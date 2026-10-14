#ifndef APS_INTERFACE_CONFIG_H
#define APS_INTERFACE_CONFIG_H

#include "APS/APSOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/ADT/StringExtras.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>

namespace apsconfig {
using Json = nlohmann::json;

inline void keys(const Json &object, std::initializer_list<const char *> allowed) {
  if (!object.is_object()) throw std::runtime_error("expected an object");
  for (auto it = object.begin(); it != object.end(); ++it) {
    bool known = false;
    for (auto key : allowed) known |= it.key() == key;
    if (!known) throw std::runtime_error("unknown configuration key: " + it.key());
  }
}

inline uint64_t integer(const Json &value) {
  if (value.is_string()) {
    auto text = value.get<std::string>();
    if (text.empty() || text.front() == '-') throw std::runtime_error("invalid unsigned integer");
    size_t used = 0;
    uint64_t number = std::stoull(text, &used, 0);
    if (used != text.size()) throw std::runtime_error("invalid integer: " + text);
    return number;
  }
  if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0))
    throw std::runtime_error("expected a non-negative integer");
  return value.get<uint64_t>();
}

inline bool identifier(llvm::StringRef name) {
  return !name.empty() && llvm::all_of(name, [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_';
  });
}

struct Config {
  Json data;
  Json csrs = Json::array();
  uint64_t maxBurstBytes = 0, inflight = 0;

  void load(llvm::StringRef path) {
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) throw std::runtime_error("cannot read " + path.str());
    data = Json::parse((*buffer)->getBuffer().str());
    keys(data, {"version", "interfaces", "chipyard", "csrs"});
    if (data.at("version") != 1) throw std::runtime_error("unsupported interface configuration version");
    if (!data.at("interfaces").is_array() || data.at("interfaces").empty())
      throw std::runtime_error("interfaces must be a non-empty array");
    keys(data.at("chipyard"), {"backend", "arch", "core_count", "top_module", "name"});
    auto &platform = data.at("chipyard");
    if (platform.at("backend") != "rocc" || platform.at("arch") != "rv32" ||
        integer(platform.at("core_count")) != 1)
      throw std::runtime_error("current APS backend supports rocc/rv32/core_count=1 only");
    for (auto key : {"top_module", "name"})
      if (!identifier(platform.at(key).get<std::string>()))
        throw std::runtime_error("invalid Chipyard name");
    std::set<std::string> names;
    for (auto &model : data.at("interfaces")) {
      keys(model, {"name", "width", "min_beats", "min_write_beats", "max_beats", "inflight",
                   "read_latency", "write_completion", "cache_line", "cache_hint"});
      auto name = model.at("name").get<std::string>();
      if ((name != "cpuitfc" && name != "busitfc") || !names.insert(name).second)
        throw std::runtime_error("configuration requires unique native cpuitfc/busitfc interfaces");
      for (auto key : {"width", "min_beats", "min_write_beats", "max_beats", "inflight",
                       "read_latency", "write_completion", "cache_line"}) {
        auto number = integer(model.at(key));
        if (!number || number > uint64_t(std::numeric_limits<int64_t>::max()))
          throw std::runtime_error(std::string("invalid positive model field: ") + key);
        model[key] = number;
      }
      auto width = integer(model.at("width")), maximum = integer(model.at("max_beats"));
      auto minimum = integer(model.at("min_beats")), writeMinimum = integer(model.at("min_write_beats"));
      if ((minimum & (minimum - 1)) || (writeMinimum & (writeMinimum - 1)) ||
          (maximum & (maximum - 1)) || minimum > maximum || writeMinimum > maximum ||
          integer(model.at("cache_line")) % width)
        throw std::runtime_error("invalid beat bounds or cache-line model");
      if (model.at("cache_hint") != "warm" && model.at("cache_hint") != "cold")
        throw std::runtime_error("cache_hint must be warm or cold");
      if (name == "cpuitfc" && (width != 4 || maximum != 1 || minimum != 1 || writeMinimum != 1 ||
                                integer(model.at("inflight")) > 2))
        throw std::runtime_error("native CPU adapter requires a single 4-byte beat and inflight<=2");
      if (name == "busitfc") {
        if (width != 8 || maximum > 16 || writeMinimum < 2)
          throw std::runtime_error("native DMA requires width=8, max_beats<=16, min_write_beats>=2");
        maxBurstBytes = width * maximum;
        inflight = integer(model.at("inflight"));
        if (inflight != 2)
          throw std::runtime_error("current native DMA wrapper has two channels; inflight must be 2");
      }
    }
    if (!maxBurstBytes) throw std::runtime_error("Chipyard configuration requires busitfc");
    if (data.contains("csrs") && !data.at("csrs").is_array())
      throw std::runtime_error("csrs must be an array");
  }

  void apply(mlir::ModuleOp module) {
    mlir::OpBuilder builder(module.getContext());
    mlir::SymbolTable symbols(module);
    // This file is authoritative. Reject ambiguous pre-existing declarations.
    for (auto op : module.getOps<aps::MemItfc>()) {
      bool present = false;
      for (auto &model : data.at("interfaces")) present |= model.at("name") == op.getSymName().str();
      if (!present) throw std::runtime_error("IR interface is absent from interface configuration");
    }
    for (auto &model : data.at("interfaces")) {
      auto name = model.at("name").get<std::string>();
      auto *existing = symbols.lookup(name);
      if (existing && !mlir::isa<aps::MemItfc>(existing))
        throw std::runtime_error("interface name conflicts with another IR symbol: " + name);
      aps::MemItfc op;
      if (existing) op = mlir::cast<aps::MemItfc>(existing);
      else {
        builder.setInsertionPointToStart(module.getBody());
        op = builder.create<aps::MemItfc>(module.getLoc(), name);
        symbols.insert(op);
      }
      for (auto it = model.begin(); it != model.end(); ++it) {
        if (it.key() == "name") continue;
        mlir::Attribute attr = it.key() == "cache_hint"
            ? mlir::Attribute(builder.getStringAttr(it.value().get<std::string>()))
            : mlir::Attribute(builder.getI64IntegerAttr(integer(it.value())));
        if (auto old = op->getAttr(it.key())) {
          bool same = old == attr;
          if (auto integerOld = mlir::dyn_cast<mlir::IntegerAttr>(old))
            if (auto integerNew = mlir::dyn_cast<mlir::IntegerAttr>(attr))
              same = integerOld.getInt() == integerNew.getInt();
          if (!same) throw std::runtime_error("IR/config conflict for " + name + "." + it.key());
        }
        op->setAttr(it.key(), attr);
      }
    }
    std::set<uint64_t> ids;
    std::set<std::string> csrNames;
    for (auto global : module.getOps<mlir::memref::GlobalOp>()) {
      if (!global->hasAttr("csr_address")) continue;
      auto address = global->getAttrOfType<mlir::IntegerAttr>("csr_address");
      auto type = global.getType();
      auto name = global.getSymName().str();
      if (!address || address.getInt() < 0 || address.getInt() > 4095 ||
          !type.hasStaticShape() || !type.getElementType().isInteger(32) || type.getNumElements() != 1 ||
          !identifier(name) || !ids.insert(address.getInt()).second || !csrNames.insert(name).second)
        throw std::runtime_error("CSR requires unique 12-bit address/name and a single i32 element: " + name);
      uint64_t mask = 0xffffffff, init = 0;
      if (auto attr = global->getAttrOfType<mlir::IntegerAttr>("csr_mask")) mask = attr.getValue().getZExtValue();
      if (auto attr = global->getAttrOfType<mlir::IntegerAttr>("csr_init")) init = attr.getValue().getZExtValue();
      bool overridden = false;
      for (auto &csr : data.value("csrs", Json::array())) {
        keys(csr, {"name", "id", "mask", "init"});
        if (csr.at("name") != name) continue;
        if (overridden) throw std::runtime_error("duplicate CSR override: " + name);
        overridden = true;
        if (csr.contains("id") && integer(csr.at("id")) != uint64_t(address.getInt()))
          throw std::runtime_error("CSR address conflicts with IR: " + name);
        if (csr.contains("mask")) mask = integer(csr.at("mask"));
        if (csr.contains("init")) init = integer(csr.at("init"));
      }
      if (mask > 0xffffffff || init > 0xffffffff)
        throw std::runtime_error("CSR mask/init exceeds rv32 width: " + name);
      csrs.push_back({{"id", address.getInt()}, {"name", name}, {"mask", mask}, {"init", init}});
    }
    for (auto &csr : data.value("csrs", Json::array()))
      if (!csrNames.count(csr.at("name").get<std::string>()))
        throw std::runtime_error("CSR override does not refer to an IR CSR");
  }

  mlir::LogicalResult emit(llvm::StringRef path, llvm::StringRef rtl) const {
    std::string error;
    auto output = mlir::openOutputFile(path, &error);
    if (!output) {
      llvm::errs() << "[ERROR] cannot write Chipyard config: " << error << "\n";
      return mlir::failure();
    }
    auto &stream = output->os();
    stream << "# Generated by aps-e2e; interface model and input IR are authoritative.\n";
    auto &platform = data.at("chipyard");
    for (auto key : {"backend", "arch", "top_module", "name"})
      stream << key << ": " << platform.at(key).dump() << "\n";
    stream << "core_count: " << integer(platform.at("core_count")) << "\n"
           << "vsrc:\n  - " << Json(std::filesystem::absolute(rtl.str()).lexically_normal().string()).dump() << "\n"
           << "maxBurstBytes: " << maxBurstBytes << "\nnXacts: " << inflight << "\n";
    if (csrs.empty()) stream << "csrs: []\n";
    else {
      stream << "csrs:\n";
      for (auto &csr : csrs)
        stream << "  - id: " << Json("0x" + llvm::utohexstr(integer(csr.at("id")), true)).dump() << "\n"
               << "    name: " << csr.at("name").dump() << "\n"
               << "    mask: " << Json("0x" + llvm::utohexstr(integer(csr.at("mask")), true)).dump() << "\n"
               << "    init: " << Json("0x" + llvm::utohexstr(integer(csr.at("init")), true)).dump() << "\n";
    }
    output->keep();
    return mlir::success();
  }
};
} // namespace apsconfig
#endif
