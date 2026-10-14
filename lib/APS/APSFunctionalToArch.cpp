#include "APS/APSOps.h"
#include "APS/PassDetail.h"
#include "APS/Passes.h"
#include "TOR/TOR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#define DEBUG_TYPE "aps-functional-to-arch"

namespace {
using namespace mlir;

constexpr llvm::StringLiteral kCpuItfc = "cpuitfc";
constexpr llvm::StringLiteral kBusItfc = "busitfc";

/// The paper's (W, M, I, L, E, C) memory-interface tuple.  Sizes are in
/// bytes, latencies are in cycles, and maxBeats/inflight are counts.
struct MemoryInterfaceModel {
  aps::MemItfc op;
  StringRef name;
  uint64_t width;
  uint64_t maxBeats;
  uint64_t minBeats;
  uint64_t minWriteBeats;
  uint64_t inflight;
  uint64_t readLatency;
  uint64_t writeCompletion;
  uint64_t cacheLine;
  StringRef cacheHint;
};

struct MemoryMapLookup {
  llvm::StringMap<aps::MemEntryOp> bankToEntry;
};

void copyAttrs(Operation *from, Operation *to) {
  for (NamedAttribute attr : from->getAttrs())
    to->setAttr(attr.getName(), attr.getValue());
}

void setModelAttr(Operation *op, StringRef name, uint64_t value) {
  op->setAttr(name, IntegerAttr::get(IntegerType::get(op->getContext(), 64),
                                    value));
}

void ensureMemItfc(ModuleOp moduleOp, llvm::StringRef name, uint64_t width,
                   uint64_t maxBeats, uint64_t inflight,
                   uint64_t readLatency, uint64_t writeCompletion,
                   uint64_t cacheLine, StringRef cacheHint) {
  SymbolTable symbolTable(moduleOp);
  OpBuilder builder(moduleOp.getContext());
  if (auto existing = dyn_cast_or_null<aps::MemItfc>(symbolTable.lookup(name))) {
    auto setIfMissing = [&](StringRef attrName, uint64_t value) {
      if (!existing->hasAttr(attrName))
        setModelAttr(existing, attrName, value);
    };
    setIfMissing("width", width);
    setIfMissing("max_beats", maxBeats);
    setIfMissing("min_beats", 1);
    setIfMissing("min_write_beats", name == kBusItfc ? 2 : 1);
    setIfMissing("inflight", inflight);
    setIfMissing("read_latency", readLatency);
    if (!existing->hasAttr("write_completion") &&
        !existing->hasAttr("write_latency"))
      setModelAttr(existing, "write_completion", writeCompletion);
    setIfMissing("cache_line", cacheLine);
    if (!existing->hasAttr("cache_hint"))
      existing->setAttr("cache_hint",
                        builder.getStringAttr(cacheHint));
    return;
  }

  builder.setInsertionPointToStart(moduleOp.getBody());
  auto itfc = builder.create<aps::MemItfc>(moduleOp.getLoc(), name);
  setModelAttr(itfc, "width", width);
  setModelAttr(itfc, "max_beats", maxBeats);
  setModelAttr(itfc, "min_beats", 1);
  setModelAttr(itfc, "min_write_beats", name == kBusItfc ? 2 : 1);
  setModelAttr(itfc, "inflight", inflight);
  setModelAttr(itfc, "read_latency", readLatency);
  setModelAttr(itfc, "write_completion", writeCompletion);
  setModelAttr(itfc, "cache_line", cacheLine);
  itfc->setAttr("cache_hint", builder.getStringAttr(cacheHint));
}

std::optional<uint64_t> getPositiveModelAttr(Operation *op, StringRef name,
                                             StringRef compatibilityName = {}) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr && !compatibilityName.empty())
    attr = op->getAttrOfType<IntegerAttr>(compatibilityName);
  if (!attr || attr.getInt() <= 0)
    return std::nullopt;
  return static_cast<uint64_t>(attr.getInt());
}

FailureOr<MemoryInterfaceModel> readInterfaceModel(aps::MemItfc op) {
  auto width = getPositiveModelAttr(op, "width");
  auto maxBeats = getPositiveModelAttr(op, "max_beats");
  auto minBeats = op->hasAttr("min_beats")
      ? getPositiveModelAttr(op, "min_beats") : std::optional<uint64_t>(1);
  auto minWriteBeats = op->hasAttr("min_write_beats")
      ? getPositiveModelAttr(op, "min_write_beats") : minBeats;
  auto inflight = getPositiveModelAttr(op, "inflight");
  auto readLatency = getPositiveModelAttr(op, "read_latency");
  auto writeCompletion =
      getPositiveModelAttr(op, "write_completion", "write_latency");
  auto cacheLine = getPositiveModelAttr(op, "cache_line");
  if (!width || !maxBeats || !minBeats || !minWriteBeats || !inflight || !readLatency ||
      !writeCompletion || !cacheLine) {
    op.emitError("memory interface requires positive width, min_beats, min_write_beats, max_beats, "
                 "inflight, read_latency, write_completion, and cache_line "
                 "attributes");
    return failure();
  }
  if ((*cacheLine % *width) != 0)
    return op.emitError("cache_line must be a multiple of width"), failure();
  if ((*width & (*width - 1)) != 0)
    return op.emitError("width must be a power of two for naturally aligned transactions"), failure();
  if (*maxBeats > uint64_t(std::numeric_limits<int64_t>::max()) / *width)
    return op.emitError("maximum transaction byte count overflows the interface model"), failure();
  if ((*minBeats & (*minBeats - 1)) != 0 || *minBeats > *maxBeats)
    return op.emitError("min_beats must be a positive power of two no greater than max_beats"), failure();
  if ((*minWriteBeats & (*minWriteBeats - 1)) != 0 || *minWriteBeats > *maxBeats)
    return op.emitError("min_write_beats must be a positive power of two no greater than max_beats"), failure();
  *minWriteBeats = std::max(*minBeats, *minWriteBeats);
  if ((op.getSymName() == kCpuItfc && *width != 4) ||
      (op.getSymName() == kBusItfc && *width != 8))
    return op.emitError("native adapter width must be 4 bytes for cpuitfc or 8 bytes for busitfc"), failure();
  if (op.getSymName() == kBusItfc && *minWriteBeats < 2)
    return op.emitError("native DMA write adapter requires min_write_beats >= 2"), failure();
  setModelAttr(op, "min_beats", *minBeats);
  setModelAttr(op, "min_write_beats", *minWriteBeats);

  auto hint = op->getAttrOfType<StringAttr>("cache_hint");
  return MemoryInterfaceModel{op, op.getSymName(), *width, *maxBeats, *minBeats, *minWriteBeats,
                              *inflight, *readLatency, *writeCompletion,
                              *cacheLine, hint ? hint.getValue() : StringRef{}};
}

FailureOr<SmallVector<MemoryInterfaceModel>>
collectInterfaceModels(ModuleOp moduleOp) {
  SmallVector<MemoryInterfaceModel> models;
  for (auto op : moduleOp.getOps<aps::MemItfc>()) {
    auto model = readInterfaceModel(op);
    if (failed(model))
      return failure();
    models.push_back(*model);
  }
  if (models.empty()) {
    moduleOp.emitError("no modeled aps.memitfc declarations available");
    return failure();
  }
  return models;
}

std::optional<int64_t> getConstantInt(Value value) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;

  auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
  if (!intAttr)
    return std::nullopt;

  return intAttr.getValue().getSExtValue();
}

Value createIntegerConstant(OpBuilder &builder, Location loc, Type type,
                            int64_t value) {
  if (type.isIndex())
    return builder.create<arith::ConstantIndexOp>(loc, value);
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType)
    return {};

  return builder
      .create<arith::ConstantOp>(loc, IntegerAttr::get(intType, value))
      .getResult();
}

uint64_t getElementSizeBytes(MemRefType memrefType) {
  unsigned bitWidth = memrefType.getElementTypeBitWidth();
  return (bitWidth + 7) / 8;
}

uint64_t getTypeSizeBytes(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return (integer.getWidth() + 7) / 8;
  if (auto floating = dyn_cast<FloatType>(type))
    return (floating.getWidth() + 7) / 8;
  return 0;
}

/// Legal transfers contain a power-of-two number of whole beats. Minimum
/// length is min_beats * W (one beat by default), independent of cache line C.
SmallVector<uint64_t> getLegalTransferSizes(const MemoryInterfaceModel &model,
                                            uint64_t granularity, bool isLoad) {
  SmallVector<uint64_t> sizes;
  uint64_t burstBeats = 1;
  while (burstBeats <= model.maxBeats / 2)
    burstBeats *= 2;
  uint64_t size = model.width * burstBeats;
  if (granularity > size)
    return sizes;
  uint64_t minimum = model.width * (isLoad ? model.minBeats : model.minWriteBeats);
  while (size >= std::max(minimum, granularity) && size != 0) {
    if (size % granularity == 0)
      sizes.push_back(size);
    size /= 2;
  }
  llvm::sort(sizes, std::greater<uint64_t>());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  return sizes;
}

SmallVector<uint64_t> decomposeTransfer(uint64_t bytes, uint64_t granularity,
                                        const MemoryInterfaceModel &model,
                                        bool isLoad,
                                        std::optional<uint64_t> baseAddress = std::nullopt) {
  SmallVector<uint64_t> result;
  uint64_t minimum = model.width * (isLoad ? model.minBeats : model.minWriteBeats);
  if (!bytes || bytes % minimum != 0)
    return result;
  auto legalSizes = getLegalTransferSizes(model, granularity, isLoad);
  uint64_t offset = 0;
  while (bytes) {
    auto size = llvm::find_if(legalSizes, [&](uint64_t size) {
      return size <= bytes && (!baseAddress || (*baseAddress + offset) % size == 0);
    });
    if (size == legalSizes.end()) return {};
    result.push_back(*size);
    offset += *size;
    bytes -= *size;
  }
  return result;
}

uint64_t estimateLatency(ArrayRef<uint64_t> transfers,
                         const MemoryInterfaceModel &model, bool isLoad) {
  SmallVector<int64_t> issue(transfers.size(), -1);
  SmallVector<int64_t> completion(transfers.size(), -1);
  for (size_t j = 0; j < transfers.size(); ++j) {
    int64_t previousIssue = j == 0 ? -1 : issue[j - 1];
    int64_t retiredSlot =
        j < model.inflight ? -1 : completion[j - model.inflight];
    issue[j] = 1 + std::max(previousIssue, retiredSlot);

    int64_t previousCompletion = j == 0 ? -1 : completion[j - 1];
    uint64_t beats = std::max<uint64_t>(1, (transfers[j] + model.width - 1) /
                                              model.width);
    if (isLoad)
      completion[j] = beats +
                      std::max<int64_t>(previousCompletion,
                                        issue[j] + model.readLatency - 1);
    else
      completion[j] = beats + model.writeCompletion +
                      std::max<int64_t>(previousCompletion, issue[j] - 1);
  }
  return completion.empty() ? 0 : static_cast<uint64_t>(completion.back());
}

uint64_t cachePenalty(uint64_t bytes, const MemoryInterfaceModel &model,
                      StringRef) {
  uint64_t lines = (bytes + model.cacheLine - 1) / model.cacheLine;
  uint64_t penalty = lines * (model.cacheLine / model.width);
  return penalty;
}

StringRef getCacheHint(Operation *op) {
  if (auto hint = op->getAttrOfType<StringAttr>("cache_hint"))
    return hint.getValue();
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return {};
  for (Value operand : op->getOperands()) {
    auto getGlobal = operand.getDefiningOp<memref::GetGlobalOp>();
    if (!getGlobal)
      continue;
    auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
        getGlobal, getGlobal.getNameAttr());
    if (!global)
      continue;
    if (auto hint = global->getAttrOfType<StringAttr>("cache_hint"))
      return hint.getValue();
  }
  return {};
}

struct InterfaceChoice {
  const MemoryInterfaceModel *model = nullptr;
  SmallVector<uint64_t> transfers;
  uint64_t cost = std::numeric_limits<uint64_t>::max();
  uint64_t latency = 0;
};

InterfaceChoice makeChoice(const MemoryInterfaceModel &model, uint64_t bytes,
                           uint64_t granularity, bool isLoad, StringRef hint,
                           std::optional<uint64_t> baseAddress = std::nullopt) {
  if (!bytes || !granularity)
    return {};
  auto transfers = decomposeTransfer(bytes, granularity, model, isLoad, baseAddress);
  if (transfers.empty())
    return {};
  uint64_t latency = estimateLatency(transfers, model, isLoad);
  return {&model, std::move(transfers),
          latency + cachePenalty(bytes, model, hint), latency};
}

struct MemoryRequest {
  Operation *op;
  uint64_t bytes;
  uint64_t granularity;
  // Fully partitioned scratchpads require individual scalar transfers.
  bool scalarized = false;
  bool bulk = false;
  std::optional<uint64_t> baseAddress;
};

using InterfaceAssignments = llvm::DenseMap<Operation *, InterfaceChoice>;

/// Minimize the paper's approximate objective using active-interface masks.
/// Startup is charged on the first use of an interface. Each transition assigns
/// one request to one legal interface: O(Q K 2^K), with no MIP search.
LogicalResult solveRegion(ArrayRef<MemoryRequest> requests,
                          ArrayRef<MemoryInterfaceModel> models, bool isLoad,
                          InterfaceAssignments &assignments) {
  if (models.size() > 63)
    return requests.front().op->emitError("interface assignment supports at most 63 visible interfaces");
  struct Cell {
    double cost;
    uint64_t previous;
    unsigned interface;
  };
  // Insertion-ordered hash maps give reproducible ties and O(1) mask lookup,
  // while retaining only reachable active sets.
  std::vector<llvm::MapVector<uint64_t, Cell>> layers(requests.size() + 1);
  layers.front()[0] = {0, 0, 0};
  SmallVector<SmallVector<InterfaceChoice>> candidates;
  for (auto [q, request] : llvm::enumerate(requests)) {
    SmallVector<InterfaceChoice> choices(models.size());
    SmallVector<double> costs(models.size(), std::numeric_limits<double>::infinity());
    bool legal = false;
    for (auto [k, model] : llvm::enumerate(models)) {
      // Current native adapters: cpuitfc has scalar issue/wait; busitfc has
      // DMA copies. Do not select an interface with no matching lowering path.
      if ((request.bulk && model.name == kCpuItfc) ||
          (!request.bulk && model.name == kBusItfc) ||
          (model.name == kCpuItfc && request.granularity != 4))
        continue;
      auto choice = makeChoice(model, request.bytes, request.granularity,
                               isLoad, getCacheHint(request.op), request.baseAddress);
      if (request.scalarized) {
        auto scalar = makeChoice(model, request.granularity,
                                 request.granularity, isLoad,
                                 getCacheHint(request.op), request.baseAddress);
        if (!scalar.model || scalar.transfers.size() != 1)
          continue;
        choice.transfers.assign(request.bytes / request.granularity,
                                request.granularity);
        choice.latency = estimateLatency(choice.transfers, model, isLoad);
        choice.cost = choice.latency + cachePenalty(
            request.bytes, model, getCacheHint(request.op));
      }
      if (!choice.model)
        continue;
      legal = true;
      double cost = cachePenalty(request.bytes, model, getCacheHint(request.op));
      for (uint64_t bytes : choice.transfers) {
        double beats = double(bytes) / model.width;
        cost += isLoad ? std::max(double(model.readLatency) / model.inflight,
                                  beats)
                       : beats + model.writeCompletion;
      }
      costs[k] = cost;
      choices[k] = std::move(choice);
    }
    if (!legal)
      return request.op->emitError()
          << "no legal memory interface for " << request.bytes
          << "-byte " << (request.bulk ? "bulk transfer" : "scalar access")
          << ": every transaction must satisfy min_beats/max_beats, whole-beat "
             "power-of-two length, natural alignment, and adapter capability; "
             "sub-beat tails cannot be rounded up";
    for (auto [mask, cell] : layers[q]) {
      for (unsigned k = 0; k < models.size(); ++k) {
        if (!choices[k].model)
          continue;
        uint64_t bit = uint64_t(1) << k;
        double startup = (mask & bit) ? 0 :
            (isLoad ? double(models[k].readLatency) - 1 : -1);
        double cost = cell.cost + costs[k] + startup;
        auto [it, inserted] = layers[q + 1].try_emplace(
            mask | bit, Cell{cost, mask, k});
        if (!inserted && cost < it->second.cost)
          it->second = {cost, mask, k};
      }
    }
    candidates.push_back(std::move(choices));
  }
  auto best = std::min_element(layers.back().begin(), layers.back().end(),
      [](const auto &a, const auto &b) { return a.second.cost < b.second.cost; });
  double objective = best->second.cost;
  uint64_t mask = best->first;
  for (size_t q = requests.size(); q-- > 0;) {
    auto cell = layers[q + 1].find(mask)->second;
    auto *op = requests[q].op;
    assignments[op] = std::move(candidates[q][cell.interface]);
    OpBuilder builder(op->getContext());
    op->setAttr("interface_solver", builder.getStringAttr("active-set-dp"));
    op->setAttr("solver_objective", builder.getF64FloatAttr(objective));
    mask = cell.previous;
  }
  return success();
}

void annotateBinding(Operation *op, const InterfaceChoice &choice,
                     uint64_t transferBytes, unsigned segment,
                     unsigned segmentCount, uint64_t transactionGroup = 0) {
  auto *ctx = op->getContext();
  auto i64 = IntegerType::get(ctx, 64);
  op->setAttr("transfer_bytes", IntegerAttr::get(i64, transferBytes));
  op->setAttr("required_alignment",
              IntegerAttr::get(i64, transferBytes));
  op->setAttr("estimated_latency",
              IntegerAttr::get(i64, choice.latency));
  op->setAttr("interface_cost", IntegerAttr::get(i64, choice.cost));
  op->setAttr("segment", IntegerAttr::get(i64, segment));
  op->setAttr("segment_count", IntegerAttr::get(i64, segmentCount));
  if (segmentCount > 1) {
    op->setAttr("transaction_group",
                IntegerAttr::get(i64, transactionGroup));
    if (segment > 0)
      op->setAttr("after_segment", IntegerAttr::get(i64, segment - 1));
  }
}

bool isSingleElementMemref(Value memref) {
  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  return memrefType && memrefType.hasStaticShape() &&
         memrefType.getNumElements() == 1;
}

std::optional<StringRef> getGlobalName(Value memref) {
  auto getGlobal = memref.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return std::nullopt;
  return getGlobal.getName();
}

MemoryMapLookup buildMemoryMapLookup(ModuleOp moduleOp) {
  MemoryMapLookup lookup;
  moduleOp.walk([&](aps::MemoryMapOp memoryMap) {
    for (auto &block : memoryMap.getRegion()) {
      for (auto entry : block.getOps<aps::MemEntryOp>()) {
        for (Attribute bankAttr : entry.getBankSymbols()) {
          auto symbol = dyn_cast<FlatSymbolRefAttr>(bankAttr);
          if (!symbol)
            continue;
          lookup.bankToEntry[symbol.getValue()] = entry;
        }
      }
    }
  });
  return lookup;
}

aps::MemEntryOp getCommonMemoryEntry(aps::Copy copy,
                                     const MemoryMapLookup &lookup) {
  aps::MemEntryOp commonEntry;
  for (Value memref : copy.getMemrefs()) {
    auto globalName = getGlobalName(memref);
    if (!globalName)
      return {};

    auto it = lookup.bankToEntry.find(*globalName);
    if (it == lookup.bankToEntry.end())
      return {};

    if (!commonEntry) {
      commonEntry = it->second;
      continue;
    }
    if (commonEntry != it->second)
      return {};
  }
  return commonEntry;
}

bool isFullyExpandedCopy(aps::Copy copy, const MemoryMapLookup &lookup) {
  if (copy.getMemrefs().empty())
    return false;

  auto entry = getCommonMemoryEntry(copy, lookup);
  if (!entry)
    return false;

  if (entry.getNumBanks() != copy.getMemrefs().size())
    return false;

  for (Value memref : copy.getMemrefs()) {
    if (!isSingleElementMemref(memref))
      return false;
  }

  return true;
}

LogicalResult lowerFullyExpandedCopy(aps::Copy copy, IRRewriter &rewriter,
                                     const InterfaceChoice &assignment,
                                     const MemoryMapLookup &lookup) {
  if (!isFullyExpandedCopy(copy, lookup))
    return failure();

  auto start = getConstantInt(copy.getStart());
  auto length = getConstantInt(copy.getLength());
  if (!start || !length) {
    copy.emitError("fully partitioned copy requires constant start and length");
    return failure();
  }
  if (*start < 0 || *length < 0) {
    copy.emitError("fully partitioned copy requires non-negative start and "
                   "length");
    return failure();
  }

  auto entry = getCommonMemoryEntry(copy, lookup);
  bool cyclic = entry.getCyclic() != 0;
  int64_t numBanks = static_cast<int64_t>(copy.getMemrefs().size());
  if (*start + *length > numBanks) {
    copy.emitError("fully partitioned copy range exceeds available banks");
    return failure();
  }

  auto firstMemrefType = dyn_cast<MemRefType>(copy.getMemrefs()[0].getType());
  if (!firstMemrefType)
    return failure();

  Location loc = copy.getLoc();
  uint64_t elementSizeBytes = getElementSizeBytes(firstMemrefType);
  if (elementSizeBytes == 0) {
    copy.emitError("cannot infer element size for fully partitioned copy");
    return failure();
  }

  Type indexType = copy.getStart().getType();
  Type cpuAddrType = copy.getCpuAddr().getType();
  Type elementType = firstMemrefType.getElementType();
  auto choice = makeChoice(*assignment.model, elementSizeBytes, elementSizeBytes,
                           copy.getDirection() == aps::CopyDirection::In,
                           getCacheHint(copy));
  if (!choice.model) {
    copy.emitError("no memory interface supports the scalarized copy access");
    return failure();
  }
  auto selectedItfc =
      FlatSymbolRefAttr::get(copy.getContext(), choice.model->name);

  rewriter.setInsertionPoint(copy);
  for (int64_t i = 0; i < *length; ++i) {
    int64_t logicalIndex = *start + i;
    int64_t bank = cyclic ? logicalIndex % numBanks : logicalIndex;
    int64_t localIndex = cyclic ? logicalIndex / numBanks : 0;
    if (bank < 0 || bank >= numBanks || localIndex != 0) {
      copy.emitError("fully partitioned copy could not map element to a "
                     "single bank");
      return failure();
    }

    Value memref = copy.getMemrefs()[bank];
    Value localIndexValue = createIntegerConstant(rewriter, loc, indexType,
                                                  localIndex);
    if (!localIndexValue)
      return failure();

    Value cpuAddr = copy.getCpuAddr();
    int64_t byteOffset = i * static_cast<int64_t>(elementSizeBytes);
    if (byteOffset != 0) {
      Value offsetValue =
          createIntegerConstant(rewriter, loc, cpuAddrType, byteOffset);
      if (!offsetValue)
        return failure();
      cpuAddr = rewriter.create<arith::AddIOp>(loc, cpuAddr, offsetValue);
    }

    if (copy.getDirection() == aps::CopyDirection::In) {
      auto load = rewriter.create<aps::LoadBy>(loc, elementType, selectedItfc,
                                               cpuAddr);
      copyAttrs(copy, load);
      annotateBinding(load, choice, elementSizeBytes, 0, 1);
      rewriter.create<aps::WriteSmem>(loc, load.getResult(), memref,
                                      ValueRange{localIndexValue});
    } else {
      auto load = rewriter.create<aps::ReadSmem>(
          loc, elementType, memref, ValueRange{localIndexValue});
      auto store = rewriter.create<aps::StoreBy>(loc, selectedItfc,
                                                 load.getResult(), cpuAddr);
      copyAttrs(copy, store);
      annotateBinding(store, choice, elementSizeBytes, 0, 1);
    }
  }

  rewriter.eraseOp(copy);
  return success();
}

struct APSFunctionalToArchPass
    : public APSFunctionalToArchBase<APSFunctionalToArchPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Defaults preserve the two interfaces used by the existing backend while
    // making their performance contract explicit.  User declarations with
    // the same names override individual fields.
    bool hasDeclaredInterfaces = !moduleOp.getOps<aps::MemItfc>().empty();
    if (!hasDeclaredInterfaces) {
      ensureMemItfc(moduleOp, kCpuItfc, /*width=*/4, /*maxBeats=*/1,
                    /*inflight=*/2, /*readLatency=*/2,
                    /*writeCompletion=*/2, /*cacheLine=*/64, "warm");
      ensureMemItfc(moduleOp, kBusItfc, /*width=*/8, /*maxBeats=*/8,
                    /*inflight=*/2, /*readLatency=*/5,
                    /*writeCompletion=*/2, /*cacheLine=*/64, "cold");
    } else {
      SymbolTable symbols(moduleOp);
      if (isa_and_nonnull<aps::MemItfc>(symbols.lookup(kCpuItfc)))
        ensureMemItfc(moduleOp, kCpuItfc, 4, 1, 2, 2, 2, 64, "warm");
      if (isa_and_nonnull<aps::MemItfc>(symbols.lookup(kBusItfc)))
        ensureMemItfc(moduleOp, kBusItfc, 8, 8, 2, 5, 2, 64, "cold");
    }

    auto modelResult = collectInterfaceModels(moduleOp);
    if (failed(modelResult))
      return signalPassFailure();
    auto &models = *modelResult;

    IRRewriter rewriter(&getContext());
    MemoryMapLookup memoryMapLookup = buildMemoryMapLookup(moduleOp);

    SmallVector<aps::Load> loads;
    SmallVector<aps::Store> stores;
    SmallVector<aps::Copy> copies;
    moduleOp.walk([&](Operation *op) {
      if (auto load = dyn_cast<aps::Load>(op))
        loads.push_back(load);
      else if (auto store = dyn_cast<aps::Store>(op))
        stores.push_back(store);
      else if (auto copy = dyn_cast<aps::Copy>(op))
        copies.push_back(copy);
      if (isa<aps::Load, aps::Store, aps::Copy>(op))
        if (auto hint = getCacheHint(op); !hint.empty())
          op->setAttr("cache_hint", StringAttr::get(op->getContext(), hint));
    });

    llvm::DenseMap<Region *, SmallVector<MemoryRequest>> readRegions, writeRegions;
    for (auto load : loads) {
      uint64_t bytes = getTypeSizeBytes(load.getResult().getType());
      auto address = getConstantInt(load.getCpuAddr());
      readRegions[load->getParentRegion()].push_back({load, bytes, bytes, false, false,
          address ? std::optional<uint64_t>(uint64_t(*address)) : std::nullopt});
    }
    for (auto store : stores) {
      uint64_t bytes = getTypeSizeBytes(store.getValue().getType());
      auto address = getConstantInt(store.getCpuAddr());
      writeRegions[store->getParentRegion()].push_back({store, bytes, bytes, false, false,
          address ? std::optional<uint64_t>(uint64_t(*address)) : std::nullopt});
    }
    for (auto copy : copies) {
      if (copy.getMemrefs().empty()) {
        copy.emitError("copy requires at least one scratchpad memref");
        return signalPassFailure();
      }
      auto type = dyn_cast<MemRefType>(copy.getMemrefs()[0].getType());
      uint64_t elementBytes = type ? getElementSizeBytes(type) : 0;
      auto length = getConstantInt(copy.getLength());
      if (!elementBytes || (length && *length < 0)) {
        copy.emitError("copy requires a sized element type and non-negative length");
        return signalPassFailure();
      }
      if (!length) {
        copy.emitError("functional-to-architectural canonicalization requires a constant transfer length; normalize dynamic transfers before interface binding");
        return signalPassFailure();
      }
      if (!*length)
        continue;
      if (uint64_t(*length) > std::numeric_limits<uint64_t>::max() / elementBytes) {
        copy.emitError("copy byte count overflows the interface model");
        return signalPassFailure();
      }
      auto &regions = copy.getDirection() == aps::CopyDirection::In
                          ? readRegions : writeRegions;
      bool scalarized = isFullyExpandedCopy(copy, memoryMapLookup);
      auto address = getConstantInt(copy.getCpuAddr());
      regions[copy->getParentRegion()].push_back(
          {copy, uint64_t(*length) * elementBytes, elementBytes, scalarized, !scalarized,
           address ? std::optional<uint64_t>(uint64_t(*address)) : std::nullopt});
    }
    InterfaceAssignments assignments;
    for (auto &entry : readRegions)
      if (failed(solveRegion(entry.second, models, true, assignments)))
        return signalPassFailure();
    for (auto &entry : writeRegions)
      if (failed(solveRegion(entry.second, models, false, assignments)))
        return signalPassFailure();

    for (auto load : loads) {
      uint64_t bytes = getTypeSizeBytes(load.getResult().getType());
      auto choice = assignments.lookup(load);
      if (!choice.model) {
        load.emitError("no memory interface supports this load width");
        return signalPassFailure();
      }
      rewriter.setInsertionPoint(load);
      auto itfc = FlatSymbolRefAttr::get(&getContext(), choice.model->name);
      auto newLoad = rewriter.create<aps::LoadBy>(
          load.getLoc(), load.getResult().getType(), itfc,
          load.getCpuAddr());
      copyAttrs(load.getOperation(), newLoad.getOperation());
      annotateBinding(newLoad, choice, bytes, 0, 1);
      rewriter.replaceOp(load, newLoad.getResult());
    }

    for (auto store : stores) {
      uint64_t bytes = getTypeSizeBytes(store.getValue().getType());
      auto choice = assignments.lookup(store);
      if (!choice.model) {
        store.emitError("no memory interface supports this store width");
        return signalPassFailure();
      }
      rewriter.setInsertionPoint(store);
      auto itfc = FlatSymbolRefAttr::get(&getContext(), choice.model->name);
      auto newStore = rewriter.create<aps::StoreBy>(
          store.getLoc(), itfc, store.getValue(), store.getCpuAddr());
      copyAttrs(store.getOperation(), newStore.getOperation());
      annotateBinding(newStore, choice, bytes, 0, 1);
      rewriter.eraseOp(store);
    }

    uint64_t transactionGroup = 0;
    for (auto copy : copies) {
      if (getConstantInt(copy.getLength()) == std::optional<int64_t>(0)) {
        rewriter.eraseOp(copy);
        continue;
      }
      if (isFullyExpandedCopy(copy, memoryMapLookup)) {
        auto assignment = assignments.lookup(copy);
        if (!assignment.model) {
          copy.emitError("fully partitioned copy requires a constant length");
          return signalPassFailure();
        }
        if (failed(lowerFullyExpandedCopy(copy, rewriter, assignment,
                                          memoryMapLookup)))
          return signalPassFailure();
        continue;
      }

      if (copy.getMemrefs().empty()) {
        copy.emitError("copy requires at least one scratchpad memref");
        return signalPassFailure();
      }
      auto memrefType = dyn_cast<MemRefType>(copy.getMemrefs()[0].getType());
      uint64_t elementBytes =
          memrefType ? getElementSizeBytes(memrefType) : 0;
      auto length = getConstantInt(copy.getLength());
      if (elementBytes == 0 || (length && *length < 0)) {
        copy.emitError("copy requires a sized element type and non-negative "
                       "length");
        return signalPassFailure();
      }

      rewriter.setInsertionPoint(copy);
      uint64_t totalBytes = static_cast<uint64_t>(*length) * elementBytes;
      if (totalBytes == 0) {
        rewriter.eraseOp(copy);
        continue;
      }
      auto choice = assignments.lookup(copy);
      if (!choice.model) {
        copy.emitError("no memory interface can legally decompose this copy");
        return signalPassFailure();
      }

      uint64_t byteOffset = 0;
      for (auto [segment, transferBytes] : llvm::enumerate(choice.transfers)) {
        uint64_t elementOffset = byteOffset / elementBytes;
        uint64_t elementCount = transferBytes / elementBytes;
        Value cpuAddr = copy.getCpuAddr();
        Value start = copy.getStart();
        if (byteOffset != 0) {
          Value cpuOffset = createIntegerConstant(
              rewriter, copy.getLoc(), cpuAddr.getType(), byteOffset);
          Value startOffset = createIntegerConstant(
              rewriter, copy.getLoc(), start.getType(), elementOffset);
          if (!cpuOffset || !startOffset) {
            copy.emitError("copy address and index types must be integers");
            return signalPassFailure();
          }
          cpuAddr = rewriter.create<arith::AddIOp>(copy.getLoc(), cpuAddr,
                                                   cpuOffset);
          start = rewriter.create<arith::AddIOp>(copy.getLoc(), start,
                                                 startOffset);
        }
        Value segmentLength = createIntegerConstant(
            rewriter, copy.getLoc(), copy.getLength().getType(), elementCount);
        if (!segmentLength) {
          copy.emitError("copy length type must be an integer");
          return signalPassFailure();
        }
        auto newCopy = rewriter.create<aps::CopyBy>(
            copy.getLoc(), copy.getDirection(), choice.model->name, cpuAddr,
            copy.getMemrefs(), start, segmentLength);
        copyAttrs(copy.getOperation(), newCopy.getOperation());
        annotateBinding(newCopy, choice, transferBytes, segment,
                        choice.transfers.size(), transactionGroup);
        newCopy->setAttr("transfer_offset", rewriter.getI64IntegerAttr(byteOffset));
        byteOffset += transferBytes;
      }
      rewriter.eraseOp(copy);
      ++transactionGroup;
    }
  }
};

} // namespace

namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>> createAPSFunctionalToArchPass() {
  return std::make_unique<APSFunctionalToArchPass>();
}
} // namespace mlir
