// Drive the selected ScaleISA program and its matching APS hardware together.
#include "APS/APSDialect.h"
#include "TOR/TORDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <thread>

using namespace mlir;
namespace fs = std::filesystem;
using json = nlohmann::json;
namespace cl = llvm::cl;
namespace {
cl::opt<std::string> input(cl::Positional, cl::desc("input MLIR"), cl::init(""));
cl::opt<std::string> output("o", cl::desc("Compile-only output file"));
cl::alias outputAlias("output", cl::aliasopt(output));
cl::opt<std::string> outputDir("output-dir", cl::desc("Full pipeline artifact directory"));
cl::opt<std::string> discoveryDir("discovery-dir", cl::desc("Reuse a completed ScaleISA run"));
cl::opt<unsigned> areaBudget("area-budget", cl::init(500));
cl::opt<std::string> selectionMethod("selection-method", cl::init("greedy"));
cl::opt<std::string> discoveryStages("discovery-stages", cl::init("linalg,affine,scf"),
                                    cl::desc("Comma-separated discovery levels"));
cl::opt<bool> compileOnly("compile-only", cl::desc("Compile selected software using --hardware"));
cl::opt<std::string> hardwareFile("hardware");
cl::opt<std::string> emit("emit", cl::init("object"), cl::desc("object or elf; full runs also emit RTL"));
cl::opt<std::string> entry("entry", cl::init("main"));
cl::opt<unsigned> maxCSRs("max-extra-csrs", cl::init(5));
cl::opt<std::string> targetProfile("target-profile", cl::desc("Realized CPU/accelerator/ABI target profile JSON"));
cl::opt<std::string> abi("abi", cl::init("ilp32"));
cl::opt<std::string> mattr("mattr", cl::init("+m,+a,+zicsr,+zifencei"));
cl::opt<std::string> memrefABI("memref-abi", cl::init("bare-pointer"));
cl::opt<unsigned> optLevel("opt-level", cl::init(2));
cl::opt<std::string> workDir("work-dir");
cl::opt<bool> allowNoCI("allow-no-ci");
cl::list<std::string> linkObjects("link-object", cl::ZeroOrMore);
cl::list<std::string> linkSources("link-source", cl::ZeroOrMore);
cl::list<std::string> linkArgs("link-arg", cl::ZeroOrMore);
cl::opt<std::string> linker("linker");
cl::opt<std::string> compiler("compiler");
cl::opt<std::string> python("python");
cl::opt<std::string> scaleisaRoot("scaleisa-root");
cl::opt<std::string> resource("resource", cl::desc("APS execution contracts (latency, II and limits); physical data uses ScaleISA's ASAP7 table"));
cl::opt<double> clockPeriod("clock", cl::desc("Scheduling period in ns (default: hardware profile)"));
cl::opt<bool> cosim("cosim", cl::desc("Build and run matching Spike + Verilator model"));
cl::opt<unsigned> jobs("jobs", cl::init(4));
cl::opt<unsigned> timeout("timeout", cl::init(3600), cl::desc("Per-stage timeout in seconds"));
cl::opt<bool> verbose("verbose", cl::desc("Stream all stage logs"));
cl::opt<std::string> renameMain("rename-main", cl::desc("Rename selected main for a supplied C harness"));
cl::opt<std::string> observeFunction("observe-function", cl::desc("Validation callback before main returns"));
cl::list<std::string> observeTypes("observe-type", cl::ZeroOrMore,
                                 cl::desc("Entry allocation type passed to validation callback"));
cl::list<unsigned> observeAllocations("observe-allocation", cl::ZeroOrMore,
    cl::desc("Zero-based entry allocation index passed to validation callback"));
const fs::path root = APS_SOURCE_ROOT;
const fs::path build = APS_BINARY_ROOT;

std::string read(const fs::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("Cannot read " + path.string());
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
void write(const fs::path &path, llvm::StringRef text) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(text.data(), text.size());
  if (!stream) throw std::runtime_error("Cannot write " + path.string());
}
void save(const fs::path &path, const json &report) {
  fs::path temp = path.string() + ".tmp";
  write(temp, report.dump(2) + "\n");
  fs::rename(temp, path);
}
std::string hash(const fs::path &path) {
  llvm::SHA256 digest;
  digest.update(read(path));
  return llvm::toHex(digest.final(), true);
}
fs::path findTool(llvm::StringRef name, const std::string &override = "") {
  if (!override.empty()) {
    fs::path path = fs::absolute(override);
    if (fs::is_regular_file(path)) return path;
    throw std::runtime_error("Tool not found: " + path.string());
  }
  for (auto path : {root / "circt/build/bin" / name.str(),
                    root / ".pixi/envs/default/bin" / name.str()})
    if (fs::is_regular_file(path)) return path;
  auto found = llvm::sys::findProgramByName(name);
  if (found) return *found;
  throw std::runtime_error("Build or install " + name.str());
}
fs::path nativeTool(llvm::StringRef name) {
  fs::path path = build / "tools/aps-scaleisa-opt" / name.str();
  if (!fs::is_regular_file(path)) throw std::runtime_error("Build " + name.str());
  return path;
}
std::string printed(Operation *op) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  op->print(stream, OpPrintingFlags().printGenericOpForm());
  return text;
}
std::string printed(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << type;
  return text;
}
OwningOpRef<ModuleOp> parse(const fs::path &path, MLIRContext &context) {
  auto module = parseSourceFile<ModuleOp>(path.string(), &context);
  if (!module) throw std::runtime_error("Invalid MLIR: " + path.string());
  return module;
}

struct Runner {
  fs::path dir, reportPath;
  json report;
  Runner(fs::path directory, llvm::StringRef reportName = "report.json")
      : dir(std::move(directory)), reportPath(dir / reportName.str()),
        report{{"status", "running"}, {"commands", json::array()}} {
    fs::create_directories(dir);
    save(reportPath, report);
  }
  void run(const std::string &stage, const std::vector<std::string> &args,
           bool streamProgress = false, const fs::path &cwd = {}) {
    report["stage"] = stage;
    save(reportPath, report);
    std::cout << "[aps-scaleisa] " << stage << std::endl;
    fs::path log = dir / (stage + ".log");
    write(log, "");
    std::string logName = log.string();
    std::array<std::optional<llvm::StringRef>, 3> redirects{
        std::nullopt, llvm::StringRef(logName), llvm::StringRef(logName)};
    std::vector<llvm::StringRef> refs(args.begin(), args.end());
    auto start = std::chrono::steady_clock::now();
    std::atomic<bool> done(false);
    std::thread tail;
    if (streamProgress || verbose) {
      tail = std::thread([&] {
        std::ifstream stream(log);
        do {
          const bool finishing = done.load();
          stream.clear();
          std::string line;
          while (std::getline(stream, line)) std::cout << line << std::endl;
          if (finishing) break;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } while (true);
      });
    }
    std::string error;
    fs::path previous = fs::current_path();
    if (!cwd.empty()) fs::current_path(cwd);
    int rc = llvm::sys::ExecuteAndWait(args.front(), refs, std::nullopt,
                                      redirects, timeout, 0, &error);
    if (!cwd.empty()) fs::current_path(previous);
    done = true;
    if (tail.joinable()) tail.join();
    double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    report["commands"].push_back({{"stage", stage}, {"argv", args},
        {"returncode", rc}, {"seconds", seconds}, {"log", logName},
        {"cwd", cwd.empty() ? fs::current_path().string() : cwd.string()}});
    save(reportPath, report);
    if (rc != 0)
      throw std::runtime_error(stage + " failed (" + std::to_string(rc) +
                               "). See " + logName + "\n" + error + "\n" + read(log));
  }
  void fail(const std::exception &error) {
    report["status"] = "failed";
    report["error"] = error.what();
    save(reportPath, report);
  }
};

std::string passPath(const fs::path &path) {
  std::string result = "\"";
  for (char c : path.string()) {
    if (c == '\\' || c == '"') result += '\\';
    result += c;
  }
  return result + "\"";
}

json compileProgram(const fs::path &source, const fs::path &hardware,
                    const fs::path &destination, bool softwareOnly = false) {
  fs::create_directories(destination.parent_path());
  fs::path work = workDir.empty() ? fs::path(destination.string() + ".scaleisa")
                                : fs::absolute(std::string(workDir));
  Runner run(work, "compile-report.json");
  try {
    run.report.update({{"input", source.string()}, {"hardware", hardware.string()},
        {"output", destination.string()}, {"emit", std::string(emit)},
        {"target", "riscv32-unknown-elf"}, {"abi", std::string(abi)},
        {"mattr", std::string(mattr)}, {"memref_abi", std::string(memrefABI)},
        {"max_extra_csrs", unsigned(maxCSRs)},
        {"input_sha256", hash(source)}, {"hardware_sha256", hash(hardware)}});
    auto ci = work / "ci.mlir", lowered = work / "llvm.mlir", ir = work / "program.ll";
    if (softwareOnly) {
      MLIRContext context;
      DialectRegistry registry;
      registry.insert<func::FuncDialect, arith::ArithDialect, memref::MemRefDialect,
          affine::AffineDialect, scf::SCFDialect, math::MathDialect,
          LLVM::LLVMDialect, linalg::LinalgDialect, tensor::TensorDialect>();
      context.appendDialectRegistry(registry);
      auto program = parse(source, context);
      program->walk([&](Operation *op) {
        if (op->hasAttr("scaleisa.ci"))
          throw std::runtime_error("Software-only selection contains a CI binding");
      });
      OpBuilder builder(&context);
      (*program)->setAttr("scaleisa.ci_sites", builder.getI64IntegerAttr(0));
      (*program)->setAttr("scaleisa.ci_instructions", builder.getI64IntegerAttr(0));
      write(ci, printed(*program));
    } else {
      run.run("lower-ci", {compiler.empty() ? nativeTool("scaleisa-compiler-opt").string()
                                         : findTool("scaleisa-compiler-opt", compiler).string(),
        source.string(), "--scaleisa-lower-ci=hardware-file=" + passPath(hardware) +
        " max-extra-csrs=" + std::to_string(maxCSRs), "-o", ci.string()});
    }
    std::string text = read(ci);
    for (auto key : {"ci_sites", "ci_instructions"}) {
      std::smatch match;
      if (!std::regex_search(text, match, std::regex("scaleisa\\." + std::string(key) + " = ([0-9]+) : i64")))
        throw std::runtime_error("Missing compiler count: " + std::string(key));
      run.report[key] = std::stoull(match[1]);
    }
    if (run.report["ci_sites"] == 0 && !allowNoCI && !softwareOnly)
      throw std::runtime_error("No exposed CI sites were compiled; check selected MLIR and hardware bindings");
    std::string bare = memrefABI == "bare-pointer" ? " use-bare-ptr-memref-call-conv=true" : "";
    std::string pipeline = "builtin.module(lower-affine,convert-scf-to-cf,convert-math-to-llvm,"
        "convert-arith-to-llvm{index-bitwidth=32},finalize-memref-to-llvm{index-bitwidth=32},"
        "convert-func-to-llvm{index-bitwidth=32" + bare + "},convert-cf-to-llvm{index-bitwidth=32},"
        "reconcile-unrealized-casts)";
    run.run("lower-llvm", {findTool("mlir-opt").string(), ci.string(), "--pass-pipeline=" + pipeline, "-o", lowered.string()});
    run.run("translate", {findTool("mlir-translate").string(), lowered.string(), "--mlir-to-llvmir", "-o", ir.string()});
    std::vector<std::string> common{findTool("llc").string(), ir.string(),
        "-mtriple=riscv32-unknown-elf", "-mattr=" + std::string(mattr),
        "-target-abi=" + std::string(abi), "-relocation-model=static", "-O=" + std::to_string(optLevel)};
    auto assembly = common, object = common;
    assembly.insert(assembly.end(), {"-filetype=asm", "-o", (work / "program.s").string()});
    object.insert(object.end(), {"-filetype=obj", "-o", (work / "program.o").string()});
    run.run("assembly", assembly);
    run.run("object", object);
    fs::path final = work / "program.o";
    if (emit == "elf") {
      std::vector<std::string> objects;
      for (const auto &path : linkObjects) objects.push_back(fs::absolute(path).string());
      for (auto [i, path] : llvm::enumerate(linkSources)) {
        fs::path obj = work / ("runtime" + std::to_string(i) + ".o");
        std::vector<std::string> args{findTool("clang").string(),
            "--target=riscv32-unknown-elf", "-mabi=" + std::string(abi),
            "-msmall-data-limit=0", "-ffreestanding", "-fno-builtin", "-ffp-contract=off", "-O2"};
        // Clang's target feature spelling is passed through to its backend.
        for (llvm::StringRef feature : llvm::split(llvm::StringRef(mattr), ',')) {
          args.insert(args.end(), {"-Xclang", "-target-feature", "-Xclang", feature.str()});
        }
        if (fs::path(path).extension() == ".cpp" || fs::path(path).extension() == ".cc")
          args.insert(args.end(), {"-fno-exceptions", "-fno-rtti"});
        args.insert(args.end(), {"-c", fs::absolute(path).string(), "-o", obj.string()});
        run.run("runtime" + std::to_string(i), args);
        objects.push_back(obj.string());
      }
      final = work / "program.elf";
      std::vector<std::string> args{findTool("ld.lld", linker).string(), "-m", "elf32lriscv",
          "--entry=" + std::string(entry), "--fatal-warnings", (work / "program.o").string()};
      args.insert(args.end(), objects.begin(), objects.end());
      args.insert(args.end(), linkArgs.begin(), linkArgs.end());
      args.insert(args.end(), {"-o", final.string()});
      run.run("link", args);
    }
    run.run("disassembly", {findTool("llvm-objdump").string(), "-d", final.string()});
    fs::copy_file(work / "disassembly.log", work / "program.disasm", fs::copy_options::overwrite_existing);
    fs::path partial = destination.string() + ".partial";
    fs::copy_file(final, partial, fs::copy_options::overwrite_existing);
    fs::rename(partial, destination);
    run.report.update({{"status", "passed"}, {"output_sha256", hash(destination)}});
    save(run.reportPath, run.report);
    std::cout << "Compiled " << run.report["ci_sites"] << " CI sites into "
              << run.report["ci_instructions"] << " custom instructions: " << destination << std::endl;
    return run.report;
  } catch (const std::exception &error) { run.fail(error); throw; }
}

void collectKernels(const fs::path &cache, ModuleOp program, const json &selection,
                    const fs::path &destination, Runner &run) {
  MLIRContext &context = *program.getContext();
  auto kernels = ModuleOp::create(program.getLoc());
  OwningOpRef<ModuleOp> owner(kernels);
  run.report["noalias_proofs"] = json::object();
  for (const auto &item : selection.at("selected_impls")) {
    std::string stage = item.at("origin_stage"), name = item.at("function_name");
    if (stage != "affine" && stage != "scf" && stage != "linalg")
      throw std::runtime_error("Unknown implementation stage " + stage);
    auto source = parse(cache / "uarch" / stage / (name + ".input.mlir"), context);
    auto function = source->lookupSymbol<func::FuncOp>(name);
    if (!function) throw std::runtime_error("Selected implementation missing: " + name);
    auto clone = cast<func::FuncOp>(function->clone());
    kernels.getBody()->push_back(clone);
    // Distinct allocation operations or defined globals prove disjoint storage.
    bool found = false, proven = true;
    unsigned memories = llvm::count_if(function.getArgumentTypes(), [](Type t) { return isa<MemRefType>(t); });
    json sites = json::array();
    program.walk([&](func::CallOp call) {
      llvm::StringRef callee = call.getCallee();
      if (callee != name && !callee.starts_with(name + "_site_")) return;
      found = true;
      llvm::SmallPtrSet<Operation *, 4> allocations;
      for (Value value : call.getOperands()) {
        if (!isa<MemRefType>(value.getType())) continue;
        while (auto cast = value.getDefiningOp<memref::CastOp>())
          value = cast.getSource();
        Operation *storage = nullptr;
        if (isa_and_nonnull<memref::AllocaOp, memref::AllocOp>(value.getDefiningOp()))
          storage = value.getDefiningOp();
        else if (auto get = value.getDefiningOp<memref::GetGlobalOp>()) {
          auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(get, get.getNameAttr());
          if (global && !global.isExternal()) storage = global;
        }
        if (!storage || !allocations.insert(storage).second) proven = false;
      }
      sites.push_back(callee.str());
    });
    if (!found) throw std::runtime_error("Selected implementation has no AU sites: " + name);
    if (memories > 1 && proven) {
      for (unsigned i = 0; i < clone.getNumArguments(); ++i)
        if (isa<MemRefType>(clone.getArgument(i).getType()))
          clone.setArgAttr(i, "scaleisa.noalias", UnitAttr::get(&context));
      run.report["noalias_proofs"][name] = {{"basis", "distinct allocations or defined globals at every call"}, {"sites", sites}};
    }
  }
  if (failed(verify(kernels))) throw std::runtime_error("Invalid selected implementation collection");
  write(destination, printed(kernels));
}

void instrument(ModuleOp program) {
  if (renameMain.empty() && observeFunction.empty()) return;
  auto main = program.lookupSymbol<func::FuncOp>("main");
  if (!main || main.isExternal()) throw std::runtime_error("Validation requires defined main");
  if (!renameMain.empty()) {
    if (program.lookupSymbol(renameMain)) throw std::runtime_error("Renamed main already exists");
    if (failed(SymbolTable::replaceAllSymbolUses(main, StringAttr::get(program.getContext(), renameMain), program)))
      throw std::runtime_error("Cannot rename main references");
    main.setName(renameMain);
  }
  if (observeFunction.empty()) return;
  if ((observeTypes.empty() == observeAllocations.empty()) || program.lookupSymbol(observeFunction))
    throw std::runtime_error("Observer needs --observe-type or --observe-allocation and a new callback name");
  SmallVector<Value> values;
  SmallVector<Value> allocations;
  for (auto alloc : main.front().getOps<memref::AllocaOp>())
    allocations.push_back(alloc.getResult());
  for (unsigned index : observeAllocations) {
    if (index >= allocations.size())
      throw std::runtime_error("Observer allocation index is outside the entry allocation list");
    values.push_back(allocations[index]);
  }
  for (const auto &type : observeTypes) {
    SmallVector<Value> matches;
    for (auto alloc : main.front().getOps<memref::AllocaOp>())
      if (printed(alloc.getType()) == type) matches.push_back(alloc.getResult());
    if (matches.size() != 1) throw std::runtime_error("Observer type must identify exactly one entry allocation: " + type);
    values.push_back(matches.front());
  }
  OpBuilder builder(program.getContext());
  builder.setInsertionPointToEnd(program.getBody());
  auto callback = builder.create<func::FuncOp>(main.getLoc(), observeFunction,
      builder.getFunctionType(ValueRange(values).getTypes(), {}));
  callback.setPrivate();
  main.walk([&](func::ReturnOp ret) {
    builder.setInsertionPoint(ret);
    builder.create<func::CallOp>(ret.getLoc(), callback, values);
  });
  if (failed(verify(program))) throw std::runtime_error("Invalid observed program");
}

void fullPipeline(MLIRContext &context) {
  fs::path out = fs::absolute(std::string(outputDir));
  Runner run(out);
  try {
    fs::path source = fs::absolute(std::string(input));
    fs::path scaleisa = scaleisaRoot.empty() ? root / "scaleisa" : fs::absolute(std::string(scaleisaRoot));
    if (!clockPeriod.getNumOccurrences()) {
      auto defaults = json::parse(read(targetProfile.empty()
          ? scaleisa / "resources/scaleisa-hector/resource.json"
          : fs::absolute(std::string(targetProfile))));
      clockPeriod = targetProfile.empty()
          ? defaults.at("_profile").at("clock_ns").get<double>()
          : defaults.at("hardware").at("_profile").at("clock_ns").get<double>();
    }
    if (!std::isfinite(double(clockPeriod)) || clockPeriod <= 0)
      throw std::runtime_error("Scheduling period must be positive and finite");
    fs::path cache = discoveryDir.empty()
        ? scaleisa / "tmp" / ("aps-scaleisa-" + out.parent_path().filename().string() + "-" + out.filename().string())
        : fs::absolute(std::string(discoveryDir));
    fs::path resourcePath = out / "resource.json";
    fs::path contractsPath = resource.empty() ? out / "aps-contracts.json"
                                            : fs::absolute(std::string(resource));
    fs::path profilePath = targetProfile.empty() ? out / "target-profile.json" : fs::absolute(std::string(targetProfile));
    if (targetProfile.empty()) {
      if (resource.empty())
        run.run("resources", {findTool("python", python).string(),
            (root / "hardware/hardfloat/build-hardfloat-ip.py").string(), "--resource-output=" + contractsPath.string()}, false, out);
      auto profile = json::parse(read(root / "hardware/performance/rocket-rv32.json"));
      auto model = json::parse(read(scaleisa / "resources/scaleisa-hector/resource.json"));
      auto actual = json::parse(read(contractsPath));
      for (auto &[name, entry] : model.items())
        if (actual.contains(name))
          for (auto key : {"latency", "II", "amount", "constr"})
            if (actual[name].contains(key)) entry[key] = actual[name][key];
      model["_profile"]["characterization_clock_ns"] = model["_profile"]["clock_ns"];
      model["_profile"]["clock_ns"] = double(clockPeriod);
      model["_profile"]["area_basis"] = "registered ASAP7 primitive proxy, not physical mapped area";
      model["_profile"]["timing_basis"] = "ASAP7 registered primitive required periods; APS execution latency/II contracts";
      model["nop"]["latency"] = json::array({0,0,0,0,0,0,0});
      // TOR needs the APS resource aliases and scalar II fields. Give both
      // schedulers the same physical delay table while retaining those aliases.
      for (auto &[name, entry] : actual.items()) {
        std::string physicalName = name;
        if (llvm::StringRef(name).ends_with("_fp32"))
          physicalName = name.substr(0, name.size() - 5);
        if (name == "fp") physicalName = "addf";
        if (model.contains(physicalName) && model[physicalName].contains("delay"))
          entry["delay"] = model[physicalName]["delay"];
      }
      write(resourcePath, actual.dump(2) + "\n");
      profile["hardware"] = model;
      profile["interface"]["max_extra_csrs"] = unsigned(maxCSRs);
      write(profilePath, profile.dump(2) + "\n");
    } else {
      if (!resource.empty()) throw std::runtime_error("--target-profile supplies resources; omit --resource");
      auto profile = json::parse(read(profilePath));
      if (!profile.at("interface").contains("max_extra_csrs") ||
          profile.at("interface").at("max_extra_csrs").get<unsigned>() != maxCSRs)
        throw std::runtime_error("Target profile max_extra_csrs must match --max-extra-csrs");
      if (profile.at("hardware").at("_profile").at("clock_ns").get<double>() != double(clockPeriod))
        throw std::runtime_error("Target profile clock does not match --clock");
      write(resourcePath, profile.at("hardware").dump(2) + "\n");
    }
    auto profileHash = hash(profilePath);
    run.report["target_profile_sha256"] = profileHash;
    run.report.update({{"input", source.string()}, {"input_sha256", hash(source)},
        {"area_budget", unsigned(areaBudget)}, {"selection_method", std::string(selectionMethod)},
        {"discovery_directory", cache.string()}, {"reused_discovery", !discoveryDir.empty()},
        {"discovery_stages", std::string(discoveryStages)}});
    if (discoveryDir.empty()) {
      fs::create_directories(cache);
      run.run("discovery", {findTool("python", python).string(), (scaleisa / "scripts/scaleisa.py").string(),
          "run", "--stage", "multi-level-stage", "--input", source.string(),
          "--output-dir", cache.string(), "--area-budget", std::to_string(areaBudget),
          "--selection-method", selectionMethod,
          "--target-profile", profilePath.string(),
          "--discovery-stages", discoveryStages,
          "--scaleisa-opt", (build / "tools/scaleisa-opt/scaleisa-opt").string(),
          "--scaleisa-engine-bin", (scaleisa / "tools/scaleisa-engine/target/release/scaleisa-engine").string()}, true);
    }
    auto discovery = json::parse(read(cache / "stage-report.json"));
    if (discovery.at("status") != "completed") throw std::runtime_error("Discovery run is incomplete");
    const auto &cfg = discovery.at("config");
    if (cfg.value("target_profile_sha256", "") != profileHash)
      throw std::runtime_error("Discovery target profile does not match; reevaluate implementations");
    if (fs::weakly_canonical(cfg.at("input_mlir").get<std::string>()) != fs::weakly_canonical(source))
      throw std::runtime_error("Discovery input does not match requested MLIR");
    if (cfg.at("area_budget").get<unsigned>() != areaBudget || cfg.at("selection_method") != std::string(selectionMethod))
      throw std::runtime_error("Discovery selection configuration does not match requested budget/method");
    std::set<std::string> requestedStages;
    for (llvm::StringRef stage : llvm::split(llvm::StringRef(discoveryStages), ','))
      requestedStages.insert(stage.str());
    std::set<std::string> cachedStages;
    for (const auto &stage : cfg.at("discovery_stages"))
      cachedStages.insert(stage.get<std::string>());
    if (requestedStages != cachedStages)
      throw std::runtime_error("Discovery stages do not match requested configuration");
    auto selection = json::parse(read(cache / "selection/selection-report.json"));
    bool softwareOnly = selection.at("selected_impls").empty();
    if (selection.at("selected_area").get<double>() > areaBudget + 1e-6)
      throw std::runtime_error("Selected area exceeds requested budget");
    run.report["discovery_metrics"] = discovery.at("metrics");
    run.report["selection"] = selection;
    run.report["selected_program_sha256"] = hash(cache / "final/selected.mlir");
    run.report["selection_report_sha256"] = hash(cache / "selection/selection-report.json");
    auto selected = parse(cache / "final/selected.mlir", context);
    fs::path candidates = out / "candidates.mlir", software = out / "selected.mlir";
    if (softwareOnly) {
      instrument(*selected);
      write(software, printed(*selected));
      fs::path hardware = out / "hardware.mlir";
      write(hardware, "module {}\n");
      fs::path binary = out / (emit == "elf" ? "program.elf" : "program.o");
      run.report["compiled"] = compileProgram(software, hardware, binary, true);
      if (cosim) {
        run.run("software-simulation", {findTool("spike").string(),
            "--isa=RV32IMAFDC_Zicsr_Zifencei", binary.string()}, true);
      }
      run.report["hardware_selected"] = false;
      run.report["status"] = "passed";
      run.report["cosimulation"] = "not applicable: software-only selection";
      run.report["software_simulation"] = cosim ? "passed" : "not requested";
      run.report["hashes"] = {{software.string(), hash(software)}, {binary.string(), hash(binary)}};
      save(run.reportPath, run.report);
      std::cout << "PASS (software only): " << run.reportPath << std::endl;
      return;
    }
    run.report["hardware_selected"] = true;
    collectKernels(cache, *selected, selection, candidates, run);
    instrument(*selected);
    write(software, printed(*selected));
    fs::path converted = out / "planned.mlir", hardware = out / "hardware.mlir";
    run.run("adapt", {nativeTool("scaleisa-adapter-opt").string(), candidates.string(),
        "--scaleisa-convert=preserve-implementation-pragmas=true",
        "--scaleisa-split=max-extra-csrs=" + std::to_string(maxCSRs),
        "--mlir-print-op-generic", "-o", converted.string()});
    run.run("export-hardware", {nativeTool("scaleisa-adapter-opt").string(), converted.string(),
        "--scaleisa-materialize-aps", "--mlir-print-op-generic", "-o", hardware.string()});
    run.run("synthesis", {(build / "tools/aps-opt/aps-e2e").string(), "--clock", std::to_string(clockPeriod),
        "-i", hardware.string(), "--resource", resourcePath.string(), "--export", "sv",
        "--dump-scheduled-tor", (out / "design.scheduled-tor.mlir").string(),
        "--dump-cmt2", (out / "design.cmt2.mlir").string(), "-o", (out / "design.sv").string()});
    context.getOrLoadDialect<tor::TORDialect>();
    auto scheduled = parse(out / "design.scheduled-tor.mlir", context);
    json pipelineOutcomes = json::array();
    unsigned fallbackCount = 0;
    scheduled->walk([&](Operation *op) {
      if (!op->hasAttr("aps.pipeline_requested")) return;
      json entry{{"operation", op->getName().getStringRef().str()}};
      for (Operation *parent = op->getParentOp(); parent; parent = parent->getParentOp())
        if (auto name = parent->getAttrOfType<StringAttr>("sym_name")) {
          entry["function"] = name.getValue().str();
          break;
        }
      if (auto ii = op->getAttrOfType<IntegerAttr>("aps.pipeline_requested_ii"))
        entry["requested_ii"] = ii.getInt();
      if (auto ii = op->getAttrOfType<IntegerAttr>("aps.pipeline_achieved_ii"))
        entry["achieved_ii"] = ii.getInt();
      auto fallback = op->getAttrOfType<StringAttr>("aps.pipeline_fallback");
      entry["realized_mode"] = fallback ? "sequential" : "pipeline";
      if (fallback) {
        entry["reason"] = fallback.getValue().str();
        ++fallbackCount;
      }
      pipelineOutcomes.push_back(entry);
    });
    run.report["backend_pipeline_outcomes"] = pipelineOutcomes;
    run.report["backend_pipeline_fallback_count"] = fallbackCount;
    if (fallbackCount)
      llvm::errs() << "[aps-scaleisa] warning: " << fallbackCount
                   << " requested pipelines realized sequentially; inspect backend_pipeline_outcomes\n";
    fs::path binary = out / (emit == "elf" ? "program.elf" : "program.o");
    run.report["compiled"] = compileProgram(software, hardware, binary);
    if (cosim) {
      auto hw = parse(hardware, context);
      std::vector<std::string> csrArgs;
      for (auto global : hw->getOps<memref::GlobalOp>()) {
        if (auto address = global->getAttrOfType<IntegerAttr>("csr_address")) {
          csrArgs.push_back("--add-csr");
          csrArgs.push_back(global.getSymName().str() + "=" + std::to_string(address.getInt()));
        }
      }
      fs::path model = out / "simulator";
      run.run("build-simulator", {findTool("make").string(), "-C", (root / "tools/aps-cosim").string(),
          "spike", "VERILATOR=" + findTool("verilator").string(), "RTL=" + (out / "design.sv").string(), "TOP=main", "SPIKE_OBJ_DIR=" + model.string(),
          "VERILATE_JOBS=" + std::to_string(jobs), "BUILD_JOBS=" + std::to_string(jobs)});
      setenv("APS_COSIM_PROGRESS", "1", 1);
      std::vector<std::string> args{(model / "Vmain").string()};
      args.insert(args.end(), csrArgs.begin(), csrArgs.end());
      args.push_back(binary.string());
      run.report["runtime_csr_arguments"] = csrArgs;
      fs::path statistics = out / "ci-runtime.json";
      setenv("APS_COSIM_STATS", statistics.c_str(), 1);
      run.run("simulation", args, true);
      auto measured = json::parse(read(statistics));
      auto estimatesPath = cache / "merged/candidate-implementations.json";
      json estimates = fs::exists(estimatesPath)
          ? json::parse(read(estimatesPath)).at("implementations") : json::array();
      auto interface = json::parse(read(profilePath)).at("interface");
      run.report["ci_cycle_comparison"] = json::array();
      for (auto function : hw->getOps<func::FuncOp>()) {
        auto opcode = function->getAttrOfType<IntegerAttr>("opcode");
        auto funct7 = function->getAttrOfType<IntegerAttr>("funct7");
        if (!opcode || !funct7) continue;
        for (const auto &sample : measured.at("instructions")) {
          if (sample.at("opcode") != opcode.getInt() || sample.at("funct7") != funct7.getInt()) continue;
          json comparison = sample;
          comparison["function"] = function.getName().str();
          for (const auto &estimate : estimates) {
            if (estimate.at("function_name") != function.getName().str()) continue;
            double predicted = estimate.at("latency_cycles");
            auto type = cast<FunctionType>(function->getAttrOfType<TypeAttr>("scaleisa.source_type").getValue());
            unsigned fpMoves = llvm::count_if(type.getInputs(), [](Type t) { return t.isF32(); }) +
                               llvm::count_if(type.getResults(), [](Type t) { return t.isF32(); });
            predicted -= fpMoves * interface.value("fp_register_move_cycles", 0.0);
            predicted -= std::max<int>(0, int(type.getNumInputs()) - 2) * interface.value("csr_write_cycles", 0.0);
            comparison["estimated_au_cycles"] = estimate.at("raw_latency_cycles");
            comparison["estimated_ci_cycles"] = predicted;
            comparison["ci_cycle_error"] = sample.at("mean_cycles").get<double>() - predicted;
          }
          run.report["ci_cycle_comparison"].push_back(comparison);
        }
      }
      run.report["ci_runtime_scope"] = measured.at("scope");
    }
    run.report["hashes"] = json::object();
    for (auto path : {software, candidates, hardware, out / "design.sv", binary})
      run.report["hashes"][path.string()] = hash(path);
    run.report["status"] = "passed";
    run.report["cosimulation"] = cosim ? "passed" : "not requested";
    save(run.reportPath, run.report);
    std::cout << "PASS: " << run.reportPath << std::endl;
  } catch (const std::exception &error) { run.fail(error); throw; }
}
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "APS ScaleISA end-to-end driver\n");
  try {
    if (input.empty()) throw std::runtime_error("Input MLIR is required");
    if (emit != "object" && emit != "elf") throw std::runtime_error("--emit must be object or elf");
    if (abi != "ilp32" && abi != "ilp32f" && abi != "ilp32d") throw std::runtime_error("Unsupported ABI");
    if (memrefABI != "bare-pointer" && memrefABI != "descriptor") throw std::runtime_error("Unsupported memref ABI");
    if (maxCSRs > 4096 || optLevel > 3 || jobs == 0 || areaBudget == 0) throw std::runtime_error("Invalid numeric option");
    bool anyStage = false;
    for (llvm::StringRef stage : llvm::split(llvm::StringRef(discoveryStages), ',')) {
      if (stage != "linalg" && stage != "affine" && stage != "scf")
        throw std::runtime_error("--discovery-stages requires linalg, affine, or scf");
      anyStage = true;
    }
    if (!anyStage) throw std::runtime_error("At least one discovery stage is required");
    if (selectionMethod != "greedy" && selectionMethod != "cbc" && selectionMethod != "eboost")
      throw std::runtime_error("--selection-method must be greedy, cbc, or eboost");
    if (!compileOnly && (!hardwareFile.empty() || !output.empty()))
      throw std::runtime_error("Use --compile-only for --hardware/-o, or --output-dir for a full run");
    if (compileOnly && (cosim || !outputDir.empty() || !discoveryDir.empty()))
      throw std::runtime_error("--compile-only cannot run discovery or cosimulation");
    if (emit == "object" && (!linkObjects.empty() || !linkSources.empty() || !linkArgs.empty()))
      throw std::runtime_error("Runtime and linker inputs require --emit=elf");
    if (cosim && emit != "elf") throw std::runtime_error("--cosim requires --emit=elf and supplied runtime objects");
    fs::path pixi = root / ".pixi/envs/default";
    if (fs::is_directory(pixi)) {
      std::string path = (pixi / "bin").string() + ":" +
                         (std::getenv("PATH") ? std::getenv("PATH") : "");
      setenv("PATH", path.c_str(), 1);
      std::string libs = (pixi / "lib").string() + ":" +
                         (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "");
      setenv("LD_LIBRARY_PATH", libs.c_str(), 1);
    }
    if (!std::getenv("MLIR_BINDINGS")) setenv("MLIR_BINDINGS", (root / "circt/build/tools/mlir/python_packages/mlir_core").c_str(), 1);
    if (compileOnly) {
      if (hardwareFile.empty() || output.empty()) throw std::runtime_error("--compile-only requires --hardware and -o");
      bool softwareOnly = false;
      if (allowNoCI) {
        DialectRegistry registry;
        registry.insert<aps::APSDialect, func::FuncDialect, arith::ArithDialect,
            memref::MemRefDialect, affine::AffineDialect, scf::SCFDialect,
            math::MathDialect, LLVM::LLVMDialect, linalg::LinalgDialect,
            tensor::TensorDialect>();
        MLIRContext context(registry);
        auto hardware = parse(fs::absolute(std::string(hardwareFile)), context);
        softwareOnly = hardware->getBody()->empty();
      }
      compileProgram(fs::absolute(std::string(input)), fs::absolute(std::string(hardwareFile)), fs::absolute(std::string(output)), softwareOnly);
    } else {
      if (outputDir.empty()) throw std::runtime_error("Full flow requires --output-dir");
      DialectRegistry registry;
      registry.insert<aps::APSDialect, affine::AffineDialect, arith::ArithDialect,
          func::FuncDialect, LLVM::LLVMDialect, linalg::LinalgDialect,
          math::MathDialect, memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect>();
      MLIRContext context(registry);
      fullPipeline(context);
    }
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "aps-scaleisa-opt: " << error.what() << "\n";
    return 1;
  }
}
