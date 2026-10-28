#include "scaleisa-compiler/Passes.h"
#include "APS/APSDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <set>

namespace mlir::scaleisa_compiler {
namespace {
struct Instruction {
  FunctionType type;
  unsigned opcode, funct3, funct7;
  SmallVector<DictionaryAttr> bindings;
};

bool carrier(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.isSignless() && integer.getWidth() <= 32;
  return type.isIndex() || type.isF32() || isa<MemRefType>(type);
}

struct LowerCIPass : PassWrapper<LowerCIPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCIPass)
  LowerCIPass() = default;
  LowerCIPass(const LowerCIPass &other) : PassWrapper(other) {}
  Option<std::string> hardwareFile{*this, "hardware-file",
      llvm::cl::desc("APS export providing exact CI encodings and operand bindings"),
      llvm::cl::init("")};
  Option<unsigned> maxExtraCSRs{*this, "max-extra-csrs", llvm::cl::init(5)};
  Option<unsigned> dmaAlignment{*this, "dma-alignment",
      llvm::cl::desc("Required CPU buffer alignment for APS DMA bursts"),
      llvm::cl::init(64)};
  StringRef getArgument() const final { return "scaleisa-lower-ci"; }
  StringRef getDescription() const final {
    return "Replace exposed ScaleISA CI calls with RV32 CSR setup and custom instructions";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aps::APSDialect, affine::AffineDialect,
                    arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  llvm::StringMap<Instruction> instructions;
  llvm::StringMap<DictionaryAttr> sequences;
  uint64_t sites = 0, emitted = 0;

  LogicalResult readHardware(ModuleOp hardware) {
    std::set<std::pair<unsigned, unsigned>> encodings;
    for (auto function : hardware.getOps<func::FuncOp>()) {
      if (!function->hasAttr("scaleisa.hardware"))
        continue;
      auto source = function->getAttrOfType<TypeAttr>("scaleisa.source_type");
      auto type = source ? dyn_cast<FunctionType>(source.getValue()) : FunctionType();
      auto bindings = function->getAttrOfType<ArrayAttr>("scaleisa.operand_bindings");
      auto opcode = function->getAttrOfType<IntegerAttr>("opcode");
      auto funct3 = function->getAttrOfType<IntegerAttr>("funct3");
      auto funct7 = function->getAttrOfType<IntegerAttr>("funct7");
      if (!type || !bindings || !opcode || !funct3 || !funct7 ||
          type.getNumResults() > 1 || bindings.size() != type.getNumInputs() ||
          llvm::any_of(type.getInputs(), [](Type t) { return !carrier(t); }) ||
          llvm::any_of(type.getResults(), [](Type t) {
            return !carrier(t) || isa<MemRefType>(t);
          }))
        return function.emitError("invalid APS CI source type or operand bindings");
      int64_t op = opcode.getInt(), f3 = funct3.getInt(), f7 = funct7.getInt();
      unsigned expected = (type.getNumResults() ? 4 : 0) |
                          (type.getNumInputs() ? 2 : 0) |
                          (type.getNumInputs() > 1 ? 1 : 0);
      if ((op != 0x0b && op != 0x2b && op != 0x5b && op != 0x7b) ||
          f3 != expected || f7 < 0 || f7 > 127 ||
          !encodings.emplace(op, f7).second)
        return function.emitError("invalid or duplicate APS CI encoding");
      Instruction instruction{type, unsigned(op), unsigned(f3), unsigned(f7), {}};
      std::set<int64_t> csrAddresses;
      for (auto [index, attr] : llvm::enumerate(bindings)) {
        auto binding = dyn_cast<DictionaryAttr>(attr);
        auto argument = binding ? binding.getAs<IntegerAttr>("arg_index") : IntegerAttr();
        auto kind = binding ? binding.getAs<StringAttr>("kind") : StringAttr();
        StringRef expectedKind = index == 0 ? "rs1" : index == 1 ? "rs2" : "csr";
        if (!argument || argument.getInt() != int64_t(index) || !kind ||
            kind.getValue() != expectedKind)
          return function.emitError("CI operand bindings must follow the APS source argument order");
        if (index >= 2) {
          auto address = binding.getAs<IntegerAttr>("address");
          if (!address || address.getInt() < 0 || address.getInt() >= 0xc00 ||
              !csrAddresses.insert(address.getInt()).second)
            return function.emitError("CI requires distinct writable 12-bit CSR addresses");
        }
        if (isa<MemRefType>(type.getInput(index))) {
          auto memoryType = binding.getAs<TypeAttr>("memref_type");
          auto valueKind = binding.getAs<StringAttr>("value_kind");
          if (!memoryType || memoryType.getValue() != type.getInput(index) ||
              !valueKind || valueKind.getValue() != "cpu_aligned_base")
            return function.emitError("CPU memref CI binding must specify its exact aligned-base type");
        }
        instruction.bindings.push_back(binding);
      }
      if (csrAddresses.size() > maxExtraCSRs)
        return function.emitError("CI exceeds max-extra-csrs");
      instructions.try_emplace(function.getName(), std::move(instruction));
    }
    if (instructions.empty())
      return hardware.emitError("hardware file contains no APS exported instructions");
    if (auto attrs = hardware->getAttrOfType<ArrayAttr>("scaleisa.sequences")) {
      for (Attribute attr : attrs) {
        auto sequence = dyn_cast<DictionaryAttr>(attr);
        auto source = sequence ? sequence.getAs<StringAttr>("source") : StringAttr();
        auto typeAttr = sequence ? sequence.getAs<TypeAttr>("source_type") : TypeAttr();
        auto type = typeAttr ? dyn_cast<FunctionType>(typeAttr.getValue()) : FunctionType();
        auto steps = sequence ? sequence.getAs<ArrayAttr>("steps") : ArrayAttr();
        auto result = sequence ? sequence.getAs<DictionaryAttr>("result") : DictionaryAttr();
        if (!source || !type || type.getNumResults() != 1 || !steps || !result ||
            instructions.count(source.getValue()) || sequences.count(source.getValue()))
          return hardware.emitError("invalid or ambiguous split CI sequence");
        SmallVector<Type> results;
        auto bindingType = [&](DictionaryAttr binding) -> Type {
          if (!binding || binding.size() != 1)
            return {};
          if (auto arg = binding.getAs<IntegerAttr>("argument")) {
            if (arg.getInt() >= 0 && uint64_t(arg.getInt()) < type.getNumInputs())
              return type.getInput(arg.getInt());
          } else if (auto step = binding.getAs<IntegerAttr>("step")) {
            if (step.getInt() >= 0 && uint64_t(step.getInt()) < results.size())
              return results[step.getInt()];
          } else if (auto value = binding.getAs<TypedAttr>("constant")) {
            return value.getType();
          }
          return {};
        };
        for (Attribute stepAttr : steps) {
          auto step = dyn_cast<DictionaryAttr>(stepAttr);
          auto callee = step ? step.getAs<FlatSymbolRefAttr>("callee") : FlatSymbolRefAttr();
          auto inputs = step ? step.getAs<ArrayAttr>("inputs") : ArrayAttr();
          auto found = callee ? instructions.find(callee.getValue()) : instructions.end();
          if (found == instructions.end() || !inputs ||
              inputs.size() != found->second.type.getNumInputs() ||
              found->second.type.getNumResults() != 1)
            return hardware.emitError("invalid split CI sequence step");
          for (auto [input, expectedType] : llvm::zip(inputs, found->second.type.getInputs()))
            if (bindingType(dyn_cast<DictionaryAttr>(input)) != expectedType)
              return hardware.emitError("split CI sequence has an invalid argument, forward reference, or type");
          results.push_back(found->second.type.getResult(0));
        }
        if (bindingType(result) != type.getResult(0))
          return hardware.emitError("split CI sequence result type mismatch");
        sequences.try_emplace(source.getValue(), sequence);
      }
    }
    return success();
  }

  // Native ScaleISA names each exposed selected helper <implementation>_site_N.
  // Use that explicit identity, never pattern-match the software computation.
  std::string resolve(StringRef name) {
    if (instructions.count(name) || sequences.count(name))
      return name.str();
    auto split = name.rsplit("_site_");
    if (!split.second.empty() &&
        llvm::all_of(split.second, [](char c) { return c >= '0' && c <= '9'; }) &&
        (instructions.count(split.first) || sequences.count(split.first)))
      return split.first.str();
    return {};
  }

  FunctionType sourceType(StringRef name) {
    auto found = instructions.find(name);
    if (found != instructions.end())
      return found->second.type;
    return cast<FunctionType>(sequences.find(name)->second.getAs<TypeAttr>("source_type").getValue());
  }

  // APS's default bus interface emits naturally aligned 64-byte bursts.
  // memref's "aligned pointer" only promises the element alignment unless an
  // allocation explicitly requests more. Strengthen compiler-owned buffers,
  // including buffers forwarded through ordinary software helper calls.
  void alignDMABuffer(Value value, llvm::SmallPtrSetImpl<Operation *> &visited) {
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
      if (!function || argument.getOwner() != &function.front() ||
          !visited.insert(function).second)
        return;
      getOperation().walk([&](func::CallOp call) {
        if (call.getCallee() == function.getName())
          alignDMABuffer(call.getOperand(argument.getArgNumber()), visited);
      });
      return;
    }
    Operation *definition = value.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      return;
    if (auto cast = dyn_cast<memref::CastOp>(definition)) {
      alignDMABuffer(cast.getSource(), visited);
      return;
    }
    if (auto global = dyn_cast<memref::GetGlobalOp>(definition))
      definition = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
          global, global.getNameAttr());
    if (!definition || !isa<memref::AllocaOp, memref::AllocOp, memref::GlobalOp>(definition))
      return;
    auto alignment = definition->getAttrOfType<IntegerAttr>("alignment");
    if (!alignment || alignment.getInt() < dmaAlignment)
      definition->setAttr("alignment", IntegerAttr::get(
          IntegerType::get(&getContext(), 64), dmaAlignment));
  }

  Value pack(OpBuilder &builder, Location loc, Value value) {
    auto i32 = builder.getI32Type();
    Type type = value.getType();
    if (isa<MemRefType>(type)) {
      llvm::SmallPtrSet<Operation *, 16> visited;
      alignDMABuffer(value, visited);
      value = builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, value);
      return builder.create<arith::IndexCastUIOp>(loc, i32, value);
    }
    if (type.isIndex())
      return builder.create<arith::IndexCastUIOp>(loc, i32, value);
    if (type.isF32())
      return builder.create<arith::BitcastOp>(loc, i32, value);
    if (type == i32)
      return value;
    return builder.create<arith::ExtUIOp>(loc, i32, value);
  }

  Value emitInstruction(OpBuilder &builder, Location loc, StringRef name,
                        ValueRange values) {
    auto &instruction = instructions.find(name)->second;
    SmallVector<Value> operands;
    for (Value value : values)
      operands.push_back(pack(builder, loc, value));
    bool hasResult = instruction.type.getNumResults();
    unsigned offset = hasResult ? 1 : 0;
    auto operand = [&](unsigned index) {
      return "$" + std::to_string(index + offset);
    };
    std::string assembly;
    bool memory = llvm::any_of(values, [](Value value) { return isa<MemRefType>(value.getType()); });
    if (memory)
      assembly += "fence rw, rw\n";
    for (unsigned index = 2; index < operands.size(); ++index) {
      auto address = instruction.bindings[index].getAs<IntegerAttr>("address").getInt();
      assembly += "csrrw x0, " + std::to_string(address) + ", " + operand(index) + "\n";
    }
    assembly += ".insn r " + std::to_string(instruction.opcode) + ", " +
        std::to_string(instruction.funct3) + ", " + std::to_string(instruction.funct7) +
        ", " + (hasResult ? "$0" : "x0") + ", " +
        (operands.empty() ? "x0" : operand(0)) + ", " +
        (operands.size() < 2 ? "x0" : operand(1));
    if (memory)
      assembly += "\nfence rw, rw";
    std::string constraints = hasResult ? "=r" : "";
    for (unsigned index = 0; index < operands.size(); ++index)
      constraints += constraints.empty() ? "r" : ",r";
    constraints += constraints.empty() ? "~{memory}" : ",~{memory}";
    Type resultType = hasResult ? Type(builder.getI32Type()) : Type();
    auto op = builder.create<LLVM::InlineAsmOp>(loc, resultType, operands,
        assembly, constraints, true, false, LLVM::tailcallkind::TailCallKind(),
        LLVM::AsmDialectAttr{}, ArrayAttr{});
    ++emitted;
    if (!hasResult)
      return {};
    Value result = op.getResult(0);
    Type type = instruction.type.getResult(0);
    if (type.isF32())
      return builder.create<arith::BitcastOp>(loc, type, result);
    if (type.isIndex())
      return builder.create<arith::IndexCastUIOp>(loc, type, result);
    if (type != builder.getI32Type())
      return builder.create<arith::TruncIOp>(loc, type, result);
    return result;
  }

  Value emit(OpBuilder &builder, Location loc, StringRef name, ValueRange values) {
    if (instructions.count(name))
      return emitInstruction(builder, loc, name, values);
    auto sequence = sequences.find(name)->second;
    SmallVector<Value> results;
    auto binding = [&](DictionaryAttr input) -> Value {
      if (auto argument = input.getAs<IntegerAttr>("argument"))
        return values[argument.getInt()];
      if (auto step = input.getAs<IntegerAttr>("step"))
        return results[step.getInt()];
      return builder.create<arith::ConstantOp>(loc, input.getAs<TypedAttr>("constant"));
    };
    for (Attribute attr : sequence.getAs<ArrayAttr>("steps")) {
      auto step = cast<DictionaryAttr>(attr);
      SmallVector<Value> operands;
      for (Attribute input : step.getAs<ArrayAttr>("inputs"))
        operands.push_back(binding(cast<DictionaryAttr>(input)));
      results.push_back(emitInstruction(builder, loc,
          step.getAs<FlatSymbolRefAttr>("callee").getValue(), operands));
    }
    return binding(sequence.getAs<DictionaryAttr>("result"));
  }

  void runOnOperation() override {
    instructions.clear();
    sequences.clear();
    sites = emitted = 0;
    if (!dmaAlignment || (dmaAlignment & (dmaAlignment - 1))) {
      getOperation().emitError("dma-alignment must be a positive power of two");
      return signalPassFailure();
    }
    if (hardwareFile.empty() || maxExtraCSRs > 4096) {
      getOperation().emitError("hardware-file is required and max-extra-csrs must fit the CSR address space");
      return signalPassFailure();
    }
    auto hardware = parseSourceFile<ModuleOp>(hardwareFile, &getContext());
    if (!hardware || failed(readHardware(*hardware)))
      return signalPassFailure();
    ModuleOp module = getOperation();
    llvm::StringSet<> helpers;
    llvm::StringMap<std::string> aliases;
    // An explicit call-site binding can identify a renamed selected helper.
    // Skip its software body too; require every remaining use to be lowered.
    llvm::StringSet<> explicitlyBound;
    module.walk([&](func::CallOp call) {
      auto binding = call->getAttrOfType<FlatSymbolRefAttr>("scaleisa.ci");
      if (binding && !resolve(binding.getValue()).empty())
        explicitlyBound.insert(call.getCallee());
    });
    for (auto function : module.getOps<func::FuncOp>()) {
      auto explicitCI = function->getAttrOfType<FlatSymbolRefAttr>("scaleisa.ci");
      std::string target = resolve(explicitCI ? explicitCI.getValue() : function.getName());
      if (explicitCI && target.empty()) {
        function.emitError("unknown explicit scaleisa.ci binding");
        return signalPassFailure();
      }
      if (target.empty()) {
        if (explicitlyBound.contains(function.getName())) {
          helpers.insert(function.getName());
          continue;
        }
        auto generated = function->getAttrOfType<BoolAttr>("au.generated");
        if ((generated && generated.getValue()) || function->hasAttr("scaleisa.kernel")) {
          function.emitError("selected CI helper has no matching hardware implementation");
          return signalPassFailure();
        }
        continue;
      }
      if (function.getFunctionType() != sourceType(target)) {
        function.emitError("selected CI signature differs from the APS source type; do not guess or reorder arguments");
        return signalPassFailure();
      }
      helpers.insert(function.getName());
      aliases[function.getName()] = target;
    }
    SmallVector<std::pair<func::CallOp, std::string>> calls;
    bool invalid = false;
    module.walk([&](func::CallOp call) {
      auto parent = call->getParentOfType<func::FuncOp>();
      if (parent && helpers.contains(parent.getName()))
        return;
      auto explicitCI = call->getAttrOfType<FlatSymbolRefAttr>("scaleisa.ci");
      auto alias = aliases.find(call.getCallee());
      std::string target = explicitCI ? resolve(explicitCI.getValue()) :
          alias != aliases.end() ? alias->second : resolve(call.getCallee());
      if (target.empty()) {
        if (explicitCI) {
          call.emitError("unknown explicit scaleisa.ci binding");
          invalid = true;
        }
        return;
      }
      auto type = FunctionType::get(&getContext(), call.getOperandTypes(), call.getResultTypes());
      if (type != sourceType(target)) {
        call.emitError("CI call does not match the exact hardware source signature");
        invalid = true;
      } else {
        calls.emplace_back(call, target);
      }
    });
    if (invalid)
      return signalPassFailure();
    for (auto &[call, target] : calls) {
      OpBuilder builder(call);
      Value result = emit(builder, call.getLoc(), target, call.getOperands());
      if (call.getNumResults())
        call.getResult(0).replaceAllUsesWith(result);
      call.erase();
      ++sites;
    }
    for (auto function : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      if (!helpers.contains(function.getName()))
        continue;
      // Address-taken helpers cannot be replaced by direct instruction sites.
      auto uses = SymbolTable::getSymbolUses(function, module);
      if (!uses || llvm::any_of(*uses, [&](const SymbolTable::SymbolUse &use) {
            auto parent = use.getUser()->getParentOfType<func::FuncOp>();
            return !parent || !helpers.contains(parent.getName());
          })) {
        function.emitError("CI helper has a non-call use; indirect CI dispatch requires explicit legalization");
        return signalPassFailure();
      }
    }
    for (auto function : llvm::make_early_inc_range(module.getOps<func::FuncOp>()))
      if (helpers.contains(function.getName()))
        function.erase();
    module->removeAttr("scaleisa.sequences");
    OpBuilder builder(&getContext());
    module->setAttr("scaleisa.ci_sites", builder.getI64IntegerAttr(sites));
    module->setAttr("scaleisa.ci_instructions", builder.getI64IntegerAttr(emitted));
  }
};
} // namespace
std::unique_ptr<Pass> createLowerCIPass() { return std::make_unique<LowerCIPass>(); }
void registerScaleISACompilerPasses() { PassRegistration<LowerCIPass>(); }
} // namespace mlir::scaleisa_compiler
