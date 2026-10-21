#include "scaleisa-adapter/Passes.h"
#include "scaleisa-adapter/MemoryABI.h"
#include "Utils.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include <algorithm>

namespace mlir::scaleisa_adapter {
#define GEN_PASS_DEF_SCALEISAMATERIALIZEAPS
#include "scaleisa-adapter/Passes.h.inc"
namespace {
bool isCarrierType(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.isSignless() && integer.getWidth() <= 32;
  return type.isIndex() || type.isF32();
}

Value unpack(OpBuilder &builder, Location loc, Value bits, Type type) {
  if (type.isF32())
    return builder.create<arith::BitcastOp>(loc, type, bits);
  if (type.isIndex())
    return builder.create<arith::IndexCastUIOp>(loc, type, bits);
  if (type.isInteger(32))
    return bits;
  return builder.create<arith::TruncIOp>(loc, type, bits);
}
Value pack(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isF32())
    return builder.create<arith::BitcastOp>(loc, builder.getI32Type(), value);
  if (value.getType().isIndex())
    return builder.create<arith::IndexCastUIOp>(loc, builder.getI32Type(), value);
  if (value.getType().isInteger(32))
    return value;
  return builder.create<arith::ExtUIOp>(loc, builder.getI32Type(), value);
}

struct MaterializeAPSPass : impl::ScaleISAMaterializeAPSBase<MaterializeAPSPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aps::APSDialect, arith::ArithDialect, memref::MemRefDialect,
                    func::FuncDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> instructions;
    uint64_t totalCSRs = 0;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!isInstruction(function))
        continue;
      auto limit = function->getAttrOfType<IntegerAttr>("scaleisa.max_extra_csrs");
      unsigned count = function.getNumArguments();
      if (!limit || limit.getInt() < 0 || limit.getInt() > 4096 ||
          count > 2 + limit.getInt() || function.getNumResults() > 1 ||
          !llvm::hasSingleElement(function.getBody()) ||
          llvm::any_of(function.getArgumentTypes(), [](Type t) {
            return !isCarrierType(t) && !isa<MemRefType>(t);
          }) ||
          llvm::any_of(function.getResultTypes(), [](Type t) { return !isCarrierType(t); })) {
        function.emitError("APS export requires planned RV32 scalar or CPU-memref inputs and at most one scalar result");
        return signalPassFailure();
      }
      for (BlockArgument argument : function.getArguments())
        if (isa<MemRefType>(argument.getType()) &&
            failed(validateCPUMemoryArgument(argument)))
          return signalPassFailure();
      if (failed(validateBufferedMemoryAliasing(function)))
        return signalPassFailure();
      auto uses = SymbolTable::getSymbolUses(function);
      if (!uses || !uses->empty()) {
        function.emitError("instruction export requires a self-contained body without external symbol references");
        return signalPassFailure();
      }
      totalCSRs += count > 2 ? count - 2 : 0;
      instructions.push_back(function);
    }
    if (instructions.empty()) {
      module.emitError("no planned instructions; run scaleisa-split first");
      return signalPassFailure();
    }
    if ((opcode != 0x0b && opcode != 0x2b && opcode != 0x5b && opcode != 0x7b) ||
        funct7Base > 127 || instructions.size() > 128 - funct7Base) {
      module.emitError("custom opcode or funct7 allocation is out of range");
      return signalPassFailure();
    }
    if (totalCSRs && (csrBase >= 0xc00 || totalCSRs > 0xc00 - csrBase)) {
      module.emitError("dedicated CSR allocation exceeds writable 12-bit addresses");
      return signalPassFailure();
    }

    // Build the complete export before replacing the input module. No software
    // caller is left with the hardware register-selector function signature.
    OwningOpRef<ModuleOp> output(ModuleOp::create(module.getLoc()));
    SymbolTable symbols(*output);
    SmallVector<func::FuncOp> exported;
    // Reserve every function name before allocating CSR symbols.
    for (auto source : instructions) {
      OpBuilder builder(&getContext());
      auto selector = builder.getIntegerType(5);
      auto function = func::FuncOp::create(source.getLoc(), source.getName(),
          builder.getFunctionType({selector, selector, selector}, {}));
      symbols.insert(function);
      exported.push_back(function);
    }
    unsigned nextCSR = csrBase, nextFunct = funct7Base;
    for (auto [source, function] : llvm::zip(instructions, exported)) {
      OpBuilder builder(&getContext());
      auto loc = source.getLoc();
      function->setAttr("opcode", builder.getI32IntegerAttr(opcode));
      function->setAttr("funct7", builder.getI32IntegerAttr(nextFunct++));
      unsigned funct3 = (source.getNumResults() ? 4 : 0) |
                        (source.getNumArguments() ? 2 : 0) |
                        (source.getNumArguments() > 1 ? 1 : 0);
      function->setAttr("funct3", builder.getI32IntegerAttr(funct3));
      function->setAttr("scaleisa.source_type", TypeAttr::get(source.getFunctionType()));
      function->setAttr("scaleisa.hardware", builder.getUnitAttr());
      SmallVector<Attribute> bindings;
      Block *entry = function.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      // Capture both register operands before unpacking values or staging any
      // scratchpad copies. Later memory lowering inserts copies at the handles.
      SmallVector<Value, 2> registerInputs;
      for (unsigned index = 0; index < std::min(2u, source.getNumArguments()); ++index)
        registerInputs.push_back(builder.create<aps::ReadIRF>(
            loc, builder.getI32Type(), entry->getArgument(index)));
      IRMapping mapping;
      SmallVector<std::pair<UnrealizedConversionCastOp, Value>> cpuMemrefs;
      for (auto [index, argument] : llvm::enumerate(source.getArguments())) {
        NamedAttrList binding;
        binding.append("arg_index", builder.getI32IntegerAttr(index));
        Value bits;
        if (index < 2) {
          bits = registerInputs[index];
          binding.append("kind", builder.getStringAttr(index == 0 ? "rs1" : "rs2"));
        } else {
          OpBuilder globalBuilder(output->getBodyRegion());
          globalBuilder.setInsertionPointToEnd(output->getBody());
          auto global = globalBuilder.create<memref::GlobalOp>(loc,
              builder.getStringAttr(source.getName().str() + "_csr" + std::to_string(index - 2)),
              builder.getStringAttr("private"),
              TypeAttr::get(MemRefType::get({}, builder.getI32Type())),
              builder.getUnitAttr(), UnitAttr(), IntegerAttr());
          symbols.insert(global);
          global->setAttr("csr_address", builder.getI32IntegerAttr(nextCSR));
          bits = builder.create<aps::ReadCSR>(loc, builder.getI32Type(), global.getSymName());
          binding.append("kind", builder.getStringAttr("csr"));
          binding.append("address", builder.getI32IntegerAttr(nextCSR++));
        }
        if (isa<MemRefType>(argument.getType())) {
          binding.append("value_kind", builder.getStringAttr("cpu_aligned_base"));
          binding.append("memref_type", TypeAttr::get(argument.getType()));
          auto type = cast<MemRefType>(argument.getType());
          binding.append("memory_mode", builder.getStringAttr(
              type.getRank() ? "scratchpad" : "scalar"));
          if (auto noalias = source.getArgAttr(index, "scaleisa.noalias"))
            binding.append("noalias", noalias);
          // A temporary typed handle permits cloning nested regions without
          // creating cross-function SSA references. It is removed below.
          auto handle = builder.create<UnrealizedConversionCastOp>(
              loc, TypeRange{argument.getType()}, ValueRange{bits});
          mapping.map(argument, handle.getResult(0));
          cpuMemrefs.emplace_back(handle, bits);
        } else {
          mapping.map(argument, unpack(builder, loc, bits, argument.getType()));
        }
        bindings.push_back(builder.getDictionaryAttr(binding));
      }
      function->setAttr("scaleisa.operand_bindings", builder.getArrayAttr(bindings));
      for (Operation &op : source.front().without_terminator())
        builder.clone(op, mapping);
      auto terminator = cast<func::ReturnOp>(source.front().getTerminator());
      if (source.getNumResults()) {
        Value bits = pack(builder, loc, mapping.lookupOrDefault(terminator.getOperand(0)));
        builder.create<aps::WriteIRF>(loc, entry->getArgument(2), bits);
      }
      builder.create<func::ReturnOp>(loc);
      if (failed(hoistScratchpadAllocations(function, symbols)))
        return signalPassFailure();
      for (auto [handle, base] : cpuMemrefs) {
        if (failed(lowerCPUMemoryArgument(handle.getResult(0), base, symbols)))
          return signalPassFailure();
        handle.erase();
      }
    }
    module.getBodyRegion().takeBody(output->getBodyRegion());
  }
};
} // namespace
std::unique_ptr<Pass> createMaterializeAPSPass() {
  return std::make_unique<MaterializeAPSPass>();
}
} // namespace mlir::scaleisa_adapter
