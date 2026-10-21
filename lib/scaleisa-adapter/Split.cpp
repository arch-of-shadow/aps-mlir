#include "scaleisa-adapter/Passes.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

namespace mlir::scaleisa_adapter {
#define GEN_PASS_DEF_SCALEISASPLIT
#include "scaleisa-adapter/Passes.h.inc"

namespace {
struct Group {
  SmallVector<Operation *> ops;
  Operation *root;
};

bool isConstant(Value value) {
  return value.getDefiningOp<arith::ConstantOp>() != nullptr;
}

SmallVector<Value> inputs(const Group &group) {
  llvm::SmallPtrSet<Operation *, 16> members(group.ops.begin(), group.ops.end());
  llvm::SetVector<Value> values;
  for (Operation *op : group.ops)
    for (Value value : op->getOperands())
      if (!isConstant(value) && !members.contains(value.getDefiningOp()))
        values.insert(value);
  return SmallVector<Value>(values.begin(), values.end());
}

// A join of computed branches is its own instruction. Do not hide one
// branch inside its consumer while passing the other through a CSR.
bool isBranchJoin(Operation *op) {
  unsigned computed = 0;
  for (Value operand : op->getOperands())
    if (operand.getDefiningOp() && !isConstant(operand))
      ++computed;
  return computed > 1;
}

bool hasSingleExit(const Group &group) {
  llvm::SmallPtrSet<Operation *, 16> members(group.ops.begin(), group.ops.end());
  for (Operation *op : group.ops)
    if (op != group.root)
      for (Operation *user : op->getUsers())
        if (!members.contains(user))
          return false;
  return true;
}

void markInstruction(func::FuncOp function, unsigned csrLimit) {
  OpBuilder builder(function.getContext());
  unsigned count = function.getNumArguments();
  function->setAttr("scaleisa.instruction", builder.getUnitAttr());
  function->setAttr("scaleisa.operand_count", builder.getI32IntegerAttr(count));
  function->setAttr("scaleisa.csr_count",
                    builder.getI32IntegerAttr(count > 2 ? count - 2 : 0));
  function->setAttr("scaleisa.max_extra_csrs", builder.getI32IntegerAttr(csrLimit));
}

struct SplitPass : impl::ScaleISASplitBase<SplitPass> {
  SplitPass() = default;
  explicit SplitPass(unsigned limit) { maxExtraCSRs = limit; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect>();
  }
  void runOnOperation() override {
    if (maxExtraCSRs > 4096) {
      getOperation().emitError("max-extra-csrs exceeds the CSR address space");
      return signalPassFailure();
    }
    SmallVector<func::FuncOp> candidates;
    for (auto function : getOperation().getOps<func::FuncOp>())
      if (isCandidate(function))
        candidates.push_back(function);
    for (auto function : candidates)
      if (failed(split(function)))
        return signalPassFailure();
  }

  LogicalResult split(func::FuncOp function) {
    if (!function->hasAttr("scaleisa.converted"))
      return function.emitError("run scaleisa-convert before instruction splitting");
    unsigned capacity = 2 + maxExtraCSRs;
    if (function.getNumArguments() <= capacity) {
      markInstruction(function, maxExtraCSRs);
      return success();
    }
    if (!llvm::hasSingleElement(function.getBody()) ||
        function.getNumResults() != 1)
      return function.emitError("oversized candidates require single-block, single-result scalar graphs");
    auto uses = SymbolTable::getSymbolUses(function, getOperation());
    if (!uses || !uses->empty())
      return function.emitError("cannot split a referenced candidate; extract the candidate first and consume scaleisa.sequences for CPU instruction emission");
    SmallVector<Group> groups;
    Block &body = function.front();
    for (Operation &op : body.without_terminator()) {
      if (isa<arith::ConstantOp>(op))
        continue;
      if (op.getNumRegions() || op.getNumResults() != 1 ||
          !isMemoryEffectFree(&op) ||
          !op.getResult(0).getType().isIntOrIndexOrFloat() ||
          llvm::any_of(op.getOperandTypes(), [](Type type) {
            return !type.isIntOrIndexOrFloat();
          }))
        return op.emitError("cannot split memory effects, regions, or non-scalar operations; legalize the memory/loop ABI first");
      groups.push_back({{&op}, &op});
      if (inputs(groups.back()).size() > capacity)
        return op.emitError("one operation exceeds the instruction operand limit");
    }

    // Fuse connected single-exit groups. Keeping every non-root result internal
    // prevents duplicating shared computations or introducing cycles in the DAG.
    while (true) {
      size_t bestSize = 0, bestProducer = 0, bestConsumer = 0;
      Group best;
      for (size_t i = 0; i < groups.size(); ++i) {
        for (size_t j = 0; j < groups.size(); ++j) {
          if (i == j || isBranchJoin(groups[i].root) ||
              isBranchJoin(groups[j].root) ||
              !llvm::is_contained(inputs(groups[j]),
                                          groups[i].root->getResult(0)))
            continue;
          Group merged = groups[j];
          llvm::append_range(merged.ops, groups[i].ops);
          llvm::sort(merged.ops, [](Operation *a, Operation *b) {
            return a->isBeforeInBlock(b);
          });
          if (merged.ops.size() <= bestSize || !hasSingleExit(merged) ||
              inputs(merged).size() > capacity)
            continue;
          bestSize = merged.ops.size();
          bestProducer = i;
          bestConsumer = j;
          best = std::move(merged);
        }
      }
      if (!bestSize)
        break;
      groups[bestConsumer] = std::move(best);
      groups.erase(groups.begin() + bestProducer);
    }

    SymbolTable symbols(getOperation());
    SmallVector<func::FuncOp> helpers;
    for (auto [index, group] : llvm::enumerate(groups)) {
      SmallVector<Value> boundary = inputs(group);
      SmallVector<Type> types;
      for (Value value : boundary)
        types.push_back(value.getType());
      auto helper = func::FuncOp::create(
          function.getLoc(), function.getName().str() + "__ci" + std::to_string(index),
          FunctionType::get(&getContext(), types,
                            {group.root->getResult(0).getType()}));
      symbols.insert(helper);
      helper->setAttr("scaleisa.kernel", UnitAttr::get(&getContext()));
      helper->setAttr("scaleisa.converted", UnitAttr::get(&getContext()));
      helper->setAttr("scaleisa.origin", StringAttr::get(&getContext(), function.getName()));
      markInstruction(helper, maxExtraCSRs);
      Block *entry = helper.addEntryBlock();
      OpBuilder builder = OpBuilder::atBlockBegin(entry);
      IRMapping mapping;
      for (auto [value, argument] : llvm::zip(boundary, entry->getArguments()))
        mapping.map(value, argument);
      for (Operation *op : group.ops) {
        for (Value value : op->getOperands())
          if (isConstant(value) && !mapping.contains(value))
            builder.clone(*value.getDefiningOp(), mapping);
        builder.clone(*op, mapping);
      }
      builder.create<func::ReturnOp>(function.getLoc(),
                                    mapping.lookup(group.root->getResult(0)));
      helpers.push_back(helper);
    }

    // Record CPU sequencing as data, never as an executable func.call wrapper.
    // Each step consumes original arguments, constants, or earlier CI results.
    OpBuilder builder(&getContext());
    DenseMap<Value, Attribute> bindings;
    for (BlockArgument argument : body.getArguments())
      bindings[argument] = builder.getDictionaryAttr({builder.getNamedAttr(
          "argument", builder.getI32IntegerAttr(argument.getArgNumber()))});
    DenseMap<Operation *, unsigned> owner;
    for (auto [index, group] : llvm::enumerate(groups))
      for (Operation *op : group.ops)
        owner[op] = index;
    SmallVector<Attribute> steps;
    std::function<Attribute(Value)> materialize = [&](Value value) -> Attribute {
      auto existing = bindings.find(value);
      if (existing != bindings.end())
        return existing->second;
      if (auto constant = value.getDefiningOp<arith::ConstantOp>())
        return bindings[value] = builder.getDictionaryAttr({
            builder.getNamedAttr("constant", constant.getValue())});
      unsigned index = owner.lookup(value.getDefiningOp());
      SmallVector<Attribute> arguments;
      for (Value input : inputs(groups[index]))
        arguments.push_back(materialize(input));
      unsigned step = steps.size();
      steps.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("callee", FlatSymbolRefAttr::get(helpers[index])),
          builder.getNamedAttr("inputs", builder.getArrayAttr(arguments))}));
      Attribute result = builder.getDictionaryAttr({builder.getNamedAttr(
          "step", builder.getI32IntegerAttr(step))});
      bindings[groups[index].root->getResult(0)] = result;
      return result;
    };
    Attribute result = materialize(
        cast<func::ReturnOp>(body.getTerminator()).getOperand(0));
    SmallVector<Attribute> sequences;
    if (auto existing = getOperation()->getAttrOfType<ArrayAttr>("scaleisa.sequences"))
      llvm::append_range(sequences, existing);
    sequences.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("source", builder.getStringAttr(function.getName())),
        builder.getNamedAttr("source_type", TypeAttr::get(function.getFunctionType())),
        builder.getNamedAttr("steps", builder.getArrayAttr(steps)),
        builder.getNamedAttr("result", result)}));
    getOperation()->setAttr("scaleisa.sequences", builder.getArrayAttr(sequences));
    function.erase();
    return success();
  }
};
} // namespace
std::unique_ptr<Pass> createSplitPass() {
  return std::make_unique<SplitPass>();
}
std::unique_ptr<Pass> createSplitPass(unsigned maxExtraCSRs) {
  return std::make_unique<SplitPass>(maxExtraCSRs);
}
} // namespace mlir::scaleisa_adapter
