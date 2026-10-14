// Storage planning on scheduled APS/TOR, before any CMT2 hardware exists.
#include "APS/APSOps.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "aps-optimize-storage"

namespace mlir {
#define GEN_PASS_DEF_APSOPTIMIZESTORAGE
#include "APS/Passes.h.inc"

namespace {
// Packing/unpacking is combinational in an existing schedule slot. Never add
// or delete a slot, or move an expression across an execution scope.
static void stampStorageOp(Operation *op, Operation *slot) {
  // Constants are virtual/rematerialized by TOR lowering, not scheduled ops.
  if (isa<arith::ConstantOp>(op))
    return;
  auto time = slot->getAttr("starttime");
  op->setAttr("starttime", time);
  op->setAttr("endtime", time);
  if (auto ref = slot->getAttr("ref_starttime")) {
    op->setAttr("ref_starttime", ref);
    op->setAttr("ref_endtime", ref);
  }
}

static bool laterInSameBlock(Operation *producer, Operation *consumer) {
  auto from = producer->getAttrOfType<IntegerAttr>("starttime");
  auto to = consumer->getAttrOfType<IntegerAttr>("starttime");
  return producer->getBlock() == consumer->getBlock() && from && to &&
         to.getInt() > from.getInt();
}

struct APSOptimizeStoragePass
    : impl::APSOptimizeStorageBase<APSOptimizeStoragePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool usesScalarMemory = false;
    bool sawResponse = false, zeroResponses = true, compactCommands = true;
    module.walk([&](Operation *op) {
      usesScalarMemory |= isa<aps::Load, aps::Store, aps::LoadBy,
                              aps::StoreBy, aps::LoadIssue, aps::StoreIssue>(op);
    });
    module.walk([&](aps::WriteIRF write) {
      sawResponse = true;
      APInt value;
      zeroResponses &= matchPattern(write.getValue(), m_ConstantInt(&value)) &&
                       value.isZero() && value.getBitWidth() == 32;
    });
    module.walk([&](tor::FuncOp function) {
      // Native lowering reads command data only through read_irf, and rd through
      // write_irf. Do not compact if a function exposes another argument use.
      for (BlockArgument argument : function.getArguments()) {
        if (!argument.getType().isInteger(5))
          compactCommands = false;
        for (OpOperand &use : argument.getUses()) {
          if (auto read = dyn_cast<aps::ReadIRF>(use.getOwner())) {
            if (use.getOperandNumber() != 0 || argument.getArgNumber() > 1)
              compactCommands = false;
          } else if (isa<aps::WriteIRF>(use.getOwner())) {
            if (use.getOperandNumber() != 0 || argument.getArgNumber() != 2)
              compactCommands = false;
          } else {
            compactCommands = false;
          }
        }
      }
    });
    module->setAttr("aps.scalar_memory_used",
                    BoolAttr::get(&getContext(), usesScalarMemory));
    module->setAttr("aps.zero_rocc_response",
                    BoolAttr::get(&getContext(), sawResponse && zeroResponses));
    module->setAttr("aps.compact_rocc_command",
                    BoolAttr::get(&getContext(), compactCommands));
    // These are compiler-generated control channels, not i1 SSA data FIFOs.
    module->setAttr("aps.elide_token_payloads", BoolAttr::get(&getContext(), true));

    SmallVector<arith::ExtUIOp> casts;
    module.walk([&](arith::ExtUIOp cast) { casts.push_back(cast); });
    unsigned rematerialized = 0;
    for (auto cast : casts) {
      auto producerTime = cast->getAttrOfType<IntegerAttr>("starttime");
      if (!producerTime)
        continue;
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : cast.getResult().getUses())
        uses.push_back(&use);
      DenseMap<Operation *, Value> materializations;
      for (OpOperand *use : uses) {
        Operation *consumer = use->getOwner();
        auto consumerTime = consumer->getAttrOfType<IntegerAttr>("starttime");
        // Only move pure widening across cycles of the same execution scope.
        // Moving through loops/branches would require a separate lifetime proof.
        if (consumer->getBlock() != cast->getBlock() || !consumerTime ||
            consumerTime.getInt() <= producerTime.getInt())
          continue;
        Value &value = materializations[consumer];
        if (!value) {
          OpBuilder builder(consumer);
          auto *copy = builder.clone(*cast.getOperation());
          copy->removeAttr("aps.storage_anchor");
          copy->setAttr("starttime", consumerTime);
          copy->setAttr("endtime", consumerTime);
          if (auto ref = consumer->getAttr("ref_starttime")) {
            copy->setAttr("ref_starttime", ref);
            copy->setAttr("ref_endtime", ref);
          }
          value = copy->getResult(0);
          ++rematerialized;
        }
        use->set(value);
      }
      // Keep an empty widening as a schedule anchor. If it was the last
      // operation in its slot, deleting it would also delete a token stage and
      // advance later side effects by a cycle. Its unused result gets no storage
      // during CMT2 generation, but the original stage remains present.
      if (cast.getResult().use_empty())
        cast->setAttr("aps.storage_anchor", UnitAttr::get(&getContext()));
    }
    SmallVector<tor::MulIOp> squares;
    module.walk([&](tor::MulIOp mul) {
      if (mul->getOperand(0) == mul->getOperand(1))
        squares.push_back(mul);
    });
    unsigned squareInputs = 0, squareOutputs = 0;
    DenseSet<Value> visitedInputs;
    for (auto square : squares) {
      Value input = square->getOperand(0);
      auto type = dyn_cast<IntegerType>(input.getType());
      Operation *producer = input.getDefiningOp();
      if (!type || type.getWidth() < 3 || !producer ||
          !visitedInputs.insert(input).second)
        continue;
      SmallVector<Operation *> consumers;
      bool legal = true;
      for (Operation *consumer : input.getUsers()) {
        legal &= isa<tor::MulIOp>(consumer) &&
                 consumer->getOperand(0) == input &&
                 consumer->getOperand(1) == input &&
                 consumer->getResult(0).getType() == type &&
                 laterInSameBlock(producer, consumer);
        if (!llvm::is_contained(consumers, consumer))
          consumers.push_back(consumer);
      }
      if (!legal)
        continue;
      // In N-bit modular arithmetic, (x + 2^(N-1))^2 == x^2.
      // Only the low N-1 bits need to survive until an exclusive square use.
      OpBuilder b(producer);
      b.setInsertionPointAfter(producer);
      auto narrow = b.create<arith::TruncIOp>(producer->getLoc(),
          IntegerType::get(&getContext(), type.getWidth() - 1), input);
      stampStorageOp(narrow, producer);
      for (Operation *consumer : consumers) {
        b.setInsertionPoint(consumer);
        auto wide = b.create<arith::ExtUIOp>(consumer->getLoc(), type, narrow);
        stampStorageOp(wide, consumer);
        consumer->setOperand(0, wide);
        consumer->setOperand(1, wide);
      }
      ++squareInputs;
    }
    for (auto square : squares) {
      Value result = square->getResult(0);
      auto type = dyn_cast<IntegerType>(result.getType());
      if (!type || type.getWidth() < 3 || result.use_empty())
        continue;
      SmallVector<OpOperand *> uses;
      bool legal = true;
      for (OpOperand &use : result.getUses()) {
        legal &= laterInSameBlock(square, use.getOwner());
        uses.push_back(&use);
      }
      if (!legal)
        continue;
      // Every integer square is 0 or 1 modulo 4: bit 1 is always zero.
      // Pack that hole out of the stored value, and restore it at each use.
      OpBuilder b(square);
      b.setInsertionPointAfter(square);
      auto make = [&](auto op, Operation *slot) {
        stampStorageOp(op, slot);
        return op.getResult();
      };
      // Scheduled TOR keeps virtual constants outside its function bodies.
      // Hoist only these operand-free constants, not the scheduled codec.
      OpBuilder constants(&getContext());
      if (auto design = square->getParentOfType<tor::DesignOp>())
        constants.setInsertionPointToStart(&design->getRegion(0).front());
      else
        constants.setInsertionPoint(square);
      Value one = constants.create<arith::ConstantIntOp>(square.getLoc(), 1, type.getWidth());
      Value two = constants.create<arith::ConstantIntOp>(square.getLoc(), 2, type.getWidth());
      Value hi = make(b.create<arith::ShRUIOp>(square.getLoc(), result, one), square);
      Value lo = make(b.create<arith::AndIOp>(square.getLoc(), result, one), square);
      Value packed = make(b.create<arith::OrIOp>(square.getLoc(), hi, lo), square);
      Value narrow = make(b.create<arith::TruncIOp>(square.getLoc(),
          IntegerType::get(&getContext(), type.getWidth() - 1), packed), square);
      DenseMap<Operation *, Value> restored;
      for (OpOperand *use : uses) {
        Operation *consumer = use->getOwner();
        Value &value = restored[consumer];
        if (!value) {
          b.setInsertionPoint(consumer);
          auto loc = consumer->getLoc();
          Value wide = make(b.create<arith::ExtUIOp>(loc, type, narrow), consumer);
          Value high = make(b.create<arith::ShRUIOp>(loc, wide, one), consumer);
          high = make(b.create<arith::ShLIOp>(loc, high, two), consumer);
          Value low = make(b.create<arith::AndIOp>(loc, wide, one), consumer);
          value = make(b.create<arith::OrIOp>(loc, high, low), consumer);
        }
        use->set(value);
      }
      ++squareOutputs;
    }
    LLVM_DEBUG(llvm::dbgs() << "[aps-optimize-storage] rematerialized "
                           << rematerialized << " widening casts; packed square inputs="
                           << squareInputs << "; outputs=" << squareOutputs
                           << "; scalar memory="
                           << usesScalarMemory << "; RoCC command payload="
                           << (compactCommands ? 69 : 96) << "; response payload="
                           << (sawResponse && zeroResponses ? 5 : 37) << '\n');
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAPSOptimizeStoragePass() {
  return std::make_unique<APSOptimizeStoragePass>();
}
} // namespace mlir
