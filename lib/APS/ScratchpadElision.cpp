#include "APS/ScratchpadElision.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <optional>

using namespace mlir;

namespace {
std::optional<int64_t> constant(Value value) {
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) return constant(cast.getIn());
  IntegerAttr attr;
  if (matchPattern(value, m_Constant(&attr))) return attr.getInt();
  return std::nullopt;
}

// A deliberately bounded affine proof: one induction variable, unit stride,
// constant offset. Casts are safe because the candidate's bounds fit signed i32.
std::optional<int64_t> offset(Value value, Value iv, int64_t lower, int64_t upper) {
  auto bounded = [&](std::optional<int64_t> delta) -> std::optional<int64_t> {
    if (!delta) return std::nullopt;
    if (auto integer = dyn_cast<IntegerType>(value.getType())) {
      if (integer.getWidth() < 32 || (integer.getWidth() == 32 &&
          (*delta < int64_t(INT32_MIN) - lower || *delta > int64_t(INT32_MAX) - (upper - 1))))
        return std::nullopt;
    }
    return delta;
  };
  if (value == iv) return 0;
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    return bounded(offset(cast.getIn(), iv, lower, upper));
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    if (auto rhs = constant(add.getRhs()))
      if (auto lhs = offset(add.getLhs(), iv, lower, upper))
        if (*rhs >= -INT32_MAX && *rhs <= INT32_MAX &&
            *lhs + *rhs >= -INT32_MAX && *lhs + *rhs <= INT32_MAX) return bounded(*lhs + *rhs);
    if (auto lhs = constant(add.getLhs()))
      if (auto rhs = offset(add.getRhs(), iv, lower, upper))
        if (*lhs >= -INT32_MAX && *lhs <= INT32_MAX &&
            *lhs + *rhs >= -INT32_MAX && *lhs + *rhs <= INT32_MAX) return bounded(*lhs + *rhs);
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>())
    if (auto rhs = constant(sub.getRhs()))
      if (auto lhs = offset(sub.getLhs(), iv, lower, upper))
        if (*rhs >= -INT32_MAX && *rhs <= INT32_MAX &&
            *lhs - *rhs >= -INT32_MAX && *lhs - *rhs <= INT32_MAX) return bounded(*lhs - *rhs);
  return std::nullopt;
}

struct AffineForm { int64_t coefficient, constant; };
std::optional<AffineForm> form(AffineExpr expr) {
  if (auto d = dyn_cast<AffineDimExpr>(expr)) {
    if (d.getPosition() == 0) return AffineForm{1, 0};
    return std::nullopt;
  }
  if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
    if (c.getValue() >= -INT32_MAX && c.getValue() <= INT32_MAX)
      return AffineForm{0, c.getValue()};
    return std::nullopt;
  }
  auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binary) return std::nullopt;
  auto a = form(binary.getLHS()), b = form(binary.getRHS());
  if (!a || !b) return std::nullopt;
  std::optional<AffineForm> result;
  if (expr.getKind() == AffineExprKind::Add)
    result = AffineForm{a->coefficient + b->coefficient, a->constant + b->constant};
  if (expr.getKind() == AffineExprKind::Mul && !b->coefficient)
    result = AffineForm{a->coefficient * b->constant, a->constant * b->constant};
  if (result && result->coefficient >= -INT32_MAX && result->coefficient <= INT32_MAX &&
      result->constant >= -INT32_MAX && result->constant <= INT32_MAX) return result;
  return std::nullopt;
}

struct Plan {
  ScratchpadElisionCandidate info;
  memref::GlobalOp global;
  func::FuncOp function;
  aps::Copy copy;
  Operation *read = nullptr;
  Value index;
  affine::AffineForOp loop;
  SmallVector<memref::GetGlobalOp> references;
};

Plan analyze(ModuleOp module, memref::GlobalOp global) {
  Plan plan;
  plan.global = global;
  plan.info.buffer = global.getSymName().str();
  auto reject = [&](StringRef reason) { plan.info.reason = reason.str(); return plan; };
  auto type = global.getType();
  if (global->hasAttr("csr_address") || type.getRank() != 1 || !type.hasStaticShape() ||
      !type.getElementType().isInteger(32) || type.getDimSize(0) > INT32_MAX / 4)
    return reject("requires a static rank-one i32 staging buffer");
  if (module.getOps<aps::MemoryMapOp>().begin() != module.getOps<aps::MemoryMapOp>().end())
    return reject("must analyze before scratchpad allocation");
  if (global->hasAttr("volatile") || global->hasAttr("no_elide"))
    return reject("volatile or explicitly retained buffer");
  auto hint = global->getAttrOfType<StringAttr>("cache_hint");
  if (!hint || hint.getValue() != "warm")
    return reject("cache residency is not explicitly warm");
  if (auto level = global->getAttrOfType<IntegerAttr>("cache_level"))
    if (level.getInt() != 0) return reject("buffer is not at the warm hierarchy level");
  bool declared = !module.getOps<aps::MemItfc>().empty();
  auto cpu = module.lookupSymbol<aps::MemItfc>("cpuitfc");
  if (declared && !cpu) return reject("no native scalar CPU interface is visible");
  if (cpu) {
    if (auto cpuHint = cpu->getAttrOfType<StringAttr>("cache_hint"))
      if (cpuHint.getValue() != "warm") return reject("CPU interface is not cache-warm");
    if (auto slots = cpu->getAttrOfType<IntegerAttr>("inflight"))
      if (slots.getInt() > 2) return reject("CPU inflight model exceeds the native two-slot adapter");
  }
  auto uses = SymbolTable::getSymbolUses(global, module);
  if (!uses) return reject("cannot resolve all symbolic buffer uses");
  for (auto use : *uses)
    if (!isa<memref::GetGlobalOp>(use.getUser()))
      return reject("buffer has non-memref symbolic users");
  bool badReference = false;
  module.walk([&](memref::GetGlobalOp reference) {
    if (reference.getName() != global.getSymName()) return;
    auto function = reference->getParentOfType<func::FuncOp>();
    if (!function || (plan.function && plan.function != function)) badReference = true;
    plan.function = function;
    plan.references.push_back(reference);
  });
  if (badReference || !plan.function || plan.function.getBody().getBlocks().size() != 1)
    return reject("buffer shared across functions or unsupported control flow");
  plan.info.function = plan.function.getSymName().str();
  if (llvm::any_of(plan.function.getArgumentTypes(), [](Type type) { return isa<MemRefType>(type); }))
    return reject("memref arguments introduce unresolved external aliases");
  for (auto reference : plan.references) {
    for (Operation *user : reference.getResult().getUsers()) {
      if (auto copy = dyn_cast<aps::Copy>(user)) {
        if (plan.copy || copy.getDirection() != aps::CopyDirection::In || copy.getMemrefs().size() != 1)
          return reject("requires exactly one copy-in and no writeback");
        plan.copy = copy;
      } else if (isa<affine::AffineLoadOp, memref::LoadOp, aps::ReadSmem>(user)) {
        if (plan.read) return reject("multiple read sites may reuse or thrash cache lines");
        plan.read = user;
      } else return reject("scratchpad write, alias, escape, or unsupported use");
    }
  }
  if (!plan.copy || !plan.read) return reject("not a main-memory staging buffer");
  for (Operation *op : {plan.copy.getOperation(), plan.read}) {
    if (op->hasAttr("volatile") || op->hasAttr("no_elide"))
      return reject("volatile or explicitly retained access");
    if (auto accessHint = op->getAttrOfType<StringAttr>("cache_hint"))
      if (accessHint.getValue() != "warm") return reject("access cache hint contradicts warm residency");
    if (auto level = op->getAttrOfType<IntegerAttr>("cache_level"))
      if (level.getInt() != 0) return reject("access is not at the warm hierarchy level");
  }
  if (plan.copy->getBlock() != &plan.function.getBody().front())
    return reject("copy-in must execute once in the function entry block");
  if (!plan.copy.getCpuAddr().getType().isInteger(32))
    return reject("native direct access requires a 32-bit CPU address");
  auto start = constant(plan.copy.getStart()), length = constant(plan.copy.getLength());
  if (!start || !length || *start < 0 || *length <= 0 ||
      *start > type.getDimSize(0) || *length > type.getDimSize(0) - *start)
    return reject("copy-in range is not statically valid");
  plan.loop = plan.read->getParentOfType<affine::AffineForOp>();
  if (!plan.loop || plan.loop->getParentOp() != plan.function ||
      plan.read->getBlock() != plan.loop.getBody())
    return reject("read must be unconditional inside one top-level affine loop");
  auto pipeline = plan.loop->getAttrOfType<IntegerAttr>("pipeline");
  if (!pipeline || pipeline.getInt() != 1)
    return reject("read is outside an explicitly pipelined loop");
  if (plan.loop.getNumIterOperands() || plan.loop.getNumResults())
    return reject("native pipelined loop lowering does not support iter_args/results");
  for (Operation *op = plan.read; op != plan.function; op = op->getParentOp())
    if (op->hasAttr("unroll") || op->hasAttr("unroll_factor") || op->hasAttr("unroll-jam"))
      return reject("buffer accessed in an unrolled region");
  if (!plan.loop.hasConstantLowerBound() || !plan.loop.hasConstantUpperBound() ||
      plan.loop.getStepAsInt() != 1)
    return reject("requires a constant-bounded unit-stride loop");
  auto lower = plan.loop.getConstantLowerBound(), upper = plan.loop.getConstantUpperBound();
  if (lower < 0 || upper <= lower || upper > INT32_MAX / 4)
    return reject("loop bounds are empty or exceed the native address model");
  std::optional<int64_t> displacement;
  if (auto read = dyn_cast<affine::AffineLoadOp>(plan.read)) {
    auto map = read.getAffineMap();
    if (map.getNumDims() == 1 && !map.getNumSymbols() && map.getNumResults() == 1 &&
        read.getMapOperands().size() == 1 && read.getMapOperands()[0] == plan.loop.getInductionVar()) {
      auto linear = form(map.getResult(0));
      if (linear && linear->coefficient == 1) displacement = linear->constant;
    }
  } else {
    ValueRange indices = isa<memref::LoadOp>(plan.read)
        ? cast<memref::LoadOp>(plan.read).getIndices()
        : cast<aps::ReadSmem>(plan.read).getIndices();
    if (indices.size() == 1) {
      plan.index = indices[0];
      displacement = offset(plan.index, plan.loop.getInductionVar(), lower, upper);
    }
  }
  if (!displacement || *displacement < *start - lower ||
      *displacement > (*start + *length) - upper)
    return reject("affine read stream is not contiguous or exceeds staged range");
  DominanceInfo dominance(plan.function);
  if (!dominance.dominates(plan.copy, plan.loop))
    return reject("copy-in does not dominate every use");
  bool unsafe = false;
  auto outsideSnapshot = [&](Operation *op) {
    return op->getBlock() == plan.copy->getBlock() &&
           (op->isBeforeInBlock(plan.copy) || plan.loop->isBeforeInBlock(op));
  };
  plan.function.walk([&](Operation *op) {
    if (isa<func::CallOp, func::CallIndirectOp>(op)) unsafe = true;
    if (isa<aps::Store, aps::StoreBy, aps::StoreIssue>(op) && !outsideSnapshot(op)) unsafe = true;
    if (auto copy = dyn_cast<aps::Copy>(op))
      if (copy.getDirection() == aps::CopyDirection::Out && !outsideSnapshot(op)) unsafe = true;
    if (isa<aps::Load, aps::LoadBy, aps::LoadIssue>(op) && plan.loop->isAncestor(op)) unsafe = true;
    if (isa<scf::WhileOp, scf::IfOp, affine::AffineIfOp>(op)) unsafe = true;
    if (auto loop = dyn_cast<affine::AffineForOp>(op)) {
      if (loop->getParentOp() != plan.function || !loop.hasConstantLowerBound() ||
          !loop.hasConstantUpperBound() || loop.getStepAsInt() != 1) unsafe = true;
    }
    // Unknown operations with side effects can mutate main memory indirectly.
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "aps" && !isa<aps::ReadIRF, aps::WriteIRF, aps::ReadCSR, aps::WriteCSR,
                                  aps::ReadSmem, aps::WriteSmem, aps::Copy, aps::Load, aps::Store>(op) &&
        !isMemoryEffectFree(op)) unsafe = true;
    if (dialect != "aps" && dialect != "arith" && dialect != "affine" &&
        dialect != "memref" && dialect != "func" && !isMemoryEffectFree(op)) unsafe = true;
  });
  if (unsafe) return reject("possible main-memory mutation, competing stream, or unsupported control flow");
  plan.info.eligible = true;
  plan.info.reason = "warm contiguous read-only stream; requires trial rescheduling";
  return plan;
}
} // namespace

SmallVector<ScratchpadElisionCandidate> mlir::analyzeScratchpadElision(ModuleOp module) {
  SmallVector<ScratchpadElisionCandidate> candidates;
  for (auto global : module.getOps<memref::GlobalOp>())
    if (!global->hasAttr("csr_address")) candidates.push_back(analyze(module, global).info);
  llvm::sort(candidates, [](const auto &a, const auto &b) { return a.buffer < b.buffer; });
  return candidates;
}

LogicalResult mlir::rewriteScratchpadElision(ModuleOp module, StringRef buffer) {
  auto global = module.lookupSymbol<memref::GlobalOp>(buffer);
  if (!global) return failure();
  auto plan = analyze(module, global);
  if (!plan.info.eligible) return failure();
  OpBuilder builder(plan.read);
  auto loc = plan.read->getLoc();
  Value index = plan.index;
  if (auto read = dyn_cast<affine::AffineLoadOp>(plan.read))
    index = builder.create<affine::AffineApplyOp>(loc, read.getAffineMap(), read.getMapOperands());
  if (index.getType().isIndex()) index = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), index);
  if (!index.getType().isInteger(32)) return failure();
  auto start = builder.create<arith::ConstantIntOp>(loc, *constant(plan.copy.getStart()), 32);
  auto delta = builder.create<arith::SubIOp>(loc, index, start);
  auto four = builder.create<arith::ConstantIntOp>(loc, 4, 32);
  auto bytes = builder.create<arith::MulIOp>(loc, delta, four);
  auto address = builder.create<arith::AddIOp>(loc, plan.copy.getCpuAddr(), bytes);
  auto load = builder.create<aps::Load>(loc, plan.read->getResult(0).getType(), address);
  load->setAttr("cache_hint", builder.getStringAttr("warm"));
  load->setAttr("elided_scratchpad", builder.getStringAttr(buffer));
  plan.read->getResult(0).replaceAllUsesWith(load.getResult());
  plan.read->erase();
  plan.copy.erase();
  for (auto reference : plan.references) {
    if (!reference.getResult().use_empty()) return failure();
    reference.erase();
  }
  global.erase();
  return success();
}
