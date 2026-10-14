#include "APS/APSToCMT2.h"

namespace mlir {

Module *APSToCMT2Pass::createTokenFIFO(Circuit &circuit, unsigned depth) {
  assert(depth == 1 || depth == 2);
  if (!elideTokenPayloads)
    return depth == 1 ? STLLibrary::createFIFO1PushModule(1, circuit)
                      : STLLibrary::createFIFO2IModule(1, circuit);

  // Preserve the original ready/occupancy contracts: FIFO1_PUSH can accept a
  // simultaneous dequeue/enqueue when full; FIFO2_I cannot. Token data is never
  // observed by APS control-flow consumers, so no payload register is needed.
  auto *module = circuit.addModule(depth == 1 ? "TokenFIFO1_PUSH" : "TokenFIFO2_I");
  auto clk = module->addClockArgument("clk");
  auto rst = module->addResetArgument("rst");
  auto &builder = module->getBuilder();
  auto saved = builder.saveInsertionPoint();
  auto *regModule = STLLibrary::createRegModule(depth == 1 ? 1 : 2, 0, circuit);
  auto *wireModule = STLLibrary::createWireDefaultModule(1, 0, circuit);
  builder.restoreInsertionPoint(saved);
  auto *state = module->addInstance("state", regModule, {clk.getValue(), rst.getValue()});
  auto *enqed = module->addInstance("enqed", wireModule, {});
  auto *deqed = module->addInstance("deqed", wireModule, {});
  auto bit = UIntType::get(builder.getContext(), 1);
  auto loc = module->getLoc();
  auto trueGuard = [&](OpBuilder &b) {
    b.create<circt::cmt2::ReturnOp>(loc, UInt::constant(1, 1, b, loc).getValue());
  };

  auto *full = module->addValue("full", {bit});
  full->guard(trueGuard);
  full->body([&](OpBuilder &b) {
    Signal count(state->callValue("read", b)[0], &b, loc);
    auto limit = UInt::constant(depth, depth == 1 ? 1 : 2, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, (count == limit).getValue());
  });
  full->finalize();

  auto *deq = module->addMethod("deq", {}, {bit});
  deq->guard([&](OpBuilder &b, ArrayRef<BlockArgument>) {
    Signal count(state->callValue("read", b)[0], &b, loc);
    auto zero = UInt::constant(0, depth == 1 ? 1 : 2, b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, (count != zero).getValue());
  });
  deq->body([&](OpBuilder &b, ArrayRef<BlockArgument>) {
    auto one = UInt::constant(1, 1, b, loc);
    deqed->callMethod("write", {one.getValue()}, b);
    b.create<circt::cmt2::ReturnOp>(loc, one.getValue());
  });
  deq->finalize();

  auto *enq = module->addMethod("enq", {{"data", bit}}, {});
  enq->guard([&](OpBuilder &b, ArrayRef<BlockArgument>) {
    Signal count(state->callValue("read", b)[0], &b, loc);
    auto limit = UInt::constant(depth, depth == 1 ? 1 : 2, b, loc);
    Signal ready = count != limit;
    if (depth == 1)
      ready = ready | Signal(deqed->callValue("read", b)[0], &b, loc);
    b.create<circt::cmt2::ReturnOp>(loc, ready.getValue());
  });
  enq->body([&](OpBuilder &b, ArrayRef<BlockArgument>) {
    enqed->callMethod("write", {UInt::constant(1, 1, b, loc).getValue()}, b);
    b.create<circt::cmt2::ReturnOp>(loc);
  });
  enq->finalize();

  auto *next = module->addRule("next");
  next->guard(trueGuard);
  next->body([&](OpBuilder &b) {
    Signal count(state->callValue("read", b)[0], &b, loc);
    Signal push(enqed->callValue("read", b)[0], &b, loc);
    Signal pop(deqed->callValue("read", b)[0], &b, loc);
    Signal updated = count;
    if (depth == 1) {
      updated = push | (count & ~pop);
    } else {
      auto zero = UInt::constant(0, 2, b, loc);
      auto one = UInt::constant(1, 2, b, loc);
      auto two = UInt::constant(2, 2, b, loc);
      auto inc = (count == zero).mux(one, two);
      auto dec = (count == two).mux(one, zero);
      updated = (push & ~pop).mux(inc, (pop & ~push).mux(dec, count));
    }
    state->callMethod("write", {updated.getValue()}, b);
    b.create<circt::cmt2::ReturnOp>(loc);
  });
  next->finalize();
  module->setPrecedence({{"full", "deq"}, {"deq", "enq"}, {"enq", "next"}});
  return module;
}

} // namespace mlir
