#include "VFIFO2_I_w32.h"
#include "verilated.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  VFIFO2_I_w32 fifo;
  auto tick = [&]() {
    fifo.clk = 0; fifo.eval();
    fifo.clk = 1; fifo.eval();
    fifo.clk = 0; fifo.eval();
  };
  auto require = [&](bool ok, const char *message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
  };
  fifo.enq_enable = fifo.deq_enable = 0;
  fifo.rst = 1; tick(); tick(); fifo.rst = 0; tick();
  auto push = [&](unsigned value) {
    fifo.eval(); require(fifo.enq_ready, "FIFO unexpectedly full");
    fifo.enq_data = value; fifo.enq_enable = 1; tick();
    fifo.enq_enable = 0; fifo.eval();
  };
  auto pop = [&](unsigned expected) {
    fifo.eval(); require(fifo.deq_ready, "FIFO unexpectedly empty");
    require(fifo.deq_ == expected, "FIFO reversed order under backpressure");
    fifo.deq_enable = 1; tick(); fifo.deq_enable = 0; fifo.eval();
  };
  push(11); push(22); tick(); tick(); pop(11); pop(22);
  push(33);
  fifo.enq_data = 44; fifo.enq_enable = fifo.deq_enable = 1; fifo.eval();
  require(fifo.deq_ == 33, "simultaneous dequeue returned wrong head");
  tick(); fifo.enq_enable = fifo.deq_enable = 0; fifo.eval(); pop(44);
  for (unsigned i = 0; i < 32; i += 2) {
    push(i); push(i + 1); tick(); pop(i); pop(i + 1);
  }
  require(!fifo.deq_ready, "FIFO did not drain");
  std::cout << "PASS: FIFO2_I ordered stalled and simultaneous traffic\n";
}
