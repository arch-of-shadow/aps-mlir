// Exercise the entire synthesized accelerator through its integer RoCC ports.
// The golden vectors come from SoftFloat, not CPU FP execution.
#include "Vmain.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>

struct Vector {
  uint32_t a, b, data, flags, rd;
};
static void require(bool condition, const char *message, uint64_t cycle) {
  if (!condition) {
    std::fprintf(stderr, "FAIL cycle=%llu: %s\n", (unsigned long long)cycle,
                 message);
    std::exit(1);
  }
}
int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  require(argc == 2, "expected vector file", 0);
  std::ifstream file(argv[1]);
  require(file.good(), "cannot open vectors", 0);
  std::vector<Vector> vectors;
  std::string line;
  while (std::getline(file, line)) {
    Vector v;
    std::istringstream fields(line);
    fields >> std::hex >> v.a >> v.b >> v.data >> v.flags;
    require(!fields.fail(), "invalid vector", 0);
    v.rd = vectors.size() % 31 + 1;
    vectors.push_back(v);
  }
  Vmain dut;
  dut.rocc_cmd_rocc_cmd_opcode = 11;
  dut.rocc_cmd_rocc_cmd_funct = 0;
  dut.rocc_cmd_rocc_cmd_rs1 = 1;
  dut.rocc_cmd_rocc_cmd_rs2 = 2;
  dut.rocc_cmd_rocc_cmd_xs1 = 1;
  dut.rocc_cmd_rocc_cmd_xs2 = 1;
  dut.rocc_cmd_rocc_cmd_xd = 1;
  dut.burst_read_0_enable = dut.burst_read_1_enable = dut.burst_write_enable =
      0;
  dut.hella_resp_enable = 0;
  dut.hella_cmd_hella_cmd_to_bus_ready = 1;
  auto edge = [&]() {
    dut.clk = 0;
    dut.eval();
    dut.clk = 1;
    dut.eval();
    dut.clk = 0;
    dut.eval();
  };
  dut.rst = 1;
  dut.rocc_cmd_enable = 0;
  dut.rocc_resp_rocc_resp_to_bus_ready = 0;
  edge();
  std::deque<Vector> pending;
  std::mt19937 rng(0x49534158);
  size_t next = 0, consumed = 0, discarded = 0;
  uint64_t cycle = 0;
  while (next < vectors.size() || !pending.empty()) {
    require(cycle < vectors.size() * 500 + 1000, "simulation timeout", cycle);
    bool reset = cycle > 1000 && cycle < 5000 && cycle % 1201 == 0;
    dut.rst = reset;
    dut.rocc_cmd_enable = !reset && next < vectors.size() && rng() % 5 != 0;
    dut.rocc_resp_rocc_resp_to_bus_ready =
        !reset && cycle % 1200 < 900 && rng() % 3 != 0;
    if (next < vectors.size()) {
      dut.rocc_cmd_rocc_cmd_rs1data = vectors[next].a;
      dut.rocc_cmd_rocc_cmd_rs2data = vectors[next].b;
      dut.rocc_cmd_rocc_cmd_rd = vectors[next].rd;
    }
    dut.eval();
    if (reset) {
      discarded += pending.size();
      pending.clear();
    } else {
      if (dut.rocc_resp_rocc_resp_to_bus_enable) {
        require(dut.rocc_resp_rocc_resp_to_bus_ready,
                "response fired while stalled", cycle);
        require(!pending.empty(), "spurious response", cycle);
        Vector v = pending.front();
        pending.pop_front();
        ++consumed;
        if (dut.rocc_resp_rocc_resp_to_bus_result_rddata != v.data ||
            dut.rocc_resp_rocc_resp_to_bus_result_rd != v.rd) {
          std::fprintf(
              stderr, "a=%08x b=%08x expected=%08x rd=%u got=%08x rd=%u\n", v.a,
              v.b, v.data, v.rd, dut.rocc_resp_rocc_resp_to_bus_result_rddata,
              dut.rocc_resp_rocc_resp_to_bus_result_rd);
          require(false, "ISAX arithmetic/context/order mismatch", cycle);
        }
      }
      if (dut.rocc_cmd_enable && dut.rocc_cmd_ready)
        pending.push_back(vectors[next++]);
    }
    edge();
    ++cycle;
  }
  dut.rocc_cmd_enable = 0;
  dut.rocc_resp_rocc_resp_to_bus_ready = 1;
  for (int i = 0; i < 20; ++i) {
    dut.eval();
    require(!dut.rocc_resp_rocc_resp_to_bus_enable, "duplicate response",
            cycle);
    edge();
  }
  require(consumed + discarded == next, "lost request", cycle);
  dut.final();
  std::printf(
      "PASS ISAX accepted=%zu consumed=%zu reset-discarded=%zu cycles=%llu\n",
      next, consumed, discarded, (unsigned long long)cycle);
}
