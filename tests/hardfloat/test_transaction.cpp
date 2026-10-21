#include "VHF32Binary.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <random>
#include <sstream>
#include <vector>
#ifndef TEST_LATENCY
#define TEST_LATENCY 3
#endif
#ifndef TEST_CAPACITY
#define TEST_CAPACITY 5
#endif

struct Vector {
  uint32_t a, b, data, flags;
  uint64_t due = 0;
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
    vectors.push_back(v);
  }
  VHF32Binary dut;
  auto edge = [&]() {
    dut.clock = 0;
    dut.eval();
    dut.clock = 1;
    dut.eval();
    dut.clock = 0;
    dut.eval();
  };
  dut.reset = 1;
  dut.issue_enable = 0;
  dut.collect_enable = 0;
  edge();
  dut.reset = 0;
  std::deque<Vector> pending;
  std::mt19937 rng(0x48544650);
  size_t next = 0, consumed = 0, discarded = 0;
  bool held = false;
  uint32_t heldData = 0, heldFlags = 0;
  uint64_t cycle = 0;
  size_t issuedAtStallStart = 0;
  while (next < vectors.size() || !pending.empty()) {
    require(cycle < vectors.size() * 40 + 1000, "simulation timeout", cycle);
    bool reset = cycle > 1000 && cycle < 5000 && cycle % 1201 == 0;
    dut.reset = reset;
    // First test II=1 (where capacity permits), then fill under a long stall,
    // then random request bubbles and long/random response stalls.
    bool longStall = (cycle >= 100 && cycle < 400) || cycle % 1500 >= 1200;
    dut.issue_enable =
        next < vectors.size() && (cycle < 400 || (rng() % 5 != 0));
    dut.collect_enable = !longStall && (cycle < 100 || (rng() % 3 != 0));
    if (next < vectors.size()) {
      dut.operand0 = vectors[next].a;
      dut.operand1 = vectors[next].b;
    }
    dut.eval();
    if (reset) {
      require(!dut.issue_ready && !dut.collect_ready, "ready during reset",
              cycle);
      discarded += pending.size();
      pending.clear();
      held = false;
    } else {
      if (held) {
        require(dut.collect_ready, "stalled response disappeared", cycle);
        require(dut.collect_data == heldData && dut.collect_flags == heldFlags,
                "stalled response changed", cycle);
      }
      if (cycle < 100 && !pending.empty() && cycle >= pending.front().due)
        require(dut.collect_ready, "response later than declared latency",
                cycle);
      if (dut.collect_ready) {
        require(!pending.empty(), "spurious/duplicated response", cycle);
        Vector v = pending.front();
        require(cycle >= v.due, "response before declared latency", cycle);
        if (dut.collect_data != v.data || dut.collect_flags != v.flags) {
          std::fprintf(
              stderr,
              "a=%08x b=%08x expected=%08x flags=%02x got=%08x flags=%02x\n",
              v.a, v.b, v.data, v.flags, dut.collect_data, dut.collect_flags);
          require(false, "arithmetic/flags/order mismatch", cycle);
        }
        if (dut.collect_enable) {
          pending.pop_front();
          ++consumed;
        }
      }
      held = dut.collect_ready && !dut.collect_enable;
      heldData = dut.collect_data;
      heldFlags = dut.collect_flags;
      if (dut.issue_enable && dut.issue_ready) {
        Vector v = vectors[next++];
        v.due = cycle + TEST_LATENCY;
        pending.push_back(v);
      }
      require(pending.size() <= TEST_CAPACITY, "outstanding capacity exceeded",
              cycle);
      if (cycle < 100 && TEST_CAPACITY > TEST_LATENCY)
        require(dut.issue_ready, "lost no-stall II=1", cycle);
      if (cycle == 100)
        issuedAtStallStart = next;
      if (cycle == 399) {
        require(next - issuedAtStallStart <= TEST_CAPACITY,
                "accepted unreserved work during stall", cycle);
        require(!dut.issue_ready && pending.size() == TEST_CAPACITY,
                "did not stop at credit limit", cycle);
      }
    }
    edge();
    ++cycle;
  }
  dut.issue_enable = 0;
  dut.collect_enable = 1;
  for (int i = 0; i < TEST_LATENCY + 4; ++i) {
    dut.eval();
    require(!dut.collect_ready, "late duplicate response", cycle);
    edge();
  }
  require(next == consumed + discarded, "lost transaction", cycle);
  dut.final();
  std::printf("PASS latency=%d capacity=%d accepted=%zu consumed=%zu "
              "reset-discarded=%zu cycles=%llu\n",
              TEST_LATENCY, TEST_CAPACITY, next, consumed, discarded,
              (unsigned long long)cycle);
}
