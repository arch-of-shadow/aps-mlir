#include "Vmemory.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <stdexcept>

struct Request { uint32_t data; unsigned cmd; unsigned tag; };
struct Slot { bool busy = false, acknowledged = false; Request request{}; };
static void check(bool ok, const char *what) {
  if (!ok) throw std::runtime_error(what);
}

int main() {
  Vmemory dut;
  std::deque<Request> awaitingBus;
  std::deque<unsigned> orderedSlots;
  std::array<Slot, 2> slots{};
  uint32_t rng = 0x13579bdu;
  auto random = [&]() {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
  };
  auto edge = [&]() { dut.clk = 1; dut.eval(); dut.clk = 0; dut.eval(); };
  dut.clk = 0; dut.rst = 1;
  dut.cmd_from_user_enable = dut.resp_from_bus_enable = dut.resp_to_user_enable = 0;
  dut.hella_cmd_hella_cmd_to_bus_ready = 0;
  dut.eval(); edge(); edge(); dut.rst = 0;
  unsigned issued = 0, retired = 0, stores = 0, reordered = 0;
  for (unsigned cycle = 0; cycle < 20200; ++cycle) {
    bool drain = cycle >= 20000;
    Request input{random(), random() & 1u, 0x3f};
    dut.cmd_from_user_enable = !drain && (random() % 3 != 0);
    // Deliberately alias all addresses AND all original user tags. Only the
    // adapter's bus tags may distinguish outstanding completions.
    dut.cmd_from_user_user_cmd_addr = 0x80000000;
    dut.cmd_from_user_user_cmd_cmd = input.cmd;
    dut.cmd_from_user_user_cmd_size = 2;
    dut.cmd_from_user_user_cmd_data = input.data;
    dut.cmd_from_user_user_cmd_mask = 0;
    dut.cmd_from_user_user_cmd_tag = input.tag;
    dut.resp_to_user_enable = drain || (random() % 4 != 0);
    dut.hella_cmd_hella_cmd_to_bus_ready = drain || (random() % 3 != 0);
    unsigned selected = random() & 1u;
    if (!slots[selected].busy || slots[selected].acknowledged) selected ^= 1;
    bool ack = slots[selected].busy && !slots[selected].acknowledged &&
               (drain || random() % 3 == 0);
    dut.resp_from_bus_enable = ack;
    dut.resp_from_bus_hella_resp_tag = selected;
    dut.resp_from_bus_hella_resp_cmd = slots[selected].request.cmd;
    dut.resp_from_bus_hella_resp_size = 2;
    dut.resp_from_bus_hella_resp_signed = 0;
    dut.resp_from_bus_hella_resp_data = slots[selected].request.data ^ 0xa5a5a5a5;
    dut.eval();
    bool push = dut.cmd_from_user_enable && dut.cmd_from_user_ready;
    bool bus = dut.hella_cmd_hella_cmd_to_bus_enable && dut.hella_cmd_hella_cmd_to_bus_ready;
    bool pop = dut.resp_to_user_enable && dut.resp_to_user_ready;
    bool receive = ack && dut.resp_from_bus_ready;
    unsigned busTag = dut.hella_cmd_hella_cmd_to_bus_cmd_tag;
    if (dut.resp_to_user_ready) {
      check(!orderedSlots.empty(), "unsolicited completion");
      auto &front = slots[orderedSlots.front()];
      check(front.acknowledged, "completion before ordered acknowledgement");
      check(dut.resp_to_user___05Ftag == front.request.tag, "user tag not preserved");
      check(dut.resp_to_user___05Fdata == (front.request.data ^ 0xa5a5a5a5), "wrong completion data");
    }
    if (bus) {
      check(!awaitingBus.empty(), "unsolicited bus request");
      check(busTag < 2 && !slots[busTag].busy, "reused occupied bus tag");
      check(dut.hella_cmd_hella_cmd_to_bus_cmd_data == awaitingBus.front().data,
            "bus request reordered or corrupted");
      check(dut.hella_cmd_hella_cmd_to_bus_cmd_cmd == awaitingBus.front().cmd,
            "wrong bus command");
    }
    if (receive && orderedSlots.size() == 2 && selected != orderedSlots.front()) ++reordered;
    edge();
    if (pop) { slots[orderedSlots.front()].busy = false; orderedSlots.pop_front(); ++retired; }
    if (receive) slots[selected].acknowledged = true;
    if (bus) {
      slots[busTag] = Slot{true, false, awaitingBus.front()};
      orderedSlots.push_back(busTag); awaitingBus.pop_front();
    }
    if (push) { awaitingBus.push_back(input); ++issued; stores += input.cmd; }
  }
  check(awaitingBus.empty() && orderedSlots.empty(), "failed to drain completions");
  check(issued == retired && stores > 100 && reordered > 100, "insufficient coverage");
  std::cout << "PASS adapter issued=" << issued << " retired=" << retired
            << " stores=" << stores << " reordered_acks=" << reordered << '\n';
}
