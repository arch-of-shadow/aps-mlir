#include "spike_memory_bridge.h"
#include <riscv/cfg.h>
#include <riscv/devices.h>
#include <riscv/sim.h>
#include <cassert>
#include <cstdio>
#include <memory>

int main() {
  cfg_t cfg;
  cfg.isa = "RV32IMAFDC_zicsr_zifencei";
  cfg.mem_layout = {mem_cfg_t(0x80000000, 65536)};
  auto memory = std::make_unique<mem_t>(65536);
  std::vector<std::pair<reg_t, abstract_mem_t *>> memories = {
      {0x80000000, memory.get()}};
  std::vector<device_factory_sargs_t> devices;
  debug_module_config_t debug = {};
  sim_t sim(&cfg, false, memories, devices, {"none"}, debug,
            nullptr, false, nullptr, false, nullptr);
  SpikeMemoryBridge bridge(*sim.get_core(0));
  constexpr uint32_t base = 0x80000100;
  constexpr uint64_t value = 0x0123456789abcdefULL;
  for (unsigned offset = 0; offset < 8; ++offset) {
    for (uint8_t mask : {uint8_t(0xff), uint8_t(0x55), uint8_t(0x80)}) {
      for (unsigned i = 0; i < 32; ++i)
        bridge.store8(base + i, 0xa5);
      bridge.store64(base + 8 + offset, value, mask);
      uint64_t expected = 0;
      for (unsigned i = 0; i < 8; ++i) {
        uint8_t byte = mask & (1u << i) ? uint8_t(value >> (8 * i)) : 0xa5;
        expected |= uint64_t(byte) << (8 * i);
      }
      assert(bridge.load64(base + 8 + offset) == expected);
      for (unsigned i = 0; i < 32; ++i) {
        unsigned position = i - 8 - offset;
        uint8_t byte = position < 8 && (mask & (1u << position))
                           ? uint8_t(value >> (8 * position)) : 0xa5;
        assert(bridge.load8(base + i) == byte);
      }
    }
  }
  std::puts("PASS DMA alignment and masked stores");
}
