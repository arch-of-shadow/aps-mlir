#include "spike_memory_bridge.h"

#include <riscv/mmu.h>
#include <riscv/processor.h>

SpikeMemoryBridge::SpikeMemoryBridge(processor_t &processor)
    : processor_(processor) {}

uint8_t SpikeMemoryBridge::load8(uint32_t addr) const {
  return processor_.get_mmu()->load<uint8_t>(addr);
}

uint32_t SpikeMemoryBridge::load32(uint32_t addr) const {
  return processor_.get_mmu()->load<uint32_t>(addr);
}

uint64_t SpikeMemoryBridge::load64(uint32_t addr) const {
  // A DMA beat is not a CPU load instruction: an f32 buffer only needs
  // four-byte alignment, even when the scratchpad bus transfers eight bytes.
  if ((addr & 7) == 0)
    return processor_.get_mmu()->load<uint64_t>(addr);
  uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= uint64_t(load8(addr + i)) << (8 * i);
  return value;
}

void SpikeMemoryBridge::store8(uint32_t addr, uint8_t value) {
  processor_.get_mmu()->store<uint8_t>(addr, value);
}

void SpikeMemoryBridge::store32(uint32_t addr, uint32_t value, uint8_t mask) {
  if (mask == 0x0f) {
    processor_.get_mmu()->store<uint32_t>(addr, value);
    return;
  }
  for (unsigned i = 0; i < 4; ++i)
    if (mask & (1u << i))
      store8(addr + i, static_cast<uint8_t>(value >> (8 * i)));
}

void SpikeMemoryBridge::store64(uint32_t addr, uint64_t value, uint8_t mask) {
  if (mask == 0xff && (addr & 7) == 0) {
    processor_.get_mmu()->store<uint64_t>(addr, value);
    return;
  }
  for (unsigned i = 0; i < 8; ++i)
    if (mask & (1u << i))
      store8(addr + i, static_cast<uint8_t>(value >> (8 * i)));
}
