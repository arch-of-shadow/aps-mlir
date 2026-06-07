#ifndef APS_COSIM_SPIKE_MEMORY_BRIDGE_H
#define APS_COSIM_SPIKE_MEMORY_BRIDGE_H

#include "memory_bridge.h"

class processor_t;

class SpikeMemoryBridge : public MemoryBridge {
public:
  explicit SpikeMemoryBridge(processor_t &processor);

  uint8_t load8(uint32_t addr) const override;
  uint32_t load32(uint32_t addr) const override;
  uint64_t load64(uint32_t addr) const override;

  void store8(uint32_t addr, uint8_t value) override;
  void store32(uint32_t addr, uint32_t value, uint8_t mask = 0x0f) override;
  void store64(uint32_t addr, uint64_t value, uint8_t mask = 0xff) override;

private:
  processor_t &processor_;
};

#endif
