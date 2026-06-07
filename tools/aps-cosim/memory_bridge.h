#ifndef APS_COSIM_MEMORY_BRIDGE_H
#define APS_COSIM_MEMORY_BRIDGE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class MemoryBridge {
public:
  virtual ~MemoryBridge() = default;

  virtual uint8_t load8(uint32_t addr) const = 0;
  virtual uint32_t load32(uint32_t addr) const = 0;
  virtual uint64_t load64(uint32_t addr) const = 0;

  virtual void store8(uint32_t addr, uint8_t value) = 0;
  virtual void store32(uint32_t addr, uint32_t value, uint8_t mask = 0x0f) = 0;
  virtual void store64(uint32_t addr, uint64_t value, uint8_t mask = 0xff) = 0;
};

class HostMemoryBridge : public MemoryBridge {
public:
  HostMemoryBridge(uint32_t base, size_t size);

  uint32_t base() const { return base_; }
  size_t size() const { return bytes_.size(); }

  uint8_t load8(uint32_t addr) const override;
  uint32_t load32(uint32_t addr) const override;
  uint64_t load64(uint32_t addr) const override;

  void store8(uint32_t addr, uint8_t value) override;
  void store32(uint32_t addr, uint32_t value, uint8_t mask = 0x0f) override;
  void store64(uint32_t addr, uint64_t value, uint8_t mask = 0xff) override;

  void loadBinary(uint32_t addr, const std::string &path);
  void dumpBinary(uint32_t addr, size_t size, const std::string &path) const;

private:
  size_t offset(uint32_t addr, size_t accessSize) const;

  uint32_t base_;
  std::vector<uint8_t> bytes_;
};

#endif
