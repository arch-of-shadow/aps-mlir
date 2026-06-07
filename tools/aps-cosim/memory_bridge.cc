#include "memory_bridge.h"

#include <fstream>
#include <stdexcept>

HostMemoryBridge::HostMemoryBridge(uint32_t base, size_t size)
    : base_(base), bytes_(size, 0) {}

size_t HostMemoryBridge::offset(uint32_t addr, size_t accessSize) const {
  if (addr < base_)
    throw std::out_of_range("memory access below base");
  uint64_t off = static_cast<uint64_t>(addr) - base_;
  if (off + accessSize > bytes_.size())
    throw std::out_of_range("memory access beyond end");
  return static_cast<size_t>(off);
}

uint8_t HostMemoryBridge::load8(uint32_t addr) const {
  return bytes_[offset(addr, 1)];
}

uint32_t HostMemoryBridge::load32(uint32_t addr) const {
  size_t off = offset(addr, 4);
  uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i)
    value |= static_cast<uint32_t>(bytes_[off + i]) << (8 * i);
  return value;
}

uint64_t HostMemoryBridge::load64(uint32_t addr) const {
  size_t off = offset(addr, 8);
  uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<uint64_t>(bytes_[off + i]) << (8 * i);
  return value;
}

void HostMemoryBridge::store8(uint32_t addr, uint8_t value) {
  bytes_[offset(addr, 1)] = value;
}

void HostMemoryBridge::store32(uint32_t addr, uint32_t value, uint8_t mask) {
  size_t off = offset(addr, 4);
  for (unsigned i = 0; i < 4; ++i)
    if (mask & (1u << i))
      bytes_[off + i] = static_cast<uint8_t>(value >> (8 * i));
}

void HostMemoryBridge::store64(uint32_t addr, uint64_t value, uint8_t mask) {
  size_t off = offset(addr, 8);
  for (unsigned i = 0; i < 8; ++i)
    if (mask & (1u << i))
      bytes_[off + i] = static_cast<uint8_t>(value >> (8 * i));
}

void HostMemoryBridge::loadBinary(uint32_t addr, const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("failed to open input binary: " + path);
  size_t off = offset(addr, 0);
  in.read(reinterpret_cast<char *>(bytes_.data() + off),
          static_cast<std::streamsize>(bytes_.size() - off));
}

void HostMemoryBridge::dumpBinary(uint32_t addr, size_t size,
                              const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("failed to open output binary: " + path);
  size_t off = offset(addr, size);
  out.write(reinterpret_cast<const char *>(bytes_.data() + off),
            static_cast<std::streamsize>(size));
}
