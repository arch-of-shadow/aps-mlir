#include "csr_file.h"

#include <stdexcept>

void ApsCsrFile::define(const std::string &name, uint32_t address,
                        bool hasAddress, uint32_t mask, uint32_t init) {
  auto it = byName_.find(name);
  if (it != byName_.end()) {
    Entry &entry = entries_[it->second];
    entry.mask = mask;
    entry.value &= mask;
    if (hasAddress && !entry.hasAddress) {
      entry.address = address;
      entry.hasAddress = true;
      byAddress_[address] = it->second;
    }
    if (hasAddress)
      entry.value = init & mask;
    return;
  }

  size_t index = entries_.size();
  entries_.push_back({name, address, mask, init & mask, hasAddress});
  byName_[name] = index;
  if (hasAddress)
    byAddress_[address] = index;
}

uint32_t ApsCsrFile::readByName(const std::string &name) const {
  return entryForName(name).value;
}

void ApsCsrFile::writeByName(const std::string &name, uint32_t value) {
  Entry &entry = entryForName(name);
  entry.value = value & entry.mask;
}

uint32_t ApsCsrFile::readByAddress(uint32_t address) const {
  auto it = byAddress_.find(address);
  if (it == byAddress_.end())
    throw std::runtime_error("unknown APS CSR address");
  return entries_[it->second].value;
}

void ApsCsrFile::writeByAddress(uint32_t address, uint32_t value) {
  auto it = byAddress_.find(address);
  if (it == byAddress_.end())
    throw std::runtime_error("unknown APS CSR address");
  Entry &entry = entries_[it->second];
  entry.value = value & entry.mask;
}

ApsCsrFile::Entry &ApsCsrFile::entryForName(const std::string &name) {
  auto it = byName_.find(name);
  if (it == byName_.end())
    throw std::runtime_error("unknown APS CSR name: " + name);
  return entries_[it->second];
}

const ApsCsrFile::Entry &
ApsCsrFile::entryForName(const std::string &name) const {
  auto it = byName_.find(name);
  if (it == byName_.end())
    throw std::runtime_error("unknown APS CSR name: " + name);
  return entries_[it->second];
}
