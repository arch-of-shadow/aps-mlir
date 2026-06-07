#ifndef APS_COSIM_CSR_FILE_H
#define APS_COSIM_CSR_FILE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class ApsCsrFile {
public:
  struct Entry {
    std::string name;
    uint32_t address;
    uint32_t mask;
    uint32_t value;
    bool hasAddress;
  };

  void define(const std::string &name, uint32_t address = 0,
              bool hasAddress = false, uint32_t mask = 0xffffffffu,
              uint32_t init = 0);

  uint32_t readByName(const std::string &name) const;
  void writeByName(const std::string &name, uint32_t value);
  uint32_t readByAddress(uint32_t address) const;
  void writeByAddress(uint32_t address, uint32_t value);

  const std::vector<Entry> &entries() const { return entries_; }

private:
  Entry &entryForName(const std::string &name);
  const Entry &entryForName(const std::string &name) const;

  std::vector<Entry> entries_;
  std::unordered_map<std::string, size_t> byName_;
  std::unordered_map<uint32_t, size_t> byAddress_;
};

#endif
