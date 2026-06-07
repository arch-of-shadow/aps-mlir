#include "csr_args.h"

#include <stdexcept>

namespace {

uint32_t parseU32(const std::string &value) {
  return static_cast<uint32_t>(std::stoul(value, nullptr, 0));
}

} // namespace

ApsCsrSpec parseApsCsrSpec(const std::string &spec) {
  size_t eq = spec.find('=');
  if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size())
    throw std::runtime_error("expected --add-csr name=addr[,mask=...,init=...]");

  ApsCsrSpec csr;
  csr.name = spec.substr(0, eq);

  size_t pos = eq + 1;
  size_t comma = spec.find(',', pos);
  csr.address = parseU32(spec.substr(pos, comma - pos));
  pos = comma;

  while (pos != std::string::npos) {
    ++pos;
    size_t next = spec.find(',', pos);
    std::string item = spec.substr(pos, next - pos);
    size_t itemEq = item.find('=');
    if (itemEq == std::string::npos)
      throw std::runtime_error("expected key=value in --add-csr option");
    std::string key = item.substr(0, itemEq);
    uint32_t value = parseU32(item.substr(itemEq + 1));
    if (key == "mask")
      csr.mask = value;
    else if (key == "init")
      csr.init = value;
    else
      throw std::runtime_error("unknown --add-csr key: " + key);
    pos = next;
  }

  return csr;
}
