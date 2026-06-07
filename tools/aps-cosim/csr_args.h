#ifndef APS_COSIM_CSR_ARGS_H
#define APS_COSIM_CSR_ARGS_H

#include <cstdint>
#include <string>

struct ApsCsrSpec {
  std::string name;
  uint32_t address = 0;
  uint32_t mask = 0xffffffffu;
  uint32_t init = 0;
};

ApsCsrSpec parseApsCsrSpec(const std::string &spec);

#endif
