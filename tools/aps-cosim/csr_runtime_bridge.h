#ifndef APS_COSIM_CSR_RUNTIME_BRIDGE_H
#define APS_COSIM_CSR_RUNTIME_BRIDGE_H

#include "csr_file.h"

#include <string>
#include <vector>

class Vmain;

class CsrRuntimeBridge {
public:
  void drive(Vmain &top, ApsCsrFile &csrs);
  void sample(Vmain &top, ApsCsrFile &csrs);

private:
  struct Handles {
    std::string name;
    void *value = nullptr;
    void *valueReady = nullptr;
    void *setData = nullptr;
    void *setReady = nullptr;
    void *setEnable = nullptr;
  };

  void bind(Vmain &top, ApsCsrFile &csrs);
  static void *requiredPort(Vmain &top, const std::string &name);
  static void putU32(void *port, uint32_t value);
  static void putU8(void *port, uint8_t value);
  static uint32_t getU32(void *port);
  static uint8_t getU8(void *port);

  bool bound_ = false;
  std::vector<Handles> handles_;
};

#endif
