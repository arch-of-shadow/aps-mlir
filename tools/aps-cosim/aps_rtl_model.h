#ifndef APS_COSIM_APS_RTL_MODEL_H
#define APS_COSIM_APS_RTL_MODEL_H

#include "fake_dma.h"
#include "csr_file.h"
#include "csr_runtime_bridge.h"
#include "memory_bridge.h"

#include <cstdint>
#include <memory>
#include <string>

class Vmain;
class VerilatedVcdC;

struct ApsCommand {
  uint8_t opcode = 0;
  uint8_t funct7 = 0;
  uint8_t rd = 0;
  uint32_t rs1 = 0;
  uint32_t rs2 = 0;
  bool xd = true;
  bool xs1 = true;
  bool xs2 = true;
};

struct ApsResult {
  uint8_t rd = 0;
  uint32_t data = 0;
  bool hasResponse = false;
  uint64_t cycles = 0;
};

class ApsRtlModel {
public:
  explicit ApsRtlModel(MemoryBridge &memory, ApsCsrFile *csrs = nullptr);
  ~ApsRtlModel();

  void reset(unsigned cycles = 8);
  void enableVcd(const std::string &path);
  ApsResult execute(const ApsCommand &cmd, uint64_t timeoutCycles = 100000);

private:
  void eval();
  void tick();
  bool tickCommandCycle();
  void driveDefaults();

  std::unique_ptr<Vmain> top_;
  ApsCsrFile localCsrs_;
  ApsCsrFile &csrs_;
  CsrRuntimeBridge csrRuntime_;
  FakeDma dma_;
  std::unique_ptr<VerilatedVcdC> trace_;
  uint64_t cycle_ = 0;
  uint64_t traceTime_ = 0;
};

#endif
