#ifndef APS_COSIM_APS_ROCC_EXTENSION_H
#define APS_COSIM_APS_ROCC_EXTENSION_H

#include "aps_rtl_model.h"
#include "csr_args.h"
#include "csr_file.h"
#include "spike_memory_bridge.h"

#include <memory>
#include <riscv/rocc.h>
#include <string>

class ApsRoccExtension : public rocc_t {
public:
  const char *name() override { return "aps"; }
  void reset() override;
  void addCsr(const ApsCsrSpec &csr);
  void setVcdPath(const std::string &path);
  void registerCsrs();
  std::vector<insn_desc_t> get_instructions() override;

  reg_t custom0(rocc_insn_t insn, reg_t xs1, reg_t xs2) override;
  reg_t custom1(rocc_insn_t insn, reg_t xs1, reg_t xs2) override;
  reg_t custom2(rocc_insn_t insn, reg_t xs1, reg_t xs2) override;
  reg_t custom3(rocc_insn_t insn, reg_t xs1, reg_t xs2) override;

private:
  reg_t execute(rocc_insn_t insn, reg_t xs1, reg_t xs2);
  void ensureModel();

  ApsCsrFile csrs_;
  std::string vcdPath_;
  bool csrsRegistered_ = false;
  std::unique_ptr<SpikeMemoryBridge> memory_;
  std::unique_ptr<ApsRtlModel> rtl_;
};

#endif
