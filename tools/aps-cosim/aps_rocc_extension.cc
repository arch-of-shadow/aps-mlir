#include "aps_rocc_extension.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <riscv/processor.h>
#include <riscv/csrs.h>
#include <stdexcept>
#include <vector>

namespace {

reg_t apsCustomInsn(processor_t *processor, insn_t insn, reg_t pc) {
  auto *aps = static_cast<ApsRoccExtension *>(processor->get_extension("aps"));
  rocc_insn_union_t decoded;
  state_t *state = processor->get_state();
  decoded.i = insn;
  reg_t xs1 = decoded.r.xs1 ? state->XPR[insn.rs1()] : -1;
  reg_t xs2 = decoded.r.xs2 ? state->XPR[insn.rs2()] : -1;
  reg_t result = 0;

  switch (decoded.r.opcode) {
  case ROCC_OPCODE0:
    result = aps->custom0(decoded.r, xs1, xs2);
    break;
  case ROCC_OPCODE1:
    result = aps->custom1(decoded.r, xs1, xs2);
    break;
  case ROCC_OPCODE2:
    result = aps->custom2(decoded.r, xs1, xs2);
    break;
  case ROCC_OPCODE3:
    result = aps->custom3(decoded.r, xs1, xs2);
    break;
  default:
    std::abort();
  }

  if (decoded.r.xd) {
    state->log_reg_write[insn.rd() << 4] = {result, 0};
    state->XPR.write(insn.rd(), result);
  }
  return pc + 4;
}

insn_desc_t apsOpcode(uint32_t opcode) {
  return {opcode, ROCC_OPCODE_MASK,
          apsCustomInsn, apsCustomInsn, apsCustomInsn, apsCustomInsn,
          apsCustomInsn, apsCustomInsn, apsCustomInsn, apsCustomInsn};
}

class ApsSpikeCsr : public csr_t {
public:
  ApsSpikeCsr(processor_t *proc, ApsCsrFile &csrs, uint32_t address)
      : csr_t(proc, address), csrs_(csrs) {}

  reg_t read() const noexcept override {
    return static_cast<reg_t>(csrs_.readByAddress(address));
  }

protected:
  bool unlogged_write(const reg_t value) noexcept override {
    csrs_.writeByAddress(address, static_cast<uint32_t>(value));
    return true;
  }

private:
  ApsCsrFile &csrs_;
};

} // namespace

void ApsRoccExtension::ensureModel() {
  if (rtl_)
    return;
  if (!p)
    throw std::runtime_error("APS RoCC extension has no Spike processor");
  memory_ = std::make_unique<SpikeMemoryBridge>(*p);
  rtl_ = std::make_unique<ApsRtlModel>(*memory_, &csrs_);
  if (!vcdPath_.empty())
    rtl_->enableVcd(vcdPath_);
  rtl_->reset();
}

void ApsRoccExtension::addCsr(const ApsCsrSpec &csr) {
  if (csrsRegistered_)
    throw std::runtime_error("cannot add APS CSR after CSR registration");
  csrs_.define(csr.name, csr.address, true, csr.mask, csr.init);
}

void ApsRoccExtension::setVcdPath(const std::string &path) {
  if (rtl_)
    throw std::runtime_error("cannot set VCD path after RTL model creation");
  vcdPath_ = path;
}

void ApsRoccExtension::registerCsrs() {
  if (csrsRegistered_)
    return;
  if (!p)
    throw std::runtime_error("APS RoCC extension has no Spike processor");

  for (const auto &entry : csrs_.entries()) {
    if (!entry.hasAddress)
      continue;
    p->get_state()->add_csr(
        entry.address, std::make_shared<ApsSpikeCsr>(p, csrs_, entry.address));
    if (std::getenv("APS_COSIM_TRACE"))
      std::cerr << "aps-csr register " << entry.name << " @0x" << std::hex
                << entry.address << " init=0x" << entry.value << std::dec
                << "\n";
  }
  csrsRegistered_ = true;
}

void ApsRoccExtension::reset() {
  if (rtl_)
    rtl_->reset();
}

std::vector<insn_desc_t> ApsRoccExtension::get_instructions() {
  return {apsOpcode(ROCC_OPCODE0), apsOpcode(ROCC_OPCODE1),
          apsOpcode(ROCC_OPCODE2), apsOpcode(ROCC_OPCODE3)};
}

reg_t ApsRoccExtension::execute(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  ensureModel();

  ApsCommand cmd;
  cmd.opcode = static_cast<uint8_t>(insn.opcode);
  cmd.funct7 = static_cast<uint8_t>(insn.funct);
  cmd.rd = static_cast<uint8_t>(insn.rd);
  cmd.rs1 = static_cast<uint32_t>(xs1);
  cmd.rs2 = static_cast<uint32_t>(xs2);
  cmd.xd = insn.xd;
  cmd.xs1 = insn.xs1;
  cmd.xs2 = insn.xs2;

  if (std::getenv("APS_COSIM_TRACE"))
    std::cerr << "aps-rocc issue opcode=0x" << std::hex << unsigned(cmd.opcode)
              << " funct7=0x" << unsigned(cmd.funct7) << " rs1=0x"
              << cmd.rs1 << " rs2=0x" << cmd.rs2 << std::dec << "\n";

  ApsResult result = rtl_->execute(cmd);
  if (std::getenv("APS_COSIM_TRACE")) {
    std::cerr << "aps-rocc opcode=0x" << std::hex << unsigned(cmd.opcode)
              << " funct7=0x" << unsigned(cmd.funct7) << " rs1=0x"
              << cmd.rs1 << " rs2=0x" << cmd.rs2 << std::dec
              << " cycles=" << result.cycles;
    if (result.hasResponse)
      std::cerr << " rd=x" << unsigned(result.rd) << " data=0x" << std::hex
                << result.data << std::dec;
    std::cerr << "\n";
  }
  return result.hasResponse ? result.data : 0;
}

reg_t ApsRoccExtension::custom0(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  return execute(insn, xs1, xs2);
}

reg_t ApsRoccExtension::custom1(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  return execute(insn, xs1, xs2);
}

reg_t ApsRoccExtension::custom2(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  return execute(insn, xs1, xs2);
}

reg_t ApsRoccExtension::custom3(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  return execute(insn, xs1, xs2);
}
