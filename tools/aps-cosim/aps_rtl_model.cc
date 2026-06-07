#include "aps_rtl_model.h"

#include "Vmain.h"

#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>

namespace {

std::unique_ptr<Vmain> makeTop() {
  Verilated::traceEverOn(true);
  return std::make_unique<Vmain>();
}

} // namespace

ApsRtlModel::ApsRtlModel(MemoryBridge &memory, ApsCsrFile *csrs)
    : top_(makeTop()), csrs_(csrs ? *csrs : localCsrs_),
      dma_(memory) {
  driveDefaults();
}

ApsRtlModel::~ApsRtlModel() = default;

void ApsRtlModel::enableVcd(const std::string &path) {
  if (trace_)
    throw std::runtime_error("VCD tracing is already enabled");
  Verilated::traceEverOn(true);
  trace_ = std::make_unique<VerilatedVcdC>();
  top_->trace(trace_.get(), 99);
  trace_->open(path.c_str());
}

void ApsRtlModel::eval() {
  top_->eval();
  if (trace_)
    trace_->dump(traceTime_++);
}

void ApsRtlModel::driveDefaults() {
  top_->clk = 0;
  top_->rst = 0;
  top_->rocc_cmd_enable = 0;
  top_->rocc_cmd_rocc_cmd_funct = 0;
  top_->rocc_cmd_rocc_cmd_rs1 = 0;
  top_->rocc_cmd_rocc_cmd_rs2 = 0;
  top_->rocc_cmd_rocc_cmd_rd = 0;
  top_->rocc_cmd_rocc_cmd_xs1 = 0;
  top_->rocc_cmd_rocc_cmd_xs2 = 0;
  top_->rocc_cmd_rocc_cmd_xd = 0;
  top_->rocc_cmd_rocc_cmd_opcode = 0;
  top_->rocc_cmd_rocc_cmd_rs1data = 0;
  top_->rocc_cmd_rocc_cmd_rs2data = 0;
  top_->rocc_resp_rocc_resp_to_bus_ready = 1;
  dma_.driveInputs(*top_);
}

void ApsRtlModel::reset(unsigned cycles) {
  top_->rst = 1;
  for (unsigned i = 0; i < cycles; ++i)
    tick();
  top_->rst = 0;
  tick();
}

void ApsRtlModel::tick() {
  dma_.driveInputs(*top_);
  csrRuntime_.drive(*top_, csrs_);
  top_->clk = 0;
  eval();
  top_->clk = 1;
  eval();
  csrRuntime_.sample(*top_, csrs_);
  dma_.sampleOutputs(*top_);
  top_->clk = 0;
  eval();
  ++cycle_;
}

bool ApsRtlModel::tickCommandCycle() {
  dma_.driveInputs(*top_);
  csrRuntime_.drive(*top_, csrs_);
  top_->clk = 0;
  eval();
  bool readyBeforeClock = top_->rocc_cmd_ready;
  top_->clk = 1;
  eval();
  csrRuntime_.sample(*top_, csrs_);
  dma_.sampleOutputs(*top_);
  top_->clk = 0;
  eval();
  ++cycle_;
  return readyBeforeClock;
}

ApsResult ApsRtlModel::execute(const ApsCommand &cmd,
                               uint64_t timeoutCycles) {
  uint64_t start = cycle_;
  bool accepted = false;
  ApsResult result;

  for (;;) {
    if (!accepted) {
      top_->rocc_cmd_enable = 1;
      top_->rocc_cmd_rocc_cmd_funct = cmd.funct7;
      top_->rocc_cmd_rocc_cmd_rs1 = 0;
      top_->rocc_cmd_rocc_cmd_rs2 = 0;
      top_->rocc_cmd_rocc_cmd_rd = cmd.rd;
      top_->rocc_cmd_rocc_cmd_xs1 = cmd.xs1;
      top_->rocc_cmd_rocc_cmd_xs2 = cmd.xs2;
      top_->rocc_cmd_rocc_cmd_xd = cmd.xd;
      top_->rocc_cmd_rocc_cmd_opcode = cmd.opcode;
      top_->rocc_cmd_rocc_cmd_rs1data = cmd.rs1;
      top_->rocc_cmd_rocc_cmd_rs2data = cmd.rs2;
    } else {
      top_->rocc_cmd_enable = 0;
    }

    bool readyBeforeClock = tickCommandCycle();

    if (!accepted && readyBeforeClock) {
      accepted = true;
      top_->rocc_cmd_enable = 0;
    }

    if (top_->rocc_resp_rocc_resp_to_bus_enable) {
      result.rd = top_->rocc_resp_rocc_resp_to_bus_result_rd;
      result.data = top_->rocc_resp_rocc_resp_to_bus_result_rddata;
      result.hasResponse = true;
      result.cycles = cycle_ - start;
      return result;
    }

    if (std::getenv("APS_COSIM_TRACE")) {
      std::cerr << "cycle=" << cycle_
                << " cmd_ready=" << unsigned(top_->rocc_cmd_ready)
                << " resp=" << unsigned(top_->rocc_resp_rocc_resp_to_bus_enable)
                << " dma_c2i0=" << unsigned(top_->dma_cpu_to_isax_ch0_enable)
                << " dma_i2c0=" << unsigned(top_->dma_isax_to_cpu_ch0_enable)
                << " poll0=" << unsigned(top_->dma_poll_for_idle_ch0_ready)
                << " hella=" << unsigned(top_->hella_cmd_hella_cmd_to_bus_enable)
                << "\n";
    }

    if (accepted && dma_.idle() && !cmd.xd) {
      result.hasResponse = false;
      result.cycles = cycle_ - start;
      return result;
    }

    if (cycle_ - start > timeoutCycles)
      throw std::runtime_error("APS RTL command timed out");
  }
}
