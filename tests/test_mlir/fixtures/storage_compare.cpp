#include "Vgold.h"
#include "Vgate.h"
#include "verilated.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Vgold gold;
  Vgate gate;
  std::mt19937 rng(0x20261014);
  uint64_t responses = 0, transfers = 0;
  unsigned cycle = 0;
#define INPUT(port, value) gold.port = gate.port = (value)
#define CHECK(port) do { if (gold.port != gate.port) { \
    std::cerr << "FAIL cycle=" << cycle << " port=" << #port \
              << " gold=" << uint64_t(gold.port) << " gate=" << uint64_t(gate.port) << '\n'; \
    std::exit(1); } } while (0)
  auto evaluate = [&]() {
    gold.eval(); gate.eval();
    CHECK(burst_read_0_ready); CHECK(burst_read_1_ready); CHECK(burst_read_1_);
    CHECK(burst_write_ready); CHECK(rocc_cmd_ready); CHECK(hella_resp_ready);
    CHECK(dma_cpu_to_isax_ch0_cpu_addr); CHECK(dma_cpu_to_isax_ch0_isax_addr);
    CHECK(dma_cpu_to_isax_ch0_length); CHECK(dma_cpu_to_isax_ch0_stride_x);
    CHECK(dma_cpu_to_isax_ch0_stride_y); CHECK(dma_cpu_to_isax_ch0_enable);
    CHECK(dma_isax_to_cpu_ch0_cpu_addr); CHECK(dma_isax_to_cpu_ch0_isax_addr);
    CHECK(dma_isax_to_cpu_ch0_length); CHECK(dma_isax_to_cpu_ch0_stride_x);
    CHECK(dma_isax_to_cpu_ch0_stride_y); CHECK(dma_isax_to_cpu_ch0_enable);
    CHECK(dma_cpu_to_isax_ch1_cpu_addr); CHECK(dma_cpu_to_isax_ch1_isax_addr);
    CHECK(dma_cpu_to_isax_ch1_length); CHECK(dma_cpu_to_isax_ch1_stride_x);
    CHECK(dma_cpu_to_isax_ch1_stride_y); CHECK(dma_cpu_to_isax_ch1_enable);
    CHECK(dma_isax_to_cpu_ch1_cpu_addr); CHECK(dma_isax_to_cpu_ch1_isax_addr);
    CHECK(dma_isax_to_cpu_ch1_length); CHECK(dma_isax_to_cpu_ch1_stride_x);
    CHECK(dma_isax_to_cpu_ch1_stride_y); CHECK(dma_isax_to_cpu_ch1_enable);
    CHECK(rocc_resp_rocc_resp_to_bus_result_rd);
    CHECK(rocc_resp_rocc_resp_to_bus_result_rddata);
    CHECK(rocc_resp_rocc_resp_to_bus_enable);
    CHECK(hella_cmd_hella_cmd_to_bus_enable);
    // An omitted scalar adapter and an idle adapter may drive different tag
    // defaults. Payload is observable only when the command is enabled; the
    // enable and all other handshake signals are still compared unconditionally.
    if (gold.hella_cmd_hella_cmd_to_bus_enable) {
      CHECK(hella_cmd_hella_cmd_to_bus_cmd_addr); CHECK(hella_cmd_hella_cmd_to_bus_cmd_tag);
      CHECK(hella_cmd_hella_cmd_to_bus_cmd_cmd); CHECK(hella_cmd_hella_cmd_to_bus_cmd_size);
      CHECK(hella_cmd_hella_cmd_to_bus_cmd_signed); CHECK(hella_cmd_hella_cmd_to_bus_cmd_phys);
      CHECK(hella_cmd_hella_cmd_to_bus_cmd_data); CHECK(hella_cmd_hella_cmd_to_bus_cmd_mask);
    }
  };
  for (cycle = 0; cycle < 200000; ++cycle) {
    INPUT(clk, 0);
    INPUT(rst, cycle < 3 || cycle % 1009 == 0);
    INPUT(burst_read_0_enable, rng() & 1);
    INPUT(burst_read_0_addr, (rng() % 56) * 8);
    INPUT(burst_read_1_enable, rng() & 1);
    INPUT(burst_write_enable, rng() & 1);
    INPUT(burst_write_addr, (rng() % 56) * 8);
    INPUT(burst_write_data, (uint64_t(rng()) << 32) | rng());
    INPUT(rocc_cmd_enable, rng() & 1);
    INPUT(rocc_cmd_rocc_cmd_funct, rng() % 8 == 0 ? 17 : 40);
    INPUT(rocc_cmd_rocc_cmd_opcode, rng() % 8 == 0 ? 43 : 11);
    INPUT(rocc_cmd_rocc_cmd_rs1, rng() & 31);
    INPUT(rocc_cmd_rocc_cmd_rs2, rng() & 31);
    INPUT(rocc_cmd_rocc_cmd_rd, rng() & 31);
    INPUT(rocc_cmd_rocc_cmd_xs1, rng() & 1);
    INPUT(rocc_cmd_rocc_cmd_xs2, rng() & 1);
    INPUT(rocc_cmd_rocc_cmd_xd, rng() & 1);
    INPUT(rocc_cmd_rocc_cmd_rs1data, rng());
    INPUT(rocc_cmd_rocc_cmd_rs2data, rng());
    INPUT(hella_resp_enable, rng() & 1);
    INPUT(hella_resp_hella_resp_data, rng());
    INPUT(hella_resp_hella_resp_tag, rng() & 255);
    INPUT(hella_resp_hella_resp_cmd, rng() & 31);
    INPUT(hella_resp_hella_resp_size, rng() & 3);
    INPUT(hella_resp_hella_resp_signed, rng() & 1);
    INPUT(dma_cpu_to_isax_ch0_ready, rng() & 1);
    INPUT(dma_isax_to_cpu_ch0_ready, rng() & 1);
    INPUT(dma_poll_for_idle_ch0_, rng() & 1);
    INPUT(dma_poll_for_idle_ch0_ready, rng() & 1);
    INPUT(dma_cpu_to_isax_ch1_ready, rng() & 1);
    INPUT(dma_isax_to_cpu_ch1_ready, rng() & 1);
    INPUT(dma_poll_for_idle_ch1_, rng() & 1);
    INPUT(dma_poll_for_idle_ch1_ready, rng() & 1);
    INPUT(rocc_resp_rocc_resp_to_bus_ready, rng() & 1);
    INPUT(hella_cmd_hella_cmd_to_bus_ready, rng() & 1);
    evaluate();
    if (!gold.rst) {
      responses += gold.rocc_resp_rocc_resp_to_bus_enable;
      transfers += gold.dma_cpu_to_isax_ch0_enable + gold.dma_cpu_to_isax_ch1_enable
                 + gold.dma_isax_to_cpu_ch0_enable + gold.dma_isax_to_cpu_ch1_enable;
    }
    INPUT(clk, 1); evaluate();
    INPUT(clk, 0); evaluate();
  }
  if (responses < 100 || transfers < 1000) {
    std::cerr << "FAIL insufficient activity\n";
    return 1;
  }
  std::cout << "PASS cycles=" << cycle << " responses=" << responses
            << " DMA_transfers=" << transfers << " all_valid_output_ports_match\n";
}
