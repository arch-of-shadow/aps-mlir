#include "aps_rocc_extension.h"
#include "csr_args.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <riscv/cfg.h>
#include <riscv/devices.h>
#include <riscv/sim.h>
#include <string>
#include <vector>

namespace {

std::vector<std::pair<reg_t, abstract_mem_t *>>
makeMems(const std::vector<mem_cfg_t> &layout) {
  std::vector<std::pair<reg_t, abstract_mem_t *>> mems;
  mems.reserve(layout.size());
  for (const auto &cfg : layout)
    mems.emplace_back(cfg.get_base(), new mem_t(cfg.get_size()));
  return mems;
}

void usage(const char *argv0) {
  std::cerr << "Usage: " << argv0 << " [--isa <isa>] <elf> [target args...]\n"
            << "  --add-csr <name=addr[,mask=...,init=...]>\n"
            << "  --vcd <path>\n"
            << "Default ISA: RV32IMAFDC_zicsr_zifencei\n";
}

} // namespace

int main(int argc, char **argv) {
  cfg_t cfg;
  cfg.isa = "RV32IMAFDC_zicsr_zifencei";

  std::vector<std::string> htifArgs;
  std::vector<ApsCsrSpec> csrSpecs;
  std::string vcdPath;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 ||
        std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      return 0;
    }
    if (std::strcmp(argv[i], "--isa") == 0) {
      if (++i >= argc) {
        usage(argv[0]);
        return 1;
      }
      cfg.isa = argv[i];
      continue;
    }
    if (std::strcmp(argv[i], "--add-csr") == 0) {
      if (++i >= argc) {
        usage(argv[0]);
        return 1;
      }
      csrSpecs.push_back(parseApsCsrSpec(argv[i]));
      continue;
    }
    if (std::strcmp(argv[i], "--vcd") == 0) {
      if (++i >= argc) {
        usage(argv[0]);
        return 1;
      }
      vcdPath = argv[i];
      continue;
    }
    htifArgs.emplace_back(argv[i]);
  }

  if (htifArgs.empty()) {
    usage(argv[0]);
    return 1;
  }

  debug_module_config_t dmConfig = {
      .progbufsize = 2,
      .max_sba_data_width = 0,
      .require_authentication = false,
      .abstract_rti = 0,
      .support_hasel = true,
      .support_abstract_csr_access = true,
      .support_abstract_fpr_access = true,
      .support_haltgroups = true,
      .support_impebreak = true};

  std::vector<device_factory_sargs_t> pluginDevices;
  auto mems = makeMems(cfg.mem_layout);

  sim_t sim(&cfg, false, mems, pluginDevices, htifArgs, dmConfig,
            nullptr, // log_path
            true,    // dtb_enabled
            nullptr, // dtb_file
            false,   // socket_enabled
            nullptr  // cmd_file
  );

  for (size_t i = 0; i < cfg.nprocs(); ++i) {
    auto *aps = new ApsRoccExtension();
    sim.get_core(i)->register_extension(aps);
    if (!vcdPath.empty())
      aps->setVcdPath(vcdPath);
    for (const auto &csr : csrSpecs)
      aps->addCsr(csr);
    aps->registerCsrs();
  }

  int rc = sim.run();

  for (auto &mem : mems)
    delete mem.second;

  return rc;
}
