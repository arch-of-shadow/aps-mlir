#include "aps_rtl_model.h"
#include "csr_args.h"
#include "memory_bridge.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint32_t parseU32(const std::string &s) {
  return static_cast<uint32_t>(std::stoul(s, nullptr, 0));
}

void usage(const char *argv0) {
  std::cerr
      << "usage: " << argv0 << " [options]\n"
      << "  --mem-base <addr>       CPU memory base, default 0x80000000\n"
      << "  --mem-size <bytes>      CPU memory size, default 0x01000000\n"
      << "  --load-bin <addr> <bin> Load raw bytes into CPU memory\n"
      << "  --dump-bin <addr> <n> <bin> Dump raw bytes after execution\n"
      << "  --add-csr <name=addr[,mask=...,init=...]>\n"
      << "  --vcd <path>\n"
      << "  --cmd <opcode> <funct7> <rd> <rs1> <rs2> [xd]\n";
}

struct Load {
  uint32_t addr;
  std::string path;
};

struct Dump {
  uint32_t addr;
  size_t size;
  std::string path;
};

} // namespace

int main(int argc, char **argv) {
  try {
    uint32_t memBase = 0x80000000u;
    size_t memSize = 0x01000000u;
    std::vector<Load> loads;
    std::vector<Dump> dumps;
    std::vector<ApsCommand> commands;
    ApsCsrFile csrs;
    std::string vcdPath;

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help") {
        usage(argv[0]);
        return 0;
      } else if (arg == "--mem-base" && i + 1 < argc) {
        memBase = parseU32(argv[++i]);
      } else if (arg == "--mem-size" && i + 1 < argc) {
        memSize = std::stoul(argv[++i], nullptr, 0);
      } else if (arg == "--load-bin" && i + 2 < argc) {
        loads.push_back({parseU32(argv[++i]), argv[++i]});
      } else if (arg == "--dump-bin" && i + 3 < argc) {
        uint32_t addr = parseU32(argv[++i]);
        size_t size = std::stoul(argv[++i], nullptr, 0);
        dumps.push_back({addr, size, argv[++i]});
      } else if (arg == "--add-csr" && i + 1 < argc) {
        ApsCsrSpec csr = parseApsCsrSpec(argv[++i]);
        csrs.define(csr.name, csr.address, true, csr.mask, csr.init);
      } else if (arg == "--vcd" && i + 1 < argc) {
        vcdPath = argv[++i];
      } else if (arg == "--cmd" && i + 5 < argc) {
        ApsCommand cmd;
        cmd.opcode = static_cast<uint8_t>(parseU32(argv[++i]));
        cmd.funct7 = static_cast<uint8_t>(parseU32(argv[++i]));
        cmd.rd = static_cast<uint8_t>(parseU32(argv[++i]));
        cmd.rs1 = parseU32(argv[++i]);
        cmd.rs2 = parseU32(argv[++i]);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
          cmd.xd = parseU32(argv[++i]) != 0;
        commands.push_back(cmd);
      } else {
        usage(argv[0]);
        return 2;
      }
    }

    HostMemoryBridge memory(memBase, memSize);
    for (const auto &load : loads)
      memory.loadBinary(load.addr, load.path);

    ApsRtlModel model(memory, &csrs);
    if (!vcdPath.empty())
      model.enableVcd(vcdPath);
    model.reset();

    for (const auto &cmd : commands) {
      ApsResult result = model.execute(cmd);
      std::cout << "cmd opcode=0x" << std::hex << unsigned(cmd.opcode)
                << " funct7=0x" << unsigned(cmd.funct7) << std::dec
                << " cycles=" << result.cycles;
      if (result.hasResponse)
        std::cout << " rd=x" << unsigned(result.rd) << " data=0x" << std::hex
                  << result.data << std::dec;
      std::cout << "\n";
    }

    for (const auto &dump : dumps)
      memory.dumpBinary(dump.addr, dump.size, dump.path);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "aps-cosim: " << e.what() << "\n";
    return 1;
  }
}
