# APS Spike + Verilator 快速 Co-Sim 方案

## 目标

用 Spike 执行 RISC-V 程序，用 Verilator 仿真 APS 生成的 RTL。自定义指令进入 APS RTL 后，RTL 通过 fake DMA 访问同一份 CPU memory，从而绕过完整 Chipyard/Rocket SoC elaboration。

第一版目标是跑通现有 bare-metal 测试，例如 `deca_e2e_opt.riscv + crypto_pqc.sv`，并以程序内部的软件参考比较输出（例如 `PQC PASS`）作为 correctness 信号。

## 核心结构

```text
RISC-V ELF
   |
   v
Spike CPU model
   |
   | custom opcode/funct7
   v
Spike APS extension
   |
   | RoCC cmd/resp
   v
Verilated APS RTL
   |
   | dma_cpu_to_isax / dma_isax_to_cpu
   | burst_read / burst_write
   | hella_cmd / hella_resp
   v
Fake DMA + memory bridge
   |
   v
Spike memory / shared host memory image
```

Spike 负责 CPU 指令、寄存器和程序控制流。Verilated APS RTL 负责 APS 自定义指令语义。Fake DMA 负责把 RTL 的数据搬运请求转换成对 CPU memory 的读写。

## 为什么不直接接 TileLink

当前 Chipyard 集成里，TileLink 是 `ApsRoccWrapper` 额外加出来的 SoC 侧接口：

- `main` 是 APS 生成 RTL 的 blackbox。
- `ApsRoccWrapper` 把 RoCC cmd/resp 接到 `main`。
- `ApsRoccWrapper` 还实例化 `TlDmaMultichannel`，把 APS DMA 请求转成 TileLink A/D。
- Rocket/Chipyard 再把这个 TL client 接到 SoC memory system。

快速 co-sim 的第一版不需要证明 TL 协议，只需要证明 APS RTL 对 CPU memory 的功能效果正确。因此第一版直接在 C++ harness 里模拟 `TlDmaMultichannel` 的功能语义：从 CPU memory 拷贝到 APS scratchpad，或从 APS scratchpad 拷贝回 CPU memory。

第二版再做 TL-faithful mini harness，单独验证 `TlDmaMultichannel` 的 A/D channel、source id、burst size、ready/valid backpressure。

## 需要驱动的 APS 顶层端口

以 `tools/aps-cosim/example/crypto_pqc.sv` 的 `main` 为例，关键端口分三组。

### RoCC 指令入口

```verilog
input         rocc_cmd_enable
output        rocc_cmd_ready
input  [6:0]  rocc_cmd_rocc_cmd_funct
input  [4:0]  rocc_cmd_rocc_cmd_rs1
input  [4:0]  rocc_cmd_rocc_cmd_rs2
input  [4:0]  rocc_cmd_rocc_cmd_rd
input         rocc_cmd_rocc_cmd_xs1
input         rocc_cmd_rocc_cmd_xs2
input         rocc_cmd_rocc_cmd_xd
input  [6:0]  rocc_cmd_rocc_cmd_opcode
input  [31:0] rocc_cmd_rocc_cmd_rs1data
input  [31:0] rocc_cmd_rocc_cmd_rs2data
```

Spike APS extension 解码 custom instruction 后，把 `opcode/funct7/rs1/rs2/rd/xd/xs1/xs2` 填入这些端口，等待 `rocc_cmd_ready` 后打一拍 `rocc_cmd_enable`。

### RoCC 响应出口

```verilog
output [4:0]  rocc_resp_rocc_resp_to_bus_result_rd
output [31:0] rocc_resp_rocc_resp_to_bus_result_rddata
input         rocc_resp_rocc_resp_to_bus_ready
output        rocc_resp_rocc_resp_to_bus_enable
```

Harness 常置 `rocc_resp_rocc_resp_to_bus_ready = 1`。当 `rocc_resp_rocc_resp_to_bus_enable` 为 1 时，把 `rddata` 写回 Spike 的 `rd`。

### Fake DMA / scratchpad 访问

APS RTL 发起 CPU memory 到 ISAX scratchpad 的读：

```verilog
output [31:0] dma_cpu_to_isax_chN_cpu_addr
output [31:0] dma_cpu_to_isax_chN_isax_addr
output [3:0]  dma_cpu_to_isax_chN_length
output [7:0]  dma_cpu_to_isax_chN_stride_x
output [7:0]  dma_cpu_to_isax_chN_stride_y
input         dma_cpu_to_isax_chN_ready
output        dma_cpu_to_isax_chN_enable
```

APS RTL 发起 ISAX scratchpad 到 CPU memory 的写：

```verilog
output [31:0] dma_isax_to_cpu_chN_cpu_addr
output [31:0] dma_isax_to_cpu_chN_isax_addr
output [3:0]  dma_isax_to_cpu_chN_length
output [7:0]  dma_isax_to_cpu_chN_stride_x
output [7:0]  dma_isax_to_cpu_chN_stride_y
input         dma_isax_to_cpu_chN_ready
output        dma_isax_to_cpu_chN_enable
```

APS RTL 内部 scratchpad 的读写入口：

```verilog
input         burst_read_0_enable
output        burst_read_0_ready
input  [31:0] burst_read_0_addr
input         burst_read_1_enable
output        burst_read_1_ready
output [63:0] burst_read_1_res0
input         burst_write_enable
output        burst_write_ready
input  [31:0] burst_write_addr
input  [63:0] burst_write_data
```

Fake DMA 的职责是：

1. 看到 `dma_cpu_to_isax_chN_enable` 时，从 CPU memory 的 `cpu_addr` 读取一段数据。
2. 通过 `burst_write_*` 把数据写入 APS scratchpad 的 `isax_addr`。
3. 看到 `dma_isax_to_cpu_chN_enable` 时，通过 `burst_read_*` 从 APS scratchpad 的 `isax_addr` 读取数据。
4. 把读出的数据写回 CPU memory 的 `cpu_addr`。

当前实现让各 DMA channel 的请求入口可以排队，scratchpad `burst_*` 访问仍串行执行。这样更接近 Chipyard 里的 `TlDmaMultichannel`：多个 channel 可以先接受请求，`dma_poll_for_idle_chN_ready/res0` 只有在 active request 和 queue 都为空时才为 1。

## Fake DMA 传输语义

第一版采用简单、可调试的队列 + 阻塞式 scratchpad 状态机。

```text
IDLE
  if queue not empty:
    pop(cpu_addr, isax_addr, length, stride)
    -> WRITE_ISAX or READ_ISAX_ADDR

LOAD_CPU_WORD
  read 64-bit word from Spike memory
  drive burst_write_addr/data/enable
  -> WRITE_ISAX

WRITE_ISAX
  wait burst_write_ready
  advance offset
  if done -> IDLE else -> LOAD_CPU_WORD

READ_ISAX_ADDR
  drive burst_read_0_addr/enable
  wait burst_read_0_ready
  -> READ_ISAX_DATA

READ_ISAX_DATA
  drive burst_read_1_enable
  wait burst_read_1_ready
  capture burst_read_1_res0
  write 64-bit word to Spike memory
  advance offset
  if done -> IDLE else -> READ_ISAX_ADDR
```

请求采样逻辑和执行逻辑分离：只要 `dma_*_enable && dma_*_ready`，fake DMA 就把请求推入队列；真正的 `burst_read/burst_write` 由一个串行状态机执行。

`length` 当前按 `bytes = 1 << length` 解释。这个规则匹配当前 `crypto_pqc.sv` 里的实测编码：`length = 5` 搬 32B，`length = 6` 搬 64B。`stride_x/stride_y` 按 Chipyard `tl_dma_multichannel` 的 tiling 语义应用在 ISAX scratchpad 地址上：`stride_x == 0` 时线性 `+8`，否则每行内按 64-bit beat 前进，行尾跳到 `addr + stride_x * stride_y - tile_offset`。CPU memory 地址保持线性。

## HellaCache 路径

部分 APS 生成物还会走 HellaCache 风格的单次 load/store：

```verilog
output [31:0] hella_cmd_hella_cmd_to_bus_cmd_addr
output [7:0]  hella_cmd_hella_cmd_to_bus_cmd_tag
output [4:0]  hella_cmd_hella_cmd_to_bus_cmd_cmd
output [1:0]  hella_cmd_hella_cmd_to_bus_cmd_size
output [31:0] hella_cmd_hella_cmd_to_bus_cmd_data
output [3:0]  hella_cmd_hella_cmd_to_bus_cmd_mask
input         hella_cmd_hella_cmd_to_bus_ready
output        hella_cmd_hella_cmd_to_bus_enable

input         hella_resp_enable
input  [31:0] hella_resp_hella_resp_data
input  [7:0]  hella_resp_hella_resp_tag
```

第一版 fake memory bridge 也要支持这条路：

- `hella_cmd_ready = 1` when bridge idle。
- load：按 `addr/size/signed` 从 CPU memory 读，下一拍或固定延迟后回 `hella_resp_enable/data/tag`。
- store：按 `addr/data/mask/size` 写 CPU memory，可以回一个 store ack，tag 原样返回。

如果当前测试只走 DMA，可以先实现但不开启复杂乱序行为。

## 当前 direct Verilator harness

当前已经落地的第一步是不接 Spike，直接用 host memory 驱动 generated RTL：

```bash
make -C tools/aps-cosim RTL=example/crypto_pqc.sv TOP=main
```

运行一条 GEMM APS 指令并 dump 回写结果：

```bash
perl -e 'print pack("C*", 1..32)' > /tmp/aps-a.bin
perl -e 'print pack("C*", 33..64)' > /tmp/aps-b.bin

./tools/aps-cosim/obj_dir/Vmain \
  --load-bin 0x80000000 /tmp/aps-a.bin \
  --load-bin 0x80000100 /tmp/aps-b.bin \
  --cmd 0x2b 0x38 1 0x80000000 0x80000100 \
  --dump-bin 0x80000120 32 /tmp/aps-gemm-pattern-out.bin

xxd -g4 /tmp/aps-gemm-pattern-out.bin
```

期望可以看到非零回写：

```text
cmd opcode=0x2b funct7=0x38 cycles=47 rd=x1 data=0x0
00000000: 00080008 00080000 00180018 00180000
00000010: 00080008 00080000 00380038 00380000
```

打开 trace 可以观察 fake DMA 请求和 scratchpad 搬运：

```bash
APS_COSIM_TRACE=1 ./tools/aps-cosim/obj_dir/Vmain ...
```

## Spike 侧集成

当前已经落地 `tools/aps-cosim` 内的 Spike runner：

```bash
make -C tools/aps-cosim spike RTL=example/crypto_pqc.sv TOP=main
```

编译 RV32 zero-copy GEMM example：

```bash
pixi run compile-native \
  tools/aps-cosim/example/gemm_zero_copy.c \
  /tmp/aps-gemm-zero-copy-example.riscv
```

运行：

```bash
pixi run aps-cosim-run /tmp/aps-gemm-zero-copy-example.riscv
```

如果 RTL 里带 CSR 端口，用 `--add-csr name=addr` 手工声明 CSR 地址。该参数可以重复多次，也支持 `mask` 和 `init`：

```bash
pixi run aps-cosim-run \
  /tmp/aps-csr-partition-loop.riscv \
  /tmp/aps-csr-partition-loop.sv \
  "--add-csr gain_cfg=0x801 --add-csr bias_cfg=0x802"
```

runner 会把这些地址注册成 Spike custom CSR，并在运行时通过 Verilator VPI 连接 `TOP.csr_<name>_value_*` 和 `TOP.csr_<name>_set_*` 端口。因此不需要 YAML/config，也不需要为不同 RTL 生成 C++ 绑定头。

期望输出：

```text
APS GEMM EXAMPLE PASS
```

打开 trace 可以看到 `.insn` 进入 APS RoCC extension，并由 fake DMA 直接读写 Spike 程序内存：

```bash
APS_COSIM_TRACE=1 ./tools/aps-cosim/obj_dir_spike/Vmain /tmp/aps-gemm-zero-copy-example.riscv
```

Spike APS extension 做三件事：

1. 注册 APS custom instruction。
2. 对每条 custom instruction 调 `ApsRtlModel::execute(opcode, funct7, rs1, rs2, rd, xd)`。
3. 让 `ApsRtlModel` 通过 `MemoryBridge` 读写 Spike memory。

接口建议：

```c++
struct ApsCommand {
  uint8_t opcode;
  uint8_t funct7;
  uint8_t rd;
  uint32_t rs1;
  uint32_t rs2;
  bool xd;
  bool xs1;
  bool xs2;
};

class ApsRtlModel {
public:
  uint32_t execute(const ApsCommand &cmd, MemoryBridge &mem);
};
```

`MemoryBridge` 第一版只需要：

```c++
class MemoryBridge {
public:
  uint64_t load64(uint32_t addr);
  uint32_t load32(uint32_t addr);
  void store64(uint32_t addr, uint64_t value, uint8_t mask = 0xff);
  void store32(uint32_t addr, uint32_t value, uint8_t mask = 0x0f);
};
```

后续如果要和 Spike 的真实 `mem_t`/MMU 语义更贴近，再把物理地址访问替换成 Spike processor/mmu 的 load/store helper。

当前代码已经把 `MemoryBridge` 抽成接口：

- `HostMemoryBridge`：direct Verilator runner 使用的 byte-array CPU memory。
- `SpikeMemoryBridge`：Spike runner 使用 `processor_t::get_mmu()->load/store` 访问 guest memory。

RoCC 指令注册使用 `ApsRoccExtension`，显式覆盖 RV32/RV64 custom opcode 映射。原因是 Chipyard 这版 Spike 的默认 `rocc_t::get_instructions()` 只在 RV64 lane 上接 custom handler，RV32 lane 是 illegal；APS runner 需要 RV32，因此不能直接继承默认映射。

## 建议落地顺序

1. `tools/aps-cosim/rtl_harness.{h,cc}`：Verilated APS model、clock/reset、RoCC cmd/resp。
2. `tools/aps-cosim/fake_dma.{h,cc}`：DMA + HellaCache 到 `MemoryBridge`。
3. `tools/aps-cosim/memory_bridge.{h,cc}`：先用 host byte array 模拟 CPU memory。
4. `tools/aps-cosim/run_direct.cc`：不接 Spike，手工喂一条 custom instruction 和一块 memory，快速验证 RTL/fake DMA。
5. `tools/aps-cosim/spike_extension.cc`：接入 Spike，跑 `.riscv`。
6. 用 `deca_e2e_opt.riscv` 验证 `PQC PASS`。

第一版先把 fake DMA 和 generated RTL 跑通，Spike extension 可以随后接入。这样调试面更小：先证明 RTL 能通过 fake DMA 正确读写一块 CPU memory，再把这块 memory 换成 Spike 的内存。

## 和 Chipyard 的关系

这个 co-sim 不是替代 Chipyard 集成验证，而是日常快速功能回归：

- 快速 co-sim：验证 APS RTL 的功能效果、DMA 数据搬运、custom instruction 软件可见结果。
- Chipyard `APSRocketConfig`：验证 Rocket/RoCC/TileLink/缓存/SoC 集成。
- TL-faithful mini harness：介于二者之间，专门验证 `TlDmaMultichannel` 协议行为。

最终回归建议保留三档：

```text
Level 0: direct Verilator APS fake-DMA test
Level 1: Spike + Verilator APS fake-DMA co-sim
Level 2: Chipyard APSRocketConfig integration test
```
