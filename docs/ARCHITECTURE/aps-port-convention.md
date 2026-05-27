# APS Port Naming and Call Convention

本文定义 APS generated top、Chipyard blackbox wrapper、软件侧配置之间的端口命名和调用约定。目标是让 APS-MLIR 生成的 `main.sv`、Chipyard `MainBlackBox`、`aps_config.yaml` 使用同一套名字，不再靠手写端口对齐。

## Scope

本文只约束 APS top 的外部接口命名和调用边界：

- RoCC command/response ports
- Rocket RoCC custom CSR ports
- DMA/burst ports
- HellaCache request/response ports

内部 CMT2 rule、APS dialect op、memory bank 名字不在本文范围内。

## Global Rules

所有 generated top ports 必须满足：

1. 使用 `lower_snake_case`。
2. 不使用 Verilog escaped identifier。
3. 不使用 Chisel/Verilog 保留字。
4. 配置驱动的名字必须先在配置解析阶段检查合法性。
5. 同一 top module 内端口名必须唯一。

方向以 APS generated top 为中心定义：

- `input` 表示 Chipyard/Rocket 给 APS。
- `output` 表示 APS 给 Chipyard/Rocket。

## YAML CSR Names

CSR 在 `aps_config.yaml` 中声明：

```yaml
csrs:
  - id: 0x810
    name: "0"
    mask: 0xffffffff
    init: 0x0
  - id: 0x811
    name: status
    mask: 0xffffffff
    init: 0x0
```

`id` 是 Rocket CSR address，`name` 是 APS 端口后缀。实际 top port 名固定加 `csr_` prefix：

```text
csr_<name>_<signal>
```

例如：

| YAML `name` | Generated prefix |
|-------------|------------------|
| `"0"` | `csr_0_` |
| `"1"` | `csr_1_` |
| `status` | `csr_status_` |
| `cfg_base` | `csr_cfg_base_` |

不要在 YAML 里写带 `csr_` 的名字，除非确实想生成 `csr_csr_*`。

## CSR Port Set

每个 CSR lane 在 APS generated top 上只暴露 committed value read 和 CSR writeback set 两个 method-style 接口：

```text
input  csr_<name>_value_res0
input  csr_<name>_value_ready
output csr_<name>_set_<name>_sdata
input  csr_<name>_set_ready
output csr_<name>_set_enable
```

推荐宽度：

| Port | Width | Meaning |
|------|-------|---------|
| `value_res0` | `xLen` | committed CSR value in Rocket `CSRFile` |
| `value_ready` | 1 | value method readiness, driven constant true by wrapper |
| `set_<name>_sdata` | `xLen` | APS update data |
| `set_ready` | 1 | set method readiness, driven constant true by wrapper |
| `set_enable` | 1 | APS updates CSR value |

第一版 generated top 如果不支持 CSR status writeback，必须显式输出：

```systemverilog
assign csr_<name>_set_enable       = 1'b0;
assign csr_<name>_set_<name>_sdata = '0;
```

Rocket `CustomCSRIO` 中的 `ren/wen/wdata/stall` 不暴露给 `main`。wrapper 内部固定 `stall=0`，把 committed `value` 接到 `value_res0`，并把 `value_ready` 和 `set_ready` 都驱动为 true。

不要省略 `set_enable/set_<name>_sdata` output port，也不要让 wrapper tie off 缺失端口。端口集合必须由 APS generated top 和 Chipyard blackbox 同时实现。

## CSR Ordering

`aps_config.yaml` 中 `csrs` 的 list 顺序定义 RoCC CSR lane 顺序：

```text
csrs[0] <-> io.rocc.csrs(0) <-> first CustomCSR
csrs[1] <-> io.rocc.csrs(1) <-> second CustomCSR
```

`name` 只影响端口名，不影响 Rocket CSR address 解码。软件通过 `id` 访问 CSR，硬件 wrapper 通过 list index 连接 CSR lane。

例如：

```yaml
csrs:
  - id: 0x810
    name: cfg
  - id: 0x811
    name: status
```

wrapper 连接关系是：

```text
io.rocc.csrs(0) <-> csr_cfg_value_* and csr_cfg_set_*
io.rocc.csrs(1) <-> csr_status_value_* and csr_status_set_*
```

## CSR Call Convention

CSR 不是 ready/valid transaction interface。`value` 是 Rocket `CSRFile` 中的 committed state，是当前传入 generated top 的唯一 CSR read path。

配置型调用约定：

```text
CPU:
  csrw csr_cfg, value
  custom0 ...

APS:
  when rocc_cmd fires:
    latch csr_cfg_value_res0 into command context
```

规则：

1. APS 在 `rocc_cmd` 接收边界采样配置 CSR。
2. 一个 RoCC transaction 启动后，只使用 latch 后的 command context。
3. 不要在长事务过程中直接反复读取裸 `csr_<name>_value_res0`。
4. wrapper 内部默认 `stall=0`，不要用 CSR stall 等待长任务完成。

状态型调用约定：

```text
APS:
  csr_status_set_enable = status_update_pulse
  csr_status_set_status_sdata = status_value

CPU:
  csrr a0, csr_status
```

如果状态来自长任务，APS 应更新 CSR committed value，CPU 后续读取最近提交状态。不要让 `ren` 阻塞直到任务完成。

## RoCC Command Ports

RoCC command port 命名保持当前 APS top 约定：

```text
input  rocc_cmd_enable
output rocc_cmd_ready
input  rocc_cmd_rocc_cmd_funct
input  rocc_cmd_rocc_cmd_rs1
input  rocc_cmd_rocc_cmd_rs2
input  rocc_cmd_rocc_cmd_rd
input  rocc_cmd_rocc_cmd_xs1
input  rocc_cmd_rocc_cmd_xs2
input  rocc_cmd_rocc_cmd_xd
input  rocc_cmd_rocc_cmd_opcode
input  rocc_cmd_rocc_cmd_rs1data
input  rocc_cmd_rocc_cmd_rs2data
```

调用语义是 ready/valid：

```text
rocc_cmd.fire = rocc_cmd_enable && rocc_cmd_ready
```

APS 必须只在 fire cycle 消费 command fields，并在同一 cycle 采样需要的 CSR `value`。

## RoCC Response Ports

RoCC response port 命名保持当前 APS top 约定：

```text
output rocc_resp_rocc_resp_to_bus_enable
input  rocc_resp_rocc_resp_to_bus_ready
output rocc_resp_rocc_resp_to_bus_result_rd
output rocc_resp_rocc_resp_to_bus_result_rddata
```

调用语义是 ready/valid：

```text
rocc_resp.fire =
  rocc_resp_rocc_resp_to_bus_enable &&
  rocc_resp_rocc_resp_to_bus_ready
```

APS 必须保持 response payload stable until fire。

## DMA and Burst Ports

DMA sideband 端口继续使用 channel-indexed naming：

```text
dma_cpu_to_isax_ch<idx>_<field>
dma_isax_to_cpu_ch<idx>_<field>
dma_poll_for_idle_ch<idx>_<field>
```

Burst memory ports 继续使用 existing generated names：

```text
burst_read_0_enable
burst_read_0_ready
burst_read_0_addr
burst_read_1_enable
burst_read_1_ready
burst_read_1_res0
burst_write_enable
burst_write_ready
burst_write_addr
burst_write_data
```

这些端口是 wrapper 内部 DMA engine 和 APS top 之间的协议，不使用 CSR naming rule。

## Implementation Requirements

Chipyard side:

- `APSConfig.csrs` 生成 Rocket `CustomCSR` list。
- `ApsRoccWrapper` 按 `config.csrs.map(_.name)` 动态生成 blackbox ports。
- 实际 blackbox CSR port prefix 固定为 `csr_${name}`。
- 每个 CSR 只向 `main` 暴露 `value_res0/value_ready/set_<name>_sdata/set_ready/set_enable`。
- `csrs.name` 必须唯一，且只能包含字母、数字、下划线。

APS-MLIR generated top side:

- 读取同一个 `aps_config.yaml` 或同源配置。
- 为每个 CSR 生成 `csr_<name>_value_res0`、`csr_<name>_value_ready`、`csr_<name>_set_<name>_sdata`、`csr_<name>_set_ready`、`csr_<name>_set_enable`。
- 对未实现写回的 CSR 输出默认 `set_enable=0,set_<name>_sdata=0`。
- 在 command receive boundary latch 需要的 CSR value result。

Software side:

- 使用 `id` 生成 CSR address macro。
- 使用 `name` 生成可读 macro/function suffix。
- 不依赖 YAML list index 作为软件 ABI。

## Example

配置：

```yaml
csrs:
  - id: 0x810
    name: "0"
    mask: 0xffffffff
    init: 0x0
  - id: 0x811
    name: status
    mask: 0xffffffff
    init: 0x0
```

生成端口：

```systemverilog
input  logic [31:0] csr_0_value_res0,
input  logic        csr_0_value_ready,
output logic [31:0] csr_0_set_0_sdata,
input  logic        csr_0_set_ready,
output logic        csr_0_set_enable,

input  logic [31:0] csr_status_value_res0,
input  logic        csr_status_value_ready,
output logic [31:0] csr_status_set_status_sdata,
input  logic        csr_status_set_ready,
output logic        csr_status_set_enable
```

软件调用：

```c
write_csr(0x810, cfg);
aps_custom0();
status = read_csr(0x811);
```

APS 调用边界：

```text
on rocc_cmd.fire:
  command.cfg = csr_0_value_res0

on done:
  csr_status_set_enable = 1
  csr_status_set_status_sdata = done_code
```
