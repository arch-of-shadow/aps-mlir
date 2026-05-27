# APS CSR Integration

本文记录 APS 接入 Rocket RoCC custom CSR 的接口和时序约定。这里的 CSR 指 Rocket `CustomCSR`/`CustomCSRIO` 路径，不是 TileLink MMIO register map。

## 背景

Rocket 已经支持给 RoCC accelerator 声明一组 custom CSR：

```scala
Seq(
  CustomCSR(0x810, mask, Some(init)),
  CustomCSR(0x811, mask, Some(init))
)
```

`CSRFile` 会为每个 `CustomCSR` 生成真实寄存器，并把访问事件和值通过 `CustomCSRIO` 接给对应的 RoCC module。APS chipyard wrapper 里已经有这条 processor-side 通路：

- `APSBlackbox(opcodes, csrList, config)` 将 `csrList` 传给 `LazyRoCC`。
- `LazyRoCCModuleImp.io.csrs` 连接到 `ApsRoccWrapper.io.rocc.csrs`。
- `WithAPSAccel` 从 APS config 生成 `CustomCSR` 列表。
- 当前 generated `main` blackbox 没有 CSR ports，CSR 信号到 `ApsRoccWrapper` 后还没有继续接入 APS top。

## Interface Shape

到 APS 侧时，CSR 地址已经在 Rocket `CSRFile` 中完成了解码。APS 不会看到 CSR address port，而是看到一组按声明顺序展开的 CSR lanes：

```scala
io.rocc.csrs(0)  // first CustomCSR in csrList
io.rocc.csrs(1)  // second CustomCSR in csrList
```

因此 APS top 的 blackbox 接口应该铺开，而不是设计成带地址的 CSR bus。建议生成命名端口，而不是只用裸 index。完整端口命名规范见 [aps-port-convention.md](aps-port-convention.md)。

```text
csr_<name>_value
csr_<name>_set
csr_<name>_sdata
```

Rocket `CustomCSRIO` 中还有 `ren/wen/wdata/stall` 等 processor-side 访问信号，但当前 APS generated `main` 不直接暴露这些信号。wrapper 内部只把 committed `value` 传给 main，并把 main 的 `set/sdata` 接回 Rocket CSRFile。

CSR 端口固定使用 `csr_<name>_<signal>`。例如 YAML 中 `name: "0"` 生成 `csr_0_*`，`name: status` 生成 `csr_status_*`。

长期应由同一个 CSR config 同时生成：

- Scala `CustomCSR(id, mask, init)`
- APS blackbox port name
- 软件侧 CSR define/header

建议配置结构：

```scala
case class APSCSRConfig(
  id: Int,
  name: String,
  mask: BigInt,
  init: BigInt = 0
)
```

## Signal Semantics

单个 CSR lane 的信号语义如下：

| Signal | Direction to APS top | Meaning |
|--------|----------------------|---------|
| `value` | input | `CSRFile` 中该 CSR 当前提交值 |
| `set` | output | APS 主动更新 `CSRFile` 中该 CSR，单周期 pulse |
| `sdata` | output | APS 写回数据，`set=1` 时有效 |

wrapper 内部固定：

```text
stall = 0
```

generated `main` 如果不需要写回 CSR，应默认：

```text
set   = 0
sdata = 0
```

这样 CPU 可以通过 `csrw` 写配置，APS 通过 `value` 读取配置。

## Timing Contract

`CustomCSRIO` 不是 ready/valid transaction bus，而是 `CSRFile` 暴露出的同步寄存器访问信号。APS 侧不应再把它包装成 `enable/ready` 方法。

CPU 写 CSR 的 processor-side 基本时序：

```text
cycle N:
  csr_wen   = 1
  csr_wdata = new_value

cycle N+1:
  csr_value = new_value
```

`wen` 是事件，`value` 是状态。当前 main 接口只使用状态：

- 配置型 CSR：读 `value`。
- 状态型 CSR：APS 用 `set/sdata` 回写 `CSRFile`，CPU 后续 `csrr` 读到最近提交状态。

如果后续确实需要 doorbell/update 事件，再单独扩展接口；不要默认把 `wen/wdata` 暴露给 generated top。

## Interaction With RoCC Command

配置型 CSR 不应在长事务执行过程中一直读取裸 `csr_value`。推荐规则是：

1. CPU 执行 `csrw csr_cfg, value`。
2. CPU 发 RoCC custom instruction。
3. APS 在 `rocc_cmd.fire` 那拍 latch 当前 CSR config。
4. 后续计算规则只使用 latch 后的 command context。

这样 CPU 在 accelerator 执行期间再次写 CSR，不会影响已经启动的 transaction。

Rocket core 对 RoCC CSR write 已经有 busy/fence 相关处理：写 RoCC CSR 时会考虑 RoCC busy 状态。因此软件顺序：

```asm
csrw 0x810, a0
custom0 ...
```

应保证后续 RoCC command 看到已经提交的 CSR 值。APS 侧的关键是只在 command 接收边界采样配置。

## Read Stall Policy

`stall` 可以让 `CSRFile` 暂停 CSR read/write，但当前 APS wrapper 不把 stall 交给 generated `main` 控制。

第一版策略：

- wrapper 内部 `stall` 恒为 `0`。
- CPU 读状态时读取最近提交的 `value`。
- 如果需要 status/done/error/counter，由 APS 周期性或事件式通过 `set/sdata` 更新。

## Wrapper Wiring

Scala wrapper 侧应逐 lane 连接：

```scala
main.io.csrValue(name) := fitWidth(io.rocc.csrs(i).value, csrWidth)

io.rocc.csrs(i).stall := false.B
io.rocc.csrs(i).set   := main.io.csrSet(name)
io.rocc.csrs(i).sdata := fitWidth(main.io.csrSdata(name), xLen)
```

`value/sdata` 最好在 wrapper 和 generated top 中统一为 `xLen`。如果 APS 内部只使用 32 bit，可以在 APS 内部截取低 32 bit；不要让 Rocket CSR 声明和 APS top 宽度产生隐式不一致。

## CMT2/Generated Top Plan

APS-to-CMT2 生成侧需要增加 CSR top ports。第一版建议最小化：

- 生成 CSR input ports：`value`。
- 生成 CSR output ports：`set/sdata`。
- 默认 `set/sdata` 为常量 0。
- 在 `rocc_cmd` 接收逻辑中，将需要的 `csr_<name>_value` 和 command fields 一起写入 command context。

这使 CSR 成为 APS top 的配置/status 边带信号，而不是调度内的普通 ready/valid interface。

## Implementation Checklist

1. 在 chipyard APS config 中加入 CSR config 列表。
2. `WithAPSAccel` 用 CSR config 生成 `CustomCSR`。
3. `MainBlackBox` 增加按 CSR name 展开的 ports。
4. `ApsRoccWrapper` 逐 lane 连接 `CustomCSRIO` 和 blackbox ports。
5. APS/CMT2 top 生成相同 CSR ports。
6. wrapper 默认 `stall=0`，generated top 默认 `set=0,sdata=0`。
7. 在 `rocc_cmd` 接收边界 latch CSR config。
8. 增加一个软件 smoke test：`csrw 0x810, value; custom0 ...`，由 accelerator 返回或写出观测到的 config。
