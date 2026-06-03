# APS to CMT2 Scope FIFO Model

这份文档描述 `aps-to-cmt2` 在 block / slot / loop 三个层级上的 FIFO 组织方式。  
更完整、更新的当前实现/目标设计对照见 [aps-to-cmt2-transfer-model.md](aps-to-cmt2-transfer-model.md)。本文保留 scope ownership 的概要，不再作为逐行实现说明。
它把两种情况分开说明：

- **非 pipeline**
  - 接受 `block` 持有自己的 `entry / exit` 连接点这一模型
  - 目标是减少重复 FIFO 设计，避免把所有传递都堆到上一级结构里

- **pipeline**
  - 接受 `FIFO` 跟着 scope 走
  - 目标是支持 loop / nested loop 的可重叠迭代

本文不讨论 adaptor、memory pool、global register、CSR 等外设路径。

主要代码入口：

- [lib/APS/APSToCMT2/RuleGeneration.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/RuleGeneration.cpp)
- [lib/APS/APSToCMT2/BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp)
- [lib/APS/APSToCMT2/BBHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BBHandler.cpp)
- [lib/APS/APSToCMT2/LoopHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/LoopHandler.cpp)

## 1. 先定义术语

### 1.1 block

这里的 block 指 `BlockHandler` 划分出来的控制流 segment。  
它可能是：

- 普通顺序块
- loop header
- if/while 的一个分段

### 1.2 slot

slot 指 `BBHandler` 根据 `starttime` 划分的同一 block 内时间片。  
slot 只描述同一 block 内部的执行顺序。

### 1.3 loop scope

loop scope 指 `LoopHandler` 管理的完整 loop：

- `entry`
- `body`
- `next`
- body 内部的若干 block

所以 loop 不是一个单块，而是一个有内部 block 子图的 scope。

## 2. 设计目标

这套模型想同时满足两件事：

1. **非 pipeline 时，结构要简单**
   - block 自己持有入口和出口
   - 不把所有 fifo 堆在上层 struct 里
   - block 内尽量少建冗余 FIFO

2. **pipeline 时，结构要可扩展**
   - FIFO 必须跟着 scope 走
   - nested loop 的 inner / outer 传递必须可分层
   - 允许同一层 scope 内部有自己的传递资源

换句话说：

- 非 pipeline 讲的是“边界持有”
- pipeline 讲的是“scope 持有”

## 3. 非 pipeline 模型

这个模型适用于：

- 普通 block
- 不需要重叠迭代的 loop
- 还没有做复杂 pipeline 的 nested loop

### 3.1 核心原则

每个 block 自己持有：

- `entry token`
- `exit token`
- 必要的 `live_in`
- 必要的 `live_out`

当前代码里，non-pipeline 的数据承载方式已经主要收敛为：

- 跨 block value：producer block 下的 reg
- 跨 slot value：basic block 内 local reg
- token：block/slot 之间仍然用 1-bit FIFO 串接

父级结构只负责：

- 把上一个 block 的出口接到下一个 block 的入口
- 把需要传下来的值接到子 block 的 `live_in`
- 把子 block 的 `live_out` 接回父级出口

这意味着 non-pipeline 不应该再为每条跨 slot/跨 block 边重复创建 FIFO；如果需要被多个 consumer 读，优先由串行 token 语义保护同一个 producer-owned reg。

### 3.2 适合的实现语义

非 pipeline 时，一个 block 可以被看成：

```text
BlockContext {
  entry_token
  exit_token
  live_in_bundle
  live_out_bundle
}
```

block 内部的 slot 目前用本地值表和 local reg 处理跨 slot 值；不需要再把每个 value 送进 FIFO。

### 3.3 对当前代码的映射

当前代码里，非 pipeline 方向最接近的思想是：

- `BlockHandler` 负责 block 级边界
- `BBHandler` 负责 block 内执行
- `LoopHandler` 负责 loop 边界

当前代码已经部分按这个方向收敛：non-pipeline 跨 block value 由 `createCrossBlockValueRegs()` 创建 producer-owned reg；pipeline 才为跨 block value 创建 FIFO。

## 4. Pipeline 模型

这个模型适用于：

- 需要 loop pipeline
- 需要 nested loop pipeline
- 需要 scope 内可重叠的迭代调度

### 4.1 核心原则

FIFO 要跟着 scope 走。

也就是说：

- block 自己有 block-owned FIFO bundle
- loop 自己有 loop-owned FIFO bundle
- nested loop 的 inner / outer 各自持有各自的 bundle

上一级结构只负责连接，不负责解释内部语义。

### 4.2 为什么 pipeline 不能把 FIFO 全丢给上一级

如果上一级 struct 统一持有全部 FIFO，会出现两个问题：

- ownership 不清
- nested scope 的重入边界不清

特别是 loop pipeline：

- outer 不能在 inner 还占用 scope 时随便发起下一次迭代
- inner 的可重入性必须由 inner 自己声明
- 如果 pipeline parent 调用 non-pipeline child scope，child 必须提供容量 1 的 context token；普通 entry/exit token 只表达顺序，不足以防止重入覆盖 child 的 reg/state
- 当前代码还没有实现这个 context token，因此它是 required-but-missing 的安全 guard

所以 pipeline 场景下，**FIFO 必须随 block / stage / loop scope 走**。

### 4.3 pipeline 时的 scope-owned 结构

一个可 pipeline 的 scope 可以抽象成：

```text
ScopeContext {
  token_in
  token_out
  data_in
  data_out
  carried_state
  local_stage_state
}
```

这不是说所有字段都一定是 FIFO，而是说：

- 这个 scope 拥有这些传递资源
- 上一级只接入口和出口

### 4.4 block 内 slot 和 pipeline

slot 级 pipeline 如果保留，应该是 block-owned，而不是 parent-owned：

- slot 自己知道自己的输入 token / 输出 token
- slot 自己知道是否需要本地缓存或延迟传递
- block 只负责把相邻 slot 连接起来

当前实现里，pipeline basic block 的 token 已经按相邻 slot 传递，但 data FIFO 还是 producer slot 直连 consumer slot。这个形态功能上可表达 def-use，但如果 FIFO 深度多数为 1，远距离 consumer 会让 producer 很快遇到 full，从而严重阻塞前级。目标形态应改成 data 和 token 一样 stage-by-stage live-through。

## 5. loop 的两种情况

loop 是最容易混淆的部分，所以必须分两类看。

### 5.1 非 pipeline loop

这时我同意一个更简单的模型：

- loop 由 block 持有自己的 `entry / exit`
- loop body 里的 block 作为子 block 来处理
- loop 不额外开放复杂的跨迭代并发能力
- 当前实现使用 `loop_state_reg`、IV reg、input state regs 和 loop-to-body FIFO

这意味着：

- outer 看到 inner loop，可以把它当作一个原子段
- inner loop 跑完后，outer 才能继续

这类模式最适合先做正确性，再做性能。

### 5.2 pipeline loop

一旦 loop 要 pipeline，就必须把跨迭代依赖显式拆出来。

当前最小实现里，pipeline loop 自己持有：

- entry token
- body token
- issue token
- done token
- loop state reg
- done counter reg
- IV FIFO
- body 内 block 的内部传递资源

当前还不是完整 per-iteration frame/tag 模型。如果 loop 里还有 inner loop，inner loop 必须按自己的 pipeline attribute 独立决定：pipeline inner 只能保证单 invocation 内的 iteration overlap；如果会被 outer 多 activation 重入，还需要 per-invocation state 隔离。non-pipeline inner 必须用容量 1 context token 阻止重入。

### 5.3 nested loop 的关键规则

nested loop 是否合法 pipeline 由前置 pass 决定。到 `aps-to-cmt2` 时，不再做依赖消解分析，只忠实实现 attr 和 scope admission。

实现规则可以写成：

- outer 有 `pipeline=true`：outer 按 pipeline loop 生成 issue/retire。
- inner 也有 `pipeline=true`：inner 也按 pipeline loop 生成自己的 issue/retire，但当前不是完整 per-invocation frame/tag。
- inner 没有 `pipeline=true`：inner 是 non-pipeline child scope，必须容量为 1；当前 context token 尚未实现。

如果 inner 不可 pipeline，那么 outer 的第二次 activation 不能在第一次 activation 还占着 inner scope 时进入 inner。

也就是说：

- **inner 不可 pipeline 时，outer 必须把 inner 当原子块，并由 context token 阻止重入**
- **inner 可 pipeline 时，outer/inner 各自通过自己的 admission/completion 结构重叠；若 inner 会被 parent 多 activation 重入，还需要额外 per-invocation 隔离**

这是当前设计里最重要的约束之一。

## 6. 推荐的最终分工

### 6.1 非 pipeline

- block 负责自己的 `entry / exit`
- block 内尽量用 local map / local register
- 不把 value 传递拆成很多层 FIFO

### 6.2 pipeline

- FIFO 跟着 scope 走
- block / loop / nested loop 各自拥有自己的传递 bundle
- parent 只负责连接，不负责持有所有细节

### 6.3 loop pipeline 的最小可行形式

如果要先做一个可控版本，建议：

- 非 pipeline block：block-owned entry/exit
- pipeline loop：loop-owned scope FIFO bundle
- nested loop：inner / outer 各自独立持有 admission/completion 结构
- pipeline parent 中的 non-pipeline child：child 必须有容量 1 context token，当前尚未实现

这比把所有 FIFO 堆到 `BlockHandler` 或 `LoopInfo` 里更干净，也更容易推理。

## 7. 一个直观例子

### 7.1 非 pipeline 的 block 串接

```text
block A
  -> A.exit
  -> block B.entry
```

这里不需要一堆 per-value FIFO。
只要 A 的出口和 B 的入口连上就够了。

### 7.2 非 pipeline 的 loop

```text
outer block
  -> inner loop
  -> outer block next part
```

如果 inner 不 pipeline，outer 必须等 inner 完整退出后再继续。

### 7.3 pipeline 的 nested loop

```text
outer loop scope
  -> inner loop scope
```

如果 outer/inner 都是 pipeline，它们各自通过自己的 admission/completion 结构重叠；当前最小实现不包含完整 per-invocation frame/tag。
如果 inner 不是 pipeline，它必须作为容量 1 child scope，outer 的后续 activation 不能进入这个 inner scope。

## 8. 和当前实现的关系

当前实现已经体现出几个正确方向：

- `BlockHandler` 负责 block 级分段和边界
- `BBHandler` 负责 block 内 slot 化
- `LoopHandler` 负责 loop entry/body/next

当前实现里还没有完成的部分：

- pipeline parent 调用 non-pipeline child 时，还缺 child-owned context token
- pipeline BB data 还是 producer slot 直连 consumer slot FIFO，不是 stage-by-stage live-through

这篇文档建议的方向是：

- 非 pipeline 时，收敛成 block-owned entry/exit
- pipeline 时，改成 scope-owned FIFO bundle

这样既能支持正确性优先的简单模式，也能支持未来的 loop pipeline 和 nested loop pipeline。

## 9. 最终结论

可以把这套设计记成两句话：

- **非 pipeline：block 持有自己的入口和出口，上层只负责连接。**
- **pipeline：FIFO 必须跟着 scope 走，nested loop 的 inner / outer 各自拥有自己的传递资源。**

如果以后要重构 `aps-to-cmt2`，先判断目标是：

- 简化非 pipeline
- 还是支持 pipeline

然后再决定 FIFO 归属，不要把两种模型混在同一套结构里。
