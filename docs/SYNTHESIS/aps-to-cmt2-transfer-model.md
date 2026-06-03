# APS to CMT2 Transfer Model

这份文档只讨论 `aps-to-cmt2` 里 **block / slot / loop** 的值传递和 token 传递，不讨论 adaptor、memory pool、global register、CSR 等外设路径。

这是一个**当前实现 + 下一步设计约束**文档。
当前代码级行为请同时看 [aps-to-cmt2-fifo-flow.md](aps-to-cmt2-fifo-flow.md)。

它把问题明确拆成两种模式：

- **非 pipeline**
  - 采用 `block` 持有自己的 `entry / exit`
  - parent 只负责把边界接起来
  - 不把所有传递细节堆到上一级结构里

- **pipeline**
  - 采用 `FIFO` 跟着 scope 走
  - block / stage / loop 各自持有自己的传递资源
  - nested loop 的 inner / outer 都必须按各自 scope 来判断是否可重叠

这两种模式不要混写成一套结构的不同别名。  
它们是两套不同的组织原则。

主要代码入口：

- [lib/APS/APSToCMT2/RuleGeneration.cpp](../../lib/APS/APSToCMT2/RuleGeneration.cpp)
- [lib/APS/APSToCMT2/BlockHandler.cpp](../../lib/APS/APSToCMT2/BlockHandler.cpp)
- [lib/APS/APSToCMT2/BBHandler.cpp](../../lib/APS/APSToCMT2/BBHandler.cpp)
- [lib/APS/APSToCMT2/LoopHandler.cpp](../../lib/APS/APSToCMT2/LoopHandler.cpp)

## 1. 先约定术语

### 1.1 block

这里的 block 指 `BlockHandler` 划分出来的控制流 segment。它可能是：

- 普通顺序块
- loop header
- if / while 的一个分段

### 1.2 slot

slot 指 `BBHandler` 按 `starttime` 划分出来的同一 block 内时间片。
slot 只描述 block 内部的执行顺序，不是控制流分段。

### 1.3 loop scope

loop scope 指 `LoopHandler` 管理的完整 loop：

- `entry`
- `body`
- `next`
- body 内部的若干 block

所以 loop 不是单个 block，而是一个带内部 block 子图的 scope。

## 2. 先给结论

### 2.1 非 pipeline

如果当前目标不是做 pipeline，那么建议采用的模型是：

- block 持有自己的 `entry / exit`
- parent 只负责连接入口和出口
- block 内尽量用 local map / local register 处理值复用
- 不把 per-value FIFO 继续往上一级聚合

这是一个“边界持有”模型。

### 2.2 pipeline

如果当前目标是支持 pipeline，那么 FIFO 不能继续集中在上一级 struct 里，而必须：

- 跟着 block / stage / loop scope 走
- 让每个 scope 自己持有自己的输入、输出、跨阶段状态
- 让 parent 只负责连接，不负责解释内部传递语义

这是一个“scope 持有”模型。

### 2.3 Current vs Target

为了避免和当前代码混淆，这里把两种状态直接并列出来：

| 状态 | 组织方式 | 说明 |
|------|----------|------|
| current non-pipeline | block token 串接，跨 block/slot 的多数数据用 producer/block-owned reg 保存 | 这是当前已经实现的方向 |
| current pipeline | block token 串接，跨 block/slot 数据用 FIFO；BB 内 value FIFO 目前是 producer slot 直连 consumer slot | 可运行，但远距离 consumer 在浅 FIFO 下会阻塞前级 |
| target non-pipeline | block owns `entry / exit`，parent 只连边界；数据尽量用 reg/local map | 正确性优先、减少重复 FIFO |
| target pipeline | scope owns FIFO bundle；BB 内 token/data 都按 stage-by-stage 传递 | 支持 II=1 时的自然 backpressure |

### 2.4 block 继承时，pipeline 不是简单布尔值

这里要特别强调一点：**pipeline 不是“父 block 开了，子 block 就自动开”的布尔继承属性**。  
更准确地说，pipeline 是一个 scope 的 admission contract：

- 这个 scope 自己是否可重入
- 这个 scope 是否持有自己的 FIFO / state bundle
- 父级是否允许把下一次请求穿透到这个 scope

所以，block 继承时要按下面的方式理解：

| parent | child | 结论 |
|--------|-------|------|
| non-pipeline | non-pipeline | 子 block 继承的是边界连接语义，整体保持单飞 |
| non-pipeline | pipeline | 父级仍然不能穿透子 scope，子 pipeline 只能在自己的边界内生效 |
| pipeline | non-pipeline | 子 scope 必须作为容量 1 的原子段；需要 context token；当前代码尚未实现这个安全 guard |
| pipeline | pipeline | 父子都需要各自的 admission contract；若 child 会被 parent 多 activation 重入，还需要 per-invocation state 隔离 |

这意味着：

- parent 的 pipeline 语义不自动传染给 child
- child 的 pipeline 能力也不自动放大 parent
- 真正决定能不能重叠的是各自 scope 的 admission contract 是否兼容

一句话说完就是：

**block 继承的是“边界”和“连接方式”，不是“是否可以 pipeline”这个事实本身。**

### 2.5 pipeline 父级里的 block 级阻塞场景

这个场景必须单独写清楚，因为它直接决定是否会执行错误：

```text
pipeline outer loop
  body block A
  non-pipeline inner loop or non-pipeline child block B
  body block C
```

如果 `outer` 按 II=1 发起多个 activation，而 `B` 是 non-pipeline，那么 `B` 内部通常会有这些单 context 状态：

- loop state reg / induction var reg
- input state regs
- block-local value regs
- non-pipeline slot local regs

如果没有一个 **block/scope 级容量 token** 保护 `B`，第二个 activation 可能在第一个 activation 还没跑完 `B` 时进入 `B.entry`，从而覆盖这些 reg。这个错误不是 FIFO 空满能自动修掉的，因为 FIFO 只保护各自通路，不知道整个 non-pipeline child scope 是否仍被占用。

因此规则是：

- 每个 block 仍然需要普通 `entry_token` / `exit_token` 表示控制流顺序。
- 但当 pipeline parent 调用 non-pipeline child scope 时，还需要一个 child-owned 的 1-bit **context token**。
- child entry 必须同时拿到 parent entry token 和 context token，才能写自己的 state/reg 并开始执行。
- child 内部 continue/backedge 不释放 context token。
- child exit 发出 output token 后释放 context token。

这不是 debug/performance counter，也不是辅助调度状态；它是 non-pipeline child scope 的容量约束本身。
当前代码还没有实现 `context_token` 字段和 acquire/release rule，因此这是 required-but-missing 的正确性 guard。

## 3. 非 pipeline 模型

这个模型适用于：

- 普通 block 串接
- 不做重叠迭代的 loop
- 还没有引入复杂 nested pipeline 的情况

### 3.1 核心原则

每个 block 自己持有：

- `entry token`
- `exit token`
- 必要的 `live_in`
- 必要的 `live_out`

parent 只负责：

- 把上一个 block 的出口接到下一个 block 的入口
- 把需要传下来的值接到子 block 的 `live_in`
- 把子 block 的 `live_out` 接回父级出口

### 3.2 这个模式下不该做什么

非 pipeline 时，不应该继续做这些“层层加 FIFO”的组织：

- 把 block 间数据传递再拆成多级总账
- 把 slot 间的值传递继续向上汇总成 parent-owned FIFO 表
- 把 loop 内部的临时传递也统一塞进同一个父级结构

这类做法会让 ownership 变得不清楚，也会让值的来源/去向越来越难看懂。

### 3.3 非 pipeline 的 block 语义

可以把一个 block 看成：

```text
BlockContext {
  entry_token
  exit_token
  live_in_bundle
  live_out_bundle
}
```

block 内部的 slot 只需要本地值表或局部寄存器映射，不需要再把每个 value 物化成额外 FIFO。

## 4. Pipeline 模型

这个模型适用于：

- loop pipeline
- nested loop pipeline
- 需要 scope 内可重叠执行的情况

### 4.1 核心原则

FIFO 必须跟着 scope 走。

也就是说：

- block 自己有 block-owned FIFO bundle
- loop 自己有 loop-owned FIFO bundle
- nested loop 的 inner / outer 各自持有各自的 bundle

上一级结构只负责连接，不负责持有所有细节。

### 4.2 为什么 pipeline 不能把 FIFO 全扔给 parent

如果 parent 统一持有所有 FIFO，会有两个问题：

- ownership 不清
- nested scope 的重入边界不清

尤其是 loop pipeline：

- outer 不能在 inner 还占着 scope 时随便发起下一次迭代
- inner 的可重入性必须由 inner 自己声明

所以 pipeline 场景下，**FIFO 必须随 block / stage / loop scope 走**。

### 4.3 pipeline 的 scope-owned 结构

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

这不表示所有字段都一定是 FIFO；它表示这些传递资源都归该 scope 所有。

### 4.4 block 内 slot 和 pipeline

如果 slot 级 pipeline 保留，它也应该是 block-owned，而不是 parent-owned：

- slot 自己知道自己的输入 token / 输出 token
- slot 自己知道是否需要本地缓存或延迟传递
- block 只负责把相邻 slot 连接起来

## 5. Loop 的两种情况

loop 是最容易混淆的部分，所以必须分两类看。

### 5.1 非 pipeline loop

这时建议采用更简单的外部语义：

- loop 对外表现为一个原子段，父级只看到 `entry / exit`
- loop body 里的 block 作为子 block 来处理
- loop 不额外开放复杂的跨迭代并发能力
- loop 内部仍然使用 `entry / body / next` 三段式控制，只是它不向外暴露并发重叠能力

这意味着：

- outer 看到 inner loop，可以把它当作一个原子段
- inner loop 跑完后，outer 才能继续

这个模式最适合先做正确性，再谈性能。

### 5.2 pipeline loop

一旦 loop 要 pipeline，就必须把跨迭代依赖显式拆出来。

当前最小实现里，pipeline loop 需要自己持有：

- `entry token`
- `body admission queue`
- `body completion token`
- `loop-carried state`
- `issue token`
- `done counter`
- pipeline IV FIFO
- body 内 block 的内部传递资源

这个模型只表示**同一次 loop invocation 内的 iteration overlap**。它不是完整 per-parent-activation frame/tag 模型，因此不能自动支持同一个 loop scope 被 pipeline parent 的多个 activation 重入。

如果 loop 里还有 inner loop，那么 inner loop 必须有自己的 admission contract：

- inner non-pipeline：必须有容量 1 context token，否则不能被 pipeline parent 重入。
- inner pipeline：只有在它也有 per-invocation 隔离或被证明不会被 parent 重入时才安全；当前最小实现还没有完整 frame/tag 隔离。

### 5.3 nested loop 的关键规则

在这一层里，**不做依赖消解分析**。
但这不等于所有 nested pipeline 结构都已经安全实现。当前应该按下面的状态区分：

- `implemented`: 当前代码能生成的结构。
- `required but missing`: 正确执行需要，但代码尚未实现的安全 guard。
- `future`: 需要完整 frame/tag 或更强前置 pass 才能支持的结构。

因此，文档里的 nested loop 规则是实现约束，不是说当前所有 case 都已经安全。

### 5.5 nested loop admission matrix

更明确地说，nested loop 至少要分这四种情况：

| outer | inner | 结论 |
|-------|-------|------|
| non-pipeline | non-pipeline | 串行，inner 作为原子段 |
| non-pipeline | pipeline | outer 不重入 inner；inner 可以在单次 invocation 内做自己的 iteration overlap |
| pipeline | non-pipeline | required but missing: inner 必须有 context token；否则 outer 的多个 activation 会覆盖 inner state |
| pipeline | pipeline | future/受限: 需要 per-invocation frame/tag 或证明 inner 不会被 parent 重入；当前最小 issue/retire 模型不足以表达任意重入 |

### 5.6 一个更精确的 admission 规则

对 nested loop，外层是否能 pipeline 的合法性应由前置 pass 处理。对当前 APSToCMT2 实现层，安全落地至少需要满足：

1. outer 自己的 recurrence / resource 约束允许重叠
2. 若 outer body 里还有 non-pipeline child scope，则 child 有 context token
3. 若 outer body 里还有 pipeline child scope，则 child 有 per-invocation state 隔离，或前置 pass 证明不会被 parent 重入

当前代码尚未实现第 2 条的 context token，也没有第 3 条的完整 frame/tag。因此文档不能把 nested pipeline 描述成已经完全支持。

### 5.7 block 继承和 pipeline 的关系

把上面的规则落到 block 继承上，可以得到一个更实用的判断：

1. **子 block 是否 pipeline，要看子 block 自己的 scope 能力**
   - 不是看父 block 是否 pipeline
   - 也不是看父 block 的 FIFO 是否多

2. **父 block 是否能重叠子 block，要看子 block 是否提供 admission contract**
   - 如果没有明确的 `acquire / release`
   - 或没有 capacity / credit
   - 父 block 就必须把子 block 当成原子段

3. **继承只传边界，不传内部调度自由度**
   - 子 block 可以继承父 block 的数据边界关系
   - 但不能默认继承父 block 的重叠能力
   - 反过来，父 block 也不能因为子 block 是 pipeline 就自动获得穿透资格

### 5.8 loop 是 pipeline 时，下面的层级默认规则

如果一个 loop 已经明确是 pipeline，那么它下面的层级要按下面的默认规则处理：

1. **它下面的 basic block 需要是 pipeline**
   - 因为这些 basic block 位于同一个可重叠 loop scope 内
   - 它们必须能接受该 loop 的 admission / token / carried-state 语义
   - 否则 loop 内部的执行会在 block 边界上重新退化成串行

2. **它下面的 loop block 需要看情况**
   - 如果 nested loop 还存在，必须按 child scope 的 admission contract 接起来
   - inner non-pipeline 需要 context token；当前尚未实现
   - inner pipeline 若可能被 parent 多 activation 重入，需要 per-invocation 隔离；当前最小模型尚未实现

3. **默认值规则**
   - `pipeline loop -> basic block`: 默认 pipeline
   - `pipeline loop -> non-pipeline nested loop`: required-but-missing context token
   - `pipeline loop -> pipeline nested loop`: 只安全覆盖单 invocation overlap；重入需要 future frame/tag 或前置证明

4. **原因**
   - basic block 只是同一 scope 内的局部执行单元，跟着外层 loop 的 pipeline 语义走是合理的
   - nested loop 是独立 scope，不能因为父 loop 是 pipeline 就默认可重入

一句话总结就是：

**pipeline loop 在这一层只实现 attr 指定的结构；nested child 的重入安全必须由 context token、per-invocation state 或前置 pass 证明承担。**

## 6. 具体实现蓝图

这一节把前面的 proposal 落成可实现的规则。  
这里的基本假设是：

- 一个 scope 的值传递必须是“单次消费、局部缓存、显式边界”
- 任何需要被多个下游使用的 value，必须先落到本地 `value_map` 或 `reg`
- `FIFO` 只承担跨边界、跨 stage、跨 iteration 的传递
- 不允许多个 consumer 直接共享同一条数据 FIFO 的 dequeue 权

### 6.1 统一原语

下面这些原语会在不同 scope 里复用：

#### 6.1.0 层级映射

为了避免术语混淆，先固定下面的映射：

- `block`
  - 一个控制流 block，作为 `BlockHandler` 的分析单元

- `basic block`
  - 一个被 `BBHandler` 处理的 block 内执行单元
  - 在非 pipeline 模式下，它等同于一个顺序执行的局部块
  - 在 pipeline 模式下，它等同于一个由多个 stage 组成的局部块

- `stage`
  - 一个 `basic block` 内的 slot
  - `stage` 只用于 block 内的局部顺序，不是独立的控制流 block

- `loop block`
  - 一个由 `LoopHandler` 处理的 loop 控制节点
  - 它本身是一个 scope，内部包含 `entry / body / next`

- `loop scope`
  - 整个 loop 的语义边界
  - 包含 loop block、loop body 中的 block、以及 loop-carried state

- `token_fifo`
  - 只表示 admission / completion
  - 不携带计算值

- `stage_token_fifo`
  - 1-bit token，连接同一个 basic block 内相邻两个 stage
  - 前一个 stage 完成后，写入下一个 stage 的 admission token
  - 如果 basic block 只有一个 stage，则不需要内部 `stage_token_fifo`

- `bundle_fifo`
  - 携带一个 scope 的输入或输出 payload
  - payload 可以是一个 value bundle；当前实现多数地方没有做统一 bundle pack，而是 per-value reg/FIFO

- `local_reg`
  - scope 内部重复使用的临时值
  - 一次 dequeue 后多次读

- `deferred_fifo`
  - 这是目标 pipeline BB 模型里的概念
  - 当前实现不是逐 stage deferred/live-through，而是 producer slot 直连 consumer slot FIFO

- `context_token`
  - non-pipeline scope 的单 context 容量
  - 当该 scope 被 pipeline parent 调用时必须存在
  - 用于防止多个 activation 同时进入并覆盖该 scope 的 reg/state

- `credit / tag`
  - 这是更完整 pipeline loop frame 模型里的概念
  - 当前实现没有 `tag`，也没有完整 per-iteration frame

### 6.2 各类 scope 的资源表

| Scope | 当前实现的 FIFO | 当前实现的 reg | 下一步约束 |
|------|----------------|---------------|------------|
| non-pipeline basic block | `entry_token_fifo`，`exit_token_fifo`，slot token FIFO | 跨 slot local value reg，跨 block producer value reg | 若被 pipeline parent 调用，需要 child-owned `context_token` |
| pipeline basic block | `entry_token_fifo`，`exit_token_fifo`，slot token FIFO，producer->consumer slot value FIFO，跨 block value FIFO | 只读 parent/loop 输入 reg；不靠 local reg 承载 overlap 数据 | value FIFO 应改成 stage-by-stage live-through FIFO |
| non-pipeline loop | `entry_token_fifo`，`loop_body_admit_fifo`，`body_done_token_fifo`，loop-to-body FIFO | `loop_state_reg`，IV reg，input state regs | 若被 pipeline parent 调用，需要 child-owned `context_token` |
| pipeline loop | `entry_token_fifo`，`issue_token_fifo`，`loop_body_admit_fifo`，`body_done_token_fifo`，IV FIFO | `loop_state_reg`，`done_counter_reg`，input state regs | 当前是 issue/retire 计数模型，不是完整 per-iteration frame/tag 模型 |

### 6.2.1 FIFO / reg 的内容约定

下面把每一类 FIFO / reg 里应该装什么写清楚。  
这里不要求固定 bit 级布局，但要求语义字段必须齐全，且读写方向唯一。

#### A. `entry_token_fifo` / `exit_token_fifo`

- 内容：
  - 1-bit token
- 作用：
  - `entry_token_fifo` 表示这个 scope 允许开始
  - `exit_token_fifo` 表示这个 scope 已完成
- 写者：
  - 上游 scope 写 `entry_token_fifo`
  - 当前 scope 写 `exit_token_fifo`
- 读者：
  - 当前 scope 读 `entry_token_fifo`
  - 下游 scope 读 `exit_token_fifo`
- 清空方式：
  - 由消费方 `deq`

#### B. `live_in_bundle_fifo` / `live_out_bundle_fifo`

- 内容：
  - 当前 block / loop / scope 入口和出口所需的跨边界值集合
  - 只包含“跨本 scope 边界仍然活着”的 value
- 作用：
  - `live_in_bundle_fifo` 在 entry 时一次性提供当前 scope 的全部输入
  - `live_out_bundle_fifo` 在 exit 时一次性导出当前 scope 的全部输出
- 字段建议：
  - `value_id -> payload` 的映射
  - 对 loop scope 还要包含必要的 carried value / state value
- 写者：
  - `live_in_bundle_fifo` 由父 scope 或 join/broadcast 边界写入
  - `live_out_bundle_fifo` 由当前 scope 在 exit 时写入
- 读者：
  - 当前 scope 在 entry 时读取 `live_in_bundle_fifo`
  - 下游 scope 在 entry 时读取 `live_out_bundle_fifo`

#### C. `loop_input_bundle_fifo` / `loop_output_bundle_fifo`

- 内容：
  - loop 入口需要从外层拿进来的值集合
  - loop 退出后需要交给外层 successor 的值集合
- 作用：
  - 作为 loop scope 的边界 payload
  - 对 non-pipeline loop，入口和出口都只发生一次
  - 当前实现没有统一的 loop input/output bundle FIFO；入口值来自 `input_fifos` 或 parent reg，出口仍走外层 output token/value 通路
- 字段建议：
  - `loop input values`
  - `loop carried init values`
  - `loop output values`
  - `final carried values`（若 loop 退出时仍需返回）
- 写者：
  - `loop_input_bundle_fifo` 由父 scope 或 join/broadcast 边界写入
  - `loop_output_bundle_fifo` 由 loop 的 `next` 在退出时写入
- 读者：
  - loop 的 `entry` 读取 `loop_input_bundle_fifo`
  - 下游 scope 在 entry 时读取 `loop_output_bundle_fifo`

#### D. `stage_input_bundle_fifo` / `stage_output_bundle_fifo`

- 内容：
  - 这个 stage / slot 入口所需的值集合
  - 这个 stage / slot 出口需要传递给后续 stage 的值集合
- 作用：
  - 承担 block 内 stage 间的边界传递
- 字段建议：
  - `operand values`
  - `stage results`
  - 必要的延迟中间量
- 写者：
  - 前一 stage / block 写 `stage_input_bundle_fifo`
  - 当前 stage 写 `stage_output_bundle_fifo`
- 读者：
  - 当前 stage 读 `stage_input_bundle_fifo`
  - 后续 stage 读 `stage_output_bundle_fifo`
- 关系：
  - `stage_output_bundle_fifo` 的内容就是下一 stage 的 `stage_input_bundle_fifo`

#### E. `deferred_fifo`

- 内容：
  - 某个 producer value 的延迟版本
  - 只给后续唯一消费者用
- 作用：
  - 目标模型里，把“当前 stage 产出、但后续 stage 才用”的值逐 stage 携带
  - 当前实现里，对应的是 producer slot -> consumer slot 的直连 FIFO
- 写者：
  - 产生该 value 的 stage / block
- 读者：
  - 第一次需要该 value 的后续 stage / block
- 清空方式：
  - `deq` 后立刻转成本地 `value_map` / `reg`

#### F. `loop_state_reg`

- 内容：
  - loop 控制状态
  - 至少包括：
    - induction counter
    - bound
    - step
    - next-condition 相关控制位
  - 如果 loop 有 carried state，还必须包含对应 carried payload
- 作用：
  - 描述当前 iteration 的状态和下一次 iteration 的控制基础
- 写者：
  - non-pipeline loop 的 `entry` 初始化
  - non-pipeline loop 的 `next` 更新
  - pipeline loop 的 `entry` 初始化
  - pipeline loop 的 `issue` 更新
- 读者：
  - non-pipeline loop 的 `body` 和 `next`
  - pipeline loop 的 `issue` 和 `retire`
- 适用范围：
  - 当前 non-pipeline 和 pipeline loop 都使用
  - 这是当前最小 pipeline 实现的一部分，不是完整 per-iteration frame 模型

#### G. `loop_body_admit_fifo`

- 内容：
  - loop body 的 admission payload
  - 对 non-pipeline loop 来说，内容是 1-bit admission token
  - 对当前 pipeline loop 来说，内容仍是 1-bit admission token；IV 通过独立 IV FIFO 传入
- 作用：
  - 作为 loop body 的唯一 admission queue
- 写者：
  - `entry` 写入第一次 iteration 的 admission payload
  - `next` 在 continue 时写入下一次 iteration 的 admission payload
  - 当前 pipeline loop 由 `issue` 写入每个 iteration 的 admission token
- 读者：
  - `body`
- 约束：
  - 这是一个多 producer、单 consumer 的 admission queue
  - non-pipeline producer 是 `entry` 和 `next`
  - 当前 pipeline producer 是 `issue`
  - consumer 只有 `body`
  - FIFO 顺序就是 admission 仲裁顺序

#### H. `loop_frame_to_next_fifo`

- 内容：
  - 目标完整 frame 模型里的当前 iteration 更新后 frame
  - 当前代码没有独立 frame FIFO；`scopeResources.loopFrameToNextFIFO` 目前复用的是 `bodyDoneTokenFIFO`
- 作用：
  - 目标模型中 body 把更新后的 frame 交给 next/retire
  - 当前实现中 body 只通过 done token 通知 retire
- 写者：
  - `body`
- 读者：
  - `next` 或 `retire`

#### I. `context_token`

- 内容：
  - 1-bit token，初始为 available
- 作用：
  - non-pipeline scope 的单 context 容量保护
  - 只在该 scope 可能被 pipeline parent 重入时需要
- 写者：
  - child `exit` 释放 token
- 读者：
  - child `entry` 获取 token
- 约束：
  - `entry` 未获取 token 时不能 dequeue parent token，也不能写 state/reg
  - loop backedge / continue 不释放 token
  - token 释放点只能是 scope 完整 exit

#### J. `credit` / `tag`

- 内容：
  - `credit`: 可接纳的 activation 数量
  - `tag`: 用于区分不同 activation 的身份，作为 future `loop_frame` 的字段被传递
- 作用：
  - 完整 pipeline loop 的 admission 与 completion 区分
- 当前状态：
  - 代码当前没有实现完整 `tag/frame` 模型
  - 代码当前使用 `issue_token_fifo + loop_state_reg + done_counter_reg` 作为 pipeline loop 的最小 issue/retire 模型
- 写者：
  - future `credit` 由 admission / completion 控制逻辑更新
  - future `tag` 在 `entry` 生成，并在 `loop_frame` 中传递
- 读者：
  - future `entry`、`body`、`next`

#### K. `value_map regs` / `stage_local regs`

- 内容：
  - 当前 scope 内部可以重复读取的 SSA 值缓存
  - 本质是“已经 dequeue 进来、当前 scope 本地复用”的临时寄存器
- 作用：
  - 避免同一个 value 被多个 op 重复 dequeue
- 写者：
  - entry / stage first-use / FIFO dequeue / reg read
- 读者：
  - 当前 scope 内部的任意 op

### 6.3 非 pipeline basic block 的入/出

non-pipeline basic block 的执行模型建议固定成三步：

1. **入**
   - 同时等待 `entry_token`、必要 `live_in_bundle` 和 `context_token` 可用
   - 如果该 block 被 pipeline parent 调用，必须先 acquire `context_token`
   - 只有 acquire 成功后，才能 dequeue `entry_token` / `live_in_bundle`
   - 把 bundle unpack 到本 block 的 `local_reg` / `value_map`

2. **执行**
   - 按调度顺序生成每个 op
   - 每个 op 的 operand 读取顺序见 6.5
   - 每个 op 的结果先进入 `value_map`
   - 只要结果还在本 block 内被使用，就不要再走 FIFO

3. **出**
   - 收集所有需要跨边界输出的值，pack 成 `live_out_bundle`
   - 一次性 enqueue `live_out_bundle`
   - 一次性发出 `exit_token`
   - 如果获取过 `context_token`，在 exit 后释放

#### 6.3.1 非 pipeline block 的 fan-in / fan-out

- 多 predecessor 到一个 block 的情况，应该由父级显式生成一个 join / mux 边界
- join 只允许选中一个来源进入该 block
- 多 successor 的情况，应该由父级显式生成 branch / broadcast 边界
- 任何 consumer 都不能直接共享同一条数据 FIFO 的 dequeue 权

### 6.4 pipeline basic block 的入/出

pipeline basic block 可以理解成若干个 stage 的串联。  
如果现在的 schedule 有 slot，那么每个 slot 就是一个 stage。

1. **入到第一个 stage**
   - 消费 block 级 `entry_token`
   - 消费 `stage0_input_bundle`
   - 把输入 unpack 到 `stage0` 的 `local_reg`

2. **stage 间传递，目标模型**
   - token 必须 `stage_i -> stage_{i+1}` 一格一格传递
   - live data 也应该随 stage 一格一格向后 shift
   - 每个 stage 计算自己的 `live_out` 集合：所有未来 stage 仍会使用的值都必须继续携带
   - 当前 stage 产出的值若属于未来 live set，就进入下一 stage 的 live-through FIFO
   - 下一 stage 若消费该值但后续还需要，也必须继续转发到再下一 stage
   - 后续 stage 从 live-through FIFO 取值后，立刻转成本地 `value_map`

3. **出到最后一个 stage**
   - 最后一个 stage 收集需要跨 block 输出的值
   - pack 成 `stage_output_bundle`
   - 发出 `exit_token`

#### 6.4.1 pipeline basic block 的规则

- 任何值都只能“先被一个 stage 取出一次，再在本地复用”
- 不允许两个 stage 同时直接 dequeue 同一条数据 FIFO
- 如果一个值既要被后续 stage 使用，又要被当前 stage 使用多次，必须先缓存到本地 `value_map`，并在 stage 边界显式转发
- 同一个 value 被多个后续 stage 使用时，不能用唯一 consumer `deferred_fifo` 表达；必须作为 live-through bundle 字段继续携带，或在明确 fanout 点复制到多条独立 FIFO
- 如果继续使用当前实现里的 producer slot 直连 consumer slot FIFO，那么 FIFO depth 至少要覆盖 producer/consumer 的 slot 距离；深度 1 会导致远距离 consumer 严重阻塞前级
- 因此下一步 pipeline BB 应改成 stage-by-stage live-through，而不是远距离直连

### 6.5 loop block 的入/出

loop block 分两种：非 pipeline 和 pipeline。  
它们的区别在于能不能有多个 iteration 在飞。

#### 6.5.1 非 pipeline loop

非 pipeline loop 的建议实现是：

1. **entry**
   - 同时等待外层 `entry_token`、loop 输入 bundle 和可选 `context_token`
   - 如果该 loop 被 pipeline parent 调用，必须先 acquire `context_token`
   - 只有 acquire 成功后，才能消费外层 `entry_token` 和 loop 输入 bundle
   - 初始化 `loop_state_reg`
   - 初始化 `loop_carried regs`
   - 写入 `loop_body_admit_fifo`

2. **body**
   - body 内部再交给普通 block 处理
   - body 的 block 默认按 non-pipeline 规则执行
   - 所有 loop-carried 值都从 `loop_state_reg` / `loop_carried regs` 读
   - body 完成后写出 `body_done_token_fifo`

3. **next**
   - 消费 `body_done_token_fifo`
   - 读出 `loop_state_reg`
   - 判断是否继续
   - 如果继续，更新 `loop_state_reg` 并重新写入 `loop_body_admit_fifo`
   - 如果退出，pack `loop_output_bundle` 并发出 `exit_token`
   - 退出时释放 `context_token`

#### 6.5.2 pipeline loop

pipeline loop 的建议实现是：

1. **entry，当前实现**
   - 消费外层 `entry_token`
   - 从父级输入 FIFO/reg 捕获 live-in
   - 初始化 `loop_state_reg`
   - 初始化 `done_counter_reg`
   - 写入 `issue_token_fifo`

2. **issue，当前实现**
   - 消费 `issue_token_fifo`
   - 读取 `loop_state_reg`
   - 如果当前 counter 仍在范围内，则把 IV 写入 `inductionVarFIFO`
   - 写入 `loop_body_admit_fifo`，允许 body 接收一个 iteration
   - 更新 `loop_state_reg`
   - 如果还有后续 iteration，则重新写入 `issue_token_fifo`

3. **body，当前实现**
   - body 内 `BlockHandler` 被设置为 pipeline mode
   - basic block 按 pipeline BB 规则生成 slot rules
   - body 完成后写出 `body_done_token_fifo`

4. **retire，当前实现**
   - 消费 `body_done_token_fifo`
   - 读取 `done_counter_reg` 和 `loop_state_reg`
   - 当前代码没有显式 issued-count/frame 集合；retire 隐含所有 iteration 都会按 counter 发行
   - done counter 达到 loop bound 推导出的最后 completion 时，向外层 `exit_token` 发 completion
   - 否则只更新 done counter

#### 6.5.3 pipeline loop 的关键约束

- 当前最小模型假设前置 pass 已经保证 pipeline loop 合法
- IV 必须用 FIFO 进入 body，不能用单个 IV reg，否则多个 iteration 会互相覆盖
- body 内普通 BB 必须按 pipeline mode 处理
- 如果 body 内仍存在 non-pipeline child scope，该 child 必须有自己的 `context_token`，否则会被多个 iteration 重入
- 当前代码没有完整 per-iteration frame/tag；因此不要在文档里假设 pipeline loop 已经能承载任意 loop-carried payload 的多 activation 区分
- loop 是否 exit 由 retire 路径根据 issue/done 状态判断；body 只负责完成一个已 admit 的 iteration
- 这个 exit 判断是单 invocation 的最小模型，不是完整 issued-frame retire

### 6.6 operand 读取顺序

对任意一个 op，`getValue` 的建议顺序是：

1. `localMap`
2. `constant`
3. `scope input bundle` 已经 unpack 到的 `local_reg`
4. `loop carried regs` 或 `input_state_registers`
5. pipeline 模式下从 slot/value FIFO dequeue 后进入当前 rule 的 `localMap`

如果某个 operand 不在这几类里，说明它没有被正确放到当前 scope 的入口、state reg 或前序 stage 中。

### 6.7 producer 写回规则

每个 op 产生结果之后，应该按下面的规则分发：

1. **只在本 stage / 本 block 内使用**
   - 写入 `value_map`
   - 不进 FIFO

2. **后续 stage / 后续 block 才使用**
   - 当前 non-pipeline 实现：写入 producer/block-owned reg
   - 当前 pipeline 实现：写入 producer->consumer FIFO
   - 目标 pipeline 实现：写入下一 stage 的 live-through FIFO
   - 只允许唯一 consumer dequeue

3. **外层 scope 才使用**
   - 当前实现写到跨 block reg/FIFO 或 loop output token/value 通路
   - future frame 模型可 pack 到 `live_out_bundle` 或 `loop_frame`
   - 在当前 scope 结束时统一出边界

4. **多个下游都要用**
   - 先由父级边界显式 fan-out
   - 或显式复制到多个独立 FIFO
   - 不能让多个 consumer 共用一条 FIFO 的 dequeue 权

### 6.8 这套实现方式的目标

这套实现方式的目标不是复刻当前代码的 FIFO 数量，而是把语义收敛成：

- 入边界时一次性拿到当前 scope 需要的全部输入
- scope 内部用 local map / reg 复用值
- 跨 stage / 跨 block / 跨 iteration 的值，才进入 FIFO、reg 或 future frame
- 所有 consumer 都是显式的、单次消费的

### 6.9 loop 特化策略

默认情况下，loop 仍然按完整框架实现，也就是保留：

- `entry_token`
- `body_token`
- `next_token`
- `loop_state`
- `body_done_token_fifo`
- pipeline loop 的 `issue_token_fifo` / `done_counter_reg` 完成路径

如果要把 loop 特化成更轻的结构，必须先满足**合法性条件**，再看**收益性条件**。

#### 6.9.1 合法性条件

loop 只有在下面这些约束都成立时，前置 pass 才会把它交给这一层实现：

1. **没有跨迭代依赖**
   - 没有 loop-carried value
   - 没有必须保留到下一次 iteration 的 state
   - 没有需要由下一次 iteration 读取的控制状态

2. **没有跨迭代控制依赖**
   - induction / trip counter / exit predicate 都是常量、外部不变值，或者能放进等价的局部控制寄存器
   - `next` 的条件不依赖前一次 iteration 留下的状态

3. **没有跨迭代副作用或资源依赖**
   - 不依赖 memory / FIFO / method / extern resource / ordered I/O 的跨迭代顺序
   - 或者这些 effect 被证明彼此独立、可交换，且不会引入冲突

4. **没有必须显式保留的 completion / admission 边界**
   - 不需要独立观察 `body_done`
   - 不需要把 loop 当作独立的调度域
   - 不需要通过 `body_done_token_fifo` / pipeline retire 路径保护 live-out、资源释放、callee completion 或 next-condition 输入

5. **没有需要保持的 nested scope admission**
   - body 内没有必须独立保留的 inner loop admission
   - 或者 inner scope 可以被当作容量 1 的原子段，并且不会被外层重入

6. **token 保守性可满足**
   - 每次迭代仍然必须精确消费一次输入 token / payload
   - 并精确产生一次 backedge 或 exit
   - 不允许多个 consumer 共享同一条数据 FIFO 的 dequeue 权

#### 6.9.2 收益性条件

只有在合法性已经成立后，才考虑这些收益性因素：

- `body` 很轻
- 只有很少的 op
- 去掉完整 loop 框架后能明显减少边界成本

#### 6.9.3 可特化时的形态

满足合法性和收益性条件时，前置 pass 可以把 loop 降级成更轻的结构，例如：

- 普通 block 串接
- 直接的 branch / join 控制
- 去掉独立的 `loop_state_reg` / issue-retire 路径

#### 6.9.4 核心原则

**默认保留完整 loop 框架；只有当前置 pass 判断合法且有收益时，才会把更轻的结构交给这一层实现。**

这样做的好处是：

- 复杂 loop 仍然安全
- 小 loop 不会被固定的边界开销拖慢
- 前置 pass 已经决定合法性和特化形态，这一层只负责忠实实现

## 7. 当前代码和这套模型的对应关系

### 7.1 `BlockHandler`

`BlockHandler` 负责：

- 把 `tor.func` 切成 block
- 识别 block 间 producer / consumer
- 建立 block 间 token/data 连接

当前实现里：

- non-pipeline 跨 block value 使用 producer-owned reg，一个 value 在同一 producer scope 内被多个 consumer 共享读取
- pipeline 跨 block value 使用 FIFO
- pipeline parent 对输入 value 的多 sub-block fanout 会创建额外 fanout FIFO

后续方向是：**non-pipeline 继续收敛到 reg/local map；pipeline 的 FIFO ownership 跟着 block/stage/scope 走**。

### 7.2 `BBHandler`

`BBHandler` 负责：

- 把 block 再按 `starttime` 切成 slot
- 建立 slot 间 token/data 连接
- 生成每个 slot 的 rule

当前实现里：

- non-pipeline BB 的跨 slot value 用 local reg
- pipeline BB 的跨 slot value 用 producer slot 直连 consumer slot FIFO

后续 pipeline BB 应改为：

- token stage-by-stage
- data stage-by-stage
- 每个 stage 只向下一 stage 传 live-through 集合

这样大多数 stage FIFO 可以保持深度 1；远距离直连 FIFO 则必须增加深度，否则会降低 II。

### 7.3 `LoopHandler`

`LoopHandler` 负责：

- 把 loop 规范化成 `entry / body / next`
- 管理 loop-carried state
- 再把 body 交回 `BlockHandler + BBHandler`

这意味着 loop 本身就是一个 scope，而不是一个单独的 block。  
所以 loop 的传递资源应该按“loop-owned”思路看待，而不是并入更上层的统一结构。

## 8. 一个直观的区分

### 8.1 非 pipeline

```text
block A
  -> A.exit
  -> block B.entry
```

这里不需要一堆 per-value FIFO。
只要 A 的出口和 B 的入口接上就够了。

### 8.2 非 pipeline 的 loop

```text
outer block
  -> inner loop
  -> outer block next part
```

如果 inner 不 pipeline，outer 必须等 inner 完整退出后再让下一个 activation 进入 inner。outer 在 inner 之前的 stage 可以继续被 FIFO/backpressure 自然调度，但 inner scope 本身必须容量为 1。

### 8.3 pipeline 的 nested loop

```text
outer loop scope
  -> inner loop scope
```

当前层不做依赖消解分析；如果 inner loop 仍存在且没有 `pipeline`，它就按 non-pipeline child scope 处理，需要 context token 阻止重入。如果 outer 和 inner 都是 pipeline，当前最小实现只保证各自单 invocation 内的 iteration overlap；若同一个 inner scope 会被 outer 多 activation 重入，还需要 per-invocation frame/tag 或前置 pass 证明不会重入。

## 9. 建议的文档阅读顺序

如果你是在做实现判断，建议按这个顺序读：

1. 先看本文件第 2 节，决定当前目标到底是 non-pipeline 还是 pipeline
2. 如果是 non-pipeline，重点看第 3 节和第 5.1 节
3. 如果是 pipeline，重点看第 4 节和第 5.2 / 5.3 / 5.4 节
4. 再回到 `BlockHandler` / `BBHandler` / `LoopHandler` 看当前代码怎么把这些边界落下去

## 10. 最终结论

可以把这套设计记成两句话：

- **非 pipeline：block 持有自己的入口和出口，上层只负责连接。**
- **pipeline：FIFO 必须跟着 scope 走，nested loop 的 inner / outer 各自拥有自己的传递资源。**

如果以后要重构 `aps-to-cmt2`，先判断目标是：

- 简化 non-pipeline
- 还是支持 pipeline

然后再决定 FIFO 归属，不要把两种模型混在同一套结构里。
