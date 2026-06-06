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

## `tor.if` Scope Lowering Model

`scf.if` is expected to be converted to `tor.if` before APSToCMT2.  The
APSToCMT2 lowering therefore targets `tor.if` directly.  `tor.if` preserves the
important SCF semantics: a single `i1` condition operand, one single-block then
region, an optional else region, variadic results, and `tor.yield` terminators
for branch result values.

### Non-pipeline `tor.if`

For the non-pipeline model, the parent block treats a `tor.if` segment as an
atomic child scope, analogous to how a non-pipeline `tor.for` is handled.  The
if scope owns its internal branch admission and join tokens, while the parent
only observes the scope entry token and scope exit token.

Required token resources:

- `ifEntryTokenFIFO`: inherited from the parent block boundary.
- `thenEntryTokenFIFO`: admits the then region.
- `elseEntryTokenFIFO`: admits the else region when the else region exists.
- `ifJoinTokenFIFO`: receives completion from the dynamically selected branch.
- `ifExitTokenFIFO`: inherited from the parent block boundary.

The generated rules are:

1. `if_dispatch_rule`

   This rule consumes `ifEntryTokenFIFO`, reads or captures the `tor.if`
   condition, and conditionally enqueues exactly one token:

   - condition true: enqueue `thenEntryTokenFIFO`.
   - condition false with an else region: enqueue `elseEntryTokenFIFO`.
   - condition false without an else region: enqueue `ifJoinTokenFIFO`
     directly, because the false path has no body.

2. Then-region block hierarchy

   The then region is processed by a nested `BlockHandler`.  Its parent entry
   token is `thenEntryTokenFIFO`, and its parent exit token is
   `ifJoinTokenFIFO`.  This preserves the existing block-level entry-token
   gating for the first sub-block in the branch.

3. Else-region block hierarchy

   If the `tor.if` has a non-empty else region, the else region is processed by
   another nested `BlockHandler`.  Its parent entry token is
   `elseEntryTokenFIFO`, and its parent exit token is `ifJoinTokenFIFO`.

4. `if_join_rule`

   This rule consumes one token from `ifJoinTokenFIFO` and enqueues one token to
   `ifExitTokenFIFO`.  Because dispatch admits exactly one branch per if-entry,
   a single shared join FIFO is sufficient for the non-pipeline model.

This gives the parent scope the same contract as a regular block or a
non-pipeline loop: one input token enters the scope, and one output token leaves
the scope after the selected branch has completed.

### Condition value handling

The condition is consumed by `if_dispatch_rule`, not by the branch body.  The
condition source follows the existing block live-in rules:

- If the condition is produced by an earlier sub-block in the same parent scope,
  it should already be available through the parent block's non-pipeline
  cross-block value register.
- If the condition is an inherited parent live-in register, the dispatch rule
  reads that register.
- If the condition is an inherited parent live-in FIFO, the dispatch rule
  dequeues it once at if entry.

If a branch body also uses the same condition value, the dispatch rule must
capture the dequeued condition into an if-scope register and pass that register
to the nested branch `BlockHandler`s as an inherited input register.  This avoids
multiple consumers racing on the same parent FIFO.

### Parent live-ins used by branches

Parent live-ins used only inside branches should be forwarded to both nested
branch handlers.  This is safe in the non-pipeline model because only one branch
is admitted for each if-entry token, so only the selected branch can dequeue its
live-in FIFO values.

If a parent live-in is consumed by both the dispatch rule and a branch body, it
must be captured into an if-scope register before branch admission, following
the same rule as the condition value above.

### `tor.if` results and `tor.yield`

`tor.if` can produce SSA results, matching `scf.if` result semantics.  The full
lowering requires an explicit merge point:

1. Create one if-result register per `tor.if` result.
2. In each branch's final block, before enqueueing `ifJoinTokenFIFO`, write the
   values from that branch's `tor.yield` operands into the corresponding
   if-result registers.
3. In `if_join_rule`, after consuming `ifJoinTokenFIFO`, publish the if-result
   registers to the parent scope's normal live-out mechanism and then enqueue
   `ifExitTokenFIFO`.

When `tor.if` has results, the else region must exist and must yield the same
number and types of values as the then region.  This mirrors the SCF contract.

A minimal first implementation may reject `tor.if` with results and support only
side-effecting/no-result `tor.if`.  That rejection must be explicit via
`emitError`; it must not fall back to regular basic-block lowering.

### Pipeline interaction

Pipeline `tor.if` is not required for the first implementation.  If the parent
scope is in pipeline mode, the lowering should reject `tor.if` explicitly unless
a re-entry-safe context-token protocol is implemented.

The future pipeline-safe model needs a capacity-one context token or equivalent
credit so that a non-pipeline if scope cannot be re-entered before its selected
branch reaches `if_join_rule`.  Without that gate, multiple dynamic instances of
the same if scope could alias condition/result registers and violate the
single-context non-pipeline assumption.

### Required APSToCMT2 integration points

The block segmenter should keep `tor.if` as a single control-flow segment and
mark it as a conditional block.  `BlockHandler::processBlock` must dispatch that
segment to an if-specific handler rather than `BBHandler`.

The if-specific handler should:

- Locate the single `tor::IfOp` in the segment.
- Reject unsupported pipeline mode explicitly.
- Reject `tor.if` results until result-register merge is implemented.
- Create branch-entry and join token FIFOs.
- Generate the dispatch and join rules.
- Process then/else regions with nested `BlockHandler`s using branch entry
  tokens and the shared join token.
- Preserve the parent block boundary contract by consuming the inherited entry
  token exactly once and producing the inherited exit token exactly once.

### Branch context lifetime

The condition value and the selected branch identity must remain valid from
`if_dispatch_rule` until `if_join_rule`.  The join rule must not observe a
condition/result context that was overwritten by a later dynamic execution of
the same `tor.if` scope.

For the non-pipeline first implementation, this is enforced by the scope token
contract:

- `if_dispatch_rule` consumes one parent entry token and admits exactly one
  branch.
- No second parent entry token may enter the same non-pipeline if scope before
  the selected branch has reached `if_join_rule` and the join rule has emitted
  the parent exit token.
- The if scope may therefore use single-entry condition/result context
  registers, because there is at most one live dynamic instance of the scope.

If the condition is needed after dispatch, for example to select branch-specific
publish logic or to debug/check the branch path at join, dispatch must write it
into an if-scope condition register before admitting the branch.  The join rule
then reads that register.  The register is single-context only and is correct
only under the non-reentry guarantee above.

A stronger and more explicit representation is to make the branch completion
FIFO carry a branch tag instead of a bare one-bit done token:

- `then` completion enqueues tag `1` to `ifJoinTokenFIFO`.
- `else` completion enqueues tag `0` to `ifJoinTokenFIFO`.
- The no-else false path enqueues tag `0` directly from dispatch to join.

With this form, `ifJoinTokenFIFO` is a one-bit payload FIFO whose value is the
selected branch tag.  The join rule dequeues the tag and can use it to guard any
branch-dependent result publishing or diagnostics.  For simple result-register
merge, the tag is not needed to choose the value because the selected branch has
already written the merge registers, but carrying the tag makes the dynamic path
explicit and avoids relying on an implicit side condition.

Pipeline support cannot reuse these single-context registers.  It must either
carry the branch tag and result payload through FIFOs, or allocate per-context
storage indexed by a tag/credit protocol.  Otherwise a later if instance may
overwrite the condition/result context before an earlier instance reaches join.

### Single-block branch finalization

A branch region with only one generated block does not need a separate final
rule.  That single block is both the branch entry block and the branch final
block:

- It consumes `thenEntryTokenFIFO` or `elseEntryTokenFIFO` at the beginning.
- It emits all operations in the branch body.
- If the `tor.if` has results, it writes the branch `tor.yield` operands to the
  if-result merge registers before completing.
- It enqueues the branch tag to `ifJoinTokenFIFO` as its branch completion.

For multi-block branch regions, the same finalization logic belongs only to the
last generated sub-block.  Intermediate branch blocks use ordinary block-to-block
tokens within the nested branch `BlockHandler`.

## Pipeline Parent Live-through Rule

When a parent scope is pipelined, values must move through the same ordered
block/scope path as the control tokens.  A value must not bypass an intermediate
block or child scope just because that intermediate scope does not use the value.

For example, in a pipelined outer loop body:

```text
A_b0: xxx_before produces %v
A_b1: non-pipeline child scope, e.g. tor.for or tor.if
A_b2: xxx_after consumes %v
```

The correct pipeline data path is block-by-block:

```text
control token: A_b0 -> A_b1 -> A_b2
data %v:       A_b0 -> A_b1 -> A_b2
```

The incorrect path is a direct bypass:

```text
control token: A_b0 -> A_b1 -> A_b2
data %v:       A_b0 --------> A_b2
```

The bypass form makes the correspondence between a dynamic token instance and
its data payload implicit.  This is especially unsafe when the intermediate
scope has variable latency, such as a non-pipeline nested loop or conditional.

Therefore, a child scope in a pipelined parent must support live-through values:

1. The child entry side dequeues/captures every parent pipeline value that is
   live across the child scope, even if the child body does not use it.
2. The child keeps the value in single-context storage if the child is
   non-pipeline and protected by a context token.
3. The child exit side re-emits the value to the next parent block before or
   together with emitting the child exit token.

For a non-pipeline nested `tor.for` inside a pipelined parent, the loop entry
rule captures live-through inputs into loop-owned state registers, and the loop
exit path re-emits those values to the parent pipeline output FIFOs.  Values used
by the loop body may additionally be forwarded to loop-body FIFOs, but body use
is not required for live-through preservation.

For a non-pipeline nested `tor.if` inside a pipelined parent, the if dispatch
rule captures live-through inputs into if-owned state registers, and the if join
rule re-emits those values after the selected branch completes.
