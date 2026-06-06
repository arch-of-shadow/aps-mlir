# APS to CMT2 FIFO Flow

本文只解释 `aps-to-cmt2` 在 **block / slot / loop** 三个层级上的 FIFO 传递，不展开 adaptor、memory pool、global register、CSR 等外设路径。  
这份文档的目标不是描述抽象架构，而是对着当前代码说明：

- 谁生产值
- 谁消费值
- 值在哪一层被物化成 FIFO
- token 如何串起执行顺序
- loop 为什么要再套一层自己的 FIFO

主要代码入口：

- [lib/APS/APSToCMT2/RuleGeneration.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/RuleGeneration.cpp)
- [lib/APS/APSToCMT2/BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp)
- [lib/APS/APSToCMT2/BBHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BBHandler.cpp)
- [lib/APS/APSToCMT2/LoopHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/LoopHandler.cpp)
- [circt/include/circt/Dialect/Cmt2/ECMT2/Signal.h](/home/zyy/aps-mlir/aps-mlir/circt/include/circt/Dialect/Cmt2/ECMT2/Signal.h)

## 1. 总体分层

当前实现可以按三层理解：

1. `BlockHandler`
   - 负责把一个 `tor.func` 切成多个 `BlockInfo`
   - 负责 block 间 token FIFO
   - 非 pipeline 时负责创建跨 block value reg
   - pipeline 时负责创建跨 block value FIFO 和输入 fanout FIFO
   - 负责把每个 block 交给更细的处理器

2. `BBHandler`
   - 负责把一个 block 再按 `starttime` 切成多个 slot
   - 负责 block 内 slot 间 token FIFO
   - 非 pipeline 时用 local reg 保存跨 slot value
   - pipeline 时用 value FIFO 连接 producer slot 和 consumer slot
   - 负责生成每个 slot 对应的 `cmt2.rule`

3. `LoopHandler`
   - 负责把 `tor.for` 规范化成 `entry -> body -> next`
   - 非 pipeline loop 使用 loop state reg、IV reg、loop-to-body FIFO
   - pipeline loop 使用 issue token FIFO、done counter reg、IV FIFO
   - 负责把 loop body 再交回 `BlockHandler + BBHandler`

这三层不是并列替代关系，而是嵌套关系：

```text
tor.func
  -> BlockHandler
     -> block
        -> BBHandler
           -> slot rules
     -> loop block
        -> LoopHandler
           -> loop entry / body / next
           -> loop body uses BlockHandler + BBHandler again
```

## 2. Block 间：`BlockHandler`

### 2.1 先切 block，再看数据流

`BlockHandler::segmentBlockIntoBlocks()` 会扫描每个 MLIR block，把遇到的控制流边界单独切出来：

- `tor.for`
- `tor.if`
- `tor.while`

普通操作会被放进当前 segment，控制流操作会单独成为一个 segment。见 [BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp#L321)。

每个 segment 最终变成一个 `BlockInfo`，其中记录：

- `operations`
- `producedValues`
- `consumedValues`
- `input_fifos`
- `output_fifos`
- `input_token_fifo`
- `output_token_fifo`

这些字段是后续所有 FIFO 传递的基础。见 [include/APS/BlockHandler.h](/home/zyy/aps-mlir/aps-mlir/include/APS/BlockHandler.h)。

### 2.2 哪些值需要跨 block 传

`analyzeOperationInBlock()` 会为每个 block 统计：

- 这个 block 里产生了什么 value
- 这个 block 里用了哪些外部 value

但并不是所有 value 都会变成 FIFO。`isVirtualValue()` 会过滤掉：

- `arith.constant`
- `memref.get_global`
- `memref.alloc` / `alloca`
- APS 的 request / collect 类操作 token

也就是说，**只有真正代表跨 block 数据依赖的值才会进入 `crossBlockFlows`**。见 [BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp#L636)。

### 2.3 block 间 value storage 是怎么建的

`analyzeCrossBlockDataflow()` 会遍历每个 producer block 的 `producedValues`，再调用 `findValueConsumers()` 找出所有 consumer block。若 consumer block 不是 producer 自己，就形成一条 `CrossBlockValueFlow`。见 [BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp#L522)。

随后 `createCrossBlockValueRegs()` 会按是否 pipeline 选择不同 storage：

- **non-pipeline mode**
  - 一个 producer value 在 producer block 下创建一个 reg
  - 同一 producer scope 里的多个 consumer 共享读取这个 reg
  - 这依赖 block token 串行保证不会有多个 activation 覆盖同一个 reg

- **pipeline mode**
  - 每条 producer -> consumer flow 创建一条 FIFO
  - consumer block 的首个 slot rule 统一 dequeue `inputFIFOs`
  - 如果实际 consumer 在更晚 slot，首个 slot 再通过 block 内 value FIFO 转发
  - 这样可以承载多个 activation overlap

命名形如：

```text
non-pipeline reg: {prefix}b{producer}v{counter}
pipeline fifo:    {prefix}b{producer}b{consumer}v{counter}
```

这一步很关键，因为它说明了当前实现已经把 non-pipeline 和 pipeline 分开：non-pipeline 主要用 reg，pipeline 才用 FIFO。见 [BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp#L514)。

### 2.4 block 间 token FIFO 怎么串

`createBlockTokenFIFOs()` 会为相邻 block 建立 1-bit token FIFO：

```text
block_i -> token_fifo_bi_bi+1 -> block_i+1
```

它的作用只有一个：**让后一个 block 知道前一个 block 结束了**。见 [BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp#L171)。

### 2.5 block 间输入分发

如果 pipeline scope 的一个父级输入 value 被多个 sub-block 共同使用，`createPipelineInputFanoutFIFOs()` 会创建从第一个使用者到后续使用者的 fanout FIFO。

这层只在 `pipelineMode` 下启用。non-pipeline mode 不走这套输入 fanout FIFO，而是依赖 reg/local value 传递。见 [BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp#L580)。

### 2.6 `processAllBlocks()` 做了什么

`processAllBlocks()` 负责把外层 block 级 FIFO 关系真正灌进每个 block：

- 第一个 sub-block：
  - token 来自父级 `inputTokenFIFO`
  - data 来自父级 `input_fifos` 或 parent input reg
- 中间 sub-block：
  - token 来自上一个 sub-block 的 `output_token_fifo`
  - data 按需要从 block-level reg 或 pipeline fanout FIFO 接收
- 最后一个 sub-block：
  - token 送回父级 `outputTokenFIFO`
  - data 送回父级 `output_fifos`

这一步决定了“block 间”与“sub-block 间”的边界是如何衔接的。见 [BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp#L683)。

## 3. Block 内：`BBHandler`

### 3.1 slot 是什么

`BBHandler::collectOperationsFromList()` 会读取每个 op 的 `starttime`，把 op 分到 `slotMap[slot]`。  
如果某个 op 没有 `starttime`，它会暂时落到 slot 0。见 [BBHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BBHandler.cpp#L1)。

因此，在这个实现里：

- block 是控制流层的粗分段
- slot 是同一个 block 内的时间片

### 3.2 block 内 value 传递怎么建

当前实现按 `pipelineMode` 分成两类。

**non-pipeline mode**：

- 遍历每个 slot 的 op result
- 如果 result 会被后续 slot 使用，就创建 local value reg
- 第一次 producer slot 写 reg
- 后续 consumer slot 读 reg
- 对 block 输入 value，如果第一 slot 后还要使用，也会创建 local reg 保存一次 dequeue/read 的结果

**pipeline mode**：

- 遍历每个 slot 的 op result
- 如果 result 会被后续 slot 使用，就创建 producer slot -> consumer slot FIFO
- 对 block 输入 value，如果后续 slot 还要使用，也会创建 first slot -> consumer slot FIFO
- producer/first slot enq，consumer slot deq

这一步与 block 间 FIFO 的差异在于：

- block 间 FIFO 关注的是“不同 block 之间的 def-use”
- slot 间 FIFO 关注的是“同一个 block 内不同时间片之间的 def-use”

所以一个 value 可以同时有 block 间 storage 和 block 内 slot storage。见 [BBHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BBHandler.cpp#L293)。

注意：pipeline mode 目前是 producer slot 直连 consumer slot，不是 stage-by-stage live-through。若这些 FIFO 深度为 1，远距离 consumer 会导致 producer 过早遇到 full，从而降低 II。下一步设计应把 pipeline BB value 传递改成随 token 一样逐 stage 前进。

### 3.3 block 内 token FIFO 怎么串

`createTokenFIFOs()` 为每个 slot 和下一个 slot 之间创建 token FIFO：

- 第一个 slot 用 `FIFO2I`
- 其他 slot 用 `FIFO1Push`

这层 token FIFO 只负责同一 block 内的 stage 顺序，不负责数据本身。见 [BBHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BBHandler.cpp#L293)。

### 3.4 slot rule 的执行顺序

`generateSlotRules()` 里，每个 slot 的 rule body 通常按这个顺序执行：

1. 如果是第一个 slot，先消费 `block_input_token_fifo`
2. 对跨 block 输入做处理
3. 如果 slot == 0，再处理 RoCC command bundle
4. 调 `generateRuleForOperation()` 逐个翻译当前 slot 的 op
5. 把当前 slot 产生的 value 写到：
   - non-pipeline: local value reg / cross-block value reg
   - pipeline: block 内后续 slot FIFO / 下游 block FIFO
6. 写入本 slot 的 stage token
7. 如果是最后一个 slot，再写 block completion token 到 `block_output_token_fifo`

见 [BBHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BBHandler.cpp#L724)。

### 3.5 `localMap` 是 op generator 的最终取值入口

`OperationGenerator::getValueInRule()` 的查找顺序是：

1. 先查 `localMap`
2. 再把 `arith.constant` 折成 FIRRTL 常量
3. non-pipeline slot rule 里，跨 slot operand 由 BBHandler 预先读 reg 放入 `localMap`
4. pipeline slot rule 里，跨 slot operand 由 BBHandler 预先 deq FIFO 放入 `localMap`

也就是说，**FIFO/reg 读取大多已经在 `BBHandler` 的 slot preamble 中完成，op generator 最终主要从 `localMap` 取值**。这就是你说的“混沌”的根源，但它是有明确优先级的。见 [BBHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BBHandler.cpp#L944)。

## 4. Loop：`LoopHandler`

### 4.1 loop 不是单独一个 block，而是一个小型调度器

`LoopHandler::processLoopBlock()` 会找到该 block 中的 `tor.for`，然后构造一个 `LoopInfo`。接着它做三件事：

1. `createLoopInfrastructure()`
2. `processLoopBodyOperations()`
3. `generateCanonicalLoopRules()`

见 [LoopHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/LoopHandler.cpp#L53)。

### 4.2 loop 内有哪些 FIFO

`createLoopInfrastructure()` 会创建：

- `entry_to_body` token FIFO
- `body_to_next` token FIFO
- `loop_state_reg`
- non-pipeline: `inductionVarReg`
- pipeline: `inductionVarFIFO`
- `input_state_registers`
- non-pipeline: `loop_to_body_fifos`
- pipeline: `issueTokenFIFO` 和 `doneCounterReg`

它们分别承担不同职责：

- `entry_to_body`：让 body 开始
- `body_to_next`：让 next 知道 body 结束
- `loop_state_reg`：保存 loop counter/bound/step/iter_args 状态
- `inductionVarReg`：non-pipeline 时给 body 提供 induction variable
- `inductionVarFIFO`：pipeline 时给 body 提供每个 iteration 的 induction variable
- `input_state_registers`：跨迭代持久保存外部输入
- `loop_to_body_fifos`：non-pipeline 时把外部输入送进 body
- `issueTokenFIFO`：pipeline loop 的 issue 触发
- `doneCounterReg`：pipeline loop 的 retire 计数

见 [LoopHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/LoopHandler.cpp#L407) 和 [LoopHandler.h](/home/zyy/aps-mlir/aps-mlir/include/APS/LoopHandler.h#L27)。

### 4.3 entry rule 在做什么

`generateLoopEntryRule()` 的 body 顺序大致是：

1. 消费外层 `inputTokenFIFO`
2. 对 loop 真的会用到的外部输入，先从 `input_fifos` dequeue
3. 如果某值需要跨迭代保存，就写入 `input_state_registers`
4. 如果某值要给 body 直接消费，就写入 `loop_to_body_fifos`
5. 初始化 `loop_state_reg`
6. non-pipeline 时如果 induction var 被 body 使用，就写 `inductionVarReg`
7. pipeline 时初始化 `doneCounterReg` 并写 `issueTokenFIFO`
8. non-pipeline 时发 token 到 `entry_to_body`

这说明 loop 的“第一次进入”并不是直接把所有值送进 body，而是先构建一层稳定状态。见 [LoopHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/LoopHandler.cpp#L131)。

### 4.4 next / issue / retire rule 在做什么

non-pipeline loop 使用 `generateLoopNextRule()`：

1. 消费 `body_to_next`
2. 从 `loop_state_reg` 读出当前状态
3. 计算 `nextCounter` 和 `shouldContinue`
4. 如果继续：
   - 回写更新后的 state
   - 从 `input_state_registers` 读出值，再送回 `loop_to_body_fifos`
   - 更新 induction var reg
   - 再发一个 token 给 `entry_to_body`
5. 如果退出：
   - 把 token 送到 `outputTokenFIFO`

见 [LoopHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/LoopHandler.cpp#L256)。

pipeline loop 使用 `entry / issue / retire` 三条 rule：

1. `entry` 消费外层 token，初始化 `loop_state_reg` 和 `doneCounterReg`，写 `issueTokenFIFO`
2. `issue` 消费 `issueTokenFIFO`，按当前 counter 发 body token 和 IV FIFO，更新 state，必要时再次写 `issueTokenFIFO`
3. `retire` 消费 `body_to_next`，更新 done counter，并在最后一个已 issue iteration 完成时发外层 output token

### 4.5 loop body 为什么再走一遍 `BlockHandler`

`processLoopBodyOperations()` 会构造一个新的 `BlockHandler`，并把：

- `loop.token_fifos.to_body` 作为输入 token FIFO
- `loop.token_fifos.body_to_next` 作为输出 token FIFO
- non-pipeline: `loop.loop_to_body_fifos` 作为输入 data FIFOs
- pipeline: `inductionVarFIFO` 作为 IV 输入 FIFO，`input_state_registers` 作为 body 可读 input reg

然后调用 `processLoopBodyAsBlocks()`。  
也就是说，**loop body 里面的 block/slot 逻辑和普通函数一样，只是外层多了一层 loop 专用状态和 token 结构**。见 [LoopHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/LoopHandler.cpp#L560)。

## 5. 一条值的完整路径怎么读

### 5.1 普通 block 间传值

如果一个 value 在 block A 里产生，在 block B 里消费：

1. `BlockHandler::analyzeCrossBlockDataflow()` 发现 A -> B
2. non-pipeline: `createCrossBlockValueRegs()` 在 A 下建 reg
3. non-pipeline: A 的 producer slot 写 reg，B 的 consumer slot 读 reg
4. pipeline: `createCrossBlockValueRegs()` 为 A/B 建 FIFO
5. pipeline: A 的 producer slot enq，B 的首个 slot rule dequeue；如果实际使用在后续 slot，再由 B 内部 FIFO 转发

### 5.2 同 block 不同 slot 传值

如果一个 value 在 slot 0 产生，slot 2 才消费：

1. non-pipeline: `BBHandler` 建 local reg，slot 0 写 reg，slot 2 读 reg
2. pipeline: `BBHandler` 建 slot 0 -> slot 2 FIFO，slot 0 enq，slot 2 deq
3. 目标 pipeline: 应改为 slot 0 -> slot 1 -> slot 2 的 stage-by-stage live-through

### 5.3 loop 里的跨迭代值

如果一个外部输入在 loop 中每次迭代都要保留：

1. `LoopHandler::createLoopInfrastructure()` 为该值建 `input_state_register`
2. non-pipeline entry rule 先读父级输入，再写进 state register 和 `loop_to_body_fifo`
3. non-pipeline next rule 读 state register，把值重新送回 `loop_to_body_fifo`
4. non-pipeline body 中通过 `BlockHandler + BBHandler` 看到的是 `loop_to_body_fifo`
5. pipeline body 目前通过 input state reg 读取 loop live-in，IV 通过 FIFO 读取

这就是 loop 里“同一个值被多次物化”的原因，但它不是重复计算，而是跨迭代状态管理。

## 6. 读代码时的建议顺序

如果要按代码彻底理解，建议按这个顺序读：

1. [RuleGeneration.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/RuleGeneration.cpp)
2. [BlockHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BlockHandler.cpp)
3. [BBHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/BBHandler.cpp)
4. [LoopHandler.cpp](/home/zyy/aps-mlir/aps-mlir/lib/APS/APSToCMT2/LoopHandler.cpp)
5. [circt/include/circt/Dialect/Cmt2/ECMT2/Signal.h](/home/zyy/aps-mlir/aps-mlir/circt/include/circt/Dialect/Cmt2/ECMT2/Signal.h)

先看层级，再看 FIFO，最后看 `Signal` 的语义，这样最不容易把局部实现误读成全局规则。

## 7. 这份实现里最重要的几个判断

- block 间和 block 内是两套 FIFO 体系，不要混成一层。
- loop 不是 block 的简单特例，而是额外再包一层 state/token 体系。
- `localMap` 不是 SSA 替代品，而是 rule 内部的值缓存。
- `getValueInRule()` 决定了“一个值到底是现算、常量、还是从 FIFO 来”。
- 大多数 guard 写成 `1'b1`，真正的顺序控制在 FIFO readiness 和 precedence。
- pipeline parent 调用 non-pipeline child scope 时，普通 block token 不足以防止重入；需要 child-owned context token。
- pipeline BB 当前的远距离 slot FIFO 是功能性实现，不是最佳 II=1 结构；stage-by-stage live-through 是下一步应改方向。

如果后续要改这条链路，先判断改动属于哪一层，再改对应 FIFO，不要直接从单个 op generator 下手。
