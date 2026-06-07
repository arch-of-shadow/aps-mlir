# APS SCF-to-Affine Wrapper Debug Notes

本文档记录本轮从 `aps-raise-scf-to-affine` wrapper 调试开始，为了让
CADL `u8/i8` loop IV 一路传到 `aps-to-cmt2` 并最终生成 8-bit IV register
所做的代码改动。

核心目标：

1. 能进入 affine 优化路径的 loop 仍然走 affine。
2. 不能进入 affine 路径的 loop 不被全局 index normalize 误伤。
3. 经过 affine path 的 loop 在回到 `scf`/`tor`/`cmt2` 时恢复原始 IV 位宽。
4. `aps-to-cmt2` 看到的 loop IV 是窄类型，例如 `i8`，从而生成
   `Reg_width8_init0`，而不是默认 `Reg_width32_init0`。

## 问题背景

原 pipeline 中使用：

```text
--normalize-scf-for-indices
--aps-mem-to-memref
--canonicalize
--raise-scf-to-affine
```

这个顺序有两个问题。

第一，`normalize-scf-for-indices` 是全局 pass，会把所有 integer-typed
`scf.for` 的 lower/upper/step/IV 变成 `index`。如果某个 loop 后面不能 raise 到
`affine.for`，它也已经失去原始位宽。

第二，即使 loop 成功进入 `affine.for`，affine dialect 的 IV 也是 `index`。
如果后续 lowering 不显式恢复原始位宽，`scf-to-tor` 和 `aps-to-cmt2` 会把 IV
当作 `index/i32` 处理，最终 CMT2 生成 32-bit loop IV reg。

真实失败现象包括：

```text
error: failed to legalize operation 'scf.for'
that was explicitly marked illegal
```

以及 `i8` loop 在 CMT2/SV 中变成 32-bit IV register。

## Pipeline 改动

### `tools/aps-opt/main.cc`

改动：

```text
--normalize-scf-for-indices
--aps-mem-to-memref
--canonicalize
--raise-scf-to-affine
```

替换为：

```text
--aps-raise-scf-to-affine
--canonicalize
```

为什么要改：

`aps-raise-scf-to-affine` 是 APS 自己的 wrapper，它只对实际被转换为
`affine.for` 的 loop 做局部 index materialization，不再全局改写所有
`scf.for`。

如果不这样改：

无法进入 affine path 的 loop 也会被提前改成 `index/i32`，原始 CADL loop
control 位宽被抹掉；后面即使想在 `scf-to-tor` 或 `aps-to-cmt2` 恢复，也很难
判断原来是 `u8/i8` 还是普通 32-bit loop。

## Pass 注册与构建

### `include/APS/Passes.h`

改动：

新增：

```c++
std::unique_ptr<Pass> createAPSRaiseSCFToAffinePass();
```

为什么要改：

让 `aps-opt` pass registry 可以创建 wrapper pass。

如果不这样改：

`Passes.td` 中即使声明了 pass，C++ 侧也没有 constructor 入口，链接或注册会失败。

### `include/APS/Passes.td`

改动：

新增 pass：

```tablegen
def APSRaiseSCFToAffine : Pass<"aps-raise-scf-to-affine"> { ... }
```

并声明依赖 dialect：

```text
affine, arith, memref, scf
```

为什么要改：

这个 pass 内部会创建 `affine.for`、`arith.index_cast`、
`memref.load/store` 和读取 `scf.for`。

如果不这样改：

`--aps-raise-scf-to-affine` 不会成为合法 pass 名；缺 dependent dialect 时，pass
pipeline 可能在创建新 op 时缺 dialect 注册。

### `lib/APS/CMakeLists.txt`

改动：

把 `APSRaiseSCFToAffine.cpp` 加进 APS library。

为什么要改：

让新 pass 实现参与编译链接。

如果不这样改：

pass 声明存在，但实际 constructor/implementation 不会进 binary。

## Wrapper Pass

### `lib/APS/APSRaiseSCFToAffine.cpp`

这是新文件，实现 `--aps-raise-scf-to-affine`。

#### `kOriginalIVTypeAttr`

改动：

定义：

```c++
constexpr StringLiteral kOriginalIVTypeAttr = "aps.original_iv_type";
```

成功 raise 时写到 `affine.for`：

```c++
affineFor->setAttr(kOriginalIVTypeAttr,
                   TypeAttr::get(forOp.getInductionVar().getType()));
```

为什么要改：

`affine.for` 的 IV 固定是 `index`，会天然丢掉 CADL loop 的原始位宽。这个 attr
是后续 `lower-affine-for` 恢复 `i8/i16/i32` IV 的来源。

如果不这样改：

能 raise 到 affine 的 loop 反而最容易丢位宽。后续只能靠 body 里
`index_cast index -> i8` 猜测，遇到 canonicalize、array partition 或 IV 没被
直接 cast 的情况会不可靠。

#### `getConstantStep`

改动：

支持从 `arith.constant index`、`arith.constant int`、通用
`arith.constant` 的 `IntegerAttr` 中提取 step。

为什么要改：

`affine.for` 需要静态正 step。本 wrapper 只处理能可靠 materialize 的 loop。

如果不这样改：

一些合法的 integer constant step loop 会被误判为不能 raise，导致 affine
load/store 和 array partition 机会丢失。

#### `materializeIndexBound`

改动：

对 lower/upper bound：

1. `index` 直接使用。
2. integer constant 变成 `arith.constant index`。
3. 非 constant integer 变成 `arith.index_cast` 到 `index`。

为什么要改：

`affine.for` bound map 使用 affine symbol identity map，需要 index-typed
symbol operand。

如果不这样改：

动态 `i8/i32` bound 不能进入 affine form；或者会创建类型不合法的 affine op。

#### `cloneOrConvertBodyOp`

改动：

在复制 loop body 时：

1. 普通 op 使用 `rewriter.clone(op, mapping)`。
2. `aps.read_smem` 转成 `memref.load`。
3. `aps.write_smem` 转成 `memref.store`。
4. memory indices 转成 index。

为什么要改：

后续 `affine-raise-from-memref` / `raise-memref-to-affine` 只认识
`memref.load/store`，不认识 APS memory op。wrapper 里局部完成 APS memory 到
memref 的转换，才能让真实带 memory 的 loop raise 到 affine load/store。

如果不这样改：

真实 case 里有 memory 的 loop 会停在 `aps.read_smem/write_smem`，后续不能生成
`affine.load/store`，array partition 也无法看到 affine memory access。

#### `APSRaiseForPattern`

改动：

匹配 `scf.for` 后执行：

1. 检查 lower/upper 可 materialize 到 index。
2. 检查 step 是正的 static constant。
3. 创建 `affine.for`，bound 用 symbol identity map。
4. 保留原 loop attrs。
5. 写 `aps.original_iv_type`。
6. 在 affine body 起始处把 affine IV cast 回旧 IV type，body clone 继续使用旧
   IV type。
7. `scf.yield` 改成 `affine.yield`。
8. 用 `affine.for` result 替换原 `scf.for` result。

为什么要改：

这是 wrapper 的核心：只在确认能够构造合法 affine loop 时提交改写，并且在 body
内部保持原始 IV use 类型，避免后续 arithmetic/memory op 看到意外的 `index`。

如果不这样改：

只能继续依赖全局 `normalize-scf-for-indices + raise-scf-to-affine`。失败 loop 会
被误伤，成功 loop 也会丢失原始 IV 位宽。

## Affine Lowering 回 SCF

### `lib/TOR/LowerAffineFor.cpp`

#### 读取 `aps.original_iv_type`

改动：

新增：

```c++
getOriginalIVType(AffineForOp op)
inferNarrowIVType(AffineForOp op)
```

lower `affine.for` 时优先使用：

```text
aps.original_iv_type
```

如果 attr 不存在，再根据 IV uses 里 `index_cast index -> iN` 推断窄类型；都推不出
时才回退到 `index`。

为什么要改：

affine path 中 `affine.for` 的 IV 只能是 `index`。要让 `scf.for` 恢复为 `i8`，
必须在 lowering 时主动选择目标 IV type。

如果不这样改：

`lower-affine-for` 会生成 index-typed `scf.for`。进入 `scf-to-tor` 后通常变成
`i32` control，CMT2 生成 `Reg_width32_init0`。

#### bound/step cast 到恢复后的 IV type

改动：

lower bound、upper bound 和 step 先按 affine 规则算出 index value，再 cast 到
恢复后的 IV type。step 对 integer IV 使用 `arith.constant int`。

为什么要改：

`scf.for` 要求 lower/upper/step 和 IV 类型一致。

如果不这样改：

会得到 `scf.for` IV 是 `i8`、但 bound/step 是 `index` 的非法 IR。

#### affine map operand 转 index

改动：

新增：

```c++
materializeAffineMapOperandsAsIndex
castToIndex
```

在 lowering `affine.min/max`、affine bound、`affine.load/store/apply` 前，把 map
operands cast 到 index。

为什么要改：

恢复窄 IV 后，body 里原来给 affine map 用的 IV 可能变成 `i8`。但是 affine map
expansion 仍要求 index operands。

如果不这样改：

会出现 affine map expansion 或生成的 `arith.cmpi/muli/addi` 混用 `i8` 与
`index` 的 verifier error。

#### `affine.load/store` lowering

改动：

新增 `lowerAffineLoadInLoop` 和 `lowerAffineStoreInLoop`，在 clone affine body 到
`scf.for` body 时：

1. map operands 先 cast 到 index。
2. 展开 affine map。
3. 生成 `memref.load/store`。

为什么要改：

普通 clone 会把 `affine.load/store` 原样带进 `scf.for`，但恢复窄 IV 后 affine op
的 map operand 类型可能不再合法。

如果不这样改：

真实 memory case 会在 `lower-affine-for` 后保留非法 affine memory op，或者在后续
pass 中因为 `i8/index` 混用失败。

#### body clone 中的 arithmetic cast repair

改动：

对 `arith.index_cast`、`addi/subi/muli`、`cmpi`、`select` 等常见 op 做特殊 clone：
从 mapping 中取新 value 后 cast 回原 op operand/result type。

为什么要改：

当 affine IV 从 `index` 恢复为 `i8` 时，原 body 中部分 op 的 operand mapping 会
变化。直接 clone 容易得到 `arith.addi i8, index` 这种非法混合类型。

如果不这样改：

`graphics` 这类真实 case 会在 `lower-affine-for` 或后续 canonicalize 阶段报
verifier error。

#### `AffineIfLowering`

改动：

补了 `affine.if` 到 `scf.if` 的 lowering，并对 then/else region clone 过程做同样
的 value mapping/cast repair。

为什么要改：

array partition 或 affine canonicalization 可能在 affine loop 内留下
`affine.if`。恢复窄 IV 后它也需要落回普通控制流。

如果不这样改：

`lower-affine-for` 后仍可能有 affine dialect control op 残留，后续
`scf-to-tor`/CMT2 flow 不能稳定处理。

## Array Partition

### `lib/TOR/NewArrayPartition.cpp`

#### `materializeIndexOperands`

改动：

新增 helper：

```c++
SmallVector<Value> materializeIndexOperands(PatternRewriter &, Location,
                                            ValueRange operands)
```

把 dynamic bank 相关 affine operands 转成 index。

为什么要改：

`new-array-partition` 在动态 bank load/store 中会创建新的 `affine.load`、
`affine.store` 和 `affine.apply`。恢复窄 IV 后，原 map operand 可能是 `i8`。

如果不这样改：

动态 bank 逻辑会创建 `affine.apply`/`affine.load/store` 使用 `i8` map operand，
后续 lowering 展开时出现 index/integer 类型错误。

#### dynamic bank load

改动：

在 dynamic bank load 路径中，所有新建 bank load 和 runtime bank
`AffineApplyOp` 使用 `indexOperands`，而不是直接使用原 `load.getMapOperands()`。

为什么要改：

bank selection affine expression 本质上是 index arithmetic。

如果不这样改：

`graphics` 中 array partition 之后的 affine load 可能带 `i8` map operand，后续
`lower-affine-for` 生成非法 arithmetic。

#### dynamic bank store

改动：

在 dynamic bank store 路径中，runtime bank `AffineApplyOp`、读取旧 bank value 的
`AffineLoadOp` 和写回每个 bank 的 `AffineStoreOp` 都使用 index operands。

为什么要改：

store 路径比 load 多了 read-modify-select-store，如果其中任何一个 affine op 保留
窄 integer map operand，都会让后续 lowering 失败。

如果不这样改：

带 partition 的 store case 会在 affine map expansion 或 verifier 中失败，array
partition 无法和 i8 loop IV 同时工作。

## SCF to TOR

### `lib/TOR/SCFToTOR.cpp`

#### `castIntegerLikeValue`

改动：

新增统一 cast helper：

1. 相同类型直接返回。
2. index 与 integer 之间用 `arith.index_cast`。
3. integer 变宽用 `arith.extui`。
4. integer 变窄用 `arith.trunci`。
5. 非 integer/index 才退到 `unrealized_conversion_cast`。

为什么要改：

原 type converter 会 materialize `builtin.unrealized_conversion_cast`。这些 cast 在
后面不一定被清理掉，并且 `index_cast i32 -> i8` 在 MLIR 中不是合法替代。

如果不这样改：

`scf-to-tor` 输出会残留 `unrealized_conversion_cast`，或生成非法
`arith.index_cast`，最终 `aps-to-cmt2` 前 verifier 失败。

#### `IndexCastOpConversion`

改动：

`arith.index_cast` 不再简单 clone，而是按 source/dest 类型转换成真正的
`index_cast`、`extui` 或 `trunci`。

为什么要改：

经过 type conversion 后，原来的 index/integer 边界可能已经变成 integer/integer
边界。

如果不这样改：

例如 `i8 -> index -> i32` 可能被保留成不合适的 unrealized cast 链，或者形成
非法 cast。

#### `CmpIOpConversion`

改动：

`CmpIOpConversion` 改为走 `IndexTypeConversionPattern`，先 prepare operands，而不是
遇到 index operand 就 failure。

为什么要改：

affine lowering 和 array partition 会生成 index-typed compare。`scf-to-tor`
需要把它们合法转成 TOR compare。

如果不这样改：

`arith.cmpi` 的 index operand 会让 conversion pattern 失败，导致后续
`scf.for` 或 arithmetic op 残留。

#### `ForOpConversion` 控制变量位宽

改动：

lower/upper/step 统一 cast 到 induction variable 的转换后类型，而不是各自旧类型：

```text
targetType = index < 3 ? controlType : convertIndexTypeToI32(oldType)
```

为什么要改：

如果 loop IV 是 `i8`，动态 bound 也必须被 cast 到 `i8`，`tor.for` 的 control
operands 才一致。

如果不这样改：

会出现 `tor.for` lower 是 `i8`、upper 是 `index/i32` 的非法 IR，或者因为类型不一致
在 conversion 中失败。

#### `ResidualForOpConversion`

改动：

新增 greedy rewrite pattern，专门把 residual `scf.for` 转成 `tor.for`，并保留
pipeline attr 逻辑。

为什么要改：

在 wrapper + affine path 调试后，仍会存在一些已经合法降回 `scf.for`、但不适合再走
dialect conversion 的 loop。它们需要被确定性地转成 `tor.for`。

如果不这样改：

`scf-to-tor` 可能留下 `scf.for`，而后面 `schedule-tor` / `aps-to-cmt2` 不接受。

#### 不再吞掉 illegal `scf.for` conversion failure

改动：

之前曾经临时做过：

```c++
target.addIllegalOp<scf::ForOp, scf::IfOp, scf::WhileOp>();
(void)applyPartialConversion(...);
```

现在改成：

1. 先运行 `ResidualForOpConversion`。
2. partial conversion 只把 `scf.if/while` 标 illegal。
3. `applyPartialConversion` failure 立即 `signalPassFailure()`。
4. 最后显式检查是否还有 `scf.for/if/while`，有就 fail。

为什么要改：

`failed to legalize operation 'scf.for' that was explicitly marked illegal`
是 MLIR conversion 的真实错误，不是可忽略 warning。

如果不这样改：

pass 可能一边输出错误诊断，一边继续生成文件。用户看到的 flow 看似成功，但 IR 的
合法性其实没有被保证，后续错误会变得很难定位。

#### `CleanupUnrealizedCasts`

改动：

增强 cleanup：

1. identity cast 删除。
2. index/integer 和 integer/integer 的 unrealized cast 改成真实 cast。
3. 识别 `index_cast` 后接 unrealized cast 的链，直接折叠成真实 cast。

为什么要改：

`scf-to-tor` 的 type conversion、affine lowering 和 canonicalize 组合后容易留下
`builtin.unrealized_conversion_cast`。

如果不这样改：

`aps-to-cmt2` 前仍会有 unrealized cast，后续 lowering 不应依赖这种占位 cast。

## CADL Real Case 输入

### `examples/cryptography/pqc.cadl`

改动：

把多个 loop IV 和 loop-derived index 从 `u32` 改成 `u8`，例如：

```cadl
with idx: u8 = (0, idx_) do { ... }
let elem_idx: u8 = idx * 4 + bit_idx;
```

为什么要改：

这是为了真实验证 CADL `u8` loop IV 是否能一路传到 CMT2/SV，而不是只测手写 MLIR。

如果不这样改：

真实 `crypto_pqc` case 仍然主要覆盖 `u32` loop，无法证明新的 wrapper 和位宽恢复
逻辑能让 CMT2 生成 8-bit IV reg。

### `examples/graphics/graphics.cadl`

改动：

把 `mean_var_fixed`、`phong_fixed`、`rgb2yuv_fixed` 中的 loop IV 和 derived index
改成 `u8`。

为什么要改：

`graphics` 是更复杂的真实 case，包含 memory、nested loop、array partition 和较多
affine load/store。它能覆盖 wrapper 与 memory/partition 的组合路径。

如果不这样改：

只测小 case 或 `crypto_pqc`，不能暴露 dynamic bank array partition 和
`lower-affine-for` 中的 `i8/index` 混用问题。

## 测试

### `tests/test_mlir/test_aps_raise_scf_to_affine_pass.py`

这是新测试文件，覆盖 wrapper、affine lowering 和真实 case。

#### u8 static bound raise

验证：

1. `scf.for i8` 能 raise 到 `affine.for`。
2. `affine.for` 带 `aps.original_iv_type = i8`。
3. body 中有必要的 `index_cast` 和 `arith.extui`。

如果没有这个测试：

后续很容易改坏 attr preservation，导致 affine path 再次丢位宽。

#### lower-affine-for 恢复 u8

验证：

1. 手写 `affine.for` body 中有 `index_cast index -> i8` 时，可以推回 `scf.for i8`。
2. 有 wrapper attr 时优先用 attr 恢复。

如果没有这个测试：

`lower-affine-for` 可能退回默认 index IV，最终 CMT2 变回 32-bit IV reg。

#### memory loop raise

验证：

1. `memref.load/store` loop 能 raise 成 `affine.load/store`。
2. `aps.read_smem/write_smem` loop 能通过 wrapper 局部转成 memref，再进入 affine memory
   path。

如果没有这个测试：

wrapper 可能只对纯 arithmetic loop 有效，真实 memory loop 仍失败。

#### dynamic/negative step

验证：

1. dynamic step loop 保持 `scf.for`。
2. negative step loop 保持 `scf.for`。

如果没有这个测试：

wrapper 可能错误创建非法 `affine.for`，因为 affine step 必须是正静态 step。

#### real cases

验证：

对 `outputs/cmt2_real_cases/*.aps.mlir` 跑真实 memory pipeline，确认：

1. `scf.for` 消失。
2. 有 `affine.for`。
3. 有 `affine.load/store`。
4. residual `memref.load/store` 数量符合预期。

如果没有这个测试：

单元 case 通过不代表真实 CADL 输出能经过 memory + affine + partition 组合路径。

#### `scf-to-tor` regression

验证：

`u8 scf.for` 经过：

```text
--convert-input
--scf-to-tor
--canonicalize
```

后：

1. log 没有 `failed to legalize operation 'scf.for'`。
2. 输出没有 `scf.for`。
3. 输出没有 `unrealized_conversion_cast`。
4. 输出有 `tor.for`，且 control type 是 `i8`。

如果没有这个测试：

很容易重新引入“把 `scf.for` 标 illegal 但吞掉 conversion failure”的错误路径。

## 本轮验证结果

已跑过：

```text
pixi run build
pixi run pytest tests/test_mlir/test_aps_raise_scf_to_affine_pass.py -q
```

结果：

```text
19 passed
```

真实 case 验证：

1. `graphics` 导出 CMT2 前 MLIR 成功。
2. `graphics.before_cmt2.log` 中没有 `error` / `failed to legalize`。
3. `graphics.before_cmt2.mlir` 中没有 `scf.*` / `unrealized_conversion_cast`。
4. `graphics.before_cmt2.mlir` 中 `tor.for` 是 `i8`。
5. `graphics` 和 `crypto_pqc` 跑 `aps-e2e` 到 CMT2 成功。
6. CMT2 中 loop IV instance/read/write 都是 `Reg_width8_init0` /
   `!firrtl.uint<8>`。
7. 继续 lower 到 SV 成功。
8. SV 中 `_iv` 匹配到的都是 `Reg_width8_init0` / `wire [7:0]`，没有匹配到
   `_iv` 的 16/32/64-bit reg/read wire。

## 设计原则总结

这轮修改的关键原则是：

1. 不再全局把所有 loop normalize 到 index。
2. 只有确认进入 affine path 的 loop 才局部 materialize index bound/memory index。
3. affine path 必须携带原始 IV type，并在 lower 回 `scf` 时恢复。
4. 所有 affine map operands 在真正使用 affine arithmetic 时显式 cast 到 index。
5. `scf-to-tor` 不能吞 MLIR conversion failure。
6. `unrealized_conversion_cast` 不能作为 APS/CMT2 flow 的正常输出。
