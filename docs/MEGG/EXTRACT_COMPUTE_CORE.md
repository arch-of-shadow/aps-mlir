# 从 APS MLIR 中提取计算核心用于匹配

## 问题定义

**输入**：APS MLIR pattern（包含硬件操作）
```mlir
func.func @flow_burst_add(%arg0: i32, %arg1: i32, %arg2: i32) {
  %0 = aps.readrf %arg0 : i32 -> i32           ← 硬件层
  %1 = aps.readrf %arg1 : i32 -> i32           ← 硬件层
  aps.memburstload %0, %mem_a[...] ...         ← 硬件层

  scf.for %i = %c0 to %c16 step %c1 {          ← 计算层（要匹配这个！）
    %a = aps.memload %mem_a[%i] : ...
    %b = aps.memload %mem_b[%i] : ...
    %sum = arith.addi %a, %b : i32
    aps.memstore %sum, %mem_a[%i] : ...
  }

  aps.memburststore %mem_a[...] ...            ← 硬件层
  aps.writerf %arg2, %c42 : ...                ← 硬件层
}
```

**需要**：自动提取计算核心
```mlir
// 自动生成的匹配 pattern
scf.for %i = %c0 to %c16 step %c1 {
  %a = memref.load %mem_a[%i] : memref<16xi32>
  %b = memref.load %mem_b[%i] : memref<16xi32>
  %sum = arith.addi %a, %b : i32
  memref.store %sum, %mem_a[%i] : memref<16xi32>
}
```

---

## 解决方案：自动抽象

### 实现位置

```python
# python/megg/rewrites/aps_pattern_extractor.py
```

### 核心逻辑

```python
class APSPatternExtractor:
    """从 APS MLIR pattern 中提取计算核心用于匹配"""

    # 硬件层操作（需要跳过）
    HARDWARE_IO_OPS = {
        'aps.readrf',
        'aps.writerf',
        'aps.memburstload',
        'aps.memburststore',
        'aps.memdeclare',
        'memref.get_global'
    }

    # 需要抽象的内存操作
    MEMORY_ABSTRACTION = {
        'aps.memload': 'memref.load',
        'aps.memstore': 'memref.store'
    }

    def extract_compute_core(self, aps_func: MOperation) -> MOperation:
        """
        提取计算核心

        策略：
        1. 识别计算 block（包含 scf.for/scf.if 的 block）
        2. 移除硬件 I/O 操作
        3. 替换 aps.memload/memstore → memref.load/store
        4. 构建新的函数（只包含计算逻辑）
        """

        # 步骤 1: 找到计算核心 block
        compute_blocks = self._find_compute_blocks(aps_func)

        if not compute_blocks:
            raise ValueError(f"No compute core found in {aps_func.name}")

        # 步骤 2: 克隆并清理
        match_func = self._create_match_function(aps_func, compute_blocks)

        return match_func

    def _find_compute_blocks(self, func: MOperation) -> List[MBlock]:
        """
        识别计算核心 block

        启发式规则：
        - 包含 scf.for/scf.if（控制流）
        - 包含 arith.*（算术）
        - 不是顶层 block（顶层通常是硬件设置）
        """
        compute_blocks = []

        def visit_block(block: MBlock, depth: int):
            has_control_flow = False
            has_arithmetic = False
            has_io = False

            for op in block.operations:
                # 检查控制流
                if op.name in ['scf.for', 'scf.if', 'scf.while']:
                    has_control_flow = True
                    # 递归检查控制流内部
                    for region in op.regions:
                        for inner_block in region.blocks:
                            visit_block(inner_block, depth + 1)

                # 检查算术
                elif op.name.startswith('arith.'):
                    has_arithmetic = True

                # 检查硬件 I/O
                elif op.name in self.HARDWARE_IO_OPS:
                    has_io = True

            # 计算 block：有控制流或算术，且不是纯 I/O
            if depth > 0 and (has_control_flow or has_arithmetic) and not has_io:
                compute_blocks.append(block)

        # 从顶层 block 开始
        for block in func.get_blocks():
            visit_block(block, depth=0)

        return compute_blocks

    def _create_match_function(
        self,
        aps_func: MOperation,
        compute_blocks: List[MBlock]
    ) -> MOperation:
        """
        创建匹配函数

        步骤：
        1. 创建新函数
        2. 调整函数签名（寄存器 → memref）
        3. 复制计算 block
        4. 抽象内存操作
        """

        # 步骤 1: 创建新函数
        match_func_name = aps_func.name + "_match"
        match_func = create_function(match_func_name)

        # 步骤 2: 调整函数签名
        # 原始: (%arg0: i32, %arg1: i32, %arg2: i32)  ← 寄存器索引
        # 匹配: (%mem_a: memref<N>, %mem_b: memref<N>) ← 内存参数

        # 收集计算 block 中使用的 memref
        memrefs = self._collect_memrefs(compute_blocks)

        # 为每个 memref 创建函数参数
        for memref in memrefs:
            arg = match_func.add_argument(memref.type, memref.name)

        # 步骤 3 & 4: 复制并清理计算 block
        for block in compute_blocks:
            cleaned_block = self._abstract_block(block)
            match_func.add_block(cleaned_block)

        return match_func

    def _collect_memrefs(self, blocks: List[MBlock]) -> List[MemrefInfo]:
        """收集 block 中使用的所有 memref"""
        memrefs = []

        for block in blocks:
            for op in block.operations:
                # aps.memload %mem[%i] → 记录 %mem
                if op.name == 'aps.memload':
                    memref = op.operands[0]
                    if memref not in memrefs:
                        memrefs.append(MemrefInfo(
                            name=memref.name,
                            type=memref.type
                        ))

                # aps.memstore %val, %mem[%i] → 记录 %mem
                elif op.name == 'aps.memstore':
                    memref = op.operands[1]
                    if memref not in memrefs:
                        memrefs.append(MemrefInfo(
                            name=memref.name,
                            type=memref.type
                        ))

        return memrefs

    def _abstract_block(self, block: MBlock) -> MBlock:
        """
        抽象 block 中的操作

        转换规则：
        - aps.memload → memref.load
        - aps.memstore → memref.store
        - 移除 memref.get_global
        - 保留所有其他操作
        """
        new_block = MBlock()

        for op in block.operations:
            # 跳过硬件 I/O 操作
            if op.name in self.HARDWARE_IO_OPS:
                continue

            # 抽象内存操作
            if op.name in self.MEMORY_ABSTRACTION:
                new_op = self._abstract_memory_op(op)
                new_block.add_operation(new_op)

            # 保留其他操作
            else:
                new_block.add_operation(op.clone())

        return new_block

    def _abstract_memory_op(self, op: MOperation) -> MOperation:
        """
        抽象单个内存操作

        aps.memload %mem[%i] : memref<16xi32>, i32 -> i32
        → memref.load %mem[%i] : memref<16xi32>

        aps.memstore %val, %mem[%i] : i32, memref<16xi32>, i32
        → memref.store %val, %mem[%i] : memref<16xi32>
        """

        if op.name == 'aps.memload':
            # aps.memload 语法: aps.memload %mem[%idx] : memref_ty, idx_ty -> result_ty
            memref = op.operands[0]
            indices = op.operands[1:]

            # memref.load 语法: memref.load %mem[%idx] : memref_ty
            return create_memref_load_op(
                memref=memref,
                indices=indices,
                result_type=op.result_types[0]
            )

        elif op.name == 'aps.memstore':
            # aps.memstore 语法: aps.memstore %val, %mem[%idx] : val_ty, memref_ty, idx_ty
            value = op.operands[0]
            memref = op.operands[1]
            indices = op.operands[2:]

            # memref.store 语法: memref.store %val, %mem[%idx] : memref_ty
            return create_memref_store_op(
                value=value,
                memref=memref,
                indices=indices
            )

        raise ValueError(f"Unknown memory op: {op.name}")
```

---

## 使用示例

### 输入（APS MLIR）

```mlir
module {
  memref.global @mem_a : memref<16xi32>
  memref.global @mem_b : memref<16xi32>

  func.func @flow_burst_add(%arg0: i32, %arg1: i32, %arg2: i32)
    attributes {funct7 = 0 : i32, opcode = 43 : i32} {

    %0 = aps.readrf %arg0 : i32 -> i32
    %1 = aps.readrf %arg1 : i32 -> i32
    %2 = memref.get_global @mem_a : memref<16xi32>
    %3 = memref.get_global @mem_b : memref<16xi32>

    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    aps.memburstload %0, %2[%c0], %c16 : i32, memref<16xi32>, i32, i32
    aps.memburstload %1, %3[%c0], %c16 : i32, memref<16xi32>, i32, i32

    %c0_idx = arith.constant 0 : index
    %c16_idx = arith.constant 16 : index
    %c1_idx = arith.constant 1 : index
    scf.for %i = %c0_idx to %c16_idx step %c1_idx {
      %5 = memref.get_global @mem_a : memref<16xi32>
      %6 = aps.memload %5[%i] : memref<16xi32>, i32 -> i32
      %7 = memref.get_global @mem_b : memref<16xi32>
      %8 = aps.memload %7[%i] : memref<16xi32>, i32 -> i32
      %9 = arith.addi %6, %8 : i32
      aps.memstore %9, %5[%i] : i32, memref<16xi32>, i32
    }

    aps.memburststore %2[%c0], %0, %c16 : memref<16xi32>, i32, i32, i32
    %c42 = arith.constant 42 : i32
    aps.writerf %arg2, %c42 : i32, i32

    return
  }
}
```

### 处理

```python
extractor = APSPatternExtractor()
match_func = extractor.extract_compute_core(aps_func)
```

### 输出（匹配 Pattern）

```mlir
func.func @flow_burst_add_match(
  %mem_a: memref<16xi32>,
  %mem_b: memref<16xi32>
) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %c16 step %c1 {
    %a = memref.load %mem_a[%i] : memref<16xi32>
    %b = memref.load %mem_b[%i] : memref<16xi32>
    %sum = arith.addi %a, %b : i32
    memref.store %sum, %mem_a[%i] : memref<16xi32>
  }

  return
}
```

---

## 集成到 Megg

### 修改 `match_rewrites.py`

```python
# python/megg/rewrites/match_rewrites.py

from megg.rewrites.aps_pattern_extractor import APSPatternExtractor

def build_ruleset_from_module(module: MModule):
    """构建 ruleset，支持 APS pattern"""

    extractor = APSPatternExtractor()
    rewrites = []
    skeletons = []

    for func_op in module.get_functions():
        instr_name = _instruction_name(func_op)

        try:
            # 🆕 检测是否为 APS pattern
            if _is_aps_pattern(func_op):
                print(f"Detected APS pattern: {instr_name}")

                # 🆕 自动提取计算核心
                match_func = extractor.extract_compute_core(func_op)
                print(f"  Extracted match pattern: {match_func.name}")

                # 保存原始 APS func（用于后端）
                original_aps_func = func_op
            else:
                # 标准 pattern（无需处理）
                match_func = func_op
                original_aps_func = None

            # 使用匹配 pattern 构建 skeleton
            skeleton, simple_pattern = _build_skeleton_from_func(match_func)

            # 保存原始 APS func（如果有）
            if skeleton and original_aps_func:
                skeleton.original_aps_func = original_aps_func

            # 生成 rewrite rules
            if simple_pattern:
                pattern, result_type, arg_vars = simple_pattern
                custom_instr = Term.custom_instr(
                    egglog.String(instr_name),
                    egglog.Vec(*arg_vars),
                    result_type
                )
                rewrite = egglog.rewrite(pattern).to(custom_instr)
                rewrites.append(rewrite)

            elif skeleton:
                # 生成 component rewrites
                for full_name, pattern in skeleton.leaf_patterns.items():
                    # ... (现有逻辑)
                    pass

                skeletons.append(skeleton)

        except Exception as e:
            print(f"Warning: Failed to process {instr_name}: {e}")
            continue

    ruleset = egglog.ruleset(*rewrites, name="match_rewrite") if rewrites else egglog.ruleset(name="match_rewrite")

    return ruleset, skeletons

def _is_aps_pattern(func: MOperation) -> bool:
    """检测函数是否包含 APS dialect 操作"""
    for block in func.get_blocks():
        for op in block.operations:
            if op.name.startswith('aps.'):
                return True
    return False
```

---

## 完整流程

```
┌────────────────────────────────────────────────────────────┐
│ 输入: APS MLIR Pattern                                      │
└────────────────────────────────────────────────────────────┘
func.func @flow_burst_add(%arg0: i32, %arg1: i32, %arg2: i32) {
  aps.readrf ...
  aps.memburstload ...
  scf.for { aps.memload, arith.addi, aps.memstore }
  aps.memburststore ...
  aps.writerf ...
}
            ↓
┌────────────────────────────────────────────────────────────┐
│ 自动提取计算核心                                            │
└────────────────────────────────────────────────────────────┘
extractor = APSPatternExtractor()
match_func = extractor.extract_compute_core(aps_func)
            ↓
┌────────────────────────────────────────────────────────────┐
│ 生成匹配 Pattern                                            │
└────────────────────────────────────────────────────────────┘
func.func @flow_burst_add_match(%mem_a, %mem_b) {
  scf.for { memref.load, arith.addi, memref.store }
}
            ↓
┌────────────────────────────────────────────────────────────┐
│ 构建 Skeleton + Rewrite Rules（现有代码）                  │
└────────────────────────────────────────────────────────────┘
skeleton = _build_skeleton_from_func(match_func)
skeleton.original_aps_func = aps_func  # 保存原始
            ↓
┌────────────────────────────────────────────────────────────┐
│ Pattern Matching（现有代码）                                │
└────────────────────────────────────────────────────────────┘
matches = megg_egraph.match_skeleton(skeleton)
            ↓
┌────────────────────────────────────────────────────────────┐
│ 输出: 标记 + 原始 APS Func                                  │
└────────────────────────────────────────────────────────────┘
"megg.custom_instr" {
  instr_name = "flow_burst_add",
  aps_definition = <保存的原始 APS MLIR>
}
```

---

## 总结

### 核心思想

**自动从 APS MLIR 中提取计算核心，用于 pattern matching**

### 三个关键步骤

1. **识别计算 block**（包含控制流和算术的 block）
2. **移除硬件层**（aps.readrf, aps.memburstload 等）
3. **抽象内存操作**（aps.memload → memref.load）

### 最终效果

- ✅ 输入：APS MLIR（完整硬件实现）
- ✅ 自动生成：匹配 pattern（只有计算逻辑）
- ✅ 匹配：在用户代码中找到
- ✅ 保留：原始 APS MLIR（用于代码生成或参考）

---

## 实现优先级

我建议先实现**简化版本**：

```python
class SimpleAPSExtractor:
    """简化版：只提取 scf.for 内部的操作"""

    def extract_compute_core(self, aps_func):
        # 1. 找到所有 scf.for
        for_loops = find_ops(aps_func, 'scf.for')

        # 2. 复制 for 循环的 body
        compute_blocks = [loop.body for loop in for_loops]

        # 3. 替换 aps.memload/memstore
        for block in compute_blocks:
            replace_ops(block, {
                'aps.memload': 'memref.load',
                'aps.memstore': 'memref.store'
            })

        # 4. 创建新函数
        return create_match_function(compute_blocks)
```

这个版本只需要 100-200 行代码，可以快速验证思路！

---

需要我开始实现吗？
