# 定制指令的正确编译架构

## 核心原则

**前端关心"做什么"（算法语义），后端关心"怎么做"（硬件实现）。**

编译器的每一层都应该 **Lower**（细化），而不是 **Raise**（抽象化）。

---

## 完整的编译流程

```
┌────────────────────────────────────────────────────────────────────┐
│                         正确的分层架构                              │
└────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════╗
║ Layer 0: 用户代码（应用层）                                      ║
╚══════════════════════════════════════════════════════════════════╝

// user.c - 用户的应用代码
void process_data(int* data_a, int* data_b, int len) {
    for (int i = 0; i < len; i++) {
        data_a[i] = data_a[i] + data_b[i];
    }
}

           ↓ (Clang frontend)

╔══════════════════════════════════════════════════════════════════╗
║ Layer 1: 标准 MLIR (算法层)                                      ║
╚══════════════════════════════════════════════════════════════════╝

// user.mlir - 标准 MLIR IR
func.func @process_data(%a: memref<?xi32>, %b: memref<?xi32>, %len: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %len step %c1 {
    %val_a = memref.load %a[%i] : memref<?xi32>
    %val_b = memref.load %b[%i] : memref<?xi32>
    %sum = arith.addi %val_a, %val_b : i32
    memref.store %sum, %a[%i] : memref<?xi32>
  }

  return
}

           ↓ (Megg pattern matching)

╔══════════════════════════════════════════════════════════════════╗
║ Layer 2: 标记后的 MLIR (带定制指令标记)                          ║
╚══════════════════════════════════════════════════════════════════╝

func.func @process_data(%a: memref<?xi32>, %b: memref<?xi32>, %len: index) {
  // Pattern matched! 标记为定制指令
  %matched = "megg.custom_instr"(%a, %b, %len) {
    instr_name = "vector_add_16"
  } : (memref<?xi32>, memref<?xi32>, index) -> ()

  return
}

           ↓ (Lower to LLVM)

╔══════════════════════════════════════════════════════════════════╗
║ Layer 3: LLVM IR (调用接口)                                      ║
╚══════════════════════════════════════════════════════════════════╝

define void @process_data(i32* %a, i32* %b, i64 %len) {
  ; 调用自定义指令 intrinsic 或 inline asm
  call void @llvm.aps.vector_add_16(i32* %a, i32* %b, i64 %len)
  ret void
}

           ↓ (APS Backend lowering)

╔══════════════════════════════════════════════════════════════════╗
║ Layer 4: APS MLIR (硬件实现层)                                   ║
╚══════════════════════════════════════════════════════════════════╝

// 硬件后端生成（不应该由前端处理！）
func.func @llvm.aps.vector_add_16(%a: i32*, %b: i32*, %len: i64) {
  // 硬件资源管理
  %mem_a = aps.memdeclare : memref<16xi32>
  %mem_b = aps.memdeclare : memref<16xi32>

  // 寄存器文件接口
  %rs1 = ... // 从调用约定获取
  %addr_a = aps.readrf %rs1 : i5 -> i64

  // DMA 传输
  aps.memburstload %addr_a, %mem_a[%c0], %c16 : ...
  aps.memburstload %addr_b, %mem_b[%c0], %c16 : ...

  // 计算核心（和 Layer 1 的算法一致）
  scf.for %i = %c0 to %c16 step %c1 : i32 {
    %a_val = aps.memload %mem_a[%i] : ...
    %b_val = aps.memload %mem_b[%i] : ...
    %sum = arith.addi %a_val, %b_val : i32
    aps.memstore %sum, %mem_a[%i] : ...
  }

  // DMA 写回
  aps.memburststore %mem_a[%c0], %addr_a, %c16 : ...

  // 寄存器写回
  aps.writerf %rd, %result : ...

  return
}

           ↓ (CADL backend codegen)

╔══════════════════════════════════════════════════════════════════╗
║ Layer 5: CADL (硬件描述)                                         ║
╚══════════════════════════════════════════════════════════════════╝

rtype vector_add_16(rs1: u5, rs2: u5, rd: u5) {
    let addr_a: u64 = _irf[rs1];
    let addr_b: u64 = _irf[rs2];

    static mem_a: [u32; 16];
    static mem_b: [u32; 16];

    mem_a[0 +: ] = _burst_read[addr_a +: 16];
    mem_b[0 +: ] = _burst_read[addr_b +: 16];

    with i: u32 = (0, i_) do {
        let a: u32 = mem_a[i];
        let b: u32 = mem_b[i];
        mem_a[i] = a + b;
        let i_: u32 = i + 1;
    } while (i_ < 16);

    _burst_write[addr_a +: 16] = mem_a[0 +: ];
    _irf[rd] = 0;
}
```

---

## 关键设计决策

### 决策 1: Pattern 定义使用 C 语义

**定制指令的 pattern 应该是纯算法描述，不包含硬件细节。**

```c
// patterns/vector_add.c - 定制指令的 C 语义
void vector_add_16(int* a, int* b) {
    for (int i = 0; i < 16; i++) {
        a[i] = a[i] + b[i];
    }
}
```

**编译为标准 MLIR**:
```bash
clang -emit-mlir patterns/vector_add.c -o patterns/vector_add.mlir
```

**生成的 pattern**:
```mlir
// patterns/vector_add.mlir
func.func @vector_add_16(%a: memref<16xi32>, %b: memref<16xi32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %c16 step %c1 {
    %val_a = memref.load %a[%i] : memref<16xi32>
    %val_b = memref.load %b[%i] : memref<16xi32>
    %sum = arith.addi %val_a, %val_b : i32
    memref.store %sum, %a[%i] : memref<16xi32>
  }

  return
}
```

**优点**:
- ✅ 纯算法描述，易于理解和维护
- ✅ 与用户代码使用相同的 IR（都是标准 MLIR）
- ✅ 前端（Megg）只需要关心算法模式
- ✅ 硬件细节完全交给后端

### 决策 2: 硬件实现由后端独立处理

**硬件实现不是 Megg 的职责！**

```
前端 (Megg):
  输入: 标准 MLIR (user code + C 语义 patterns)
  处理: Pattern matching + E-graph optimization
  输出: 标记了定制指令的 MLIR 或 LLVM IR

后端 (APS Backend):
  输入: LLVM IR with custom instruction markers
  处理:
    1. 资源分配 (寄存器、内存)
    2. 调用约定 (参数传递)
    3. DMA 调度 (burst load/store)
    4. 生成 APS MLIR / CADL
  输出: CADL 硬件描述 / 二进制
```

**硬件实现模板** (由后端维护):
```python
# aps_backend/templates/vector_add_16.py
class VectorAdd16Template:
    """硬件实现模板（后端专用）"""

    def __init__(self):
        self.name = "vector_add_16"
        self.opcode = 43
        self.memory_size = 16  # elements

    def generate_aps_mlir(self, call_context):
        """生成 APS MLIR 实现"""
        return f"""
        func.func @{self.name}_impl(%rs1: i5, %rs2: i5, %rd: i5) {{
          // 硬件资源分配
          %mem_a = aps.memdeclare : memref<{self.memory_size}xi32>
          %mem_b = aps.memdeclare : memref<{self.memory_size}xi32>

          // 获取 CPU 地址
          %addr_a = aps.readrf %rs1 : i5 -> i64
          %addr_b = aps.readrf %rs2 : i5 -> i64

          // DMA burst load
          aps.memburstload %addr_a, %mem_a[%c0], %c{self.memory_size} : ...
          aps.memburstload %addr_b, %mem_b[%c0], %c{self.memory_size} : ...

          // 计算核心（和 C 语义一致）
          scf.for %i = %c0 to %c{self.memory_size} step %c1 : i32 {{
            %a = aps.memload %mem_a[%i] : ...
            %b = aps.memload %mem_b[%i] : ...
            %sum = arith.addi %a, %b : i32
            aps.memstore %sum, %mem_a[%i] : ...
          }}

          // DMA burst store
          aps.memburststore %mem_a[%c0], %addr_a, %c{self.memory_size} : ...

          // 写回结果
          aps.writerf %rd, %c0 : i5, i32

          return
        }}
        """

    def generate_cadl(self):
        """生成 CADL 代码"""
        # ... 后端逻辑 ...
```

### 决策 3: Megg 只负责标记，不生成硬件代码

**Megg 的输出**: 带标记的 MLIR 或 LLVM IR

**选项 A: 输出标记的 MLIR**
```mlir
func.func @my_function(%a: memref<16xi32>, %b: memref<16xi32>) {
  // Megg 插入的标记
  %0 = "megg.custom_instr"(%a, %b) {
    instr_name = "vector_add_16",
    operands = ["memref<16xi32>", "memref<16xi32>"]
  } : (memref<16xi32>, memref<16xi32>) -> ()

  return
}
```

**选项 B: 输出 LLVM IR with intrinsic**
```llvm
define void @my_function(i32* %a, i32* %b) {
  ; Megg 生成的 intrinsic 调用
  call void @llvm.aps.custom.vector_add_16(i32* %a, i32* %b)
  ret void
}
```

**后端识别标记并生成实现**:
```python
# aps_backend/lower.py
def lower_custom_instruction(instr_marker):
    """将 Megg 的标记转换为硬件实现"""
    instr_name = instr_marker.get_attr("instr_name")

    # 查找硬件模板
    template = get_hardware_template(instr_name)

    # 生成 APS MLIR
    aps_mlir = template.generate_aps_mlir(instr_marker.context)

    # 或直接生成 CADL
    cadl_code = template.generate_cadl()

    return aps_mlir  # or cadl_code
```

---

## 完整工作流程

### 步骤 1: 定义定制指令（C 语义）

```c
// patterns/my_instructions.c

// 定制指令 1: 向量加法
void vector_add_16(int* a, int* b) {
    for (int i = 0; i < 16; i++) {
        a[i] = a[i] + b[i];
    }
}

// 定制指令 2: 向量点积
int dot_product_16(int* a, int* b) {
    int sum = 0;
    for (int i = 0; i < 16; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

**编译为 MLIR pattern**:
```bash
clang -O2 -emit-mlir patterns/my_instructions.c \
  -o patterns/my_instructions.mlir
```

### 步骤 2: Megg Pattern Matching

```bash
# 输入: 用户代码 (标准 MLIR)
# Pattern: C 语义定义 (标准 MLIR)
./megg-opt user_code.mlir \
  --custom-instructions patterns/my_instructions.mlir \
  --output-format mlir \  # 或 llvm
  -o optimized.mlir
```

**Megg 内部流程**:
```python
# python/megg/compiler.py
class Compiler:
    def schedule(self, custom_instructions: str):
        # 1. 加载 C 语义 patterns (标准 MLIR)
        pattern_module = load_mlir(custom_instructions)

        # 2. 从 patterns 构建 ruleset + skeletons
        #    (现有逻辑完全可用！)
        ruleset, skeletons = build_ruleset_from_module(pattern_module)

        # 3. E-graph optimization + pattern matching
        #    (现有逻辑完全可用！)
        self.egraph.run(ruleset)

        # 4. 提取优化后的 terms
        optimized_terms = self.extract_best_terms()

        # 5. 检测匹配的定制指令
        for skeleton in skeletons:
            matches = self.megg_egraph.match_skeleton(skeleton)
            for match in matches:
                # 🆕 标记为定制指令（而不是生成硬件代码）
                self._mark_custom_instruction(match, skeleton.instr_name)

        # 6. 转换回 MLIR（带标记）
        optimized_mlir = terms_to_func(optimized_terms)

        return optimized_mlir

    def _mark_custom_instruction(self, match, instr_name):
        """在匹配的区域插入定制指令标记"""
        # 创建 custom_instr marker
        marker = create_custom_instr_op(
            name=instr_name,
            operands=match.operands,
            result_type=match.result_type
        )

        # 替换原始 term
        self.egraph.replace(match.root_term, marker)
```

**输出** (带标记的 MLIR):
```mlir
func.func @my_function(%a: memref<16xi32>, %b: memref<16xi32>) {
  // Megg 识别的定制指令标记
  "megg.custom_instr"(%a, %b) {
    instr_name = "vector_add_16"
  } : (memref<16xi32>, memref<16xi32>) -> ()

  return
}
```

### 步骤 3: 后端 Lowering（独立工具链）

```bash
# APS 后端处理 Megg 的输出
./aps-backend optimized.mlir \
  --templates templates/ \  # 硬件实现模板
  --output-format cadl \
  -o output.cadl
```

**后端处理**:
```python
# aps_backend/main.py
def compile_to_aps(mlir_file, templates_dir):
    module = load_mlir(mlir_file)

    for func in module.get_functions():
        for op in func.operations:
            if op.name == "megg.custom_instr":
                instr_name = op.get_attr("instr_name")

                # 加载硬件模板
                template = load_template(templates_dir, instr_name)

                # 生成 APS MLIR 实现
                aps_impl = template.generate_aps_mlir(op.operands)

                # 替换标记
                replace_op(op, aps_impl)

    # 继续 lowering 到 CADL
    cadl_code = lower_to_cadl(module)

    return cadl_code
```

**最终输出** (CADL):
```cadl
rtype vector_add_16(rs1: u5, rs2: u5, rd: u5) {
    let addr_a: u64 = _irf[rs1];
    let addr_b: u64 = _irf[rs2];

    static mem_a: [u32; 16];
    static mem_b: [u32; 16];

    // ... (完整的硬件实现)
}
```

---

## 与现有 Megg 架构的兼容性

**好消息：现有的 Megg pattern matching 完全可用！**

### 现有功能可以直接复用

1. **Pattern extraction** (`match_rewrites.py`)
   - ✅ 输入从 APS MLIR 改为 C 语义 MLIR
   - ✅ 逻辑完全不变（都是标准 dialect）

2. **Skeleton matching** (`megg_egraph.py`)
   - ✅ 完全不需要修改
   - ✅ 匹配的是计算模式，不是硬件操作

3. **E-graph optimization**
   - ✅ 完全不需要修改
   - ✅ 优化的是算法逻辑

### 需要新增的功能

1. **Custom instruction marker** (`terms_to_func.py`)
   ```python
   # 新增: 生成定制指令标记
   def _term_to_operation(self, term: Term) -> MOperation:
       if term.head == "custom_instr":
           # 生成 megg.custom_instr operation
           return self._create_custom_instr_marker(term)
       # ... 现有逻辑 ...
   ```

2. **LLVM IR output** (可选)
   ```python
   # 新增: 输出 LLVM IR instead of MLIR
   class LLVMBackend:
       def emit(self, module: MModule):
           # 转换 MLIR → LLVM IR
           # 保留 custom_instr markers
   ```

---

## 职责划分清单

### Megg (前端优化器)

**负责**:
- ✅ 加载 C 语义 patterns (标准 MLIR)
- ✅ Pattern matching (识别算法模式)
- ✅ E-graph optimization (等价变换)
- ✅ 插入定制指令标记
- ✅ 输出带标记的 MLIR/LLVM IR

**不负责**:
- ❌ 硬件资源分配
- ❌ DMA 调度
- ❌ 寄存器分配
- ❌ 生成 APS MLIR / CADL

### APS Backend (后端编译器)

**负责**:
- ✅ 识别定制指令标记
- ✅ 加载硬件实现模板
- ✅ 资源分配 (内存、寄存器)
- ✅ 调用约定处理
- ✅ 生成 APS MLIR / CADL
- ✅ 硬件代码优化

**不负责**:
- ❌ 算法模式识别
- ❌ E-graph 优化

---

## 总结

### 你的观点（完全正确）

1. **编译器应该分层 Lower，而不是 Raise**
   - ✅ C 语义 → MLIR → LLVM → 硬件
   - ❌ 硬件 MLIR → 抽象 → 匹配

2. **前端关心算法，后端关心实现**
   - ✅ Megg: 识别 "向量加法" 模式
   - ✅ 后端: 实现 "DMA + 寄存器 + burst"

3. **硬件细节不应该在前端出现**
   - ✅ Pattern 定义: 纯 C 语义
   - ✅ 硬件实现: 后端模板

### 正确的实现路径

```
1. 用 C 定义定制指令语义
   └─> 编译为标准 MLIR pattern

2. Megg 在标准 MLIR 上做 pattern matching
   └─> 输出带标记的 MLIR/LLVM IR

3. APS 后端处理标记
   └─> 生成 APS MLIR / CADL 实现
```

### 下一步行动

我建议：
1. **放弃之前的 APS 抽象方案**（逆向思维，不符合编译原则）
2. **采用 C 语义 pattern 定义**（算法描述，易于维护）
3. **Megg 只负责标记**（职责清晰，架构简洁）
4. **后端独立处理硬件实现**（分离关注点）

你觉得这样对吗？需要我帮你实现新的方案吗？