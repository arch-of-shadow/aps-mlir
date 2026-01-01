# 正确方案：在 aps-mlir 中写 Pass 转换为标准 MLIR

## 你的正确理解

**在 aps-mlir 项目中实现一个 MLIR Pass，将 APS dialect 转换为标准 MLIR（软件语义）**

```
aps-mlir 项目:
┌─────────────────────────────────────────────────────────────┐
│ APS Dialect → Standard MLIR Pass                            │
└─────────────────────────────────────────────────────────────┘

CADL 定义
    ↓ (CADL 编译器)
APS MLIR (硬件语义)
    ↓ (新的 Pass: APS → Standard)
标准 MLIR (软件语义)
    ↓ (传给 Megg)
Megg Pattern Matching
```

---

## 转换规则

### 规则 1: 硬件 I/O → 移除（或转为注释）

```mlir
// 原始 APS MLIR
%0 = aps.readrf %arg0 : i32 -> i32

// 转换后（软件语义）
// 移除（因为软件层面不关心寄存器读取）
// 或者保留为注释：
// @hardware: aps.readrf %arg0
```

### 规则 2: Burst Load → For 循环

```mlir
// 原始 APS MLIR
%mem = aps.memdeclare : memref<16xi32>
aps.memburstload %cpu_addr, %mem[%c0], %c16
  : i64, memref<16xi32>, i32, i32

// 转换后（软件语义）
%mem = memref.alloc() : memref<16xi32>
%c0 = arith.constant 0 : index
%c16 = arith.constant 16 : index
%c1 = arith.constant 1 : index
scf.for %i = %c0 to %c16 step %c1 {
  %addr = arith.addi %cpu_addr, %i : i64
  %val = memref.load %cpu_mem[%addr] : memref<?xi32>
  memref.store %val, %mem[%i] : memref<16xi32>
}
```

**重要**：Burst 需要展开为循环，因为：
- 软件语义中没有 "burst" 的概念
- 用户代码可能手写了同样的循环模式

### 规则 3: Burst Store → For 循环

```mlir
// 原始 APS MLIR
aps.memburststore %mem[%c0], %cpu_addr, %c16
  : memref<16xi32>, i32, i64, i32

// 转换后（软件语义）
%c0 = arith.constant 0 : index
%c16 = arith.constant 16 : index
%c1 = arith.constant 1 : index
scf.for %i = %c0 to %c16 step %c1 {
  %val = memref.load %mem[%i] : memref<16xi32>
  %addr = arith.addi %cpu_addr, %i : i64
  memref.store %val, %cpu_mem[%addr] : memref<?xi32>
}
```

### 规则 4: APS 内存操作 → 标准内存操作

```mlir
// 原始 APS MLIR
%val = aps.memload %mem[%i] : memref<16xi32>, i32 -> i32

// 转换后（软件语义）
%val = memref.load %mem[%i] : memref<16xi32>
```

```mlir
// 原始 APS MLIR
aps.memstore %val, %mem[%i] : i32, memref<16xi32>, i32

// 转换后（软件语义）
memref.store %val, %mem[%i] : memref<16xi32>
```

### 规则 5: 硬件输出 → 移除

```mlir
// 原始 APS MLIR
aps.writerf %rd, %result : i5, i32

// 转换后（软件语义）
// 移除（或作为 return 值）
return %result : i32
```

---

## 完整示例

### 输入（APS MLIR）

```mlir
module {
  memref.global @mem_a : memref<16xi32> {impl = "1rw"}
  memref.global @mem_b : memref<16xi32>

  func.func @flow_burst_add(%arg0: i32, %arg1: i32, %arg2: i32)
    attributes {funct7 = 0 : i32, opcode = 43 : i32} {

    // 硬件接口
    %0 = aps.readrf %arg0 : i32 -> i32
    %1 = aps.readrf %arg1 : i32 -> i32

    // 硬件资源
    %2 = memref.get_global @mem_a : memref<16xi32>
    %3 = memref.get_global @mem_b : memref<16xi32>

    // Burst load
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    aps.memburstload %0, %2[%c0_i32], %c16_i32
      : i32, memref<16xi32>, i32, i32
    aps.memburstload %1, %3[%c0_i32], %c16_i32
      : i32, memref<16xi32>, i32, i32

    // 计算核心
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c16 step %c1 : i32 {
      %5 = memref.get_global @mem_a : memref<16xi32>
      %6 = aps.memload %5[%i] : memref<16xi32>, i32 -> i32
      %7 = memref.get_global @mem_b : memref<16xi32>
      %8 = aps.memload %7[%i] : memref<16xi32>, i32 -> i32
      %9 = arith.addi %6, %8 : i32
      aps.memstore %9, %5[%i] : i32, memref<16xi32>, i32
    }

    // Burst store
    aps.memburststore %2[%c0_i32], %0, %c16_i32
      : memref<16xi32>, i32, i32, i32

    // 硬件输出
    %c42_i32 = arith.constant 42 : i32
    aps.writerf %arg2, %c42_i32 : i32, i32

    return
  }
}
```

### 输出（标准 MLIR - 软件语义）

```mlir
module {
  func.func @flow_burst_add_software(
    %cpu_mem_a: memref<?xi32>,  // CPU 内存地址空间
    %cpu_mem_b: memref<?xi32>,  // CPU 内存地址空间
    %addr_a: index,              // 数组 A 的起始地址
    %addr_b: index               // 数组 B 的起始地址
  ) -> i32 {

    // 分配局部内存（模拟 APS 片上存储）
    %mem_a = memref.alloc() : memref<16xi32>
    %mem_b = memref.alloc() : memref<16xi32>

    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index

    // Burst load → For 循环（模拟 DMA）
    scf.for %i = %c0 to %c16 step %c1 {
      %src_idx = arith.addi %addr_a, %i : index
      %val_a = memref.load %cpu_mem_a[%src_idx] : memref<?xi32>
      memref.store %val_a, %mem_a[%i] : memref<16xi32>

      %src_idx_b = arith.addi %addr_b, %i : index
      %val_b = memref.load %cpu_mem_b[%src_idx_b] : memref<?xi32>
      memref.store %val_b, %mem_b[%i] : memref<16xi32>
    }

    // 计算核心（保持不变，只替换 aps.memload/memstore）
    scf.for %i = %c0 to %c16 step %c1 {
      %a = memref.load %mem_a[%i] : memref<16xi32>
      %b = memref.load %mem_b[%i] : memref<16xi32>
      %sum = arith.addi %a, %b : i32
      memref.store %sum, %mem_a[%i] : memref<16xi32>
    }

    // Burst store → For 循环
    scf.for %i = %c0 to %c16 step %c1 {
      %val = memref.load %mem_a[%i] : memref<16xi32>
      %dst_idx = arith.addi %addr_a, %i : index
      memref.store %val, %cpu_mem_a[%dst_idx] : memref<?xi32>
    }

    // 释放局部内存
    memref.dealloc %mem_a : memref<16xi32>
    memref.dealloc %mem_b : memref<16xi32>

    // 返回值（模拟 writerf）
    %result = arith.constant 42 : i32
    return %result : i32
  }
}
```

---

## 在 aps-mlir 中实现 Pass

### 实现位置

```
aps-mlir/lib/APS/Transforms/APSToStandard.cpp
```

### Pass 框架

```cpp
// aps-mlir/lib/APS/Transforms/APSToStandard.cpp

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "APS/APSOps.h"

namespace {

class APSToStandardPass : public PassWrapper<APSToStandardPass,
                                              OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    // 遍历所有函数
    module.walk([&](func::FuncOp func) {
      convertFunction(func, builder);
    });
  }

private:
  void convertFunction(func::FuncOp func, OpBuilder &builder) {
    // 转换函数体中的操作
    func.walk([&](Operation *op) {
      if (auto burstLoad = dyn_cast<aps::MemBurstLoadOp>(op)) {
        convertBurstLoad(burstLoad, builder);
      } else if (auto burstStore = dyn_cast<aps::MemBurstStoreOp>(op)) {
        convertBurstStore(burstStore, builder);
      } else if (auto memLoad = dyn_cast<aps::MemLoadOp>(op)) {
        convertMemLoad(memLoad, builder);
      } else if (auto memStore = dyn_cast<aps::MemStoreOp>(op)) {
        convertMemStore(memStore, builder);
      } else if (auto readrf = dyn_cast<aps::CpuRfReadOp>(op)) {
        // 移除或转为注释
        removeOp(readrf);
      } else if (auto writerf = dyn_cast<aps::CpuRfWriteOp>(op)) {
        // 移除或转为注释
        removeOp(writerf);
      }
    });
  }

  void convertBurstLoad(aps::MemBurstLoadOp burstLoad, OpBuilder &builder) {
    /*
    原始:
      aps.memburstload %cpu_addr, %mem[%start], %length
        : i64, memref<16xi32>, i32, i32

    转换为:
      scf.for %i = %c0 to %length step %c1 {
        %src_idx = arith.addi %cpu_addr, %i
        %val = memref.load %cpu_mem[%src_idx]
        %dst_idx = arith.addi %start, %i
        memref.store %val, %mem[%dst_idx]
      }
    */

    builder.setInsertionPoint(burstLoad);
    Location loc = burstLoad.getLoc();

    Value cpuAddr = burstLoad.getCpuAddr();
    Value memref = burstLoad.getMemref();
    Value start = burstLoad.getStart();
    Value length = burstLoad.getLength();

    // 创建循环常量
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);

    // 转换 length 为 index 类型
    Value lengthIndex = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), length);

    // 创建 for 循环
    auto forOp = builder.create<scf::ForOp>(loc, c0, lengthIndex, c1);

    // 在循环体中生成 load/store
    builder.setInsertionPointToStart(forOp.getBody());
    Value i = forOp.getInductionVar();

    // src_idx = cpu_addr + i
    Value srcIdx = builder.create<arith::AddIOp>(loc, cpuAddr, i);

    // 假设有一个全局的 CPU 内存空间
    // %val = memref.load %cpu_mem[%src_idx]
    // 这里需要根据实际情况处理

    // dst_idx = start + i
    Value dstIdx = builder.create<arith::AddIOp>(loc, start, i);

    // memref.store %val, %mem[%dst_idx]
    // ... (具体实现)

    // 移除原始操作
    burstLoad.erase();
  }

  void convertBurstStore(aps::MemBurstStoreOp burstStore, OpBuilder &builder) {
    // 类似 convertBurstLoad
  }

  void convertMemLoad(aps::MemLoadOp apsLoad, OpBuilder &builder) {
    /*
    原始:
      %val = aps.memload %mem[%i] : memref<16xi32>, i32 -> i32

    转换为:
      %val = memref.load %mem[%i] : memref<16xi32>
    */

    builder.setInsertionPoint(apsLoad);
    Location loc = apsLoad.getLoc();

    Value memref = apsLoad.getMemref();
    ValueRange indices = apsLoad.getIndices();
    Type resultType = apsLoad.getResult().getType();

    // 创建标准 memref.load
    auto stdLoad = builder.create<memref::LoadOp>(
        loc, resultType, memref, indices);

    // 替换使用
    apsLoad.getResult().replaceAllUsesWith(stdLoad.getResult());
    apsLoad.erase();
  }

  void convertMemStore(aps::MemStoreOp apsStore, OpBuilder &builder) {
    /*
    原始:
      aps.memstore %val, %mem[%i] : i32, memref<16xi32>, i32

    转换为:
      memref.store %val, %mem[%i] : memref<16xi32>
    */

    builder.setInsertionPoint(apsStore);
    Location loc = apsStore.getLoc();

    Value value = apsStore.getValue();
    Value memref = apsStore.getMemref();
    ValueRange indices = apsStore.getIndices();

    // 创建标准 memref.store
    builder.create<memref::StoreOp>(loc, value, memref, indices);

    apsStore.erase();
  }

  void removeOp(Operation *op) {
    // 移除操作（或转为注释 attribute）
    op->erase();
  }
};

} // namespace

std::unique_ptr<Pass> createAPSToStandardPass() {
  return std::make_unique<APSToStandardPass>();
}
```

### 注册 Pass

```cpp
// aps-mlir/lib/APS/Transforms/Passes.cpp

void registerAPSPasses() {
  PassRegistration<APSToStandardPass>(
      "aps-to-standard",
      "Convert APS dialect to standard MLIR dialects");
}
```

---

## 使用流程

### 步骤 1: 在 aps-mlir 中运行 Pass

```bash
# 在 aps-mlir 项目中
/home/cloud/aps-mlir/build/bin/aps-mlir-opt \
  flow_burst_add.mlir \
  --aps-to-standard \
  -o flow_burst_add_software.mlir
```

### 步骤 2: 传给 Megg 做 Pattern Matching

```bash
# 在 megg 项目中
./megg-opt user_code.mlir \
  --custom-instructions /path/to/flow_burst_add_software.mlir \
  -o optimized.mlir
```

**现在可以匹配了！** 因为：
- Pattern: 标准 MLIR（for 循环 + memref.load/store）
- 用户代码: 标准 MLIR（for 循环 + memref.load/store）

---

## 关键设计决策

### 问题 1: Burst 展开后，模式会变复杂

**原始 Pattern（带 burst）**：
```mlir
aps.memburstload ...
scf.for { 计算 }
aps.memburststore ...
```

**展开后（标准 MLIR）**：
```mlir
scf.for { burst load }   ← 循环 1
scf.for { 计算 }         ← 循环 2
scf.for { burst store }  ← 循环 3
```

**用户代码可能只写了计算循环**：
```mlir
scf.for { 计算 }  ← 只有这个
```

**解决方案**：Pattern 应该只包含**计算核心**，不包含 burst 循环。

### 方案 A: Pass 生成两个版本

```cpp
class APSToStandardPass {
  void convertFunction(func::FuncOp func) {
    // 生成两个函数：

    // 1. 完整版本（包含 burst）
    func::FuncOp fullVersion = cloneFunction(func);
    expandAllAPSOps(fullVersion);  // 展开所有 APS 操作

    // 2. 计算核心版本（只保留计算，用于匹配）
    func::FuncOp matchVersion = cloneFunction(func);
    extractComputeCore(matchVersion);  // 只保留计算循环

    // 重命名
    fullVersion.setName(func.getName() + "_full");
    matchVersion.setName(func.getName() + "_match");
  }

  void extractComputeCore(func::FuncOp func) {
    // 移除 burst load/store
    // 保留计算循环
    // 替换 aps.memload/memstore → memref.load/store
  }
};
```

**输出两个函数**：
```mlir
// 完整版本（如果需要参考）
func.func @flow_burst_add_full(...) {
  scf.for { burst load }
  scf.for { 计算 }
  scf.for { burst store }
}

// 匹配版本（传给 Megg）
func.func @flow_burst_add_match(%mem_a, %mem_b) {
  scf.for { 计算 }
}
```

### 方案 B: 用 Attribute 标记

```mlir
func.func @flow_burst_add_match(%mem_a, %mem_b)
  attributes {
    original_aps_function = "flow_burst_add",
    is_compute_core = true
  } {
  scf.for { 计算 }
}
```

---

## 总结

### 正确的职责划分

```
aps-mlir 项目:
  - 定义 APS dialect
  - 实现 APS → Standard Pass
  - 输出: 标准 MLIR（软件语义）

Megg 项目:
  - 接收: 标准 MLIR pattern
  - Pattern matching
  - 输出: 匹配结果 + 标记

后端:
  - 读取匹配结果
  - 使用原始 CADL 定义生成硬件代码
```

### 关键点

1. **Burst 需要展开为循环**（你说得对！）
   - 软件层面用循环表示 burst
   - 用户代码可能手写了同样的循环

2. **生成匹配 Pattern 和完整 Pattern**
   - 匹配 Pattern: 只有计算核心
   - 完整 Pattern: 包含 burst（如果需要参考）

3. **Megg 完全不需要理解 APS**
   - 只处理标准 MLIR
   - 现有代码可用

---

## 下一步

我建议：
1. 在 aps-mlir 中实现 `APSToStandardPass`
2. 生成两个版本：完整版 + 匹配版
3. 测试转换结果

需要我帮你写具体的 C++ 实现吗？
