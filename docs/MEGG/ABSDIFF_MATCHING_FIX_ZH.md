# Absdiff 模式匹配修复总结

## 问题描述

`tests/benchmarks/absdiff/absdiff.mlir` 无法匹配 `tests/benchmarks/absdiff/instr.mlir` 模式。

## 根本原因

**操作数顺序提取错误**。

### 问题细节

Pattern文件 (`instr.mlir`):
```mlir
%0 = arith.cmpi uge, %arg0, %arg1 : i32
%3 = scf.if %0 -> (i32) {
  %1 = arith.subi %arg0, %arg1 : i32  // then: arg0 - arg1
  scf.yield %1 : i32
} else {
  %2 = arith.subi %arg1, %arg0 : i32  // else: arg1 - arg0
  scf.yield %2 : i32
}
```

旧的约束生成逻辑:
- Then分支: `Term.sub(arg0, arg1)` → 提取为 `[arg0, arg1]` ✅
- Else分支: `Term.sub(arg1, arg0)` → 错误地提取为 `[arg0, arg1]` ❌

原因: `add_leaf_pattern` 方法中，当直接操作数不在 `arg_var_set` 时（如 `Term.sub(...)`），会通过正则表达式在 pattern 字符串中搜索变量名，但搜索顺序是按照 `self.arg_vars` 的声明顺序（总是 `[arg0, arg1]`），而不是按照变量在 pattern 中出现的顺序。

Code 实际值:
- Then分支提取: `['Term-0', 'Term-1']` (arg0, arg1)
- Else分支提取: `['Term-1', 'Term-0']` (arg1, arg0)

约束验证失败:
```
1. flow_absdiff_then_stmt1[0] = Term-0 → 绑定 arg0 = Term-0 ✓
2. flow_absdiff_then_stmt1[1] = Term-1 → 绑定 arg1 = Term-1 ✓
3. flow_absdiff_else_stmt2[0] = Term-1 → 检查 arg0 == Term-1? NO! (arg0已绑定到Term-0) ✗
```

## 修复方案

修改 `python/megg/rewrites/match_rewrites.py` 中的 `_extract_arg_vars_from_pattern` 方法：

```python
def _extract_arg_vars_from_pattern(self, pattern: Term) -> List[Term]:
    """
    Extract argument variables from pattern Term in the correct order.
    Parses the pattern string representation and extracts variables
    in the order they appear in the pattern.
    """
    result = []
    pattern_str = str(pattern)

    # Create a mapping from variable names to variable objects
    var_name_to_var: Dict[str, Term] = {}
    for var in self.arg_vars:
        var_name = _get_var_name(var)
        var_name_to_var[var_name] = var

    # Find all variable occurrences in order using regex
    var_pattern = r'\b(' + '|'.join(re.escape(_get_var_name(v)) for v in self.arg_vars) + r')\b'
    matches = re.finditer(var_pattern, pattern_str)

    seen = set()
    for match in matches:
        var_name = match.group(1)
        if var_name not in seen:
            seen.add(var_name)
            result.append(var_name_to_var[var_name])

    return result
```

### 关键改进

- 使用 `re.finditer` 按照变量在 pattern 字符串中**首次出现的顺序**提取
- 维护 `seen` 集合避免重复
- 保持操作数的正确顺序

## 验证结果

修复后：

```
leaf_operands:
  flow_absdiff_then_stmt1: [arg0, arg1]  ✅
  flow_absdiff_else_stmt2: [arg1, arg0]  ✅

operand_constraints:
  ('flow_absdiff_then_stmt1', 0, 'arg0')  ✅
  ('flow_absdiff_then_stmt1', 1, 'arg1')  ✅
  ('flow_absdiff_else_stmt2', 0, 'arg1')  ✅
  ('flow_absdiff_else_stmt2', 1, 'arg0')  ✅

匹配结果: 1 total matches ✅
```

输出文件 (`outputs/absdiff_opt_both.mlir`):
```mlir
func.func @absdiff(%arg0: i32, %arg1: i32) -> i32 {
  %0 = llvm.inline_asm "flow_absdiff", "=r,r,r" %arg0, %arg1 : (i32, i32) -> i32
  return %0 : i32
}
```

## 影响范围

此修复影响所有包含**不同分支使用相同参数但顺序不同**的控制流模式匹配。

### 受益场景

1. 条件分支中操作数顺序不同 (如 absdiff)
2. 循环体中参数使用顺序变化
3. 任何需要精确匹配操作数顺序的复杂模式

## 测试

```bash
pixi run python megg-opt.py tests/benchmarks/absdiff/absdiff.mlir \
  --custom-instructions tests/benchmarks/absdiff/instr.mlir \
  -o /tmp/absdiff_output.mlir

# 验证输出包含 llvm.inline_asm "flow_absdiff"
grep "flow_absdiff" /tmp/absdiff_output.mlir
```

## 相关文件

- 修复: `python/megg/rewrites/match_rewrites.py` (line 112-140)
- 测试: `tests/benchmarks/absdiff/`
  - `absdiff.mlir` - 待优化代码
  - `instr.mlir` - 自定义指令模式
