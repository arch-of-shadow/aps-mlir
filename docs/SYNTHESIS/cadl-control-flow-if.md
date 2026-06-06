# CADL Control-Flow If

This note defines the CADL frontend behavior for statement-level `if`.
Expression-level `if` already exists and remains a value-selection expression
lowered to `arith.select`.

## Syntax

CADL has two separate `if` forms.

Expression-level `if` is used where an expression is expected:

```cadl
let x: u32 = if cond { a } else { b };
```

Statement-level `if` is used where a statement is expected:

```cadl
if cond {
    stmt*
} else {
    stmt*
}
```

The `else` block is optional:

```cadl
if cond {
    stmt*
}
```

Statement-level `if` does not use a trailing semicolon. A branch body is a
statement block, not an expression body. This keeps value selection and control
flow distinct in both AST and MLIR lowering.

## AST

Statement-level `if` is represented as a statement:

```python
IfStmt(
    condition: Expr,
    then_body: list[Stmt],
    else_body: list[Stmt] | None,
)
```

It is intentionally separate from `IfExpr`:

```python
IfExpr(
    condition: Expr,
    then_branch: Expr,
    else_branch: Expr,
)
```

## Lowering

`IfExpr` continues to lower to `arith.select`.

`IfStmt` lowers to `scf.if`. Branch side effects, such as `_irf`, `_mem`, or
`_csr` writes, stay inside their branch region.

For ordinary SSA variables edited inside a branch and used after the `if`, the
lowering creates `scf.if` results. This is the structured-control-flow
equivalent of a phi node.

Example:

```cadl
let x: u32 = a;

if cond {
    x = a + 1;
} else {
    x = b + 1;
}

_irf[rd] = x;
```

Expected MLIR shape:

```mlir
%x1 = scf.if %cond -> (i32) {
  %then = arith.addi %a, %c1 : i32
  scf.yield %then : i32
} else {
  %else = arith.addi %b, %c1 : i32
  scf.yield %else : i32
}
aps.writerf ..., %x1
```

If only one branch edits an outer variable, the other branch yields the original
value:

```cadl
if cond {
    x = x + 1;
}
```

Expected MLIR shape:

```mlir
%x1 = scf.if %cond -> (i32) {
  %then = arith.addi %x, %c1 : i32
  scf.yield %then : i32
} else {
  scf.yield %x : i32
}
```

## Merge Rules

The first implementation supports conservative merge candidates:

- The assignment target is an identifier.
- The identifier existed in the parent scope before the `if`.
- The existing binding is a regular SSA value, not a global symbol or constant.
- A branch-local `let` does not escape the branch.
- Index assignments and range-slice assignments are side effects, not merge
  candidates.

The merge set is collected before constructing `scf.if`, because SCF result
types must be known at operation construction time.

After `scf.if`, the parent scope rebinds each merged variable to the
corresponding `scf.if` result.

## Unsupported For Now

The following are intentionally not part of the first implementation:

- Branch-declared variable merge, such as declaring `let x` in both branches
  and using `x` after the `if`.
- Merging globals, constants, memories, or register-file aliases.
- `break`, `continue`, and early `return` inside `if`.
- Type-changing assignment across branches.

