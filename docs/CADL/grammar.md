# CADL Grammar Reference

Complete syntax specification for the Computer Architecture Description Language.

## Table of Contents

- [Lexical Structure](#lexical-structure)
- [Keywords](#keywords)
- [Operators](#operators)
- [Syntax Rules](#syntax-rules)
- [Grammar Production Rules](#grammar-production-rules)

---

## Lexical Structure

### Identifiers

```
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
```

Identifiers start with a letter or underscore, followed by any combination of letters, digits, or underscores.

**Examples:**
- `rs1`, `rd`, `my_variable`
- `_irf`, `_mem`
- `x0`, `sum_`

### Type Identifiers

```
VARTYPE: /[fui]([1-9][0-9]*)c*|usize/
```

Type identifiers specify bit widths for numeric types:

| Pattern | Type | Examples |
|---------|------|----------|
| `u{N}` | Unsigned N-bit integer | `u1`, `u5`, `u32`, `u64` |
| `i{N}` | Signed N-bit integer | `i8`, `i16`, `i32` |
| `f{N}` | Float (32 or 64) | `f32`, `f64` |
| `usize` | Architecture size | `usize` |

### Number Literals

#### With Explicit Width

```
NUMBER_LIT:
  | /{width}'[sS]?{format}{digits}(_[ui]{override_width})?/
```

**Format specifiers:**

| Format | Prefix | Base | Example | Value | Type |
|--------|--------|------|---------|-------|------|
| Binary | `b` or `B` | 2 | `5'b101010` | 42 | `u5` |
| Octal | `o` or `O` | 8 | `3'o7` | 7 | `u3` |
| Decimal | `d` or `D` | 10 | `15'd123` | 123 | `u15` |
| Hexadecimal | `h` or `H` | 16 | `8'hFF` | 255 | `u8` |

**Signed literals:** Add `s` or `S` prefix after the format specifier:
```cadl
8'sHFF    // Signed 8-bit hex: -1
16'sd100  // Signed 16-bit decimal: 100
```

#### Without Explicit Width (Default: u32)

```cadl
42           // Decimal: u32
0x1234       // Hexadecimal: u32
0b1010       // Binary: u32
0o755        // Octal: u32
```

#### Floating Point

```
NUMBER_LIT: /-?[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?(_[f]{width})?/
```

**Examples:**
```cadl
3.14159         // f64 (default)
2.5e-3          // f64 scientific notation
1.0_f32         // Explicit f32
```

### String Literals

```
STRING_LIT: /"[^"]*"/
```

Simple double-quoted strings (no escape sequences in current grammar):
```cadl
"hello world"
"error: invalid input"
```

### Comments

```
COMMENT:
  | /\/\/[^\n]*/                           // Line comment
  | /\/\*[^*]*\*+(?:[^\/*][^*]*\*+)*\//   // Block comment
```

**Examples:**
```cadl
// This is a line comment

/* This is a
   block comment */
```

---

## Keywords

CADL reserves the following keywords:

### Control Flow
- `if`, `else` - Conditional expressions
- `while` - Loop condition
- `with`, `do` - Loop state threading
- `return` - Return statement
- `break`, `continue` - Loop control (reserved)

### Declarations
- `static` - Global state variable
- `let` - Variable binding
- `flow` - Flow definition
- `rtype` - R-type instruction definition
- `regfile` - Register file declaration

### Types
- `Instance` - Instance type for flow instantiation

### Special Constructs
- `sel` - Select expression (conditional multiplexer)
- `spawn` - Parallel execution
- `in` - Reserved for iteration

### Literals
- `true`, `false` - Boolean literals

---

## Operators

### Arithmetic Operators

| Operator | Name | Precedence | Associativity | Example |
|----------|------|------------|---------------|---------|
| `*` | Multiply | 7 | Left | `a * b` |
| `/` | Divide | 7 | Left | `a / b` |
| `%` | Remainder | 7 | Left | `a % b` |
| `+` | Add | 6 | Left | `a + b` |
| `-` | Subtract | 6 | Left | `a - b` |

### Bitwise Operators

| Operator | Name | Precedence | Associativity | Example |
|----------|------|------------|---------------|---------|
| `~` | Bitwise NOT | 8 (unary) | Right | `~a` |
| `<<` | Left Shift | 5 | Left | `a << 2` |
| `>>` | Right Shift | 5 | Left | `a >> 2` |
| `&` | Bitwise AND | 4 (also 3*) | Left | `a & b` |
| `^` | Bitwise XOR | 3 | Left | `a ^ b` |
| `|` | Bitwise OR | 2 | Left | `a | b` |

*Note: `&` appears in both bitwise_and_expr (precedence 4) and is also used as logical AND

### Comparison Operators

| Operator | Name | Precedence | Associativity | Example |
|----------|------|------------|---------------|---------|
| `==` | Equal | 4 | Left | `a == b` |
| `!=` | Not Equal | 4 | Left | `a != b` |
| `<` | Less Than | 4 | Left | `a < b` |
| `<=` | Less or Equal | 4 | Left | `a <= b` |
| `>` | Greater Than | 4 | Left | `a > b` |
| `>=` | Greater or Equal | 4 | Left | `a >= b` |

### Logical Operators

| Operator | Name | Precedence | Associativity | Example |
|----------|------|------------|---------------|---------|
| `!` | Logical NOT | 8 (unary) | Right | `!condition` |
| `&&` | Logical AND | 3* | Left | `a && b` |
| `||` | Logical OR | 1 | Left | `a || b` |

### Unary Operators

| Operator | Name | Example |
|----------|------|---------|
| `-` | Negation | `-x` |
| `!` | Logical NOT | `!flag` |
| `~` | Bitwise NOT | `~bits` |

### Type Cast Operators (not supported yet!)

| Operator | Description | Example |
|----------|-------------|---------|
| `$signed(expr)` | Cast to signed | `$signed(x)` |
| `$unsigned(expr)` | Cast to unsigned | `$unsigned(x)` |
| `$f32(expr)` | Cast to f32 | `$f32(x)` |
| `$f64(expr)` | Cast to f64 | `$f64(x)` |
| `$int(expr)` | Cast to integer | `$int(x)` |
| `$uint(expr)` | Cast to unsigned int | `$uint(x)` |

### Punctuation and Delimiters

| Symbol | Name | Usage |
|--------|------|-------|
| `=` | Assignment | `x = 5` |
| `:` | Type annotation | `x: u32` |
| `;` | Statement terminator | `let x = 5;` |
| `,` | Separator | `(a, b, c)` |
| `{` `}` | Braces | Block, aggregate |
| `(` `)` | Parentheses | Grouping, calls |
| `[` `]` | Brackets | Indexing, arrays |
| `[[` `]]` | Double brackets | Directives |
| `->` | Arrow | (Reserved) |
| `=>` | Fat arrow | (Reserved) |
| `@` | At sign | (Reserved) |
| `#` | Hash | Attributes |

---

## Syntax Rules

### Operator Precedence (Lowest to Highest)

1. **Logical OR** (`||`)
2. **Bitwise OR** (`|`)
3. **Bitwise XOR** (`^`)
4. **Equality & Relational** (`==`, `!=`, `<`, `<=`, `>`, `>=`)
5. **Bitwise/Logical AND** (`&`, `&&`)
6. **Shift** (`<<`, `>>`)
7. **Additive** (`+`, `-`)
8. **Multiplicative** (`*`, `/`, `%`)
9. **Unary** (`-`, `!`, `~`)
10. **Postfix** (indexing `[]`, slicing `[:]`, calls `()`)

### Expression Hierarchy (from grammar.lark)

```
expr
  └─ logical_or_expr (||)
       └─ bitwise_or_expr (|)
            └─ bitwise_xor_expr (^)
                 └─ eq_expr (==, !=)
                      └─ bitwise_and_expr (&, &&)
                           └─ rel_expr (<, <=, >, >=)
                                └─ shift_expr (<<, >>)
                                     └─ add_expr (+, -)
                                          └─ mul_expr (*, /, %)
                                               └─ unary_expr (-, !, ~)
                                                    └─ postfix_expr ([], [start:end], ())
                                                         └─ primary_expr
```

---

## Grammar Production Rules

### Top-Level Structure

```ebnf
proc ::= proc_part*

proc_part ::= regfile
            | flow
            | static
```

A CADL program consists of zero or more processor parts: register files, flows, and static variables.

### Data Types

```ebnf
data_type ::= VARTYPE                                    // u32, i16, etc.
            | '[' VARTYPE (';' NUMBER_LIT)* ']'          // [u32; 8]
            | 'Instance'                                  // Instance type

compound_type ::= data_type
```

### Static Variables

```ebnf
static ::= attribute* 'static' IDENTIFIER ':' data_type ('=' expr)? ';'
```

**Example:**
```cadl
#[impl("regs")]
static counter: u32 = 0;
```

### Register Files (DO NOT USE)

```ebnf
regfile ::= 'regfile' IDENTIFIER '(' NUMBER_LIT ',' NUMBER_LIT ')' ';'
```

Declares a register file with width and depth:
```cadl
regfile my_regs(32, 16);  // 16 registers of 32 bits each
```

### Flow Definitions

```ebnf
flow ::= attribute* 'flow' IDENTIFIER '(' fn_arg_list? ')' body
       | attribute* 'rtype' IDENTIFIER '(' fn_arg_list? ')' body

fn_arg_list ::= fn_arg (',' fn_arg)*
fn_arg ::= IDENTIFIER ':' compound_type

body ::= ';'                    // Empty body
       | '{' stmt* '}'          // Block body
```

**Example:**
```cadl
#[opcode(7'b0101011)]
rtype add_instr(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  _irf[rd] = r1 + _irf[rs2];
}
```

### Attributes

```ebnf
attribute ::= '#' '[' IDENTIFIER ']'                           // Simple
            | '#' '[' IDENTIFIER '(' expr ')' ']'              // With parameter
```

**Examples:**
```cadl
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
#[impl("regs")]
```

### Statements

```ebnf
stmt ::= expr ';'                                              // Expression statement
       | 'let'? expr (':' data_type)? '=' expr ';'            // Assignment
       | 'return' expr_list ';'                               // Return
       | '[' expr ']' ':' stmt                                 // Guard
       | 'with' with_binding* 'do' body 'while' expr ';'      // Do-while
       | '[[' IDENTIFIER ('(' expr ')')? ']]'                 // Directive
       | 'spawn' '{' stmt* '}'                                // Spawn
       | static                                                // Static declaration

with_binding ::= IDENTIFIER ':' VARTYPE '=' '(' expr? ',' expr? ')'

expr_list ::= expr (',' expr)*
```

### Expressions

```ebnf
// Primary expressions
primary_expr ::= '(' expr_list ')'                             // Tuple/grouping
               | NUMBER_LIT                                    // Number literal
               | 'true' | 'false'                             // Boolean
               | STRING_LIT                                    // String
               | IDENTIFIER                                    // Variable reference
               | IDENTIFIER '(' expr_list? ')'                // Function call
               | cast_op '(' expr ')'                         // Type cast
               | 'if' expr '{' expr '}' 'else' '{' expr '}'  // If expression
               | 'sel' '{' sel_arm+ '}'                       // Select expression
               | '{' expr_list '}'                            // Aggregate

sel_arm ::= expr ':' expr ','

cast_op ::= '$signed' | '$unsigned' | '$f32' | '$f64' | '$int' | '$uint'

// Postfix expressions
postfix_expr ::= postfix_expr '[' expr_list ']'                // Indexing
               | postfix_expr '[' add_expr ':' add_expr ']'    // Slicing
               | postfix_expr '[' add_expr '+:' add_expr? ']'  // Range slice
               | primary_expr

// Binary expressions (simplified, see precedence above)
expr ::= expr binary_op expr
       | unary_op expr
       | postfix_expr
       | primary_expr
```

---

## Special Constructs

### Indexing and Slicing

```cadl
array[0]           // Single index
array[i, j]        // Multi-dimensional index
array[7:0]         // Slice from bit 7 to bit 0
array[start +: 8]  // Range slice: 8 elements starting at 'start'
```

### If Expression

```cadl
let result: u32 = if condition {
    true_value
} else {
    false_value
};
```

### Select Expression (Not supported yet!)

```cadl
sel {
    cond1: value1,
    cond2: value2,
    cond3: value3,
}
```

Similar to a priority multiplexer - evaluates conditions in order and returns the first matching value.

### Do-While with State Threading

```cadl
with
  i: u32 = (i0, i_)
  sum: u32 = (sum0, sum_)
do {
  let sum_: u32 = sum + 4;
  let i_: u32 = i + 1;
} while (i < n);
```

The `with` clause defines loop-carried state: `(initial_value, next_value)`.

### Directives

```cadl
[[unroll]]           // Directive without parameter
[[pragma(2)]]        // Directive with parameter
```

Compiler hints for optimization or synthesis.

### Spawn Statement (DO NOT USE)

```cadl
spawn {
  stmt1;
  stmt2;
  stmt3;
}
```

Indicates statements can execute in parallel (hardware threads).

---

## Reserved Hardware Constructs

### Register File Access

```cadl
_irf[rs1]       // Read from integer register file
_irf[rd] = x;   // Write to integer register file
```

### Memory Access

```cadl
_mem[addr]           // Read from memory
_mem[addr] = value;  // Write to memory
```

---

## Grammar File Location

The complete Lark grammar is defined in:
```
cadl_frontend/grammar.lark
```

---

## See Also

- [Type System](type-system.md) - Detailed type information
- [Expressions](expressions.md) - Expression semantics
- [Statements](statements.md) - Statement details
- [Flows](flows.md) - Flow and rtype definitions
- [Examples](examples.md) - Working code examples

---

**Note**: This grammar reference is based on the Lark LALR parser implementation and matches the production rules in `grammar.lark`.
