# CADL Statements

This document describes the statement constructs in CADL, including variable declarations, assignments, control flow, and hardware-specific directives.

## Statement Categories

### Expression Statements

Execute an expression and discard its result:

```cadl
expr;
```

**Examples:**
```cadl
foo(x, y);              // Call a function
_irf[rd] = result;      // Assignment to register
_mem[addr] = value;     // Memory write
```

### Assignment Statements

Assign values to variables or memory locations.

#### Basic Assignment

```cadl
lvalue = expression;
```

**Examples:**
```cadl
x = 42;
result = a + b;
_irf[rd] = sum;
_mem[addr] = data;
```

#### Let Binding (Variable Declaration)

Declare and optionally initialize a new variable:

```cadl
let identifier: type = expression;
let identifier = expression;  // Type inferred
```

**Examples:**
```cadl
let r1: u32 = _irf[rs1];       // Explicit type
let sum: u32 = a + b + c;      // Explicit type
let x = 42;                     // Type inferred as u32
let mask = 8'hFF;               // Type inferred as u8
```

**Type Annotation (Optional):**
```cadl
// With type annotation
x: u32 = 100;

// Without 'let' keyword (re-assignment)
x = 200;
```

### Control Flow Statements

#### Guard Statements (DO NOT USE)

Conditionally execute a statement based on a guard condition:

```cadl
[condition]: statement
```

**Examples:**
```cadl
[x > 0]: result = positive_val;
[enable]: _mem[addr] = data;
[count < limit]: count = count + 1;
```

Guards are hardware-friendly conditional execution - the statement executes only when the condition is true.

#### Do-While Loops

Hardware-oriented loop with explicit state threading:

```cadl
with
  var1: type = (init1, next1)
  var2: type = (init2, next2)
  ...
do {
  // Loop body
} while condition;
```

**Components:**
- **with bindings**: State variables with initial and next values
- **do block**: Loop body
- **while condition**: Loop continuation condition

**Example - Simple Counter:**
```cadl
let sum0: u32 = 0;
let i0: u32 = 0;
let n0: u32 = 10;

with
  i: u32 = (i0, i_)
  sum: u32 = (sum0, sum_)
  n: u32 = (n0, n_)
do {
  let sum_: u32 = sum + 4;
  let i_: u32 = i + 1;
  let n_: u32 = n;
} while (i_ < n);
```

**Key Concepts:**
- Each `with` binding has an **initial value** (e.g., `i0`) and a **next value** (e.g., `i_`)
- Inside the loop body, compute the next values (suffixed with `_`)
- The loop continues while the condition is true
- This explicit threading makes dataflow clear for hardware synthesis

**Example - Array Processing:**
```cadl
let s0: u32 = _irf[rs1];  // Source address
let d0: u32 = _irf[rs2];  // Destination address
let it0: u32 = 0;
let n0: u32 = 8;

with
  s: u32 = (s0, s_)
  d: u32 = (d0, d_)
  it: u32 = (it0, it_)
  n: u32 = (n0, n_)
do {
  let mem1: u32 = _mem[s];
  let mem2: u32 = _mem[s + 4];
  _mem[d] = mem1 + mem2;
  let s_: u32 = s + 8;
  let d_: u32 = d + 4;
  let it_: u32 = it + 1;
  let n_: u32 = n;
} while (it < n);
```

#### Return Statements (DO NOT USE)

Return values from a flow:

```cadl
return expression;
return (expr1, expr2, ...);  // Multiple values
```

**Examples:**
```cadl
return result;
return (sum, carry);
return ();  // Empty return
```

Note: In `rtype` flows, the return is implicit through register file writes.

### Directive Statements

Compiler hints and optimization directives:

```cadl
[[directive_name]]
[[directive_name(expression)]]
```

**Examples:**
```cadl
[[unroll]]              // Unroll loop
[[pipeline(2)]]         // Pipeline with initiation interval 2
[[inline]]              // Inline function
```

Directives provide metadata to guide hardware synthesis and optimization.

### Spawn Statements (DO NOT USE)

Parallel execution blocks:

```cadl
spawn {
  statement1;
  statement2;
  ...
}
```

**Example:**
```cadl
spawn {
  _mem[addr1] = val1;
  _mem[addr2] = val2;
  _mem[addr3] = val3;
}
```

Statements inside `spawn` blocks execute in parallel, representing concurrent hardware operations.

### Static Declarations

Declare static (persistent) variables:

```cadl
static identifier: type;
static identifier: type = initializer;
```

**Examples:**
```cadl
static counter: u32 = 0;
static addr: u32;
static thetas: [u32; 8] = {1474560, 870484, 459940, 233473, 117189, 58652, 29333, 14667};
```

**Attributes on Statics:**
```cadl
#[impl("regs")]
static thetas: [u32; 8] = {...};

#[impl("1rw")]
static mem_a: [u32; 16];
```

Static variables:
- Retain their value across flow invocations
- Map to hardware registers or memory
- Can be initialized with constants
- Can have implementation hints via attributes

## Statement Blocks

Group multiple statements:

```cadl
{
  statement1;
  statement2;
  statement3;
}
```

Blocks are used in:
- Flow bodies
- Do-while loop bodies
- Spawn blocks

## Complete Examples

### Memory Accumulation

```cadl
rtype accum(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let a : u32 = _mem[r1];
  let b : u32 = _mem[r1 + 4];
  let c : u32 = _mem[r1 + 8];
  let d : u32 = _mem[r1 + 12];
  let rst: u32 = a + b + c + d;
  _mem[r1 + 16] = rst;
  _irf[rd] = rst;
}
```

### CORDIC Algorithm

```cadl
rtype cordic(rs1: u5, rs2: u5, rd: u5) {
    let x0 : u32 = 19898;
    let y0 : u32 = 0;
    let z0 : u32 = _irf[rs1];
    let n0 : u32 = 8;
    let it0: u32 = 0;

    with
      it: u32 = (it0, it_)
      x: u32 = (x0, x_)
      y: u32 = (y0, y_)
      z: u32 = (z0, z_)
      n: u32 = (n0, n_)
    do {
      let z_neg: u1  = z[31:31];
      let theta: u32 = thetas[it];
      let x_shift: u32 = x >> it;
      let y_shift: u32 = y >> it;
      let x_ : u32 = if z_neg {x + y_shift} else {x - y_shift};
      let y_ : u32 = if z_neg {y - x_shift} else {y + x_shift};
      let z_ : u32 = if z_neg {z + theta} else {z - theta};
      let it_: u32 = it + 1;
      let n_ : u32 = n;
    } while (it < n);

    _irf[rd] = y;
}
```

### Stateful Processing

```cadl
static st: u32 = 0;

rtype state_priority(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let mr1: u32 = _mem[r1];
  _irf[rd] = mr1 + st;
  st = st + 1;  // Modify static state
}
```

### Auto-Incrementing Address

```cadl
static addr: u32 = 0;

rtype autoinc(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  addr = if r2 == 0 {addr + 4} else {r1};
  _irf[rd] = _mem[addr];
}
```

### Burst Memory Operations

```cadl
#[impl("1rw")]
static mem_a: [u32; 16];
static mem_b: [u32; 16];

rtype burst_add(rs1: u5, rs2: u5, rd: u5) {
    let r1: u32 = _irf[rs1];
    let r2: u32 = _irf[rs2];

    // Burst read from memory to local arrays
    mem_a[0 +: ] = _burst_read[r1 +: 16];
    mem_b[0 +: ] = _burst_read[r2 +: 16];

    // Process locally
    with i: u32 = (0, i_) do {
        let a: u32 = mem_a[i];
        let b: u32 = mem_b[i];
        let c: u32 = a + b;
        mem_a[i] = c;
        let i_: u32 = i + 1;
    } while (i_ < 16);

    // Burst write back to memory
    _burst_write[r1 +: 16] = mem_a[0 +: ];
    _irf[rd] = 42;
}
```

## Control Flow Patterns

### Loop Iteration

**Fixed iteration count:**
```cadl
with i: u32 = (0, i_) do {
  // ... loop body ...
  let i_: u32 = i + 1;
} while (i < 10);
```

**Variable iteration count:**
```cadl
let n: u32 = _irf[rs2];
with i: u32 = (0, i_) do {
  // ... loop body ...
  let i_: u32 = i + 1;
} while (i < n);
```

### Conditional State Update

```cadl
let new_state: u32 = if condition {
  state + increment
} else {
  state
};
```

### Pipeline Stage Guards

```cadl
[stage1_valid]: stage1_data = input;
[stage2_valid]: stage2_data = process(stage1_data);
[stage3_valid]: output = stage2_data;
```

## Statement Best Practices

1. **Declare Variables Close to Use**: Keep declarations near where variables are used
   ```cadl
   let addr: u32 = base + offset;
   let data: u32 = _mem[addr];  // Use immediately
   ```

2. **Explicit Types for Hardware State**: Always specify types for static variables
   ```cadl
   static counter: u32 = 0;  // Good
   ```

3. **Name Loop Variables Consistently**: Use clear naming for state threading
   ```cadl
   with i: u32 = (i0, i_) do {  // '_' suffix for next value
     let i_: u32 = i + 1;
   } while (i < n);
   ```

4. **Thread All Loop State**: Include all state variables in `with` bindings
   ```cadl
   with
     i: u32 = (i0, i_)
     sum: u32 = (sum0, sum_)  // Don't forget accumulator
   do { ... }
   ```

5. **Use Guards for Conditional Hardware**: Prefer guards over `if` for conditional execution
   ```cadl
   [enable]: _mem[addr] = data;  // Hardware-friendly
   ```

## See Also

- [Expressions](expressions.md) - Expression syntax and operators
- [Flows](flows.md) - Flow definitions and structure
- [Type System](type-system.md) - Data types
- [Examples](examples.md) - Complete working examples
