# CADL Documentation

**Computer Architecture Description Language (CADL)** - A hardware description language for specifying custom RISC-V instruction extensions and hardware accelerator behavior.

## Overview

CADL is designed for:
- **Custom RISC-V Instructions**: Define R-type and other instruction formats with custom opcodes and funct7 codes
- **Hardware Accelerators**: Describe dataflow-oriented hardware behavior with precise bit-width control
- **Hardware Synthesis**: Generate synthesizable hardware from high-level descriptions
- **APS-MLIR Integration**: Lower CADL code to MLIR for optimization and code generation

## Quick Start

```cadl
// Define a custom RISC-V instruction
#[opcode(7'b0101011)]
#[funct7(7'b0000000)]
rtype add_custom(rs1: u5, rs2: u5, rd: u5) {
  let r1: u32 = _irf[rs1];
  let r2: u32 = _irf[rs2];
  _irf[rd] = r1 + r2;
}
```

## Documentation Structure

### Core Language Reference

1. **[Grammar Reference](grammar.md)**
   - Complete syntax specification
   - Keywords and operators
   - Lexical structure

2. **[Type System](type-system.md)**
   - BasicType: `u32`, `i16`, `f32`, etc.
   - DataType: arrays and compound types
   - Literal system with width inference

3. **[Expressions](expressions.md)**
   - Operators and precedence
   - Function calls and indexing
   - Conditionals and select expressions
   - Type casts

4. **[Statements](statements.md)**
   - Variable declarations
   - Assignments
   - Control flow (loops, guards)
   - Directives and spawn

5. **[Flows](flows.md)**
   - Flow vs. rtype definitions
   - Attributes and annotations
   - RISC-V integration
   - Register and memory access

### Examples and Patterns

6. **[Examples](examples.md)**
   - Complete working examples
   - Common patterns
   - Best practices

### MLIR Integration

7. **[APS Dialect](aps-dialect.md)**
   - MLIR operations
   - Memory model
   - Register file access
   - Burst transfers

## Key Features

### Width-Aware Literals

CADL supports precise bit-width specification for numeric literals:

```cadl
5'b101010    // 5-bit binary literal (value 42, type u5)
8'hFF        // 8-bit hexadecimal literal (value 255, type u8)
32'd1000     // 32-bit decimal literal (value 1000, type u32)
```

### Hardware-Specific Constructs

- **Register File Access**: `_irf[rs1]` - Read from CPU register file
- **Memory Access**: `_mem[addr]` - Hardware memory operations
- **Static Variables**: `static x: u32 = 0;` - Hardware state
- **Do-While Loops**: Hardware-friendly loop constructs with explicit state threading

### Attributes System

Annotate flows with hardware-specific metadata:

```cadl
#[opcode(7'b0101011)]    // RISC-V opcode
#[funct7(7'b0000000)]    // RISC-V funct7
#[impl("regs")]          // Implementation hint
rtype my_instruction(...) { ... }
```

## Language Philosophy

CADL is designed with several key principles:

1. **Explicit Bit Widths**: All types have explicit bit widths for precise hardware generation
2. **Dataflow Semantics**: Focus on data movement and transformation
3. **Hardware Realism**: Constructs map directly to hardware structures
4. **Type Safety**: Strong static typing prevents common hardware errors
5. **RISC-V Integration**: First-class support for custom RISC-V extensions

## Parser Implementation

CADL is implemented using:
- **Lark Parser**: LALR parser for Python
- **Grammar File**: `cadl_frontend/grammar.lark`
- **AST Classes**: `cadl_frontend/ast.py` (matches Rust IR structures)
- **Type System**: Aligned with `type_sys_ir.rs` from the Rust implementation

## Related Projects

- **cadl_rust**: Original Rust implementation using LALRPOP
- **APS-MLIR**: MLIR compiler infrastructure with TOR and Schedule dialects
- **CIRCT**: Circuit IR Compilers and Tools (MLIR foundation)

## Getting Started

### Parse a CADL file

```bash
# Parse and display AST
pixi run parse examples/simple.cadl

# Parse with summary
pixi run parse-summary examples/zyy.cadl
```

### Python API

```python
from cadl_frontend import parse_proc

with open("example.cadl", "r") as f:
    source = f.read()

ast = parse_proc(source, "example.cadl")
print(f"Found {len(ast.flows)} flows")
```

### Run Tests

```bash
# All tests
pixi run pytest tests/ -v

# Specific test suites
pytest tests/test_parser.py -v          # Basic parser tests
pytest tests/test_zyy_examples.py -v    # Real-world examples
pytest tests/test_literal_widths.py -v  # Number literal parsing
```

## Language Status

### Working Features
✅ Number literals with width specification
✅ All basic types (signed, unsigned, float)
✅ Arrays and compound types
✅ Expression parsing with correct precedence
✅ Flow and rtype definitions
✅ Attributes system
✅ Static variables
✅ Control flow (loops, conditionals)
✅ Register and memory access

### In Development
🚧 Full MLIR lowering pipeline
🚧 Advanced optimizations
🚧 Expanded type system features

## Contributing

When contributing to CADL:
- Follow existing AST structure patterns
- Add tests for new features
- Update grammar documentation
- Ensure type system consistency

## Further Reading

- [Grammar Reference](grammar.md) - Complete syntax specification
- [Type System](type-system.md) - Detailed type information
- [Examples](examples.md) - Working code examples
- [APS Dialect](aps-dialect.md) - MLIR integration

---

**Project**: APS-MLIR
**License**: See repository root
**Contact**: See CLAUDE.md for project structure
