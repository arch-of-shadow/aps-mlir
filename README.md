# CADL Frontend (Python)

A Python implementation of the CADL (Computer Architecture Description Language) parser and MLIR compiler using Lark.

This project provides a complete toolchain from CADL source code to optimized MLIR IR, with intelligent loop transformations and directive support.

## Features

### Parser
- Full CADL language parsing using Lark LALR parser
- AST generation with type-aware literals
- Width-specified number literals (e.g., `5'b101010` → `u5` type)
- Support for flows, functions, loops, directives, and more

### MLIR Compiler
- Direct lowering from CADL AST to MLIR operations
- Integration with CIRCT dialects (SCF, Comb, HW, APS)
- Supports control flow, memory operations, and function calls

### Loop Transformation
- **Automatic loop optimization**: Raises `do-while` loops to `scf.for` when pattern-detectable
- **Pattern detection**: Recognizes affine loops with constant bounds and steps
- **Loop-carried variables**: Full support via `iter_args` mechanism
- **Directive support**: Applies `[[unroll(N)]]` and other directives as MLIR attributes
- **Fallback to `scf.while`**: Handles general loops that don't match patterns

## Installation

Using pixi (recommended):
```bash
pixi install
```

Using pip:
```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

### Parse CADL to AST
```python
from cadl_frontend import parse_proc

with open("example.cadl", "r") as f:
    source = f.read()

ast = parse_proc(source, "example.cadl")
print(f"Flows: {len(ast.flows)}, Functions: {len(ast.functions)}")
```

### Convert CADL to MLIR
```bash
pixi run mlir examples/simple.cadl
pixi run mlir examples/raise_while_to_for.cadl
```

Or programmatically:
```python
from cadl_frontend import parse_proc
from cadl_frontend.mlir_converter import CADLMLIRConverter

ast = parse_proc(source, filename)
converter = CADLMLIRConverter()
mlir_module = converter.convert_proc(ast)
print(mlir_module)
```

## Loop Transformation Examples

### Pattern 1: Binding Expression Increment
```cadl
with i: u8 = (0, i + 1), crc: u8 = (0, crc_)
do {
    let crc_: u8 = crc ^ (data >> i);
} while i + 1 < 8;
```
Output: `Loop -> scf.for: i=0..8 step 1, iter_args=[crc]`

### Pattern 2: With Directive
```cadl
[[unroll(4)]]
with i: u8 = (0, i + 1), crc: u8 = (0, crc_)
do { ... } while i + 1 < 8;
```
Generates: `scf.for ... { ... } {unroll = 4 : i32}`

### Pattern 3: Backward Loop
```cadl
let i0: u8 = 8;
with i: u8 = (i0, i - 1), crc: u8 = (0, crc_)
do { ... } while i > 0;
```
Output: `Loop -> scf.for: i=8..0 step -1, iter_args=[crc]`

## Development

Run tests:
```bash
pixi run pytest tests/ -v
# OR
make test
```

Test specific suites:
```bash
pixi run pytest tests/test_parser.py -v
pixi run pytest tests/test_zyy_examples.py -v
```

Format code:
```bash
make format
make lint
```

## Architecture

```
cadl_frontend/
├── grammar.lark          # Lark grammar definition
├── parser.py             # AST transformers
├── ast.py                # AST node definitions
├── mlir_converter.py     # CADL → MLIR lowering
└── loop_transform.py     # Loop optimization logic
```

### Key Components
- **Parser**: Lark-based parser with custom transformers
- **Type System**: Width-aware types matching Rust implementation
- **MLIR Converter**: Direct lowering to MLIR SCF/CIRCT dialects
- **Loop Transformer**: AST-level pattern detection and optimization