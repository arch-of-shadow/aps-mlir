# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
A hybrid project combining:
1. **CADL Frontend** - Python parser for Computer Architecture Description Language using Lark
2. **APS-MLIR** - MLIR-based compiler infrastructure with custom TOR and Schedule dialects

## Build Commands

### C++/MLIR Build
```bash
# Initial setup (builds CIRCT from source)
pixi run setup

# Build the MLIR project
pixi run build

### Python/CADL Commands
```bash
# Run all tests
pixi run pytest tests/
# OR
make test

# Run specific test suites
pixi run pytest tests/test_integration.py     # Real-world CADL tests (15 tests)

# Parse CADL files
pixi run parse examples/simple.cadl
pixi run parse-summary examples/zyy.cadl

# Convert CADL to MLIR
pixi run mlir examples/simple.cadl
pixi run mlir examples/raise_while_to_for.cadl

# Development tools
make format  # Format Python code with black/isort
make lint    # Run linting checks
```

## Architecture

### MLIR Components (`lib/` and `include/`)
- **TOR Dialect** (`lib/TOR/`, `include/TOR/`) - Custom MLIR dialect for hardware transformations
  - Key passes: `ArrayPartition`, `LoopTripcount`, `HlsUnroll`, `MemrefReuse`, `DependenceAnalysis`
  - Pragmas and optimizations for hardware synthesis
- **Schedule** (`lib/Schedule/`, `include/Schedule/`) - Scheduling algorithms and SDC solver
  - `SDCSchedule`, `SDCSolver` - System of Difference Constraints scheduling
  - `CDFG` - Control/Data Flow Graph representations

### CADL Frontend (`cadl_frontend/`)
- **Parser Pipeline**: `grammar.lark` → Lark LALR parser → `parser.py` transformers → `ast.py` AST
- **Type System** (matches Rust `type_sys_ir.rs`):
  - `BasicType`: `ApFixed(width)`, `ApUFixed(width)`, `Float32`, `Float64`
  - `DataType`: `Single`, `Array`, `Instance`
  - `CompoundType`: `Basic`, `FnTy`
- **Literal System**: Width-aware number literals (e.g., `5'b101010` → `u5` type)
- **MLIR Converter** (`mlir_converter.py`): Converts CADL AST to MLIR operations
  - Supports SCF dialect (Structured Control Flow)
  - Handles loops, conditionals, function calls, memory operations
  - Integrates with CIRCT dialects (comb, hw, aps)
- **Loop Transformation** (`loop_transform.py`): AST-level loop optimization
  - Automatically raises `do-while` loops to `scf.for` when pattern-detectable
  - Falls back to `scf.while` for general cases
  - Supports loop-carried variables via iter_args
  - Applies directives as MLIR attributes (e.g., `[[unroll(4)]]`)

### Key Integration Points
- CADL AST directly lowers to MLIR IR via `mlir_converter.py`
- Type system matches between Python frontend and Rust/MLIR backend
- Frontend validates RISC-V attributes (opcode, funct7) for hardware generation
- Loop transformation happens at AST level before MLIR emission

## Environment Setup
- **Build System**: Pixi (conda-forge based) + CMake/Ninja
- **Dependencies**: CIRCT, MLIR, LLVM (built from source via `pixi run setup`)
- **Python**: 3.13 (managed by Pixi)
- **Key Libraries**: lp_solve, nanobind, pybind11, nlohmann_json

## Testing Strategy
- Python tests use pytest with comprehensive AST validation
- Test real-world CADL examples from `examples/zyy.cadl`
- Validate operator precedence, type inference, and width extraction

## Loop Transformation Details

The loop transformer (`cadl_frontend/loop_transform.py`) performs AST-level analysis to raise CADL `do-while` loops to MLIR `scf.for` operations when possible.

### Pattern Detection
A loop can be raised to `scf.for` if it matches these criteria:
1. **Affine induction variable**: One binding with constant init and constant step (e.g., `i: u8 = (0, i+1)`)
2. **Constant bounds**: Loop condition compares IV to a constant (supports `<`, `<=`, `>`, `>=`, `!=`)
3. **No bound modification**: Bound variables are not modified in loop body
4. **Constant loop-carried state**: Other bindings have constant initializers (for iter_args)

### Supported Patterns
```cadl
// Pattern 1: Binding expression increment
with i: u8 = (0, i + 1), crc: u8 = (0, crc_)
do { ... } while i + 1 < 8;
// → scf.for %i = 0 to 8 step 1 iter_args(%crc = 0)

// Pattern 2: Body-assigned increment
with i: u8 = (0, i_), crc: u8 = (0, crc_)
do { i_ = i + 1; ... } while i_ < 8;
// → scf.for %i = 0 to 8 step 1 iter_args(%crc = 0)

// Pattern 3: Backward loop with constant variables
let i0: u8 = 8; let iend: u8 = 0;
with i: u8 = (i0, i_), crc: u8 = (0, crc_)
do { i_ = i - 1; ... } while i_ > iend;
// → scf.for %i = 8 to 0 step -1 iter_args(%crc = 0)
```

### Directive Support
Directives are applied as MLIR attributes on operations:
```cadl
[[unroll(4)]]
with i: u8 = (0, i + 1), crc: u8 = (0, crc_)
do { ... } while i + 1 < 8;
// → scf.for ... { ... } {unroll = 4 : i32}
```

Any directive with an integer literal becomes an integer attribute.
Directives without arguments become boolean attributes.

### Output Format
Loop transformation produces one-line summaries:
```
Loop -> scf.for: i=0..8 step 1, iter_args=[crc]
Loop -> scf.while: Invalid comparison operator: ==
```

## Common Tasks

### Adding a new MLIR pass
1. Create pass files in `lib/TOR/` and `include/TOR/`
2. Register in `lib/TOR/CMakeLists.txt`
3. Follow existing patterns (e.g., `ArrayPartition.cpp`)

### Modifying CADL grammar
1. Edit `cadl_frontend/grammar.lark`
2. Update transformers in `cadl_frontend/parser.py`
3. Add/update AST nodes in `cadl_frontend/ast.py`
4. Add tests to `tests/test_*.py`

### Extending loop transformation
1. Modify pattern detection in `cadl_frontend/loop_transform.py`
2. Update `ForLoopPattern` dataclass if needed
3. Test with examples in `examples/raise_while_to_for.cadl`

### Debugging parser issues
```python
# Use parse_with_errors.py for detailed error messages
python parse_with_errors.py examples/simple.cadl
```