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
pixi run pytest tests/ -v
# OR
make test

# Run specific test suites
pytest tests/test_parser.py -v           # Basic parser tests (9 tests)
pytest tests/test_zyy_examples.py -v     # Real-world CADL tests (15 tests)
pytest tests/test_literal_widths.py -v   # Number literal width tests

# Parse CADL files
pixi run parse examples/simple.cadl
pixi run parse-summary examples/zyy.cadl

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

### Key Integration Points
- CADL AST designed to eventually lower to MLIR IR
- Type system matches between Python frontend and Rust/MLIR backend
- Frontend validates RISC-V attributes (opcode, funct7) for hardware generation

## Environment Setup
- **Build System**: Pixi (conda-forge based) + CMake/Ninja
- **Dependencies**: CIRCT, MLIR, LLVM (built from source via `pixi run setup`)
- **Python**: 3.13 (managed by Pixi)
- **Key Libraries**: lp_solve, nanobind, pybind11, nlohmann_json

## Testing Strategy
- Python tests use pytest with comprehensive AST validation
- Test real-world CADL examples from `examples/zyy.cadl`
- Validate operator precedence, type inference, and width extraction

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

### Debugging parser issues
```python
# Use parse_with_errors.py for detailed error messages
python parse_with_errors.py examples/simple.cadl
```