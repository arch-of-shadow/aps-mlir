# CADL Frontend Project Development Progress

## Project Overview
This project contains a Python-based parser for CADL (Computer Architecture Description Language), originally ported from a Rust implementation using LALRPOP to Python using Lark parser generator.

## Project Structure

### Main Components
- **`cadl_rust/`** - Original Rust implementation with LALRPOP
- **`cadl-frontend/`** - New Python implementation with Lark
- **`hector/`** - Related tooling (contains debugger and other tools)

### Python Frontend (`cadl-frontend/`)
```
cadl-frontend/
   cadl_frontend/
      __init__.py          # Package entry point
      ast.py              # AST classes and type system
      grammar.lark        # Lark grammar file
      parser.py           # Parser and transformer
   tests/
      test_parser.py         # Basic parser tests (9 tests)
      test_zyy_examples.py   # Comprehensive test suite (15 tests)
      test_literal_widths.py # Number literal width tests
   examples/
      simple.cadl         # Example CADL code
      zyy.cadl           # Real-world CADL examples
      usage_example.py    # Usage demonstration
   pyproject.toml          # Python project configuration
   Makefile               # Development commands
   README.md              # Project documentation
```

## Type System Architecture

### Current Type System (matches `type_sys_ir.rs`)
The Python implementation uses the new IR type system from `cadl_rust/type_sys_ir.rs`:

**BasicType variants:**
- `BasicType_ApFixed(width: int)` - Signed fixed-point (e.g., `i32`)
- `BasicType_ApUFixed(width: int)` - Unsigned fixed-point (e.g., `u32`)  
- `BasicType_Float32()` - 32-bit float
- `BasicType_Float64()` - 64-bit float
- `BasicType_String()` - String type
- `BasicType_USize()` - USize type

**DataType variants:**
- `DataType_Single(BasicType)` - Single type
- `DataType_Array(BasicType, List[int])` - Array with dimensions
- `DataType_Instance()` - Instance type

**CompoundType variants:**
- `CompoundType_Basic(DataType)` - Basic compound type
- `CompoundType_FnTy(args: List[CompoundType], ret: List[CompoundType])` - Function type

### Literal System (matches `literal.rs`)
**LiteralInner variants:**
- `LiteralInner_Fixed(value: int)` - Fixed-point literal  
- `LiteralInner_Float(value: float)` - Float literal

**Literal struct:**
- `Literal(lit: LiteralInner, ty: BasicType)` - Literal with type information

## Development Progress

### Completed Tasks

#### Initial Implementation & Type System
- ✅ Created comprehensive Lark grammar file (grammar.lark) with proper operator precedence
- ✅ Implemented AST data structures matching Rust IR implementation
- ✅ **Replaced entire type system** to match `cadl_rust/type_sys_ir.rs`
- ✅ **Added Literal system** matching `cadl_rust/literal.rs`
- ✅ Built transformer classes to convert parse tree to AST
- ✅ Added support for all CADL language constructs:
  - Flows (default and rtype)
  - Functions with arguments and return types
  - Static variables and regfiles
  - Memory and register file operations
  - Complex expressions with proper precedence
  - Do-while loops with with-bindings
  - Attributes (opcode, funct7)

#### Grammar Enhancements
- ✅ Fixed expression precedence hierarchy to match LALRPOP implementation
- ✅ Added support for all number literal formats (binary, octal, decimal, hex with prefixes)
- ✅ Implemented cast operators ($signed, $unsigned, $f32, $f64, $int, $uint)
- ✅ **CRITICAL FIX**: Moved cast expressions from unary_expr to primary_expr to support `~$unsigned(val)` syntax
- ✅ Added comprehensive operator support (arithmetic, bitwise, logical, comparison, shift)

#### Parser Implementation
- ✅ Fixed transformer token indexing issues across multiple expression types
- ✅ Enhanced attribute parsing for opcode/funct7 RISC-V instruction encoding
- ✅ Fixed array type parsing and aggregate expression handling
- ✅ **CRITICAL FIX**: Fixed function call argument parsing to properly handle expressions instead of raw tokens
- ✅ Added proper index and slice expression support
- ✅ Implemented comprehensive literal parsing with width extraction

#### Test Coverage & Comprehensive Enhancement
- ✅ **Implemented width extraction** for sized number literals: `5'b101010` → `u5` type
- ✅ **Added comprehensive test suite** for literal width parsing (test_literal_widths.py)
- ✅ Created comprehensive test suite (test_zyy_examples.py) with 15 test methods
- ✅ Added detailed AST validation with type checking and structural verification
- ✅ Tested all operators, expressions, and language constructs
- ✅ Added edge case and error handling tests
- ✅ Validated complex real-world examples from zyy.cadl
- ✅ Ensured backward compatibility with existing basic tests (test_parser.py)

### Technical Achievements

#### Grammar Precedence (Lowest to Highest)
```
logical_or_expr (||)
bitwise_or_expr (|)
bitwise_xor_expr (^)
eq_expr (==, !=)
bitwise_and_expr (&), logical_and_expr (&&)
rel_expr (<, <=, >, >=)
shift_expr (<<, >>)
add_expr (+, -)
mul_expr (*, /, %)
unary_expr (-, !, ~)
postfix_expr ([], [start:end])
primary_expr (literals, identifiers, function calls, casts)
```

#### Key Syntax Support
- ✅ Complex number literals: `5'b101010`, `8'hFF`, `32'hEDB88320`
- ✅ Memory access: `_mem[addr]`, `_irf[rs1]`
- ✅ Function calls: `helper(offset)`, nested calls
- ✅ Type casts: `$signed(val)`, `$unsigned(val)`
- ✅ Unary operators with casts: `~$unsigned(val)` ← **User requirement**
- ✅ Complex expressions: `a ^ (0xEDB88320 & ~((x & 1) - 1))`
- ✅ Slice operations: `r1[31:24]`, `vec[1:3]`
- ✅ Aggregate expressions: `{r1[31:24], r2[23:16]}`

#### AST Structure
- Comprehensive type hierarchy: BasicType → DataType → CompoundType
- Expression hierarchy: BinaryExpr, UnaryExpr, CallExpr, IndexExpr, SliceExpr
- Statement types: AssignStmt, ReturnStmt, DoWhileStmt, GuardStmt
- Flow definitions with attributes and type signatures
- Static variables and regfile declarations

## Key Features

### Number Literal Parsing
The parser correctly handles width-specified number literals:
- `5'b101010` → `u5` type with value 42
- `8'hFF` → `u8` type with value 255
- `15'd123` → `u15` type with value 123
- `3'o123` → `u3` type with value 83

Default literals without width specification get 32-bit unsigned type:
- `0x1234` → `u32` type
- `42` → `u32` type

### Grammar Features
The Lark grammar supports all CADL constructs:
- **Flow definitions**: `flow` and `rtype` flows with attributes
- **Functions**: Function definitions with arguments and return types
- **Statements**: Assignment, loops, guards, directives, spawn, threads
- **Expressions**: All operators with correct precedence, function calls, indexing, slicing
- **Types**: Basic types, arrays, compound types, function types
- **Literals**: Number literals (various formats), string literals, booleans

### Parser Implementation Details
- Uses **LALR parser** (not Earley) to support embedded transformers
- **`?` operator** used extensively in grammar to eliminate unnecessary parse tree nesting
- **Transformer classes** convert parse trees to proper AST objects
- **Type-aware parsing** creates proper `Literal` objects with correct bit widths

## Development Environment

### Python Environment
- Uses **pixi** for environment management
- Python 3.10 required
- Key dependencies: `lark>=1.1.0`, `pytest>=8.4.2`

### Development Commands
```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_literal_widths.py -v
pytest tests/test_zyy_examples.py -v
pytest tests/test_parser.py -v

# Format code
make format

# Run linting
make lint
```

### Test Coverage
- **`test_parser.py`** - Basic parser functionality tests (9 tests)
- **`test_zyy_examples.py`** - Comprehensive real-world CADL tests (15 tests)
- **`test_literal_widths.py`** - Number literal width parsing tests
- Tests cover: number formats, type inference, width extraction, operators, complex expressions

## Usage Examples

### Basic Parsing
```python
from cadl_frontend import parse_proc

# Parse CADL source
with open("example.cadl", "r") as f:
    source = f.read()

ast = parse_proc(source, "example.cadl")
print(f"Found {len(ast.flows)} flows and {len(ast.functions)} functions")
```

### Accessing Literal Information
```python
# Parse a static declaration with literal
ast = parse_proc("static x: u32 = 5'b101010;")
static_var = list(ast.statics.values())[0]
literal = static_var.expr.literal

print(f"Type: {type(literal.ty).__name__}")  # BasicType_ApUFixed
print(f"Width: {literal.ty.width}")          # 5
print(f"Value: {literal.lit.value}")         # 42
```

## Current Status
- **COMPLETED**: Comprehensive CADL frontend with full parsing capabilities
- **ALL TESTS PASSING**: 24 total tests (15 enhanced + 9 basic compatibility)
- **READY FOR USE**: Can parse real-world CADL files like zyy.cadl
- **FULL FEATURE PARITY**: Matches Rust LALRPOP implementation functionality

## Known Working Features
✅ Number literal parsing with correct width extraction  
✅ All basic CADL syntax (flows, functions, statements)  
✅ Type system matching Rust IR implementation  
✅ Expression parsing with operator precedence  
✅ Static variable declarations  
✅ Flow definitions with attributes (opcode, funct7)
✅ Memory and register file operations
✅ Complex expressions and function calls
✅ Do-while loops with with-bindings
✅ All unary and binary operators including `~$unsigned(val)`

## File Locations
- **Main parser**: `cadl-frontend/cadl_frontend/parser.py`
- **AST definitions**: `cadl-frontend/cadl_frontend/ast.py`  
- **Grammar**: `cadl-frontend/cadl_frontend/grammar.lark`
- **Tests**: `cadl-frontend/tests/`
- **Examples**: `cadl-frontend/examples/`

## Python Environment Path
Use this Python interpreter: `/home/zyy/aps-mlir/cadl-frontend/.pixi/envs/default/bin/python`

## Testing Commands
```bash
# Test number literal width parsing specifically
/home/zyy/aps-mlir/cadl-frontend/.pixi/envs/default/bin/python -m pytest tests/test_literal_widths.py -v

# Test comprehensive real-world examples
/home/zyy/aps-mlir/cadl-frontend/.pixi/envs/default/bin/python -m pytest tests/test_zyy_examples.py -v

# Test basic parsing functionality  
/home/zyy/aps-mlir/cadl-frontend/.pixi/envs/default/bin/python -m pytest tests/test_parser.py -v

# Run all tests
/home/zyy/aps-mlir/cadl-frontend/.pixi/envs/default/bin/python -m pytest tests/ -v
```

---
*Last updated: 2025-09-10*
*Total development time: Multiple sessions with comprehensive testing and validation*
*Status: Production-ready CADL parser with full feature support*