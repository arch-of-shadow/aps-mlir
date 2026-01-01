# End-to-End C Compilation Guide

This guide explains how to use Megg's end-to-end (E2E) C compilation feature to optimize C code with custom instructions.

## Overview

The E2E compilation feature allows you to:
1. Mark specific C functions for optimization using `#pragma megg optimize`
2. Automatically split code into optimized (Megg path) and standard (LLVM path) sections
3. Compile and link everything into a final executable

This is ideal for **ASIP development** where you want to match custom hardware instructions in performance-critical functions while keeping the rest of your code standard.

## Quick Start

### 1. Mark Functions for Optimization

Add `#pragma megg optimize` before any function you want to optimize:

```c
#include <stdio.h>

// Regular function - compiled through LLVM
void helper() {
    printf("Helper function\n");
}

// Optimized function - compiled through Megg
#pragma megg optimize
int compute_intensive(int a, int b, int n) {
    int result = a;
    for (int i = 0; i < n; i++) {
        if (result < 100) {
            result += b;
        }
    }
    return result;
}

int main() {
    int result = compute_intensive(10, 5, 20);
    printf("Result: %d\n", result);
    helper();
    return 0;
}
```

### 2. Prepare Custom Instruction Pattern

Create an MLIR file defining your custom instruction pattern (see [CUSTOM_INSTRUCTION_MATCHING.md](CUSTOM_INSTRUCTION_MATCHING.md)):

```mlir
// pattern.mlir
module {
  func.func @my_custom_instr(%n: index, %init: i32, %val: i32, %threshold: i32) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
      %cond = arith.cmpi slt, %acc, %threshold : i32
      %next = scf.if %cond -> (i32) {
        %sum = arith.addi %acc, %val : i32
        scf.yield %sum : i32
      } else {
        scf.yield %acc : i32
      }
      scf.yield %next : i32
    }

    return %result : i32
  }
}
```

### 3. Compile

Use the `--mode c-e2e` flag:

```bash
./megg-opt.py --mode c-e2e app.c \
  --custom-instructions pattern.mlir \
  -o app.out
```

### 4. Run

```bash
./app.out
```

## Compilation Pipeline

The E2E compiler follows this workflow:

```
┌─────────────────────────────────────────────────────────┐
│                     Input: app.c                        │
│  - Regular functions (helper, main)                     │
│  - #pragma megg optimize functions (compute_intensive)  │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        ↓                                   ↓
┌──────────────────┐              ┌──────────────────┐
│   Pragma Parser  │              │                  │
│  Split code:     │              │                  │
│  • target.c      │              │                  │
│  • rest.c        │              │                  │
└──────────────────┘              └──────────────────┘
        ↓                                   ↓
┌──────────────────┐              ┌──────────────────┐
│  Target Path     │              │  Rest Path       │
│  (Megg)          │              │  (LLVM)          │
└──────────────────┘              └──────────────────┘
        ↓                                   ↓
  C → Polygeist                      C → Clang
        ↓                                   ↓
  MLIR → Megg                        LLVM IR → LLC
        ↓                                   ↓
  Optimized MLIR                     Object file
        ↓                                   ↓
  LLVM IR → LLC                             ↓
        ↓                                   ↓
  Object file                               ↓
        │                                   │
        └─────────────────┬─────────────────┘
                          ↓
                ┌──────────────────┐
                │   Linker (ld)    │
                │  Executable      │
                └──────────────────┘
```

## Command-Line Options

### Basic Usage

```bash
./megg-opt.py --mode c-e2e INPUT.c \
  --custom-instructions PATTERN.mlir \
  -o OUTPUT
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode c-e2e` | Enable E2E compilation mode | Required |
| `--custom-instructions FILE` | Custom instruction MLIR pattern | Required |
| `-o, --output FILE` | Output executable path | Required |
| `--keep-intermediate` | Keep temporary files for debugging | False |
| `--verbose` | Enable verbose logging | False |

### Examples

**Basic compilation**:
```bash
./megg-opt.py --mode c-e2e app.c \
  --custom-instructions pattern.mlir \
  -o app.out
```

**With debugging**:
```bash
./megg-opt.py --mode c-e2e app.c \
  --custom-instructions pattern.mlir \
  -o app.out \
  --keep-intermediate \
  --verbose
```

**Multiple marked functions**:
```c
#pragma megg optimize
int func1(int a, int b) { ... }

#pragma megg optimize
int func2(int x, int y) { ... }
```

## Debugging

### Enable Intermediate Files

Use `--keep-intermediate` to preserve temporary files:

```bash
./megg-opt.py --mode c-e2e app.c \
  --custom-instructions pattern.mlir \
  -o app.out \
  --keep-intermediate
```

Temporary files will be in `/tmp/megg_e2e_XXXXX/`:
- `target.c`: Functions marked for optimization
- `rest.c`: Other functions with extern declarations
- `target.mlir`: Initial MLIR from Polygeist
- `optimized.mlir`: After Megg optimization
- `target.ll`: LLVM IR for optimized functions
- `rest.ll`: LLVM IR for rest functions
- `target.o`, `rest.o`: Object files

### Verbose Output

Use `--verbose` to see detailed compilation steps:

```bash
./megg-opt.py --mode c-e2e app.c \
  --custom-instructions pattern.mlir \
  -o app.out \
  --verbose
```

### Common Issues

**Issue**: "Polygeist conversion failed"
- **Solution**: Check that your C code is valid and doesn't use unsupported features
- **Debug**: Look at `target.c` to see what's being converted

**Issue**: "Megg optimization failed"
- **Solution**: Verify your custom instruction pattern matches the function structure
- **Debug**: Check `target.mlir` to see the MLIR representation

**Issue**: "Linking failed"
- **Solution**: Ensure function signatures match between target.c and rest.c
- **Debug**: Check extern declarations in `rest.c`

## Limitations

Current limitations of the E2E compiler:

1. **Pragma Parsing**: Uses simple regex-based parsing (no full C parser)
   - May fail with complex macros or conditional compilation
   - Assumes standard C function syntax

2. **Function Splitting**:
   - Functions are extracted by brace counting
   - May have issues with nested structures or comments containing braces

3. **Dependencies**:
   - Marked functions should be self-contained or only call other marked functions
   - External dependencies must be handled manually

4. **Build System**:
   - Currently single-file compilation only
   - No support for multi-file projects (yet)

## Advanced Usage

### Multiple Patterns

You can match different custom instructions by combining patterns:

```bash
# First, merge patterns into one MLIR file
cat pattern1.mlir pattern2.mlir > combined.mlir

# Then compile
./megg-opt.py --mode c-e2e app.c \
  --custom-instructions combined.mlir \
  -o app.out
```

### Integration with Build Systems

Create a wrapper script for Makefile integration:

```bash
#!/bin/bash
# megg-cc wrapper

if grep -q "#pragma megg optimize" "$1"; then
    # Has pragma - use Megg E2E
    ./megg-opt.py --mode c-e2e "$1" \
      --custom-instructions patterns.mlir \
      -o "$2"
else
    # No pragma - use standard compiler
    gcc "$1" -o "$2"
fi
```

## Comparison with MLIR Mode

Megg supports two compilation modes:

| Feature | MLIR Mode | C E2E Mode |
|---------|-----------|------------|
| Input | MLIR file | C source file |
| Output | Optimized MLIR | Executable |
| Use Case | MLIR development | End-to-end application |
| Function Selection | Manual | `#pragma` directive |
| Linking | Manual | Automatic |

**MLIR Mode** (original):
```bash
# Manual workflow
./megg-opt.py input.mlir --custom-instructions pattern.mlir -o output.mlir
mlir-opt output.mlir --test-lower-to-llvm -o output.ll
llc -filetype=obj output.ll -o output.o
clang output.o -o executable
```

**C E2E Mode** (new):
```bash
# Automatic workflow
./megg-opt.py --mode c-e2e input.c --custom-instructions pattern.mlir -o executable
```

## Next Steps

- See [CUSTOM_INSTRUCTION_MATCHING.md](CUSTOM_INSTRUCTION_MATCHING.md) for pattern creation
- See [tests/e2e/](../tests/e2e/) for example applications
- See [CLAUDE.md](../CLAUDE.md) for overall architecture

## Contributing

To extend the E2E compiler:

1. **Pragma Parser** (`python/megg/frontend/pragma_parser.py`):
   - Add support for pragma options (e.g., `#pragma megg optimize(level=3)`)
   - Implement full C parser using libclang

2. **E2E Compiler** (`python/megg/e2e_compiler.py`):
   - Add multi-file support
   - Implement incremental compilation
   - Add cross-compilation support

3. **CLI** (`python/megg/cli.py`):
   - Add more command-line options
   - Integrate with build systems (CMake, Make)
