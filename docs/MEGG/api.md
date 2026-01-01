# Megg API Reference

## Overview

This document covers Megg's public interfaces:
- **CLI**: `megg-opt` command-line tool
- **Python API**: `Compiler` class and modules
- **Cost Functions**: Customizable extraction strategies
- **Backend**: LLVM lowering and compilation

---

## Command-Line Interface

### Basic Usage

```bash
# Optimize MLIR file (internal + external rewrites)
./megg-opt.py input.mlir

# Specify output path
./megg-opt.py input.mlir -o output.mlir

# Internal rewrites only
./megg-opt.py input.mlir --rewrite-mode internal

# Target specific functions
./megg-opt.py input.mlir --target-functions "matmul,conv2d"

# With debugging
./megg-opt.py input.mlir --verbose --dump-egraph debug.svg
```

### Arguments Reference

#### Core Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input` | path | *required* | Input MLIR file |
| `-o, --output` | path | `outputs/<name>_opt_<mode>.mlir` | Output file path |
| `--rewrite-mode` | `internal\|external\|both\|none` | `both` | Which rewrites to apply |
| `--mlir-passes` | string | None | Semicolon-separated MLIR passes |
| `--target-functions` | string | None | Comma-separated function names |

#### Optimization Parameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--iter` | int | 100 | Max iterations (currently unused) |
| `--time-limit` | float | 300.0 | Wall-clock time limit (seconds) |
| `--extract-cost-model` | `json\|builtin` | `builtin` | Cost model for extraction |
| `--disable-safeguards` | flag | False | Disable time limits and error handling |

#### Debugging Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose` | flag | False | Verbose output |
| `--log-level` | `DEBUG\|INFO\|WARNING\|ERROR` | `INFO` | Logging level |
| `--dump-egraph` | path | None | Save e-graph visualization (.svg/.dot) |
| `--dump-state` | path | None | Save compiler state (⚠️ not implemented) |

#### Backend Options
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--emit-llvm` | flag | False | Emit LLVM IR instead of MLIR |
| `--compile` | `object\|executable` | None | Compile to object/executable |
| `--opt-level` | `O0\|O1\|O2\|O3` | `O2` | LLVM optimization level |
| `--target` | string | host | Target triple (e.g., `x86_64-linux-gnu`) |
| `--link-libs` | string | None | Comma-separated libraries to link |
| `--custom-instructions` | path | None | MLIR file with custom instruction definitions |

### Examples

```bash
# Basic optimization
./megg-opt.py tests/mlir/simple_arith.mlir

# Internal rewrites with debugging
./megg-opt.py tests/mlir/matmul.mlir \
  --rewrite-mode internal \
  --verbose \
  --dump-egraph matmul_egraph.svg

# External MLIR passes only
./megg-opt.py input.mlir \
  --rewrite-mode external \
  --mlir-passes "canonicalize;cse;loop-fusion"

# Compile to executable with optimizations
./megg-opt.py input.mlir \
  --compile executable \
  --opt-level O3 \
  -o program

# Custom instruction matching
./megg-opt.py input.mlir \
  --custom-instructions custom_ops.mlir \
  -o output.mlir
```

### Exit Codes
- `0`: Success
- `1`: Failure (parse error, compilation error, or runtime exception)

---

## Python API

### Module Import

```python
# Core compiler
from megg.compiler import Compiler, CompilationResult

# E-graph components
from megg.egraph.func_to_terms import FuncToTerms
from megg.egraph.terms_to_func import ExprTreeToMLIR
from megg.egraph.megg_egraph import MeggEGraph, ExpressionNode
from megg.egraph.extract import Extractor, AstSize, AstDepth, OpWeightedCost

# Rewrite rules
from megg.rewrites.internal_rewrites import basic_math_laws, constant_folding_laws
from megg.rewrites.match_rewrites import build_ruleset_from_module

# Backend
from megg.backend.llvm_backend import MLIRToLLVMBackend

# CLI utilities
from megg.cli import load_mlir, save_mlir
```

### Compiler Class

#### Constructor

```python
compiler = Compiler(
    module: mlir.ir.Module,              # Input MLIR module
    target_functions: List[str] = None,  # Optional: specific functions to optimize
    cost_function: CostFunction = None,  # Default: AstSize()
    match_ruleset: egglog.Ruleset = None # Optional: custom instruction matching
)
```

**Parameters:**
- `module`: MLIR module (created via `mlir.ir.Module.parse()` or `load_mlir()`)
- `target_functions`: List of function names to optimize (default: all functions)
- `cost_function`: Cost function for extraction (default: `AstSize()`)
  - Options: `AstSize()`, `AstDepth()`, `OpWeightedCost({...})`, `MeggCost()`, custom
  - **`MeggCost()`**: Hardware-aware cost model with operation-specific costs
  - **`MeggCost(custom_costs={...})`**: Override specific operation costs
- `match_ruleset`: Custom egglog ruleset for pattern matching

**Attributes:**
- `egraphs: Dict[str, EGraph]` - One e-graph per function
- `transformers: Dict[str, FuncToTerms]` - MLIR ↔ e-graph mappings
- `original_module: mlir.ir.Module` - Input module reference
- `cost_function: CostFunction` - Extraction cost function

#### schedule() Method

```python
result = compiler.schedule(
    max_iterations: int = 10,            # Max iterations (unused currently)
    time_limit: float = 60.0,           # Wall-clock time limit (seconds)
    internal_rewrites: bool = True,      # Apply egglog rules
    external_passes: List[str] = None,   # MLIR passes (TODO)
    enable_safeguards: bool = True       # Enable time limits and error handling
) -> CompilationResult
```

**Returns:** `CompilationResult` with:
- `optimized_module: mlir.ir.Module` - Optimized module (or original on failure)
- `internal_rewrites: int` - Number of internal rules applied (Phase 1)
- `external_rewrites: int` - Number of external passes applied (Phase 2)
- `custom_rewrites: int` - Number of custom instruction patterns applied (Phase 3)
- `time_elapsed: float` - Total time in seconds
- `success: bool` - Whether optimization succeeded
- `error_message: str` - Error message if `success=False`

**Example:**
```python
import mlir.ir as ir
from megg.compiler import Compiler
from megg.egraph.extract import OpWeightedCost

# Load MLIR
with ir.Context() as ctx:
    ctx.load_all_available_dialects()
    module = ir.Module.parse("""
        func.func @add(%arg0: i32, %arg1: i32) -> i32 {
            %0 = arith.addi %arg0, %arg1 : i32
            return %0 : i32
        }
    """)

# Create compiler with custom cost function
cost_fn = OpWeightedCost({'Mul': 10.0, 'Add': 1.0})
compiler = Compiler(module, cost_function=cost_fn)

# Run optimization
result = compiler.schedule(
    time_limit=30.0,
    internal_rewrites=True
)

if result.success:
    print(f"Optimization succeeded in {result.time_elapsed:.2f}s")
    print(f"Applied {result.internal_rewrites} rewrites")
    print(result.optimized_module)
else:
    print(f"Failed: {result.error_message}")
```

#### visualize_egraph() Method

```python
compiler.visualize_egraph(
    output_path: str,        # Output file path (.svg, .png, .pdf)
    format: str = "svg",     # GraphViz output format
    **kwargs                 # Additional GraphViz options
) -> str                     # Path to generated file
```

Renders the first e-graph using GraphViz.

---

## Cost Functions

### CostFunction Interface

```python
from megg.egraph.extract import CostFunction

class MyCostFunction(CostFunction[float]):
    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        """Calculate cost of a node given children costs."""
        # Custom cost logic
        total = 1.0
        for child_eclass in node.children:
            total += costs(child_eclass)
        return total
```

### Built-in Cost Functions

#### AstSize
```python
from megg.egraph.extract import AstSize

cost_fn = AstSize()  # Minimize total AST node count
```
Each node has cost `1 + sum(child_costs)`. Encourages smaller expressions.

#### AstDepth
```python
from megg.egraph.extract import AstDepth

cost_fn = AstDepth()  # Minimize maximum expression depth
```
Each node has cost `1 + max(child_costs)`. Encourages shallower expressions.

#### OpWeightedCost
```python
from megg.egraph.extract import OpWeightedCost

# Prefer additions over multiplications
cost_fn = OpWeightedCost({
    'Mul': 10.0,
    'Div': 20.0,
    'Add': 1.0,
    'Sub': 1.0,
}, default_cost=5.0)
```
Assigns custom costs per operation type. Useful for targeting specific architectures.

#### ConstantFoldingCost
```python
from megg.egraph.extract import ConstantFoldingCost

cost_fn = ConstantFoldingCost()
```
Heavily favors constant expressions over operations. Literals have cost 1.0, other operations have cost 5.0 + children.

#### MeggCost
```python
from megg.egraph.extract import MeggCost

# Use default hardware-aware costs
cost_fn = MeggCost()

# Override specific operation costs
cost_fn = MeggCost(custom_costs={'Term.custom_instr': 5.0, 'Term.mul': 10.0})
```

Hardware-aware cost model with operation-specific costs. Default costs: literals (1.0), basic arithmetic (2.0), mul (4.0), div (8.0), control flow (10-20), memory (5-15), custom instructions (15.0). Customizable for ASIP optimization.

### Extractor Usage

```python
from megg.egraph.extract import Extractor, OpWeightedCost
from megg.egraph.megg_egraph import MeggEGraph

# Create MeggEGraph from egglog EGraph
megg_egraph = MeggEGraph.from_egraph(egraph, func_transformer)

# Create extractor with cost function
cost_fn = OpWeightedCost({'Mul': 10.0, 'Add': 1.0})
extractor = Extractor(megg_egraph, cost_fn)

# Extract best expression from e-class
result = extractor.find_best(eclass_id)

print(f"Cost: {result.cost}")
print(f"Expression: {result.expr}")
```

**ExtractionResult attributes:**
- `expr: ExpressionNode` - Best expression tree
- `cost: Cost` - Total cost of expression
- `eclass_id: str` - E-class ID that was extracted

---

## LLVM Backend

### MLIRToLLVMBackend Class

```python
from megg.backend.llvm_backend import MLIRToLLVMBackend

backend = MLIRToLLVMBackend(verbose=False)
```

### process() Method

```python
result = backend.process(
    module: mlir.ir.Module,              # Input MLIR module
    output_format: str = "llvm-ir",      # "llvm-ir" | "object" | "executable"
    output_path: str = None,             # Output file path
    optimization_level: str = "O2",      # "O0" | "O1" | "O2" | "O3"
    target: str = None,                  # Target triple (e.g., "x86_64-linux-gnu")
    link_libs: List[str] = None          # Libraries to link
) -> Optional[str]                       # Returns LLVM IR string for "llvm-ir", None otherwise
```

**Example:**
```python
from megg.backend.llvm_backend import MLIRToLLVMBackend

backend = MLIRToLLVMBackend(verbose=True)

# Emit LLVM IR
llvm_ir = backend.process(
    module,
    output_format="llvm-ir",
    output_path="output.ll",
    optimization_level="O3"
)

# Compile to object file
backend.process(
    module,
    output_format="object",
    output_path="output.o",
    optimization_level="O2"
)

# Compile to executable
backend.process(
    module,
    output_format="executable",
    output_path="program",
    optimization_level="O3",
    link_libs=["m", "pthread"]
)
```

**Tool Dependencies:**
- `mlir-opt`: MLIR optimization passes
- `mlir-translate`: MLIR to LLVM IR translation
- `llc`: LLVM IR to object code compilation
- `clang`: Linking executables

Missing tools trigger graceful degradation with warning messages.

---

## Utility Functions

### load_mlir / save_mlir

```python
from megg.cli import load_mlir, save_mlir

# Load MLIR from file
module = load_mlir("input.mlir")

# Save MLIR to file
save_mlir(module, "output.mlir")
```

---

## Extending Megg

### Adding Internal Rewrites

Create new rules in `megg/rewrites/internal_rewrites.py`:

```python
from egglog import rewrite, vars_
from megg.egraph.term import Term, LitTerm

def my_custom_laws() -> egglog.Ruleset:
    a, b = vars_("a b", Term)
    ty = egglog.String('i32')

    rules = [
        # x + (-x) = 0
        rewrite(
            Term.add(a, Term.neg(a, ty), ty)
        ).to(
            Term.lit(LitTerm.int(egglog.i64(0)), ty)
        )
    ]

    return egglog.ruleset(*rules, name='my_custom_laws')
```

Then apply in `RewriteEngine.apply_internal_rewrites()`.

### Adding Custom Cost Functions

```python
from megg.egraph.extract import CostFunction

class LatencyCost(CostFunction[float]):
    def __init__(self, latency_table: Dict[str, float]):
        self.latency_table = latency_table

    def cost(self, node: MeggENode, costs: Callable[[str], float]) -> float:
        op_cost = self.latency_table.get(node.op, 1.0)
        child_cost = sum(costs(c) for c in node.children)
        return op_cost + child_cost

# Use in compiler
compiler = Compiler(module, cost_function=LatencyCost({'Mul': 5.0, 'Add': 1.0}))
```

### Custom Instruction Matching

Define custom instructions in MLIR and load via CLI:

```mlir
// custom_ops.mlir
func.func @mac(%a: i32, %b: i32, %c: i32) -> i32 {
  %0 = arith.muli %a, %b : i32
  %1 = arith.addi %0, %c : i32
  return %1 : i32
}
```

```bash
./megg-opt.py input.mlir --custom-instructions custom_ops.mlir
```

---

## Troubleshooting

### MLIR Python Bindings Not Found
Ensure MLIR is built with Python bindings enabled and added to `PYTHONPATH`:
```bash
export PYTHONPATH=/path/to/llvm-project/install/python_packages/mlir_core:$PYTHONPATH
```

### Functions Skipped During Optimization
Functions containing unsupported operations (e.g., `call`) are automatically skipped. Check logs with `--verbose`.

### E-graph Visualization Fails
Requires GraphViz installed:
```bash
sudo apt-get install graphviz  # Ubuntu/Debian
brew install graphviz          # macOS
```

### External MLIR Passes Not Applied
External rewrites are currently stubbed (TODO). Only internal egglog rewrites are functional.

