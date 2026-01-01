# Megg Compiler Architecture

## Overview
Megg is a compiler framework for application-specific instruction processor (ASIP) design that integrates MLIR with e-graph reasoning to mitigate phase-ordering challenges. The system performs a round-trip translation between MLIR and egglog terms, enabling algebraic saturation and custom instruction selection under a single orchestration layer.

## Unified MLIR ↔ E-graph Pipeline
- **Bidirectional translation layer**: `FuncToTerms` lowers MLIR SSA graphs (arithmetic, memory, and structured control flow) into egglog terms, while `ExprTreeToMLIR` reconstructs optimized expression trees back into MLIR, preserving block structure, types, and side-effect semantics.
- **Three-stage schedule**: `Compiler.schedule()` coordinates (i) internal algebraic rewrites inside the e-graph, (ii) external MLIR passes (interface provided for future integration), and (iii) the custom instruction phase. This arrangement unifies traditional MLIR passes with e-graph saturation within the same control loop.
- **Cost-driven extraction**: The `Extractor` uses architecture-aware models such as `MeggCost` to score candidate expression trees before re-materializing MLIR, ensuring that e-graph exploration and MLIR reconstruction are guided by the same cost objective.

## Two-Stage Pattern Matching Engine
- **Skeleton construction**: `_build_skeleton_from_func()` derives a hierarchical `SkeletonNode` representation for complex patterns containing `scf.for`, `scf.if`, or nested regions. The skeleton captures block structure and statement-level constraints, serving as a reusable control-flow blueprint.
- **Component instrumentation**: Leaf statements within the skeleton are transformed into lightweight `component_instr` rewrites that tag the corresponding subgraphs during saturation. This tagging makes subsequent structural verification tractable even for deeply nested control flow.
- **Indexed matching**: The `SkeletonMatcher` first queries control-flow indices in `MeggEGraph` to identify candidates with the same container type, then validates component bindings against skeleton constraints. Successful matches insert a unified `Term.custom_instr` node into the e-class, which later lowers to `llvm.inline_asm`, aligning the handling of simple and complex patterns.

## CADL Semantic Lifting
- **Local abstraction passes**: `APSPatternExtractor` and `CADLPatternAbstractor` provide transpiler-style passes that strip APS/CADL-specific I/O constructs—such as register file accesses and burst DMA—from pattern definitions while retaining their computation kernels.
- **Memory interface normalization**: During abstraction, `aps.memload`/`aps.memstore` operations are rewritten into standard `memref.load`/`memref.store`, and function signatures are reconstructed to expose memref arguments. This normalization allows patterns authored in CADL to match MLIR generated from conventional frontends.
- **Dual-view preservation**: Each skeleton retains `full_definition` (the complete hardware-aware APS function) alongside the derived `match_pattern`. Matching operates on the abstracted view, whereas code generation refers back to the full definition, ensuring that semantic lifting behaves as a projection rather than an information-destroying raise.

