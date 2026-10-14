"""Simplified top-level Compiler for MLIR + e-graph optimization."""

from __future__ import annotations

import logging
import signal
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import egglog
from egglog import EGraph

from megg.egraph.extract import AffineCost, Extractor, MeggCost
from megg.egraph.func_to_terms import FuncToTerms
from megg.egraph.megg_egraph import MeggEGraph, SkeletonMatcher
from megg.egraph.terms_to_func import ExprTreeToMLIR
from megg.rewrites.external_loop_pass import apply_loop_pass_by_name
from megg.rewrites.internal_rewrites import (
    basic_math_laws,
    constant_folding_laws,
    type_annotation_ruleset,
)
from megg.utils import (
    IRBuilder,
    MModule,
    MOperation,
    OperationType,
    count_nested_loops,
    get_loop_nest_level,
)

logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Operation timed out")

@contextmanager
def timeout(seconds: int):
    """Context manager for timeout protection."""
    if seconds <= 0:
        yield
        return

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

DEFAULT_EXTERNAL_PASSES: Tuple[str, ...] = ("loop-unroll", "loop-tile")

@dataclass
class FunctionContext:
    """Context information for a function being optimized."""
    func_name: str
    func_op: MOperation
    egraph: EGraph
    transformer: FuncToTerms
    generation: int = 0  # Track transformation generation
    parent_id: Optional[int] = None  # Track where this came from

    def create_child(self, new_func_op: MOperation, new_transformer: FuncToTerms) -> 'FunctionContext':
        """Create a new context from a transformation, preserving history."""
        return FunctionContext(
            func_name=self.func_name,
            func_op=new_func_op,
            egraph=self.egraph,
            transformer=new_transformer,
            generation=self.generation + 1,
            parent_id=id(self),
        )

def _get_var_name(var) -> str:
    """Return a consistent string identifier for egglog variables."""
    name = getattr(var, "name", None)
    if isinstance(name, str):
        return name

    text = str(var)
    if text.startswith('<Var ') and text.endswith('>'):
        return text[5:-1].strip()
    return text

@dataclass
class CompilationResult:
    """Result of the compilation process."""
    optimized_module: MModule
    internal_rewrites: int
    external_rewrites: int
    custom_rewrites: int
    time_elapsed: float
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class PhaseSnapshot:
    """Snapshot of compiler state at a specific phase (for debugging/visualization)."""
    phase_name: str
    phase_index: int
    egraph_stats: Dict[str, Any]  # {func_name: {total_eclasses, total_nodes, ...}}
    cumulative_stats: Dict[str, Any]  # {internal_rewrites, external_rewrites, custom_rewrites}
    details: Optional[Dict[str, Any]] = None  # Phase-specific details (rules applied, etc.)
    timestamp: float = 0.0

DebugCallback = Callable[['PhaseSnapshot'], None]

class StateManager:
    """Manages compilation state and safeguards."""

    def __init__(self, max_iterations: int = 10, time_limit: float = 60.0):
        self.max_iterations = max_iterations
        self.time_limit = time_limit
        self.start_time = 0.0
        self.internal_rewrites = 0
        self.external_rewrites = 0
        self.custom_rewrites = 0
        self.internal_rewrites_details: List[Dict[str, Any]] = []
        self.external_rewrites_details: List[Dict[str, Any]] = []
        self.initial_egraph_statistics: Dict[str, Any] = {}
        self.egraph_statistics: Dict[str, Any] = {}

    def start_compilation(self):
        """Start compilation tracking."""
        self.start_time = time.time()
        self.internal_rewrites = 0
        self.external_rewrites = 0
        self.custom_rewrites = 0
        self.internal_rewrites_details = []
        self.external_rewrites_details = []
        self.initial_egraph_statistics = {}
        self.egraph_statistics = {}
        logger.info("Starting compilation")

    def check_time_limit(self) -> bool:
        """Check if time limit exceeded."""
        elapsed = time.time() - self.start_time
        if elapsed >= self.time_limit:
            logger.warning(f"Time limit ({self.time_limit}s) reached")
            return True
        return False

    def record_internal_rewrites(self, count: int, details: Optional[List[Dict[str, Any]]] = None):
        """Record internal rewrites applied."""
        self.internal_rewrites += count
        if details:
            self.internal_rewrites_details.extend(details)

    def record_external_rewrites(self, count: int, details: Optional[List[Dict[str, Any]]] = None):
        """Record external rewrites applied."""
        self.external_rewrites += count
        if details:
            self.external_rewrites_details.extend(details)

    def record_custom_rewrites(self, count: int):
        """Record custom instruction rewrites applied."""
        self.custom_rewrites += count

    def record_initial_egraph_statistics(self, stats: Dict[str, Any]):
        """Record e-graph statistics at initialization (before any rewrites)."""
        self.initial_egraph_statistics = stats

    def record_egraph_statistics(self, stats: Dict[str, Any]):
        """Record e-graph statistics after saturation."""
        self.egraph_statistics = stats

    def get_result(self, module: MModule, success: bool = True, error_msg: Optional[str] = None) -> CompilationResult:
        """Generate final compilation result."""
        elapsed = time.time() - self.start_time
        return CompilationResult(
            optimized_module=module,
            internal_rewrites=self.internal_rewrites,
            external_rewrites=self.external_rewrites,
            custom_rewrites=self.custom_rewrites,
            time_elapsed=elapsed,
            success=success,
            error_message=error_msg
        )

class RewriteEngine:
    """Handles internal and external rewrite execution."""

    def __init__(self, compiler: Compiler):
        self.compiler = compiler

    def apply_internal_rewrites(self) -> int:
        rulesets = (basic_math_laws(), constant_folding_laws(), type_annotation_ruleset())
        matches = {}
        for graph in self.compiler.egraphs.values():
            for _ in range(5):
                for ruleset in rulesets:
                    report = graph.run(ruleset.saturate())
                    for rule, count in report.num_matches_per_rule.items():
                        if count:
                            matches[rule] = matches.get(rule, 0) + count
        details = [{'rule': rule, 'num_matches': count} for rule, count in matches.items()]
        if self.compiler.state_manager:
            self.compiler.state_manager.internal_rewrites_details.extend(details)
        return len(details)

    def apply_custom_rewrites(self) -> int:
        """Apply custom instruction matching rewrites (final lightweight pass)."""
        try:
            if self.compiler.match_ruleset is None:
                logger.info("No custom instruction rewrites provided")
                return 0

            math_laws = basic_math_laws()
            const_laws = constant_folding_laws()
            type_rules = type_annotation_ruleset()

            logger.info("Re-simplifying e-graph before custom matching...")
            for func_name, egraph in self.compiler.egraphs.items():
                for round_idx in range(5):
                    egraph.run(math_laws.saturate())
                    egraph.run(const_laws.saturate())
                    egraph.run(type_rules.saturate())

            logger.info(
                "Applying custom instruction matching (final pass)...")
            rule_count = len(self.compiler.match_ruleset.__egg_ruleset__.rules)

            for func_name, egraph in self.compiler.egraphs.items():

                egraph.run(self.compiler.match_ruleset)
                logger.debug(
                    f"Applied custom rewrites to function: {func_name}")

            logger.info(
                f"Applied {rule_count} custom instruction patterns")
            return rule_count

        except Exception as e:
            raise RuntimeError(f"Custom instruction matching failed: {e}") from e

    def _get_loop_characteristics(self, loop_op: MOperation) -> Optional[Dict[str, int]]:
        """Collect bounds, trip count, and structural info for a loop."""
        characteristics: Dict[str, int] = {}

        bounds = loop_op._op.get_scf_loop_bounds()

        if not bounds:
            return None

        lower = int(bounds['lower'])
        upper = int(bounds['upper'])
        step = int(bounds['step'])

        if step == 0:
            return None

        trip_count = (upper - lower) // step

        characteristics.update({
            'lower': lower,
            'upper': upper,
            'step': step,
            'trip_count': trip_count,
            'nest_level': get_loop_nest_level(loop_op),
            'nested_loop_count': count_nested_loops(loop_op),
        })

        return characteristics

    def _determine_loop_transformations(
        self,
        current_loop: Dict[str, int],
        hint: Dict[str, int],
        loop_index: int = -1,
        hint_name: str = "",
    ) -> List[Tuple[str, str, int]]:
        """Return all viable loop transformations based on the hint."""

        required = {'lower', 'upper', 'step',
                    'nest_level', 'nested_loop_count'}
        if not hint or not required.issubset(hint):
            return []

        custom_step = hint['step']
        if custom_step == 0:
            return []

        custom_trip = (hint['upper'] - hint['lower']) // custom_step
        if custom_trip <= 0:
            return []

        current_trip = current_loop.get('trip_count', 0)
        if current_trip <= 0:
            return []

        current_nested = current_loop.get('nested_loop_count', 0)
        target_nested = hint['nested_loop_count']

        if current_trip == custom_trip and current_nested == target_nested:
            return []

        decisions: List[Tuple[str, str, int]] = []

        if target_nested > current_nested:
            if current_nested > 0 and custom_trip > 0 and current_trip % custom_trip == 0:
                factor = current_trip // custom_trip
                expected_nested = current_nested * factor
                if factor > 1 and expected_nested == target_nested:
                    decisions.append(('loop-unroll', 'unroll_factor', factor))
                    return decisions

            if custom_trip > 0 and custom_trip < current_trip and current_trip % custom_trip == 0:
                factor = current_trip // custom_trip
                decisions.append(('loop-tile', 'tile_size', factor))
            return decisions

        if target_nested < current_nested:
            if custom_trip > 0 and current_trip % custom_trip == 0:
                factor = current_trip // custom_trip
                if factor > 1:
                    decisions.append(('loop-unroll', 'unroll_factor', factor))
                    if current_nested > 0:
                        decisions.append(('loop-unroll-jam', 'unroll_factor', factor))
            return decisions

        if current_trip > custom_trip and current_trip % custom_trip == 0:
            factor = current_trip // custom_trip
            if factor > 1:
                decisions.append(('loop-unroll', 'unroll_factor', factor))
                if current_nested > 0:
                    decisions.append(('loop-unroll-jam', 'unroll_factor', factor))
            if custom_trip > 0 and custom_trip <= current_trip and current_nested > 0:
                decisions.append(('loop-tile', 'tile_size', factor))
        elif current_trip < custom_trip:
            # Cannot increase iterations via tiling/unrolling; no suitable transform.
            return []

        return decisions

    def _try_apply_external_pass(self, *, loop_index: int,
                                decision: Tuple[str, str, int],
                                func_context: FunctionContext,
                                apply_loop_pass_by_name: Callable):
        pass_name, parameter, value = decision
        if value <= 1:
            return False, None, None
        result = apply_loop_pass_by_name(
            func=func_context.func_op, loop_index=loop_index, pass_name=pass_name,
            func_to_terms=func_context.transformer, egraph=func_context.egraph,
            **{parameter: value})
        if result is None:
            return False, None, None
        transformed, _ = result
        # Operations returned by the pass borrow their module's lifetime.
        self.compiler._external_pass_modules[func_context.func_name].append(transformed["module"])
        return True, transformed["function"], transformed["transformer"]

    def apply_external_rewrites(self) -> int:
        try:
            pass_count = 0
            external_details = []  # Track external pass details

            all_contexts: List[FunctionContext] = []
            contexts_by_func: Dict[str, List[FunctionContext]] = {}

            for func_name in list(self.compiler.transformers.keys()):
                egraph = self.compiler.egraphs.get(func_name)
                transformer = self.compiler.transformers.get(func_name)

                original_func_op = None
                for op in self.compiler.original_module.get_functions():
                    if getattr(op, 'symbol_name', None) == func_name:
                        original_func_op = op
                        break

                if egraph is None or transformer is None or original_func_op is None:
                    continue

                loops = transformer.loop_to_term
                needs_transformation = False

                if loops and self.compiler.loop_hints:
                    for loop_index, (loop_op, _) in list(loops.items()):
                        current_info = self._get_loop_characteristics(loop_op)
                        if not current_info:
                            continue

                        current_nest_level = current_info.get('nest_level', -1)
                        matching_hints = {
                            name: info for name, info in self.compiler.loop_hints.items()
                            if info.get('nest_level', -1) == current_nest_level
                        }

                        for ci_loop_name, ci_loop_info in matching_hints.items():
                            decisions = self._determine_loop_transformations(
                                current_info,
                                ci_loop_info,
                                loop_index=loop_index,
                                hint_name=ci_loop_name,
                            )
                            if decisions:  # Non-empty means transformation needed
                                needs_transformation = True
                                logger.info(f"Loop {loop_index} needs transformation based on hint '{ci_loop_name}': {decisions}")
                                break

                        if needs_transformation:
                            break

                if needs_transformation:
                    try:

                        logger.info(f"Extracting affine-friendly version of {func_name} for loop passes...")

                        megg_egraph = MeggEGraph.from_egraph(egraph, func_transformer=transformer)

                        extractor = Extractor(megg_egraph, AffineCost())
                        extraction_result = extractor.find_best()

                        logger.info(f"✓ Extracted with cost: {extraction_result.cost}")

                        func_op = ExprTreeToMLIR.reconstruct(
                            original_func=original_func_op,
                            body_exprs=[extraction_result.expr],
                            target_module=self.compiler.original_module
                        )

                        logger.info(f"✓ Reconstructed affine-friendly MLIR for {func_name}")

                        standalone_module = self.compiler.original_module.empty_like()
                        standalone_module.append_to_module(func_op)
                        func_op = standalone_module.get_functions()[0]

                        logger.info(f"Creating transformer for affine-extracted {func_name} in original egraph")
                        new_transformer = FuncToTerms.transform(func_op, egraph)
                        logger.info(f"✓ Transformer created with {len(new_transformer.loop_to_term)} loops")

                        transformer = new_transformer

                    except Exception as e:
                        logger.warning(f"Failed to extract affine version for {func_name}: {e}")
                        traceback.print_exc()
                        raise RuntimeError(f"Affine extraction failed for {func_name}: {e}") from e
                else:
                    logger.info(f"Skipping affine extraction for {func_name} (no transformations needed)")
                    func_op = original_func_op

                initial_ctx = FunctionContext(
                    func_name=func_name,
                    func_op=func_op,  # Use affine-extracted function only if needed
                    egraph=egraph,
                    transformer=transformer,  # Use transformer with loop terms in original egraph
                    generation=0,
                    parent_id=None,
                )

                all_contexts.append(initial_ctx)
                contexts_by_func[func_name] = [initial_ctx]

            ctx_index = 0
            max_contexts = 50  # Limit total contexts to prevent runaway
            max_generation = 0  # Only process generation 0 (initial affine-extracted version)

            while ctx_index < len(all_contexts) and ctx_index < max_contexts:
                ctx = all_contexts[ctx_index]

                logger.info(f"Processing {ctx.func_name} generation {ctx.generation} "
                           f"(context {ctx_index + 1}/{len(all_contexts)})")

                if ctx.generation > max_generation:
                    logger.info(f"Skipping generation {ctx.generation} (max={max_generation})")
                    ctx_index += 1
                    continue

                loops = ctx.transformer.loop_to_term
                logger.info(f"Context has {len(loops)} loop(s) in loop_to_term")
                if not loops:
                    ctx_index += 1
                    continue

                try:
                    with timeout(60):
                        for loop_index, (loop_op, _) in list(loops.items()):
                            logger.info(f"Processing loop {loop_index}")
                            current_info = self._get_loop_characteristics(loop_op)
                            if not current_info:
                                logger.info(f"  No characteristics for loop {loop_index}")
                                continue

                            logger.info(f"  Loop {loop_index} info: {current_info}")

                            current_nest_level = current_info.get('nest_level', -1)
                            logger.info(f"  Current nest_level: {current_nest_level}, available hints: {list(self.compiler.loop_hints.keys())}")

                            matching_hints = {
                                name: info for name, info in self.compiler.loop_hints.items()
                                if info.get('nest_level', -1) == current_nest_level
                            }
                            logger.info(f"  Matching hints for nest_level={current_nest_level}: {list(matching_hints.keys())}")

                            if not matching_hints:
                                logger.debug(f"No matching hints for loop at nest_level={current_nest_level}")
                                continue

                            for ci_loop_name, ci_loop_info in matching_hints.items():
                                logger.info(f"  Trying to match with hint '{ci_loop_name}': {ci_loop_info}")
                                decisions = self._determine_loop_transformations(
                                    current_info,
                                    ci_loop_info,
                                    loop_index=loop_index,
                                    hint_name=ci_loop_name,
                                )

                                logger.info(f"  Decisions for '{ci_loop_name}': {decisions}")

                                if not decisions:
                                    continue

                                for decision in decisions:
                                    logger.info(f"  Applying decision: {decision}")
                                    success, new_func_op, new_transformer = self._try_apply_external_pass(
                                        loop_index=loop_index,
                                        decision=decision,
                                        func_context=ctx,
                                        apply_loop_pass_by_name=apply_loop_pass_by_name,
                                    )

                                    logger.info(f"  Apply result: success={success}")

                                    if not success:
                                        logger.info(f"  Skipping due to failure")
                                        continue

                                    logger.info(f"  Creating new context for generation {ctx.generation + 1}")

                                    new_ctx = ctx.create_child(new_func_op, new_transformer)

                                    all_contexts.append(new_ctx)
                                    contexts_by_func[ctx.func_name].append(new_ctx)

                                    self.compiler.transformers[ctx.func_name] = new_transformer

                                    pass_count += 1
                                    pass_name, param_name, param_value = decision
                                    external_details.append({
                                        'pass': pass_name,
                                        'parameter': param_name,
                                        'value': param_value
                                    })
                                    logger.info(f"  Incremented pass_count to {pass_count}")

                                    logger.info(f"✓ Applied transformation, created generation {new_ctx.generation}")

                except TimeoutException:
                    logger.warning(f"Processing context {ctx_index + 1} timed out after 60s, skipping...")

                ctx_index += 1

            if len(all_contexts) >= max_contexts:
                logger.warning(f"Reached max_contexts limit ({max_contexts}), stopped processing")

            for func_name, contexts in contexts_by_func.items():
                logger.info(f"Function {func_name}: {len(contexts)} versions "
                           f"(generations 0-{contexts[-1].generation if contexts else 0})")

            if self.compiler.state_manager:
                self.compiler.state_manager.record_external_rewrites(pass_count, external_details)

            logger.info(f"Applied {pass_count} external pass instances")
            return pass_count

        except Exception as e:
            raise RuntimeError(f"External rewrites failed: {e}") from e

    def _find_loops_in_function(self, func_op: MOperation) -> List[tuple]:
        """Find all named loops in a function."""

        loops = []  # List of (name, op) tuples
        loop_counter = 0

        def find_loops_recursively(op: MOperation):
            nonlocal loop_counter

            if hasattr(op, 'type'):
                if op.type in [OperationType.SCF_FOR, OperationType.AFFINE_FOR, OperationType.SCF_WHILE]:
                    loops.append((len(loops), op))
            for region in op.get_regions():
                for block in region.get_blocks():
                    for nested_op in block.get_operations():
                        find_loops_recursively(nested_op)

        regions = func_op.get_regions()
        if regions:
            for region in regions:
                for block in region.get_blocks():
                    for op in block.get_operations():
                        find_loops_recursively(op)

        return loops

class Compiler:
    """Simplified top-level compiler for MLIR + e-graph optimization."""

    def __init__(self, module: MModule, target_functions: Optional[List[str]] = None,
                 cost_function: Optional[Any] = None,
                 match_ruleset: Optional[egglog.Ruleset] = None,
                 skeletons: Optional[List] = None,
                 loop_hints: Optional[Dict[str, Dict[str, int]]] = None,
                 instr_encodings: Optional[Dict[str, Dict[str, str]]] = None,
                 debug_callback: Optional[DebugCallback] = None):
        """Initialize compiler with MLIR module."""
        self.original_module = module
        self.target_functions = target_functions or []
        self.cost_function = cost_function if cost_function is not None else MeggCost()
        self.loop_hints = loop_hints or {}  # Store loop optimization hints
        self.instr_encodings = instr_encodings or {}  # Store instruction encodings
        self.debug_callback = debug_callback  # Callback for phase snapshots
        self._phase_index = 0  # Track phase number for snapshots

        self.egraphs: Dict[str, EGraph] = {}
        self.transformers: Dict[str, FuncToTerms] = {}
        self.modified_megg_egraphs: Dict[str, MeggEGraph] = {}
        self.loop_transform_records: Dict[str, List[Dict[str, Any]]] = {}
        self.latest_func_ops: Dict[str, MOperation] = {}
        self._external_pass_modules: Dict[str, List[MModule]] = {}
        self.rewrite_engine = RewriteEngine(self)
        self.state_manager: Optional[StateManager] = None
        self.match_ruleset = match_ruleset
        self.skeletons = skeletons or []  # 新增：控制流骨架列表
        self.custom_instr_properties: Dict[str, Dict[str, Any]] = {}
        self._init_custom_instr_properties()
        self._initialize_egraph()

    def _init_custom_instr_properties(self):
        """Record side-effect metadata and encodings for known custom instructions."""
        self.custom_instr_properties = {}

        for skeleton in self.skeletons:
            props = {
                'has_side_effects': getattr(skeleton, 'has_side_effects', False),
                'clobbers': list(getattr(skeleton, 'clobbers', []) or []),
            }
            if skeleton.instr_name in self.instr_encodings:
                props['encoding'] = self.instr_encodings[skeleton.instr_name]

            self.custom_instr_properties[skeleton.instr_name] = props

        for instr_name, encoding in self.instr_encodings.items():
            if instr_name not in self.custom_instr_properties:
                self.custom_instr_properties[instr_name] = {
                    'has_side_effects': False,  # Pure computation
                    'clobbers': [],
                    'encoding': encoding
                }

    def _initialize_egraph(self):
        """Convert MLIR functions to e-graph terms."""
        logger.info("Converting MLIR functions to e-graph terms")

        try:
            for op in self.original_module.get_functions():
                func_name = op.symbol_name
                logger.info(
                    f"Processing function: {func_name} in targets {self.target_functions}")
                if self.target_functions and func_name not in self.target_functions:
                    continue

                if self.original_module.has_unsupported_ops(op):
                    logger.warning(
                        f"Skipping function {func_name} (unsupported operations)")
                    continue
                logger.info(f"Converting function: {func_name}")
                egraph = EGraph()

                transformer = FuncToTerms.transform(op, egraph)
                logger.info(f"FuncToTerms transformer: {transformer}")
                type_rules = type_annotation_ruleset()
                egraph.run(type_rules.saturate())
                self.egraphs[func_name] = egraph
                self.transformers[func_name] = transformer
                self.latest_func_ops[func_name] = op
                self._external_pass_modules.setdefault(
                    func_name, []).append(self.original_module)

                logger.info(f"Processed function: {func_name}")

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Failed to initialize e-graph: {e}")
            logger.error(f"Traceback:\n{error_details}")
            raise RuntimeError(f"Failed to initialize e-graph: {e}") from e

    def _emit_phase_snapshot(self, phase_name: str, details: Optional[Dict[str, Any]] = None):
        """Emit a snapshot of current compiler state to the debug callback."""
        if self.debug_callback is None:
            return

        try:
            egraph_stats = {}
            for func_name, egraph in self.egraphs.items():
                transformer = self.transformers.get(func_name)
                if transformer:
                    try:
                        megg = MeggEGraph.from_egraph(egraph, func_transformer=transformer)
                        egraph_stats[func_name] = megg.get_statistics()
                    except Exception as e:
                        egraph_stats[func_name] = {"error": str(e)}

            cumulative_stats = {}
            if self.state_manager:
                cumulative_stats = {
                    "internal_rewrites": self.state_manager.internal_rewrites,
                    "external_rewrites": self.state_manager.external_rewrites,
                    "custom_rewrites": self.state_manager.custom_rewrites,
                    "internal_details": self.state_manager.internal_rewrites_details,
                    "external_details": self.state_manager.external_rewrites_details,
                }

            snapshot = PhaseSnapshot(
                phase_name=phase_name,
                phase_index=self._phase_index,
                egraph_stats=egraph_stats,
                cumulative_stats=cumulative_stats,
                details=details,
                timestamp=time.time()
            )

            self._phase_index += 1

            self.debug_callback(snapshot)

        except Exception as e:
            logger.warning(f"Failed to emit phase snapshot: {e}")

    def schedule(self,
                 max_iterations: int = 10,
                 time_limit: float = 60.0,
                 internal_rewrites: bool = True,
                 external_passes: bool = True,
                 custom_rewrites: bool = True,
                 enable_safeguards: bool = True,
                 output_path: Optional[str] = None) -> CompilationResult:
        """Run optimization pipeline: expand e-graph, then match custom instructions."""
        state_manager = StateManager(max_iterations, time_limit)
        self.state_manager = state_manager
        state_manager.start_compilation()

        try:
            logger.info("Starting optimization schedule")
            logger.info(
                "Goal: Expand e-graph with rewrites, then match custom instructions")

            self._collect_initial_egraph_statistics()

            self._emit_phase_snapshot("0_egraph_init", {"description": "E-Graph initialized from MLIR"})

            if internal_rewrites:
                if enable_safeguards and state_manager.check_time_limit():
                    return state_manager.get_result(self.original_module, False, "Time limit exceeded during internal rewrites")
                internal_count = self.rewrite_engine.apply_internal_rewrites()
                state_manager.record_internal_rewrites(internal_count)
                logger.info(
                    f"Phase 1 complete: {internal_count} internal rules applied (e-graph expanded)")

                self._emit_phase_snapshot("1_internal_rewrites", {
                    "description": "Applied algebraic laws (commutativity, associativity, constant folding)",
                    "rules_applied": internal_count
                })

            if external_passes:
                if enable_safeguards and state_manager.check_time_limit():
                    return state_manager.get_result(self._extract_optimized_module(), False, "Time limit exceeded during external rewrites")

                external_count = self.rewrite_engine.apply_external_rewrites()
                logger.info(
                    f"Phase 2 complete: {external_count} external passes applied (e-graph expanded)")

                self._emit_phase_snapshot("2_external_rewrites", {
                    "description": "Applied MLIR loop passes (unroll, tile)",
                    "passes_applied": external_count
                })

                if external_count > 0:
                    logger.info("Phase 2.5: Re-applying internal rewrites to simplify transformed code...")
                    additional_internal = self.rewrite_engine.apply_internal_rewrites()
                    state_manager.record_internal_rewrites(additional_internal)
                    logger.info(f"Phase 2.5 complete: {additional_internal} additional internal rules applied")

                    self._emit_phase_snapshot("2.5_re_simplify", {
                        "description": "Re-applied internal rewrites after loop transforms",
                        "additional_rules": additional_internal
                    })

            if custom_rewrites and self.match_ruleset is not None:
                if enable_safeguards and state_manager.check_time_limit():
                    return state_manager.get_result(self._extract_optimized_module(), False, "Time limit exceeded during custom rewrites")

                custom_count = self.rewrite_engine.apply_custom_rewrites()
                state_manager.record_custom_rewrites(custom_count)
                logger.info(
                    f"Phase 3 complete: {custom_count} custom instruction patterns applied (final pass)")

                self._emit_phase_snapshot("3_custom_rewrites", {
                    "description": "Applied egglog pattern matching rules",
                    "patterns_matched": custom_count
                })

            if custom_rewrites and self.skeletons:
                if enable_safeguards and state_manager.check_time_limit():
                    return state_manager.get_result(self._extract_optimized_module(), False, "Time limit exceeded during skeleton matching")

                skeleton_count = self._apply_skeleton_matching()
                state_manager.record_custom_rewrites(skeleton_count)
                logger.info(
                    f"Phase 3.5 complete: {skeleton_count} complex instructions matched via skeletons")

                self._emit_phase_snapshot("3.5_skeleton_matching", {
                    "description": "Applied skeleton-based control flow matching",
                    "skeletons_matched": skeleton_count,
                    "skeleton_names": [s.instr_name for s in self.skeletons]
                })

            self._collect_egraph_statistics()

            self._emit_phase_snapshot("4_before_extraction", {
                "description": "E-Graph saturated, ready for extraction"
            })

            optimized_module = self._extract_optimized_module(
                output_path=output_path)

            result = state_manager.get_result(optimized_module, True)
            logger.info(
                f"Optimization successful: {result.internal_rewrites} internal + {result.external_rewrites} external + {result.custom_rewrites} custom in {result.time_elapsed:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return state_manager.get_result(self.original_module, False, str(e))

    def _extract_optimized_module(self, output_path: Optional[str] = None) -> MModule:
        """Reconstruct complete functions from their top-level block roots."""
        optimized_module = self.original_module.empty_like()
        builder = IRBuilder()
        with builder.set_context(optimized_module.get_context()):
            for operation in self.original_module.get_operations():
                parent = operation.get_parent_op()
                if parent is not None and parent.name == 'builtin.module' and operation.type != OperationType.FUNC_FUNC:
                    optimized_module.append_to_module(operation)
            for original_func in self.original_module.get_functions():
                func_name = original_func.symbol_name
                if func_name not in self.transformers:
                    optimized_module.append_to_module(original_func)
                    continue
                try:
                    transformer = self.transformers[func_name]
                    graph = self.modified_megg_egraphs.get(func_name)
                    if graph is None:
                        graph = MeggEGraph.from_egraph(
                            self.egraphs[func_name], func_transformer=transformer)
                    if not graph.root_eclasses:
                        raise ValueError("Function has no top-level block root")
                    extractor = Extractor(graph, self.cost_function)
                    body_exprs = [extractor.find_best(root).expr for root in graph.root_eclasses]
                    with timeout(60):
                        function = ExprTreeToMLIR.reconstruct(
                            original_func, body_exprs=body_exprs, output_terms=body_exprs,
                            instr_properties=self.custom_instr_properties,
                            target_module=optimized_module)
                    if function is None:
                        raise ValueError("Reconstruction returned no function")
                    optimized_module.append_to_module(function)
                except Exception as exc:
                    raise RuntimeError(f"Failed to reconstruct function {func_name}: {exc}") from exc
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(optimized_module.to_string())
        # Keep the owning module alive while bindings reference its operations.
        self._optimized_module_cache = optimized_module
        return optimized_module

    def visualize_egraph(self, output_path: str, format: str = "svg", **kwargs):
        """Generate e-graph visualization."""
        egraph = next(iter(self.egraphs.values()), None)
        if egraph is None:
            logger.warning("No e-graph available for visualization")
            return None
        src = egraph._graphviz(**kwargs)
        src.format = format

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = src.render(filename=str(
            path.with_suffix('')), cleanup=True, quiet=True)
        logger.info(f"EGraph visualization saved to: {result}")
        return result

    def _apply_skeleton_matching(self) -> int:

        match_count = 0
        for func_name, egraph in self.egraphs.items():
            graph = MeggEGraph.from_egraph(
                egraph, func_transformer=self.transformers[func_name])
            matcher = SkeletonMatcher(graph, cost_function=self.cost_function)
            for skeleton in self.skeletons:
                for eclass_id, bindings in matcher.match_skeleton(skeleton):
                    operands = [bindings[_get_var_name(arg)] for arg in skeleton.arg_vars]
                    graph.add_custom_instr_node(
                        eclass_id=eclass_id, instr_name=skeleton.instr_name,
                        operands=operands,
                        result_type=self._extract_result_type(skeleton.result_type))
                    match_count += 1
            self.modified_megg_egraphs[func_name] = graph
        return match_count

    def _extract_result_type(self, result_type) -> str:
        """Extract type string from egglog.String or other representations."""
        if result_type is None:
            return "void"

        type_str = str(result_type)

        if type_str.startswith('String("') and type_str.endswith('")'):
            return type_str[8:-2]  # Remove 'String("' and '")'

        return type_str.strip('"')

    def _collect_initial_egraph_statistics(self):
        self._collect_egraph_statistics(initial=True)

    def _collect_egraph_statistics(self, initial: bool = False):
        totals = {'num_eclasses': 0, 'num_enodes': 0}
        for func_name, egraph in self.egraphs.items():
            graph = MeggEGraph.from_egraph(
                egraph, func_transformer=self.transformers[func_name])
            stats = graph.get_statistics()
            totals['num_eclasses'] += stats['total_eclasses']
            totals['num_enodes'] += stats['total_nodes']
            logger.debug("E-graph statistics for %s: %s", func_name, stats)
        if self.state_manager:
            if initial:
                self.state_manager.record_initial_egraph_statistics(totals)
            else:
                self.state_manager.record_egraph_statistics(totals)

    def generate_statistics_report(self) -> Dict[str, Any]:
        """Generate a comprehensive statistics report suitable for JSON output."""
        if not self.state_manager:
            return {}

        used_internal_count = len([
            r for r in self.state_manager.internal_rewrites_details
            if r.get('num_matches', 0) > 0
        ])

        return {
            'internal_rewrites': used_internal_count,
            'external_rewrites': self.state_manager.external_rewrites_details,
            'custom_rewrites': self.state_manager.custom_rewrites,
            'initial_egraph_statistics': self.state_manager.initial_egraph_statistics,
            'egraph_statistics': self.state_manager.egraph_statistics,
            'time_elapsed': time.time() - self.state_manager.start_time if self.state_manager.start_time else 0
        }
