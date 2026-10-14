"""自定义指令匹配的 rewrite 规则生成"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import egglog

from megg.egraph.datatype import DataType
from megg.egraph.extract import AstSize, Extractor
from megg.egraph.func_to_terms import (
    FuncToTerms,
    mlir_type_to_egraph_ty_string,
    mlir_type_to_megg_type,
)
from megg.egraph.megg_egraph import MeggEGraph
from megg.egraph.term import Term
from megg.egraph.terms_to_func import ExprTreeToMLIR
from megg.rewrites.internal_rewrites import (
    basic_math_laws,
    constant_folding_laws,
    type_annotation_ruleset,
)
from megg.utils.ir_builder import IRBuilder
from megg.utils.mlir_utils import MBlock, MModule, MOperation, MValue

logger = logging.getLogger(__name__)

def normalize_pattern_module(pattern_module: MModule, verbose: bool = False) -> MModule:
    """Normalize pattern functions through the same egraph extraction pipeline."""

    normalized_module = pattern_module.empty_like()
    rulesets = (basic_math_laws(), constant_folding_laws(), type_annotation_ruleset())
    with IRBuilder().set_context(normalized_module.get_context()):
        for function in pattern_module.get_functions():
            graph = egglog.EGraph()
            transformer = FuncToTerms.transform(function, graph)
            for _ in range(5):
                for ruleset in rulesets:
                    graph.run(ruleset.saturate())
            megg_graph = MeggEGraph.from_egraph(graph, func_transformer=transformer)
            if not megg_graph.root_eclasses:
                raise ValueError(f"Pattern {function.symbol_name} has no block root")
            extractor = Extractor(megg_graph, AstSize())
            expressions = [extractor.find_best(root).expr for root in megg_graph.root_eclasses]
            normalized = ExprTreeToMLIR.reconstruct(
                original_func=function, body_exprs=expressions, output_terms=expressions,
                target_module=normalized_module)
            if normalized is None:
                raise ValueError(f"Pattern {function.symbol_name} reconstruction failed")
            normalized_module.append_to_module(normalized)
    if verbose:
        logger.debug("Normalized pattern module: %s", normalized_module)
    return normalized_module

def _extract_constant_value(value: MValue) -> Optional[int]:
    operation = value.get_defining_op()
    if operation is None or operation.name != 'arith.constant':
        return None
    return int(operation.get_attr('value'))

def _get_var_name(var) -> str:
    """Return a consistent string identifier for egglog variables."""
    name = getattr(var, "name", None)
    if isinstance(name, str):
        return name

    text = str(var)
    if text.startswith('<Var ') and text.endswith('>'):
        return text[5:-1].strip()
    return text

def simplify_pattern_str(pattern_str: str) -> str:
    """Simplify a pattern string for display."""

    s = pattern_str

    s = re.sub(r'\bTerm\.', '', s)

    s = re.sub(r'Vec\[Term\]\(([^)]*)\)', r'\1', s)

    s = re.sub(r'egglog\.String\("([^"]*)"\)', r'"\1"', s)
    s = re.sub(r'String\("([^"]*)"\)', r'"\1"', s)

    s = re.sub(r'egglog\.i64\((\d+)\)', r'\1', s)
    s = re.sub(r'i64\((\d+)\)', r'\1', s)

    s = re.sub(r'"__expr_\w+"', '...', s)

    s = re.sub(r'\b_arg(\d+)\b', r'arg\1', s)

    s = re.sub(r'\b(\w+)_\b', r'\1', s)

    s = re.sub(r'\s+', ' ', s).strip()

    return s

def _extract_used_args_from_pattern(pattern: Term, arg_vars: List[Term]) -> List[Term]:
    """Extract which argument variables are actually used in the pattern."""
    pattern_str = str(pattern)
    used_args = []

    for var in arg_vars:
        var_name = _get_var_name(var)
        if re.search(r'\b' + re.escape(var_name) + r'\b', pattern_str):
            used_args.append(var)

    return used_args

@dataclass
class SkeletonStmt:
    """Block 中的一个 statement（Vec 元素）"""
    name: str
    pattern_term: Optional[Term] = None  # 叶子 pattern
    nested_skeleton: Optional['SkeletonNode'] = None  # 嵌套控制流
    operand_terms: List[Term] = field(default_factory=list)  # 参与匹配的操作数

    def is_leaf(self) -> bool:
        return self.pattern_term is not None

    def is_nested(self) -> bool:
        return self.nested_skeleton is not None

@dataclass
class SkeletonBlock:
    """一个 block（包含 statements）"""
    name: str
    statements: List[SkeletonStmt] = field(default_factory=list)

@dataclass
class SkeletonNode:
    """控制流节点（包含一个或多个 blocks）"""
    container_type: str  # "func.body", "scf.for", "scf.if", "scf.while"
    blocks: List[SkeletonBlock] = field(default_factory=list)
    result_type: Optional[str] = None
    condition_term: Optional[Term] = None  # scf.if的condition, scf.while的condition
    init_values_term: Optional[Term] = None  # scf.for的init_vals, scf.while的init_vals
    loop_bounds: Optional[Dict[str, Any]] = None  # scf.for的lower/upper/step (存储常量值)

@dataclass
class Skeleton:
    """完整的控制流骨架"""
    instr_name: str
    root: SkeletonNode
    leaf_patterns: Dict[str, Term] = field(default_factory=dict)
    leaf_operands: Dict[str, List[Term]] = field(default_factory=dict)
    arg_vars: List = field(default_factory=list)  # 函数参数的 generic variables
    result_type: Optional[egglog.String] = None  # 函数返回类型
    arg_var_to_index: Dict[str, int] = field(default_factory=dict)
    arg_types: Dict[str, DataType] = field(default_factory=dict)
    operand_constraints: List[Tuple[str, int, str]] = field(default_factory=list)
    has_side_effects: bool = False
    clobbers: List[str] = field(default_factory=list)

    def format_tree(self) -> str:
        """Format skeleton as a tree structure like for(for(stmt0), for(stmt1, stmt2))"""
        return self._format_node(self.root)

    def _format_node(self, node: SkeletonNode) -> str:
        """Recursively format a skeleton node."""
        type_name = node.container_type.replace("scf.", "").replace("func.", "")

        children = []
        for block in node.blocks:
            for stmt in block.statements:
                if stmt.is_nested():
                    children.append(self._format_node(stmt.nested_skeleton))
                else:
                    children.append(stmt.name)

        if children:
            return f"{type_name}({', '.join(children)})"
        else:
            return type_name

    def add_leaf_pattern(self, name: str, pattern: Term, operands: Optional[List[Term]] = None):
        full_name = f"{self.instr_name}_{name}"
        self.leaf_patterns[full_name] = pattern
        operands = operands or []

        arg_var_set = set(self.arg_vars) if self.arg_vars else set()

        filtered_operands: List[Term] = [op for op in operands if op in arg_var_set]

        if self.arg_vars:
            # CSR parameters and input pointers that must also be bound.
            filtered_operands = self._extract_arg_vars_from_pattern(pattern)

        self.leaf_operands[full_name] = filtered_operands

    def _extract_arg_vars_from_pattern(self, pattern: Term) -> List[Term]:
        """Extract argument variables from pattern Term in the correct order."""
        result = []
        pattern_str = str(pattern)

        var_name_to_var: Dict[str, Term] = {}
        for var in self.arg_vars:
            var_name = _get_var_name(var)
            var_name_to_var[var_name] = var

        var_pattern = r'\b(' + '|'.join(re.escape(_get_var_name(v)) for v in self.arg_vars) + r')\b'
        matches = re.finditer(var_pattern, pattern_str)

        seen = set()
        for match in matches:
            var_name = match.group(1)
            if var_name not in seen:
                seen.add(var_name)
                result.append(var_name_to_var[var_name])

        return result

    def add_operand_constraint(self, component_name: str, operand_idx: int, var_name: str):
        constraint = (component_name, operand_idx, var_name)
        if constraint not in self.operand_constraints:
            self.operand_constraints.append(constraint)

def _populate_operand_constraints(skeleton: Skeleton) -> None:
    """Populate argument index mapping and operand equality constraints."""
    if skeleton.arg_vars:
        skeleton.arg_var_to_index = {
            _get_var_name(var): idx for idx, var in enumerate(skeleton.arg_vars)
        }
    else:
        skeleton.arg_var_to_index = {}

    for component_name, operands in skeleton.leaf_operands.items():
        for idx, term in enumerate(operands):
            var_name = _get_var_name(term)
            skeleton.add_operand_constraint(component_name, idx, var_name)

def _specialize_leaf_pattern(
    skeleton: Skeleton,
    component_name: str,
    pattern: Term,
    operand_terms: List[Term],
) -> Tuple[Term, List[Term]]:
    """Make component rewrites more specific when the pattern is ambiguous."""

    if len(operand_terms) != 1:
        return pattern, operand_terms

    operand = operand_terms[0]
    var_name = _get_var_name(operand)
    if not var_name:
        return pattern, operand_terms

    arg_index = skeleton.arg_var_to_index.get(var_name)
    if arg_index is None:
        return pattern, operand_terms

    dtype = skeleton.arg_types.get(var_name)
    if dtype is None:
        return pattern, operand_terms

    pattern_str = str(pattern).replace(" ", "")
    candidates = {
        f'Term.yield_(Vec[Term]({var_name}),"void")',
        f'Term.yield_(Vec[Term](_{var_name}),"void")',
    }
    if pattern_str not in candidates:
        return pattern, operand_terms

    arg_term = Term.arg(egglog.i64(arg_index), egglog.String(str(dtype)))
    specialized_pattern = Term.yield_(egglog.Vec[Term](arg_term), egglog.String("void"))

    return specialized_pattern, [arg_term]

def _instruction_name(func_op: MOperation) -> str:
    """提取函数名"""
    return func_op.symbol_name

def _process_block_statements(
    block: MBlock,
    block_name: str,
    helper: FuncToTerms,
    ssa_to_term: Dict[MValue, Term],
    instr_name: str,
    stmt_counter: int
) -> Tuple[SkeletonBlock, int]:
    """处理 block 中的 statements（Vec 元素）"""
    skeleton_block = SkeletonBlock(name=block_name)

    local_ssa_to_term = dict(ssa_to_term)
    block_args = block.arguments
    unmapped_args = [arg for arg in block_args if arg not in local_ssa_to_term]

    if len(unmapped_args) > 0:
        raw_block = getattr(block, '_block', None)
        block_id = id(raw_block) if raw_block is not None else id(block)
        block_tag = f"{block_name}_{block_id}"
        for i, arg in enumerate(unmapped_args):
            if i == 0:
                var_name = f"{block_tag}_loop_idx"
            else:
                var_name = f"{block_tag}_acc{i-1}"
            local_ssa_to_term[arg] = egglog.var(var_name, Term)

    operations = block.get_operations()

    for op in operations:
        if op.name in ['scf.for', 'scf.if', 'scf.while']:
            for i, res in enumerate(op.results):
                cf_var = egglog.var(f"cf_{len(local_ssa_to_term)}_{i}", Term)
                local_ssa_to_term[res] = cf_var
            continue

        operand_terms = [local_ssa_to_term[o] for o in op.operands if o in local_ssa_to_term]
        result_term = helper._operation_to_term(op, operand_terms)
        if result_term is not None:
            if op.results:
                local_ssa_to_term[op.results[0]] = result_term
            else:
                local_ssa_to_term[op] = result_term

    for op in operations:
        if op.name in ['scf.for', 'scf.if', 'scf.while']:
            stmt_name = f"{block_name}_stmt{stmt_counter}"
            stmt_counter += 1
            nested_node, stmt_counter = _process_control_flow(
                op,
                op.name,
                helper,
                local_ssa_to_term,
                instr_name,
                stmt_counter
            )
            stmt = SkeletonStmt(name=stmt_name, nested_skeleton=nested_node)
            skeleton_block.statements.append(stmt)
            continue

        # 仅将副作用操作视为 Vec 元素（store/yield）
        is_vec_element = op.name in ['scf.yield', 'memref.store', 'memref.alloc', 'memref.alloca']
        if not is_vec_element:
            continue

        stmt_name = f"{block_name}_stmt{stmt_counter}"
        stmt_counter += 1
        pattern_term = None

        if op.results:
            pattern_term = local_ssa_to_term.get(op.results[0])
        elif op in local_ssa_to_term:
            pattern_term = local_ssa_to_term[op]

        stmt_operands: List[Term] = []
        for operand in op.operands:
            if operand in local_ssa_to_term:
                stmt_operands.append(local_ssa_to_term[operand])
            else:
                pass

        if op.name == 'scf.yield':
            if len(op.operands)==0:
                continue

        stmt = SkeletonStmt(
            name=stmt_name,
            pattern_term=pattern_term,
            operand_terms=stmt_operands
        )
        skeleton_block.statements.append(stmt)

    return skeleton_block, stmt_counter

def _process_control_flow(
    control_op: MOperation,
    control_type: str,
    helper: FuncToTerms,
    ssa_to_term: Dict[MValue, Term],
    instr_name: str,
    stmt_counter: int
) -> Tuple[SkeletonNode, int]:
    """处理控制流操作，生成 nested skeleton node"""
    regions = control_op.get_regions()

    if control_type == "scf.for":
        node = SkeletonNode(container_type="scf.for")

        if control_op.operands and len(control_op.operands) >= 3:
            lower_value = control_op.operands[0]
            upper_value = control_op.operands[1]
            step_value = control_op.operands[2]

            lower_bound = _extract_constant_value(lower_value)
            upper_bound = _extract_constant_value(upper_value)
            step = _extract_constant_value(step_value)

            if lower_bound is not None and upper_bound is not None and step is not None:
                node.loop_bounds = {
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'step': step
                }
                logger.info(f"  Extracted loop bounds: lower={lower_bound}, upper={upper_bound}, step={step}")
            else:
                logger.debug(f"  Could not extract all loop bounds (lower={lower_bound}, upper={upper_bound}, step={step})")

        body_mlir_block = regions[0].get_blocks()[0]
        body_block, stmt_counter = _process_block_statements(
            body_mlir_block,
            "body",
            helper,
            ssa_to_term,
            instr_name,
            stmt_counter
        )
        node.blocks.append(body_block)

    elif control_type == "scf.if":
        node = SkeletonNode(container_type="scf.if")

        if control_op.operands and len(control_op.operands) > 0:
            condition_value = control_op.operands[0]
            condition_term = ssa_to_term.get(condition_value)
            if condition_term is not None:
                node.condition_term = condition_term

        then_mlir_block = regions[0].get_blocks()[0]
        then_block, stmt_counter = _process_block_statements(
            then_mlir_block,
            "then",
            helper,
            ssa_to_term,
            instr_name,
            stmt_counter
        )
        node.blocks.append(then_block)

        if len(regions) > 1:
            else_mlir_block = regions[1].get_blocks()[0]
            else_block, stmt_counter = _process_block_statements(
                else_mlir_block,
                "else",
                helper,
                ssa_to_term,
                instr_name,
                stmt_counter
            )
            node.blocks.append(else_block)

    elif control_type == "scf.while":
        node = SkeletonNode(container_type="scf.while")

        before_mlir_block = regions[0].get_blocks()[0]
        before_block, stmt_counter = _process_block_statements(
            before_mlir_block,
            "before",
            helper,
            ssa_to_term,
            instr_name,
            stmt_counter
        )
        node.blocks.append(before_block)

        if len(regions) > 1:
            after_mlir_block = regions[1].get_blocks()[0]
            after_block, stmt_counter = _process_block_statements(
                after_mlir_block,
                "after",
                helper,
                ssa_to_term,
                instr_name,
                stmt_counter
            )
            node.blocks.append(after_block)

    else:
        raise ValueError(f"Unknown control flow type: {control_type}")

    return node, stmt_counter

def _build_skeleton_from_func(func_op: MOperation) -> Tuple[Optional[Skeleton], Optional[Tuple[Term, egglog.String, List]]]:
    """从函数构建 skeleton 或 simple pattern"""
    instr_name = _instruction_name(func_op)

    egraph = egglog.EGraph()
    helper = FuncToTerms(
        func=func_op,
        egraph=egraph,
        next_id=0,
        ssa_to_id={},
        ssa_to_term={},
        loop_to_term={},
        top_block=None
    )

    entry_block = func_op.get_regions()[0].get_blocks()[0]
    num_args = len(entry_block.arguments)
    arg_vars: List[Term] = []
    arg_types_map: Dict[str, DataType] = {}

    if num_args > 0:
        arg_names = ' '.join(f"arg{i}" for i in range(num_args))
        arg_vars = list(egglog.vars_(arg_names, Term))

    ssa_to_term: Dict[MValue, Term] = {}
    for i, arg in enumerate(entry_block.arguments):
        if i >= len(arg_vars):
            var = egglog.var(f"arg{i}", Term)
            arg_vars.append(var)
        else:
            var = arg_vars[i]

        ssa_to_term[arg] = var

        var_name = _get_var_name(var)
        arg_type = mlir_type_to_megg_type(arg.type)
        if arg_type is not None:
            arg_types_map[var_name] = arg_type

    # 简单模式：无控制流、无副作用（除了 func.return）
    operations = entry_block.get_operations()

    has_control_flow = False
    has_side_effects = False
    return_op = None

    for op in operations:
        if op.name in ['scf.for', 'scf.if', 'scf.while']:
            has_control_flow = True
            # FIXME: 暂时不考虑memref.load作为副作用
        elif op.name in ['scf.yield', 'memref.store', 'scf.cond', 'memref.alloc', 'memref.alloca']:
            has_side_effects = True
        elif op.name == 'func.return':
            return_op = op

    # 情况1：简单计算模式（无控制流、无副作用）
    if not has_control_flow and not has_side_effects and return_op is not None:

        local_ssa_to_term = dict(ssa_to_term)

        for op in operations:
            if op.name == 'func.return':
                continue

            operand_terms = []
            for operand in op.operands:
                if operand in local_ssa_to_term:
                    operand_terms.append(local_ssa_to_term[operand])

            result_term = helper._operation_to_term(op, operand_terms)
            if result_term is not None and op.results:
                local_ssa_to_term[op.results[0]] = result_term

        if return_op.operands:
            return_value = return_op.operands[0]
            if return_value in local_ssa_to_term:
                pattern = local_ssa_to_term[return_value]
                result_type = mlir_type_to_egraph_ty_string(return_value.type)
                return None, (pattern, result_type, arg_vars)

        return None, None

    root = SkeletonNode(container_type="func.body")

    body_block, _ = _process_block_statements(
        entry_block,
        "body",
        helper,
        ssa_to_term,
        instr_name,
        0
    )
    root.blocks.append(body_block)

    for stmt in body_block.statements:
        if stmt.is_nested():
            pass
        else:
            pass

    if len(body_block.statements) == 1 and body_block.statements[0].is_nested():
        root = body_block.statements[0].nested_skeleton

    result_type = None
    if return_op and return_op.operands:
        result_type = mlir_type_to_egraph_ty_string(return_op.operands[0].type)

    skeleton = Skeleton(
        instr_name=instr_name,
        root=root,
        arg_vars=arg_vars,
        result_type=result_type
    )
    skeleton.arg_types = arg_types_map
    skeleton.has_side_effects = has_side_effects
    skeleton.clobbers = ['memory'] if has_side_effects else []

    _extract_leaf_patterns_recursive(skeleton.root, skeleton)

    _populate_operand_constraints(skeleton)

    return skeleton, None

def _extract_leaf_patterns_recursive(node: SkeletonNode, skeleton: Skeleton):
    """递归提取所有 leaf patterns"""
    for block in node.blocks:
        for stmt in block.statements:
            if stmt.is_leaf():
                skeleton.add_leaf_pattern(stmt.name, stmt.pattern_term, stmt.operand_terms)
            elif stmt.is_nested():
                _extract_leaf_patterns_recursive(stmt.nested_skeleton, skeleton)

def build_ruleset_from_module(module: MModule, normalize: bool = True, verbose: bool = False,
                              instr_encodings: Optional[Dict[str, dict]] = None) -> Tuple[egglog.Ruleset, List[Skeleton]]:
    """从模块构建 ruleset 和 skeletons"""
    if normalize:
        module = normalize_pattern_module(module, verbose=verbose)

    rewrites: List[egglog.Rewrite] = []
    skeletons: List[Skeleton] = []

    for func_op in module.get_functions():
        instr_name = _instruction_name(func_op)

        try:
            skeleton, simple_pattern_with_type_and_args = _build_skeleton_from_func(func_op)

            if simple_pattern_with_type_and_args is not None:
                pattern, result_type, arg_vars = simple_pattern_with_type_and_args

                used_args = _extract_used_args_from_pattern(pattern, arg_vars)
                if instr_encodings is not None and instr_name in instr_encodings:
                    # positions so CSR parameters cannot become rs1/rs2 operands.
                    used_names = {_get_var_name(used) for used in used_args}
                    instr_encodings[instr_name]['operand_indices'] = [
                        i for i, arg in enumerate(arg_vars)
                        if _get_var_name(arg) in used_names
                    ]

                if len(used_args) < len(arg_vars):
                    unused_count = len(arg_vars) - len(used_args)

                operands_vec = egglog.Vec[Term](*used_args) if used_args else egglog.Vec[Term]()
                custom_instr = Term.custom_instr(
                    egglog.String(instr_name),
                    operands_vec,
                    result_type
                )
                rewrite = egglog.rewrite(pattern).to(custom_instr)
                rewrites.append(rewrite)

            elif skeleton is not None:
                # Bind only actual dependencies; never guess unused operands.
                used_names = {_get_var_name(arg) for pattern in skeleton.leaf_patterns.values()
                              for arg in _extract_used_args_from_pattern(pattern, skeleton.arg_vars)}
                indices = [i for i, arg in enumerate(skeleton.arg_vars)
                           if _get_var_name(arg) in used_names]
                if instr_encodings is not None and instr_name in instr_encodings:
                    instr_encodings[instr_name]['operand_indices'] = indices
                skeleton.arg_vars = [skeleton.arg_vars[i] for i in indices]

                for full_name, pattern in skeleton.leaf_patterns.items():
                    operand_terms = skeleton.leaf_operands.get(full_name, [])
                    if not operand_terms:
                        continue

                    specialized_pattern, specialized_operands = _specialize_leaf_pattern(
                        skeleton,
                        full_name,
                        pattern,
                        operand_terms,
                    )

                    operand_vec = (
                        egglog.Vec[Term](*specialized_operands)
                        if specialized_operands else egglog.Vec[Term]()
                    )
                    comp_instr = Term.component_instr(
                        egglog.String(full_name),
                        operand_vec,
                        egglog.String("void")
                    )
                    pattern_str = str(specialized_pattern)
                    pattern_len = len(pattern_str)

                    simplified = simplify_pattern_str(pattern_str)
                    short_name = full_name.replace(f"{instr_name}_", "")

                    if pattern_len > 5000:
                        pass

                    rewrite = egglog.rewrite(specialized_pattern).to(comp_instr)
                    rewrites.append(rewrite)

                skeletons.append(skeleton)

        except Exception as exc:
            raise ValueError(f"Failed to build instruction pattern {instr_name}: {exc}") from exc

    if rewrites:
        ruleset = egglog.ruleset(*rewrites, name="match_rewrite")
    else:
        ruleset = egglog.ruleset(name="match_rewrite")

    return ruleset, skeletons
