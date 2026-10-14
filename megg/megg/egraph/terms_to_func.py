"""MeggEGraph Expression Tree to MLIR Conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from megg.backend.llvm_backend import MLIRToLLVMBackend
from megg.egraph.datatype import IntType, VoidType
from megg.egraph.func_to_terms import megg_to_mlir_type
from megg.egraph.megg_egraph import ExpressionNode
from megg.utils import IRBuilder, MBlock, MModule, MOperation, MType, MValue


@dataclass
class ExprTreeToMLIR:
    """Convert MeggEGraph expression trees to MLIR operations."""

    original_func: MOperation

    body_exprs: List[ExpressionNode]

    output_terms: List[Any] = field(default_factory=list)

    expr_to_ssa: Dict[Tuple[str, int], MValue] = field(default_factory=dict)

    reconstructed_func: Optional[MOperation] = None

    constants_cache: Dict[tuple, MValue] = field(default_factory=dict)

    loop_contexts: Dict[str, MBlock] = field(default_factory=dict)
    loop_block_stack: List[Tuple[MBlock, Set[str]]] = field(default_factory=list)
    builder: IRBuilder = IRBuilder()
    instr_properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    current_block_context: Optional[MBlock] = None

    block_first_nested_op: Dict[int, Any] = field(default_factory=dict)

    target_module: Optional[Any] = None

    @classmethod
    def reconstruct(cls, original_func: MOperation, body_exprs: List[ExpressionNode], output_terms: List[Any] = None,
                    instr_properties: Optional[Dict[str, Dict[str, Any]]] = None,
                    target_module: Optional[Any] = None) -> MOperation:
        """Reconstruct a FuncOp from MeggEGraph expression trees."""
        converter = cls(
            original_func=original_func,
            body_exprs=body_exprs,
            output_terms=output_terms or [],
            instr_properties=instr_properties or {},
            target_module=target_module
        )
        converter._reconstruct()
        return converter.reconstructed_func

    def _has_custom_instruction(self, exprs: List[ExpressionNode]) -> bool:
        """Check if expression tree contains custom instructions."""
        def traverse(expr: ExpressionNode) -> bool:
            if expr.op == 'Custom_instr':
                return True
            for child in expr.children:
                if traverse(child):
                    return True
            return False

        for expr in exprs:
            if traverse(expr):
                return True
        return False

    def _reconstruct(self):
        """Main reconstruction method."""
        logger = logging.getLogger(__name__)

        func_name = self.original_func.symbol_name
        logger.info(f"[RECONSTRUCT] Starting reconstruction of {func_name}")
        func_type = self.original_func.get_function_type
        arg_types = func_type.get_inputs()
        outputs = func_type.get_results()
        logger.info(f"[RECONSTRUCT] Function has {len(arg_types)} args, {len(outputs)} outputs")

        logger.info(f"[RECONSTRUCT] Checking for custom instructions...")
        has_custom_instr = self._has_custom_instruction(self.body_exprs)
        logger.info(f"[RECONSTRUCT] Has custom instructions: {has_custom_instr}")

        # If has custom instructions, replace memref args with llvm.ptr for ABI compatibility
        if has_custom_instr:
            modified_arg_types = []
            for arg_type in arg_types:
                type_str = str(arg_type)
                if type_str.startswith('memref<'):
                    # This prevents ABI mismatch (memref=5 params vs ptr=1 param)
                    modified_arg_types.append(self.builder.llvm_ptr())
                else:
                    modified_arg_types.append(arg_type)
            arg_types = modified_arg_types

        if self.target_module is not None:
            with self.builder.set_context(self.target_module.get_context()):
                self.reconstructed_func, entry_block = self.builder.create_function(
                    func_name, arg_types, outputs)
        else:
            self.reconstructed_func, entry_block = self.builder.create_function(
                func_name, arg_types, outputs)
        for i, arg in enumerate(entry_block.arguments):
            arg_key = f"arg_{i}"
            self.expr_to_ssa[arg_key] = arg

        self.current_block_context = entry_block

        logger.info(f"[RECONSTRUCT] Reconstructing {len(self.body_exprs)} body expressions")
        last_expr_results: List[MValue] = []
        for idx, body_expr in enumerate(self.body_exprs):
            logger.info(f"[RECONSTRUCT] Processing body_expr {idx+1}/{len(self.body_exprs)}: op={body_expr.op}")
            result = self._reconstruct_expr(body_expr, entry_block)
            logger.info(f"[RECONSTRUCT] Completed body_expr {idx+1}/{len(self.body_exprs)}")
            if result is None:
                continue
            if isinstance(result, MValue):
                last_expr_results = [result]
            elif hasattr(result, 'results'):
                op_results = result.results
                if op_results:
                    last_expr_results = op_results

        logger.info(f"[RECONSTRUCT] Ensuring block termination...")
        self._ensure_block_terminated(entry_block, outputs, last_expr_results)
        logger.info(f"[RECONSTRUCT] Reconstruction complete for {func_name}")

    def _reconstruct_expr(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Recursively reconstruct an expression tree to MLIR operations."""
        # Check if already reconstructed (but skip cache for literals - they must be scope-aware)
        target_block = self.current_block_context if self.current_block_context else block
        cache_key = (expr.node_id, id(target_block))

        if expr.op != 'Lit' and cache_key in self.expr_to_ssa:
            cached = self.expr_to_ssa[cache_key]
            if cached == "PROCESSED":
                return None
            return cached

        op_handlers = {
            'Lit': self._reconstruct_literal,
            'Arg': self._reconstruct_argument,
            'Loop_index': self._reconstruct_loop_index,
            'Loop_iter_arg': self._reconstruct_loop_iter_arg,
            'Block_arg': self._reconstruct_block_arg,

            'Add': self._reconstruct_binary_op,
            'Sub': self._reconstruct_binary_op,
            'Mul': self._reconstruct_binary_op,
            'Div': self._reconstruct_binary_op,
            'Rem': self._reconstruct_binary_op,

            'And': self._reconstruct_binary_op,
            'And_': self._reconstruct_binary_op,  # Alternative name from extraction
            'Or': self._reconstruct_binary_op,
            'Or_': self._reconstruct_binary_op,  # Alternative name from extraction
            'Xor': self._reconstruct_binary_op,
            'Xor_': self._reconstruct_binary_op,  # Alternative name from extraction

            'Shl': self._reconstruct_binary_op,
            'ShrSI': self._reconstruct_binary_op,
            'Shrsi': self._reconstruct_binary_op,  # Alternative capitalization
            'ShrUI': self._reconstruct_binary_op,
            'Shrui': self._reconstruct_binary_op,  # Alternative capitalization

            'Neg': self._reconstruct_unary_op,

            'Cmp': self._reconstruct_comparison,
            'Select': self._reconstruct_select,

            'Index_cast': self._reconstruct_cast,
            'SIToFP': self._reconstruct_cast,
            'Sitofp': self._reconstruct_cast,  # Alternative capitalization
            'UIToFP': self._reconstruct_cast,
            'Uitofp': self._reconstruct_cast,  # Alternative capitalization
            'FPToSI': self._reconstruct_cast,
            'Fptosi': self._reconstruct_cast,  # Alternative capitalization
            'FPToUI': self._reconstruct_cast,
            'Fptoui': self._reconstruct_cast,  # Alternative capitalization
            'ExtSI': self._reconstruct_cast,
            'Extsi': self._reconstruct_cast,  # Alternative capitalization
            'ExtUI': self._reconstruct_cast,
            'Extui': self._reconstruct_cast,  # Alternative capitalization
            'TruncI': self._reconstruct_cast,
            'Trunci': self._reconstruct_cast,  # Alternative capitalization
            'Bitcast': self._reconstruct_cast,

            'If_': self._reconstruct_if,
            'While_': self._reconstruct_while,
            'For_': self._reconstruct_for,
            'For_with_carry': self._reconstruct_for_with_carry,
            'Affine_for': self._reconstruct_affine_for,
            'Affine_for_with_carry': self._reconstruct_affine_for_with_carry,
            'Block': self._reconstruct_block,
            'Yield_': self._reconstruct_yield,
            'Affine_yield': self._reconstruct_affine_yield,
            'Custom_instr': self._reconstruct_custom_instr,
            'Return_': self._reconstruct_return,
            'Condition': self._reconstruct_condition,

            'Alloc': self._reconstruct_memref_alloc,
            'Alloca': self._reconstruct_memref_alloca,
            'Get_global': self._reconstruct_memref_get_global,
            'Store': self._reconstruct_memref_store,
            'Load': self._reconstruct_memref_load,

            "Vec": self._reconstruct_vec,
        }

        handler = op_handlers.get(expr.op)
        if handler is None:
            raise ValueError(f"Unsupported operation: {expr.op!r}")

        result = handler(expr, block)

        # Cache the result, but NOT for literals (constants must be scope-aware)
        if result is not None and expr.op != 'Lit':
            self.expr_to_ssa[cache_key] = result
        elif expr.op in ['Yield_', 'Affine_yield', 'Return_', 'Condition']:
            self.expr_to_ssa[cache_key] = "PROCESSED"

        return result

    @staticmethod
    def _canonical_op(op: Optional[str]) -> str:
        if not op:
            return ''
        if op == 'For_':
            return 'For'
        if op == 'While_':
            return 'While'
        if op == 'Affine_for_':
            return 'Affine_for'
        return op

    def _reconstruct_literal(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct literal constants directly from metadata and dtype."""
        target_block = block

        lit_value = expr.metadata.get('value', None)
        if lit_value is None:
            raise ValueError(
                f"Literal node {expr.node_id} missing 'value' metadata")

        mlir_type = megg_to_mlir_type(expr.dtype)

        type_str = str(mlir_type)
        is_float = any(ft in type_str for ft in ['f16', 'f32', 'f64'])

        with self.builder.set_insertion_point_to_end(target_block):
            if is_float:
                cache_key = ('float', float(lit_value), type_str, id(target_block))
                if cache_key in self.constants_cache:
                    return self.constants_cache[cache_key]
                const_value = self.builder.constant(
                    float(lit_value), mlir_type._type)
            else:
                cache_key = ('int', int(lit_value), type_str, id(target_block))
                if cache_key in self.constants_cache:
                    return self.constants_cache[cache_key]
                const_value = self.builder.constant(int(lit_value), mlir_type)

        self.constants_cache[cache_key] = const_value
        return const_value

    def _reconstruct_argument(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct function arguments."""

        arg_idx = expr.metadata.get('arg_info', None)
        if arg_idx is None:
            raise ValueError(
                f"Arg node {expr.node_id} missing 'arg_info' metadata")
        arg_key = f"arg_{arg_idx}"
        return self.expr_to_ssa.get(arg_key)

    def _reconstruct_loop_index(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct loop index variable."""
        loop_id_info = expr.metadata.get('arg_info', None)
        loop_block = None
        loop_key = None

        if loop_id_info is not None:
            loop_id = loop_id_info[0] if isinstance(
                loop_id_info, list) and len(loop_id_info) > 0 else loop_id_info
            loop_key = f"loop_{loop_id}"
            loop_block = self.loop_contexts.get(loop_key)

        if loop_block is None and self.loop_block_stack:
            loop_id_str = str(loop_id_info) if loop_id_info is not None else None
            selected_block = None

            for block_ctx, block_ids in self.loop_block_stack:
                if loop_id_str and loop_id_str in block_ids:
                    selected_block = block_ctx
                    break

            # A nested loop's binder cannot substitute for the recorded loop binder.
            if selected_block is None:
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not find loop_id {loop_id_str} in loop_block_stack. "
                             f"Available ids: {[ids for _, ids in self.loop_block_stack]}")
                return None  # Return None instead of using wrong loop!

            loop_block = selected_block
            if loop_key and loop_block:
                self.loop_contexts[loop_key] = loop_block

        if loop_block and len(loop_block.arguments) > 0:
            result = loop_block.arguments[0]
            return result

        return None

    def _reconstruct_block_arg(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct block argument (loop-carried values)."""
        arg_info = expr.metadata.get('arg_info', None)
        if not arg_info or len(arg_info) < 2:
            return None

        block_id = arg_info[0]
        arg_idx = arg_info[1]

        loop_key = f"loopblock_{block_id}"

        if loop_key in self.loop_contexts:
            loop_block = self.loop_contexts[loop_key]
            actual_idx = int(arg_idx)

            if actual_idx < len(loop_block.arguments):
                return loop_block.arguments[actual_idx]

        return None

    def _reconstruct_loop_iter_arg(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct loop iteration argument (loop-carried values)."""
        arg_info = expr.metadata.get('arg_info', None)
        if not arg_info or len(arg_info) < 2:
            return None

        loop_id = arg_info[0]
        iter_arg_idx = arg_info[1]

        loop_key = f"loop_{loop_id}"
        if loop_key in self.loop_contexts:
            loop_block = self.loop_contexts[loop_key]
            actual_idx = int(iter_arg_idx) + 1
            if actual_idx < len(loop_block.arguments):

                return loop_block.arguments[actual_idx]

        return None

    def _type_str_to_mtype(self, type_str: str):
        """Convert a type string (e.g., 'i32', 'index') to an MType object."""
        type_str = type_str.strip()
        if type_str == 'index':
            return self.builder.index()
        elif type_str == 'i32':
            return self.builder.i32()
        elif type_str == 'i64':
            return self.builder.i64()
        elif type_str == 'i1':
            return self.builder.i1()
        elif type_str == 'f32':
            return self.builder.f32()
        elif type_str == 'f64':
            return self.builder.f64()
        elif type_str.startswith('i') and type_str[1:].isdigit():
            width = int(type_str[1:])
            return self.builder.integer(width)
        else:
            return self.builder.i32()

    def _ensure_same_type(self, lhs: MValue, rhs: MValue, block: MBlock) -> Tuple[MValue, MValue]:
        """Ensure two operands have the same type, inserting index_cast if needed."""
        lhs_type_str = str(lhs.type)
        rhs_type_str = str(rhs.type)

        if lhs_type_str == rhs_type_str:
            return lhs, rhs

        with self.builder.set_insertion_point_to_end(block):
            if lhs_type_str == 'index' and 'i' in rhs_type_str:
                target_type = self._type_str_to_mtype(rhs_type_str)
                lhs = self.builder.index_cast(lhs, target_type)
            elif rhs_type_str == 'index' and 'i' in lhs_type_str:
                target_type = self._type_str_to_mtype(lhs_type_str)
                rhs = self.builder.index_cast(rhs, target_type)
            elif 'i' in lhs_type_str and 'i' in rhs_type_str:
                pass

        return lhs, rhs

    def _reconstruct_binary_op(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct binary arithmetic operations."""
        target_block = self.current_block_context if self.current_block_context else block

        if len(expr.children) < 2:
            return None

        lhs = self._reconstruct_expr(expr.children[0], block)
        rhs = self._reconstruct_expr(expr.children[1], block)

        if lhs is None or rhs is None:
            return None

        lhs, rhs = self._ensure_same_type(lhs, rhs, target_block)

        with self.builder.set_insertion_point_to_end(target_block):
            if expr.op == 'Add':
                result = self.builder.add(lhs, rhs)
            elif expr.op == 'Sub':
                result = self.builder.sub(lhs, rhs)
            elif expr.op == 'Mul':
                result = self.builder.mul(lhs, rhs)
            elif expr.op == 'Div':
                result = self.builder.div(lhs, rhs)
            elif expr.op == 'Rem':
                result = self.builder.rem(lhs, rhs)
            elif expr.op in ('And', 'And_'):
                result = self.builder.and_(lhs, rhs)
            elif expr.op in ('Or', 'Or_'):
                result = self.builder.or_(lhs, rhs)
            elif expr.op in ('Xor', 'Xor_'):
                result = self.builder.xor_(lhs, rhs)
            elif expr.op == 'Shl':
                result = self.builder.shl(lhs, rhs)
            elif expr.op in ('ShrSI', 'Shrsi'):
                result = self.builder.shrsi(lhs, rhs)
            elif expr.op in ('ShrUI', 'Shrui'):
                result = self.builder.shrui(lhs, rhs)
            else:
                return None

        return result

    def _reconstruct_unary_op(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct unary arithmetic operations."""
        if len(expr.children) < 1:
            return None

        operand = self._reconstruct_expr(expr.children[0], block)
        if operand is None:
            return None

        mlir_type = megg_to_mlir_type(expr.dtype)

        with self.builder.set_insertion_point_to_end(block):
            if expr.op == 'Neg':
                type_str = str(mlir_type)
                is_float = any(ft in type_str for ft in ['f16', 'f32', 'f64'])

                x = 0.0 if is_float else 0
                zero = self.builder.constant(x, mlir_type)
                result = self.builder.sub(zero, operand)
                return result
            else:
                raise RuntimeError(f"meet unsupport unary op: {expr.op}")

        return None

    def _reconstruct_comparison(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct comparison operations."""
        if len(expr.children) < 2:
            return None

        lhs = self._reconstruct_expr(expr.children[0], block)
        rhs = self._reconstruct_expr(expr.children[1], block)

        if lhs is None or rhs is None:
            return None

        predicate = expr.metadata.get('cmp_info', None)

        with self.builder.set_insertion_point_to_end(block):
            result = self.builder.cmp(predicate, lhs, rhs)

        return result

    def _reconstruct_select(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct arith.select operation (ternary conditional)."""
        if len(expr.children) < 3:
            return None

        cond = self._reconstruct_expr(expr.children[0], block)
        true_val = self._reconstruct_expr(expr.children[1], block)
        false_val = self._reconstruct_expr(expr.children[2], block)

        if cond is None or true_val is None or false_val is None:
            return None

        with self.builder.set_insertion_point_to_end(block):
            result = self.builder.select(cond, true_val, false_val)

        return result

    def _reconstruct_cast(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct cast operations."""
        if len(expr.children) < 1:
            return None

        operand = self._reconstruct_expr(expr.children[0], block)
        if operand is None:
            return None

        target_type = megg_to_mlir_type(expr.dtype)

        with self.builder.set_insertion_point_to_end(block):
            if expr.op == 'Index_cast':
                result = self.builder.index_cast(operand, target_type)
            elif expr.op in ('SIToFP', 'Sitofp'):
                result = self.builder.sitofp(operand, target_type)
            elif expr.op in ('UIToFP', 'Uitofp'):
                result = self.builder.uitofp(operand, target_type)
            elif expr.op in ('FPToSI', 'Fptosi'):
                result = self.builder.fptosi(operand, target_type)
            elif expr.op in ('FPToUI', 'Fptoui'):
                result = self.builder.fptoui(operand, target_type)
            elif expr.op in ('ExtSI', 'Extsi'):
                result = self.builder.extsi(operand, target_type)
            elif expr.op in ('ExtUI', 'Extui'):
                result = self.builder.extui(operand, target_type)
            elif expr.op in ('TruncI', 'Trunci'):
                result = self.builder.trunci(operand, target_type)
            elif expr.op == 'Bitcast':
                result = self.builder.bitcast(operand, target_type)
            else:
                return None
        return result

    def _pre_materialize_branch_dependencies(self, branch_expr: ExpressionNode, block: MBlock):
        """Pre-materialize all pure operations that a branch will use."""
        if not branch_expr:
            return

        target_block = self.current_block_context if self.current_block_context else block
        cache_key = (branch_expr.node_id, id(target_block))
        if cache_key in self.expr_to_ssa:
            return

        control_flow_ops = {'If_', 'While_', 'For', 'For_with_carry', 'Affine_for', 'Affine_for_with_carry',
                           'Block', 'Vec', 'Yield_', 'Affine_yield', 'Return_', 'Condition',
                           'Store', 'Load', 'Alloc', 'Alloca', 'Get_global'}  # Memory ops with side effects

        if branch_expr.op in control_flow_ops:
            # For control flow and side-effect operations, recursively pre-materialize their dependencies
            for child in branch_expr.children:
                self._pre_materialize_branch_dependencies(child, block)
        else:
            self._reconstruct_expr(branch_expr, block)

    def _branch_yields_values(self, branch_expr: ExpressionNode) -> bool:
        """Check if a branch yields non-empty values."""
        if branch_expr.op == 'Block' and branch_expr.children:
            vec_expr = branch_expr.children[0]
            if vec_expr.op == 'Vec' and vec_expr.children:
                for child in vec_expr.children:
                    if child.op in ['Yield_', 'Affine_yield']:
                        if child.children and len(child.children) > 0:
                            yield_vec = child.children[0]
                            if yield_vec.op == 'Vec':
                                return len(yield_vec.children) > 0
                        return False
        if branch_expr.op in ['Yield_', 'Affine_yield']:
            if branch_expr.children and len(branch_expr.children) > 0:
                yield_vec = branch_expr.children[0]
                if yield_vec.op == 'Vec':
                    return len(yield_vec.children) > 0
        return False

    def _reconstruct_if(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct if-then-else operations."""
        if len(expr.children) < 3:
            return None

        cond = self._reconstruct_expr(expr.children[0], block)
        if cond is None:
            return None

        then_expr = expr.children[1]
        else_expr = expr.children[2]

        old_prematerialize_context = self.current_block_context
        self.current_block_context = block
        self._pre_materialize_branch_dependencies(then_expr, block)
        self._pre_materialize_branch_dependencies(else_expr, block)
        self.current_block_context = old_prematerialize_context

        has_then_yield_values = self._branch_yields_values(then_expr)
        has_else_yield_values = self._branch_yields_values(else_expr)

        if not has_then_yield_values and not has_else_yield_values:
            return_types = []
        else:
            result_t = expr.dtype
            if not result_t or isinstance(result_t, VoidType):
                return_types = []
            else:
                result_type = megg_to_mlir_type(result_t)
                if result_type is None:
                    return_types = []
                else:
                    return_types = [result_type]

        has_non_empty_else = not (
            else_expr.op == 'Yield_' and
            (not else_expr.children or
             (len(else_expr.children) == 1 and
              else_expr.children[0].op == 'Vec' and
              len(else_expr.children[0].children) == 0))
        )

        with self.builder.set_insertion_point_to_end(block):
            if_op_m = self.builder.scf_if(cond, return_types, has_else=has_non_empty_else)

        old_context = self.current_block_context

        then_expr = expr.children[1]
        then_block = self.builder.get_if_then_block(if_op_m)
        self.current_block_context = then_block
        self._reconstruct_branch(then_expr, then_block)

        if has_non_empty_else:
            else_block = self.builder.get_if_else_block(if_op_m)
            self.current_block_context = else_block
            self._reconstruct_branch(else_expr, else_block)

        self.current_block_context = old_context

        if_result_block = old_context if old_context else block
        if_result_cache_key = (expr.node_id, id(if_result_block))

        if len(return_types) == 0:
            result = None
        else:
            result = self.builder.get_operation_result(if_op_m, 0)

        if if_result_cache_key not in self.expr_to_ssa:
            if result is not None:
                self.expr_to_ssa[if_result_cache_key] = result
            else:
                self.expr_to_ssa[if_result_cache_key] = "PROCESSED"

        return result

    def _reconstruct_condition(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct scf.condition operation."""

        if len(expr.children) < 2:
            return None

        cond_val = self._reconstruct_expr(expr.children[0], block)
        if cond_val is None:
            return None

        cond_type_str = str(cond_val.get_type())

        if cond_type_str != 'i1':
            cond_type = cond_val.get_type()
            with self.builder.set_insertion_point_to_end(block):
                zero = self.builder.constant(0, cond_type)
                cond_val = self.builder.cmp("ne", cond_val, zero)

        if len(expr.children) >= 2:
            vec_expr = expr.children[1]
            yield_values = self._collect_values_from_expr(vec_expr, block)
        else:
            yield_values = []
        with self.builder.set_insertion_point_to_end(block):
            self.builder.scf_condition(cond_val, yield_values)

        return None

    def _reconstruct_while(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct scf.while using first child as init_args, second as condition, third as body."""
        if len(expr.children) < 3:
            return None

        init_vals_expr = expr.children[0]
        init_vals = []
        if init_vals_expr.op == "Vec":
            for child in init_vals_expr.children:
                val = self._reconstruct_expr(child, block)
                if val is not None:
                    init_vals.append(val)
        else:
            val = self._reconstruct_expr(init_vals_expr, block)
            if val is not None:
                init_vals.append(val)
        if not init_vals:
            return None

        result_types = []
        for i, v in enumerate(init_vals):
            if isinstance(v, MValue):
                result_types.append(v.get_type())
            else:
                type_name = type(v).__name__
                if type_name == 'MOperation' or hasattr(v, 'results'):
                    try:
                        op_results = v.results
                        if op_results and len(op_results) > 0:
                            init_vals[i] = op_results[0]
                            result_types.append(op_results[0].get_type())
                        else:
                            raise RuntimeError(
                                f"MOperation in init_vals has no results: {v}")
                    except AttributeError as e:
                        if hasattr(v, '_op') and type(v._op).__name__ == 'MLIRValue':
                            init_vals[i] = MValue(v._op)
                            result_types.append(MValue(v._op).get_type())
                        else:
                            raise RuntimeError(
                                f"Expected MValue or MOperation in init_vals, got {type(v)} with error: {e}")
                else:
                    raise RuntimeError(
                        f"Expected MValue or MOperation in init_vals, got {type(v)}")

        with self.builder.set_insertion_point_to_end(block):
            while_op_m = self.builder.scf_while(result_types, init_vals)

        before_block = self.builder.get_while_before_block(while_op_m)
        after_block = self.builder.get_while_after_block(while_op_m)

        loop_ids = expr.metadata.get('while_info', [])
        if len(loop_ids) < 2:
            raise ValueError(
                f"While node {expr.node_id} missing 'while_info' metadata")
        before_block_id = loop_ids[0]
        after_block_id = loop_ids[1]
        self.loop_contexts[f"loopblock_{before_block_id}"] = before_block
        self.loop_contexts[f"loopblock_{after_block_id}"] = after_block

        old_context = self.current_block_context

        cond_expr = expr.children[1]
        self.current_block_context = before_block
        self._reconstruct_branch(cond_expr, before_block)

        body_expr = expr.children[2]
        self.current_block_context = after_block
        self._reconstruct_branch(body_expr, after_block)

        self.current_block_context = old_context

        result = self.builder.get_operation_result(while_op_m, 0)
        return result

    def _reconstruct_for(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct for loop without iter_args."""
        if len(expr.children) < 4:
            return None

        lower = self._reconstruct_expr(expr.children[0], block)
        upper = self._reconstruct_expr(expr.children[1], block)
        step = self._reconstruct_expr(expr.children[2], block)

        if not (lower and upper and step):
            return None

        loop_id_expr = expr.children[3] if len(expr.children) >= 5 else None
        body_expr_index = 4 if loop_id_expr else 3
        body_expr = expr.children[body_expr_index]

        loop_id = None
        if loop_id_expr is not None and hasattr(loop_id_expr, 'metadata'):
            loop_id = loop_id_expr.metadata.get('arg_info', None)

        loop_key = f"loop_{loop_id}" if loop_id is not None else None

        with self.builder.set_insertion_point_to_end(block):
            for_op_m = self.builder.scf_for(lower, upper, step, [])

        body_block = self.builder.get_for_body_block(for_op_m)

        local_loop_ids: Set[str] = set()
        if loop_id is not None:
            local_loop_ids.add(str(loop_id))
            self.loop_contexts[f"loop_{loop_id}"] = body_block

        if loop_key:
            self.loop_contexts[loop_key] = body_block

        old_context = self.current_block_context
        self.current_block_context = body_block

        self.loop_block_stack.append((body_block, local_loop_ids))
        try:
            self._reconstruct_branch(
                body_expr,
                body_block,
                allow_inferred_yield_values=False,
            )
        finally:
            self.loop_block_stack.pop()
            self.current_block_context = old_context

        return None

    def _reconstruct_for_with_carry(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:

        if len(expr.children) < 6:
            return None

        lower = self._reconstruct_expr(expr.children[0], block)
        upper = self._reconstruct_expr(expr.children[1], block)
        step = self._reconstruct_expr(expr.children[2], block)

        if not (lower and upper and step):
            return None

        loop_id = expr.children[3].metadata.get('arg_info', None)
        loop_key = f"loop_{loop_id}"

        iter_args = []
        init_vals_expr = expr.children[4]
        for child in init_vals_expr.children:
            val = self._reconstruct_expr(child, block)
            if val:
                iter_args.append(val)

        with self.builder.set_insertion_point_to_end(block):
            for_op_m = self.builder.scf_for(lower, upper, step, iter_args)

        body_block = self.builder.get_for_body_block(for_op_m)
        body_expr = expr.children[5]

        local_loop_ids: Set[str] = set()
        if loop_id is not None:
            local_loop_ids.add(str(loop_id))
            self.loop_contexts[f"loop_{loop_id}"] = body_block

        self.loop_contexts[loop_key] = body_block

        old_context = self.current_block_context
        self.current_block_context = body_block
        self.loop_block_stack.append((body_block, local_loop_ids))

        try:
            self._reconstruct_branch(
                body_expr,
                body_block,
                allow_inferred_yield_values=False,
            )

            ops = body_block.get_operations()
            if ops:
                last_op = ops[-1]
                if self._is_terminator(last_op):
                    yield_operands = last_op.operands if hasattr(last_op, 'operands') else []
                    expected_count = len(iter_args)

                    if len(yield_operands) != expected_count:
                        raise ValueError(
                            f"Loop expects {expected_count} yielded values, got {len(yield_operands)}")
        finally:
            self.loop_block_stack.pop()
            self.current_block_context = old_context

        return self.builder.get_operation_result(for_op_m, 0)
    
    def _reconstruct_affine_for(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct affine.for without iter_args."""
        if len(expr.children) < 7:
            return None

        step_value = self._extract_int_value(expr.children[2])

        loop_metadata = expr.children[3].metadata if expr.children[3].metadata else {}
        loop_id = loop_metadata.get('arg_info', None)
        loop_key = f"loop_{loop_id}" if loop_id is not None else None

        lower_bound = 0
        upper_bound = 10  # TODO: extract from affine maps

        with self.builder.set_insertion_point_to_end(block):
            for_op = self.builder.affine_for(lower_bound, upper_bound, step_value)

        if loop_key:
            self.loop_contexts[loop_key] = for_op.body

        body_stmts_expr = expr.children[6]
        if body_stmts_expr.op == 'Vec':
            for stmt in body_stmts_expr.children:
                self._reconstruct_expr(stmt, for_op.body)

        results = list(for_op.results_)
        return results[0] if results else None

    def _reconstruct_affine_for_with_carry(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct affine.for with iter_args."""
        if len(expr.children) < 8:
            return None

        step_value = self._extract_int_value(expr.children[2])

        loop_metadata = expr.children[3].metadata if expr.children[3].metadata else {}
        loop_id = loop_metadata.get('arg_info', None)
        loop_key = f"loop_{loop_id}" if loop_id is not None else None

        init_vals = []
        init_vec_expr = expr.children[6]
        if init_vec_expr.op == 'Vec':
            for child in init_vec_expr.children:
                val = self._reconstruct_expr(child, block)
                if val:
                    init_vals.append(val)

        lower_bound = 0
        upper_bound = 10  # TODO: extract from affine maps

        with self.builder.set_insertion_point_to_end(block):
            for_op = self.builder.affine_for(lower_bound, upper_bound, step_value)

        if loop_key:
            self.loop_contexts[loop_key] = for_op.body

        body_stmts_expr = expr.children[7]
        if body_stmts_expr.op == 'Vec':
            for stmt in body_stmts_expr.children:
                self._reconstruct_expr(stmt, for_op.body)

        results = list(for_op.results_)
        return results[0] if results else None
    
    def _reconstruct_vec(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct a Vec node: this represents an ordered sequence of sub-terms."""
        last_result = None
        for child in expr.children:
            res = self._reconstruct_expr(child, block)
            if res is not None:
                last_result = res
        return last_result

    def _collect_values_from_expr(self, expr: ExpressionNode, block: MBlock) -> List[MValue]:
        """Collect values for yield/return contexts."""
        values: List[MValue] = []
        op_name = getattr(expr, "op", None)
        if op_name == 'Vec':
            for child in expr.children:
                val = self._reconstruct_expr(child, block)
                if val is not None:
                    if isinstance(val, MValue):
                        values.append(val)
                    elif hasattr(val, 'results'):
                        op_results = val.results  # results is a @property, not a method
                        if op_results and len(op_results) > 0:
                            values.append(op_results[0])
        else:
            val = self._reconstruct_expr(expr, block)
            if val is not None:
                if isinstance(val, MValue):
                    values.append(val)
                elif hasattr(val, 'results'):
                    op_results = val.results  # results is a @property, not a method
                    if op_results and len(op_results) > 0:
                        values.append(op_results[0])
        return values

    def _ensure_block_terminated(
        self,
        block: MBlock,
        expected_results: List[MType],
        body_results: List[MValue],
    ) -> None:
        """Ensure a block ends with a terminator."""
        ops = block.get_operations()
        if ops and self._is_terminator(ops[-1]):
            return

        with self.builder.set_insertion_point_to_end(block):
            if expected_results:
                if not body_results or len(body_results) != len(expected_results):
                    raise ValueError(
                        "Function expects return values but block has no terminator and no body results"
                    )
                casted_vals = [
                    self._cast_value_to_type(val, expected_results[idx])
                    for idx, val in enumerate(body_results)
                ]
                self.builder.func_return(casted_vals)
            else:
                self.builder.func_return([])

    def _cast_value_to_type(self, value: MValue, target_type: MType) -> MValue:
        """Ensure a value matches the expected MLIR type."""
        if value is None or target_type is None:
            return value

        src_type_str = str(value.get_type())
        tgt_type_str = str(target_type)

        if src_type_str == tgt_type_str:
            return value

        def parse_int_width(type_str: str) -> Optional[int]:
            if type_str.startswith('i') and type_str[1:].isdigit():
                return int(type_str[1:])
            return None

        src_width = parse_int_width(src_type_str)
        tgt_width = parse_int_width(tgt_type_str)

        if src_width is not None and tgt_width is not None:
            if src_width > tgt_width:
                return self.builder.trunci(value, target_type)
            elif src_width < tgt_width:
                return self.builder.extsi(value, target_type)
            return value

        if tgt_type_str == 'index':
            return self.builder.index_cast(value, target_type)

        return value

    def _collect_loop_index_ids(self, expr: ExpressionNode, stop_ops: Optional[set[str]] = None, debug: bool = False) -> set[str]:
        """Collect loop index identifiers from an expression, stopping at nested loops."""
        if stop_ops is None:
            stop_ops = {
                'For', 'For_', 'For_with_carry',
                'Affine_for', 'Affine_for_with_carry',
                'While_', 'While',
            }

        loop_ids: set[str] = set()
        stack = [expr]
        while stack:
            node = stack.pop()
            op_name = getattr(node, 'op', None)
            if op_name is None:
                continue
            canonical_op = self._canonical_op(op_name)
            if canonical_op in stop_ops and node is not expr:
                # Do not traverse into nested loops
                if debug:
                    pass
                continue
            if canonical_op == 'Loop_index':
                info = node.metadata.get('arg_info') if node.metadata else None
                if info is not None:
                    if debug:
                        pass
                    loop_ids.add(str(info))
            stack.extend(node.children or [])
        return loop_ids

    def _reconstruct_block(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct a Block: currently each Block has exactly one child (a Vec)."""
        if not expr.children:
            return None
        return self._reconstruct_expr(expr.children[0], block)

    def _reconstruct_branch(
        self,
        branch_expr: ExpressionNode,
        target_block: MBlock,
        *,
        terminator_dialect: str = 'scf',
        allow_inferred_yield_values: bool = True,
    ):
        """Reconstruct a branch (then/else/body)."""
        initial_ops = target_block.get_operations()
        if initial_ops and self._is_terminator(initial_ops[-1]):
            initial_ops[-1].erase()

        value = self._reconstruct_expr(branch_expr, target_block)
        operations = target_block.get_operations()
        if operations and self._is_terminator(operations[-1]):
            return
        values = [value] if allow_inferred_yield_values and value is not None else []
        with self.builder.set_insertion_point_to_end(target_block):
            if terminator_dialect == 'affine':
                self.builder.affine_yield(values)
            else:
                self.builder.scf_yield(values)

    def _reconstruct_yield(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """A Yield always has a single Vec child, whose elements are the yielded values."""
        if not expr.children:
            yield_values = []
        else:
            vec_expr = expr.children[0]
            yield_values = self._collect_values_from_expr(vec_expr, block)

        ops = block.get_operations()
        if ops and self._is_terminator(ops[-1]):
            if len(yield_values) > 0:
                pass
            else:
                return None

        with self.builder.set_insertion_point_to_end(block):
            result = self.builder.scf_yield(yield_values)
        return None

    def _reconstruct_affine_yield(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        if not expr.children:
            with self.builder.set_insertion_point_to_end(block):
                self.builder.affine_yield([])
            return None

        vec_expr = expr.children[0]
        values = self._collect_values_from_expr(vec_expr, block)

        with self.builder.set_insertion_point_to_end(block):
            self.builder.affine_yield(values)
        return None

    def _reconstruct_custom_instr(self, expr: ExpressionNode, block:MBlock) -> Optional[MValue]:
        """Reconstruct custom instruction as a call (or inline asm) with typed operands."""
        instr_name = expr.metadata.get('instr_name') if expr.metadata else None
        if not instr_name:
            raise ValueError(f"Custom instruction node {expr.node_id} missing 'instr_name' metadata")

        operand_values: List[MValue] = []
        if expr.children:
            if len(expr.children) == 1 and expr.children[0].op == 'Vec':
                operands_expr = expr.children[0]
                operand_values = self._collect_values_from_expr(operands_expr, block)
            else:
                for child_expr in expr.children:
                    val = self._reconstruct_expr(child_expr, block)
                    if val is not None:
                        operand_values.append(val)

        result_type = (megg_to_mlir_type(expr.dtype)
                       if expr.dtype is not None and not isinstance(expr.dtype, VoidType)
                       else None)
        properties = self.instr_properties.get(instr_name, {})
        with self.builder.set_insertion_point_to_end(block):
            return MLIRToLLVMBackend.emit_custom_instruction(
                self.builder, operand_values, result_type, instr_name, properties)

    def _reconstruct_return(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """A Return always has a single Vec child, whose elements are the return values."""
        if not expr.children:
            with self.builder.set_insertion_point_to_end(block):
                self.builder.func_return([])
            return None

        vec_expr = expr.children[0]
        values = self._collect_values_from_expr(vec_expr, block)

        func_type = self.original_func.get_function_type
        expected_types = func_type.get_results()

        adjusted_values = []
        for i, val in enumerate(values):
            if i < len(expected_types):
                expected_type = expected_types[i]
                if hasattr(val, '_value') and hasattr(val._value, 'get_defining_op'):
                    defining_op = val._value.get_defining_op()
                    if defining_op and hasattr(defining_op, 'get_num_results'):
                        num_results = defining_op.get_num_results()
                        if num_results > 1:
                            for res_idx in range(num_results):
                                result_val = self.builder.get_operation_result(MOperation(defining_op), res_idx)
                                if str(result_val.get_type()) == str(expected_type):
                                    adjusted_values.append(result_val)
                                    break
                            else:
                                adjusted_values.append(val)
                        else:
                            adjusted_values.append(val)
                    else:
                        adjusted_values.append(val)
                else:
                    adjusted_values.append(val)
            else:
                adjusted_values.append(val)

        converted_values: List[MValue] = []
        for i, val in enumerate(adjusted_values):
            if i < len(expected_types):
                converted_values.append(
                    self._cast_value_to_type(val, expected_types[i])
                )
            else:
                converted_values.append(val)

        with self.builder.set_insertion_point_to_end(block):
            self.builder.func_return(converted_values)
        return None

    def _reconstruct_memref_alloc(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.alloc."""
        memref_type = megg_to_mlir_type(expr.dtype)  # Returns MType

        with self.builder.set_insertion_point_to_end(block):
            alloc_op = self.builder.alloc(memref_type)

        return alloc_op

    def _reconstruct_memref_alloca(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.alloca (stack allocation)."""
        memref_type = megg_to_mlir_type(expr.dtype)  # Returns MType

        with self.builder.set_insertion_point_to_end(block):
            alloca_op = self.builder.alloca(memref_type)

        return alloca_op

    def _reconstruct_memref_get_global(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.get_global."""
        symbol_name = expr.metadata.get('global_name', None)
        if symbol_name and symbol_name.startswith('@'):
            symbol_name = symbol_name[1:]
        if expr.children:
            name_child = expr.children[0]
            if name_child.op.startswith('Boxed('):
                symbol_name = name_child.op[7:-2]  # Remove 'Boxed("' and '")'

        memref_type = megg_to_mlir_type(expr.dtype)

        with self.builder.set_insertion_point_to_end(block):
            get_global_result = self.builder.get_global(
                memref_type, symbol_name)

        return get_global_result

    def _reconstruct_memref_store(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.store."""
        if len(expr.children) < 3:
            return None

        value = self._reconstruct_expr(expr.children[0], block)
        memref_val = self._reconstruct_expr(expr.children[1], block)

        if not (value and memref_val):
            return None

        index_expr = expr.children[2]
        index_value = self._reconstruct_expr(index_expr, block)
        if not index_value:
            return None
        indices = [index_value]

        with self.builder.set_insertion_point_to_end(block):
            memref_type_str = str(memref_val.type)
            value_type_str = str(value.type)

            if 'memref<' in memref_type_str:
                element_type_start = memref_type_str.rfind('x') + 1
                element_type_end = memref_type_str.find('>', element_type_start)
                if element_type_end == -1:
                    element_type_end = len(memref_type_str)
                element_type = memref_type_str[element_type_start:element_type_end]

                if value_type_str != element_type:
                    if value_type_str == 'index' and element_type.startswith('i'):
                        if element_type.startswith('i'):
                            bit_width = int(element_type[1:])
                            target_type = IntType(width=bit_width, signed=True)
                            mlir_target_type = megg_to_mlir_type(target_type)
                            value = self.builder.index_cast(value, mlir_target_type)

            self.builder.store(value, memref_val, indices)

        return None

    def _reconstruct_memref_load(self, expr: ExpressionNode, block: MBlock) -> Optional[MValue]:
        """Reconstruct memref.load."""
        if len(expr.children) < 2:
            return None

        memref_val = self._reconstruct_expr(expr.children[0], block)
        if not memref_val:
            return None

        index_expr = expr.children[1]
        index_value = self._reconstruct_expr(index_expr, block)
        if not index_value:
            return None
        indices = [index_value]

        with self.builder.set_insertion_point_to_end(block):
            load_result = self.builder.load(memref_val, indices)

        return load_result

    def _extract_int_value(self, expr: ExpressionNode) -> int:
        if expr.op != 'Lit' or 'value' not in expr.metadata:
            raise ValueError(f"Expected integer literal, got {expr.op}")
        return int(expr.metadata['value'])

    def _is_terminator(self, operation: MOperation) -> bool:
        return operation.is_terminator
