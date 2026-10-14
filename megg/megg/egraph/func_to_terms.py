"""Direct transformation between MLIR FuncOp and Megg Terms."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import egglog
from egglog import EGraph

from megg.egraph.datatype import (
    DataType,
    FloatType as MeggFloatType,
    IndexType,
    IntType,
    MemRefType,
    PointerType,
    TensorType,
    TupleType,
    VectorType,
    VoidType,
)
from megg.egraph.term import LitTerm, Term, id_of
from megg.utils import IRBuilder, MBlock, MOperation, MType, MValue, OperationType

_session_counter = 0

def _next_session_id() -> str:
    """Generate a unique session ID for each FuncToTerms transformation."""
    global _session_counter
    _session_counter += 1
    return f"s{_session_counter}"

CONTROL_FLOW_OPS = {
    OperationType.SCF_FOR,
    OperationType.SCF_IF,
    OperationType.SCF_WHILE,
    OperationType.AFFINE_FOR,
    OperationType.AFFINE_IF,
}

@dataclass
class FuncToTerms:
    """Transform a FuncOp directly to Terms in an e-graph."""

    func: MOperation
    egraph: EGraph
    next_id: int
    ssa_to_id: Dict[MValue, int]
    # Essential: Mapping from SSA values to Terms (def-use chain)
    ssa_to_term: Dict[MValue, Term]
    top_block: Term
    loop_to_term: Dict[int, Tuple[MOperation,Term]]
    loop_index: int = 0
    session_id: str = field(default_factory=_next_session_id)

    @classmethod
    def transform(cls, func: MOperation, egraph: Optional[EGraph] = None) -> 'FuncToTerms':
        """Transform a FuncOp to Terms."""
        if egraph is None:
            egraph = EGraph(save_egglog_string=True)

        transformer = cls(
            func=func,
            egraph=egraph,
            next_id=0,
            ssa_to_id={},
            ssa_to_term={},
            loop_to_term={},
            top_block=None,
        )
        transformer._transform()
        return transformer

    def get_id_of_term(self, term: Term) -> Optional[int]:
        """Get the numeric ID of a Term if it's tracked."""
        for ssa_val, t in self.ssa_to_term.items():
            if t == term:
                return self.ssa_to_id.get(ssa_val, None)
        return None

    def get_tracked_terms(self) -> Dict[int, Term]:
        """Get all tracked terms with their numeric IDs."""
        terms = {}
        for ssa_val, term_id in self.ssa_to_id.items():
            if ssa_val in self.ssa_to_term:
                terms[term_id] = self.ssa_to_term[ssa_val]
        return terms

    def _transform(self):
        """Perform the transformation from FuncOp to Terms (only top block)."""
        regions = self.func.get_regions()
        if len(regions) == 0:
            return egglog.ruleset()  # Return empty ruleset

        blocks = regions[0].get_blocks()
        if len(blocks) == 0:
            return egglog.ruleset()  # Return empty ruleset

        block = blocks[0]

        self.top_block = self._block_to_term(block, is_function_block=True)

    def _operation_to_term(self, op: MOperation, operand_terms: List[Term], loop_id: Optional[int] = None, num_iter_args: int = 0) -> Optional[Term]:
        """Convert an operation to a Term."""
        if len(op.results) == 0:
            ty_str = egglog.String("void")
        elif len(op.results) == 1:
            ty_str = mlir_type_to_egraph_ty_string(op.results[0].type)
        else:
            result_types = [r.type for r in op.results]
            ty_str = create_tuple_type_string(result_types)


        op_type = op.type if hasattr(op, 'type') else OperationType.UNKNOWN

        if op_type in [OperationType.ARITH_ADD, OperationType.ARITH_ADDF]:
            if len(operand_terms) == 2:
                return Term.add(operand_terms[0], operand_terms[1], ty_str)
        elif op_type in [OperationType.ARITH_SUB, OperationType.ARITH_SUBF]:
            if len(operand_terms) == 2:
                return Term.sub(operand_terms[0], operand_terms[1], ty_str)
        elif op_type in [OperationType.ARITH_MUL, OperationType.ARITH_MULF]:
            if len(operand_terms) == 2:
                return Term.mul(operand_terms[0], operand_terms[1], ty_str)
        elif op_type in [OperationType.ARITH_DIV, OperationType.ARITH_DIVF]:
            if len(operand_terms) == 2:
                return Term.div(operand_terms[0], operand_terms[1], ty_str)
        elif op_type in [OperationType.ARITH_REM, OperationType.ARITH_REMF]:
            if len(operand_terms) == 2:
                return Term.rem(operand_terms[0], operand_terms[1], ty_str)

        elif op_type == OperationType.ARITH_ANDI:
            if len(operand_terms) == 2:
                return Term.and_(operand_terms[0], operand_terms[1], ty_str)
        elif op_type == OperationType.ARITH_ORI:
            if len(operand_terms) == 2:
                return Term.or_(operand_terms[0], operand_terms[1], ty_str)
        elif op_type == OperationType.ARITH_XORI:
            if len(operand_terms) == 2:
                return Term.xor_(operand_terms[0], operand_terms[1], ty_str)

        elif op_type == OperationType.ARITH_SHLI:
            if len(operand_terms) == 2:
                return Term.shl(operand_terms[0], operand_terms[1], ty_str)
        elif op_type == OperationType.ARITH_SHRSI:
            if len(operand_terms) == 2:
                return Term.shrsi(operand_terms[0], operand_terms[1], ty_str)
        elif op_type == OperationType.ARITH_SHRUI:
            if len(operand_terms) == 2:
                return Term.shrui(operand_terms[0], operand_terms[1], ty_str)

        elif op_type == OperationType.ARITH_INDEX_CAST:
            if len(operand_terms) == 1:
                return Term.index_cast(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_SITOFP:
            if len(operand_terms) == 1:
                return Term.sitofp(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_UITOFP:
            if len(operand_terms) == 1:
                return Term.uitofp(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_FPTOSI:
            if len(operand_terms) == 1:
                return Term.fptosi(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_FPTOUI:
            if len(operand_terms) == 1:
                return Term.fptoui(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_EXTSI:
            if len(operand_terms) == 1:
                return Term.extsi(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_EXTUI:
            if len(operand_terms) == 1:
                return Term.extui(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_TRUNCI:
            if len(operand_terms) == 1:
                return Term.trunci(operand_terms[0], ty_str)
        elif op_type == OperationType.ARITH_BITCAST:
            if len(operand_terms) == 1:
                return Term.bitcast(operand_terms[0], ty_str)

        elif op_type == OperationType.ARITH_CMPI:
            if len(operand_terms) == 2:
                predicate = predicate_to_string(op.get_attr('predicate'))
                return Term.cmp(operand_terms[0], operand_terms[1], egglog.String(predicate), ty_str)
        elif op_type == OperationType.ARITH_CMPF:
            if len(operand_terms) == 2:
                predicate = predicate_to_string(
                    op.get_attr('predicate'), cmp_type='f')
                return Term.cmp(operand_terms[0], operand_terms[1], egglog.String(predicate), ty_str)

        elif op_type == OperationType.ARITH_SELECT:
            if len(operand_terms) == 3:
                return Term.select(operand_terms[0], operand_terms[1], operand_terms[2], ty_str)

        elif op_type == OperationType.SCF_IF:
            cond_term = operand_terms[0] if operand_terms else None

            then_term = None
            if len(op.regions) > 0 and len(op.regions[0].get_blocks()) > 0:
                then_term = self._block_to_term(
                    op.regions[0].get_blocks()[0], loop_id, num_iter_args)

            else_term = None
            if len(op.regions) > 1 and len(op.regions[1].get_blocks()) > 0:
                else_term = self._block_to_term(
                    op.regions[1].get_blocks()[0], loop_id, num_iter_args)
            else:
                else_term = Term.yield_(
                    egglog.Vec[Term](), egglog.String("void"))

            if cond_term is not None and then_term is not None and else_term is not None:
                return Term.if_(cond_term, then_term, else_term, ty_str)
        elif op_type == OperationType.SCF_WHILE:
            init_terms = operand_terms  # Initial values for the while loop

            cond_term = None
            if len(op.regions) > 0 and len(op.regions[0].get_blocks()) > 0:
                cond_term = self._block_to_term(op.regions[0].get_blocks()[0])

            body_term = None
            if len(op.regions) > 1 and len(op.regions[1].get_blocks()) > 0:
                body_term = self._block_to_term(op.regions[1].get_blocks()[0])

            if cond_term is not None and body_term is not None:
                return Term.while_(init_terms, cond_term, body_term, ty_str)
        elif op_type == OperationType.SCF_FOR:
            if len(operand_terms) >= 3:
                lower = operand_terms[0]
                upper = operand_terms[1]
                step = operand_terms[2]

                loop_name = self.loop_index
                self.loop_index += 1
                loop_id = id(op)
                loop_id_i64 = egglog.i64(loop_id)
                loop_index_term = Term.loop_index(
                    loop_id_i64, egglog.String("index"))

                # Check if there are iter_args (loop-carried values)
                iter_args = operand_terms[3:] if len(operand_terms) > 3 else []

                body_term = None
                if op.regions and len(op.regions) > 0 and len(op.regions[0].get_blocks()) > 0:
                    body_term = self._block_to_term(
                        op.regions[0].get_blocks()[0], loop_id=loop_id, num_iter_args=len(
                            iter_args)
                    )
                elif hasattr(op, 'body') and len(op.body.get_blocks()) > 0:
                    body_term = self._block_to_term(
                        op.body.get_blocks()[0], loop_id=loop_id, num_iter_args=len(
                            iter_args)
                    )

                if body_term is not None:
                    if iter_args:
                        # Use for_with_carry for loops with loop-carried values
                        loop_term = Term.for_with_carry(
                            lower, upper, step,
                            loop_index_term,  # This is a Term
                            egglog.Vec[Term](*iter_args),
                            body_term, ty_str
                        )
                    else:
                        loop_term = Term.for_(
                            lower, upper, step,
                            loop_index_term,  # This is a Term (Loop_index node)
                            body_term, ty_str
                        )
                    self.loop_to_term[loop_name] = (op,loop_term)

                    return loop_term
        elif op_type == OperationType.FUNC_RETURN:
            return Term.return_(egglog.Vec[Term](*operand_terms), ty_str)

        elif op_type == OperationType.SCF_YIELD:
            # YieldOp can yield multiple values (for loop-carried values)
            return Term.yield_(egglog.Vec[Term](*operand_terms), ty_str)

        elif op_type == OperationType.SCF_CONDITION:
            if len(operand_terms) >= 1:
                return Term.yield_(egglog.Vec(*operand_terms[1:]), ty_str)
            return Term.yield_(egglog.Vec(), ty_str)

        elif op_type == OperationType.ARITH_CONSTANT:
            return self._constant_to_term(op)

        elif op_type == OperationType.MEMREF_ALLOC:
            return Term.alloc(ty_str)
        elif op_type == OperationType.MEMREF_ALLOCA:
            return Term.alloca(ty_str)
        elif op_type == OperationType.MEMREF_GET_GLOBAL:
            symbol_name = op.get_attr('name')
            return Term.get_global(egglog.String(symbol_name), ty_str)
        elif op_type == OperationType.MEMREF_STORE:
            if len(operand_terms) < 3:
                raise ValueError(
                    "memref.store currently requires exactly one index term")
            value_term = operand_terms[0]  # Value to store
            memref_term = operand_terms[1]  # MemRef to store into
            index_terms = operand_terms[2:]
            if len(index_terms) != 1:
                raise ValueError(
                    f"memref.store expects one index term, got {len(index_terms)}")
            return Term.store(value_term, memref_term, index_terms[0], ty_str)
        elif op_type == OperationType.MEMREF_LOAD:
            if len(operand_terms) < 2:
                raise ValueError(
                    "memref.load currently requires exactly one index term")
            memref_term = operand_terms[0]  # MemRef to load from
            index_terms = operand_terms[1:]
            if len(index_terms) != 1:
                raise ValueError(
                    f"memref.load expects one index term, got {len(index_terms)}")
            return Term.load(memref_term, index_terms[0], ty_str)

        raise ValueError(f"MOperation {op.name} is not supported")

    def _is_pure_operation(self, op: MOperation) -> bool:
        """Check if an operation is pure (no side effects and no control flow)."""
        op_type = op.type if hasattr(op, 'type') else OperationType.UNKNOWN

        # Control flow and side-effect operations must be serialized
        side_effect_types = {
            *CONTROL_FLOW_OPS,
            OperationType.MEMREF_STORE, OperationType.AFFINE_STORE,
            OperationType.MEMREF_ALLOC, OperationType.MEMREF_DEALLOC,
            OperationType.MEMREF_ALLOCA,
            OperationType.FUNC_RETURN, OperationType.SCF_YIELD,
        }

        if op_type in side_effect_types:
            return False

        return True

    def _get_block_stmts(self, block: MBlock, loop_id: Optional[int] = None, num_iter_args: int = 0) -> egglog.Vec[Term]:
        """Process a block and return only the statement vector (Vec[Term]) without wrapping in Term.block."""
        for i, arg in enumerate(block.arguments):
            if loop_id is not None and i < num_iter_args + 1:
                if i == 0:
                    term = Term.loop_index(egglog.i64(
                        loop_id), mlir_type_to_egraph_ty_string(arg.type))
                    term_ref = self.egraph.let(f"{self.session_id}_loop_{loop_id}_index", term)
                else:
                    iter_arg_index = i - 1
                    term = Term.loop_iter_arg(egglog.i64(loop_id), egglog.i64(
                        iter_arg_index), mlir_type_to_egraph_ty_string(arg.type))
                    term_ref = self.egraph.let(
                        f"{self.session_id}_loop_{loop_id}_iter_arg_{iter_arg_index}", term)
            else:
                term = Term.block_arg(id(block), egglog.i64(
                    i), mlir_type_to_egraph_ty_string(arg.type))
                term_ref = self.egraph.let(f"{self.session_id}_block_{id(block)}_arg_{i}", term)

            self.ssa_to_term[arg] = term_ref
            arg_id = self._next_id()
            self.ssa_to_id[arg] = arg_id
            self.egraph.register(egglog.set_(
                id_of(term_ref)).to(egglog.i64(arg_id)))

        # Collect serialized terms (side effects + terminators)
        serialized_terms = []

        for op in block.operations:
            op_ty = op.type
            if op_ty in [OperationType.SCF_YIELD, OperationType.AFFINE_YIELD, OperationType.AFFINE_STORE, OperationType.MEMREF_STORE, OperationType.MEMREF_ALLOC, OperationType.MEMREF_ALLOCA]:
                operand_terms = self._get_operand_terms(
                    op, block, loop_id, num_iter_args)
                term = self._operation_to_term(
                    op, operand_terms, loop_id, num_iter_args)
                if term is not None:
                    serialized_terms.append(term)
            else:
                operand_terms = self._get_operand_terms(
                    op, block, loop_id, num_iter_args)
                term = self._operation_to_term(
                    op, operand_terms, loop_id, num_iter_args)
                if term is not None:
                    for result in op.results:
                        self.ssa_to_term[result] = term

        return egglog.Vec[Term](*serialized_terms)

    def _block_to_term(
        self,
        block: MBlock,
        loop_id: Optional[int] = None,
        num_iter_args: int = 0,
        is_function_block: bool = False
    ) -> Optional[Term]:
        """Unified block processing for all contexts (function, loop, if/else, etc.)."""
        if not block or not block.operations:
            return None

        for i, arg in enumerate(block.arguments):
            if is_function_block:
                term = Term.arg(egglog.i64(
                    i), mlir_type_to_egraph_ty_string(arg.type))
                term_ref = self.egraph.let(f"{self.session_id}_func_{id(block)}_arg_{i}", term)
            elif loop_id is not None:
                if i == 0:
                    term = Term.loop_index(egglog.i64(
                        loop_id), mlir_type_to_egraph_ty_string(arg.type))
                    term_ref = self.egraph.let(f"{self.session_id}_loop_{loop_id}_index", term)
                else:
                    iter_arg_index = i - 1
                    term = Term.loop_iter_arg(egglog.i64(loop_id), egglog.i64(
                        iter_arg_index), mlir_type_to_egraph_ty_string(arg.type))
                    term_ref = self.egraph.let(
                        f"{self.session_id}_loop_{loop_id}_iter_arg_{iter_arg_index}", term)
            else:
                term = Term.block_arg(id(block), egglog.i64(
                    i), mlir_type_to_egraph_ty_string(arg.type))
                term_ref = self.egraph.let(f"{self.session_id}_block_{id(block)}_arg_{i}", term)

            self.ssa_to_term[arg] = term_ref
            arg_id = self._next_id()
            self.ssa_to_id[arg] = arg_id
            self.egraph.register(egglog.set_(
                id_of(term_ref)).to(egglog.i64(arg_id)))

        # Collect operations that must be serialized (side effects + terminators)
        serialized_terms = []
        if (terminator := block.get_terminator()):
            if len(terminator.operands) > 0:
                ret_type = terminator.operands[0].type
                block_type = mlir_type_to_egraph_ty_string(ret_type)
            else:
                block_type = egglog.String("void")
        else:
            block_type = egglog.String("void")  # Default type

        # Collect operations that MUST be serialized in block.vec
        # 1. Side-effect operations (memref.store, control flow, etc.)

        all_ops = list(block.operations)
        need_serialize_ops = set()

        for op in all_ops:
            op_ty = op.type
            if not self._is_pure_operation(op) or op_ty in [OperationType.FUNC_RETURN, OperationType.SCF_YIELD, OperationType.SCF_CONDITION]:
                need_serialize_ops.add(op_ty)

        for op in all_ops:
            op_ty = op.type
            if op_ty in [OperationType.FUNC_RETURN, OperationType.SCF_YIELD, OperationType.SCF_CONDITION]:
                if op_ty == OperationType.SCF_CONDITION:
                    if not op.operands:
                        raise RuntimeError("scf.ConditionOp has no operands")

                    cond_operand = op.operands[0]
                    if cond_operand in self.ssa_to_term:
                        cond_term = self.ssa_to_term[cond_operand]
                    else:
                        cond_term = Term.block_arg(id(block), egglog.i64(
                            0), mlir_type_to_egraph_ty_string(cond_operand.type))
                        cond_term = self.egraph.let(
                            f"{self.session_id}_block_{id(block)}_cond", cond_term)
                        self.ssa_to_term[cond_operand] = cond_term
                        cond_id = self._next_id()
                        self.ssa_to_id[cond_operand] = cond_id
                        self.egraph.register(egglog.set_(
                            id_of(cond_term)).to(egglog.i64(cond_id)))

                    iter_terms = []
                    for i, iter_operand in enumerate(op.operands[1:]):
                        if iter_operand in self.ssa_to_term:
                            iter_term = self.ssa_to_term[iter_operand]
                        else:
                            iter_term = Term.block_arg(id(block), egglog.i64(
                                i), mlir_type_to_egraph_ty_string(iter_operand.type))
                            iter_term = self.egraph.let(
                                f"{self.session_id}_block_{id(block)}_cond_arg_{i}", iter_term)
                            self.ssa_to_term[iter_operand] = iter_term
                            iter_id = self._next_id()
                            self.ssa_to_id[iter_operand] = iter_id
                            self.egraph.register(egglog.set_(
                                id_of(iter_term)).to(egglog.i64(iter_id)))
                        iter_terms.append(iter_term)

                    cond_block_term = Term.condition(
                        cond_term, egglog.Vec(*iter_terms), block_type)
                    serialized_terms.append(cond_block_term)
                else:
                    operand_terms = self._get_operand_terms(
                        op, block, loop_id, num_iter_args)
                    term = self._operation_to_term(
                        op, operand_terms, loop_id, num_iter_args)
                    if term is not None:
                        serialized_terms.append(term)

                continue

            operand_terms = self._get_operand_terms(
                op, block, loop_id, num_iter_args)

            term = self._operation_to_term(
                op, operand_terms, loop_id, num_iter_args)

            if term is None:
                continue

            if is_function_block:
                serialize_op = (op.type in need_serialize_ops)
                if len(op.results) == 0:
                    # No results (side effect operation) - must serialize
                    unique_id = self._next_id()
                    name_hint = f"{self.session_id}_side_effect_{id(op)}_{unique_id}"
                    term_ref = self.egraph.let(name_hint, term)
                    serialized_terms.append(term_ref)
                elif len(op.results) == 1:
                    result = op.results[0]
                    name_hint = f"{self.session_id}_op_{id(result)}"
                    term_ref = self.egraph.let(name_hint, term)
                    self.ssa_to_term[result] = term_ref
                    result_id = self._next_id()
                    self.ssa_to_id[result] = result_id
                    self.egraph.register(egglog.set_(
                        id_of(term_ref)).to(egglog.i64(result_id)))

                    if serialize_op:
                        serialized_terms.append(term_ref)
                    else:
                        pass
                else:
                    main_result = op.results[0]
                    name_hint = f"{self.session_id}_op_{id(main_result)}"
                    term_ref = self.egraph.let(name_hint, term)
                    for i, result in enumerate(op.results):
                        self.ssa_to_term[result] = term_ref
                        result_id = self._next_id()
                        self.ssa_to_id[result] = result_id

                    if serialize_op:
                        serialized_terms.append(term_ref)
            else:
                for result in op.results:
                    self.ssa_to_term[result] = term

                is_terminator = op_ty in [
                    OperationType.FUNC_RETURN, OperationType.SCF_YIELD, OperationType.SCF_CONDITION]
                should_serialize = not self._is_pure_operation(
                    op) or is_terminator
                if should_serialize:
                    serialized_terms.append(term)
                else:
                    pass

        # Create block term only with serialized (side-effect + terminator) operations
        if serialized_terms:
            block_id = id(block)
            block_term = Term.block(egglog.i64(
                block_id), egglog.Vec(*serialized_terms), block_type)

            if is_function_block:
                block_name = f"{self.session_id}_func_body_{id(block)}"
            else:
                block_name = f"{self.session_id}_block_{id(block)}"

            term_ref = self.egraph.let(block_name, block_term)

            self.ssa_to_term[block] = term_ref
            block_id = self._next_id()
            self.ssa_to_id[block] = block_id
            self.egraph.register(egglog.set_(
                id_of(term_ref)).to(egglog.i64(block_id)))

            if is_function_block:
                self.top_block = term_ref

            return term_ref
        else:
            return Term.block(
                id(block),
                egglog.Vec(Term.yield_(egglog.Vec(), egglog.String("void"))),
                block_type
            )

    def _get_operand_terms(
        self,
        op: MOperation,
        block: MBlock,
        loop_id: Optional[int] = None,
        num_iter_args: int = 0
    ) -> List[Term]:
        """Extract operand terms for an operation, resolving block arguments."""
        operand_terms = []
        for operand in op.operands:

            found = False
            if operand in self.ssa_to_term:
                operand_terms.append(self.ssa_to_term[operand])
                found = True
            else:
                operand_str = str(operand)
                for key, value in self.ssa_to_term.items():
                    if str(key) == operand_str:
                        operand_terms.append(value)
                        found = True
                        break

            if not found:
                block_arg_index = None
                for i, block_arg in enumerate(block.arguments):
                    if str(operand) == str(block_arg):
                        block_arg_index = i
                        break

                if block_arg_index is not None:
                    if loop_id is not None and block_arg_index == 0:
                        arg_term = Term.loop_index(
                            egglog.i64(loop_id),
                            mlir_type_to_egraph_ty_string(operand.type)
                        )
                    elif loop_id is not None and 0 < block_arg_index <= num_iter_args:
                        # Subsequent arguments are loop-carried values
                        arg_term = Term.loop_iter_arg(
                            egglog.i64(loop_id),
                            egglog.i64(block_arg_index - 1),
                            mlir_type_to_egraph_ty_string(operand.type)
                        )
                    else:
                        arg_term = Term.arg(
                            id(block),
                            egglog.i64(block_arg_index),
                            mlir_type_to_egraph_ty_string(operand.type)
                        )
                    operand_terms.append(arg_term)
        return operand_terms

    def _constant_to_term(self, op: MOperation) -> Term:
        if not op.has_attr('value') or len(op.results) != 1:
            raise ValueError(f"Malformed constant operation: {op}")
        value = op.get_attr('value')
        type_string = str(op.results[0].type)
        dtype = mlir_type_to_egraph_ty_string(type_string)
        if type_string in {'f16', 'f32', 'f64'}:
            return Term.lit(LitTerm.float(egglog.f64(float(value))), dtype)
        if type_string == 'index' or re.fullmatch(r'[iu][1-9][0-9]*', type_string):
            return Term.lit(LitTerm.int(egglog.i64(int(value))), dtype)
        raise ValueError(f"Unsupported constant type: {type_string}")

    def _next_id(self) -> int:
        self.next_id += 1
        return self.next_id

def create_tuple_type_string(types: list) -> egglog.String:
    """Create a tuple type string for multiple values."""
    megg_types = []
    for t in types:
        megg_type = mlir_type_to_megg_type(t)
        if megg_type:
            megg_types.append(megg_type)

    if len(megg_types) == 0:
        return egglog.String("void")
    elif len(megg_types) == 1:
        return egglog.String(str(megg_types[0]))
    else:
        tuple_type = TupleType(element_types=tuple(megg_types))
        return egglog.String(str(tuple_type))

def _parse_type_from_string(type_str: str) -> Optional[DataType]:
    """Parse a type string to DataType (helper for MLIR -> Megg types)."""
    type_str = type_str.strip().lower()

    if type_str in ("f16", "half"):
        return MeggFloatType.float_type("f16")
    elif type_str in ("f32", "single", "float"):
        return MeggFloatType.float_type("f32")
    elif type_str in ("f64", "double"):
        return MeggFloatType.float_type("f64")

    elif type_str == "index":
        return IndexType()
    elif type_str == "void":
        return VoidType()

    elif type_str.startswith("i") and type_str[1:].isdigit():
        width = int(type_str[1:])
        return IntType(width=width, signed=True)

    elif type_str.startswith("u") and type_str[1:].isdigit():
        width = int(type_str[1:])
        return IntType(width=width, signed=False)

    elif type_str == '!llvm.ptr':
        return PointerType()
    elif type_str.startswith('(') and type_str.endswith(')'):
        elements = tuple(_parse_type_from_string(part.strip())
                         for part in type_str[1:-1].split(','))
        return TupleType(elements) if all(elements) else None
    match = re.fullmatch(r'(memref|tensor|vector)<((?:[0-9?]+x)*)([ifu][0-9]+)>', type_str)
    if match:
        kind, dimensions, element = match.groups()
        shape = tuple(-1 if dim == '?' else int(dim)
                      for dim in dimensions.rstrip('x').split('x') if dim)
        element_type = _parse_type_from_string(element)
        if kind == 'memref':
            return MemRefType(shape, element_type)
        if kind == 'tensor':
            return TensorType(shape, element_type)
        if kind == 'vector' and len(shape) == 1 and shape[0] > 0:
            return VectorType(element_type, shape[0])
    return None

def megg_to_mlir_type(megg_type: DataType) -> MType:
    """Convert Megg DataType to MLIR Type using C++ wrapper."""
    builder = IRBuilder()
    type_str = str(megg_type)

    if type_str in ("f16", "half"):
        raise ValueError("f16 reconstruction is not supported by the MEGG builder")
    elif type_str in ("f32", "single", "float"):
        return builder.f32()
    elif type_str in ("f64", "double"):
        return builder.f64()

    elif type_str == "index":
        return builder.index()
    elif type_str == "void":
        raise ValueError("Void values have no MLIR value type")

    elif type_str.startswith("i") and type_str[1:].isdigit():
        width = int(type_str[1:])
        if width == 32:
            return builder.i32()
        elif width == 64:
            return builder.i64()
        else:
            return builder.integer(width)

    elif type_str.startswith("u") and type_str[1:].isdigit():
        width = int(type_str[1:])
        return builder.integer(width)

    elif isinstance(megg_type, PointerType):
        return builder.llvm_ptr()
    elif isinstance(megg_type, MemRefType):
        shape = [-(1 << 63) if dim == -1 else dim for dim in megg_type.shape]
        return builder.memref(shape, megg_to_mlir_type(megg_type.element_type))
    raise ValueError(f"Unsupported MEGG type: {megg_type}")

def mlir_type_to_megg_type(mlir_type) -> DataType:
    if isinstance(mlir_type, (tuple, list)):
        return TupleType(tuple(mlir_type_to_megg_type(item) for item in mlir_type))
    result = _parse_type_from_string(str(mlir_type))
    if result is None:
        raise ValueError(f"Unsupported MLIR type: {mlir_type}")
    return result

def mlir_type_to_egraph_ty_string(mlir_type) -> egglog.String:
    """Convert an MLIR type to an e-graph type string."""
    megg_ty = mlir_type_to_megg_type(mlir_type)
    if megg_ty is None:
        raise ValueError(f"Unsupported MLIR type: {mlir_type}")
    return egglog.String(str(megg_ty))

def predicate_to_string(predicate_enum, cmp_type="i"):
    """Convert MLIR compare predicate enum to a human-readable string."""
    if cmp_type == "i":
        mapping = {
            0: "eq",   # equal
            1: "ne",   # not equal
            2: "slt",  # signed less than
            3: "sle",  # signed less or equal
            4: "sgt",  # signed greater than
            5: "sge",  # signed greater or equal
            6: "ult",  # unsigned less than
            7: "ule",  # unsigned less or equal
            8: "ugt",  # unsigned greater than
            9: "uge",  # unsigned greater or equal
            "eq": "eq",
            "ne": "ne",
            "slt": "slt",
            "sle": "sle",
            "sgt": "sgt",
            "sge": "sge",
            "ult": "ult",
            "ule": "ule",
            "ugt": "ugt",
            "uge": "uge",
        }
    elif cmp_type == "f":
        mapping = {
            0: "false",      # always false
            1: "oeq",        # ordered and equal
            2: "ogt",        # ordered and greater than
            3: "oge",        # ordered and greater or equal
            4: "olt",        # ordered and less than
            5: "ole",        # ordered and less or equal
            6: "one",        # ordered and not equal
            7: "ord",        # ordered (no nans)
            8: "uno",        # unordered (any operand is nan)
            9: "ueq",        # unordered or equal
            10: "ugt",       # unordered or greater than
            11: "uge",       # unordered or greater or equal
            12: "ult",       # unordered or less than
            13: "ule",       # unordered or less or equal
            14: "une",       # unordered or not equal
            15: "true",      # always true
            "oeq": "oeq",
            "ogt": "ogt",
            "oge": "oge",
            "olt": "olt",
            "ole": "ole",
            "one": "one",
            "ord": "ord",
            "uno": "uno",
            "ueq": "ueq",
            "ugt": "ugt",
            "uge": "uge",
            "ult": "ult",
            "ule": "ule",
            "une": "une",
        }
    else:
        raise ValueError(f"Unknown cmp_type: {cmp_type}")

    if predicate_enum in mapping:
        return mapping[predicate_enum]
    else:
        raise ValueError(f"Unknown predicate_enum: {predicate_enum}")
