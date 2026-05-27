from __future__ import annotations

import circt.ir as ir
import circt.dialects.aps as aps
import circt.dialects.func as func
from .. import cadl_ast



class StatementEmitter:
    """Emit CADL statements into MLIR operations."""

    def __init__(self, converter):
        self.converter = converter

    def emit_list(self, stmts: list[cadl_ast.Stmt]) -> None:
        for stmt in stmts:
            self.emit(stmt)

    def emit(self, stmt: cadl_ast.Stmt) -> None:
        c = self.converter

        if isinstance(stmt, cadl_ast.ExprStmt):
            c.expr_emitter.emit(stmt.expr)
            return

        if isinstance(stmt, cadl_ast.AssignStmt):
            self.emit_assign(stmt)
            return

        if isinstance(stmt, cadl_ast.ReturnStmt):
            ret_values = [c.expr_emitter.emit(expr) for expr in stmt.exprs]
            func.ReturnOp(ret_values)
            return

        if isinstance(stmt, cadl_ast.DoWhileStmt):
            self.emit_do_while(stmt)
            c.pending_directives = []
            return

        if isinstance(stmt, cadl_ast.DirectiveStmt):
            c.pending_directives.append(stmt)
            return

        raise NotImplementedError(f"Statement type not yet supported: {type(stmt)}")

    def emit_assign(self, stmt: cadl_ast.AssignStmt) -> None:
        c = self.converter

        if c.memory_emitter.is_burst_operation(stmt):
            c.memory_emitter.convert_burst_operation(stmt)
            return

        self._track_constant_assignment(stmt)

        rhs_value = c.expr_emitter.emit(stmt.rhs)
        rhs_value = c.expr_emitter.convert_type_if_needed(
            rhs_value, stmt.type_annotation
        )

        if isinstance(stmt.lhs, cadl_ast.IdentExpr):
            symbol_value = c.get_symbol(stmt.lhs.name)
            if isinstance(symbol_value, str) and c.global_emitter.is_scalar(
                symbol_value
            ):
                target_type = c.global_emitter.element_type(symbol_value)
                if target_type is not None:
                    rhs_value = c.expr_emitter.cast_type(rhs_value, target_type)
                symbol_ref = ir.FlatSymbolRefAttr.get(symbol_value)
                aps.GlobalStore(rhs_value, symbol_ref)
            else:
                c.set_symbol(stmt.lhs.name, rhs_value, stmt.type_annotation)
            return

        if isinstance(stmt.lhs, cadl_ast.IndexExpr):
            c.memory_emitter.convert_index_assignment(stmt.lhs, rhs_value)
            return

        if isinstance(stmt.lhs, cadl_ast.RangeSliceExpr):
            raise NotImplementedError(
                "Non-burst range slice assignments not yet supported"
            )

        raise NotImplementedError(
            f"Complex LHS assignment not yet supported: {type(stmt.lhs)}"
        )

    def emit_do_while(self, stmt: cadl_ast.DoWhileStmt) -> None:
        c = self.converter
        c.loop_transformer.validate_loop_body_assignments(stmt)
        for_pattern = c.loop_transformer.analyze_and_detect_for_pattern(stmt)

        if for_pattern:
            c.loop_transformer.emit_scf_for(stmt, for_pattern)
        else:
            c.loop_transformer.emit_scf_while(stmt)

    def _track_constant_assignment(self, stmt: cadl_ast.AssignStmt) -> None:
        c = self.converter
        if not isinstance(stmt.lhs, cadl_ast.IdentExpr):
            return

        if isinstance(stmt.rhs, cadl_ast.LitExpr):
            if isinstance(
                stmt.rhs.literal.lit, (cadl_ast.LiteralInner_Fixed, cadl_ast.LiteralInner_Float)
            ):
                c.constant_vars[stmt.lhs.name] = stmt.rhs.literal.lit.value
            return

        if stmt.lhs.name in c.constant_vars:
            del c.constant_vars[stmt.lhs.name]
