from __future__ import annotations

import circt.ir as ir
import circt.dialects.aps as aps
import circt.dialects.func as func
import circt.dialects.scf as scf
from .. import cadl_ast
from .state import SymbolKind



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

        if isinstance(stmt, cadl_ast.IfStmt):
            self.emit_if(stmt)
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

    def emit_if(self, stmt: cadl_ast.IfStmt) -> None:
        c = self.converter

        merge_names = self._collect_merge_names(stmt)
        merge_name_set = set(merge_names)
        shadowed = merge_name_set & self._collect_let_names(stmt.then_body)
        if stmt.else_body is not None:
            shadowed |= merge_name_set & self._collect_let_names(stmt.else_body)
        if shadowed:
            names = ", ".join(sorted(shadowed))
            raise RuntimeError(
                f"CADL semantic error: branch-local let cannot shadow merged "
                f"outer variable(s): {names}"
            )

        merge_values = [c.get_symbol(name) for name in merge_names]
        result_types = [value.type for value in merge_values]

        condition = c.expr_emitter.emit(stmt.condition)
        condition = c.expr_emitter.cast_type(
            condition, ir.IntegerType.get_signless(1), condition_context=True
        )

        has_else = stmt.else_body is not None or bool(merge_names)
        if_op = scf.IfOp(condition, result_types, hasElse=has_else)

        with ir.InsertionPoint(if_op.then_block):
            self._emit_if_branch(stmt.then_body, merge_names, merge_values)

        if has_else:
            with ir.InsertionPoint(if_op.else_block):
                self._emit_if_branch(stmt.else_body or [], merge_names, merge_values)

        for i, name in enumerate(merge_names):
            c.set_symbol(name, if_op.results[i], c.get_symbol_type(name))

    def _emit_if_branch(
        self,
        stmts: list[cadl_ast.Stmt],
        merge_names: list[str],
        fallback_values: list[ir.Value],
    ) -> None:
        c = self.converter
        c.push_scope()
        self.emit_list(stmts)

        yield_values = []
        for name, fallback_value in zip(merge_names, fallback_values):
            value = c.get_symbol(name)
            if value is None:
                value = fallback_value
            if isinstance(value, str):
                raise RuntimeError(
                    f"CADL semantic error: cannot yield global symbol '{name}' "
                    "from if branch"
                )
            value = c.expr_emitter.cast_type(value, fallback_value.type)
            yield_values.append(value)

        scf.YieldOp(yield_values)
        c.pop_scope()

    def _collect_merge_names(self, stmt: cadl_ast.IfStmt) -> list[str]:
        names = []
        seen = set()
        for name in self._assigned_outer_names(stmt.then_body):
            if name not in seen and self._is_mergeable_outer_value(name):
                names.append(name)
                seen.add(name)
        for name in self._assigned_outer_names(stmt.else_body or []):
            if name not in seen and self._is_mergeable_outer_value(name):
                names.append(name)
                seen.add(name)
        return names

    def _is_mergeable_outer_value(self, name: str) -> bool:
        binding = self.converter.current_scope.get(name)
        if binding is None:
            return False
        return binding.kind == SymbolKind.VALUE

    def _assigned_outer_names(self, stmts: list[cadl_ast.Stmt]) -> list[str]:
        names = []
        for stmt in stmts:
            if isinstance(stmt, cadl_ast.AssignStmt):
                if not stmt.is_let and isinstance(stmt.lhs, cadl_ast.IdentExpr):
                    names.append(stmt.lhs.name)
            elif isinstance(stmt, cadl_ast.IfStmt):
                names.extend(self._assigned_outer_names(stmt.then_body))
                names.extend(self._assigned_outer_names(stmt.else_body or []))
            elif isinstance(stmt, cadl_ast.GuardStmt):
                names.extend(self._assigned_outer_names([stmt.stmt]))
            elif isinstance(stmt, cadl_ast.SpawnStmt):
                names.extend(self._assigned_outer_names(stmt.stmts))
            elif isinstance(stmt, cadl_ast.DoWhileStmt):
                names.extend(binding.id for binding in stmt.bindings)
        return names

    def _collect_let_names(self, stmts: list[cadl_ast.Stmt]) -> set[str]:
        names = set()
        for stmt in stmts:
            if isinstance(stmt, cadl_ast.AssignStmt):
                if stmt.is_let and isinstance(stmt.lhs, cadl_ast.IdentExpr):
                    names.add(stmt.lhs.name)
            elif isinstance(stmt, cadl_ast.IfStmt):
                names.update(self._collect_let_names(stmt.then_body))
                names.update(self._collect_let_names(stmt.else_body or []))
            elif isinstance(stmt, cadl_ast.GuardStmt):
                names.update(self._collect_let_names([stmt.stmt]))
            elif isinstance(stmt, cadl_ast.SpawnStmt):
                names.update(self._collect_let_names(stmt.stmts))
            elif isinstance(stmt, cadl_ast.DoWhileStmt):
                names.update(self._collect_let_names(stmt.body))
        return names

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
