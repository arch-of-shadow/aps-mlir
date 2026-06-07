"""
CADL AST to MLIR Converter

This module provides a framework for converting CADL Abstract Syntax Trees
to MLIR Intermediate Representation, leveraging CIRCT dialects for hardware
synthesis and optimization.
"""

from __future__ import annotations
from typing import List, Optional, Union

import circt
import circt.ir as ir

from .loop import LoopTransformer
from .expr import ExprEmitter
from .memory import GlobalEmitter, MemoryEmitter
from .state import ConversionState, SymbolScope
from .stmt import StatementEmitter
from .types import cast_cadl_type_to_mlir
import circt.dialects.func as func
import circt.dialects.memref as memref
from .. import cadl_ast



class CADLMLIRConverter:
    """
    Main converter class for transforming CADL AST to MLIR IR

    Maintains MLIR context, module, and symbol table for SSA form generation.
    Uses CIRCT dialects for hardware-oriented operations.
    """

    def __init__(self):
        self.context = ir.Context()
        self.state = ConversionState(self.context)
        self.global_emitter = GlobalEmitter(self.state)
        self.expr_emitter = ExprEmitter(self)
        self.memory_emitter = MemoryEmitter(self)
        self.stmt_emitter = StatementEmitter(self)
        self.module: Optional[ir.Module] = None

        self._load_dialects()

        self.builder: Optional[ir.InsertionPoint] = None
        self.loop_transformer = LoopTransformer(self)

    @property
    def module(self) -> Optional[ir.Module]:
        return self.state.module

    @module.setter
    def module(self, value: Optional[ir.Module]) -> None:
        self.state.module = value

    @property
    def current_scope(self) -> SymbolScope:
        return self.state.current_scope

    @current_scope.setter
    def current_scope(self, value: SymbolScope) -> None:
        self.state.current_scope = value

    @property
    def scope_stack(self) -> List[SymbolScope]:
        return self.state.scope_stack

    @property
    def current_global_refs(self) -> dict[str, ir.Value]:
        return self.state.globals.refs

    @current_global_refs.setter
    def current_global_refs(self, value: dict[str, ir.Value]) -> None:
        self.state.globals.refs = value

    @property
    def global_ops(self) -> dict[str, memref.GlobalOp]:
        return self.state.globals.ops

    @property
    def constant_vars(self) -> dict[str, int]:
        return self.state.constant_vars

    @property
    def pending_directives(self) -> List[cadl_ast.DirectiveStmt]:
        return self.state.pending_directives

    @pending_directives.setter
    def pending_directives(self, value: List[cadl_ast.DirectiveStmt]) -> None:
        self.state.pending_directives = value

    def _load_dialects(self) -> None:
        """Load required MLIR and CIRCT dialects"""
        with self.context:
            circt.register_dialects(self.context)

    def push_scope(self) -> None:
        """Push new symbol scope onto stack"""
        self.state.push_scope()

    def pop_scope(self) -> None:
        """Pop symbol scope from stack"""
        self.state.pop_scope()

    def get_symbol(self, name: str) -> Optional[Union[ir.Value, str]]:
        """Get SSA value for symbol name"""
        typed_val = self.current_scope.get(name)
        return typed_val.value if typed_val else None

    def get_symbol_type(self, name: str) -> Optional[cadl_ast.DataType]:
        """Get CADL type for symbol name"""
        typed_val = self.current_scope.get(name)
        return typed_val.cadl_type if typed_val else None

    def set_symbol(
        self,
        name: str,
        value: Union[ir.Value, str],
        cadl_type: Optional[cadl_ast.DataType] = None,
    ) -> None:
        """Set SSA value and CADL type for symbol name in current scope"""
        self.current_scope.set(name, value, cadl_type)

    def convert_cadl_type(
        self, cadl_type: Union[cadl_ast.BasicType, cadl_ast.DataType, cadl_ast.CompoundType]
    ) -> ir.Type:
        """
        Convert CADL type to MLIR type

        Maps CADL type system to appropriate MLIR types.
        Both signed and unsigned CADL types map to signless MLIR integers.
        Signedness is handled by operation semantics (e.g., divsi vs divui).
        """
        return cast_cadl_type_to_mlir(cadl_type)

    def convert_proc(self, proc: cadl_ast.Proc) -> ir.Module:
        """
        Convert CADL Proc to MLIR Module

        Creates top-level MLIR module containing all functions, flows,
        and global variables from the processor definition.
        """
        self.proc = proc

        with self.context, ir.Location.unknown():
            self.module = ir.Module.create()

            with ir.InsertionPoint(self.module.body):
                self.builder = ir.InsertionPoint.current

                for static in proc.statics.values():
                    self.global_emitter.convert_static(static, self.set_symbol)

                for register in proc.registers.values():
                    self.global_emitter.convert_register(register, self.set_symbol)

                for flow in proc.flows.values():
                    self._convert_flow(flow)

        return self.module

    def _convert_flow(self, flow: cadl_ast.Flow) -> ir.Operation:
        """Convert CADL Flow to MLIR function (for now)"""
        # TODO: Implement hardware-specific flow conversion

        arg_types = [self.convert_cadl_type(dtype) for _, dtype in flow.inputs]
        ret_types = []  # TODO: Determine from flow analysis

        func_type = ir.FunctionType.get(arg_types, ret_types)
        func_op = func.FuncOp(f"flow_{flow.name}", func_type)

        if flow.attrs and flow.attrs.attrs:
            for attr_name, attr_expr in flow.attrs.attrs.items():
                if attr_expr and isinstance(attr_expr, cadl_ast.LitExpr):
                    attr_value = attr_expr.literal.lit.value
                    attr = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(32), attr_value
                    )
                    func_op.attributes[attr_name] = attr
                elif attr_expr is None:
                    func_op.attributes[attr_name] = ir.UnitAttr.get()

        entry_block = func_op.add_entry_block()

        with ir.InsertionPoint(entry_block):
            self.push_scope()

            for i, (name, _) in enumerate(flow.inputs):
                arg_value = entry_block.arguments[i]
                self.set_symbol(name, arg_value)

            if flow.body:
                self.stmt_emitter.emit_list(flow.body)

            block_ops = list(entry_block.operations)
            if not block_ops or not isinstance(block_ops[-1], func.ReturnOp):
                func.ReturnOp([])

            self.pop_scope()

        return func_op

    def _convert_stmt_list(self, stmts: List[cadl_ast.Stmt]) -> None:
        """Compatibility callback used by LoopTransformer."""
        self.stmt_emitter.emit_list(stmts)

    def _convert_expr(self, expr: cadl_ast.Expr) -> ir.Value:
        """Convert expression to MLIR SSA value"""
        return self.expr_emitter.emit(expr)


def convert_cadl_to_mlir(proc: cadl_ast.Proc, run_cse: bool = True) -> ir.Module:
    """
    Main entry point for converting CADL Proc to MLIR Module

    Args:
        proc: CADL processor AST to convert
        run_cse: Whether to run Common Subexpression Elimination pass (default: True)

    Returns:
        MLIR module containing the converted representation
    """
    converter = CADLMLIRConverter()
    module = converter.convert_proc(proc)

    if run_cse:
        with converter.context:
            from circt.passmanager import PassManager

            pm = PassManager.parse("builtin.module(cse)")
            pm.run(module.operation)

    return module

def validate_mlir(mlir_text: str) -> bool:
    with ir.Context() as ctx:
        circt.register_dialects(ctx)
        ir.Module.parse(mlir_text)
    return True
