from __future__ import annotations

from typing import Callable, Optional

import circt.ir as ir
import circt.dialects.arith as arith
import circt.dialects.aps as aps
import circt.dialects.memref as memref

from .state import ConversionState
from .types import cast_cadl_type_to_mlir
from .. import cadl_ast


def _create_symbol_read_op(
    op_class_name: str,
    op_name: str,
    result_type: ir.Type,
    symbol_ref: ir.FlatSymbolRefAttr,
) -> ir.Value:
    op_class = getattr(aps, op_class_name, None)
    if op_class is not None:
        return op_class(result_type, symbol_ref).result

    op = ir.Operation.create(
        op_name,
        results=[result_type],
        attributes={"global_name": symbol_ref},
    )
    return op.results[0]


def _create_symbol_write_op(
    op_class_name: str,
    op_name: str,
    value: ir.Value,
    symbol_ref: ir.FlatSymbolRefAttr,
) -> None:
    op_class = getattr(aps, op_class_name, None)
    if op_class is not None:
        op_class(value, symbol_ref)
        return

    ir.Operation.create(
        op_name,
        operands=[value],
        attributes={"global_name": symbol_ref},
    )


def _create_gmem_load(result_type: ir.Type, cpu_addr: ir.Value) -> ir.Value:
    op_class = getattr(aps, "Load", None)
    if op_class is not None:
        return op_class(result_type, cpu_addr).result

    op = ir.Operation.create("aps.load", operands=[cpu_addr], results=[result_type])
    return op.results[0]


def _create_gmem_store(value: ir.Value, cpu_addr: ir.Value) -> None:
    op_class = getattr(aps, "Store", None)
    if op_class is not None:
        op_class(value, cpu_addr)
        return

    ir.Operation.create("aps.store", operands=[value, cpu_addr])


def _create_gmem_copy(
    cpu_addr: ir.Value,
    memrefs: list[ir.Value],
    start: ir.Value,
    length: ir.Value,
    direction: str,
):
    if direction not in {"in", "out"}:
        raise ValueError(f"Unsupported gmem copy direction: {direction}")

    i32_type = ir.IntegerType.get_signless(32)
    attrs = {
        "direction": ir.IntegerAttr.get(i32_type, 0 if direction == "in" else 1)
    }

    return ir.Operation.create(
        "aps.copy",
        operands=[cpu_addr, *memrefs, start, length],
        attributes=attrs,
    )


def function_uses_memory(function) -> bool:
    """Return true when a function-like AST node uses _mem."""
    if not function.body:
        return False
    return stmt_list_uses_memory(function.body)


def flow_uses_memory(flow) -> bool:
    """Return true when a flow uses _mem."""
    if not flow.body:
        return False
    return stmt_list_uses_memory(flow.body)


def stmt_list_uses_memory(stmts) -> bool:
    """Return true when any statement in the list uses _mem."""
    return any(stmt_uses_memory(stmt) for stmt in stmts)


def stmt_uses_memory(stmt) -> bool:
    """Return true when a statement uses _mem.

    This intentionally preserves the previous converter behavior. Broader
    expression coverage should be added as a separate semantic change.
    """
    if isinstance(stmt, cadl_ast.AssignStmt):
        if isinstance(stmt.lhs, (cadl_ast.IndexExpr, cadl_ast.RangeSliceExpr)):
            if isinstance(stmt.lhs.expr, cadl_ast.IdentExpr) and stmt.lhs.expr.name == "_mem":
                return True
        return expr_uses_memory(stmt.rhs)
    if isinstance(stmt, cadl_ast.ExprStmt):
        return expr_uses_memory(stmt.expr)
    if isinstance(stmt, cadl_ast.ReturnStmt):
        return any(expr_uses_memory(expr) for expr in stmt.exprs)
    if isinstance(stmt, cadl_ast.DoWhileStmt):
        return stmt_list_uses_memory(stmt.body) or expr_uses_memory(stmt.condition)
    return False


def expr_uses_memory(expr) -> bool:
    """Return true when an expression uses _mem."""
    if isinstance(expr, (cadl_ast.IndexExpr, cadl_ast.RangeSliceExpr)) and isinstance(expr.expr, cadl_ast.IdentExpr):
        return expr.expr.name == "_mem"
    if isinstance(expr, cadl_ast.BinaryExpr):
        return expr_uses_memory(expr.left) or expr_uses_memory(expr.right)
    if isinstance(expr, cadl_ast.UnaryExpr):
        return expr_uses_memory(expr.operand)
    if isinstance(expr, cadl_ast.CallExpr):
        return any(expr_uses_memory(arg) for arg in expr.args)
    return False


class GlobalEmitter:
    """Emit and resolve module-level memref globals."""

    def __init__(self, state: ConversionState):
        self.state = state

    @property
    def global_ops(self) -> dict[str, memref.GlobalOp]:
        return self.state.globals.ops

    @property
    def global_refs(self) -> dict[str, ir.Value]:
        return self.state.globals.refs

    def get_reference(self, global_name: str) -> ir.Value:
        """Get a cached memref.get_global reference for a declared global."""
        if global_name not in self.global_refs:
            if global_name not in self.global_ops:
                raise RuntimeError(
                    f"Global {global_name} not declared (no GlobalOp found)"
                )

            memory_type = self._memory_type(global_name)
            global_ref = memref.GetGlobalOp(memory_type, global_name)
            self.global_refs[global_name] = global_ref.result
        return self.global_refs[global_name]

    def is_scalar(self, global_name: str) -> bool:
        """Check if a global is represented as rank-0 memref."""
        if global_name not in self.global_ops:
            return False

        memory_type = self._memory_type(global_name)
        if isinstance(memory_type, ir.MemRefType):
            return len(memory_type.shape) == 0
        return False

    def element_type(self, global_name: str) -> Optional[ir.Type]:
        """Return the element type of a global memref."""
        if global_name not in self.global_ops:
            return None

        memory_type = self._memory_type(global_name)
        if isinstance(memory_type, ir.MemRefType):
            return memory_type.element_type
        return None

    def declare_cpu_memory(self, set_symbol: Callable[[str, str], None]) -> None:
        """Declare the implicit CPU memory backing _mem accesses."""
        if "_cpu_memory" in self.global_ops:
            return

        element_type = ir.IntegerType.get_signless(32)
        memory_type = memref.MemRefType.get([1024], element_type)
        global_name = "_cpu_memory"
        global_op = memref.GlobalOp(global_name, memory_type)

        self.global_ops[global_name] = global_op
        set_symbol("_cpu_memory", global_name)

    def convert_static(
        self, static: cadl_ast.Static, set_symbol: Callable[[str, str], None]
    ) -> None:
        """Convert a CADL static declaration to memref.global."""
        self._convert_global_decl(
            name=static.id,
            ty=static.ty,
            attrs=static.attrs,
            expr=static.expr,
            set_symbol=set_symbol,
        )

    def convert_register(
        self, register: cadl_ast.Register, set_symbol: Callable[[str, str], None]
    ) -> None:
        """Convert a CADL register declaration to a scalar memref.global."""
        self._convert_global_decl(
            name=register.name,
            ty=register.ty,
            attrs=register.attrs,
            expr=None,
            set_symbol=set_symbol,
        )

    def _convert_global_decl(
        self,
        name: str,
        ty: cadl_ast.DataType,
        attrs: dict[str, Optional[cadl_ast.Expr]],
        expr: Optional[cadl_ast.Expr],
        set_symbol: Callable[[str, str], None],
    ) -> None:
        """Convert a top-level storage declaration to memref.global."""
        mlir_type = cast_cadl_type_to_mlir(ty)
        global_name = name

        initial_value = None
        initial_values_list = None
        if expr:
            if isinstance(expr, cadl_ast.LitExpr):
                initial_value = expr.literal.lit.value
            elif isinstance(expr, cadl_ast.AggregateExpr):
                initial_values_list = []
                for elem_expr in expr.elements:
                    if isinstance(elem_expr, cadl_ast.LitExpr):
                        initial_values_list.append(elem_expr.literal.lit.value)
                    else:
                        initial_values_list = None
                        break

        if isinstance(mlir_type, ir.MemRefType):
            memref_type = mlir_type
        else:
            memref_type = ir.MemRefType.get([], mlir_type)

        if initial_value is not None:
            if isinstance(mlir_type, ir.MemRefType):
                raise RuntimeError(
                    f"Scalar initialization provided for array type: {mlir_type}"
                )

            element_attr = ir.IntegerAttr.get(mlir_type, initial_value)
            attr = ir.DenseElementsAttr.get_splat(
                ir.RankedTensorType.get([], mlir_type), element_attr
            )
            global_op = memref.GlobalOp(
                global_name, memref_type, initial_value=attr, constant=True
            )
        elif initial_values_list is not None:
            if not isinstance(mlir_type, ir.MemRefType):
                raise RuntimeError(
                    f"Array initialization provided for non-array type: {mlir_type}"
                )

            element_type = mlir_type.element_type
            element_attrs = [
                ir.IntegerAttr.get(element_type, val) for val in initial_values_list
            ]
            tensor_type = ir.RankedTensorType.get(mlir_type.shape, element_type)
            dense_attr = ir.DenseElementsAttr.get(element_attrs, tensor_type)
            global_op = memref.GlobalOp(
                global_name, memref_type, initial_value=dense_attr, constant=True
            )
        else:
            global_op = memref.GlobalOp(global_name, memref_type)

        global_op.attributes["var_name"] = ir.StringAttr.get(global_name)
        self.global_ops[global_name] = global_op

        if attrs:
            for attr_name, attr_value in attrs.items():
                mlir_attr = self.convert_attribute_value(attr_value)
                if mlir_attr is not None:
                    global_op.attributes[attr_name] = mlir_attr

        set_symbol(name, global_name)

    def convert_attribute_value(self, expr: Optional[cadl_ast.Expr]) -> Optional[ir.Attribute]:
        """Convert CADL static/directive attribute syntax to MLIR attributes."""
        if expr is None:
            return ir.UnitAttr.get()

        if isinstance(expr, cadl_ast.StringLitExpr):
            return ir.StringAttr.get(expr.value)

        if isinstance(expr, cadl_ast.LitExpr):
            literal = expr.literal
            if isinstance(literal.lit, (cadl_ast.LiteralInner_Fixed, cadl_ast.LiteralInner_Float)):
                value = literal.lit.value
                mlir_type = cast_cadl_type_to_mlir(literal.ty)
                if isinstance(value, int):
                    return ir.IntegerAttr.get(mlir_type, value)
                if isinstance(value, float):
                    return ir.FloatAttr.get(mlir_type, value)

        if isinstance(expr, cadl_ast.ArrayLiteralExpr):
            element_attrs = []
            for elem_expr in expr.elements:
                elem_attr = self.convert_attribute_value(elem_expr)
                if elem_attr is not None:
                    element_attrs.append(elem_attr)
            return ir.ArrayAttr.get(element_attrs)

        if isinstance(expr, cadl_ast.IdentExpr):
            return ir.StringAttr.get(expr.name)

        return ir.StringAttr.get(str(expr))

    def _memory_type(self, global_name: str) -> ir.Type:
        type_attr = self.global_ops[global_name].type_
        if isinstance(type_attr, ir.TypeAttr):
            return type_attr.value
        return type_attr


class MemoryEmitter:
    """Emit APS memory/register-file access operations."""

    def __init__(self, converter):
        self.converter = converter

    def convert_index_expr(self, expr: cadl_ast.IndexExpr) -> ir.Value:
        """
        Convert IndexExpr to appropriate MLIR operation based on the base expression.

        Handles:
        - _irf[rs] -> aps.ReadIRF
        - _mem[addr] -> aps.load
        - regular array[idx] -> aps.read_smem
        """
        c = self.converter
        if isinstance(expr.expr, cadl_ast.IdentExpr):
            base_name = expr.expr.name

            if base_name == "_irf":
                if len(expr.indices) != 1:
                    raise ValueError("_irf access requires exactly one index")

                reg_index = c._convert_expr(expr.indices[0])
                result_type = ir.IntegerType.get_signless(32)
                return aps.ReadIRF(result_type, reg_index).result

            if base_name == "_csr":
                if len(expr.indices) != 1:
                    raise ValueError("_csr access requires exactly one index")
                csr_name = self.get_csr_name(expr.indices[0])
                symbol_ref = ir.FlatSymbolRefAttr.get(csr_name)
                element_type = c.global_emitter.element_type(csr_name)
                if element_type is None:
                    raise RuntimeError(
                        f"Cannot infer CSR register element type for {csr_name}"
                    )
                return _create_symbol_read_op(
                    "ReadCSR", "aps.read_csr", element_type, symbol_ref
                )

            if base_name == "_mem":
                if len(expr.indices) != 1:
                    raise ValueError("_mem access requires exactly one index")
                addr = c._convert_expr(expr.indices[0])
                result_type = ir.IntegerType.get_signless(32)
                return _create_gmem_load(result_type, addr)
            else:
                symbol_value = c.get_symbol(base_name)
                if isinstance(symbol_value, str):
                    memref = c.global_emitter.get_reference(symbol_value)
                else:
                    memref = c._convert_expr(expr.expr)
                indices = [c._convert_expr(idx) for idx in expr.indices]
        else:
            memref = c._convert_expr(expr.expr)
            indices = [c._convert_expr(idx) for idx in expr.indices]

        element_type = self.get_memref_element_type(memref)
        if element_type is None:
            element_type = ir.IntegerType.get_signless(32)

        return aps.ReadSmem(element_type, memref, indices).result

    def get_memref_element_type(self, memref_value: ir.Value) -> Optional[ir.Type]:
        """Get the element type of a memref value."""
        if ir.MemRefType.isinstance(memref_value.type):
            return ir.MemRefType(memref_value.type).element_type
        return None

    def get_cpu_memory(self) -> ir.Value:
        """Return the declared CPU memory global reference."""
        c = self.converter
        if "_cpu_memory" not in c.global_ops:
            raise RuntimeError(
                "Global CPU memory should be declared at module level before use"
            )
        return c.global_emitter.get_reference("_cpu_memory")

    def convert_index_assignment(self, lhs: cadl_ast.IndexExpr, rhs_value: ir.Value) -> None:
        """
        Convert indexed assignment to appropriate MLIR operation.

        Handles:
        - _irf[rd] = value -> aps.WriteIRF
        - _mem[addr] = value -> aps.store
        - regular array[idx] = value -> aps.write_smem
        """
        c = self.converter
        if isinstance(lhs.expr, cadl_ast.IdentExpr):
            base_name = lhs.expr.name

            if base_name == "_irf":
                if len(lhs.indices) != 1:
                    raise ValueError("_irf assignment requires exactly one index")

                reg_index = c._convert_expr(lhs.indices[0])
                aps.WriteIRF(reg_index, rhs_value)
                return

            if base_name == "_csr":
                if len(lhs.indices) != 1:
                    raise ValueError("_csr assignment requires exactly one index")
                csr_name = self.get_csr_name(lhs.indices[0])
                target_type = c.global_emitter.element_type(csr_name)
                if target_type is not None:
                    rhs_value = c.expr_emitter.cast_type(rhs_value, target_type)
                symbol_ref = ir.FlatSymbolRefAttr.get(csr_name)
                _create_symbol_write_op(
                    "WriteCSR", "aps.write_csr", rhs_value, symbol_ref
                )
                return

            if base_name == "_mem":
                if len(lhs.indices) != 1:
                    raise ValueError("_mem assignment requires exactly one index")

                addr = c._convert_expr(lhs.indices[0])
                rhs_value = c.expr_emitter.cast_type(
                    rhs_value, ir.IntegerType.get_signless(32)
                )
                _create_gmem_store(rhs_value, addr)
                return

            base_value = self.get_memref_for_symbol(base_name)
            indices = [c._convert_expr(idx) for idx in lhs.indices]
        else:
            base_value = c._convert_expr(lhs.expr)
            indices = [c._convert_expr(idx) for idx in lhs.indices]

        target_type = self.get_memref_element_type(base_value)
        if target_type is not None:
            rhs_value = c.expr_emitter.cast_type(rhs_value, target_type)

        aps.WriteSmem(rhs_value, base_value, indices)

    def is_burst_operation(self, stmt: cadl_ast.AssignStmt) -> bool:
        """Detect burst read/write assignments."""
        if isinstance(stmt.rhs, cadl_ast.RangeSliceExpr) and isinstance(
            stmt.rhs.expr, cadl_ast.IdentExpr
        ):
            if stmt.rhs.expr.name == "_mem":
                return True

        if isinstance(stmt.lhs, cadl_ast.RangeSliceExpr) and isinstance(
            stmt.lhs.expr, cadl_ast.IdentExpr
        ):
            if stmt.lhs.expr.name == "_mem":
                return True

        return False

    def convert_burst_operation(self, stmt: cadl_ast.AssignStmt) -> None:
        """Convert burst read/write operations to APS burst ops."""
        if isinstance(stmt.rhs, cadl_ast.RangeSliceExpr) and isinstance(
            stmt.rhs.expr, cadl_ast.IdentExpr
        ):
            if stmt.rhs.expr.name == "_mem":
                self.convert_burst_load(stmt)
                return

        if isinstance(stmt.lhs, cadl_ast.RangeSliceExpr) and isinstance(
            stmt.lhs.expr, cadl_ast.IdentExpr
        ):
            if stmt.lhs.expr.name == "_mem":
                self.convert_burst_store(stmt)
                return

        raise ValueError("Invalid burst operation pattern")

    def extract_literal_value(self, expr: cadl_ast.Expr) -> int:
        """Extract constant integer value from LitExpr."""
        if not isinstance(expr, cadl_ast.LitExpr):
            raise ValueError(f"Expected LitExpr, got {type(expr).__name__}")

        if not isinstance(expr.literal.lit, (cadl_ast.LiteralInner_Fixed, cadl_ast.LiteralInner_Float)):
            raise ValueError("Literal has no value attribute")

        return expr.literal.lit.value

    def convert_burst_load(self, stmt: cadl_ast.AssignStmt) -> None:
        """Convert buffer[offset +: ] = _mem[cpu_addr +: length]."""
        c = self.converter
        lhs = stmt.lhs
        rhs = stmt.rhs

        if not isinstance(lhs, cadl_ast.RangeSliceExpr):
            raise ValueError("Burst load LHS must be a range slice expression")
        if not isinstance(rhs, cadl_ast.RangeSliceExpr):
            raise ValueError("Burst load RHS must be a range slice expression")

        cpu_addr = c._convert_expr(rhs.start)
        if rhs.length is None:
            raise ValueError("Burst read must have explicit length")

        rhs_length_val = self.extract_literal_value(rhs.length)

        if isinstance(lhs.expr, cadl_ast.IdentExpr):
            buffer_memref = self.get_memref_for_symbol(lhs.expr.name)
        else:
            buffer_memref = c._convert_expr(lhs.expr)

        start_offset = c._convert_expr(lhs.start)

        if lhs.length is not None:
            lhs_length_val = self.extract_literal_value(lhs.length)
            if lhs_length_val != rhs_length_val:
                raise ValueError(
                    f"Burst length mismatch: buffer[+:{lhs_length_val}] = "
                    f"_mem[+:{rhs_length_val}]"
                )

        i32_type = ir.IntegerType.get_signless(32)
        length = arith.ConstantOp(i32_type, rhs_length_val).result
        burst_op = _create_gmem_copy(
            cpu_addr, [buffer_memref], start_offset, length, "in"
        )
        self.apply_pending_directives(burst_op)

    def convert_burst_store(self, stmt: cadl_ast.AssignStmt) -> None:
        """Convert _mem[cpu_addr +: length] = buffer[offset +: ]."""
        c = self.converter
        lhs = stmt.lhs
        rhs = stmt.rhs

        if not isinstance(lhs, cadl_ast.RangeSliceExpr):
            raise ValueError("Burst store LHS must be a range slice expression")
        if not isinstance(rhs, cadl_ast.RangeSliceExpr):
            raise ValueError("Burst store RHS must be a range slice expression")

        cpu_addr = c._convert_expr(lhs.start)
        if lhs.length is None:
            raise ValueError("Burst write must have explicit length")

        lhs_length_val = self.extract_literal_value(lhs.length)

        if isinstance(rhs.expr, cadl_ast.IdentExpr):
            buffer_memref = self.get_memref_for_symbol(rhs.expr.name)
        else:
            buffer_memref = c._convert_expr(rhs.expr)

        start_offset = c._convert_expr(rhs.start)

        if rhs.length is not None:
            rhs_length_val = self.extract_literal_value(rhs.length)
            if lhs_length_val != rhs_length_val:
                raise ValueError(
                    f"Burst length mismatch: _mem[+:{lhs_length_val}] = "
                    f"buffer[+:{rhs_length_val}]"
                )

        i32_type = ir.IntegerType.get_signless(32)
        length = arith.ConstantOp(i32_type, lhs_length_val).result
        burst_op = _create_gmem_copy(
            cpu_addr, [buffer_memref], start_offset, length, "out"
        )
        self.apply_pending_directives(burst_op)

    def get_memref_for_symbol(self, symbol_name: str) -> ir.Value:
        """Get a memref value for local or global array-like symbols."""
        c = self.converter
        symbol_value = c.get_symbol(symbol_name)

        if symbol_value is None:
            raise ValueError(f"Undefined symbol: {symbol_name}")

        if isinstance(symbol_value, str):
            static_var = None
            for static in c.proc.statics.values():
                if static.id == symbol_name:
                    static_var = static
                    break

            if static_var:
                return c.global_emitter.get_reference(symbol_value)
            raise ValueError(f"Cannot find static definition for global: {symbol_name}")

        return symbol_value

    def get_csr_name(self, expr: cadl_ast.Expr) -> str:
        """Resolve _csr[name] to a declared CSR register symbol."""
        c = self.converter
        if not isinstance(expr, cadl_ast.IdentExpr):
            raise ValueError("_csr access expects a register name, e.g. _csr[cfg]")

        csr_name = expr.name
        if csr_name not in c.proc.csrs:
            raise ValueError(f"Undefined CSR register: {csr_name}")
        if csr_name not in c.global_ops:
            raise RuntimeError(f"CSR register {csr_name} was not lowered as a global")
        return csr_name

    def apply_pending_directives(self, op) -> None:
        """Apply and clear pending directives on a burst operation."""
        c = self.converter
        if not c.pending_directives:
            return

        for directive in c.pending_directives:
            attr_name = directive.name
            if directive.expr and isinstance(directive.expr, cadl_ast.LitExpr):
                value = directive.expr.literal.lit.value
                attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)
                op.operation.attributes[attr_name] = attr
            else:
                op.operation.attributes[attr_name] = ir.BoolAttr.get(True)
        c.pending_directives = []
