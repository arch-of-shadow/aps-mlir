from __future__ import annotations

from typing import Optional, Union

import circt.ir as ir
import circt.dialects.arith as arith
import circt.dialects.aps as aps
import circt.dialects.comb as comb

from .. import cadl_ast
from .types import cast_cadl_type_to_mlir


class ExprEmitter:
    """Emit expression-local arithmetic and type conversion operations."""

    def __init__(self, converter):
        self.converter = converter

    def emit(self, expr: cadl_ast.Expr) -> ir.Value:
        """Convert a CADL expression to an MLIR SSA value."""
        c = self.converter

        if isinstance(expr, cadl_ast.LitExpr):
            literal = expr.literal
            mlir_type = cast_cadl_type_to_mlir(literal.ty)

            if isinstance(literal.lit, (cadl_ast.LiteralInner_Fixed, cadl_ast.LiteralInner_Float)):
                value = literal.lit.value
                return arith.ConstantOp(mlir_type, value).result
            raise NotImplementedError(
                f"Literal type not supported: {type(literal.lit)}"
            )

        if isinstance(expr, cadl_ast.IdentExpr):
            return self.emit_ident(expr)

        if isinstance(expr, cadl_ast.BinaryExpr):
            left = self.emit(expr.left)
            right = self.emit(expr.right)
            return self.convert_binary_op(expr.op, left, right, expr)

        if isinstance(expr, cadl_ast.UnaryExpr):
            operand = self.emit(expr.operand)
            return self.convert_unary_op(expr.op, operand)

        if isinstance(expr, cadl_ast.IndexExpr):
            return c.memory_emitter.convert_index_expr(expr)

        if isinstance(expr, cadl_ast.SliceExpr):
            return self.convert_slice_expr(expr)

        if isinstance(expr, cadl_ast.IfExpr):
            return self.convert_if_expr(expr)

        if isinstance(expr, cadl_ast.SelectExpr):
            return self.convert_select_expr(expr)

        raise NotImplementedError(f"Expression type not yet supported: {type(expr)}")

    def convert_type_if_needed(
        self, value: ir.Value, type_annotation: Optional[cadl_ast.DataType]
    ) -> ir.Value:
        """Convert a value to an assignment annotation type when needed."""
        if type_annotation is None:
            return value

        use_sign_extend = False
        if isinstance(type_annotation, cadl_ast.DataType_Single):
            use_sign_extend = isinstance(
                type_annotation.basic_type, cadl_ast.BasicType_ApFixed
            )
        return self.cast_type(
            value,
            cast_cadl_type_to_mlir(type_annotation),
            use_sign_extend=use_sign_extend,
        )

    def emit_ident(self, expr: cadl_ast.IdentExpr) -> ir.Value:
        """Resolve an identifier as either an SSA value or scalar global load."""
        c = self.converter
        value = c.get_symbol(expr.name)
        if value is None:
            raise ValueError(f"Undefined symbol: {expr.name}")

        if not isinstance(value, str):
            return value

        if not c.global_emitter.is_scalar(value):
            raise TypeError(
                f"Global array '{expr.name}' cannot be used as a scalar expression; "
                "use an indexed access or burst slice"
            )

        element_type = c.global_emitter.element_type(value)
        if element_type is None:
            raise RuntimeError(
                f"Cannot infer scalar global element type for {expr.name}"
            )

        symbol_ref = ir.FlatSymbolRefAttr.get(value)
        return aps.GlobalLoad(element_type, symbol_ref).result

    def cast_type(
        self,
        value: ir.Value,
        target_type: ir.Type,
        use_sign_extend: bool = False,
        condition_context: bool = False,
    ) -> ir.Value:
        """Convert a value to a concrete target MLIR type when needed."""
        source_type = value.type

        if source_type == target_type:
            return value

        if isinstance(source_type, ir.IntegerType) and isinstance(
            target_type, ir.IntegerType
        ):
            if condition_context and target_type == ir.IntegerType.get_signless(1):
                zero = arith.ConstantOp(source_type, 0).result
                return arith.CmpIOp(arith.CmpIPredicate.ne, value, zero).result

            source_width = source_type.width
            target_width = target_type.width

            if source_width < target_width:
                if use_sign_extend:
                    return arith.ExtSIOp(target_type, value).result
                return arith.ExtUIOp(target_type, value).result

            if source_width > target_width:
                return arith.TruncIOp(target_type, value).result
        else:
            raise TypeError(
                f"Attempt to convert unknown type: {source_type} to {target_type}"
            )

        return value

    def promote_operands(
        self,
        left: ir.Value,
        right: ir.Value,
        expr: Optional[cadl_ast.BinaryExpr] = None,
    ) -> tuple[ir.Value, ir.Value]:
        """Promote integer operands to matching widths."""
        if isinstance(left.type, ir.IntegerType) and isinstance(
            right.type, ir.IntegerType
        ):
            left_width = left.type.width
            right_width = right.type.width

            if left_width < right_width:
                is_left_signed = expr and self.get_expr_signedness(expr.left)
                return (
                    self.cast_type(
                        left,
                        right.type,
                        use_sign_extend=bool(is_left_signed),
                    ),
                    right,
                )

            if right_width < left_width:
                is_right_signed = expr and self.get_expr_signedness(expr.right)
                return (
                    left,
                    self.cast_type(
                        right,
                        left.type,
                        use_sign_extend=bool(is_right_signed),
                    ),
                )

        return (left, right)

    def is_signed_type(
        self, ty: Optional[Union[cadl_ast.BasicType, cadl_ast.DataType, cadl_ast.CompoundType]]
    ) -> bool:
        """Check whether a CADL type should use signed operation semantics."""
        if isinstance(ty, cadl_ast.BasicType_ApFixed):
            return True
        if isinstance(ty, cadl_ast.DataType_Single):
            return self.is_signed_type(ty.basic_type)
        if isinstance(ty, cadl_ast.CompoundType_Basic):
            return self.is_signed_type(ty.data_type)
        return False

    def get_expr_signedness(self, expr: cadl_ast.Expr) -> bool:
        """Check whether an expression is known to be signed."""
        if isinstance(expr, cadl_ast.IdentExpr):
            cadl_type = self.converter.get_symbol_type(expr.name)
            if cadl_type:
                return self.is_signed_type(cadl_type)
        return False

    def convert_binary_op(
        self,
        op: cadl_ast.BinaryOp,
        left: ir.Value,
        right: ir.Value,
        expr: Optional[cadl_ast.BinaryExpr] = None,
    ) -> ir.Value:
        """Convert a CADL binary operator to MLIR."""
        is_signed = expr and self.get_expr_signedness(expr.left)

        if op == cadl_ast.BinaryOp.AND:
            i1_type = ir.IntegerType.get_signless(1)
            left = self.cast_type(left, i1_type, condition_context=True)
            right = self.cast_type(right, i1_type, condition_context=True)
            return arith.AndIOp(left, right).result
        if op == cadl_ast.BinaryOp.OR:
            i1_type = ir.IntegerType.get_signless(1)
            left = self.cast_type(left, i1_type, condition_context=True)
            right = self.cast_type(right, i1_type, condition_context=True)
            return arith.OrIOp(left, right).result

        if op == cadl_ast.BinaryOp.LSHIFT:
            right = self.cast_type(right, left.type)
            return arith.ShLIOp(left, right).result
        if op == cadl_ast.BinaryOp.RSHIFT:
            right = self.cast_type(right, left.type)
            if is_signed:
                return arith.ShRSIOp(left, right).result
            return arith.ShRUIOp(left, right).result

        left, right = self.promote_operands(left, right, expr)

        if op == cadl_ast.BinaryOp.ADD:
            return arith.AddIOp(left, right).result
        if op == cadl_ast.BinaryOp.SUB:
            return arith.SubIOp(left, right).result
        if op == cadl_ast.BinaryOp.MUL:
            return arith.MulIOp(left, right).result
        if op == cadl_ast.BinaryOp.DIV:
            if is_signed:
                return arith.DivSIOp(left, right).result
            return arith.DivUIOp(left, right).result
        if op == cadl_ast.BinaryOp.REM:
            if is_signed:
                return arith.RemSIOp(left, right).result
            return arith.RemUIOp(left, right).result

        if op == cadl_ast.BinaryOp.EQ:
            return arith.CmpIOp(arith.CmpIPredicate.eq, left, right).result
        if op == cadl_ast.BinaryOp.NE:
            return arith.CmpIOp(arith.CmpIPredicate.ne, left, right).result
        if op == cadl_ast.BinaryOp.LT:
            if is_signed:
                return arith.CmpIOp(arith.CmpIPredicate.slt, left, right).result
            return arith.CmpIOp(arith.CmpIPredicate.ult, left, right).result
        if op == cadl_ast.BinaryOp.LE:
            if is_signed:
                return arith.CmpIOp(arith.CmpIPredicate.sle, left, right).result
            return arith.CmpIOp(arith.CmpIPredicate.ule, left, right).result
        if op == cadl_ast.BinaryOp.GT:
            if is_signed:
                return arith.CmpIOp(arith.CmpIPredicate.sgt, left, right).result
            return arith.CmpIOp(arith.CmpIPredicate.ugt, left, right).result
        if op == cadl_ast.BinaryOp.GE:
            if is_signed:
                return arith.CmpIOp(arith.CmpIPredicate.sge, left, right).result
            return arith.CmpIOp(arith.CmpIPredicate.uge, left, right).result

        if op == cadl_ast.BinaryOp.BIT_AND:
            return arith.AndIOp(left, right).result
        if op == cadl_ast.BinaryOp.BIT_OR:
            return arith.OrIOp(left, right).result
        if op == cadl_ast.BinaryOp.BIT_XOR:
            return arith.XOrIOp(left, right).result

        raise NotImplementedError(f"Binary operation not yet supported: {op}")

    def convert_unary_op(self, op: cadl_ast.UnaryOp, operand: ir.Value) -> ir.Value:
        """Convert a CADL unary operator to MLIR."""
        if op == cadl_ast.UnaryOp.NEG:
            zero = arith.ConstantOp(operand.type, 0).result
            return arith.SubIOp(zero, operand).result
        if op == cadl_ast.UnaryOp.NOT:
            operand = self.cast_type(
                operand, ir.IntegerType.get_signless(1), condition_context=True
            )
            one_i1 = arith.ConstantOp(ir.IntegerType.get_signless(1), 1).result
            return arith.XOrIOp(operand, one_i1).result
        if op == cadl_ast.UnaryOp.BIT_NOT:
            all_ones = arith.ConstantOp(operand.type, -1).result
            return arith.XOrIOp(operand, all_ones).result

        raise NotImplementedError(f"Unary operation not yet supported: {op}")

    def convert_slice_expr(self, expr: cadl_ast.SliceExpr) -> ir.Value:
        """Convert bit slice expressions."""
        base_value = self.emit(expr.expr)

        if isinstance(expr.start, cadl_ast.LitExpr) and isinstance(expr.end, cadl_ast.LitExpr):
            start_bit = expr.start.literal.lit.value
            end_bit = expr.end.literal.lit.value

            if start_bit == end_bit:
                result_type = ir.IntegerType.get_signless(1)
                return comb.ExtractOp(result_type, base_value, start_bit).result

            width = abs(start_bit - end_bit) + 1
            result_type = ir.IntegerType.get_signless(width)
            low_bit = min(start_bit, end_bit)
            return comb.ExtractOp(result_type, base_value, low_bit).result

        raise NotImplementedError(
            "Dynamic bit slices are not supported in MLIR conversion"
        )

    def convert_if_expr(self, expr: cadl_ast.IfExpr) -> ir.Value:
        """Convert if expressions to arith.select."""
        condition = self.emit(expr.condition)
        then_value = self.emit(expr.then_branch)
        else_value = self.emit(expr.else_branch)

        condition = self.cast_type(
            condition, ir.IntegerType.get_signless(1), condition_context=True
        )
        return arith.SelectOp(condition, then_value, else_value).result

    def convert_select_expr(self, expr: cadl_ast.SelectExpr) -> ir.Value:
        """Convert select expressions to a priority chain of arith.select."""
        result = self.emit(expr.default)

        for cond_expr, val_expr in reversed(expr.arms):
            condition = self.cast_type(
                self.emit(cond_expr),
                ir.IntegerType.get_signless(1),
                condition_context=True,
            )
            value = self.emit(val_expr)
            result = arith.SelectOp(condition, value, result).result

        return result
