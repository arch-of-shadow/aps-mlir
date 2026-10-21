from __future__ import annotations

from typing import Optional, Union

import circt.ir as ir
import circt.dialects.arith as arith
import circt.dialects.aps as aps
import circt.dialects.comb as comb

from .. import cadl_ast
from .literals import exact_float_attr
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
                if isinstance(literal.lit, cadl_ast.LiteralInner_Float) and literal.lit.text:
                    value = exact_float_attr(mlir_type, literal.lit.text)
                return arith.ConstantOp(mlir_type, value).result
            raise NotImplementedError(
                f"Literal type not supported: {type(literal.lit)}"
            )

        if isinstance(expr, cadl_ast.CallExpr):
            return self.emit_float_call(expr)

        if isinstance(expr, cadl_ast.IdentExpr):
            return self.emit_ident(expr)

        if isinstance(expr, cadl_ast.BinaryExpr):
            left = self.emit(expr.left)
            right = self.emit(expr.right)
            return self.convert_binary_op(expr.op, left, right, expr)

        if isinstance(expr, cadl_ast.BitcastExpr):
            value = self.emit(expr.operand)
            target = cast_cadl_type_to_mlir(cadl_ast.parse_basic_type_from_string(expr.target_type))
            def width(ty):
                if isinstance(ty, ir.IntegerType):
                    return ty.width
                if isinstance(ty, ir.F32Type):
                    return 32
                if isinstance(ty, ir.BF16Type):
                    return 16
                return None
            if width(value.type) is None or width(value.type) != width(target):
                raise TypeError("bitcast requires same-width integer/bf16/f32 types")
            if value.type == target:
                return value
            return arith.BitcastOp(target, value).result

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

    def emit_float_call(self, expr):
        import re
        match = re.fullmatch(r"(bf16|fp32)_(add|sub|mul|div|sqrt|fma|fmsub|fnmadd|fnmsub|cmp|from_i32|from_u32|to_i32|to_u32|widen|narrow|neg|abs|copysign|min|max|classify)(_flags)?", expr.name)
        if not match:
            raise NotImplementedError(f"Unknown intrinsic: {expr.name}")
        precision, name, flags = match.groups()
        variant = name if name in {'fmsub','fnmadd','fnmsub'} else None
        if variant:
            name = 'fma'
        if name == 'widen' and precision != 'bf16' or name == 'narrow' and precision != 'fp32':
            raise TypeError('Use bf16_widen / fp32_narrow for cross-format conversion')
        names = ["add","sub","mul","div","sqrt","fma","cmp","from_i32","from_u32","to_i32","to_u32","widen","narrow","neg","abs","copysign","min","max","classify"]
        kind = names.index(name)
        width = 16 if precision == "bf16" else 32
        arity = 3 if name == "fma" else 2 if name in {"add","sub","mul","div","cmp","copysign","min","max"} else 1
        args = list(expr.args)
        predicate = 0
        def integer_literal(value):
            if not isinstance(value, cadl_ast.LitExpr) or not isinstance(value.literal.lit, cadl_ast.LiteralInner_Fixed):
                raise TypeError("rounding mode/predicate must be an integer literal")
            return value.literal.lit.value
        if name == "cmp":
            if len(args) != 3:
                raise TypeError("cmp requires a,b,predicate (MLIR predicates 0..15)")
            predicate = integer_literal(args.pop())
        rm = 1 if name in {"to_i32","to_u32"} else 0
        if len(args) == arity + 1:
            rm = integer_literal(args.pop())
        if len(args) != arity or rm not in range(5) or predicate not in range(16):
            raise TypeError("invalid floating intrinsic arity/rounding/predicate")
        operands = [self.emit(a) for a in args]
        fp_type = ir.BF16Type.get() if width == 16 else ir.F32Type.get()
        expected = ir.F32Type.get() if name == "narrow" else ir.BF16Type.get() if name == "widen" else fp_type
        if name in {"from_i32","from_u32"}:
            if not isinstance(operands[0].type, ir.IntegerType) or operands[0].type.width != 32:
                raise TypeError("from_i32/u32 requires a 32-bit integer")
        elif any(v.type != expected for v in operands):
            raise TypeError(f"{expr.name} requires {expected} operands")
        output = ir.F32Type.get() if name == "widen" else ir.BF16Type.get() if name == "narrow" else ir.IntegerType.get_signless(32) if name in {"to_i32","to_u32","classify"} else ir.IntegerType.get_signless(1) if name == "cmp" else fp_type
        attrs = {k: ir.IntegerAttr.get(ir.IntegerType.get_signless(32), v) for k,v in {"kind":kind,"width":width,"rm":rm,"predicate":predicate}.items()}
        if variant:
            for index in ([2] if variant == 'fmsub' else [0,2] if variant == 'fnmadd' else [0]):
                neg_attrs = dict(attrs)
                neg_attrs['kind'] = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 13)
                neg = ir.Operation.create('aps.fp', results=[fp_type,ir.IntegerType.get_signless(5)], operands=[operands[index]], attributes=neg_attrs)
                operands[index] = neg.results[0]
        op = ir.Operation.create("aps.fp", results=[output, ir.IntegerType.get_signless(5)], operands=operands, attributes=attrs)
        return op.results[1 if flags else 0]

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

        float_types = (ir.BF16Type, ir.F32Type, ir.F64Type)
        if isinstance(left.type, float_types) or isinstance(right.type, float_types):
            if left.type != right.type or not isinstance(left.type, (ir.BF16Type, ir.F32Type)):
                raise TypeError("floating arithmetic requires matching bf16/f32 operands")
            operations = {
                cadl_ast.BinaryOp.ADD: arith.AddFOp,
                cadl_ast.BinaryOp.SUB: arith.SubFOp,
                cadl_ast.BinaryOp.MUL: arith.MulFOp,
                cadl_ast.BinaryOp.DIV: arith.DivFOp,
            }
            operation = operations.get(op)
            predicates = {cadl_ast.BinaryOp.EQ: 1, cadl_ast.BinaryOp.NE: 14,
                          cadl_ast.BinaryOp.LT: 4, cadl_ast.BinaryOp.LE: 5,
                          cadl_ast.BinaryOp.GT: 2, cadl_ast.BinaryOp.GE: 3}
            if op in predicates:
                return arith.CmpFOp(predicates[op], left, right).result
            if operation is None:
                raise NotImplementedError(f"Floating operation not supported: {op}")
            return operation(left, right).result

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
        if isinstance(operand.type, (ir.BF16Type, ir.F32Type, ir.F64Type)):
            if op == cadl_ast.UnaryOp.NEG:
                return arith.NegFOp(operand).result
            raise NotImplementedError(f"Floating unary operation not supported: {op}")
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
