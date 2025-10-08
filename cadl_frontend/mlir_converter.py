"""
CADL AST to MLIR Converter

This module provides a framework for converting CADL Abstract Syntax Trees
to MLIR Intermediate Representation, leveraging CIRCT dialects for hardware
synthesis and optimization.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

# MLIR Python bindings
import circt
import circt.ir as ir
import circt.dialects.func as func
import circt.dialects.arith as arith
import circt.dialects.scf as scf
import circt.dialects.memref as memref

# CIRCT Python bindings
import circt.dialects.comb as comb
import circt.dialects.hw as hw

# CADL AST imports
from .ast import (
    Proc, Function, Flow, Static, Regfile,
    Stmt, Expr, BasicType, DataType, CompoundType,
    BasicType_ApFixed, BasicType_ApUFixed, BasicType_Float32, BasicType_Float64,
    BasicType_String, BasicType_USize,
    DataType_Single, DataType_Array, DataType_Instance,
    CompoundType_Basic, CompoundType_FnTy,
    LitExpr, IdentExpr, BinaryExpr, UnaryExpr, CallExpr,
    AssignStmt, ReturnStmt, ForStmt, DoWhileStmt, ExprStmt,
    BinaryOp, UnaryOp, FlowKind
)


@dataclass
class SymbolScope:
    """Symbol scope for managing variable bindings in different contexts"""
    symbols: Dict[str, ir.Value] = field(default_factory=dict)
    parent: Optional[SymbolScope] = None

    def get(self, name: str) -> Optional[ir.Value]:
        """Get symbol value, checking parent scopes if not found"""
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent:
            return self.parent.get(name)
        return None

    def set(self, name: str, value: ir.Value) -> None:
        """Set symbol value in current scope"""
        self.symbols[name] = value

    def has(self, name: str) -> bool:
        """Check if symbol exists in any scope"""
        return self.get(name) is not None


class CADLMLIRConverter:
    """
    Main converter class for transforming CADL AST to MLIR IR

    Maintains MLIR context, module, and symbol table for SSA form generation.
    Uses CIRCT dialects for hardware-oriented operations.
    """

    def __init__(self):
        # MLIR context and module
        self.context = ir.Context()
        self.module: Optional[ir.Module] = None

        # Load required dialects and configure context
        self._load_dialects()

        # Symbol table management for SSA form
        self.current_scope = SymbolScope()
        self.scope_stack: List[SymbolScope] = []

        # Builder for generating operations
        self.builder: Optional[ir.InsertionPoint] = None

    def _load_dialects(self) -> None:
        """Load required MLIR and CIRCT dialects"""
        # Register CIRCT dialects with the context
        with self.context:
            circt.register_dialects(self.context)

    def push_scope(self) -> None:
        """Push new symbol scope onto stack"""
        self.scope_stack.append(self.current_scope)
        self.current_scope = SymbolScope(parent=self.current_scope)

    def pop_scope(self) -> None:
        """Pop symbol scope from stack"""
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()

    def get_symbol(self, name: str) -> Optional[ir.Value]:
        """Get SSA value for symbol name"""
        return self.current_scope.get(name)

    def set_symbol(self, name: str, value: ir.Value) -> None:
        """Set SSA value for symbol name in current scope"""
        self.current_scope.set(name, value)

    def convert_cadl_type(self, cadl_type: Union[BasicType, DataType, CompoundType]) -> ir.Type:
        """
        Convert CADL type to MLIR type

        Maps CADL type system to appropriate MLIR types, with fallbacks
        for unsupported fixed-width types.
        """
        if isinstance(cadl_type, BasicType_ApFixed):
            # TODO: Investigate !hw.int<width> for fixed-width signed integers
            # For now, use standard MLIR integer types as fallback
            if cadl_type.width <= 32:
                return ir.IntegerType.get_signed(32)
            else:
                return ir.IntegerType.get_signed(64)

        elif isinstance(cadl_type, BasicType_ApUFixed):
            # TODO: Investigate !hw.int<width> for fixed-width unsigned integers
            # For now, use standard MLIR integer types as fallback
            if cadl_type.width <= 32:
                return ir.IntegerType.get_signless(32)
            else:
                return ir.IntegerType.get_signless(64)

        elif isinstance(cadl_type, BasicType_Float32):
            return ir.F32Type.get()

        elif isinstance(cadl_type, BasicType_Float64):
            return ir.F64Type.get()

        elif isinstance(cadl_type, BasicType_String):
            # No direct string type in MLIR, use memref<i8> for now
            return memref.MemRefType.get([ir.ShapedType.get_dynamic_size()], ir.IntegerType.get_signless(8))

        elif isinstance(cadl_type, BasicType_USize):
            return ir.IndexType.get()

        elif isinstance(cadl_type, DataType_Single):
            return self.convert_cadl_type(cadl_type.basic_type)

        elif isinstance(cadl_type, DataType_Array):
            element_type = self.convert_cadl_type(cadl_type.element_type)
            return memref.MemRefType.get(cadl_type.dimensions, element_type)

        elif isinstance(cadl_type, CompoundType_Basic):
            return self.convert_cadl_type(cadl_type.data_type)

        elif isinstance(cadl_type, CompoundType_FnTy):
            # Function types need special handling
            arg_types = [self.convert_cadl_type(arg) for arg in cadl_type.args]
            ret_types = [self.convert_cadl_type(ret) for ret in cadl_type.ret]
            return ir.FunctionType.get(arg_types, ret_types)

        else:
            raise NotImplementedError(f"Unsupported CADL type: {type(cadl_type)}")

    def convert_proc(self, proc: Proc) -> ir.Module:
        """
        Convert CADL Proc to MLIR Module

        Creates top-level MLIR module containing all functions, flows,
        and global variables from the processor definition.
        """
        with self.context, ir.Location.unknown():
            # Create module
            self.module = ir.Module.create()

            with ir.InsertionPoint(self.module.body):
                self.builder = ir.InsertionPoint.current

                # Convert static variables to global declarations
                for static in proc.statics.values():
                    self._convert_static(static)

                # Convert functions
                for function in proc.functions.values():
                    self._convert_function(function)

                # Convert flows (as functions for now)
                for flow in proc.flows.values():
                    self._convert_flow(flow)

                # TODO: Convert register files to appropriate MLIR constructs

        return self.module

    def _convert_static(self, static: Static) -> None:
        """Convert static variable to MLIR global"""
        # For now, create as function-level variable
        # TODO: Implement proper global variable handling
        pass

    def _convert_function(self, function: Function) -> ir.Operation:
        """Convert CADL Function to MLIR func.func operation"""
        # Convert argument types
        arg_types = [self.convert_cadl_type(arg.ty) for arg in function.args]

        # Convert return types
        ret_types = [self.convert_cadl_type(ret) for ret in function.ret]

        # Create function type
        func_type = ir.FunctionType.get(arg_types, ret_types)

        # Create function operation
        func_op = func.FuncOp(function.name, func_type)

        # Add entry block
        entry_block = func_op.add_entry_block()

        with ir.InsertionPoint(entry_block):
            # Push new scope for function
            self.push_scope()

            # Bind function arguments to symbols
            for i, arg in enumerate(function.args):
                arg_value = entry_block.arguments[i]
                self.set_symbol(arg.id, arg_value)

            # Convert function body
            if function.body:
                self._convert_stmt_list(function.body)

            # Check if last operation is already a return
            # If not, add a default return
            block_ops = list(entry_block.operations)
            if not block_ops or not isinstance(block_ops[-1], func.ReturnOp):
                func.ReturnOp([])

            # Pop function scope
            self.pop_scope()

        return func_op

    def _convert_flow(self, flow: Flow) -> ir.Operation:
        """Convert CADL Flow to MLIR function (for now)"""
        # For now, treat flows as functions
        # TODO: Implement hardware-specific flow conversion

        # Convert input types
        arg_types = [self.convert_cadl_type(dtype) for _, dtype in flow.inputs]

        # Flows typically return void or single value
        ret_types = []  # TODO: Determine from flow analysis

        # Create function type
        func_type = ir.FunctionType.get(arg_types, ret_types)

        # Create function operation with flow name
        func_op = func.FuncOp(f"flow_{flow.name}", func_type)

        # Add entry block
        entry_block = func_op.add_entry_block()

        with ir.InsertionPoint(entry_block):
            # Push new scope for flow
            self.push_scope()

            # Bind flow inputs to symbols
            for i, (name, _) in enumerate(flow.inputs):
                arg_value = entry_block.arguments[i]
                self.set_symbol(name, arg_value)

            # Convert flow body
            if flow.body:
                self._convert_stmt_list(flow.body)

            # Check if last operation is already a return
            # If not, add a default return
            block_ops = list(entry_block.operations)
            if not block_ops or not isinstance(block_ops[-1], func.ReturnOp):
                func.ReturnOp([])

            # Pop flow scope
            self.pop_scope()

        return func_op

    def _convert_stmt_list(self, stmts: List[Stmt]) -> None:
        """Convert list of statements maintaining SSA form"""
        for stmt in stmts:
            self._convert_stmt(stmt)

    def _convert_stmt(self, stmt: Stmt) -> None:
        """Convert single statement to MLIR operations"""
        if isinstance(stmt, ExprStmt):
            self._convert_expr(stmt.expr)

        elif isinstance(stmt, AssignStmt):
            # Convert RHS expression
            rhs_value = self._convert_expr(stmt.rhs)

            # Handle LHS (should be identifier for now)
            if isinstance(stmt.lhs, IdentExpr):
                self.set_symbol(stmt.lhs.name, rhs_value)
            else:
                raise NotImplementedError(f"Complex LHS assignment not yet supported: {type(stmt.lhs)}")

        elif isinstance(stmt, ReturnStmt):
            # Convert return expressions
            ret_values = [self._convert_expr(expr) for expr in stmt.exprs]
            func.ReturnOp(ret_values)

        elif isinstance(stmt, DoWhileStmt):
            # Use scf.while as advised - perfect semantic match
            self._convert_do_while(stmt)

        elif isinstance(stmt, ForStmt):
            # Convert for loops using scf.for
            self._convert_for_loop(stmt)

        else:
            raise NotImplementedError(f"Statement type not yet supported: {type(stmt)}")

    def _convert_expr(self, expr: Expr) -> ir.Value:
        """Convert expression to MLIR SSA value"""
        if isinstance(expr, LitExpr):
            # Convert literal to constant operation
            literal = expr.literal
            mlir_type = self.convert_cadl_type(literal.ty)

            if hasattr(literal.lit, 'value'):
                value = literal.lit.value
                return arith.ConstantOp(mlir_type, value).result
            else:
                raise NotImplementedError(f"Literal type not supported: {type(literal.lit)}")

        elif isinstance(expr, IdentExpr):
            # Look up symbol in symbol table
            value = self.get_symbol(expr.name)
            if value is None:
                raise ValueError(f"Undefined symbol: {expr.name}")
            return value

        elif isinstance(expr, BinaryExpr):
            # Convert binary operations using appropriate dialects
            left = self._convert_expr(expr.left)
            right = self._convert_expr(expr.right)

            return self._convert_binary_op(expr.op, left, right)

        elif isinstance(expr, UnaryExpr):
            # Convert unary operations
            operand = self._convert_expr(expr.operand)
            return self._convert_unary_op(expr.op, operand)

        elif isinstance(expr, CallExpr):
            # Convert function calls
            args = [self._convert_expr(arg) for arg in expr.args]
            # For now, assume function exists in symbol table or is built-in
            return self._convert_call(expr.name, args)

        else:
            raise NotImplementedError(f"Expression type not yet supported: {type(expr)}")

    def _convert_binary_op(self, op: BinaryOp, left: ir.Value, right: ir.Value) -> ir.Value:
        """Convert binary operation to appropriate MLIR operation"""
        # Arithmetic operations (prefer arith dialect for arithmetic)
        if op == BinaryOp.ADD:
            return arith.AddIOp(left, right).result
        elif op == BinaryOp.SUB:
            return arith.SubIOp(left, right).result
        elif op == BinaryOp.MUL:
            return arith.MulIOp(left, right).result
        elif op == BinaryOp.DIV:
            # Use signed or unsigned based on type
            return arith.DivSIOp(left, right).result  # TODO: Handle signedness
        elif op == BinaryOp.REM:
            return arith.RemSIOp(left, right).result  # TODO: Handle signedness

        # Comparison operations
        elif op == BinaryOp.EQ:
            return arith.CmpIOp(arith.CmpIPredicate.eq, left, right).result
        elif op == BinaryOp.NE:
            return arith.CmpIOp(arith.CmpIPredicate.ne, left, right).result
        elif op == BinaryOp.LT:
            return arith.CmpIOp(arith.CmpIPredicate.slt, left, right).result  # TODO: Handle signedness
        elif op == BinaryOp.LE:
            return arith.CmpIOp(arith.CmpIPredicate.sle, left, right).result
        elif op == BinaryOp.GT:
            return arith.CmpIOp(arith.CmpIPredicate.sgt, left, right).result
        elif op == BinaryOp.GE:
            return arith.CmpIOp(arith.CmpIPredicate.sge, left, right).result

        # Logical operations (use comb for hardware-style operations)
        elif op == BinaryOp.AND:
            # Logical AND - need to handle boolean context
            return arith.AndIOp(left, right).result
        elif op == BinaryOp.OR:
            # Logical OR - need to handle boolean context
            return arith.OrIOp(left, right).result

        # Bitwise operations (use comb dialect for hardware operations)
        elif op == BinaryOp.BIT_AND:
            return comb.AndOp([left, right]).result
        elif op == BinaryOp.BIT_OR:
            return comb.OrOp([left, right]).result
        elif op == BinaryOp.BIT_XOR:
            return comb.XorOp([left, right]).result

        # Shift operations (prefer comb for hardware)
        elif op == BinaryOp.LSHIFT:
            return comb.ShlOp(left, right).result  # Shift ops might use different syntax
        elif op == BinaryOp.RSHIFT:
            return comb.ShrUOp(left, right).result  # TODO: Handle arithmetic vs logical shift

        else:
            raise NotImplementedError(f"Binary operation not yet supported: {op}")

    def _convert_unary_op(self, op: UnaryOp, operand: ir.Value) -> ir.Value:
        """Convert unary operation to appropriate MLIR operation"""
        if op == UnaryOp.NEG:
            # Arithmetic negation
            zero = arith.ConstantOp(operand.type, 0).result
            return arith.SubIOp(zero, operand).result
        elif op == UnaryOp.NOT:
            # Logical NOT
            one = arith.ConstantOp(operand.type, 1).result
            return arith.XOrIOp(operand, one).result
        elif op == UnaryOp.BIT_NOT:
            # Bitwise NOT (invert all bits)
            all_ones = arith.ConstantOp(operand.type, -1).result
            return comb.XorOp([operand, all_ones]).result

        # Type cast operations
        elif op == UnaryOp.SIGNED_CAST:
            # Cast to signed interpretation
            # For now, just return operand (type system handles interpretation)
            return operand
        elif op == UnaryOp.UNSIGNED_CAST:
            # Cast to unsigned interpretation
            return operand
        elif op == UnaryOp.F32_CAST:
            # Cast to f32
            if operand.type != ir.F32Type.get():
                return arith.SIToFPOp(ir.F32Type.get(), operand).result
            return operand
        elif op == UnaryOp.F64_CAST:
            # Cast to f64
            if operand.type != ir.F64Type.get():
                return arith.SIToFPOp(ir.F64Type.get(), operand).result
            return operand

        else:
            raise NotImplementedError(f"Unary operation not yet supported: {op}")

    def _convert_call(self, func_name: str, args: List[ir.Value]) -> ir.Value:
        """Convert function call to MLIR call operation"""
        # For now, assume function exists and has compatible signature
        # TODO: Implement proper function lookup and type checking
        call_op = func.CallOp([], func_name, args)

        # Return first result if any, otherwise None
        if call_op.results:
            return call_op.results[0]
        else:
            # For procedures/void functions, create a dummy value
            # This is a temporary workaround
            dummy_type = ir.IntegerType.get_signless(32)
            return arith.ConstantOp(dummy_type, 0).result

    def _convert_do_while(self, stmt: DoWhileStmt) -> None:
        """
        Convert do-while loop to scf.while operation

        CADL do-while semantics: body executes at least once, condition checked after.
        The condition uses variables defined in the body (like i_).

        We use scf.while with a first_iteration flag to ensure at least one execution.
        """
        # Handle with bindings (loop variables with init/next values)
        init_values = []
        loop_var_types = []

        for binding in stmt.bindings:
            # Convert initial value if provided
            if binding.init:
                init_val = self._convert_expr(binding.init)
                init_values.append(init_val)
                loop_var_types.append(init_val.type)
            else:
                # Default initialization for the type
                var_type = self.convert_cadl_type(binding.ty)
                zero_val = arith.ConstantOp(var_type, 0).result
                init_values.append(zero_val)
                loop_var_types.append(var_type)

        # Add a boolean flag to track first iteration
        bool_type = ir.IntegerType.get_signless(1)
        true_val = arith.ConstantOp(bool_type, 1).result
        init_values.append(true_val)
        loop_var_types.append(bool_type)

        # Create scf.while operation
        while_op = scf.WhileOp(loop_var_types, init_values)

        # Before region: check if we should continue
        before_block = while_op.before.blocks.append(*loop_var_types)
        with ir.InsertionPoint(before_block):
            # Get the first_iteration flag (last argument)
            first_iter = before_block.arguments[-1]

            # For do-while, we always continue on first iteration
            # Otherwise, we need to check the condition based on the current values

            # If it's the first iteration, always continue
            # Otherwise, the condition has already been evaluated with the new values
            # So we just pass through the values
            scf.ConditionOp(first_iter, before_block.arguments)

        # After region: execute loop body and check condition
        after_block = while_op.after.blocks.append(*loop_var_types)
        with ir.InsertionPoint(after_block):
            # Push scope for loop body
            self.push_scope()

            # Update loop variables with block arguments (excluding the flag)
            for i, binding in enumerate(stmt.bindings):
                self.set_symbol(binding.id, after_block.arguments[i])

            # Execute loop body - this defines i_, sum_, n_ etc.
            self._convert_stmt_list(stmt.body)

            # Now evaluate the condition using variables defined in the body
            condition_val = self._convert_expr(stmt.condition)

            # Compute next values for loop variables from bindings
            next_values = []
            for binding in stmt.bindings:
                if binding.next:
                    # The next expression references variables defined in the body
                    if isinstance(binding.next, IdentExpr):
                        next_val = self.get_symbol(binding.next.name)
                    else:
                        next_val = self._convert_expr(binding.next)
                    next_values.append(next_val)
                else:
                    # Keep current value if no next expression
                    current_val = self.get_symbol(binding.id)
                    next_values.append(current_val)

            # Pass the condition as the new first_iteration flag
            # This way, the before region will check it on the next iteration
            next_values.append(condition_val)

            # Yield the next values
            scf.YieldOp(next_values)
            self.pop_scope()

        # Make loop variables available in the parent scope (exclude the flag)
        for i, binding in enumerate(stmt.bindings):
            if while_op.results and i < len(while_op.results) - 1:
                self.set_symbol(binding.id, while_op.results[i])

    def _convert_for_loop(self, stmt: ForStmt) -> None:
        """Convert for loop to appropriate MLIR constructs"""
        # Push new scope for loop
        self.push_scope()

        # Execute initialization
        self._convert_stmt(stmt.init)

        # For now, convert to scf.while (more general than scf.for)
        # TODO: Detect when we can use scf.for for better optimization

        # Create condition check function
        def create_while_body():
            # Check condition
            condition_val = self._convert_expr(stmt.condition)

            # Create while operation
            # For simplicity, use empty arguments for now
            while_op = scf.WhileOp([], [])

            # Before region: condition check
            before_block = while_op.before.blocks.append()
            with ir.InsertionPoint(before_block):
                scf.ConditionOp(condition_val, [])

            # After region: body + update
            after_block = while_op.after.blocks.append()
            with ir.InsertionPoint(after_block):
                # Execute loop body
                self._convert_stmt_list(stmt.body)

                # Execute update statement
                self._convert_stmt(stmt.update)

                # Yield (no arguments for this simple case)
                scf.YieldOp([])

        create_while_body()

        # Pop loop scope
        self.pop_scope()


def convert_cadl_to_mlir(proc: Proc) -> ir.Module:
    """
    Main entry point for converting CADL Proc to MLIR Module

    Args:
        proc: CADL processor AST to convert

    Returns:
        MLIR module containing the converted representation
    """
    converter = CADLMLIRConverter()
    return converter.convert_proc(proc)