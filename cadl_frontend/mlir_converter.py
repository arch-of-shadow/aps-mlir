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
import circt.dialects.aps as aps

# CADL AST imports
from .ast import (
    Proc, Flow, Static, Regfile,
    Stmt, Expr, BasicType, DataType, CompoundType,
    BasicType_ApFixed, BasicType_ApUFixed, BasicType_Float32, BasicType_Float64,
    BasicType_String, BasicType_USize,
    DataType_Single, DataType_Array, DataType_Instance,
    CompoundType_Basic,
    LitExpr, IdentExpr, BinaryExpr, UnaryExpr, CallExpr, IndexExpr, SliceExpr, RangeSliceExpr, IfExpr, AggregateExpr, StringLitExpr,
    AssignStmt, ReturnStmt, ForStmt, DoWhileStmt, ExprStmt,
    BinaryOp, UnaryOp, FlowKind
)


@dataclass
class SymbolScope:
    """Symbol scope for managing variable bindings in different contexts"""
    symbols: Dict[str, Union[ir.Value, str]] = field(default_factory=dict)
    parent: Optional[SymbolScope] = None

    def get(self, name: str) -> Optional[Union[ir.Value, str]]:
        """Get symbol value, checking parent scopes if not found"""
        if name in self.symbols:
            return self.symbols[name]
        elif self.parent:
            return self.parent.get(name)
        return None

    def set(self, name: str, value: Union[ir.Value, str]) -> None:
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

        # CPU memory instance management
        self.cpu_memory_instance: Optional[ir.Value] = None
        self.functions_using_memory: set = set()  # Track which functions use _mem
        self.global_memory_declared: bool = False  # Track if global memory is declared

        # Cache for global references within current function
        self.current_global_refs: Dict[str, ir.Value] = {}

    def _load_dialects(self) -> None:
        """Load required MLIR and CIRCT dialects"""
        # Register CIRCT dialects with the context
        with self.context:
            circt.register_dialects(self.context)

    def push_scope(self) -> None:
        """Push new symbol scope onto stack"""
        self.scope_stack.append(self.current_scope)
        self.current_scope = SymbolScope(parent=self.current_scope)
        # Reset global reference cache for new function scope
        self.current_global_refs = {}

    def pop_scope(self) -> None:
        """Pop symbol scope from stack"""
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()
            # Clear global reference cache when exiting scope to avoid referencing
            # SSA values from inner scopes that are no longer valid
            self.current_global_refs = {}

    def get_symbol(self, name: str) -> Optional[Union[ir.Value, str]]:
        """Get SSA value for symbol name"""
        return self.current_scope.get(name)

    def set_symbol(self, name: str, value: Union[ir.Value, str]) -> None:
        """Set SSA value for symbol name in current scope"""
        self.current_scope.set(name, value)

    def get_cpu_memory_instance(self) -> ir.Value:
        """Get reference to the global CPU memory instance"""
        if not self.global_memory_declared:
            raise RuntimeError("Global CPU memory should be declared at module level before use")

        # Get reference to the global memory using cached helper
        element_type = ir.IntegerType.get_signless(32)
        memory_size = 1024  # Must match the size used in _declare_global_memory
        memory_type = memref.MemRefType.get([memory_size], element_type)
        return self._get_global_reference("_cpu_memory", memory_type)

    def set_cpu_memory_instance(self, memory_instance: ir.Value) -> None:
        """Set the CPU memory instance for current function"""
        self.cpu_memory_instance = memory_instance

    def _get_global_reference(self, global_name: str, memory_type: ir.Type) -> ir.Value:
        """Get reference to global variable, with caching to avoid duplicates"""
        if global_name not in self.current_global_refs:
            # Create new global reference and cache it
            global_ref = memref.GetGlobalOp(memory_type, global_name)
            self.current_global_refs[global_name] = global_ref.result
        return self.current_global_refs[global_name]

    def _declare_global_memory(self) -> None:
        """Declare global CPU memory at module level using memref.global"""
        if not self.global_memory_declared:
            # Create global memory using memref.global with static size
            element_type = ir.IntegerType.get_signless(32)
            # Use static size (e.g., 1024 elements) instead of dynamic size
            memory_size = 1024
            memory_type = memref.MemRefType.get([memory_size], element_type)

            # Create a global memref variable
            global_name = "_cpu_memory"
            global_op = memref.GlobalOp(global_name, memory_type)

            # Store the global reference for symbol resolution
            self.set_symbol("_cpu_memory", global_name)
            self.global_memory_declared = True

    def _function_uses_memory(self, function) -> bool:
        """Check if a function uses _mem operations"""
        # We'll need to analyze the function body to see if it contains _mem operations
        # For now, let's implement a simple visitor pattern
        if not function.body:
            return False

        return self._stmt_list_uses_memory(function.body)

    def _flow_uses_memory(self, flow) -> bool:
        """Check if a flow uses _mem operations"""
        if not flow.body:
            return False

        return self._stmt_list_uses_memory(flow.body)

    def _stmt_list_uses_memory(self, stmts) -> bool:
        """Check if a list of statements uses _mem operations"""
        for stmt in stmts:
            if self._stmt_uses_memory(stmt):
                return True
        return False

    def _stmt_uses_memory(self, stmt) -> bool:
        """Check if a statement uses _mem operations"""
        if isinstance(stmt, AssignStmt):
            # Check LHS for _mem assignment
            if isinstance(stmt.lhs, IndexExpr) and isinstance(stmt.lhs.expr, IdentExpr):
                if stmt.lhs.expr.name == "_mem":
                    return True
            # Check RHS for _mem read
            if self._expr_uses_memory(stmt.rhs):
                return True
        elif isinstance(stmt, ExprStmt):
            return self._expr_uses_memory(stmt.expr)
        elif isinstance(stmt, ReturnStmt):
            return any(self._expr_uses_memory(expr) for expr in stmt.exprs)
        # Add other statement types as needed
        return False

    def _expr_uses_memory(self, expr) -> bool:
        """Check if an expression uses _mem operations"""
        if isinstance(expr, IndexExpr) and isinstance(expr.expr, IdentExpr):
            if expr.expr.name == "_mem":
                return True
        elif isinstance(expr, BinaryExpr):
            return self._expr_uses_memory(expr.left) or self._expr_uses_memory(expr.right)
        elif isinstance(expr, UnaryExpr):
            return self._expr_uses_memory(expr.operand)
        elif isinstance(expr, CallExpr):
            return any(self._expr_uses_memory(arg) for arg in expr.args)
        # Add other expression types as needed
        return False

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
        # Store proc reference for later access
        self.proc = proc

        with self.context, ir.Location.unknown():
            # Create module
            self.module = ir.Module.create()

            with ir.InsertionPoint(self.module.body):
                self.builder = ir.InsertionPoint.current

                # Check if any flows use memory
                any_uses_memory = any(self._flow_uses_memory(flow) for flow in proc.flows.values())

                # Declare global memory if needed
                if any_uses_memory:
                    self._declare_global_memory()

                # Convert static variables to global declarations
                for static in proc.statics.values():
                    self._convert_static(static)

                # Convert flows (as functions for now)
                for flow in proc.flows.values():
                    self._convert_flow(flow)

                # TODO: Convert register files to appropriate MLIR constructs

        return self.module

    def _convert_static(self, static: Static) -> None:
        """Convert static variable to MLIR global"""
        mlir_type = self.convert_cadl_type(static.ty)

        # Create a global variable using memref.global
        global_name = static.id

        # Get initial value if provided
        initial_value = None
        initial_values_list = None
        if static.expr:
            if isinstance(static.expr, LitExpr):
                # Single literal value
                initial_value = static.expr.literal.lit.value
            elif isinstance(static.expr, AggregateExpr):
                # Array initialization like {1474560, 870484, ...}
                initial_values_list = []
                for elem_expr in static.expr.elements:
                    if isinstance(elem_expr, LitExpr):
                        initial_values_list.append(elem_expr.literal.lit.value)
                    else:
                        # For non-literal elements, we'll skip initialization for now
                        initial_values_list = None
                        break

        # Determine the correct memref type based on the CADL type
        if isinstance(mlir_type, ir.MemRefType):
            # mlir_type is already a memref (for arrays), use it directly
            memref_type = mlir_type
        else:
            # mlir_type is a scalar element type, wrap it in memref<>
            memref_type = ir.MemRefType.get([], mlir_type)

        # Create a tensor type for the initial value if provided
        if initial_value is not None:
            # Single scalar initialization
            if isinstance(mlir_type, ir.MemRefType):
                # This shouldn't happen for single values, but handle it
                global_op = memref.GlobalOp(global_name, memref_type)
            else:
                # Create attribute from the integer value for scalars
                element_attr = ir.IntegerAttr.get(mlir_type, initial_value)
                attr = ir.DenseElementsAttr.get_splat(ir.RankedTensorType.get([], mlir_type), element_attr)
                global_op = memref.GlobalOp(global_name, memref_type, initial_value=attr, constant=True)
        elif initial_values_list is not None:
            # Array initialization with list of values
            if isinstance(mlir_type, ir.MemRefType):
                # Create dense elements attribute for array initialization
                element_type = mlir_type.element_type
                shape = mlir_type.shape

                # Create integer attributes for each value
                element_attrs = []
                for val in initial_values_list:
                    element_attrs.append(ir.IntegerAttr.get(element_type, val))

                # Create tensor type and dense elements attribute
                tensor_type = ir.RankedTensorType.get(shape, element_type)
                dense_attr = ir.DenseElementsAttr.get(element_attrs, tensor_type)

                # Create global with initialization
                global_op = memref.GlobalOp(global_name, memref_type, initial_value=dense_attr, constant=True)
            else:
                # This shouldn't happen for arrays, fallback to uninitialized
                global_op = memref.GlobalOp(global_name, memref_type)
        else:
            # Create uninitialized global
            global_op = memref.GlobalOp(global_name, memref_type)

        # Add custom attributes if present
        if static.attrs:
            for attr_name, attr_value in static.attrs.items():
                # Convert CADL attribute value to MLIR attribute
                mlir_attr = self._convert_attribute_value(attr_value)
                if mlir_attr is not None:
                    global_op.attributes[attr_name] = mlir_attr

        # Store the global reference for symbol resolution
        self.set_symbol(static.id, global_name)

    def _convert_attribute_value(self, expr: Optional[Expr]) -> Optional[ir.Attribute]:
        """
        Convert CADL expression to MLIR attribute for use in operation attributes

        Handles:
        - StringLitExpr -> StringAttr
        - LitExpr with integer -> IntegerAttr
        - None -> UnitAttr (for presence-only attributes)
        """
        if expr is None:
            # Attribute without value, use UnitAttr
            return ir.UnitAttr.get()

        if isinstance(expr, StringLitExpr):
            # String attribute
            return ir.StringAttr.get(expr.value)

        if isinstance(expr, LitExpr):
            # Numeric attribute
            literal = expr.literal
            if hasattr(literal.lit, 'value'):
                value = literal.lit.value
                if isinstance(value, int):
                    # Integer attribute
                    mlir_type = self.convert_cadl_type(literal.ty)
                    return ir.IntegerAttr.get(mlir_type, value)
                elif isinstance(value, float):
                    # Float attribute
                    mlir_type = self.convert_cadl_type(literal.ty)
                    return ir.FloatAttr.get(mlir_type, value)

        if isinstance(expr, IdentExpr):
            # Identifier - treat as string symbol
            return ir.StringAttr.get(expr.name)

        # For other expression types, try to convert to string representation
        return ir.StringAttr.get(str(expr))

    def _convert_function(self, function: Function) -> ir.Operation:
        """Convert CADL Function to MLIR func.func operation"""
        # Convert argument types - handle case where args might be empty or Token
        if hasattr(function.args, '__iter__') and not isinstance(function.args, str):
            # function.args is a proper list
            arg_types = [self.convert_cadl_type(arg.ty) for arg in function.args]
            function_args = function.args
        else:
            # function.args is probably a Token (empty args case)
            arg_types = []
            function_args = []

        # Check if function uses _mem (but don't add it as argument anymore)
        uses_memory = self._function_uses_memory(function)

        # Convert return types - handle case where ret might be empty or Token
        if hasattr(function.ret, '__iter__') and not isinstance(function.ret, str):
            # function.ret is a proper list
            ret_types = [self.convert_cadl_type(ret) for ret in function.ret]
        else:
            # function.ret is probably a Token (empty return case)
            ret_types = []

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
            for i, arg in enumerate(function_args):
                arg_value = entry_block.arguments[i]
                self.set_symbol(arg.id, arg_value)

            # If function uses memory, it will be declared via aps.memdeclare when first accessed
            # No need to set up CPU memory instance here anymore

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

        # Check if flow uses _mem (but don't add it as argument anymore)
        uses_memory = self._flow_uses_memory(flow)

        # Flows typically return void or single value
        ret_types = []  # TODO: Determine from flow analysis

        # Create function type
        func_type = ir.FunctionType.get(arg_types, ret_types)

        # Create function operation with flow name
        func_op = func.FuncOp(f"flow_{flow.name}", func_type)

        # Add all attributes from flow to MLIR function
        if flow.attrs and flow.attrs.attrs:
            for attr_name, attr_expr in flow.attrs.attrs.items():
                if attr_expr and isinstance(attr_expr, LitExpr):
                    # Extract the literal value
                    attr_value = attr_expr.literal.lit.value
                    # Create integer attribute for any attribute
                    attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), attr_value)
                    func_op.attributes[attr_name] = attr
                elif attr_expr is None:
                    # Simple attribute without value (like #[inline])
                    # Create a unit attribute (boolean true)
                    func_op.attributes[attr_name] = ir.UnitAttr.get()
                # TODO: Handle other expression types for attributes if needed

        # Add entry block
        entry_block = func_op.add_entry_block()

        with ir.InsertionPoint(entry_block):
            # Push new scope for flow
            self.push_scope()

            # Bind flow inputs to symbols
            for i, (name, _) in enumerate(flow.inputs):
                arg_value = entry_block.arguments[i]
                self.set_symbol(name, arg_value)

            # If flow uses memory, it will be declared via aps.memdeclare when first accessed
            # No need to set up CPU memory instance here anymore

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
            # Check for burst operations first
            if self._is_burst_operation(stmt):
                self._convert_burst_operation(stmt)
                return

            # Convert RHS expression
            rhs_value = self._convert_expr(stmt.rhs)

            # Handle LHS assignment
            if isinstance(stmt.lhs, IdentExpr):
                self.set_symbol(stmt.lhs.name, rhs_value)
            elif isinstance(stmt.lhs, IndexExpr):
                # Handle indexed assignment (e.g., _irf[rd] = value, _mem[addr] = value)
                self._convert_index_assignment(stmt.lhs, rhs_value)
            elif isinstance(stmt.lhs, RangeSliceExpr):
                # Handle range slice assignment
                self._convert_range_slice_assignment(stmt.lhs, rhs_value)
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

            # If it's a global variable reference, load from it
            if isinstance(value, str):
                # Get reference to global and load from it (with caching)
                # For now, assume u32 type - we could improve this by storing type info with the global name
                i32_type = ir.IntegerType.get_signless(32)
                memref_type = ir.MemRefType.get([], i32_type)
                global_ref = self._get_global_reference(value, memref_type)
                return aps.MemLoad(i32_type, global_ref, []).result
            else:
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

        elif isinstance(expr, IndexExpr):
            # Convert index operations - handle special cases for _irf, _mem, etc.
            return self._convert_index_expr(expr)

        elif isinstance(expr, SliceExpr):
            # Convert slice operations like z[31:31]
            return self._convert_slice_expr(expr)

        elif isinstance(expr, IfExpr):
            # Convert if expressions to conditional operations
            return self._convert_if_expr(expr)

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

    def _convert_index_expr(self, expr: IndexExpr) -> ir.Value:
        """
        Convert IndexExpr to appropriate MLIR operation based on the base expression

        Handles special cases:
        - _irf[rs] -> aps.CpuRfRead
        - _mem[addr] -> memref.LoadOp
        - regular array[idx] -> memref.LoadOp
        """
        # Check if this is a special builtin operation
        if isinstance(expr.expr, IdentExpr):
            base_name = expr.expr.name

            if base_name == "_irf":
                # Integer register file read: _irf[rs] -> aps.CpuRfRead
                if len(expr.indices) != 1:
                    raise ValueError("_irf access requires exactly one index")

                # Convert the register index
                reg_index = self._convert_expr(expr.indices[0])

                # Determine result type (assume i32 for now, could be made configurable)
                result_type = ir.IntegerType.get_signless(32)

                # Create APS register file read operation
                return aps.CpuRfRead(result_type, reg_index).result

            elif base_name == "_mem":
                # CPU memory read: _mem[addr] -> aps.memload %cpu_mem[%addr]
                if len(expr.indices) != 1:
                    raise ValueError("_mem access requires exactly one index")

                # Convert the memory address
                addr = self._convert_expr(expr.indices[0])

                # Get CPU memory instance
                cpu_mem = self.get_cpu_memory_instance()

                # Determine result type (assume i32 for now)
                result_type = ir.IntegerType.get_signless(32)

                # Generate APS memload operation
                return aps.MemLoad(result_type, cpu_mem, [addr]).result

            else:
                # Regular array/memref indexing - check if it's a global array access
                if isinstance(expr.expr, IdentExpr):
                    array_name = expr.expr.name
                    # Check if this is a global array by looking up in symbol table
                    symbol_value = self.get_symbol(array_name)
                    if isinstance(symbol_value, str):  # It's a global reference
                        # For global arrays, we need to get the actual type of the global
                        # For now, we'll determine this by looking up the static variable definition
                        # TODO: Store type information with global references for better lookup

                        # Find the static variable definition to get the correct type
                        static_var = None
                        for static in self.proc.statics.values():
                            if static.id == array_name:
                                static_var = static
                                break

                        if static_var:
                            # Convert the CADL type to MLIR type to get correct dimensions
                            cadl_mlir_type = self.convert_cadl_type(static_var.ty)
                            if isinstance(cadl_mlir_type, ir.MemRefType):
                                # Use the actual memref type from the static definition
                                global_ref = self._get_global_reference(symbol_value, cadl_mlir_type)
                            else:
                                # Fallback for scalars
                                memref_type = ir.MemRefType.get([], cadl_mlir_type)
                                global_ref = self._get_global_reference(symbol_value, memref_type)
                        else:
                            # Fallback if we can't find the static definition
                            element_type = ir.IntegerType.get_signless(32)
                            array_size = 8  # Default size for arrays like thetas
                            memref_type = ir.MemRefType.get([array_size], element_type)
                            global_ref = self._get_global_reference(symbol_value, memref_type)

                        # Convert indices
                        indices = [self._convert_expr(idx) for idx in expr.indices]

                        # Determine the element type from the memref
                        if hasattr(global_ref.type, 'element_type'):
                            element_type = global_ref.type.element_type
                        else:
                            # Fallback to i32
                            element_type = ir.IntegerType.get_signless(32)

                        # Use APS memload with the memref and indices
                        return aps.MemLoad(element_type, global_ref, indices).result
                    else:
                        # Regular local array access
                        base_value = self._convert_expr(expr.expr)
                        indices = [self._convert_expr(idx) for idx in expr.indices]

                        # Determine result type from the base memref type
                        if hasattr(base_value.type, 'element_type'):
                            result_type = base_value.type.element_type
                        else:
                            # Fallback to i32
                            result_type = ir.IntegerType.get_signless(32)

                        # Use APS memload for regular array access
                        return aps.MemLoad(result_type, base_value, indices).result
                else:
                    # Complex base expression
                    base_value = self._convert_expr(expr.expr)
                    indices = [self._convert_expr(idx) for idx in expr.indices]

                    # Determine result type from the base memref type
                    if hasattr(base_value.type, 'element_type'):
                        result_type = base_value.type.element_type
                    else:
                        # Fallback to i32
                        result_type = ir.IntegerType.get_signless(32)

                    # Use APS memload for general indexing
                    return aps.MemLoad(result_type, base_value, indices).result
        else:
            # Non-identifier base expression (e.g., function_call()[idx])
            base_value = self._convert_expr(expr.expr)
            indices = [self._convert_expr(idx) for idx in expr.indices]

            # Determine result type from the base memref type
            if hasattr(base_value.type, 'element_type'):
                result_type = base_value.type.element_type
            else:
                # Fallback to i32
                result_type = ir.IntegerType.get_signless(32)

            # Use APS memload for general indexing
            return aps.MemLoad(result_type, base_value, indices).result

    def _convert_index_assignment(self, lhs: IndexExpr, rhs_value: ir.Value) -> None:
        """
        Convert indexed assignment to appropriate MLIR operation

        Handles special cases:
        - _irf[rd] = value -> aps.CpuRfWrite
        - _mem[addr] = value -> memref.StoreOp
        - regular array[idx] = value -> memref.StoreOp
        """
        # Check if this is a special builtin operation
        if isinstance(lhs.expr, IdentExpr):
            base_name = lhs.expr.name

            if base_name == "_irf":
                # Integer register file write: _irf[rd] = value -> aps.CpuRfWrite
                if len(lhs.indices) != 1:
                    raise ValueError("_irf assignment requires exactly one index")

                # Convert the register index
                reg_index = self._convert_expr(lhs.indices[0])

                # Create APS register file write operation
                aps.CpuRfWrite(reg_index, rhs_value)

            elif base_name == "_mem":
                # CPU memory write: _mem[addr] = value -> aps.memstore %value, %cpu_mem[%addr]
                if len(lhs.indices) != 1:
                    raise ValueError("_mem assignment requires exactly one index")

                # Convert the memory address
                addr = self._convert_expr(lhs.indices[0])

                # Get CPU memory instance
                cpu_mem = self.get_cpu_memory_instance()

                # Generate APS memstore operation
                aps.MemStore(rhs_value, cpu_mem, [addr])

            else:
                # Regular array/memref assignment
                if isinstance(lhs.expr, IdentExpr):
                    # Get the memref itself, not a loaded value
                    base_value = self._get_memref_for_symbol(lhs.expr.name)
                else:
                    base_value = self._convert_expr(lhs.expr)
                indices = [self._convert_expr(idx) for idx in lhs.indices]

                # Use APS memstore for regular array assignment
                aps.MemStore(rhs_value, base_value, indices)
        else:
            # Complex base expression
            if isinstance(lhs.expr, IdentExpr):
                # Get the memref itself, not a loaded value
                base_value = self._get_memref_for_symbol(lhs.expr.name)
            else:
                base_value = self._convert_expr(lhs.expr)
            indices = [self._convert_expr(idx) for idx in lhs.indices]

            # Use APS memstore for general indexed assignment
            aps.MemStore(rhs_value, base_value, indices)

    def _is_burst_operation(self, stmt: AssignStmt) -> bool:
        """
        Detect if an assignment is a burst operation

        Burst load:  mem[start +: ] = _burst_read[cpu_addr +: length]
        Burst store: _burst_write[cpu_addr +: length] = mem[start +: ]
        """
        # Check for burst read (RHS is _burst_read with range slice)
        if isinstance(stmt.rhs, RangeSliceExpr) and isinstance(stmt.rhs.expr, IdentExpr):
            if stmt.rhs.expr.name == "_burst_read":
                return True

        # Check for burst write (LHS is _burst_write with range slice)
        if isinstance(stmt.lhs, RangeSliceExpr) and isinstance(stmt.lhs.expr, IdentExpr):
            if stmt.lhs.expr.name == "_burst_write":
                return True

        return False

    def _convert_burst_operation(self, stmt: AssignStmt) -> None:
        """
        Convert burst read/write operations to MLIR aps.memburstload/memburststore

        Burst load:  buffer[offset +: ] = _burst_read[cpu_addr +: length]
                     -> aps.memburstload %cpu_addr, %buffer[%offset], %length

        Burst store: _burst_write[cpu_addr +: length] = buffer[offset +: ]
                     -> aps.memburststore %buffer[%offset], %cpu_addr, %length
        """
        # Burst load: RHS is _burst_read
        if isinstance(stmt.rhs, RangeSliceExpr) and isinstance(stmt.rhs.expr, IdentExpr):
            if stmt.rhs.expr.name == "_burst_read":
                self._convert_burst_load(stmt)
                return

        # Burst store: LHS is _burst_write
        if isinstance(stmt.lhs, RangeSliceExpr) and isinstance(stmt.lhs.expr, IdentExpr):
            if stmt.lhs.expr.name == "_burst_write":
                self._convert_burst_store(stmt)
                return

        raise ValueError("Invalid burst operation pattern")

    def _convert_burst_load(self, stmt: AssignStmt) -> None:
        """
        Convert burst load: buffer[offset +: ] = _burst_read[cpu_addr +: length]
        to: aps.memburstload %cpu_addr, %buffer[%offset], %length
        """
        lhs = stmt.lhs  # buffer[offset +: ]
        rhs = stmt.rhs  # _burst_read[cpu_addr +: length]

        if not isinstance(lhs, RangeSliceExpr):
            raise ValueError("Burst load LHS must be a range slice expression")
        if not isinstance(rhs, RangeSliceExpr):
            raise ValueError("Burst load RHS must be a range slice expression")

        # Extract components from RHS (_burst_read[cpu_addr +: length])
        cpu_addr = self._convert_expr(rhs.start)
        if rhs.length is None:
            raise ValueError("Burst read must have explicit length")
        length = self._convert_expr(rhs.length)

        # Extract components from LHS (buffer[offset +: ])
        # Get memref for the buffer
        if isinstance(lhs.expr, IdentExpr):
            buffer_name = lhs.expr.name
            buffer_memref = self._get_memref_for_symbol(buffer_name)
        else:
            buffer_memref = self._convert_expr(lhs.expr)

        start_offset = self._convert_expr(lhs.start)

        # Infer length if not specified on LHS
        if lhs.length is not None:
            # Length specified on both sides - could validate they match
            pass

        # Generate aps.memburstload operation
        # Arguments: cpu_addr, memref, start, length
        aps.MemBurstLoad(cpu_addr, buffer_memref, start_offset, length)

    def _convert_burst_store(self, stmt: AssignStmt) -> None:
        """
        Convert burst store: _burst_write[cpu_addr +: length] = buffer[offset +: ]
        to: aps.memburststore %buffer[%offset], %cpu_addr, %length
        """
        lhs = stmt.lhs  # _burst_write[cpu_addr +: length]
        rhs = stmt.rhs  # buffer[offset +: ]

        if not isinstance(lhs, RangeSliceExpr):
            raise ValueError("Burst store LHS must be a range slice expression")
        if not isinstance(rhs, RangeSliceExpr):
            raise ValueError("Burst store RHS must be a range slice expression")

        # Extract components from LHS (_burst_write[cpu_addr +: length])
        cpu_addr = self._convert_expr(lhs.start)
        if lhs.length is None:
            raise ValueError("Burst write must have explicit length")
        length = self._convert_expr(lhs.length)

        # Extract components from RHS (buffer[offset +: ])
        # Get memref for the buffer
        if isinstance(rhs.expr, IdentExpr):
            buffer_name = rhs.expr.name
            buffer_memref = self._get_memref_for_symbol(buffer_name)
        else:
            buffer_memref = self._convert_expr(rhs.expr)

        start_offset = self._convert_expr(rhs.start)

        # Infer length if not specified on RHS
        if rhs.length is not None:
            # Length specified on both sides - could validate they match
            pass

        # Generate aps.memburststore operation
        # Arguments: memref, start, cpu_addr, length
        aps.MemBurstStore(buffer_memref, start_offset, cpu_addr, length)

    def _convert_range_slice_assignment(self, lhs: RangeSliceExpr, rhs_value: ir.Value) -> None:
        """Handle regular range slice assignments (not burst operations)"""
        # For now, this is not a common case - range slices are primarily used for burst ops
        raise NotImplementedError("Non-burst range slice assignments not yet supported")

    def _get_memref_for_symbol(self, symbol_name: str) -> ir.Value:
        """Get memref value for a symbol (handling both local and global variables)"""
        symbol_value = self.get_symbol(symbol_name)

        if symbol_value is None:
            raise ValueError(f"Undefined symbol: {symbol_name}")

        if isinstance(symbol_value, str):
            # It's a global reference - get the memref
            # Find the static variable definition to get the type
            static_var = None
            for static in self.proc.statics.values():
                if static.id == symbol_name:
                    static_var = static
                    break

            if static_var:
                cadl_mlir_type = self.convert_cadl_type(static_var.ty)
                return self._get_global_reference(symbol_value, cadl_mlir_type)
            else:
                raise ValueError(f"Cannot find static definition for global: {symbol_name}")
        else:
            # It's a local value
            return symbol_value

    def _convert_slice_expr(self, expr: SliceExpr) -> ir.Value:
        """
        Convert slice expression to MLIR bit extraction

        Handles expressions like z[31:31] (extract bit 31) or z[15:8] (extract bits 15 to 8)
        Uses CIRCT's comb dialect for bit manipulation
        """
        # Convert the base expression
        base_value = self._convert_expr(expr.expr)

        # Convert start and end indices
        start_val = self._convert_expr(expr.start)
        end_val = self._convert_expr(expr.end)

        # For now, we'll assume constant indices (most common case)
        # and use comb.ExtractOp for bit extraction

        # Get the constant values if possible
        if (hasattr(expr.start, 'literal') and hasattr(expr.start.literal, 'lit') and
            hasattr(expr.end, 'literal') and hasattr(expr.end.literal, 'lit')):

            start_bit = expr.start.literal.lit.value
            end_bit = expr.end.literal.lit.value

            # Determine the width of the extracted slice
            if start_bit == end_bit:
                # Single bit extraction - result is i1
                result_type = ir.IntegerType.get_signless(1)
                # Use comb.extract to get a single bit
                return comb.ExtractOp(result_type, base_value, start_bit).result
            else:
                # Multi-bit extraction
                width = abs(start_bit - end_bit) + 1
                result_type = ir.IntegerType.get_signless(width)
                low_bit = min(start_bit, end_bit)
                # Use comb.extract to get multiple bits
                return comb.ExtractOp(result_type, base_value, low_bit).result
        else:
            # Dynamic slice indices - more complex, use shift and mask
            # This is a fallback for non-constant indices
            # For now, assume single bit extraction and return bit 0
            result_type = ir.IntegerType.get_signless(1)
            # Extract bit at dynamic position using shift and mask
            # result = (base_value >> start_val) & 1
            shifted = comb.ShrUOp(base_value, start_val).result
            one = arith.ConstantOp(base_value.type, 1).result
            return comb.AndOp([shifted, one]).result

    def _convert_if_expr(self, expr: IfExpr) -> ir.Value:
        """
        Convert if expression to MLIR conditional operation

        Converts CADL if expressions like:
            if z_neg {x + y_shift} else {x - y_shift}

        Uses comb.MuxOp for hardware-oriented conditional selection
        """
        # Convert the condition
        condition = self._convert_expr(expr.condition)

        # Convert then and else branches
        then_value = self._convert_expr(expr.then_branch)
        else_value = self._convert_expr(expr.else_branch)

        # Ensure condition is a single bit (i1)
        # If the condition is not i1, we need to check if it's non-zero
        if condition.type != ir.IntegerType.get_signless(1):
            # Convert to boolean by comparing with zero
            zero = arith.ConstantOp(condition.type, 0).result
            condition = arith.CmpIOp(arith.CmpIPredicate.ne, condition, zero).result

        # Use comb.MuxOp for hardware-style conditional selection
        # MuxOp selects then_value when condition is true, else_value when false
        return comb.MuxOp(condition, then_value, else_value).result


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