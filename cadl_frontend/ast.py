"""
AST (Abstract Syntax Tree) classes for CADL

These classes mirror the Rust structures in the original cadl_rust implementation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


# Type aliases
Ident = str
Map = Dict


# Expression types
@dataclass
class Expr:
    """Base class for all expressions"""
    pass


@dataclass
class LitExpr(Expr):
    """Literal expression with proper type information"""
    literal: Literal


@dataclass
class StringLitExpr(Expr):
    """String literal expression"""
    value: str


@dataclass
class ComplexExpr(Expr):
    """Complex number expression"""
    real: Expr
    imag: Expr


@dataclass
class IdentExpr(Expr):
    """Identifier expression"""
    name: str


@dataclass
class TupleExpr(Expr):
    """Tuple expression"""
    elements: List[Expr]


@dataclass
class BinaryExpr(Expr):
    """Binary operation expression"""
    op: BinaryOp
    left: Expr
    right: Expr


@dataclass
class UnaryExpr(Expr):
    """Unary operation expression"""
    op: UnaryOp
    operand: Expr


@dataclass
class CallExpr(Expr):
    """Function call expression"""
    name: str
    args: List[Expr]


@dataclass
class IndexExpr(Expr):
    """Array/vector indexing expression"""
    expr: Expr
    indices: List[Expr]


@dataclass
class SliceExpr(Expr):
    """Array/vector slicing expression"""
    expr: Expr
    start: Expr
    end: Expr


@dataclass
class MatchExpr(Expr):
    """Match expression"""
    expr: Expr
    arms: List[tuple[Expr, Expr]]


@dataclass
class SelectExpr(Expr):
    """Select expression"""
    arms: List[tuple[Expr, Expr]]
    default: Expr


@dataclass
class IfExpr(Expr):
    """If expression"""
    condition: Expr
    then_branch: Expr
    else_branch: Expr


@dataclass
class AggregateExpr(Expr):
    """Aggregate expression (like struct literals)"""
    elements: List[Expr]


# Binary operators
class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    REM = "%"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "&&"
    OR = "||"
    BIT_AND = "&"
    BIT_OR = "|"
    BIT_XOR = "^"
    LSHIFT = "<<"
    RSHIFT = ">>"


# Unary operators
class UnaryOp(Enum):
    NEG = "-"
    NOT = "!"
    BIT_NOT = "~"
    SIGNED_CAST = "$signed"
    UNSIGNED_CAST = "$unsigned"
    F32_CAST = "$f32"
    F64_CAST = "$f64"
    INT_CAST = "$int"
    UINT_CAST = "$uint"


# New IR Type System - matches type_sys_ir.rs
from enum import Enum

class BasicType:
    """Base class for BasicType enum variants"""
    pass

@dataclass  
class BasicType_ApFixed(BasicType):
    """Signed fixed-point type - BasicType::ApFixed(u32)"""
    width: int

@dataclass
class BasicType_ApUFixed(BasicType): 
    """Unsigned fixed-point type - BasicType::ApUFixed(u32)"""
    width: int

@dataclass
class BasicType_Float32(BasicType):
    """32-bit float type - BasicType::Float32"""
    pass

@dataclass
class BasicType_Float64(BasicType):
    """64-bit float type - BasicType::Float64"""  
    pass

@dataclass
class BasicType_String(BasicType):
    """String type - BasicType::String"""
    pass

@dataclass
class BasicType_USize(BasicType):
    """USize type - BasicType::USize"""
    pass


class DataType:
    """Base class for DataType enum variants"""
    pass

@dataclass
class DataType_Single(DataType):
    """Single data type - DataType::Single(BasicType)"""
    basic_type: BasicType

@dataclass
class DataType_Array(DataType):
    """Array data type - DataType::Array(BasicType, Vec<usize>)"""
    element_type: BasicType
    dimensions: List[int]

@dataclass
class DataType_Instance(DataType):
    """Instance data type - DataType::Instance"""
    pass


class CompoundType:
    """Base class for CompoundType enum variants"""
    
    def to_basic(self) -> DataType:
        """Convert to basic type - matches Rust as_basic() method"""
        if isinstance(self, CompoundType_Basic):
            return self.data_type
        elif isinstance(self, CompoundType_FnTy):
            raise RuntimeError("Cannot convert function type to basic type")
        else:
            raise RuntimeError(f"Unknown CompoundType variant: {type(self)}")

@dataclass
class CompoundType_Basic(CompoundType):
    """Basic compound type - CompoundType::Basic(DataType)"""  
    data_type: DataType

@dataclass
class CompoundType_FnTy(CompoundType):
    """Function compound type - CompoundType::FnTy(Vec<CompoundType>, Vec<CompoundType>)"""
    args: List['CompoundType']
    ret: List['CompoundType']


# Literal System - matches literal.rs
class LiteralInner:
    """Base class for LiteralInner enum variants"""
    pass

@dataclass
class LiteralInner_Fixed(LiteralInner):
    """Fixed-point literal - LiteralInner::Fixed(BigInt)"""
    value: int  # Using int instead of BigInt for simplicity

@dataclass
class LiteralInner_Float(LiteralInner):
    """Float literal - LiteralInner::Float(f64)"""
    value: float

@dataclass
class Literal:
    """Literal with type information - matches Rust Literal struct"""
    lit: LiteralInner
    ty: BasicType


# Statements
@dataclass
class Stmt:
    """Base class for all statements"""
    pass


@dataclass
class ExprStmt(Stmt):
    """Expression statement"""
    expr: Expr


@dataclass
class AssignStmt(Stmt):
    """Assignment statement"""
    is_let: bool
    lhs: Expr
    rhs: Expr
    type_annotation: Optional[DataType] = None


@dataclass
class ReturnStmt(Stmt):
    """Return statement"""
    exprs: List[Expr]


@dataclass
class ForStmt(Stmt):
    """For loop statement"""
    init: Stmt
    condition: Expr
    update: Stmt
    body: List[Stmt]


@dataclass
class StaticStmt(Stmt):
    """Static variable declaration statement"""
    static: 'Static'


@dataclass
class GuardStmt(Stmt):
    """Guard statement (conditional execution)"""
    condition: Expr
    stmt: Stmt


@dataclass
class DoWhileStmt(Stmt):
    """Do-while loop statement"""
    bindings: List['WithBinding']
    body: List[Stmt]
    condition: Expr


@dataclass
class DirectiveStmt(Stmt):
    """Directive statement (compiler hints)"""
    name: str
    expr: Optional[Expr] = None


@dataclass
class FunctionStmt(Stmt):
    """Function declaration statement"""
    function: 'Function'


@dataclass
class SpawnStmt(Stmt):
    """Spawn statement (parallel execution)"""
    stmts: List[Stmt]


@dataclass
class ThreadStmt(Stmt):
    """Thread statement"""
    thread: 'Thread'


# Function-related structures
@dataclass
class WithBinding:
    """With binding for loop constructs"""
    id: str
    ty: BasicType
    init: Optional[Expr] = None
    next: Optional[Expr] = None


@dataclass
class FnArg:
    """Function argument"""
    id: str
    ty: CompoundType


@dataclass
class Function:
    """Function definition"""
    name: str
    args: List[FnArg]
    ret: List[CompoundType]
    body: List[Stmt] = field(default_factory=list)


@dataclass
class Static:
    """Static variable declaration"""
    id: str
    ty: DataType
    expr: Optional[Expr] = None


@dataclass
class Thread:
    """Thread definition"""
    id: str
    body: List[Stmt] = field(default_factory=list)


# Flow-related structures
class FlowKind(Enum):
    DEFAULT = "default"
    RTYPE = "rtype"


@dataclass
class FlowAttributes:
    """Flow attributes (decorators)"""
    activator: Optional[Expr] = None
    funct3: Optional[Expr] = None
    funct7: Optional[Expr] = None
    opcode: Optional[Expr] = None

    @classmethod
    def from_tuples(cls, tuples: List[tuple[str, Optional[Expr]]]) -> FlowAttributes:
        """Create from list of attribute tuples"""
        attrs = cls()
        for name, value in tuples:
            if name == "activator":
                attrs.activator = value
            elif name == "funct3":
                attrs.funct3 = value
            elif name == "funct7":
                attrs.funct7 = value
            elif name == "opcode":
                attrs.opcode = value
        return attrs

    def with_activator(self, activator: Expr) -> FlowAttributes:
        """Add activator attribute"""
        self.activator = activator
        return self


@dataclass
class Flow:
    """Flow definition"""
    name: str
    kind: FlowKind
    inputs: List[tuple[str, DataType]] = field(default_factory=list)
    attrs: FlowAttributes = field(default_factory=FlowAttributes)
    body: Optional[List[Stmt]] = None

    def fields(self) -> List[tuple[str, DataType]]:
        """Get flow input fields"""
        return self.inputs

    def get_body(self) -> Optional[List[Stmt]]:
        """Get flow body"""
        return self.body


@dataclass
class Regfile:
    """Register file definition"""
    name: str
    width: int
    depth: int


# Processor structure
@dataclass
class ProcPart:
    """Base class for processor parts"""
    pass


@dataclass
class RegfilePart(ProcPart):
    """Regfile processor part"""
    regfile: Regfile


@dataclass
class FlowPart(ProcPart):
    """Flow processor part"""
    flow: Flow


@dataclass
class FunctionPart(ProcPart):
    """Function processor part"""
    function: Function


@dataclass
class StaticPart(ProcPart):
    """Static processor part"""
    static: Static


@dataclass
class Proc:
    """Main processor structure"""
    regfiles: Map[str, Regfile] = field(default_factory=dict)
    flows: Map[str, Flow] = field(default_factory=dict)
    functions: Map[str, Function] = field(default_factory=dict)
    statics: Map[str, Static] = field(default_factory=dict)

    def get_flows(self) -> List[Flow]:
        """Get all flows"""
        return list(self.flows.values())

    def add_part(self, part: ProcPart) -> None:
        """Add a processor part"""
        if isinstance(part, RegfilePart):
            self.regfiles[part.regfile.name] = part.regfile
        elif isinstance(part, FlowPart):
            self.flows[part.flow.name] = part.flow
        elif isinstance(part, FunctionPart):
            self.functions[part.function.name] = part.function
        elif isinstance(part, StaticPart):
            self.statics[part.static.id] = part.static

    @classmethod
    def from_parts(cls, parts: List[ProcPart]) -> Proc:
        """Create processor from parts"""
        proc = cls()
        for part in parts:
            proc.add_part(part)
        return proc


# Helper methods for expressions
def expr_is_lval(expr: Expr) -> bool:
    """Check if expression is an lvalue"""
    return isinstance(expr, (IdentExpr, IndexExpr, SliceExpr))


def expr_as_literal(expr: Expr) -> Optional[str]:
    """Get expression as literal if possible"""
    if isinstance(expr, LitExpr):
        return expr.value
    return None


def expr_flatten(expr: Expr) -> List[Expr]:
    """Flatten tuple expressions"""
    if isinstance(expr, TupleExpr):
        result = []
        for element in expr.elements:
            result.extend(expr_flatten(element))
        return result
    return [expr]


# Convenience factory methods for type system  
def parse_basic_type_from_string(type_str: str) -> BasicType:
    """Parse a basic type from string (like 'u32', 'i8', 'f32')"""
    if type_str == "usize":
        return BasicType_USize()
    elif type_str.startswith('u'):
        width = int(type_str[1:])
        return BasicType_ApUFixed(width)
    elif type_str.startswith('i'):
        width = int(type_str[1:])  
        return BasicType_ApFixed(width)
    elif type_str == "f32":
        return BasicType_Float32()
    elif type_str == "f64":
        return BasicType_Float64()
    elif type_str == "string":
        return BasicType_String()
    else:
        raise ValueError(f"Unknown basic type: {type_str}")


def parse_literal_from_string(literal_str: str) -> Literal:
    """Parse a number literal string into a Literal with proper type"""
    # Simple implementation - will be enhanced later
    if "'" in literal_str:
        # Handle width-specified literals like "5'b101010"
        parts = literal_str.split("'", 1)
        width = int(parts[0])
        format_and_value = parts[1]
        
        if format_and_value.startswith(('b', 'B')):
            # Binary format
            value = int(format_and_value[1:], 2)
            return Literal(
                LiteralInner_Fixed(value),
                BasicType_ApUFixed(width)
            )
        elif format_and_value.startswith(('h', 'H')):
            # Hex format
            value = int(format_and_value[1:], 16)
            return Literal(
                LiteralInner_Fixed(value),
                BasicType_ApUFixed(width)
            )
        elif format_and_value.startswith(('o', 'O')):
            # Octal format
            value = int(format_and_value[1:], 8)
            return Literal(
                LiteralInner_Fixed(value), 
                BasicType_ApUFixed(width)
            )
        elif format_and_value.startswith(('d', 'D')):
            # Decimal format
            value = int(format_and_value[1:])
            return Literal(
                LiteralInner_Fixed(value),
                BasicType_ApUFixed(width)
            )
    else:
        # Simple integer or hex without width specification
        if literal_str.startswith('0x') or literal_str.startswith('0X'):
            value = int(literal_str, 16)
        else:
            value = int(literal_str)
        
        # Default to 32-bit unsigned
        return Literal(
            LiteralInner_Fixed(value),
            BasicType_ApUFixed(32)
        )