"""
CADL Parser module

Provides the main parsing functionality using Lark parser generator.
Converts parse trees into AST nodes matching the Rust implementation.
"""

from pathlib import Path
from typing import Optional
try:
    from lark import Lark, Transformer
except ImportError:
    raise ImportError("lark package is required. Install with: pip install lark")
from .ast import *


class CADLTransformer(Transformer):
    """Transformer to convert Lark parse tree to CADL AST"""

    # Literals and identifiers  
    def number_lit(self, items):
        literal_str = str(items[0])
        literal = parse_literal_from_string(literal_str)
        return LitExpr(literal)

    def string_lit(self, items):
        return StringLitExpr(str(items[0]).strip('"'))

    def true_lit(self, items):
        return LitExpr("true")

    def false_lit(self, items):
        return LitExpr("false")

    def identifier(self, items):
        return IdentExpr(str(items[0]))

    # Type system
    def single_type(self, items):
        basic_type = parse_basic_type_from_string(str(items[0]))
        return DataType_Single(basic_type)

    def array_type(self, items):
        # Grammar: LBRACKET VARTYPE (SEMICOLON NUMBER_LIT)* RBRACKET
        element_type = parse_basic_type_from_string(str(items[1]))  # Skip LBRACKET
        dimensions = []
        # Extract dimensions from SEMICOLON NUMBER_LIT pairs
        i = 2
        while i < len(items) - 1:  # Skip final RBRACKET
            if hasattr(items[i], 'type') and items[i].type == 'SEMICOLON':
                dimensions.append(int(str(items[i + 1])))
                i += 2
            else:
                i += 1
        return DataType_Array(element_type, dimensions)

    def instance_type(self, items):
        return DataType_Instance()

    def basic_type(self, items):
        return CompoundType_Basic(items[0])

    def fn_type(self, items):
        args = items[0] if items[0] else []
        ret = items[1] if items[1] else []
        return CompoundType_FnTy(args, ret)

    def compound_type_list(self, items):
        return items

    # Function arguments
    def fn_arg(self, items):
        name = str(items[0])
        # items[1] is the COLON token, items[2] is the compound_type
        type_info = items[2]
        return FnArg(name, type_info)

    def fn_arg_list(self, items):
        # Filter out comma tokens and return only FnArg objects
        return [item for item in items if isinstance(item, FnArg)]

    # With bindings
    def with_binding(self, items):
        name = str(items[0])      # IDENTIFIER
        type_name = str(items[2]) # VARTYPE (skip COLON)
        basic_type = parse_basic_type_from_string(type_name)
        init_expr = items[5] if len(items) > 5 and items[5] is not None else None  # first expr
        next_expr = items[7] if len(items) > 7 and items[7] is not None else None  # second expr
        return WithBinding(name, basic_type, init_expr, next_expr)

    # Expressions - Binary operations
    def add_op(self, items):
        return BinaryExpr(BinaryOp.ADD, items[0], items[2])

    def sub_op(self, items):
        return BinaryExpr(BinaryOp.SUB, items[0], items[2])

    def mul_op(self, items):
        return BinaryExpr(BinaryOp.MUL, items[0], items[2])

    def div_op(self, items):
        return BinaryExpr(BinaryOp.DIV, items[0], items[2])

    def rem_op(self, items):
        return BinaryExpr(BinaryOp.REM, items[0], items[2])

    def eq_op(self, items):
        return BinaryExpr(BinaryOp.EQ, items[0], items[2])

    def ne_op(self, items):
        return BinaryExpr(BinaryOp.NE, items[0], items[2])

    def lt_op(self, items):
        return BinaryExpr(BinaryOp.LT, items[0], items[2])

    def le_op(self, items):
        return BinaryExpr(BinaryOp.LE, items[0], items[2])

    def gt_op(self, items):
        return BinaryExpr(BinaryOp.GT, items[0], items[2])

    def ge_op(self, items):
        return BinaryExpr(BinaryOp.GE, items[0], items[2])

    def and_op(self, items):
        return BinaryExpr(BinaryOp.AND, items[0], items[2])

    def or_op(self, items):
        return BinaryExpr(BinaryOp.OR, items[0], items[2])

    def lshift_op(self, items):
        return BinaryExpr(BinaryOp.LSHIFT, items[0], items[2])

    def rshift_op(self, items):
        return BinaryExpr(BinaryOp.RSHIFT, items[0], items[2])

    def bit_and_op(self, items):
        return BinaryExpr(BinaryOp.BIT_AND, items[0], items[2])

    def bit_or_op(self, items):
        return BinaryExpr(BinaryOp.BIT_OR, items[0], items[2])

    def bit_xor_op(self, items):
        return BinaryExpr(BinaryOp.BIT_XOR, items[0], items[2])

    # Expressions - Unary operations
    def neg_op(self, items):
        return UnaryExpr(UnaryOp.NEG, items[0])

    def not_op(self, items):
        return UnaryExpr(UnaryOp.NOT, items[0])

    def bit_not_op(self, items):
        return UnaryExpr(UnaryOp.BIT_NOT, items[0])

    def signed_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.SIGNED_CAST, items[2])

    def unsigned_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.UNSIGNED_CAST, items[2])

    def f32_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.F32_CAST, items[2])

    def f64_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.F64_CAST, items[2])

    def int_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.INT_CAST, items[2])

    def uint_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return UnaryExpr(UnaryOp.UINT_CAST, items[2])

    # Complex expressions
    def call_expr(self, items):
        name = str(items[0])  # IDENTIFIER
        # items[1] is LPAREN, items[2] is expr_list (optional), items[3] is RPAREN
        args = items[2] if len(items) > 3 and items[2] else []
        return CallExpr(name, args)

    def index_expr(self, items):
        expr = items[0]
        indices = items[2]  # Skip LBRACKET token, get expr_list
        return IndexExpr(expr, indices)

    def slice_expr(self, items):
        expr = items[0]
        start = items[2]  # Skip LBRACKET token
        end = items[4]    # Skip COLON token
        return SliceExpr(expr, start, end)

    def paren_expr(self, items):
        # items = [LPAREN, expr_list, RPAREN]
        expr_list = items[1] if len(items) > 1 else []
        if isinstance(expr_list, list):
            if len(expr_list) == 1:
                return expr_list[0]
            return TupleExpr(expr_list)
        else:
            return expr_list

    def if_expr(self, items):
        # items = [KW_IF, condition, LBRACE, then_branch, RBRACE, KW_ELSE, LBRACE, else_branch, RBRACE]
        condition = items[1]      # Skip KW_IF
        then_branch = items[3]    # Skip LBRACE
        else_branch = items[7]    # Skip KW_ELSE, LBRACE
        return IfExpr(condition, then_branch, else_branch)

    def match_expr(self, items):
        expr = items[0]
        arms = items[1:]
        return MatchExpr(expr, arms)

    def select_expr(self, items):
        arms = items[:-1]
        default = items[-1]
        return SelectExpr(arms, default)

    def aggregate_expr(self, items):
        # Grammar: LBRACE expr_list RBRACE
        expr_list = items[1]  # Skip LBRACE, get expr_list, skip RBRACE
        return AggregateExpr(expr_list)

    def match_arm(self, items):
        return (items[0], items[1])

    def sel_arm(self, items):
        return (items[0], items[1])

    def expr_list(self, items):
        # Filter out COMMA tokens, keep only expressions
        return [item for item in items if not (hasattr(item, 'type') and item.type == 'COMMA')]

    # Statements
    def expr_stmt(self, items):
        return ExprStmt(items[0])

    def assign_stmt(self, items):
        # Grammar: KW_LET? expr (COLON data_type)? ASSIGN expr SEMICOLON
        is_let = any(hasattr(item, 'type') and item.type == 'KW_LET' for item in items)
        
        # Find indices of key tokens
        assign_idx = next(i for i, item in enumerate(items) if hasattr(item, 'type') and item.type == 'ASSIGN')
        
        if is_let:
            lhs = items[1]  # expr after KW_LET
        else:
            lhs = items[0]  # first expr
            
        # RHS is the expression after ASSIGN (before SEMICOLON)
        rhs = items[assign_idx + 1]
        
        # Type annotation is between COLON and ASSIGN if present
        type_annotation = None
        colon_idx = next((i for i, item in enumerate(items) if hasattr(item, 'type') and item.type == 'COLON'), None)
        if colon_idx is not None and colon_idx < assign_idx:
            type_annotation = items[colon_idx + 1]
            
        return AssignStmt(is_let, lhs, rhs, type_annotation)

    def return_stmt(self, items):
        # items = [KW_RETURN, expr_list, SEMICOLON]
        expr_list = items[1] if len(items) > 1 else []
        return ReturnStmt(expr_list if isinstance(expr_list, list) else [expr_list])

    def guard_stmt(self, items):
        condition = items[0]
        stmt = items[1]
        return GuardStmt(condition, stmt)

    def do_while_stmt(self, items):
        # Grammar: KW_WITH with_binding* KW_DO body KW_WHILE expr SEMICOLON
        bindings = []
        
        # Find KW_DO to separate bindings from body
        do_idx = next(i for i, item in enumerate(items) if hasattr(item, 'type') and item.type == 'KW_DO')
        while_idx = next(i for i, item in enumerate(items) if hasattr(item, 'type') and item.type == 'KW_WHILE')
        
        # Extract bindings (between KW_WITH and KW_DO)
        for i in range(1, do_idx):  # Skip KW_WITH
            if isinstance(items[i], WithBinding):
                bindings.append(items[i])
        
        # Extract body (between KW_DO and KW_WHILE)
        body = items[do_idx + 1]  # Should be the transformed body
        
        # Extract condition (between KW_WHILE and SEMICOLON)
        condition = items[while_idx + 1]
        
        return DoWhileStmt(bindings, body, condition)

    def directive_stmt(self, items):
        name = str(items[0])
        expr = items[1] if len(items) > 1 else None
        return DirectiveStmt(name, expr)

    def function_stmt(self, items):
        return FunctionStmt(items[0])

    def spawn_stmt(self, items):
        return SpawnStmt(items)

    def static_stmt(self, items):
        return StaticStmt(items[0])

    def thread_stmt(self, items):
        return ThreadStmt(items[0])

    # Body
    def empty_body(self, items):
        return None

    def block_body(self, items):
        # Filter out LBRACE and RBRACE tokens, return only statements
        return [item for item in items if hasattr(item, '__class__') and 'Stmt' in item.__class__.__name__]

    # Function, static, thread definitions
    def function(self, items):
        name = str(items[1])      # Skip KW_FN
        args = items[3] if len(items) > 3 and items[3] else []      # fn_arg_list after LPAREN
        ret_types = items[7] if len(items) > 7 and items[7] else [] # compound_type_list after second LPAREN  
        body = items[9] if len(items) > 9 else []                   # body after second RPAREN
        return Function(name, args, ret_types, body)

    def static(self, items):
        # Expected structure: KW_STATIC IDENTIFIER COLON data_type (ASSIGN expr)? SEMICOLON
        name = str(items[1])  # IDENTIFIER token
        type_info = items[3]  # data_type (already transformed)
        expr = items[5] if len(items) > 5 else None  # expr (already transformed)
        return Static(name, type_info, expr)

    def thread(self, items):
        name = str(items[0])
        body = items[1] if items[1] else []
        return Thread(name, body)

    # Attributes
    def simple_attr(self, items):
        return (str(items[0]), None)

    def param_attr(self, items):
        return (str(items[0]), items[1])

    # Flow definition
    def default_flow(self, items):
        attrs = []
        idx = 0
        
        # Extract attributes
        while idx < len(items) and isinstance(items[idx], tuple):
            attrs.append(items[idx])
            idx += 1
        
        # Skip KW_FLOW token
        name_idx = idx + 1
        
        name = str(items[name_idx])
        
        # Find the inputs and body by looking for the right types
        inputs = []
        body = None
        
        for i in range(name_idx + 1, len(items)):
            item = items[i]
            if isinstance(item, list) and len(item) > 0 and all(isinstance(arg, FnArg) for arg in item):
                # This is the fn_arg_list
                inputs = item
            elif item is None:
                # This is an empty body
                body = item
                break
            elif isinstance(item, list):
                # This could be the body (list of statements)
                body = item
                break
        
        flow_attrs = FlowAttributes.from_tuples(attrs)
        input_pairs = [(arg.id, arg.ty.to_basic()) for arg in inputs]
        
        return Flow(name, FlowKind.DEFAULT, input_pairs, flow_attrs, body)

    def rtype_flow(self, items):
        attrs = []
        idx = 0
        
        # Extract attributes
        while idx < len(items) and isinstance(items[idx], tuple):
            attrs.append(items[idx])
            idx += 1
        
        # Skip KW_RTYPE token
        name_idx = idx + 1
        
        name = str(items[name_idx])
        
        # Find the inputs and body by looking for the right types
        inputs = []
        body = None
        
        for i in range(name_idx + 1, len(items)):
            item = items[i]
            if isinstance(item, list) and len(item) > 0 and all(isinstance(arg, FnArg) for arg in item):
                # This is the fn_arg_list
                inputs = item
            elif item is None:
                # This is an empty body
                body = item
                break
            elif isinstance(item, list):
                # This could be the body (list of statements)
                body = item
                break
        
        flow_attrs = FlowAttributes.from_tuples(attrs)
        input_pairs = [(arg.id, arg.ty.to_basic()) for arg in inputs]
        
        return Flow(name, FlowKind.RTYPE, input_pairs, flow_attrs, body)

    # Regfile definition
    def regfile(self, items):
        name = str(items[1])      # Skip KW_REGFILE
        width = int(str(items[3])) # Skip LPAREN  
        depth = int(str(items[5])) # Skip COMMA
        return Regfile(name, width, depth)

    # Attribute transformers
    def simple_attr(self, items):
        # Grammar: HASH LBRACKET IDENTIFIER RBRACKET
        name = str(items[2])  # Skip HASH, LBRACKET, get IDENTIFIER
        return (name, None)
    
    def param_attr(self, items):
        # Grammar: HASH LBRACKET IDENTIFIER LPAREN expr RPAREN RBRACKET
        name = str(items[2])  # Skip HASH, LBRACKET, get IDENTIFIER
        expr = items[4]       # Skip LPAREN, get expr
        return (name, expr)

    # Processor parts
    def proc_part(self, items):
        part = items[0]
        if isinstance(part, Regfile):
            return RegfilePart(part)
        elif isinstance(part, Flow):
            return FlowPart(part)
        elif isinstance(part, Function):
            return FunctionPart(part)
        elif isinstance(part, Static):
            return StaticPart(part)
        return part

    # Main processor
    def proc(self, items):
        return Proc.from_parts(items)

    def start(self, items):
        return items[0]


class CADLParser:
    """Main CADL parser class"""

    def __init__(self):
        """Initialize the parser with Lark grammar"""
        grammar_path = Path(__file__).parent / "grammar.lark"
        with open(grammar_path, 'r') as f:
            grammar = f.read()
        
        self.parser = Lark(
            grammar,
            parser='lalr',  # Using LALR parser for better performance with transformers
            start='start'
        )
        self.transformer = CADLTransformer()

    def parse(self, source: str, filename: Optional[str] = None) -> Proc:
        """Parse CADL source code into AST"""
        try:
            parse_tree = self.parser.parse(source)
            result = self.transformer.transform(parse_tree)
            return result
        except Exception as e:
            if filename:
                raise ValueError(f"Parse error in {filename}: {e}") from e
            else:
                raise ValueError(f"Parse error: {e}") from e


# Global parser instance
_parser = None


def get_parser() -> CADLParser:
    """Get or create global parser instance"""
    global _parser
    if _parser is None:
        _parser = CADLParser()
    return _parser


def parse_proc(source: str, filename: Optional[str] = None) -> Proc:
    """Parse a CADL processor from source code
    
    Args:
        source: CADL source code string
        filename: Optional filename for error reporting
        
    Returns:
        Proc: Parsed processor AST
        
    Raises:
        ValueError: On parse errors
    """
    parser = get_parser()
    return parser.parse(source, filename)