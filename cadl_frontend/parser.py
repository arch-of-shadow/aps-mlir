"""
CADL Parser module

Provides the main parsing functionality using Lark parser generator.
Converts parse trees into AST nodes matching the Rust implementation.
"""

from pathlib import Path
from typing import Optional
from lark import Lark, Transformer, Token, UnexpectedInput
from lark.exceptions import (
    UnexpectedToken,
    UnexpectedCharacters,
    ParseError,
    VisitError,
)
from . import cadl_ast

def _is_token(item, token_type: str) -> bool:
    return isinstance(item, Token) and item.type == token_type


class CADLParseError(Exception):
    """Enhanced CADL parse error with formatting"""

    def __init__(
        self,
        message: str,
        line: int,
        column: int,
        filename: Optional[str] = None,
        source_lines: Optional[list] = None,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.line = line
        self.column = column
        self.filename = filename
        self.source_lines = source_lines or []
        self.suggestion = suggestion
        super().__init__(self.format_error())

    def format_error(self) -> str:
        """Format a pretty error message with source context"""
        lines = []

        # Header
        if self.filename:
            lines.append(f"Error in {self.filename}:")
        else:
            lines.append("Parse Error:")

        lines.append("")

        # Main error message
        lines.append(f"  {self.message}")
        lines.append("")

        # Source context
        if self.source_lines and self.line > 0:
            # Show 1-2 lines before and after the error
            start_line = max(1, self.line - 1)
            end_line = min(len(self.source_lines), self.line + 1)

            for line_num in range(start_line, end_line + 1):
                line_content = (
                    self.source_lines[line_num - 1]
                    if line_num <= len(self.source_lines)
                    else ""
                )
                prefix = "  → " if line_num == self.line else "    "
                lines.append(f"{prefix}{line_num:4} | {line_content}")

                # Add pointer to error column
                if line_num == self.line and self.column > 0:
                    pointer_line = "    " + " " * 5 + " " * (self.column - 1) + "^"
                    lines.append(pointer_line)

        # Location info
        if self.line > 0 and self.column > 0:
            lines.append("")
            lines.append(f"  at line {self.line}, column {self.column}")

        # Suggestion if available
        if self.suggestion:
            lines.append("")
            lines.append(f"  💡 Suggestion: {self.suggestion}")

        return "\n".join(lines)


ERROR_EXAMPLES = {
    "Missing semicolon": [
        "rtype bad(rd: u5) { let x: u32 = 1 _irf[rd] = x; }",
        "rtype bad(rd: u5) { _irf[rd] = 1 }",
    ],
    "Missing type annotation": [
        "rtype bad(rd: u5) { let x = 1; }",
        "rtype bad(rd: u5) { let x: = 1; }",
    ],
    "Malformed flow declaration": [
        "rtype bad(rs1 u5) { }",
        "rtype bad(rs1: u5, ) { }",
    ],
    "Malformed static declaration": [
        "static data [u32; 4];",
        "static data: [u32; ];",
    ],
    "Malformed expression": [
        "rtype bad(rd: u5) { _irf[rd] = 1 + ; }",
        "rtype bad(rd: u5) { _irf[rd] = if 1 { 2 }; }",
    ],
    "Unclosed delimiter": [
        "rtype bad(rd: u5) { _irf[rd] = (1 + 2; }",
        "rtype bad(rd: u5) { _irf[rd] = arr[0; }",
    ],
}


ERROR_SUGGESTIONS = {
    "Missing semicolon": "End statements with ';'.",
    "Missing type annotation": "Use 'let name: type = value;'.",
    "Malformed flow declaration": "Use 'rtype name(arg: type, ...) { ... }'.",
    "Malformed static declaration": "Use 'static name: type;' or 'static name: type = value;'.",
    "Malformed expression": "Check the expression around the highlighted token.",
    "Unclosed delimiter": "Check matching parentheses, brackets, and braces.",
}


EXPECTED_TOKEN_HINTS = {
    "SEMICOLON": "Expected ';'.",
    "COLON": "Expected ':'.",
    "ASSIGN": "Expected '='.",
    "COMMA": "Expected ','.",
    "RBRACE": "Expected '}'.",
    "RBRACKET": "Expected ']'.",
    "RPAREN": "Expected ')'.",
}


def _token_value(token) -> str:
    return token.value if isinstance(token, Token) else str(token)


def _expected_hint(expected) -> Optional[str]:
    for token_name, hint in EXPECTED_TOKEN_HINTS.items():
        if token_name in expected:
            return hint
    return None


def _match_example_error(e: UnexpectedInput, parse_fn) -> Optional[str]:
    if parse_fn is None:
        return None
    return e.match_examples(parse_fn, ERROR_EXAMPLES, use_accepts=True)


def format_lark_error(
    e: Exception,
    source: str,
    filename: Optional[str] = None,
    parse_fn=None,
) -> CADLParseError:
    """Convert Lark parsing exception to pretty CADLParseError"""
    source_lines = source.splitlines()

    if isinstance(e, UnexpectedToken):
        label = _match_example_error(e, parse_fn)
        token_value = _token_value(e.token)
        if label:
            message = label
            suggestion = ERROR_SUGGESTIONS.get(label)
        else:
            message = f"Unexpected '{token_value}'"
            suggestion = _expected_hint(e.expected)
        return CADLParseError(
            message, e.line, e.column, filename, source_lines, suggestion
        )

    elif isinstance(e, UnexpectedCharacters):
        char = (
            source[e.pos_in_stream]
            if e.pos_in_stream is not None and e.pos_in_stream < len(source)
            else "?"
        )
        suggestion = "Check the character at the highlighted position."
        return CADLParseError(
            f"Unexpected character '{char}'",
            e.line,
            e.column,
            filename,
            source_lines,
            suggestion,
        )

    else:
        return CADLParseError(
            f"Parse error: {e}",
            1,
            1,
            filename,
            source_lines,
            "Check overall file syntax and structure.",
        )


class CADLTransformer(Transformer):
    """Transformer to convert Lark parse tree to CADL AST"""

    def _validate_burst_operation(self, lhs: cadl_ast.Expr, rhs: cadl_ast.Expr) -> None:
        """
        Validate burst operation constraints:
        - Burst lengths must be compile-time constants (literals)
        """

        def check_burst_length(expr: cadl_ast.Expr, side_name: str) -> None:
            """Check if expr is a burst operation with non-constant length"""
            if isinstance(expr, cadl_ast.RangeSliceExpr):
                # Check if this is a burst operation
                if isinstance(expr.expr, cadl_ast.IdentExpr):
                    if expr.expr.name in ("_burst_read", "_burst_write"):
                        if expr.length is None:
                            raise ValueError(
                                f"Burst operation on {side_name} must have explicit length"
                            )
                        # Require length to be a literal expression
                        if not isinstance(expr.length, cadl_ast.LitExpr):
                            raise ValueError(
                                f"Burst operation length on {side_name} must be a compile-time constant literal. "
                                f"Got: {type(expr.length).__name__}"
                            )

        # Check both sides
        check_burst_length(lhs, "LHS")
        check_burst_length(rhs, "RHS")

    # Literals and identifiers
    def number_lit(self, items):
        literal_str = str(items[0])
        literal = cadl_ast.parse_literal_from_string(literal_str)
        return cadl_ast.LitExpr(literal)

    def string_lit(self, items):
        return cadl_ast.StringLitExpr(str(items[0]).strip('"'))

    def true_lit(self, items):
        return cadl_ast.LitExpr("true")

    def false_lit(self, items):
        return cadl_ast.LitExpr("false")

    def identifier(self, items):
        return cadl_ast.IdentExpr(str(items[0]))

    # Type system
    def single_type(self, items):
        basic_type = cadl_ast.parse_basic_type_from_string(str(items[0]))
        return cadl_ast.DataType_Single(basic_type)

    def array_type(self, items):
        # Grammar: LBRACKET VARTYPE (SEMICOLON NUMBER_LIT)* RBRACKET
        element_type = cadl_ast.parse_basic_type_from_string(str(items[1]))  # Skip LBRACKET
        dimensions = []
        # Extract dimensions from SEMICOLON NUMBER_LIT pairs
        i = 2
        while i < len(items) - 1:  # Skip final RBRACKET
            if _is_token(items[i], "SEMICOLON"):
                dimensions.append(int(str(items[i + 1])))
                i += 2
            else:
                i += 1
        return cadl_ast.DataType_Array(element_type, dimensions)

    def instance_type(self, items):
        return cadl_ast.DataType_Instance()

    def basic_type(self, items):
        return cadl_ast.CompoundType_Basic(items[0])

    # Function arguments
    def fn_arg(self, items):
        name = str(items[0])
        # items[1] is the COLON token, items[2] is the compound_type
        type_info = items[2]
        return cadl_ast.FnArg(name, type_info)

    def fn_arg_list(self, items):
        # Filter out comma tokens and return only FnArg objects
        return [item for item in items if isinstance(item, cadl_ast.FnArg)]

    # With bindings
    def with_binding(self, items):
        name = str(items[0])  # IDENTIFIER
        type_name = str(items[2])  # VARTYPE (skip COLON)
        basic_type = cadl_ast.parse_basic_type_from_string(type_name)
        init_expr = (
            items[5] if len(items) > 5 and items[5] is not None else None
        )  # first expr
        next_expr = (
            items[7] if len(items) > 7 and items[7] is not None else None
        )  # second expr
        return cadl_ast.WithBinding(name, basic_type, init_expr, next_expr)

    # Expressions - Binary operations
    def add_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.ADD, items[0], items[2])

    def sub_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.SUB, items[0], items[2])

    def mul_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.MUL, items[0], items[2])

    def div_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.DIV, items[0], items[2])

    def rem_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.REM, items[0], items[2])

    def eq_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.EQ, items[0], items[2])

    def ne_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.NE, items[0], items[2])

    def lt_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.LT, items[0], items[2])

    def le_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.LE, items[0], items[2])

    def gt_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.GT, items[0], items[2])

    def ge_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.GE, items[0], items[2])

    def and_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.AND, items[0], items[2])

    def or_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.OR, items[0], items[2])

    def lshift_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.LSHIFT, items[0], items[2])

    def rshift_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.RSHIFT, items[0], items[2])

    def bit_and_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.BIT_AND, items[0], items[2])

    def bit_or_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.BIT_OR, items[0], items[2])

    def bit_xor_op(self, items):
        return cadl_ast.BinaryExpr(cadl_ast.BinaryOp.BIT_XOR, items[0], items[2])

    # Expressions - Unary operations
    def neg_op(self, items):
        return cadl_ast.UnaryExpr(cadl_ast.UnaryOp.NEG, items[1])  # Skip OP_MINUS token

    def not_op(self, items):
        return cadl_ast.UnaryExpr(cadl_ast.UnaryOp.NOT, items[1])  # Skip OP_NOT token

    def bit_not_op(self, items):
        return cadl_ast.UnaryExpr(cadl_ast.UnaryOp.BIT_NOT, items[1])  # Skip OP_BIT_NOT token

    def signed_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return cadl_ast.UnaryExpr(cadl_ast.UnaryOp.SIGNED_CAST, items[2])

    def unsigned_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return cadl_ast.UnaryExpr(cadl_ast.UnaryOp.UNSIGNED_CAST, items[2])

    def f32_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return cadl_ast.UnaryExpr(cadl_ast.UnaryOp.F32_CAST, items[2])

    def f64_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return cadl_ast.UnaryExpr(cadl_ast.UnaryOp.F64_CAST, items[2])

    def int_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return cadl_ast.UnaryExpr(cadl_ast.UnaryOp.INT_CAST, items[2])

    def uint_cast(self, items):
        # items = [CAST_TOKEN, LPAREN, expr, RPAREN]
        return cadl_ast.UnaryExpr(cadl_ast.UnaryOp.UINT_CAST, items[2])

    # Complex expressions
    def call_expr(self, items):
        name = str(items[0])  # IDENTIFIER
        # items[1] is LPAREN, items[2] is expr_list (optional), items[3] is RPAREN
        args = items[2] if len(items) > 3 and items[2] else []
        return cadl_ast.CallExpr(name, args)

    def index_expr(self, items):
        expr = items[0]
        indices = items[2]  # Skip LBRACKET token, get expr_list
        return cadl_ast.IndexExpr(expr, indices)

    def slice_expr(self, items):
        expr = items[0]
        start = items[2]  # Skip LBRACKET token
        end = items[4]  # Skip COLON token
        return cadl_ast.SliceExpr(expr, start, end)

    def range_slice_expr(self, items):
        # Grammar: postfix_expr LBRACKET expr OP_PLUS COLON expr? RBRACKET
        expr = items[0]
        start = items[2]  # Skip LBRACKET token
        # items[3] is OP_PLUS, items[4] is COLON
        length = items[5] if len(items) > 6 and items[5] is not None else None
        return cadl_ast.RangeSliceExpr(expr, start, length)

    def paren_expr(self, items):
        # items = [LPAREN, expr_list, RPAREN]
        expr_list = items[1] if len(items) > 1 else []
        if isinstance(expr_list, list):
            if len(expr_list) == 1:
                return expr_list[0]
            return cadl_ast.TupleExpr(expr_list)
        else:
            return expr_list

    def if_expr(self, items):
        # items = [KW_IF, condition, LBRACE, then_branch, RBRACE, KW_ELSE, LBRACE, else_branch, RBRACE]
        condition = items[1]  # Skip KW_IF
        then_branch = items[3]  # Skip LBRACE
        else_branch = items[7]  # Skip KW_ELSE, LBRACE
        return cadl_ast.IfExpr(condition, then_branch, else_branch)

    def select_expr(self, items):
        # Grammar: KW_SEL LBRACE sel_arm+ RBRACE
        # items[0] = KW_SEL token
        # items[1] = LBRACE token
        # items[2:-1] = sel_arm tuples (condition, value)
        # items[-1] = RBRACE token

        # Filter out tokens, keep only sel_arm tuples
        arms_raw = [item for item in items if isinstance(item, tuple)]

        if len(arms_raw) == 0:
            raise ValueError("select expression must have at least one arm")

        if len(arms_raw) == 1:
            # Only one arm - use it as default with no conditional arms
            arms = []
            default = arms_raw[0][1]  # Value part of the only arm
        else:
            # Multiple arms - all but last are conditional, last is default
            arms = arms_raw[:-1]  # List of (condition, value) tuples
            default = arms_raw[-1][1]  # Value part of last arm (ignore its condition)

        return cadl_ast.SelectExpr(arms, default)

    def aggregate_expr(self, items):
        # Grammar: LBRACE expr_list RBRACE
        expr_list = items[1]  # Skip LBRACE, get expr_list, skip RBRACE
        return cadl_ast.AggregateExpr(expr_list)

    def sel_arm(self, items):
        # Grammar: expr COLON expr COMMA
        # items[0] = condition expr
        # items[1] = COLON token
        # items[2] = value expr
        # items[3] = COMMA token
        return (items[0], items[2])

    def expr_list(self, items):
        # Filter out COMMA tokens, keep only expressions
        return [item for item in items if not _is_token(item, "COMMA")]

    # Statements
    def expr_stmt(self, items):
        return cadl_ast.ExprStmt(items[0])

    def assign_stmt(self, items):
        # Grammar: KW_LET? expr (COLON data_type)? ASSIGN expr SEMICOLON
        is_let = any(_is_token(item, "KW_LET") for item in items)

        # Find indices of key tokens
        assign_idx = next(
            i for i, item in enumerate(items) if _is_token(item, "ASSIGN")
        )

        if is_let:
            lhs = items[1]  # expr after KW_LET
        else:
            lhs = items[0]  # first expr

        # RHS is the expression after ASSIGN (before SEMICOLON)
        rhs = items[assign_idx + 1]

        # Type annotation is between COLON and ASSIGN if present
        type_annotation = None
        colon_idx = next(
            (i for i, item in enumerate(items) if _is_token(item, "COLON")), None
        )
        if colon_idx is not None and colon_idx < assign_idx:
            type_annotation = items[colon_idx + 1]

        # Type checking rule: let statements must have explicit type annotations
        if is_let and type_annotation is None:
            raise ValueError(
                "'let' statements require explicit type annotation. Use 'let var: type = value;'"
            )

        # Validate burst operation constraints
        self._validate_burst_operation(lhs, rhs)

        return cadl_ast.AssignStmt(is_let, lhs, rhs, type_annotation)

    def return_stmt(self, items):
        # items = [KW_RETURN, expr_list, SEMICOLON]
        expr_list = items[1] if len(items) > 1 else []
        return cadl_ast.ReturnStmt(expr_list if isinstance(expr_list, list) else [expr_list])

    def guard_stmt(self, items):
        condition = items[0]
        stmt = items[1]
        return cadl_ast.GuardStmt(condition, stmt)

    def if_stmt(self, items):
        condition = next(
            item
            for item in items
            if isinstance(item, cadl_ast.Expr)
        )
        bodies = [item for item in items if isinstance(item, list)]
        then_body = bodies[0] if bodies else []
        else_body = bodies[1] if len(bodies) > 1 else None
        return cadl_ast.IfStmt(condition, then_body, else_body)

    def do_while_stmt(self, items):
        # Grammar: KW_WITH with_binding* KW_DO body KW_WHILE expr SEMICOLON
        bindings = []

        # Find KW_DO to separate bindings from body
        do_idx = next(i for i, item in enumerate(items) if _is_token(item, "KW_DO"))
        while_idx = next(
            i for i, item in enumerate(items) if _is_token(item, "KW_WHILE")
        )

        # Extract bindings (between KW_WITH and KW_DO)
        for i in range(1, do_idx):  # Skip KW_WITH
            if isinstance(items[i], cadl_ast.WithBinding):
                bindings.append(items[i])

        # Extract body (between KW_DO and KW_WHILE)
        body = items[do_idx + 1]  # Should be the transformed body

        # Extract condition (between KW_WHILE and SEMICOLON)
        condition = items[while_idx + 1]

        return cadl_ast.DoWhileStmt(bindings, body, condition)

    def directive_stmt(self, items):
        # Grammar: LBRACKET_BRACKET IDENTIFIER (LPAREN expr RPAREN)? RBRACKET_BRACKET
        # items[0] = [[, items[1] = IDENTIFIER, items[2] = (, items[3] = expr, items[4] = ), items[5] = ]]
        name = str(items[1])  # IDENTIFIER is at index 1
        expr = (
            items[3] if len(items) > 4 else None
        )  # expr is at index 3 if present (after LPAREN)
        return cadl_ast.DirectiveStmt(name, expr)

    def spawn_stmt(self, items):
        return cadl_ast.SpawnStmt(items)

    def static_stmt(self, items):
        return cadl_ast.StaticStmt(items[0])

    def thread_stmt(self, items):
        return ThreadStmt(items[0])

    # Body
    def empty_body(self, items):
        return None

    def block_body(self, items):
        # Filter out LBRACE and RBRACE tokens, return only statements
        return [item for item in items if isinstance(item, cadl_ast.Stmt)]

    def stmt_block(self, items):
        return [item for item in items if isinstance(item, cadl_ast.Stmt)]

    # Static and thread definitions
    def static(self, items):
        # Expected structure: attribute* KW_STATIC IDENTIFIER COLON data_type (ASSIGN expr)? SEMICOLON
        # Extract attributes from beginning
        attrs = []
        idx = 0
        while idx < len(items) and isinstance(items[idx], tuple):
            attrs.append(items[idx])
            idx += 1

        # Now parse the rest: KW_STATIC IDENTIFIER COLON data_type ...
        # items[idx] is KW_STATIC
        name = str(items[idx + 1])  # IDENTIFIER token
        type_info = items[idx + 3]  # data_type (already transformed)
        expr = (
            items[idx + 5]
            if len(items) > idx + 5 and items[idx + 5] is not None
            else None
        )  # expr (already transformed)

        # Convert attributes to dict
        attr_dict = dict(attrs) if attrs else {}

        return cadl_ast.Static(name, type_info, expr, attr_dict)

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
            if (
                isinstance(item, list)
                and len(item) > 0
                and all(isinstance(arg, cadl_ast.FnArg) for arg in item)
            ):
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

        flow_attrs = cadl_ast.FlowAttributes.from_tuples(attrs)
        input_pairs = [(arg.id, arg.ty.to_basic()) for arg in inputs]

        return cadl_ast.Flow(name, cadl_ast.FlowKind.DEFAULT, input_pairs, flow_attrs, body)

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
            if (
                isinstance(item, list)
                and len(item) > 0
                and all(isinstance(arg, cadl_ast.FnArg) for arg in item)
            ):
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

        flow_attrs = cadl_ast.FlowAttributes.from_tuples(attrs)
        input_pairs = [(arg.id, arg.ty.to_basic()) for arg in inputs]

        return cadl_ast.Flow(name, cadl_ast.FlowKind.RTYPE, input_pairs, flow_attrs, body)

    # Regfile definition
    def regfile(self, items):
        name = str(items[1])  # Skip KW_REGFILE
        width = int(str(items[3]))  # Skip LPAREN
        depth = int(str(items[5]))  # Skip COMMA
        return cadl_ast.Regfile(name, width, depth)

    # Register definition
    def register(self, items):
        attrs = []
        idx = 0
        while idx < len(items) and isinstance(items[idx], tuple):
            attrs.append(items[idx])
            idx += 1

        # Grammar: attribute* KW_REGISTER IDENTIFIER COLON data_type SEMICOLON
        name = str(items[idx + 1])
        type_info = items[idx + 3]
        attr_dict = dict(attrs) if attrs else {}
        register = cadl_ast.Register(name, type_info, attr_dict)
        if not register.is_csr:
            raise NotImplementedError(
                "Only #[csr_address(...)] register declarations are supported for now"
            )
        return register

    # Processor parts
    def proc_part(self, items):
        part = items[0]
        if isinstance(part, cadl_ast.Regfile):
            return cadl_ast.RegfilePart(part)
        elif isinstance(part, cadl_ast.Flow):
            return cadl_ast.FlowPart(part)
        elif isinstance(part, cadl_ast.Static):
            return cadl_ast.StaticPart(part)
        elif isinstance(part, cadl_ast.Register):
            return cadl_ast.RegisterPart(part)
        return part

    # Main processor
    def proc(self, items):
        return cadl_ast.Proc.from_parts(items)

    def simple_attr(self, items):
        # items = [HASH, LBRACKET, IDENTIFIER, RBRACKET]
        attr_name = items[2].value  # IDENTIFIER value
        return (attr_name, None)

    def param_attr(self, items):
        # items = [HASH, LBRACKET, IDENTIFIER, LPAREN, attr_expr, RPAREN, RBRACKET]
        attr_name = items[2].value  # IDENTIFIER value
        attr_expr = items[4]  # attr_expr (could be expr or array_literal)
        return (attr_name, attr_expr)

    def array_literal(self, items):
        # Grammar: LBRACKET attr_expr_list RBRACKET
        # items[0] = LBRACKET token
        # items[1] = attr_expr_list (list of expressions)
        # items[2] = RBRACKET token
        expr_list = items[1] if len(items) > 1 else []
        return cadl_ast.ArrayLiteralExpr(
            expr_list if isinstance(expr_list, list) else [expr_list]
        )

    def attr_expr_list(self, items):
        # Filter out COMMA tokens, keep only expressions
        return [item for item in items if not _is_token(item, "COMMA")]

    def start(self, items):
        return items[0]


class CADLParser:
    """Main CADL parser class"""

    def __init__(self):
        """Initialize the parser with Lark grammar"""
        grammar_path = Path(__file__).parent / "grammar.lark"
        with open(grammar_path, "r") as f:
            grammar = f.read()

        self.parser = Lark(
            grammar,
            parser="lalr",  # Using LALR parser for better performance with transformers
            start="start",
        )
        self.transformer = CADLTransformer()

    def parse(self, source: str, filename: Optional[str] = None) -> cadl_ast.Proc:
        """Parse CADL source code into AST"""
        try:
            parse_tree = self.parser.parse(source)
            result = self.transformer.transform(parse_tree)
            return result
        except (UnexpectedToken, UnexpectedCharacters, ParseError) as e:
            # Convert Lark errors to pretty CADL errors (no chaining to hide traceback)
            raise format_lark_error(e, source, filename, self.parser.parse)
        except VisitError as e:
            if isinstance(e.orig_exc, (ValueError, NotImplementedError)):
                raise CADLParseError(
                    str(e.orig_exc), 0, 0, filename, source.splitlines()
                )
            raise CADLParseError(
                f"Internal error: {e}", 1, 1, filename, source.splitlines()
            )
        except (ValueError, NotImplementedError) as e:
            raise CADLParseError(str(e), 1, 1, filename, source.splitlines())
        except Exception as e:
            # Handle transformer errors and other issues
            raise CADLParseError(
                f"Internal error: {e}", 1, 1, filename, source.splitlines()
            )


# Global parser instance
_parser = None


def get_parser() -> CADLParser:
    """Get or create global parser instance"""
    global _parser
    if _parser is None:
        _parser = CADLParser()
    return _parser


def parse_proc(source: str, filename: Optional[str] = None) -> cadl_ast.Proc:
    """Parse a CADL processor from source code

    Args:
        source: CADL source code string
        filename: Optional filename for error reporting

    Returns:
        Proc: Parsed processor AST

    Raises:
        CADLParseError: On parse errors (with pretty formatting)
    """
    parser = get_parser()
    try:
        return parser.parse(source, filename)
    except CADLParseError:
        # Re-raise our pretty errors as-is
        raise
    except Exception as e:
        # Wrap any other errors
        raise CADLParseError(
            f"Unexpected error: {e}", 1, 1, filename, source.splitlines()
        )
