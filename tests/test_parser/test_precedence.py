#!/usr/bin/env python3
"""
Test to verify that operator precedence matches LALRPOP implementation
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestPrecedence:
    """Test operator precedence"""

    def test_arithmetic_precedence(self):
        """Test that multiplication has higher precedence than addition"""
        source = """
        flow test_precedence(a: u32, b: u32, c: u32) {
            return (a + b * c);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # The parsed expression should be: a + (b * c), not (a + b) * c
        # This means the return statement contains a BinaryExpr with ADD
        # where the right operand is a BinaryExpr with MUL
        return_stmt = flow.body[0]
        assert isinstance(return_stmt, ReturnStmt)
        assert len(return_stmt.exprs) == 1
        
        # The actual expression should be ADD with right operand being MUL
        expr = return_stmt.exprs[0]
        assert isinstance(expr, BinaryExpr)
        assert expr.op == BinaryOp.ADD
        
        # Right operand should be MUL operation
        assert isinstance(expr.right, BinaryExpr)
        assert expr.right.op == BinaryOp.MUL

    def test_bitwise_precedence(self):
        """Test that bitwise AND has higher precedence than bitwise OR"""
        source = """
        flow test_bitwise(a: u32, b: u32, c: u32) {
            return (a | b & c);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # The parsed expression should be: a | (b & c), not (a | b) & c
        return_stmt = flow.body[0]
        expr = return_stmt.exprs[0]
        
        assert isinstance(expr, BinaryExpr)
        assert expr.op == BinaryOp.BIT_OR
        
        # Right operand should be BIT_AND operation
        assert isinstance(expr.right, BinaryExpr)
        assert expr.right.op == BinaryOp.BIT_AND

    def test_shift_vs_arithmetic_precedence(self):
        """Test that addition has higher precedence than shifts"""
        source = """
        flow test_shift(a: u32, b: u32, c: u32) {
            return (a << b + c);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # The parsed expression should be: a << (b + c), not (a << b) + c
        return_stmt = flow.body[0]
        expr = return_stmt.exprs[0]
        
        assert isinstance(expr, BinaryExpr)
        assert expr.op == BinaryOp.LSHIFT
        
        # Right operand should be ADD operation
        assert isinstance(expr.right, BinaryExpr)
        assert expr.right.op == BinaryOp.ADD

    def test_comparison_vs_bitwise_precedence(self):
        """Test that bitwise operations have higher precedence than comparisons"""
        source = """
        flow test_comparison(a: u32, b: u32, c: u32) {
            return (a & b == c);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # The parsed expression should be: (a & b) == c, not a & (b == c)
        return_stmt = flow.body[0]
        expr = return_stmt.exprs[0]
        
        assert isinstance(expr, BinaryExpr)
        assert expr.op == BinaryOp.EQ
        
        # Left operand should be BIT_AND operation
        assert isinstance(expr.left, BinaryExpr)
        assert expr.left.op == BinaryOp.BIT_AND


if __name__ == "__main__":
    pytest.main([__file__])