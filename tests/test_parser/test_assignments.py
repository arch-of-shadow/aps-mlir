#!/usr/bin/env python3
"""
Test to verify that let assignments are parsed correctly
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestAssignments:
    """Test assignment statement parsing"""

    def test_simple_let_assignment(self):
        """Test basic let assignment with type annotation"""
        source = """
        rtype test_let(a: u32, b: u32) {
            let sum: u32 = a + b;
            return (sum);
        }
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "test_let"
        
        # Should have assignment and return statements
        assert len(flow.body) == 2
        
        # Check the assignment statement
        assign_stmt = flow.body[0]
        assert isinstance(assign_stmt, AssignStmt)
        assert assign_stmt.is_let == True
        
        # Check LHS is identifier
        assert isinstance(assign_stmt.lhs, IdentExpr)
        assert assign_stmt.lhs.name == "sum"
        
        # Check RHS is binary expression
        assert isinstance(assign_stmt.rhs, BinaryExpr)
        assert assign_stmt.rhs.op == BinaryOp.ADD
        
        # Should have type annotation
        assert assign_stmt.type_annotation is not None
        assert isinstance(assign_stmt.type_annotation, DataType_Single)
        assert isinstance(assign_stmt.type_annotation.basic_type, BasicType_ApUFixed)
        assert assign_stmt.type_annotation.basic_type.width == 32

    def test_let_assignment_with_type(self):
        """Test let assignment with explicit type annotation"""
        source = """
        rtype test_typed_let(x: u32, y: u32) {
            let result: u32 = x * y;
            return (result);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check the assignment statement
        assign_stmt = flow.body[0]
        assert isinstance(assign_stmt, AssignStmt)
        assert assign_stmt.is_let == True
        
        # Check LHS 
        assert isinstance(assign_stmt.lhs, IdentExpr)
        assert assign_stmt.lhs.name == "result"
        
        # Check RHS is multiplication
        assert isinstance(assign_stmt.rhs, BinaryExpr)
        assert assign_stmt.rhs.op == BinaryOp.MUL
        
        # Check type annotation exists
        assert assign_stmt.type_annotation is not None

    def test_assignment_without_let(self):
        """Test assignment without let keyword"""
        source = """
        rtype test_assign(a: u32) {
            let x: u32 = 10;
            x = a + 5;
            return (x);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have let assignment, regular assignment, and return
        assert len(flow.body) == 3
        
        # Check first is let assignment
        let_stmt = flow.body[0]
        assert isinstance(let_stmt, AssignStmt)
        assert let_stmt.is_let == True
        
        # Check second is regular assignment  
        assign_stmt = flow.body[1]
        assert isinstance(assign_stmt, AssignStmt)
        assert assign_stmt.is_let == False
        
        # Check LHS and RHS
        assert isinstance(assign_stmt.lhs, IdentExpr)
        assert assign_stmt.lhs.name == "x"
        assert isinstance(assign_stmt.rhs, BinaryExpr)
        assert assign_stmt.rhs.op == BinaryOp.ADD

    def test_multiple_assignments(self):
        """Test multiple assignment statements"""
        source = """
        rtype test_multiple(a: u32, b: u32) {
            let x: u32 = a;
            let y: u32 = b;
            let sum: u32 = x + y;
            let product: u32 = x * y;
            return (sum + product);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have 4 assignments + 1 return
        assert len(flow.body) == 5
        
        # Check all assignments are let statements
        for i in range(4):
            stmt = flow.body[i]
            assert isinstance(stmt, AssignStmt)
            assert stmt.is_let == True
        
        # Check variable names
        expected_names = ["x", "y", "sum", "product"]
        for i, expected_name in enumerate(expected_names):
            stmt = flow.body[i]
            assert isinstance(stmt.lhs, IdentExpr)
            assert stmt.lhs.name == expected_name

    def test_assignment_with_literals(self):
        """Test assignments with different types of literals"""
        source = """
        rtype test_literals() {
            let num: u32 = 42;
            let hex: u32 = 0xFF;
            let binary: u8 = 5'b10101;
            let decimal_width: u16 = 10'd255;
            return (num + hex);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check assignments with literals
        assignments = [stmt for stmt in flow.body if isinstance(stmt, AssignStmt)]
        assert len(assignments) == 4
        
        # Check first assignment (simple decimal)
        assert isinstance(assignments[0].rhs, LitExpr)
        
        # Check second assignment (hex)
        assert isinstance(assignments[1].rhs, LitExpr)
        
        # Check third assignment (binary with width)
        assert isinstance(assignments[2].rhs, LitExpr)
        
        # Check fourth assignment (decimal with width)
        assert isinstance(assignments[3].rhs, LitExpr)

    def test_assignment_with_function_calls(self):
        """Test assignment with function call on RHS"""
        source = """
        fn helper(x: u32) -> (u32) {
            return (x * 2);
        }
        
        rtype test_call_assign(input: u32) {
            let doubled: u32 = helper(input);
            let tripled: u32 = helper(input) + input;
            return (doubled + tripled);
        }
        """
        
        ast = parse_proc(source)
        
        # Should have function and flow
        assert len(ast.functions) == 1
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        
        # Check first assignment has function call
        assign1 = flow.body[0]
        assert isinstance(assign1, AssignStmt)
        assert isinstance(assign1.rhs, CallExpr)
        assert assign1.rhs.name == "helper"
        
        # Check second assignment has expression with function call
        assign2 = flow.body[1]
        assert isinstance(assign2, AssignStmt)
        assert isinstance(assign2.rhs, BinaryExpr)
        assert isinstance(assign2.rhs.left, CallExpr)

    def test_assignment_with_complex_expressions(self):
        """Test assignments with complex arithmetic expressions"""
        source = """
        rtype test_complex(a: u32, b: u32, c: u32) {
            let expr1: u32 = a + b * c;
            let expr2: u32 = (a + b) * c;
            let expr3: u32 = a << 2 + b;
            let expr4: u32 = a & b | c;
            return (expr1 + expr2);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        assignments = [stmt for stmt in flow.body if isinstance(stmt, AssignStmt)]
        assert len(assignments) == 4
        
        # Check first: a + b * c (should be a + (b * c))
        expr1 = assignments[0].rhs
        assert isinstance(expr1, BinaryExpr)
        assert expr1.op == BinaryOp.ADD
        assert isinstance(expr1.right, BinaryExpr)
        assert expr1.right.op == BinaryOp.MUL
        
        # Check second: (a + b) * c  
        expr2 = assignments[1].rhs
        assert isinstance(expr2, BinaryExpr)
        assert expr2.op == BinaryOp.MUL
        
        # Check third: shift and addition precedence
        expr3 = assignments[2].rhs
        assert isinstance(expr3, BinaryExpr)
        assert expr3.op == BinaryOp.LSHIFT
        assert isinstance(expr3.right, BinaryExpr)
        assert expr3.right.op == BinaryOp.ADD
        
        # Check fourth: bitwise operations
        expr4 = assignments[3].rhs
        assert isinstance(expr4, BinaryExpr)
        assert expr4.op == BinaryOp.BIT_OR

    def test_assignment_with_type_casts(self):
        """Test assignments with type casting operations"""
        source = """
        rtype test_casts(signed_val: i32) {
            let unsigned_val: u32 = $unsigned(signed_val);
            let float_val: u32 = $uint($f32(signed_val));
            return (unsigned_val + float_val);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        assignments = [stmt for stmt in flow.body if isinstance(stmt, AssignStmt)]
        assert len(assignments) == 2
        
        # Check first assignment has unsigned cast
        assert isinstance(assignments[0].rhs, UnaryExpr)
        assert assignments[0].rhs.op == UnaryOp.UNSIGNED_CAST
        
        # Check second assignment has nested casts
        assert isinstance(assignments[1].rhs, UnaryExpr)
        assert assignments[1].rhs.op == UnaryOp.UINT_CAST
        assert isinstance(assignments[1].rhs.operand, UnaryExpr)
        assert assignments[1].rhs.operand.op == UnaryOp.F32_CAST


if __name__ == "__main__":
    pytest.main([__file__])