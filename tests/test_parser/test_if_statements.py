#!/usr/bin/env python3
"""
Test to verify that if statements are parsed correctly
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestIfStatements:
    """Test if statement parsing"""

    def test_simple_if_expression(self):
        """Test basic if expression in return statement"""
        source = """
        rtype test_if(a: u32, b: u32) {
            return (if a > b {a} else {b});
        }
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "test_if"
        assert flow.kind == FlowKind.RTYPE
        
        # Check that body exists and has statements
        assert flow.body is not None
        assert len(flow.body) == 1
        
        # Check the return statement contains an if expression
        return_stmt = flow.body[0]
        assert isinstance(return_stmt, ReturnStmt)
        assert len(return_stmt.exprs) == 1
        
        # The if expression should be parsed correctly
        if_expr = return_stmt.exprs[0]
        assert isinstance(if_expr, IfExpr)
        
        # Check condition is a comparison
        assert isinstance(if_expr.condition, BinaryExpr)
        assert if_expr.condition.op == BinaryOp.GT
        
        # Check branches are identifiers
        assert isinstance(if_expr.then_branch, IdentExpr)
        assert if_expr.then_branch.name == "a"
        assert isinstance(if_expr.else_branch, IdentExpr)
        assert if_expr.else_branch.name == "b"

    def test_if_expression_in_assignment(self):
        """Test if expression used in let assignment"""
        source = """
        rtype conditional_assign(x: u32, y: u32) {
            let max_val: u32 = if x >= y {x} else {y};
            return (max_val);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have assignment and return statements
        assert len(flow.body) == 2
        
        # Check the assignment statement
        assign_stmt = flow.body[0]
        assert isinstance(assign_stmt, AssignStmt)
        assert assign_stmt.is_let == True
        
        # The RHS should be an if expression
        assert isinstance(assign_stmt.rhs, IfExpr)
        
        # Check condition is greater-equal comparison
        if_expr = assign_stmt.rhs
        assert isinstance(if_expr.condition, BinaryExpr)
        assert if_expr.condition.op == BinaryOp.GE

    def test_nested_if_expressions(self):
        """Test nested if expressions"""
        source = """
        rtype nested_if(a: u32, b: u32, c: u32) {
            let result: u32 = if a > b {
                if a > c {a} else {c}
            } else {
                if b > c {b} else {c}
            };
            return (result);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check assignment statement has nested if
        assign_stmt = flow.body[0]
        outer_if = assign_stmt.rhs
        assert isinstance(outer_if, IfExpr)
        
        # Both branches should be if expressions  
        assert isinstance(outer_if.then_branch, IfExpr)
        assert isinstance(outer_if.else_branch, IfExpr)
        
        # Check inner if expressions have proper structure
        then_if = outer_if.then_branch
        assert isinstance(then_if.condition, BinaryExpr)
        assert then_if.condition.op == BinaryOp.GT

    def test_if_with_complex_conditions(self):
        """Test if expressions with complex boolean conditions"""
        source = """
        rtype complex_conditions(x: u32, y: u32, z: u32) {
            let result1: u32 = if (x > 0) && (y < 100) {x + y} else {0};
            let result2: u32 = if (x == 0) || (z >= 50) {z * 2} else {x};
            return (result1 + result2);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have two assignments and return
        assert len(flow.body) == 3
        
        # Check first if expression with AND condition
        assign1 = flow.body[0]
        if_expr1 = assign1.rhs
        assert isinstance(if_expr1, IfExpr)
        assert isinstance(if_expr1.condition, BinaryExpr)
        assert if_expr1.condition.op == BinaryOp.AND
        
        # Check second if expression with OR condition  
        assign2 = flow.body[1]
        if_expr2 = assign2.rhs
        assert isinstance(if_expr2, IfExpr)
        assert isinstance(if_expr2.condition, BinaryExpr)
        assert if_expr2.condition.op == BinaryOp.OR

    def test_if_with_arithmetic_expressions(self):
        """Test if expressions with arithmetic in branches"""
        source = """
        rtype arithmetic_if(a: u32, b: u32) {
            let sum: u32 = if a > 10 {a * 2 + b} else {a + b * 3};
            let diff: u32 = if sum > 50 {sum - 25} else {sum + 10};
            return (diff);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check first if expression has arithmetic in branches
        assign1 = flow.body[0]
        if_expr1 = assign1.rhs
        
        # Then branch: a * 2 + b
        then_branch = if_expr1.then_branch
        assert isinstance(then_branch, BinaryExpr)
        assert then_branch.op == BinaryOp.ADD
        assert isinstance(then_branch.left, BinaryExpr)  # a * 2
        assert then_branch.left.op == BinaryOp.MUL
        
        # Else branch: a + b * 3  
        else_branch = if_expr1.else_branch
        assert isinstance(else_branch, BinaryExpr)
        assert else_branch.op == BinaryOp.ADD
        assert isinstance(else_branch.right, BinaryExpr)  # b * 3
        assert else_branch.right.op == BinaryOp.MUL

    def test_if_with_literals_and_casts(self):
        """Test if expressions with literals and type casts"""
        source = """
        rtype if_with_casts(value: i32) {
            let unsigned_val: u32 = if value >= 0 {$unsigned(value)} else {0};
            let comparison: u32 = if unsigned_val > 32'd100 {1} else {0};
            return (comparison);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check first if expression has cast in then branch
        assign1 = flow.body[0]
        if_expr1 = assign1.rhs
        
        # Then branch should be unsigned cast
        then_branch = if_expr1.then_branch
        assert isinstance(then_branch, UnaryExpr)
        assert then_branch.op == UnaryOp.UNSIGNED_CAST
        
        # Else branch should be literal 0
        else_branch = if_expr1.else_branch
        assert isinstance(else_branch, LitExpr)
        
        # Check second if expression uses width-specified literal
        assign2 = flow.body[1]
        if_expr2 = assign2.rhs
        condition = if_expr2.condition
        assert isinstance(condition, BinaryExpr)
        assert isinstance(condition.right, LitExpr)  # 32'd100

    def test_chained_if_expressions(self):
        """Test chained/cascaded if expressions (else-if pattern)"""
        source = """
        rtype chained_if(score: u32) {
            let grade: u32 = if score >= 90 {
                4  // A
            } else {
                if score >= 80 {
                    3  // B  
                } else {
                    if score >= 70 {
                        2  // C
                    } else {
                        1  // F
                    }
                }
            };
            return (grade);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check the chained structure
        assign_stmt = flow.body[0]
        outer_if = assign_stmt.rhs
        assert isinstance(outer_if, IfExpr)
        
        # Then branch should be literal 4
        assert isinstance(outer_if.then_branch, LitExpr)
        
        # Else branch should be another if expression
        middle_if = outer_if.else_branch
        assert isinstance(middle_if, IfExpr)
        
        # Continue checking the chain
        inner_if = middle_if.else_branch
        assert isinstance(inner_if, IfExpr)
        
        # Final else should be literal 1
        assert isinstance(inner_if.else_branch, LitExpr)

    def test_if_with_function_calls(self):
        """Test if expressions containing function calls"""
        source = """
        fn helper_func(x: u32) -> (u32) {
            return (x * 2);
        }
        
        rtype if_with_calls(input: u32) {
            let result: u32 = if input > 5 {helper_func(input)} else {input};
            return (result);
        }
        """
        
        ast = parse_proc(source)
        
        # Should have both function and rtype flow
        assert len(ast.functions) == 1
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assign_stmt = flow.body[0]
        if_expr = assign_stmt.rhs
        
        # Then branch should be function call
        then_branch = if_expr.then_branch  
        assert isinstance(then_branch, CallExpr)
        assert then_branch.name == "helper_func"
        assert len(then_branch.args) == 1


if __name__ == "__main__":
    pytest.main([__file__])