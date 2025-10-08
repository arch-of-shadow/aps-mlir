"""
Test cases for CADL parser

These tests mirror the test cases from the Rust implementation.
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestLiterals:
    """Test literal parsing"""

    def test_number_literals(self):
        """Test various number format parsing"""
        test_cases = [
            "1231231",
            "12345678901234566789",
            "0x1234",
            "5'b101010",
            "3'o123",
            "15'd123",
            "8'hFF",
        ]
        
        # For now, we'll test basic parsing doesn't crash
        # TODO: Add more specific assertions based on expected AST structure
        for case in test_cases:
            source = f"static x: u32 = {case};"
            try:
                ast = parse_proc(source)
                assert ast is not None
            except Exception as e:
                pytest.fail(f"Failed to parse number literal '{case}': {e}")


class TestExpressions:
    """Test expression parsing"""

    def test_basic_expressions(self):
        """Test basic expression parsing"""
        test_cases = [
            "mdfasdf",
            "(a,b,c)",
            "(a)",
            "xyz(xxx)",
            "a[asdaadasd]",
            "a[1]",
            "vec[1:3]",
        ]
        
        for case in test_cases:
            source = f"static x: u32; fn test() -> () {{ {case}; }}"
            try:
                ast = parse_proc(source)
                assert ast is not None
            except Exception as e:
                pytest.fail(f"Failed to parse expression '{case}': {e}")


class TestStatements:
    """Test statement parsing"""

    def test_assignment_statements(self):
        """Test assignment statement parsing"""
        test_cases = [
            "let x: u32 = 123123;",
            "x = 123123;",
            "dfasdfas;",
        ]
        
        for case in test_cases:
            source = f"fn test() -> () {{ {case} }}"
            try:
                ast = parse_proc(source)
                assert ast is not None
            except Exception as e:
                pytest.fail(f"Failed to parse statement '{case}': {e}")

    def test_do_while_loop(self):
        """Test comprehensive do-while loop parsing based on zyy.cadl examples"""
        source = """
        rtype loop_test(rs1: u5, rs2: u5, rd: u5) {
            let sum0: u32 = _irf[rs1];
            let i0: u32 = 0;
            let n0: u32 = _irf[rs2];
            with 
                i: u32 = (i0, i_)
                sum: u32 = (sum0, sum_)
                n: u32 = (n0, n_)
            do {
                let n_: u32 = n;
                let sum_: u32 = sum + 4;
                let i_: u32 = i + 1;
            } while (i_ < n);
            _irf[rd] = sum;
        }
        """
        try:
            ast = parse_proc(source)
            assert ast is not None
            assert len(ast.flows) == 1
            
            flow = list(ast.flows.values())[0]
            assert flow.name == "loop_test"
            assert flow.kind == FlowKind.RTYPE
            assert len(flow.inputs) == 3
            
            # Should have: 3 let assignments + 1 do-while loop + 1 irf assignment  
            assert len(flow.body) == 5
            
            # Check the do-while loop is parsed correctly
            do_while_stmt = flow.body[3]  # Fourth statement should be the loop
            assert isinstance(do_while_stmt, DoWhileStmt)
            
            # Check that we have 3 with bindings
            assert len(do_while_stmt.bindings) == 3
            
            # Verify binding names and types
            binding_names = [b.id for b in do_while_stmt.bindings]
            assert "i" in binding_names
            assert "sum" in binding_names  
            assert "n" in binding_names
            
            # Check that loop body has 3 assignments
            assert len(do_while_stmt.body) == 3
            
            # Verify while condition is a comparison
            assert isinstance(do_while_stmt.condition, BinaryExpr)
            assert do_while_stmt.condition.op == BinaryOp.LT
            
        except Exception as e:
            pytest.fail(f"Failed to parse complex do-while loop: {e}")

    def test_crc8_loop(self):
        """Test more complex loop with bitwise operations from crc8 example"""
        source = """
        rtype crc8(rs1: u5, rs2: u5, rd: u5) {
            let x0: u32 = _irf[rs1];
            let i0: u32 = 0;
            let n0: u32 = 8;
            with
                i: u32 = (i0, i_)
                x: u32 = (x0, x_)
                n: u32 = (n0, n_)
            do {
                let a: u32 = (x >> 1);
                let x_: u32 = a ^ (32'hEDB88320 & ~((x & 1) - 1));
                let i_: u32 = i + 1;
                let n_: u32 = n;
            } while (i < n);
            _irf[rd] = x;
        }
        """
        try:
            ast = parse_proc(source)
            assert ast is not None
            flow = list(ast.flows.values())[0]
            
            # Should have complex bitwise operations in the loop body
            do_while_stmt = flow.body[3]
            assert isinstance(do_while_stmt, DoWhileStmt)
            
            # Check that we have shift, XOR, AND, and NOT operations in the loop body
            loop_body = do_while_stmt.body
            assert len(loop_body) == 4
            
            # Second assignment should have XOR operation
            xor_assignment = loop_body[1]
            assert isinstance(xor_assignment, AssignStmt)
            assert isinstance(xor_assignment.rhs, BinaryExpr)
            assert xor_assignment.rhs.op == BinaryOp.BIT_XOR
            
        except Exception as e:
            pytest.fail(f"Failed to parse crc8 loop with bitwise operations: {e}")


class TestFlows:
    """Test flow parsing"""

    def test_basic_flow(self):
        """Test basic flow definition"""
        source = """
        flow add(a: u32, b: u32) {
            return (a + b);
        }
        """
        try:
            ast = parse_proc(source)
            assert ast is not None
            assert len(ast.flows) == 1
            flow = list(ast.flows.values())[0]
            assert flow.name == "add"
            assert flow.kind == FlowKind.DEFAULT
        except Exception as e:
            pytest.fail(f"Failed to parse flow: {e}")

    def test_rtype_flow(self):
        """Test rtype flow definition"""
        source = """
        rtype multiply(a: u32, b: u32) {
            return (a * b);
        }
        """
        try:
            ast = parse_proc(source)
            assert ast is not None
            assert len(ast.flows) == 1
            flow = list(ast.flows.values())[0]
            assert flow.name == "multiply"
            assert flow.kind == FlowKind.RTYPE
        except Exception as e:
            pytest.fail(f"Failed to parse rtype flow: {e}")


class TestRegfiles:
    """Test regfile parsing"""

    def test_regfile_definition(self):
        """Test regfile definition parsing"""
        source = "regfile rf(32, 16);"
        try:
            ast = parse_proc(source)
            assert ast is not None
            assert len(ast.regfiles) == 1
            regfile = list(ast.regfiles.values())[0]
            assert regfile.name == "rf"
            assert regfile.width == 32
            assert regfile.depth == 16
        except Exception as e:
            pytest.fail(f"Failed to parse regfile: {e}")


class TestFunctions:
    """Test function parsing"""

    def test_function_definition(self):
        """Test function definition parsing"""
        source = """
        fn factorial(n: u32) -> (u32) {
            return (if n <= 1 {1} else {n * factorial(n - 1)});
        }
        """
        try:
            ast = parse_proc(source)
            assert ast is not None
            assert len(ast.functions) == 1
            function = list(ast.functions.values())[0]
            assert function.name == "factorial"
            assert len(function.args) == 1
            assert function.args[0].id == "n"
        except Exception as e:
            pytest.fail(f"Failed to parse function: {e}")


if __name__ == "__main__":
    pytest.main([__file__])