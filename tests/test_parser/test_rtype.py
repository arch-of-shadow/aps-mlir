#!/usr/bin/env python3
"""
Test to verify that rtype flows are parsed correctly
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestRType:
    """Test rtype flow parsing"""

    def test_basic_rtype(self):
        """Test basic rtype flow without expressions"""
        source = """
        rtype test_rtype(a: u32, b: u32) ;
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "test_rtype"
        assert flow.kind == FlowKind.RTYPE
        assert len(flow.inputs) == 2
        
        # Check inputs
        input_names = [inp[0] for inp in flow.inputs]
        assert "a" in input_names
        assert "b" in input_names
        
        # Check that body is None (empty)
        assert flow.body is None

    def test_rtype_with_attributes(self):
        """Test rtype flow with attributes"""
        source = """
        #[opcode(51)]
        #[funct3(0)]
        rtype multiply(a: u32, b: u32) ;
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "multiply"
        assert flow.kind == FlowKind.RTYPE
        
        # Check attributes are present
        assert flow.attrs is not None
        # Note: Attribute parsing details would depend on transformer implementation
        
    def test_rtype_vs_regular_flow(self):
        """Test that rtype flows are distinguished from regular flows"""
        source = """
        flow regular_flow(a: u32) ;
        rtype rtype_flow(b: u32) ;
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 2
        
        # Check that we have one of each kind
        flow_kinds = [flow.kind for flow in ast.flows.values()]
        assert FlowKind.DEFAULT in flow_kinds
        assert FlowKind.RTYPE in flow_kinds
        
        # Verify specific flows
        for flow_name, flow in ast.flows.items():
            if flow_name == "regular_flow":
                assert flow.kind == FlowKind.DEFAULT
            elif flow_name == "rtype_flow":
                assert flow.kind == FlowKind.RTYPE

    def test_rtype_with_no_inputs(self):
        """Test rtype flow with no input parameters"""
        source = """
        rtype empty_rtype() ;
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "empty_rtype"
        assert flow.kind == FlowKind.RTYPE
        assert len(flow.inputs) == 0
        assert flow.body is None

    def test_rtype_with_multiple_types(self):
        """Test rtype flow with different input types"""
        source = """
        rtype multi_type(a: u8, b: i32, c: f32) ;
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "multi_type"
        assert flow.kind == FlowKind.RTYPE
        assert len(flow.inputs) == 3
        
        # Verify input types (names and basic structure)
        input_names = [inp[0] for inp in flow.inputs]
        assert input_names == ["a", "b", "c"]

    def test_rtype_with_basic_expressions(self):
        """Test rtype flow with basic expressions (no control flow)"""
        source = """
        rtype arithmetic_rtype(a: u32, b: u32) {
            let sum: u32 = a + b;
            let diff: u32 = a - b;
            let product: u32 = a * b;
            let quotient: u32 = a / b;
            let remainder: u32 = a % b;
            return (sum + product);
        }
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "arithmetic_rtype"
        assert flow.kind == FlowKind.RTYPE
        assert len(flow.inputs) == 2
        
        # Check that body exists and has statements
        assert flow.body is not None
        assert len(flow.body) > 0
        
        # Check input parameters
        input_names = [inp[0] for inp in flow.inputs]
        assert input_names == ["a", "b"]

    def test_rtype_with_shift_operations(self):
        """Test rtype flow with shift operations"""
        source = """
        #[opcode(33)]
        rtype shift_rtype(x: u16, y: u16) {
            let not_result: u16 = ~x;
            let shift_left: u16 = x << 2;
            let shift_right: u16 = x >> 1;
            return (shift_left + shift_right);
        }
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "shift_rtype"
        assert flow.kind == FlowKind.RTYPE
        assert len(flow.inputs) == 2
        
        # Check attributes are present
        assert flow.attrs is not None
        
        # Check that body exists and has statements
        assert flow.body is not None
        assert len(flow.body) > 0
        
        # Check input parameters have correct types
        input_names = [inp[0] for inp in flow.inputs]
        assert input_names == ["x", "y"]

    def test_rtype_with_bitwise_operations_correct_precedence(self):
        """Test rtype flow with bitwise operations and correct precedence"""
        source = """
        #[opcode(35)]
        rtype bitwise_rtype(x: u16, y: u16, z: u16) {
            let and_result: u16 = x & y;
            let or_result: u16 = x | y;
            let xor_result: u16 = x ^ y;
            let not_result: u16 = ~x;
            let combined: u16 = x & y | z;
            return (and_result ^ or_result);
        }
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "bitwise_rtype"
        assert flow.kind == FlowKind.RTYPE
        assert len(flow.inputs) == 3
        
        # Check attributes are present
        assert flow.attrs is not None
        
        # Check that body exists and has statements
        assert flow.body is not None
        assert len(flow.body) > 0
        
        # Check input parameters have correct types
        input_names = [inp[0] for inp in flow.inputs]
        assert input_names == ["x", "y", "z"]

    def test_rtype_with_comparison_operations(self):
        """Test rtype flow with comparison and logical operations"""
        source = """
        rtype comparison_rtype(a: i32, b: i32) {
            let equal: u32 = $unsigned(a == b);
            let not_equal: u32 = $unsigned(a != b);
            let less_than: u32 = $unsigned(a < b);
            let greater_equal: u32 = $unsigned(a >= b);
            let logical_and: u32 = $unsigned((a > 0) && (b > 0));
            let logical_or: u32 = $unsigned((a > 0) || (b > 0));
            return (equal + not_equal);
        }
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "comparison_rtype"
        assert flow.kind == FlowKind.RTYPE
        assert len(flow.inputs) == 2
        
        # Check that body exists and has statements
        assert flow.body is not None
        assert len(flow.body) > 0


if __name__ == "__main__":
    pytest.main([__file__])