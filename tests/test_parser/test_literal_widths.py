#!/usr/bin/env python3
"""
Test to verify that number literal types have correct widths
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestLiteralWidths:
    """Test literal width parsing and type inference"""

    def test_width_specified_literals(self):
        """Test that width-specified literals get correct bit widths"""
        test_cases = [
            ("5'b101010", 5, 42),     # 5-bit binary
            ("8'hFF", 8, 255),        # 8-bit hex
            ("15'd123", 15, 123),     # 15-bit decimal  
            ("3'o123", 3, 83),        # 3-bit octal
        ]
        
        for literal_str, expected_width, expected_value in test_cases:
            source = f"static x: u32 = {literal_str};"
            
            ast = parse_proc(source)
            assert len(ast.statics) == 1
            
            static_var = list(ast.statics.values())[0]
            assert static_var.id == "x"
            
            # Check that we have a LitExpr with a Literal
            assert isinstance(static_var.expr, LitExpr)
            literal = static_var.expr.literal
            
            # Check the literal type and width
            assert isinstance(literal.ty, BasicType_ApUFixed)
            assert literal.ty.width == expected_width, f"Expected width {expected_width} for {literal_str}, got {literal.ty.width}"
            
            # Check the literal value
            assert isinstance(literal.lit, LiteralInner_Fixed)
            assert literal.lit.value == expected_value, f"Expected value {expected_value} for {literal_str}, got {literal.lit.value}"

    def test_default_width_literals(self):
        """Test that non-width-specified literals get default 32-bit width"""
        test_cases = [
            ("0x1234", 4660),    # Hex without width
            ("1231231", 1231231), # Decimal without width
            ("42", 42),          # Simple decimal
        ]
        
        for literal_str, expected_value in test_cases:
            source = f"static x: u32 = {literal_str};"
            
            ast = parse_proc(source)
            static_var = list(ast.statics.values())[0]
            literal = static_var.expr.literal
            
            # Should default to 32-bit unsigned
            assert isinstance(literal.ty, BasicType_ApUFixed)
            assert literal.ty.width == 32, f"Expected default width 32 for {literal_str}, got {literal.ty.width}"
            
            assert isinstance(literal.lit, LiteralInner_Fixed)
            assert literal.lit.value == expected_value

    def test_literal_type_consistency(self):
        """Test that all parsed literals use the new type system consistently"""
        source = """
        static a: u32 = 5'b101010;
        static b: u32 = 8'hFF;  
        static c: u32 = 15'd123;
        static d: u32 = 42;
        """
        
        ast = parse_proc(source)
        assert len(ast.statics) == 4
        
        # Check that all literals use the new Literal system
        for static_name, static_var in ast.statics.items():
            assert isinstance(static_var.expr, LitExpr), f"Static {static_name} should have LitExpr"
            assert isinstance(static_var.expr.literal, Literal), f"Static {static_name} should have Literal"
            assert isinstance(static_var.expr.literal.ty, BasicType), f"Static {static_name} should have BasicType"
            assert isinstance(static_var.expr.literal.lit, LiteralInner), f"Static {static_name} should have LiteralInner"

    def test_number_format_parsing(self):
        """Test that different number formats are parsed correctly"""
        format_tests = [
            ("5'b101010", 2, "101010"),   # Binary
            ("8'hFF", 16, "FF"),          # Hex uppercase  
            ("8'hff", 16, "ff"),          # Hex lowercase
            ("15'd123", 10, "123"),       # Decimal
            ("3'o123", 8, "123"),         # Octal uppercase
            ("3'O123", 8, "123"),         # Octal lowercase  
        ]
        
        for literal_str, expected_base, digits in format_tests:
            source = f"static x: u32 = {literal_str};"
            ast = parse_proc(source)
            literal = list(ast.statics.values())[0].expr.literal
            
            # Verify the value was parsed correctly for the given base
            expected_value = int(digits, expected_base)
            assert literal.lit.value == expected_value, f"Format {literal_str} should parse to {expected_value}, got {literal.lit.value}"


if __name__ == "__main__":
    pytest.main([__file__])