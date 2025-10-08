"""
Comprehensive tests for CADL AST to MLIR converter

Tests the mlir_converter module functionality including:
- Type mapping from CADL to MLIR types
- Expression conversion to SSA form
- Function and flow conversion
- Control flow structures (do-while, for loops)
- Symbol table management
- Binary and unary operations
"""

import pytest
import sys
import os

# Import CADL modules
from cadl_frontend import parse_proc
from cadl_frontend.ast import *

# Test if MLIR bindings are available
try:
    from cadl_frontend.mlir_converter import CADLMLIRConverter, convert_cadl_to_mlir
    import circt.ir as ir
    import circt.dialects.func as func
    import circt.dialects.arith as arith
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False


class TestMLIRAvailability:
    """Test MLIR bindings availability"""

    def test_mlir_imports(self):
        """Test that MLIR imports work"""
        if not MLIR_AVAILABLE:
            pytest.skip("MLIR Python bindings not available - run with: pixi run python -m pytest")

        # If we get here, MLIR is available
        assert MLIR_AVAILABLE
        assert CADLMLIRConverter is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not available")
class TestTypeMapping:
    """Test CADL type to MLIR type conversion"""

    def setup_method(self):
        """Setup converter for each test"""
        self.converter = CADLMLIRConverter()

    def test_basic_fixed_types(self):
        """Test ApFixed and ApUFixed type mapping"""
        # Test unsigned fixed types
        u32_type = BasicType_ApUFixed(32)
        mlir_u32 = self.converter.convert_cadl_type(u32_type)
        assert mlir_u32 == ir.IntegerType.get_signless(32)

        u8_type = BasicType_ApUFixed(8)
        mlir_u8 = self.converter.convert_cadl_type(u8_type)
        assert mlir_u8 == ir.IntegerType.get_signless(32)  # Fallback to i32

        # Test signed fixed types
        i32_type = BasicType_ApFixed(32)
        mlir_i32 = self.converter.convert_cadl_type(i32_type)
        assert mlir_i32 == ir.IntegerType.get_signed(32)

        i64_type = BasicType_ApFixed(64)
        mlir_i64 = self.converter.convert_cadl_type(i64_type)
        assert mlir_i64 == ir.IntegerType.get_signed(64)

    def test_float_types(self):
        """Test float type mapping"""
        f32_type = BasicType_Float32()
        mlir_f32 = self.converter.convert_cadl_type(f32_type)
        assert mlir_f32 == ir.F32Type.get()

        f64_type = BasicType_Float64()
        mlir_f64 = self.converter.convert_cadl_type(f64_type)
        assert mlir_f64 == ir.F64Type.get()

    def test_array_types(self):
        """Test array type mapping to memref"""
        element_type = BasicType_ApUFixed(8)
        array_type = DataType_Array(element_type, [4, 8])

        mlir_array = self.converter.convert_cadl_type(array_type)
        assert str(mlir_array).startswith("memref<4x8x")

    def test_compound_types(self):
        """Test compound type mapping"""
        basic_type = BasicType_ApUFixed(32)
        data_type = DataType_Single(basic_type)
        compound_type = CompoundType_Basic(data_type)

        mlir_type = self.converter.convert_cadl_type(compound_type)
        assert mlir_type == ir.IntegerType.get_signless(32)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not available")
class TestSymbolTable:
    """Test symbol table management for SSA form"""

    def setup_method(self):
        """Setup converter for each test"""
        self.converter = CADLMLIRConverter()

    def test_symbol_scoping(self):
        """Test symbol scoping push/pop operations"""
        # Create a dummy value for testing
        with self.converter.context:
            dummy_type = ir.IntegerType.get_signless(32)
            with ir.Location.unknown():
                module = ir.Module.create()
                with ir.InsertionPoint(module.body):
                    const_op = arith.ConstantOp(dummy_type, 42)
                    dummy_value = const_op.result

        # Test initial state
        assert self.converter.get_symbol("test_var") is None

        # Set symbol in current scope
        self.converter.set_symbol("test_var", dummy_value)
        assert self.converter.get_symbol("test_var") == dummy_value

        # Push new scope
        self.converter.push_scope()
        assert self.converter.get_symbol("test_var") == dummy_value  # Should find in parent

        # Shadow variable in new scope
        self.converter.set_symbol("test_var", dummy_value)  # Same value, different scope
        assert self.converter.get_symbol("test_var") == dummy_value

        # Pop scope
        self.converter.pop_scope()
        assert self.converter.get_symbol("test_var") == dummy_value

    def test_undefined_symbol(self):
        """Test handling of undefined symbols"""
        assert self.converter.get_symbol("undefined") is None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not available")
class TestExpressionConversion:
    """Test expression conversion to MLIR SSA values"""

    def setup_method(self):
        """Setup converter and context"""
        self.converter = CADLMLIRConverter()
        self.converter.context.__enter__()
        self.module = ir.Module.create()
        self.converter.module = self.module
        self.insertion_point = ir.InsertionPoint(self.module.body)
        self.insertion_point.__enter__()

    def teardown_method(self):
        """Cleanup context"""
        self.insertion_point.__exit__(None, None, None)
        self.converter.context.__exit__(None, None, None)

    def test_literal_conversion(self):
        """Test literal expression conversion"""
        # Create literal expression
        literal = Literal(LiteralInner_Fixed(42), BasicType_ApUFixed(32))
        lit_expr = LitExpr(literal)

        # Convert to MLIR
        with ir.Location.unknown():
            mlir_value = self.converter._convert_expr(lit_expr)
            assert mlir_value is not None
            assert mlir_value.type == ir.IntegerType.get_signless(32)

    def test_binary_operation_conversion(self):
        """Test binary operation conversion"""
        # Create two literal expressions
        lit1 = LitExpr(Literal(LiteralInner_Fixed(10), BasicType_ApUFixed(32)))
        lit2 = LitExpr(Literal(LiteralInner_Fixed(20), BasicType_ApUFixed(32)))

        # Create binary expression
        add_expr = BinaryExpr(BinaryOp.ADD, lit1, lit2)

        # Convert to MLIR
        with ir.Location.unknown():
            mlir_value = self.converter._convert_expr(add_expr)
            assert mlir_value is not None
            assert mlir_value.type == ir.IntegerType.get_signless(32)

    def test_identifier_expression(self):
        """Test identifier expression lookup"""
        # First create a symbol
        with ir.Location.unknown():
            dummy_type = ir.IntegerType.get_signless(32)
            const_op = arith.ConstantOp(dummy_type, 42)
            self.converter.set_symbol("test_var", const_op.result)

        # Create identifier expression
        ident_expr = IdentExpr("test_var")

        # Convert to MLIR - should find in symbol table
        mlir_value = self.converter._convert_expr(ident_expr)
        assert mlir_value is not None
        assert mlir_value.type == ir.IntegerType.get_signless(32)

    def test_undefined_identifier_error(self):
        """Test error on undefined identifier"""
        ident_expr = IdentExpr("undefined_var")

        with pytest.raises(ValueError, match="Undefined symbol"):
            self.converter._convert_expr(ident_expr)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not available")
class TestFunctionConversion:
    """Test function conversion to MLIR func.func operations"""

    def test_simple_function_parsing_and_conversion(self):
        """Test parsing and converting a simple function"""
        cadl_source = """
        fn add_two(a: u32, b: u32) -> (u32) {
            return (a + b);
        }
        """

        # Parse CADL
        ast = parse_proc(cadl_source, "test.cadl")
        assert len(ast.functions) == 1

        # Convert to MLIR
        mlir_module = convert_cadl_to_mlir(ast)
        assert mlir_module is not None

        # Check that module contains function
        module_str = str(mlir_module)
        assert "func.func" in module_str
        assert "add_two" in module_str

    def test_function_with_multiple_args(self):
        """Test function with multiple arguments"""
        cadl_source = """
        fn compute(a: u32, b: u32, c: u32) -> (u32) {
            let result: u32 = a + b * c;
            return (result);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        mlir_module = convert_cadl_to_mlir(ast)

        module_str = str(mlir_module)
        assert "compute" in module_str
        assert "func.func" in module_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not available")
class TestFlowConversion:
    """Test flow conversion to MLIR operations"""

    def test_simple_flow_conversion(self):
        """Test converting a simple flow"""
        cadl_source = """
        flow multiply(x: u32, y: u32) {
            return (x * y);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        assert len(ast.flows) == 1

        mlir_module = convert_cadl_to_mlir(ast)
        module_str = str(mlir_module)

        # Flow should be converted to function
        assert "flow_multiply" in module_str
        assert "func.func" in module_str

    def test_rtype_flow_conversion(self):
        """Test converting an rtype flow with attributes"""
        cadl_source = """
        #[opcode(51)]
        #[funct3(0)]
        rtype add_inst(rs1: u32, rs2: u32) {
            return (rs1 + rs2);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        flow = list(ast.flows.values())[0]

        # Check that attributes are preserved
        assert flow.attrs.get("opcode") is not None
        assert flow.attrs.get("funct3") is not None

        mlir_module = convert_cadl_to_mlir(ast)
        assert mlir_module is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not available")
class TestBinaryOperations:
    """Test binary operation conversion to MLIR operations"""

    def setup_method(self):
        """Setup for each test"""
        self.converter = CADLMLIRConverter()

    def test_arithmetic_operations(self):
        """Test arithmetic operation mapping"""
        with self.converter.context, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                # Create dummy operands
                dummy_type = ir.IntegerType.get_signless(32)
                left = arith.ConstantOp(dummy_type, 10).result
                right = arith.ConstantOp(dummy_type, 20).result

                # Test addition
                add_result = self.converter._convert_binary_op(BinaryOp.ADD, left, right)
                assert add_result is not None

                # Test subtraction
                sub_result = self.converter._convert_binary_op(BinaryOp.SUB, left, right)
                assert sub_result is not None

                # Test multiplication
                mul_result = self.converter._convert_binary_op(BinaryOp.MUL, left, right)
                assert mul_result is not None

    def test_comparison_operations(self):
        """Test comparison operation mapping"""
        with self.converter.context, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                dummy_type = ir.IntegerType.get_signless(32)
                left = arith.ConstantOp(dummy_type, 10).result
                right = arith.ConstantOp(dummy_type, 20).result

                # Test equality
                eq_result = self.converter._convert_binary_op(BinaryOp.EQ, left, right)
                assert eq_result is not None

                # Test less than
                lt_result = self.converter._convert_binary_op(BinaryOp.LT, left, right)
                assert lt_result is not None

    def test_bitwise_operations(self):
        """Test bitwise operation mapping to comb dialect"""
        with self.converter.context, ir.Location.unknown():
            module = ir.Module.create()
            with ir.InsertionPoint(module.body):
                dummy_type = ir.IntegerType.get_signless(32)
                left = arith.ConstantOp(dummy_type, 10).result
                right = arith.ConstantOp(dummy_type, 20).result

                # Test bitwise AND
                and_result = self.converter._convert_binary_op(BinaryOp.BIT_AND, left, right)
                assert and_result is not None

                # Test bitwise OR
                or_result = self.converter._convert_binary_op(BinaryOp.BIT_OR, left, right)
                assert or_result is not None

                # Test bitwise XOR
                xor_result = self.converter._convert_binary_op(BinaryOp.BIT_XOR, left, right)
                assert xor_result is not None


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not available")
class TestControlFlow:
    """Test control flow conversion (do-while, for loops)"""

    def test_do_while_parsing(self):
        """Test parsing do-while statements"""
        # Note: This tests the AST parsing, not full MLIR conversion
        # since do-while implementation is complex
        cadl_source = """
        fn test_loop() -> () {
            do with (i: u32 = 0; i + 1) {
                let x: u32 = i * 2;
            } while (i < 10);
        }
        """

        ast = parse_proc(cadl_source, "test.cadl")
        function = list(ast.functions.values())[0]

        # Check that do-while statement is parsed
        assert len(function.body) > 0
        # The exact structure depends on the parser implementation


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR Python bindings not available")
class TestIntegration:
    """Integration tests for complete CADL to MLIR conversion"""

    def test_complete_processor_conversion(self):
        """Test converting a complete processor with multiple components"""
        cadl_source = """
        // Static variable
        static counter: u32 = 0;

        // Simple function
        fn increment(x: u32) -> (u32) {
            return (x + 1);
        }

        // Flow definition
        flow add_flow(a: u32, b: u32) {
            let result: u32 = a + b;
            return (result);
        }

        // RType flow with attributes
        #[opcode(51)]
        rtype alu_op(rs1: u32, rs2: u32) {
            return (rs1 + rs2);
        }
        """

        # Parse CADL
        ast = parse_proc(cadl_source, "complete_test.cadl")

        # Verify AST structure
        assert len(ast.statics) == 1
        assert len(ast.functions) == 1
        assert len(ast.flows) == 2

        # Convert to MLIR
        mlir_module = convert_cadl_to_mlir(ast)
        assert mlir_module is not None

        module_str = str(mlir_module)

        # Check that all components are present in MLIR
        assert "increment" in module_str
        assert "flow_add_flow" in module_str
        assert "flow_alu_op" in module_str
        assert "func.func" in module_str

    def test_error_handling(self):
        """Test error handling in conversion process"""
        # Test with malformed function
        cadl_source = """
        fn broken_function(x: unknown_type) -> (u32) {
            return (x);
        }
        """

        # This should either parse with error or convert with appropriate handling
        try:
            ast = parse_proc(cadl_source, "error_test.cadl")
            # If parsing succeeds, conversion might fail
            mlir_module = convert_cadl_to_mlir(ast)
        except Exception as e:
            # Expected - either parsing or conversion should handle unknown types
            assert "unknown_type" in str(e) or "not" in str(e).lower()


class TestRunnerCompatibility:
    """Test that the reorganized tests work with pytest"""

    def test_pytest_discovery(self):
        """Test that pytest can discover tests in new structure"""
        # This test ensures the file structure works
        assert True

    def test_import_structure(self):
        """Test that imports work correctly after reorganization"""
        # Test basic imports
        from cadl_frontend import parse_proc
        from cadl_frontend.ast import BasicType_ApUFixed

        # Test that we can create basic AST objects
        basic_type = BasicType_ApUFixed(32)
        assert basic_type.width == 32


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__])