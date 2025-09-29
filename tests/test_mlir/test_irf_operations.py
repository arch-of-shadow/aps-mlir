"""
Test __irf register file operations in CADL to MLIR conversion
"""

import pytest
import sys
import os
from textwrap import dedent

# Try importing MLIR bindings
try:
    import circt
    import circt.ir as ir
    from cadl_frontend.mlir_converter import CADLMLIRConverter, convert_cadl_to_mlir
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False

from cadl_frontend.parser import parse_proc
from cadl_frontend.ast import *


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestIrfOperations:
    """Test MLIR conversion for __irf register file operations"""

    def verify_mlir_conversion(self, cadl_source: str, expected_ops: list = None):
        """Helper to verify MLIR conversion succeeds and contains expected operations"""
        # Parse CADL
        ast = parse_proc(cadl_source, "test.cadl")

        # Convert to MLIR
        mlir_module = convert_cadl_to_mlir(ast)
        assert mlir_module is not None

        # Convert to string and verify
        mlir_str = str(mlir_module)
        assert "module" in mlir_str or "builtin.module" in mlir_str

        # Check for expected operations if provided
        if expected_ops:
            for op in expected_ops:
                assert op in mlir_str, f"Expected operation '{op}' not found in MLIR output"

        return mlir_str

    def test_simple_irf_read(self):
        """Test basic _irf read operation"""
        cadl_source = dedent("""
        rtype test_irf_read(rs1: u5) {
            let value: u32 = _irf[rs1];
            return (value);
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source, ["aps.readrf"])
        print("=== Simple IRF Read MLIR ===")
        print(mlir_str)

    def test_simple_irf_write(self):
        """Test basic _irf write operation"""
        cadl_source = dedent("""
        rtype test_irf_write(rd: u5, value: u32) {
            _irf[rd] = value;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source, ["aps.writerf"])
        print("=== Simple IRF Write MLIR ===")
        print(mlir_str)

    def test_irf_read_write_combined(self):
        """Test _irf read and write in same function"""
        cadl_source = dedent("""
        rtype test_irf_copy(rs1: u5, rd: u5) {
            let value: u32 = _irf[rs1];
            _irf[rd] = value;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source, ["aps.readrf", "aps.writerf"])
        print("=== IRF Copy MLIR ===")
        print(mlir_str)

    def test_irf_arithmetic(self):
        """Test _irf operations with arithmetic"""
        cadl_source = dedent("""
        rtype test_irf_add(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            let sum: u32 = r1 + r2;
            _irf[rd] = sum;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source,
                                             ["aps.readrf", "aps.writerf", "arith.addi"])
        print("=== IRF Arithmetic MLIR ===")
        print(mlir_str)

    def test_irf_complex_expression(self):
        """Test _irf with complex expressions"""
        cadl_source = dedent("""
        rtype test_irf_complex(rs1: u5, rs2: u5, rs3: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            let r3: u32 = _irf[rs3];
            let result: u32 = (r1 + r2) * r3;
            _irf[rd] = result;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source,
                                             ["aps.readrf", "aps.writerf", "arith.addi", "arith.muli"])
        print("=== IRF Complex Expression MLIR ===")
        print(mlir_str)

    def test_irf_with_constants(self):
        """Test _irf operations with literal constants"""
        cadl_source = dedent("""
        rtype test_irf_const(rs1: u5, rd: u5) {
            let value: u32 = _irf[rs1];
            let incremented: u32 = value + 42;
            _irf[rd] = incremented;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source,
                                             ["aps.readrf", "aps.writerf", "arith.constant", "arith.addi"])
        print("=== IRF with Constants MLIR ===")
        print(mlir_str)

    def test_multiple_irf_operations(self):
        """Test multiple sequential _irf operations"""
        cadl_source = dedent("""
        rtype test_multiple_irf(rs1: u5, rs2: u5, rd1: u5, rd2: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];

            // Write r1 to rd1
            _irf[rd1] = r1;

            // Write r2 to rd2
            _irf[rd2] = r2;

            // Write sum to rs1 (reuse register)
            _irf[rs1] = r1 + r2;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source, ["aps.readrf", "aps.writerf"])

        # Count occurrences to verify multiple operations
        read_count = mlir_str.count("aps.readrf")
        write_count = mlir_str.count("aps.writerf")

        assert read_count >= 2, f"Expected at least 2 readrf operations, found {read_count}"
        assert write_count >= 3, f"Expected at least 3 writerf operations, found {write_count}"

        print("=== Multiple IRF Operations MLIR ===")
        print(mlir_str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])