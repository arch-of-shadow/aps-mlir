"""
Test _mem CPU memory operations in CADL to MLIR conversion
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
class TestMemOperations:
    """Test MLIR conversion for _mem CPU memory operations"""

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

    def test_simple_mem_read(self):
        """Test basic _mem read operation"""
        cadl_source = dedent("""
        rtype test_mem_read(addr: u5) {
            let value: u32 = _mem[addr];
            return (value);
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source, ["memref.load", "memref<?xi32>"])

        # Verify function signature includes memory argument
        assert "memref<?xi32>" in mlir_str
        assert "memref.load" in mlir_str

        print("=== Simple Memory Read MLIR ===")
        print(mlir_str)

    def test_simple_mem_write(self):
        """Test basic _mem write operation"""
        cadl_source = dedent("""
        rtype test_mem_write(addr: u5, value: u32) {
            _mem[addr] = value;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source, ["memref.store", "memref<?xi32>"])

        # Verify function signature includes memory argument
        assert "memref<?xi32>" in mlir_str
        assert "memref.store" in mlir_str

        print("=== Simple Memory Write MLIR ===")
        print(mlir_str)

    def test_mem_and_irf_combined(self):
        """Test _mem and _irf operations together"""
        cadl_source = dedent("""
        rtype test_combined(rs1: u5, addr: u5, rd: u5) {
            let reg_val: u32 = _irf[rs1];
            let mem_val: u32 = _mem[addr];
            let result: u32 = reg_val + mem_val;
            _irf[rd] = result;
            _mem[addr + 4] = result;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source,
                                             ["aps.readrf", "aps.writerf", "memref.load", "memref.store"])

        # Verify both register file and memory operations
        assert "aps.readrf" in mlir_str
        assert "aps.writerf" in mlir_str
        assert "memref.load" in mlir_str
        assert "memref.store" in mlir_str
        assert "memref<?xi32>" in mlir_str

        print("=== Combined Memory and Register File MLIR ===")
        print(mlir_str)

    def test_complex_mem_address(self):
        """Test _mem with complex address expressions"""
        cadl_source = dedent("""
        rtype test_complex_addr(base: u5, offset: u5, stride: u5) {
            let addr1: u32 = base + offset;
            let addr2: u32 = base + (stride * 2);
            let val1: u32 = _mem[addr1];
            let val2: u32 = _mem[addr2];
            _mem[base] = val1 + val2;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source,
                                             ["memref.load", "memref.store", "arith.addi", "arith.muli"])

        # Verify arithmetic operations for address calculation
        assert "arith.addi" in mlir_str
        assert "arith.muli" in mlir_str
        assert mlir_str.count("memref.load") == 2  # Two loads
        assert mlir_str.count("memref.store") == 1  # One store

        print("=== Complex Memory Address MLIR ===")
        print(mlir_str)

    def test_mem_without_memory_operations(self):
        """Test that functions without _mem don't get memory argument"""
        cadl_source = dedent("""
        rtype test_no_mem(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """)

        mlir_str = self.verify_mlir_conversion(cadl_source, ["aps.readrf", "aps.writerf"])

        # Verify NO memory argument is added
        assert "memref<?xi32>" not in mlir_str
        assert "memref.load" not in mlir_str
        assert "memref.store" not in mlir_str

        print("=== No Memory Operations MLIR ===")
        print(mlir_str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])