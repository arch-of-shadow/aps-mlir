"""
Simplified MLIR converter tests using examples from zyy.cadl
All _irf and _mem accesses are replaced with constants to test core conversion logic
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
class TestZyyRTypeSimplified:
    """Test MLIR conversion for simplified rtype instructions (no array access)"""

    def verify_mlir_output(self, cadl_source: str, expected_functions: list = None):
        """Helper to verify MLIR conversion succeeds and contains expected functions"""
        # Parse CADL
        ast = parse_proc(cadl_source, "test.cadl")

        # Convert to MLIR
        mlir_module = convert_cadl_to_mlir(ast)
        assert mlir_module is not None

        # Convert to string and verify
        mlir_str = str(mlir_module)
        print(mlir_str)
        
        assert "module" in mlir_str or "builtin.module" in mlir_str

        # Check for expected functions if provided
        if expected_functions:
            for func_name in expected_functions:
                assert func_name in mlir_str, f"Expected function '{func_name}' not found in MLIR output"

        return mlir_str

    def test_simple_constant_rtype(self):
        """Test the simplest rtype: constant function"""
        cadl_source = """
        rtype constant(rs1: u5, rs2: u5, rd: u5) {
            let r0: u32 = 0;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_constant"])

        # Verify basic structure
        assert "func.func" in mlir_str
        assert "arith.constant" in mlir_str  # Should have constant 0

    def test_add_instruction_simplified(self):
        """Test add instruction with constants instead of _irf access"""
        cadl_source = """
        #[opcode(7'b0001011)]  // custom0
        #[funct7(7'b0000000)]
        rtype add(rs1: u5, rs2: u5, rd: u5) {
            // Original: let r1: u32 = _irf[rs1];
            // Original: let r2: u32 = _irf[rs2];
            let r1: u32 = 100;  // Simulating _irf[rs1]
            let r2: u32 = 200;  // Simulating _irf[rs2]
            let result: u32 = r1 + r2;
            // Original: _irf[rd] = result;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_add"])

        # Verify arithmetic operations
        assert "arith.addi" in mlir_str
        assert "100" in mlir_str or "arith.constant" in mlir_str
        assert "200" in mlir_str or "arith.constant" in mlir_str

    def test_many_multiply_simplified(self):
        """Test instruction with many multiplications using constants"""
        cadl_source = """
        #[opcode(7'b0001011)]  // custom0
        #[funct7(7'b0000000)]
        rtype many_mult(rs1: u5, rs2: u5, rd: u5) {
            // Simulating _irf access with constants
            let r1: u32 = 2;  // _irf[rs1]
            let r2: u32 = 3;  // _irf[rs2]
            let result: u32 = r1 * r2 * r2 * r2 * r2;
            // Result would be 2 * 3 * 3 * 3 * 3 = 162
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_many_mult"])

        # Should have multiple multiply operations
        assert "arith.muli" in mlir_str
        # At least 4 multiplications
        assert mlir_str.count("arith.muli") >= 4

    def test_many_add_sequence_simplified(self):
        """Test instruction with sequence of additions using constants"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b1111100)]
        rtype many_add_test(rs1: u5, rs2: u5, rd: u5) {
            // Simulating _irf access
            let r1: u32 = 10;  // _irf[rs1]
            let r2: u32 = 20;  // _irf[rs2]
            let d1: u32 = r1 + r2;  // 30
            let d2: u32 = d1 + r1;  // 40
            let d3: u32 = d2 + r1;  // 50
            let d4: u32 = d3 + r1;  // 60
            let d5: u32 = d4 + r1;  // 70
            let d6: u32 = d5 + r1;  // 80
            let d7: u32 = d6 + r1;  // 90
            let d8: u32 = d7 + r1;  // 100
            // _irf[rd] = d8;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_many_add_test"])

        # Should have multiple add operations
        assert mlir_str.count("arith.addi") >= 8

    def test_memory_write_simplified(self):
        """Test memory write instruction with constants"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b0000000)]
        rtype mem_simplewrite(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = 1000;  // Simulating _irf[rs1]
            // Original: _mem[r1] = _irf[rs2];
            let mem_value: u32 = 42;  // Value to write
            let result: u32 = 1437;  // Return value
            // Original: _irf[rd] = 1437;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_mem_simplewrite"])

        # Should have constant 1437
        assert "1437" in mlir_str or "arith.constant" in mlir_str

    def test_memory_read_simplified(self):
        """Test memory read instruction with constants"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b0000000)]
        rtype mem_read_(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = 1000;  // Simulating _irf[rs1]
            let r2: u32 = 4;     // Simulating _irf[rs2]
            // Original: let rst: u32 = _mem[r1 + r2];
            let addr: u32 = r1 + r2;  // Address calculation
            let rst: u32 = 999;  // Simulating memory read result
            // Original: _irf[rd] = rst;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_mem_read_"])

        # Should have add for address calculation
        assert "arith.addi" in mlir_str

    def test_memory_accumulate_simplified(self):
        """Test accumulate instruction with constants simulating memory reads"""
        cadl_source = """
        #[opcode(7'b1011011)]  // custom2
        #[funct7(7'b0000000)]
        rtype accum(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = 1000;  // Base address from _irf[rs1]
            // Simulating memory reads at different offsets
            let a: u32 = 10;  // _mem[r1]
            let b: u32 = 20;  // _mem[r1 + 4]
            let c: u32 = 30;  // _mem[r1 + 8]
            let d: u32 = 40;  // _mem[r1 + 12]
            let rst: u32 = a + b + c + d;  // Should be 100
            // Original: _mem[r1 + 16] = rst;
            // Original: _irf[rd] = rst;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_accum"])

        # Should have multiple additions for accumulation
        assert mlir_str.count("arith.addi") >= 3  # At least for a+b+c+d

    def test_complex_arithmetic_simplified(self):
        """Test complex arithmetic without memory/register access"""
        cadl_source = """
        rtype complex_math(rs1: u5, rs2: u5, rd: u5) {
            // Simulating register values
            let x: u32 = 15;
            let y: u32 = 25;

            // Complex calculations
            let a: u32 = x + y;       // 40
            let b: u32 = x * y;       // 375
            let c: u32 = a - x;       // 25
            let d: u32 = b / 5;       // 75 (if division works)
            let e: u32 = c * 2;       // 50
            let result: u32 = d + e;  // 125
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_complex_math"])

        # Should have various arithmetic operations
        assert "arith.addi" in mlir_str
        assert "arith.muli" in mlir_str
        assert "arith.subi" in mlir_str
        assert "arith.divsi" in mlir_str

    def test_bitwise_operations_simplified(self):
        """Test bitwise operations with constants"""
        cadl_source = """
        rtype bitwise_ops(rs1: u5, rs2: u5, rd: u5) {
            let a: u32 = 0xFF00;  // Simulating _irf[rs1]
            let b: u32 = 0x00FF;  // Simulating _irf[rs2]

            let and_result: u32 = a & b;    // Should be 0
            let or_result: u32 = a | b;     // Should be 0xFFFF
            let xor_result: u32 = a ^ b;    // Should be 0xFFFF
            let shift_left: u32 = a << 4;   // Should be 0xFF000
            let shift_right: u32 = b >> 4;  // Should be 0x000F
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_bitwise_ops"])

        # Should have bitwise operations
        assert "comb.and" in mlir_str or "arith.andi" in mlir_str
        assert "comb.or" in mlir_str or "arith.ori" in mlir_str
        assert "comb.xor" in mlir_str or "arith.xori" in mlir_str
        assert "comb.shl" in mlir_str or "arith.shli" in mlir_str

    def test_comparison_operations_simplified(self):
        """Test comparison operations with constants"""
        cadl_source = """
        rtype compare_ops(rs1: u5, rs2: u5, rd: u5) {
            let a: u32 = 100;
            let b: u32 = 200;

            // These would need if-expressions to be useful, but test the comparisons
            let eq: u32 = if (a == b) {1} else {0};
            let ne: u32 = if (a != b) {1} else {0};
            let lt: u32 = if (a < b) {1} else {0};
            let gt: u32 = if (a > b) {1} else {0};
            let le: u32 = if (a <= b) {1} else {0};
            let ge: u32 = if (a >= b) {1} else {0};
        }
        """

        # This will likely fail due to if-expressions, but test the parsing
        with pytest.raises(Exception):
            self.verify_mlir_output(cadl_source)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestComplexSimplified:
    """Test more complex patterns with simplified memory/register access"""

    def verify_mlir_output(self, cadl_source: str, expected_functions: list = None):
        """Helper to verify MLIR conversion succeeds and contains expected functions"""
        # Parse CADL
        ast = parse_proc(cadl_source, "test.cadl")

        # Convert to MLIR
        mlir_module = convert_cadl_to_mlir(ast)
        assert mlir_module is not None

        # Convert to string and verify
        mlir_str = str(mlir_module)
        assert "module" in mlir_str or "builtin.module" in mlir_str

        # Check for expected functions if provided
        if expected_functions:
            for func_name in expected_functions:
                assert func_name in mlir_str, f"Expected function '{func_name}' not found in MLIR output"

        return mlir_str

    def test_fibonacci_style_computation(self):
        """Test Fibonacci-style iterative computation"""
        cadl_source = """
        rtype fibonacci_iter(rs1: u5, rs2: u5, rd: u5) {
            // Compute first N fibonacci numbers
            let n: u32 = 10;  // Simulating _irf[rs1]

            let f0: u32 = 0;
            let f1: u32 = 1;
            let f2: u32 = f0 + f1;  // 1
            let f3: u32 = f1 + f2;  // 2
            let f4: u32 = f2 + f3;  // 3
            let f5: u32 = f3 + f4;  // 5
            let f6: u32 = f4 + f5;  // 8
            let f7: u32 = f5 + f6;  // 13
            let f8: u32 = f6 + f7;  // 21
            let result: u32 = f8;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_fibonacci_iter"])

        # Should have many additions
        assert mlir_str.count("arith.addi") >= 7

    def test_polynomial_evaluation(self):
        """Test polynomial evaluation: ax^3 + bx^2 + cx + d"""
        cadl_source = """
        rtype polynomial(rs1: u5, rs2: u5, rd: u5) {
            // Coefficients
            let a: u32 = 2;
            let b: u32 = 3;
            let c: u32 = 4;
            let d: u32 = 5;
            let x: u32 = 10;  // Simulating _irf[rs1]

            // Compute x^2, x^3
            let x2: u32 = x * x;      // 100
            let x3: u32 = x2 * x;     // 1000

            // Compute polynomial
            let term1: u32 = a * x3;  // 2000
            let term2: u32 = b * x2;  // 300
            let term3: u32 = c * x;   // 40

            let sum1: u32 = term1 + term2;  // 2300
            let sum2: u32 = sum1 + term3;   // 2340
            let result: u32 = sum2 + d;     // 2345
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_polynomial"])

        # Should have multiplications and additions
        assert "arith.muli" in mlir_str
        assert "arith.addi" in mlir_str

    def test_nested_arithmetic(self):
        """Test deeply nested arithmetic expressions"""
        cadl_source = """
        rtype nested(rs1: u5, rs2: u5, rd: u5) {
            let a: u32 = 5;
            let b: u32 = 10;
            let c: u32 = 15;
            let d: u32 = 20;

            // ((a + b) * (c - d)) + ((a * b) - (c + d))
            let sum1: u32 = a + b;     // 15
            let diff1: u32 = c - d;    // -5 (as unsigned)
            let prod1: u32 = sum1 * diff1;

            let prod2: u32 = a * b;    // 50
            let sum2: u32 = c + d;     // 35
            let diff2: u32 = prod2 - sum2;  // 15

            let result: u32 = prod1 + diff2;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_nested"])

        # Should have all types of operations
        assert "arith.addi" in mlir_str
        assert "arith.subi" in mlir_str
        assert "arith.muli" in mlir_str


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestRealWorldSimplified:
    """Test real-world algorithms with simplified I/O"""

    def verify_mlir_output(self, cadl_source: str, expected_functions: list = None):
        """Helper to verify MLIR conversion succeeds and contains expected functions"""
        # Parse CADL
        ast = parse_proc(cadl_source, "test.cadl")

        # Convert to MLIR
        mlir_module = convert_cadl_to_mlir(ast)
        assert mlir_module is not None

        # Convert to string and verify
        mlir_str = str(mlir_module)
        assert "module" in mlir_str or "builtin.module" in mlir_str

        # Check for expected functions if provided
        if expected_functions:
            for func_name in expected_functions:
                assert func_name in mlir_str, f"Expected function '{func_name}' not found in MLIR output"

        return mlir_str

    def test_gcd_algorithm(self):
        """Test GCD algorithm (without loops)"""
        cadl_source = """
        rtype gcd_step(rs1: u5, rs2: u5, rd: u5) {
            // Single step of GCD algorithm
            let a: u32 = 48;  // Simulating _irf[rs1]
            let b: u32 = 18;  // Simulating _irf[rs2]

            // One iteration: a % b
            let quotient: u32 = a / b;  // 2
            let product: u32 = quotient * b;  // 36
            let remainder: u32 = a - product;  // 12

            // Next values would be b=18, remainder=12
            let next_a: u32 = b;
            let next_b: u32 = remainder;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_gcd_step"])

        # Should have division, multiplication, subtraction
        assert "arith.divsi" in mlir_str
        assert "arith.muli" in mlir_str
        assert "arith.subi" in mlir_str

    def test_checksum_computation(self):
        """Test simple checksum computation"""
        cadl_source = """
        rtype checksum(rs1: u5, rs2: u5, rd: u5) {
            // Simulating data values
            let data0: u32 = 0x1234;
            let data1: u32 = 0x5678;
            let data2: u32 = 0x9ABC;
            let data3: u32 = 0xDEF0;

            // Simple additive checksum
            let sum0: u32 = data0 + data1;
            let sum1: u32 = sum0 + data2;
            let sum2: u32 = sum1 + data3;

            // XOR checksum
            let xor0: u32 = data0 ^ data1;
            let xor1: u32 = xor0 ^ data2;
            let xor2: u32 = xor1 ^ data3;

            // Combine both
            let result: u32 = sum2 + xor2;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_checksum"])

        # Should have additions and XOR operations
        assert "arith.addi" in mlir_str
        assert "xor" in mlir_str.lower()


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestZyyLoopsSimplified:
    """Test loop constructs directly from zyy.cadl with simplified memory/register access"""

    def verify_mlir_output(self, cadl_source: str, expected_functions: list = None):
        """Helper to verify MLIR conversion succeeds and contains expected functions"""
        # Parse CADL
        ast = parse_proc(cadl_source, "test.cadl")

        # Convert to MLIR
        mlir_module = convert_cadl_to_mlir(ast)
        assert mlir_module is not None

        # Convert to string and verify
        mlir_str = str(mlir_module)
        print(mlir_str)

        assert "module" in mlir_str or "builtin.module" in mlir_str

        # Check for expected functions if provided
        if expected_functions:
            for func_name in expected_functions:
                assert func_name in mlir_str, f"Expected function '{func_name}' not found in MLIR output"

        return mlir_str

    def test_loop_test_from_zyy(self):
        """Test loop_test from zyy.cadl with do-while (simplified)"""
        cadl_source = """
        #[opcode(7'b1011011)]
        #[funct7(7'b1111100)]
        rtype loop_test(rs1: u5, rs2: u5, rd: u5) {
            let sum0: u32 = 100;  // Simulating _irf[rs1]
            let i0: u32 = 0;
            let n0: u32 = 5;      // Simulating _irf[rs2]

            with
                i: u32 = (i0, i_)
                sum: u32 = (sum0, sum_)
                n: u32 = (n0, n_)
            do {
                let n_: u32 = n;
                let sum_: u32 = sum + 4;
                let i_: u32 = i + 1;
            } while (i_ < n);

            let result: u32 = sum;  // Would be _irf[rd] = sum
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_loop_test"])

        # Should have loop structure
        assert "scf.while" in mlir_str or "scf.for" in mlir_str
        assert "arith.addi" in mlir_str  # For i+1 and sum+4
        assert "arith.cmpi" in mlir_str  # For i < n comparison

    def test_crc8_from_zyy(self):
        """Test crc8 from zyy.cadl with do-while loop"""
        cadl_source = """
        #[opcode(7'b0101011)]
        #[funct7(7'b0000000)]
        rtype crc8(rs1: u5, rs2: u5, rd: u5) {
            let x0: u32 = 255;  // Simulating _irf[rs1]
            with
                i: u32 = (0, i_)
                x: u32 = (x0, x_)
            do {
                let a: u32 = x >> 1;
                let x_: u32 = a ^ (0xEDB88320 & ~((x & 1) - 1));
                let i_: u32 = i + 1;
            } while (i < 8);

            let result: u32 = x + 1;  // Would be _irf[rd] = x
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_crc8"])

        # Should have loop and CRC operations
        assert "scf.while" in mlir_str or "scf.for" in mlir_str
        assert "comb.shru" in mlir_str or "shr" in mlir_str.lower()
        assert "comb.and" in mlir_str
        assert "comb.xor" in mlir_str

    def test_cplx_mult_from_zyy(self):
        """Test complex multiplication from zyy.cadl (without bit slicing)"""
        cadl_source = """
        #[opcode(7'b0101011)]
        #[funct7(7'b0000000)]
        rtype cplx_mult(rs1: u5, rs2: u5, rd: u5) {
            let r1: i32 = 100;  // Simulating _irf[rs1] as signed
            let r2: i32 = 200;  // Simulating _irf[rs2] as signed

            // Original extracts 16-bit parts: ar = r1[31:16], ai = r1[15:0], etc.
            // Since bit slicing isn't supported, simulate with constants

            let ar: i32 = 10;  // Real part of first complex number
            let ai: i32 = 20;  // Imaginary part of first complex number
            let br: i32 = 30;  // Real part of second complex number
            let bi: i32 = 40;  // Imaginary part of second complex number

            // Complex multiplication: (ar + ai*j) * (br + bi*j)
            let zr: i32 = ar * br - ai * bi;  // Real part: ar*br - ai*bi
            let zi: i32 = ar * bi + ai * br;  // Imaginary part: ar*bi + ai*br

            // Would pack result back, but just use zr for now
            let result: i32 = zr;
        }
        """

        mlir_str = self.verify_mlir_output(cadl_source, ["flow_cplx_mult"])

        # Should have multiplications, addition, and subtraction
        assert mlir_str.count("arith.muli") >= 4  # ar*br, ai*bi, ar*bi, ai*br
        assert "arith.addi" in mlir_str
        assert "arith.subi" in mlir_str


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    exit(result.returncode)