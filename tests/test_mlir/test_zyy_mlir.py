"""
Structured MLIR converter tests using representative custom instructions.
"""

import pytest

try:
    import circt  # noqa: F401
    import circt.ir as ir  # noqa: F401
    from tests.test_mlir.mlir_test_utils import MLIRCheck

    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False


def verify_mlir(cadl_source: str) -> MLIRCheck:
    return MLIRCheck.from_cadl(cadl_source)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestZyyRTypeInstructions:
    def test_simple_constant_rtype(self):
        mlir = verify_mlir("""
        rtype constant(rs1: u5, rs2: u5, rd: u5) {
            let r0: u32 = 0;
            _irf[rd] = r0;
        }
        """)
        mlir.assert_func("flow_constant", function_type="(i5, i5, i5) -> ()")
        mlir.assert_op_count("arith.constant", exactly=1)
        mlir.assert_op_count("aps.write_irf", exactly=1)
        mlir.assert_operand_types("aps.write_irf", ["i5", "i32"])

    def test_add_instruction_with_attributes(self):
        mlir = verify_mlir("""
        #[opcode(7'b0001011)]
        #[funct7(7'b0000000)]
        rtype add(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """)
        mlir.assert_func("flow_add", opcode=11, funct7=0, function_type="(i5, i5, i5) -> ()")
        mlir.assert_op_count("aps.read_irf", exactly=2)
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("aps.write_irf", exactly=1)
        addi = mlir.single_op("arith.addi")
        write_irf = mlir.single_op("aps.write_irf")
        mlir.assert_operand_producer(addi, 0, "aps.read_irf")
        mlir.assert_operand_producer(addi, 1, "aps.read_irf")
        mlir.assert_operand_producer(write_irf, 1, "arith.addi")

    def test_simd_add_instruction_currently_rejected(self):
        source = """
        #[opcode(7'b0101011)]
        #[funct7(7'b1111111)]
        rtype simd_add(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = {
                (r1[31:24] + r2[31:24])[7:0],
                (r1[23:16] + r2[23:16])[7:0],
                (r1[15:8] + r2[15:8])[7:0],
                (r1[7:0] + r2[7:0])[7:0]
            };
        }
        """
        with pytest.raises(Exception) as exc_info:
            verify_mlir(source)
        assert "aggregate" in str(exc_info.value).lower() or "slice" in str(exc_info.value).lower()

    def test_many_multiply_instruction(self):
        mlir = verify_mlir("""
        #[opcode(7'b0001011)]
        #[funct7(7'b0000000)]
        rtype many_mult(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 * r2 * r2 * r2 * r2;
        }
        """)
        mlir.assert_func("flow_many_mult", opcode=11, funct7=0)
        mlir.assert_op_count("aps.read_irf", exactly=2)
        mlir.assert_op_count("arith.muli", exactly=4)
        mlir.assert_op_count("aps.write_irf", exactly=1)
        mlir.assert_operand_producer(mlir.single_op("aps.write_irf"), 1, "arith.muli")

    def test_if_conditional_instruction(self):
        mlir = verify_mlir("""
        #[opcode(7'b1011011)]
        #[funct7(7'b1111111)]
        rtype if_test(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = if (r1 > 32'd6) {r1} else {r2};
        }
        """)
        mlir.assert_func("flow_if_test", opcode=91, funct7=127)
        mlir.assert_op_count("arith.cmpi", exactly=1)
        mlir.assert_op_count("arith.select", exactly=1)
        mlir.assert_op_count("aps.write_irf", exactly=1)
        select = mlir.single_op("arith.select")
        write_irf = mlir.single_op("aps.write_irf")
        mlir.assert_operand_producer(select, 0, "arith.cmpi")
        mlir.assert_operand_producer(select, 1, "aps.read_irf")
        mlir.assert_operand_producer(select, 2, "aps.read_irf")
        mlir.assert_operand_producer(write_irf, 1, "arith.select")

    def test_loop_instruction(self):
        mlir = verify_mlir("""
        #[opcode(7'b1011011)]
        #[funct7(7'b1111100)]
        rtype loop_test(rs1: u5, rs2: u5, rd: u5) {
            let sum0: u32 = _irf[rs1];
            let i0: u32 = 0;
            let n0: u32 = _irf[rs2];

            with
                i: u32 = (i0, i + 1)
                sum: u32 = (sum0, sum + 4)
                n: u32 = (n0, n)
            do {
                let n_: u32 = n;
                let sum_: u32 = sum + 4;
                let i_: u32 = i + 1;
            } while (i < n);

            _irf[rd] = sum;
        }
        """)
        mlir.assert_func("flow_loop_test", opcode=91, funct7=124)
        assert mlir.op_counts()["scf.while"] + mlir.op_counts()["scf.for"] == 1, mlir.text
        mlir.assert_op_count("aps.write_irf", exactly=1)

    def test_many_add_sequence(self):
        mlir = verify_mlir("""
        #[opcode(7'b1011011)]
        #[funct7(7'b1111100)]
        rtype many_add_test(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            let d1: u32 = r1 + r2;
            let d2: u32 = d1 + r1;
            let d3: u32 = d2 + r1;
            let d4: u32 = d3 + r1;
            let d5: u32 = d4 + r1;
            let d6: u32 = d5 + r1;
            let d7: u32 = d6 + r1;
            let d8: u32 = d7 + r1;
            _irf[rd] = d8;
        }
        """)
        mlir.assert_func("flow_many_add_test", opcode=91, funct7=124)
        mlir.assert_op_count("aps.read_irf", exactly=2)
        mlir.assert_op_count("arith.addi", exactly=8)
        mlir.assert_op_count("aps.write_irf", exactly=1)

    def test_memory_write_instruction(self):
        mlir = verify_mlir("""
        #[opcode(7'b1011011)]
        #[funct7(7'b0000000)]
        rtype mem_simplewrite(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            _mem[r1] = _irf[rs2];
            _irf[rd] = 1437;
        }
        """)
        mlir.assert_func("flow_mem_simplewrite", opcode=91, funct7=0)
        mlir.assert_op_count("aps.read_irf", exactly=2)
        mlir.assert_op_count("aps.store", exactly=1)
        mlir.assert_op_count("arith.constant", exactly=1)
        mlir.assert_op_count("aps.write_irf", exactly=1)
        store = mlir.single_op("aps.store")
        write_irf = mlir.single_op("aps.write_irf")
        mlir.assert_operand_producer(store, 0, "aps.read_irf")
        mlir.assert_operand_producer(store, 1, "aps.read_irf")
        mlir.assert_operand_producer(write_irf, 1, "arith.constant")

    def test_memory_read_instruction(self):
        mlir = verify_mlir("""
        #[opcode(7'b1011011)]
        #[funct7(7'b0000000)]
        rtype mem_read_(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            let rst: u32 = _mem[r1 + r2];
            _irf[rd] = rst;
        }
        """)
        mlir.assert_func("flow_mem_read_", opcode=91, funct7=0)
        mlir.assert_op_count("aps.read_irf", exactly=2)
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("aps.load", exactly=1)
        mlir.assert_op_count("aps.write_irf", exactly=1)
        load = mlir.single_op("aps.load")
        mlir.assert_operand_producer(load, 0, "arith.addi")
        mlir.assert_operand_producer(mlir.single_op("aps.write_irf"), 1, "aps.load")

    def test_memory_accumulate_instruction(self):
        mlir = verify_mlir("""
        #[opcode(7'b1011011)]
        #[funct7(7'b0000000)]
        rtype accum(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let a: u32 = _mem[r1];
            let b: u32 = _mem[r1 + 4];
            let c: u32 = _mem[r1 + 8];
            let d: u32 = _mem[r1 + 12];
            let rst: u32 = a + b + c + d;
            _mem[r1 + 16] = rst;
            _irf[rd] = rst;
        }
        """)
        mlir.assert_func("flow_accum", opcode=91, funct7=0)
        mlir.assert_op_count("aps.load", exactly=4)
        mlir.assert_op_count("aps.store", exactly=1)
        mlir.assert_op_count("arith.addi", at_least=7)
        mlir.assert_operand_producer(mlir.single_op("aps.write_irf"), 1, "arith.addi")

    def test_crc8_instruction(self):
        mlir = verify_mlir("""
        #[opcode(7'b0101011)]
        #[funct7(7'b0000000)]
        rtype crc8(rs1: u5, rs2: u5, rd: u5) {
            let x0: u32 = _irf[rs1];
            let i0: u32 = 0;
            let n0: u32 = 8;

            let x: u32 = x0;
            let flag: u32 = x >> 7;
            let x_shifted: u32 = x << 1;
            let x_final: u32 = if (flag != 0) { x_shifted ^ 7 } else { x_shifted };

            _irf[rd] = x_final;
        }
        """)
        mlir.assert_func("flow_crc8", opcode=43, funct7=0)
        mlir.assert_op_count("arith.shrui", exactly=1)
        mlir.assert_op_count("arith.shli", exactly=1)
        mlir.assert_op_count("arith.xori", exactly=1)
        mlir.assert_op_count("arith.select", exactly=1)
        select = mlir.single_op("arith.select")
        mlir.assert_operand_producer(select, 0, "arith.cmpi")
        mlir.assert_operand_producer(select, 1, "arith.xori")
        mlir.assert_operand_producer(select, 2, "arith.shli")
        mlir.assert_operand_producer(mlir.single_op("aps.write_irf"), 1, "arith.select")

    def test_cordic_instruction(self):
        mlir = verify_mlir("""
        static thetas: [u32; 8] = {1474560, 870484, 459940, 233473, 117189, 58652, 29333, 14667};
        #[opcode(7'b0101011)]
        #[funct7(7'b0000000)]
        rtype cordic(rs1: u5, rs2: u5, rd: u5) {
            let x0 : u32 = 19898;
            let y0 : u32 = 0;
            let z0 : u32 = _irf[rs1];
            let n0 : u32 = 8;
            let it0: u32 = 0;
            with
              it: u32 = (it0, it_)
              x: u32 = (x0, x_)
              y: u32 = (y0, y_)
              z: u32 = (z0, z_)
              n: u32 = (n0, n_)
            do {
              let z_neg: u1  = z[31:31];
              let theta: u32 = thetas[it];
              let x_shift: u32 = x >> it;
              let y_shift: u32 = y >> it;
              let x_ : u32 = if z_neg {x + y_shift} else {x - y_shift};
              let y_ : u32 = if z_neg {y - x_shift} else {y + x_shift};
              let z_ : u32 = if z_neg {z + theta} else {z - theta};
              let it_: u32 = it + 1;
              let n_ : u32 = n;
            } while (it < n);

            _irf[rd] = y;
        }
        """)
        mlir.assert_func("flow_cordic", opcode=43, funct7=0)
        mlir.assert_global("thetas", type_="memref<8xi32>", constant=True)
        mlir.assert_op_count("aps.read_smem", exactly=1)
        assert mlir.op_counts()["scf.while"] + mlir.op_counts()["scf.for"] == 1, mlir.text
        mlir.assert_op_count("aps.write_irf", exactly=1)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestSimpleMLIRConversions:
    def test_basic_arithmetic_function(self):
        mlir = verify_mlir("""
        rtype add(a: u32, b: u32, c: u32) {
            _irf[c] = (a + b);
        }
        """)
        mlir.assert_func("flow_add", function_type="(i32, i32, i32) -> ()")
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("aps.write_irf", exactly=1)
        mlir.assert_op_count("func.return", exactly=1)

    def test_multiple_operations(self):
        mlir = verify_mlir("""
        rtype compute(a: u32, b: u32, c: u32, d: u32) {
            let temp1: u32 = a + b;
            let temp2: u32 = temp1 * c;
            let result: u32 = temp2 - a;
            _irf[d] = (result);
        }
        """)
        mlir.assert_func("flow_compute")
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("arith.muli", exactly=1)
        mlir.assert_op_count("arith.subi", exactly=1)

    def test_simple_flow(self):
        mlir = verify_mlir("""
        flow process(x: u32, y: u32, z: u5) {
            let sum: u32 = x + y;
            let product: u32 = x * y;
            _irf[z] = (sum + product);
        }
        """)
        mlir.assert_func("flow_process", function_type="(i32, i32, i5) -> ()")
        mlir.assert_op_count("arith.addi", exactly=2)
        mlir.assert_op_count("arith.muli", exactly=1)
        mlir.assert_op_count("aps.write_irf", exactly=1)

    def test_static_variable(self):
        mlir = verify_mlir("""
        static counter: u32 = 42;

        rtype get_counter(a: u32) {
            _irf[a] = (counter);
        }
        """)
        mlir.assert_func("flow_get_counter")
        mlir.assert_global(
            "counter",
            type_="memref<i32>",
            constant=True,
            attrs={"initial_value": "dense<42> : tensor<i32>", "var_name": '"counter"'},
        )
        mlir.assert_op_count("aps.globalload", exactly=1)
        mlir.assert_no_op("memref.get_global")
        mlir.assert_no_op("aps.read_smem")
        mlir.assert_operand_producer(mlir.single_op("aps.write_irf"), 1, "aps.globalload")


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestMLIROutputValidation:
    def test_mlir_module_structure(self):
        mlir = verify_mlir("""
        rtype test(a: u5) {
            _irf[a] = (123);
        }
        """)
        mlir.assert_op_count("builtin.module", exactly=1)
        mlir.assert_func("flow_test", function_type="(i5) -> ()")
        mlir.assert_op_count("func.return", exactly=1)

    def test_ssa_value_uniqueness(self):
        mlir = verify_mlir("""
        rtype complex(a: u32, b: u32, c: u5) {
            let x: u32 = a + b;
            let y: u32 = x * 2;
            let z: u32 = y - a;
            _irf[c] = (z);
        }
        """)
        for op_name in ["arith.addi", "arith.muli", "arith.subi"]:
            mlir.assert_op_count(op_name, exactly=1)
            mlir.assert_result_types(op_name, ["i32"])

    def test_type_consistency(self):
        mlir = verify_mlir("""
        rtype typed(a: u32, b: u32, c: u5) {
            let x: u32 = a;
            let y: u32 = b;
            _irf[c] = (x + y);
        }
        """)
        mlir.assert_func("flow_typed", function_type="(i32, i32, i5) -> ()")
        mlir.assert_operand_types("arith.addi", ["i32", "i32"])
        mlir.assert_result_types("arith.addi", ["i32"])
        mlir.assert_operand_producer(mlir.single_op("aps.write_irf"), 1, "arith.addi")
