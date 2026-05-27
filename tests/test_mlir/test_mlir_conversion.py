"""
Structured MLIR conversion tests.

These tests inspect generated MLIR as IR operations instead of using broad
string membership checks. The text form is still printed in assertion failures
by MLIRCheck for debugging.
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
class TestBasicMLIRConversion:
    def test_simple_arithmetic(self):
        mlir = verify_mlir("""
        rtype add(a: u32, b: u32, c: u32) {
            let result: u32 = a + b;
            _irf[c] = result;
        }
        """)
        mlir.assert_func("flow_add", function_type="(i32, i32, i32) -> ()")
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("aps.writerf", exactly=1)
        mlir.assert_operand_types("arith.addi", ["i32", "i32"])
        mlir.assert_result_types("arith.addi", ["i32"])
        addi = mlir.single_op("arith.addi")
        writerf = mlir.single_op("aps.writerf")
        mlir.assert_operand_producer(addi, 0, None)
        mlir.assert_operand_producer(addi, 1, None)
        mlir.assert_operand_producer(writerf, 1, "arith.addi")
        mlir.assert_result_has_user(addi, "aps.writerf")

    def test_multiple_operations(self):
        mlir = verify_mlir("""
        rtype compute(a: u32, b: u32, c: u32, d: u32) {
            let temp1: u32 = a + b;
            let temp2: u32 = temp1 * c;
            let result: u32 = temp2 - a;
            _irf[d] = result;
        }
        """)
        mlir.assert_func("flow_compute", function_type="(i32, i32, i32, i32) -> ()")
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("arith.muli", exactly=1)
        mlir.assert_op_count("arith.subi", exactly=1)
        mlir.assert_op_count("aps.writerf", exactly=1)

    def test_bitwise_operations(self):
        mlir = verify_mlir("""
        rtype bitwise(a: u32, b: u32, c: u32) {
            let and_result: u32 = a & b;
            let or_result: u32 = a | b;
            let xor_result: u32 = a ^ b;
            _irf[c] = and_result + or_result + xor_result;
        }
        """)
        mlir.assert_func("flow_bitwise")
        mlir.assert_op_count("arith.andi", exactly=1)
        mlir.assert_op_count("arith.ori", exactly=1)
        mlir.assert_op_count("arith.xori", exactly=1)
        mlir.assert_op_count("arith.addi", exactly=2)

    def test_shift_operations(self):
        mlir = verify_mlir("""
        rtype shifter(value: u32, amount: u5, rd: u5) {
            let left: u32 = value << amount;
            let right: u32 = value >> amount;
            _irf[rd] = left + right;
        }
        """)
        mlir.assert_func("flow_shifter", function_type="(i32, i5, i5) -> ()")
        mlir.assert_op_count("arith.shli", exactly=1)
        mlir.assert_op_count("arith.shrui", exactly=1)
        mlir.assert_op_count("arith.extui", at_least=1)
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_operand_types("arith.shli", ["i32", "i32"])
        mlir.assert_operand_types("arith.shrui", ["i32", "i32"])
        mlir.assert_operand_producer(mlir.single_op("arith.shli"), 1, "arith.extui")
        mlir.assert_operand_producer(mlir.single_op("arith.shrui"), 1, "arith.extui")

    def test_comparison_operations(self):
        mlir = verify_mlir("""
        rtype compare(a: u32, b: u32, rd: u32) {
            let eq: u1 = a == b;
            let lt: u1 = a < b;
            let result: u32 = if eq {1} else {0};
            _irf[rd] = result;
        }
        """)
        mlir.assert_func("flow_compare")
        # The `lt` binding is unused and is eliminated by the frontend CSE pass.
        mlir.assert_op_count("arith.cmpi", exactly=1)
        mlir.assert_op_count("arith.select", exactly=1)
        mlir.assert_result_types("arith.cmpi", ["i1"])


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestRegisterFileMLIR:
    def test_irf_read(self):
        mlir = verify_mlir("""
        rtype test(rs1: u5, rd: u5) {
            let value: u32 = _irf[rs1];
            _irf[rd] = value;
        }
        """)
        mlir.assert_func("flow_test", function_type="(i5, i5) -> ()")
        mlir.assert_op_count("aps.readrf", exactly=1)
        mlir.assert_op_count("aps.writerf", exactly=1)
        mlir.assert_operand_types("aps.readrf", ["i5"])
        mlir.assert_result_types("aps.readrf", ["i32"])
        writerf = mlir.single_op("aps.writerf")
        mlir.assert_operand_producer(writerf, 1, "aps.readrf")
        mlir.assert_result_has_user(mlir.single_op("aps.readrf"), "aps.writerf")

    def test_irf_arithmetic(self):
        mlir = verify_mlir("""
        rtype add(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """)
        mlir.assert_func("flow_add")
        mlir.assert_op_count("aps.readrf", exactly=2)
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("aps.writerf", exactly=1)
        addi = mlir.single_op("arith.addi")
        writerf = mlir.single_op("aps.writerf")
        mlir.assert_operand_producer(addi, 0, "aps.readrf")
        mlir.assert_operand_producer(addi, 1, "aps.readrf")
        mlir.assert_operand_producer(writerf, 1, "arith.addi")

    def test_irf_complex(self):
        mlir = verify_mlir("""
        rtype complex(rs1: u5, rs2: u5, rs3: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            let r3: u32 = _irf[rs3];
            let result: u32 = (r1 + r2) * r3;
            _irf[rd] = result;
        }
        """)
        mlir.assert_func("flow_complex")
        mlir.assert_op_count("aps.readrf", exactly=3)
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("arith.muli", exactly=1)
        mlir.assert_op_count("aps.writerf", exactly=1)
        addi = mlir.single_op("arith.addi")
        muli = mlir.single_op("arith.muli")
        writerf = mlir.single_op("aps.writerf")
        mlir.assert_operand_producer(addi, 0, "aps.readrf")
        mlir.assert_operand_producer(addi, 1, "aps.readrf")
        mlir.assert_operand_producer(muli, 0, "arith.addi")
        mlir.assert_operand_producer(muli, 1, "aps.readrf")
        mlir.assert_operand_producer(writerf, 1, "arith.muli")


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestMemoryMLIR:
    def test_mem_read(self):
        mlir = verify_mlir("""
        rtype test(addr: u32, rd: u5) {
            let value: u32 = _mem[addr];
            _irf[rd] = value;
        }
        """)
        mlir.assert_global("_cpu_memory", type_="memref<1024xi32>")
        mlir.assert_op_count("memref.get_global", exactly=1)
        mlir.assert_op_count("aps.memload", exactly=1)
        mlir.assert_operand_types("aps.memload", ["memref<1024xi32>", "i32"])
        mlir.assert_result_types("aps.memload", ["i32"])
        load = mlir.single_op("aps.memload")
        writerf = mlir.single_op("aps.writerf")
        mlir.assert_operand_producer(load, 0, "memref.get_global")
        mlir.assert_operand_producer(load, 1, None)
        mlir.assert_operand_producer(writerf, 1, "aps.memload")

    def test_mem_write(self):
        mlir = verify_mlir("""
        rtype test(addr: u32, value: u32) {
            _mem[addr] = value;
        }
        """)
        mlir.assert_global("_cpu_memory", type_="memref<1024xi32>")
        mlir.assert_op_count("memref.get_global", exactly=1)
        mlir.assert_op_count("aps.memstore", exactly=1)
        mlir.assert_operand_types("aps.memstore", ["i32", "memref<1024xi32>", "i32"])
        store = mlir.single_op("aps.memstore")
        mlir.assert_operand_producer(store, 0, None)
        mlir.assert_operand_producer(store, 1, "memref.get_global")
        mlir.assert_operand_producer(store, 2, None)

    def test_mem_and_irf(self):
        mlir = verify_mlir("""
        rtype test(rs1: u5, addr: u32, rd: u5) {
            let reg_val: u32 = _irf[rs1];
            let mem_val: u32 = _mem[addr];
            let result: u32 = reg_val + mem_val;
            _irf[rd] = result;
        }
        """)
        mlir.assert_op_count("aps.readrf", exactly=1)
        mlir.assert_op_count("aps.memload", exactly=1)
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("aps.writerf", exactly=1)

    def test_no_memory_when_unused(self):
        mlir = verify_mlir("""
        rtype test(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """)
        mlir.assert_no_op("memref.global")
        mlir.assert_no_op("memref.get_global")
        mlir.assert_no_op("aps.memload")
        mlir.assert_no_op("aps.memstore")


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestStaticMLIR:
    def test_static_simple_scalar_load(self):
        mlir = verify_mlir("""
        static counter: u32 = 42;

        rtype test(rd: u5) {
            _irf[rd] = counter;
        }
        """)
        mlir.assert_global(
            "counter",
            type_="memref<i32>",
            constant=True,
            attrs={"initial_value": "dense<42> : tensor<i32>", "var_name": '"counter"'},
        )
        mlir.assert_op_count("aps.globalload", exactly=1)
        mlir.assert_op_attr("aps.globalload", "global_name", "@counter")
        mlir.assert_no_op("aps.memload")
        mlir.assert_operand_producer(mlir.single_op("aps.writerf"), 1, "aps.globalload")

    def test_static_array(self):
        mlir = verify_mlir("""
        static buffer: [i32; 1024];

        rtype test() {
            buffer[0] = 42;
        }
        """)
        mlir.assert_global("buffer", type_="memref<1024xi32>", attrs={"var_name": '"buffer"'})
        mlir.assert_op_count("memref.get_global", exactly=1)
        mlir.assert_op_count("aps.memstore", exactly=1)
        mlir.assert_operand_types("aps.memstore", ["i32", "memref<1024xi32>", "i32"])
        store = mlir.single_op("aps.memstore")
        mlir.assert_operand_producer(store, 1, "memref.get_global")

    def test_static_array_bare_identifier_is_rejected(self):
        with pytest.raises(TypeError, match="Global array 'buffer' cannot be used"):
            verify_mlir("""
            static buffer: [i32; 1024];

            rtype test(rd: u5) {
                _irf[rd] = buffer;
            }
            """)

    def test_static_with_impl_attribute(self):
        mlir = verify_mlir("""
        #[impl("1rw")]
        static buffer: [i32; 1024];

        rtype test() {
            buffer[0] = 42;
        }
        """)
        mlir.assert_global(
            "buffer",
            type_="memref<1024xi32>",
            attrs={"impl": '"1rw"', "var_name": '"buffer"'},
        )

    def test_static_with_multiple_attributes(self):
        mlir = verify_mlir("""
        #[impl("2rw")]
        #[partition("cyclic")]
        static scratch: [i32; 512];

        rtype test() {
            scratch[0] = 1;
        }
        """)
        mlir.assert_global(
            "scratch",
            type_="memref<512xi32>",
            attrs={"impl": '"2rw"', "partition": '"cyclic"', "var_name": '"scratch"'},
        )


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestCSRRegisterMLIR:
    def test_csr_register_lowers_to_global_with_address_attr(self):
        mlir = verify_mlir("""
        #[csr_address(0x801)]
        register cfg: u32;

        rtype test(rd: u5) {
            let value: u32 = _csr[cfg];
            _irf[rd] = value;
        }
        """)
        mlir.assert_global(
            "cfg",
            type_="memref<i32>",
            attrs={"csr_address": "2049 : i32", "var_name": '"cfg"'},
        )
        mlir.assert_op_count("aps.readcsr", exactly=1)
        mlir.assert_op_attr("aps.readcsr", "global_name", "@cfg")
        mlir.assert_operand_producer(mlir.single_op("aps.writerf"), 1, "aps.readcsr")

    def test_csr_register_store_uses_writecsr(self):
        mlir = verify_mlir("""
        #[csr_address(0x801)]
        register cfg: u32;

        rtype test(value: u32) {
            _csr[cfg] = value;
        }
        """)
        mlir.assert_global(
            "cfg",
            type_="memref<i32>",
            attrs={"csr_address": "2049 : i32", "var_name": '"cfg"'},
        )
        mlir.assert_op_count("aps.writecsr", exactly=1)
        mlir.assert_op_attr("aps.writecsr", "global_name", "@cfg")


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestControlFlowMLIR:
    def test_if_expression(self):
        mlir = verify_mlir("""
        rtype test(a: u32, b: u32, rd: u5) {
            let result: u32 = if (a > b) {a} else {b};
            _irf[rd] = result;
        }
        """)
        mlir.assert_op_count("arith.cmpi", exactly=1)
        mlir.assert_op_count("arith.select", exactly=1)
        mlir.assert_result_types("arith.select", ["i32"])
        select = mlir.single_op("arith.select")
        writerf = mlir.single_op("aps.writerf")
        mlir.assert_operand_producer(select, 0, "arith.cmpi")
        mlir.assert_operand_producer(writerf, 1, "arith.select")

    def test_nested_if(self):
        mlir = verify_mlir("""
        rtype test(a: u32, b: u32, c: u32, rd: u5) {
            let result: u32 = if (a > b) {
                if (a > c) {a} else {c}
            } else {
                if (b > c) {b} else {c}
            };
            _irf[rd] = result;
        }
        """)
        mlir.assert_op_count("arith.cmpi", exactly=3)
        mlir.assert_op_count("arith.select", exactly=3)

    def test_select_expression_simple(self):
        mlir = verify_mlir("""
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: 10,
                x > 0: 20,
            };
            _irf[rd] = result;
        }
        """)
        # The final arm is the default fallback and does not require a select.
        mlir.assert_op_count("arith.cmpi", exactly=1)
        mlir.assert_op_count("arith.select", exactly=1)

    def test_select_expression_multiple_arms(self):
        mlir = verify_mlir("""
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: 100,
                x < 10: 200,
                x < 20: 300,
                x >= 20: 400,
            };
            _irf[rd] = result;
        }
        """)
        # The final `x >= 20` arm becomes the fallback after previous predicates.
        mlir.assert_op_count("arith.cmpi", exactly=3)
        mlir.assert_op_count("arith.select", exactly=3)

    def test_select_with_complex_values(self):
        mlir = verify_mlir("""
        rtype test(rs1: u5, rd: u5) {
            let x: u32 = _irf[rs1];
            let result: u32 = sel {
                x == 0: x + 10,
                x == 1: x * 20,
                x > 10: x << 2,
                1: 0,
            };
            _irf[rd] = result;
        }
        """)
        # The `1: 0` arm is the default fallback.
        mlir.assert_op_count("arith.select", exactly=3)
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("arith.muli", exactly=1)
        mlir.assert_op_count("arith.shli", exactly=1)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestLoopMLIR:
    def test_counted_do_while_raises_to_scf_for(self):
        mlir = verify_mlir("""
        rtype loop_for(rs1: u5, rd: u5) {
            let sum0: u32 = _irf[rs1];
            let i0: u32 = 0;
            with
                i: u32 = (i0, i + 1)
                sum: u32 = (sum0, sum + 4)
            do {
                let sum_: u32 = sum + 4;
                let i_: u32 = i + 1;
            } while (i < 8);

            _irf[rd] = sum;
        }
        """)
        mlir.assert_func("flow_loop_for", function_type="(i5, i5) -> ()")
        mlir.assert_op_count("scf.for", exactly=1)
        mlir.assert_no_op("scf.while")
        mlir.assert_op_count("scf.yield", exactly=1)

        loop = mlir.single_op("scf.for")
        mlir.assert_operand_types("scf.for", ["i32", "i32", "i32", "i32"])
        mlir.assert_result_types("scf.for", ["i32"])
        mlir.assert_region_block_arg_types(loop, [["i32", "i32"]])
        mlir.assert_region_block_ops(loop, [["arith.constant", "arith.addi", "scf.yield"]])

        mlir.assert_operand_producer(loop, 0, "arith.constant")
        mlir.assert_operand_producer(loop, 1, "arith.constant")
        mlir.assert_operand_producer(loop, 2, "arith.constant")
        mlir.assert_operand_producer(loop, 3, "aps.readrf")
        mlir.assert_operand_producer(mlir.single_op("aps.writerf"), 1, "scf.for")

    def test_dynamic_bound_do_while_lowers_to_scf_while(self):
        mlir = verify_mlir("""
        rtype loop_while(rs1: u5, rs2: u5, rd: u5) {
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
        mlir.assert_func("flow_loop_while", function_type="(i5, i5, i5) -> ()")
        mlir.assert_op_count("scf.while", exactly=1)
        mlir.assert_no_op("scf.for")
        mlir.assert_op_count("scf.condition", exactly=1)
        mlir.assert_op_count("scf.yield", exactly=1)
        mlir.assert_op_count("arith.cmpi", exactly=1)

        loop = mlir.single_op("scf.while")
        mlir.assert_operand_types("scf.while", ["i32", "i32", "i32", "i1"])
        mlir.assert_result_types("scf.while", ["i32", "i32", "i32", "i1"])
        mlir.assert_region_block_arg_types(loop, [["i32", "i32", "i32", "i1"],
                                                 ["i32", "i32", "i32", "i1"]])
        mlir.assert_region_block_ops(loop, [["scf.condition"],
                                           ["arith.constant", "arith.constant", "arith.cmpi",
                                            "arith.addi", "arith.addi", "scf.yield"]])

        mlir.assert_operand_producer(loop, 0, "arith.constant")
        mlir.assert_operand_producer(loop, 1, "aps.readrf")
        mlir.assert_operand_producer(loop, 2, "aps.readrf")
        mlir.assert_operand_producer(loop, 3, "arith.constant")
        mlir.assert_operand_producer(mlir.single_op("aps.writerf"), 1, "scf.while")


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestComplexExamples:
    def test_risc_v_add(self):
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
        mlir.assert_op_count("aps.readrf", exactly=2)
        mlir.assert_op_count("arith.addi", exactly=1)
        mlir.assert_op_count("aps.writerf", exactly=1)
        addi = mlir.single_op("arith.addi")
        writerf = mlir.single_op("aps.writerf")
        mlir.assert_operand_producer(addi, 0, "aps.readrf")
        mlir.assert_operand_producer(addi, 1, "aps.readrf")
        mlir.assert_operand_producer(writerf, 1, "arith.addi")

    def test_memory_accumulate(self):
        mlir = verify_mlir("""
        #[opcode(7'b1011011)]
        #[funct7(7'b0000000)]
        rtype accum(rs1: u5, rd: u5) {
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
        mlir.assert_global("_cpu_memory", type_="memref<1024xi32>")
        mlir.assert_op_count("aps.memload", exactly=4)
        mlir.assert_op_count("aps.memstore", exactly=1)
        mlir.assert_op_count("arith.addi", at_least=7)
        for load in mlir.ops_named("aps.memload"):
            mlir.assert_operand_producer(load, 0, "memref.get_global")
        mlir.assert_operand_producer(mlir.single_op("aps.memstore"), 1, "memref.get_global")

    def test_crc8_simplified(self):
        mlir = verify_mlir("""
        #[opcode(7'b0101011)]
        #[funct7(7'b0000000)]
        rtype crc8(rs1: u5, rd: u5) {
            let x0: u32 = _irf[rs1];
            let flag: u32 = x0 >> 7;
            let x_shifted: u32 = x0 << 1;
            let x_final: u32 = if (flag != 0) { x_shifted ^ 7 } else { x_shifted };
            _irf[rd] = x_final;
        }
        """)
        mlir.assert_func("flow_crc8", opcode=43, funct7=0)
        mlir.assert_op_count("arith.shrui", exactly=1)
        mlir.assert_op_count("arith.shli", exactly=1)
        mlir.assert_op_count("arith.xori", exactly=1)
        mlir.assert_op_count("arith.select", exactly=1)


@pytest.mark.skipif(not MLIR_AVAILABLE, reason="MLIR/CIRCT bindings not available")
class TestMLIRStructure:
    def test_module_structure(self):
        mlir = verify_mlir("""
        rtype test(a: u32, b: u32, rd: u5) {
            _irf[rd] = a + b;
        }
        """)
        mlir.assert_op_count("builtin.module", exactly=1)
        mlir.assert_func("flow_test", function_type="(i32, i32, i5) -> ()")
        mlir.assert_op_count("func.return", exactly=1)

    def test_ssa_values(self):
        mlir = verify_mlir("""
        rtype complex(a: u32, b: u32, c: u32, rd: u5) {
            let x: u32 = a + b;
            let y: u32 = x * 2;
            let z: u32 = y - a;
            _irf[rd] = z;
        }
        """)
        for op_name in ["arith.addi", "arith.muli", "arith.subi"]:
            mlir.assert_result_types(op_name, ["i32"])
        mlir.assert_op_count("aps.writerf", exactly=1)

    def test_type_consistency(self):
        mlir = verify_mlir("""
        rtype typed(a: u32, b: u32, c: u32, rd: u5) {
            let x: u32 = a;
            let y: u32 = b;
            _irf[rd] = x + y;
        }
        """)
        mlir.assert_func("flow_typed", function_type="(i32, i32, i32, i5) -> ()")
        mlir.assert_operand_types("arith.addi", ["i32", "i32"])
        mlir.assert_result_types("arith.addi", ["i32"])
        mlir.assert_operand_producer(mlir.single_op("aps.writerf"), 1, "arith.addi")
