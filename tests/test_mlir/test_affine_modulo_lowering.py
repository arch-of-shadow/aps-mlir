from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APS_OPT = ROOT / "build/tools/aps-opt/aps-opt"
MLIR_BIN = ROOT / "circt/build/bin"


def lower(tmp_path: Path, source: str, *passes: str) -> str:
    if not APS_OPT.exists():
        pytest.skip("aps-opt is not built")
    input_file = tmp_path / "input.mlir"
    input_file.write_text(source)
    result = subprocess.run(
        [str(APS_OPT), str(input_file), "--allow-unregistered-dialect", *passes],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.mark.parametrize("divisor", [1, 2, 4, 8, 3])
@pytest.mark.parametrize("lower_bound", [0, -3])
def test_affine_modulo_in_narrowed_loop(tmp_path, divisor, lower_bound):
    source = f"""
module {{
  func.func @bank_sum() -> i32 {{
    %zero = arith.constant 0 : i32
    %sum = affine.for %i = {lower_bound} to 5 iter_args(%acc = %zero) -> (i32) {{
      %bank = affine.apply affine_map<(d0) -> (d0 mod {divisor})>(%i)
      %value = arith.index_cast %bank : index to i32
      %next = arith.addi %acc, %value : i32
      affine.yield %next : i32
    }}
    return %sum : i32
  }}
}}
"""
    result = lower(tmp_path, source, "--lower-affine-for", "--canonicalize")
    assert "affine." not in result
    if divisor != 1:
        assert "scf.for" in result
    if divisor == 3:
        assert "arith.remsi" in result
    else:
        assert "arith.remsi" not in result
        if divisor > 1:
            assert "arith.andi" in result


def test_preexpanded_remainder_preserves_operand_types(tmp_path):
    result = lower(
        tmp_path,
        """
module {
  func.func @preexpanded(%mem: memref<3xi32>) {
    %three = arith.constant 3 : index
    affine.for %i = 0 to 5 {
      %bank = arith.remsi %i, %three : index
      %v = memref.load %mem[%bank] : memref<3xi32>
      "test.consume"(%v) : (i32) -> ()
    }
    return
  }
}
""",
        "--lower-affine-for",
        "--canonicalize",
    )
    assert "arith.remsi" in result
    assert "affine.for" not in result


def test_nested_conditional_preserves_captured_index_types(tmp_path):
    result = lower(tmp_path, """
module {
  func.func @conditional(%flag: i1, %mem: memref<40xi32>) {
    affine.for %i = 0 to 40 {
      scf.if %flag {
        %value = arith.index_cast %i : index to i32
        memref.store %value, %mem[%i] : memref<40xi32>
      }
    }
    return
  }
}
""", "--lower-affine-for", "--canonicalize")
    assert "scf.if" in result
    assert "memref.store" in result
    assert "affine.for" not in result


def test_nested_loops_and_modulo_expression(tmp_path):
    result = lower(
        tmp_path,
        """
module {
  func.func @nested() {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 6 {
        %bank = affine.apply affine_map<(d0, d1) -> ((d0 + d1) mod 4)>(%i, %j)
        "test.consume"(%bank) : (index) -> ()
      }
    }
    return
  }
}
""",
        "--lower-affine-for",
        "--canonicalize",
    )
    assert "affine." not in result
    assert result.count("scf.for") == 2
    assert "arith.andi" in result
    assert "arith.remsi" not in result


def test_csr_scale_synthesizes_without_unrolling(tmp_path):
    if not APS_OPT.exists():
        pytest.skip("aps-opt is not built")
    source = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/aps-frontend"),
            "mlir",
            str(ROOT / "examples/megg_example/csr_scale/csr_scale.cadl"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    input_file = tmp_path / "csr_scale.mlir"
    output_file = tmp_path / "csr_scale.sv"
    input_file.write_text(source.stdout)
    subprocess.run(
        [
            str(APS_OPT.with_name("aps-e2e")),
            "--clock=4.0",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--export=sv",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    rtl = output_file.read_text()
    assert "module main(" in rtl
    assert "csr_gain_cfg_value_" in rtl
    assert "csr_bias_cfg_value_" in rtl


@pytest.mark.parametrize("divisor", [2, 4, 2147483648])
def test_unsigned_remainder_becomes_mask(tmp_path, divisor):
    result = lower(
        tmp_path,
        f"""
module {{
  func.func @unsigned_rem(%x: i32) -> i32 {{
    %divisor = arith.constant {divisor} : i32
    %r = arith.remui %x, %divisor : i32
    return %r : i32
  }}
}}
""",
        "--arith-muldiv-to-shift",
    )
    assert "arith.remui" not in result
    assert "arith.andi" in result


@pytest.mark.parametrize("divisor", [0, 2, 4, 3, -2])
def test_signed_remainder_not_masked_without_proof(tmp_path, divisor):
    result = lower(
        tmp_path,
        f"""
module {{
  func.func @signed_rem(%x: i32) -> i32 {{
    %divisor = arith.constant {divisor} : i32
    %r = arith.remsi %x, %divisor : i32
    return %r : i32
  }}
}}
""",
        "--arith-muldiv-to-shift",
    )
    assert "arith.remsi" in result
    assert "arith.andi" not in result


@pytest.mark.parametrize("upper,step,masked", [(5, 1, True), (7, 3, False)])
def test_signed_loop_remainder_requires_nonwrapping_control(
    tmp_path, upper, step, masked
):
    result = lower(
        tmp_path,
        f"""
module {{
  func.func @signed_loop() {{
    %zero = arith.constant 0 : i4
    %upper = arith.constant {upper} : i4
    %step = arith.constant {step} : i4
    %two = arith.constant 2 : i4
    scf.for %i = %zero to %upper step %step : i4 {{
      %r = arith.remsi %i, %two : i4
      "test.consume"(%r) : (i4) -> ()
    }}
    return
  }}
}}
""",
        "--arith-muldiv-to-shift",
    )
    assert ("arith.andi" in result) == masked
    assert ("arith.remsi" in result) != masked


def test_modulo_semantics_execute_correctly(tmp_path):
    result = lower(
        tmp_path,
        """
module {
  func.func @affine_mod(%x: index) -> index {
    %r = affine.apply affine_map<(d0) -> (d0 mod 4)>(%x)
    return %r : index
  }
  func.func @signed_rem(%x: i32) -> i32 {
    %four = arith.constant 4 : i32
    %r = arith.remsi %x, %four : i32
    return %r : i32
  }
  func.func @unsigned_rem(%x: i32) -> i32 {
    %four = arith.constant 4 : i32
    %r = arith.remui %x, %four : i32
    return %r : i32
  }
  func.func @unsigned_large(%x: i32) -> i32 {
    %divisor = arith.constant 2147483648 : i32
    %r = arith.remui %x, %divisor : i32
    return %r : i32
  }
}
""",
        "--lower-affine-for",
        "--arith-muldiv-to-shift",
        "--canonicalize",
    )
    assert "arith.remsi" in result
    source = tmp_path / "lowered.mlir"
    llvm_mlir = tmp_path / "llvm.mlir"
    llvm_ir = tmp_path / "test.ll"
    source.write_text(result)
    subprocess.run(
        [
            str(MLIR_BIN / "mlir-opt"),
            "--test-lower-to-llvm",
            str(source),
            "-o",
            str(llvm_mlir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    translation = subprocess.run(
        [str(MLIR_BIN / "mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir)],
        check=True,
        capture_output=True,
        text=True,
    )
    llvm_ir.write_text(translation.stdout)
    harness = tmp_path / "check.c"
    harness.write_text(
        """
#include <stdint.h>
#include <limits.h>
extern int64_t affine_mod(int64_t);
extern int32_t signed_rem(int32_t);
extern uint32_t unsigned_rem(uint32_t);
extern uint32_t unsigned_large(uint32_t);
static int check(int32_t x) {
  uint32_t u = (uint32_t)x;
  return affine_mod(x) != ((x % 4 + 4) % 4)
      || signed_rem(x) != x % 4 || unsigned_rem(u) != u % 4
      || unsigned_large(u) != u % UINT32_C(2147483648);
}
int main(void) {
  for (int32_t x = -257; x <= 257; ++x)
    if (check(x)) return 1;
  return check(INT_MIN) || check(INT_MAX);
}
"""
    )
    binary = tmp_path / "check"
    subprocess.run(
        ["clang", "-O2", str(llvm_ir), str(harness), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_symbolic_banks_preserve_values_and_avoid_global_name_collisions(tmp_path):
    result = lower(
        tmp_path,
        """
module {
  memref.global @buffer : memref<8xi32> = dense<[1,2,3,4,5,6,7,8]> {partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], partition_cyclic_array = [1 : i32]}
  memref.global @buffer_0 : memref<8xi32> = dense<[11,12,13,14,15,16,17,18]> {partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], partition_cyclic_array = [1 : i32]}
  func.func @read_banks(%i: index) -> i32 {
    %a = memref.get_global @buffer : memref<8xi32>
    %b = memref.get_global @buffer_0 : memref<8xi32>
    %x = affine.load %a[symbol(%i)] : memref<8xi32>
    %y = affine.load %b[symbol(%i)] : memref<8xi32>
    %sum = arith.addi %x, %y : i32
    return %sum : i32
  }
}
""",
        "--new-array-partition",
        "--canonicalize",
    )
    assert result.count("memref.global") == 4
    source = tmp_path / "partitioned.mlir"
    llvm_mlir = tmp_path / "llvm.mlir"
    llvm_ir = tmp_path / "test.ll"
    source.write_text(result)
    subprocess.run(
        [str(MLIR_BIN / "mlir-opt"), "--test-lower-to-llvm", str(source), "-o", str(llvm_mlir)],
        check=True, capture_output=True, text=True,
    )
    translation = subprocess.run(
        [str(MLIR_BIN / "mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir)],
        check=True, capture_output=True, text=True,
    )
    llvm_ir.write_text(translation.stdout)
    harness = tmp_path / "check.c"
    harness.write_text("""
#include <stdint.h>
extern int32_t read_banks(int64_t);
int main(void) {
  for (int64_t i = 0; i < 8; ++i)
    if (read_banks(i) != 12 + 2 * i) return 1;
  return 0;
}
""")
    binary = tmp_path / "check"
    subprocess.run(
        ["clang", "-O2", str(llvm_ir), str(harness), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.mark.parametrize("factor", [2, 4])
def test_nonaffine_banks_preserve_loads_and_guard_stores(tmp_path, factor):
    result = lower(tmp_path, f"""
module {{
  memref.global @buffer : memref<8xi32> = dense<[1,2,3,4,5,6,7,8]> {{partition_dim_array = [0 : i32], partition_factor_array = [{factor} : i32], partition_cyclic_array = [1 : i32]}}
  func.func @write_slot(%raw: index, %value: i32) {{
    %zero = arith.constant 0 : index
    %last = arith.constant 7 : index
    %negative = arith.cmpi slt, %raw, %zero : index
    %lo = arith.select %negative, %zero, %raw : index
    %large = arith.cmpi sgt, %lo, %last : index
    %index = arith.select %large, %last, %lo : index
    %mem = memref.get_global @buffer : memref<8xi32>
    memref.store %value, %mem[%index] : memref<8xi32>
    return
  }}
  func.func @read_slot(%index: index) -> i32 {{
    %mem = memref.get_global @buffer : memref<8xi32>
    %value = memref.load %mem[%index] : memref<8xi32>
    return %value : i32
  }}
}}
""", "--new-array-partition", "--canonicalize")
    assert result.count("memref.global") == factor
    assert result.count("aps.write_smem_if") == factor
    # Execute both reads and writes; every store must preserve all other banks.
    source = tmp_path / "partitioned.mlir"
    source.write_text(result)
    lowered = subprocess.run(
        [str(APS_OPT), str(source), "--aps-mem-to-memref"],
        check=True, capture_output=True, text=True,
    )
    source.write_text(lowered.stdout)
    llvm_mlir = tmp_path / "llvm.mlir"
    subprocess.run(
        [str(MLIR_BIN / "mlir-opt"), "--test-lower-to-llvm", str(source), "-o", str(llvm_mlir)],
        check=True, capture_output=True, text=True,
    )
    translation = subprocess.run(
        [str(MLIR_BIN / "mlir-translate"), "--mlir-to-llvmir", str(llvm_mlir)],
        check=True, capture_output=True, text=True,
    )
    llvm_ir = tmp_path / "test.ll"
    llvm_ir.write_text(translation.stdout)
    harness = tmp_path / "check.c"
    harness.write_text("""
#include <stdint.h>
extern void write_slot(int64_t, int32_t);
extern int32_t read_slot(int64_t);
int main(void) {
  int32_t expected[8] = {1,2,3,4,5,6,7,8};
  for (int i = 0; i < 128; ++i) {
    int index = (i * 7) % 13 - 3;
    int clamped = index < 0 ? 0 : index > 7 ? 7 : index;
    expected[clamped] = i * 19 - 37;
    write_slot(index, expected[clamped]);
    for (int j = 0; j < 8; ++j)
      if (read_slot(j) != expected[j]) return 1;
  }
  return 0;
}
""")
    binary = tmp_path / "check"
    subprocess.run(
        ["clang", "-O2", str(llvm_ir), str(harness), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(binary)], check=True, timeout=10)
