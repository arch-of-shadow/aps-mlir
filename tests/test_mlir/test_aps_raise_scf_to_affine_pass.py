from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
APS_OPT = REPO_ROOT / "build" / "tools" / "aps-opt" / "aps-opt"
REAL_CASE_DIR = REPO_ROOT / "outputs" / "cmt2_real_cases"
REAL_CASES = sorted(REAL_CASE_DIR.glob("*.aps.mlir"))
REAL_MEMORY_PIPELINE = [
    "--allow-unregistered-dialect",
    "--aps-raise-scf-to-affine",
    "--canonicalize",
    "--affine-raise-from-memref",
    "--raise-memref-to-affine",
    "--canonicalize",
]
EXPECTED_RESIDUAL_MEMREF_LOADS = {
    "crypto_pqc": 1,
    "crypto_pqc_zip": 1,
}


def run_aps_raise(
    tmp_path: Path,
    source: str,
    *,
    extra_passes: list[str] | None = None,
) -> str:
    if not APS_OPT.exists():
        pytest.skip(f"{APS_OPT} is not built")

    input_file = tmp_path / "input.mlir"
    output_file = tmp_path / "output.mlir"
    input_file.write_text(source)

    cmd = [
        str(APS_OPT),
        str(input_file),
        "--allow-unregistered-dialect",
        "--aps-raise-scf-to-affine",
    ]
    cmd.extend(extra_passes or [])
    cmd.extend(["-o", str(output_file)])

    subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return output_file.read_text()


def run_aps_opt_file(tmp_path: Path, input_file: Path, passes: list[str]) -> str:
    if not APS_OPT.exists():
        pytest.skip(f"{APS_OPT} is not built")

    output_file = tmp_path / f"{input_file.stem}.out.mlir"
    cmd = [str(APS_OPT), str(input_file)]
    cmd.extend(passes)
    cmd.extend(["-o", str(output_file)])

    subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return output_file.read_text()


def run_aps_opt_source(
    tmp_path: Path, source: str, passes: list[str]
) -> tuple[str, str]:
    if not APS_OPT.exists():
        pytest.skip(f"{APS_OPT} is not built")

    input_file = tmp_path / "input.mlir"
    output_file = tmp_path / "output.mlir"
    input_file.write_text(source)

    cmd = [str(APS_OPT), str(input_file)]
    cmd.extend(passes)
    cmd.extend(["-o", str(output_file)])

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return output_file.read_text(), result.stderr + result.stdout


def test_raises_u8_iv_static_bound_and_preserves_body_use_width(tmp_path: Path):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @u8_loop(%x: i32) -> i32 {
    %c0 = arith.constant 0 : i8
    %c8 = arith.constant 8 : i8
    %c1 = arith.constant 1 : i8
    %0 = scf.for %i = %c0 to %c8 step %c1 iter_args(%sum = %x) -> (i32) : i8 {
      %i32 = arith.extui %i : i8 to i32
      %next = arith.addi %sum, %i32 : i32
      scf.yield %next : i32
    }
    return %0 : i32
  }
}
""",
    )

    assert "affine.for" in output
    assert "aps.original_iv_type" not in output
    assert "scf.for" not in output
    assert "arith.index_cast" in output
    assert ": index to i8" in output
    assert "arith.extui" in output
    assert "affine.yield" in output


def test_scf_to_tor_legalizes_residual_u8_for_without_conversion_error(
    tmp_path: Path,
):
    output, log = run_aps_opt_source(
        tmp_path,
        """
module {
  func.func @flow_iv_u8_loop(%arg0: i5, %arg1: i5) attributes {funct7 = 122 : i32, opcode = 91 : i32} {
    %0 = aps.read_irf %arg0 : i5 -> i32
    %c0_i8 = arith.constant 0 : i8
    %c8_i32 = arith.constant 8 : i32
    %ub = arith.trunci %c8_i32 : i32 to i8
    %c1_i8 = arith.constant 1 : i8
    %1 = scf.for %i = %c0_i8 to %ub step %c1_i8 iter_args(%sum = %0) -> (i32) : i8 {
      %i32 = arith.extui %i : i8 to i32
      %next = arith.addi %sum, %i32 : i32
      scf.yield %next : i32
    }
    aps.write_irf %arg1, %1 : i5, i32
    return
  }
}
""",
        [
            "--allow-unregistered-dialect",
            "--convert-input=clock=4.0 resource=examples/resource_ihp130.json output-path=/tmp",
            "--scf-to-tor",
            "--canonicalize",
        ],
    )

    assert "failed to legalize operation 'scf.for'" not in log
    assert "error:" not in log
    assert "scf.for" not in output
    assert "unrealized_conversion_cast" not in output
    assert "tor.for" in output
    assert ": i4) to" in output
    assert "step (%c1_i4 : i4)" in output


def test_lower_affine_for_infers_static_control_width(tmp_path: Path):
    input_file = tmp_path / "u8_affine.mlir"
    input_file.write_text(
        """
module {
  func.func @u8_affine(%x: i32) -> i32 {
    %0 = affine.for %i = 0 to 8 iter_args(%sum = %x) -> (i32) {
      %i8 = arith.index_cast %i : index to i8
      %i32 = arith.extui %i8 : i8 to i32
      %next = arith.addi %sum, %i32 : i32
      affine.yield %next : i32
    }
    return %0 : i32
  }
}
"""
    )

    output = run_aps_opt_file(
        tmp_path,
        input_file,
        ["--allow-unregistered-dialect", "--lower-affine-for", "--canonicalize"],
    )

    assert "affine.for" not in output
    assert "scf.for" in output
    assert ": i5 {" in output
    assert ": index to i8" not in output
    assert "arith.extui" in output
    assert ": i5 to i32" in output


def test_lower_affine_for_infers_static_control_width_after_memory_raise(
    tmp_path: Path,
):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @u8_memory_index(%mem: memref<16xi32>, %x: i32) -> i32 {
    %c0 = arith.constant 0 : i8
    %c8 = arith.constant 8 : i8
    %c1 = arith.constant 1 : i8
    %0 = scf.for %i = %c0 to %c8 step %c1 iter_args(%sum = %x) -> (i32) : i8 {
      %idx = arith.index_cast %i : i8 to index
      %v = memref.load %mem[%idx] : memref<16xi32>
      %next = arith.addi %sum, %v : i32
      scf.yield %next : i32
    }
    return %0 : i32
  }
}
""",
        extra_passes=[
            "--canonicalize",
            "--affine-raise-from-memref",
            "--lower-affine-for",
            "--canonicalize",
        ],
    )

    assert "affine.for" not in output
    assert "affine.load" not in output
    assert "scf.for" in output
    assert ": i5 {" in output
    assert "memref.load" in output
    assert "arith.index_cast" in output
    assert ": i5 to index" in output


def test_raises_dynamic_integer_bound_with_local_index_cast(tmp_path: Path):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @dynamic_bound(%ub: i32, %x: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %0 = scf.for %i = %c0 to %ub step %c1 iter_args(%sum = %x) -> (i32) : i32 {
      %next = arith.addi %sum, %i : i32
      scf.yield %next : i32
    }
    return %0 : i32
  }
}
""",
    )

    assert "affine.for" in output
    assert "scf.for" not in output
    assert "arith.index_cast" in output
    assert ": i32 to index" in output
    assert ": index to i32" in output
    assert "affine.yield" in output


def test_raises_multiple_iter_args_without_changing_carry_types(tmp_path: Path):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @multi_iter(%x: i32, %y: i16) -> (i32, i16) {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    %0:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%sum = %x, %acc = %y) -> (i32, i16) : i32 {
      %next_sum = arith.addi %sum, %i : i32
      %next_acc = arith.addi %acc, %acc : i16
      scf.yield %next_sum, %next_acc : i32, i16
    }
    return %0#0, %0#1 : i32, i16
  }
}
""",
    )

    assert "affine.for" in output
    assert "iter_args" in output
    assert "-> (i32, i16)" in output
    assert "affine.yield" in output
    assert ": i32, i16" in output


def test_memory_load_loop_feeds_affine_memory_raise(tmp_path: Path):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @mem_load(%mem: memref<16xi32>, %x: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    %0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%sum = %x) -> (i32) : i32 {
      %idx = arith.index_cast %i : i32 to index
      %v = memref.load %mem[%idx] : memref<16xi32>
      %next = arith.addi %sum, %v : i32
      scf.yield %next : i32
    }
    return %0 : i32
  }
}
""",
        extra_passes=["--canonicalize", "--affine-raise-from-memref"],
    )

    assert "affine.for" in output
    assert "affine.load" in output
    assert "memref.load" not in output
    assert "scf.for" not in output
    assert "index_cast" not in output
    assert "affine.yield" in output


def test_memory_store_loop_feeds_affine_memory_raise(tmp_path: Path):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @mem_store(%mem: memref<16xi32>, %x: i32) {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      %idx = arith.index_cast %i : i32 to index
      memref.store %x, %mem[%idx] : memref<16xi32>
    }
    return
  }
}
""",
        extra_passes=["--canonicalize", "--affine-raise-from-memref"],
    )

    assert "affine.for" in output
    assert "affine.store" in output
    assert "memref.store" not in output
    assert "scf.for" not in output
    assert "index_cast" not in output


def test_aps_memory_loop_feeds_affine_memory_raise(tmp_path: Path):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @aps_mem(%mem: memref<16xi32>, %x: i32) {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      %v = aps.read_smem %mem[%i] : memref<16xi32>, i32 -> i32
      %next = arith.addi %v, %x : i32
      aps.write_smem %next, %mem[%i] : i32, memref<16xi32>, i32
    }
    return
  }
}
""",
        extra_passes=["--canonicalize", "--affine-raise-from-memref"],
    )

    assert "affine.for" in output
    assert "affine.load" in output
    assert "affine.store" in output
    assert "aps.read_smem" not in output
    assert "aps.write_smem" not in output
    assert "memref.load" not in output
    assert "memref.store" not in output


def test_leaves_dynamic_step_loop_unchanged(tmp_path: Path):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @dynamic_step(%lb: i32, %ub: i32, %step: i32, %x: i32) -> i32 {
    %0 = scf.for %i = %lb to %ub step %step iter_args(%sum = %x) -> (i32) : i32 {
      %next = arith.addi %sum, %i : i32
      scf.yield %next : i32
    }
    return %0 : i32
  }
}
""",
    )

    assert "affine.for" not in output
    assert "scf.for" in output
    assert "step %arg2" in output
    assert " : i32 {" in output


def test_dynamic_step_loop_keeps_aps_memory_unchanged(tmp_path: Path):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @dynamic_step_aps_mem(%mem: memref<16xi32>, %lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      %v = aps.read_smem %mem[%i] : memref<16xi32>, i32 -> i32
      aps.write_smem %v, %mem[%i] : i32, memref<16xi32>, i32
    }
    return
  }
}
""",
    )

    assert "affine.for" not in output
    assert "scf.for" in output
    assert "aps.read_smem" in output
    assert "aps.write_smem" in output
    assert "memref.load" not in output
    assert "memref.store" not in output


def test_leaves_negative_step_loop_unchanged(tmp_path: Path):
    output = run_aps_raise(
        tmp_path,
        """
module {
  func.func @negative_step(%x: i32) -> i32 {
    %c8 = arith.constant 8 : i32
    %c0 = arith.constant 0 : i32
    %cm1 = arith.constant -1 : i32
    %0 = scf.for %i = %c8 to %c0 step %cm1 iter_args(%sum = %x) -> (i32) : i32 {
      %next = arith.addi %sum, %i : i32
      scf.yield %next : i32
    }
    return %0 : i32
  }
}
""",
    )

    assert "affine.for" not in output
    assert "scf.for" in output
    assert "constant -1" in output


def test_scf_to_tor_supports_negative_step_loop(tmp_path: Path):
    output, log = run_aps_opt_source(
        tmp_path,
        """
module {
  func.func @negative_step(%arg0: i5, %arg1: i5) attributes {funct7 = 123 : i32, opcode = 91 : i32} {
    %0 = aps.read_irf %arg0 : i5 -> i32
    %c4 = arith.constant 4 : i4
    %c0 = arith.constant 0 : i4
    %cm1 = arith.constant -1 : i4
    %1 = scf.for %i = %c4 to %c0 step %cm1 iter_args(%sum = %0) -> (i32) : i4 {
      %i32 = arith.extui %i : i4 to i32
      %next = arith.addi %sum, %i32 : i32
      scf.yield %next : i32
    }
    aps.write_irf %arg1, %1 : i5, i32
    return
  }
}
""",
        [
            "--allow-unregistered-dialect",
            "--convert-input=clock=4.0 resource=examples/resource_ihp130.json output-path=/tmp",
            "--scf-to-tor",
            "--canonicalize",
        ],
    )

    assert "error:" not in log
    assert "scf.for" not in output
    assert "tor.for" in output
    assert "step (%c-1_i3 : i3)" in output
    assert ": i3) to" in output


def test_aps_to_cmt2_supports_negative_step_loop(tmp_path: Path):
    output, log = run_aps_opt_source(
        tmp_path,
        """
module {
  func.func @negative_step(%arg0: i5, %arg1: i5) attributes {funct7 = 123 : i32, opcode = 91 : i32} {
    %0 = aps.read_irf %arg0 : i5 -> i32
    %c4 = arith.constant 4 : i4
    %c0 = arith.constant 0 : i4
    %cm1 = arith.constant -1 : i4
    %1 = scf.for %i = %c4 to %c0 step %cm1 iter_args(%sum = %0) -> (i32) : i4 {
      %i32 = arith.extui %i : i4 to i32
      %next = arith.addi %sum, %i32 : i32
      scf.yield %next : i32
    }
    aps.write_irf %arg1, %1 : i5, i32
    return
  }
}
""",
        [
            "--allow-unregistered-dialect",
            "--convert-input=clock=4.0 resource=examples/resource_ihp130.json output-path=/tmp",
            "--scf-to-tor",
            "--canonicalize",
            "--schedule-tor",
            "--tor-time-graph",
            "--canonicalize",
            "--aps-to-cmt2",
        ],
    )

    assert "error:" not in log
    assert "Reg_width3_init0" in output
    assert "firrtl.geq" in output
    assert "firrtl.leq" not in output
    assert "firrtl.bits" in output
    assert "!firrtl.uint<3>" in output


@pytest.mark.parametrize("input_file", REAL_CASES, ids=lambda path: path.stem)
def test_real_cases_raise_aps_memory_loops_to_affine(
    tmp_path: Path, input_file: Path
):
    if not REAL_CASES:
        pytest.skip(f"no real APS cases found under {REAL_CASE_DIR}")

    output = run_aps_opt_file(tmp_path, input_file, REAL_MEMORY_PIPELINE)
    case_name = input_file.name.removesuffix(".aps.mlir")

    assert "scf.for" not in output
    assert "affine.for" in output
    assert ("affine.load" in output) or ("affine.store" in output)
    assert output.count("memref.store") == 0
    assert output.count("memref.load") == EXPECTED_RESIDUAL_MEMREF_LOADS.get(
        case_name, 0
    )
