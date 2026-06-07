from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
APS_OPT = REPO_ROOT / "build" / "tools" / "aps-opt" / "aps-opt"

FUNCTIONAL_TO_ARCH_PIPELINE = (
    "builtin.module("
    "func.func(place-readrf-at-entry),"
    "auto-burst-partition,"
    "aps-memory-map,"
    "aps-raise-scf-to-affine,"
    "canonicalize,"
    "func.func(raise-memref-to-affine),"
    "raise-memref-to-affine,"
    "canonicalize,"
    "hls-unroll,"
    "cse,"
    "canonicalize,"
    "func.func(affine-loop-normalize),"
    "canonicalize,"
    "new-array-partition,"
    "canonicalize,"
    "func.func(affine-mem-to-aps,memref-to-aps),"
    "aps-functional-to-arch,"
    "func.func(promote-singleton-memref-to-global)"
    ")"
)


def run_functional_to_arch(tmp_path: Path, source: str) -> str:
    if not APS_OPT.exists():
        pytest.skip(f"{APS_OPT} is not built")
    if shutil.which("pixi") is None:
        pytest.skip("pixi is not available")

    cadl_file = tmp_path / "input.cadl"
    mlir_file = tmp_path / "input.mlir"
    output_file = tmp_path / "output.mlir"
    cadl_file.write_text(source)

    subprocess.run(
        ["pixi", "run", "mlir", str(cadl_file), str(mlir_file)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(APS_OPT),
            "-p",
            FUNCTIONAL_TO_ARCH_PIPELINE,
            str(mlir_file),
            "-o",
            str(output_file),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return output_file.read_text()


def test_full_partition_copy_in_lowers_to_scalar_loads(tmp_path: Path):
    output = run_functional_to_arch(
        tmp_path,
        """
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static buf: [u32; 4];

rtype full_copy(rs1: u5, rd: u5) {
  let addr: u32 = _irf[rs1];
  buf[0 +: ] = _mem[addr +: 4];
  _irf[rd] = 0;
}
""",
    )

    assert output.count("aps.load_by") == 4
    assert output.count("aps.globalstore") == 4
    assert "aps.copy_by" not in output
    assert "aps.copy " not in output


def test_full_partition_copy_out_lowers_to_scalar_stores(tmp_path: Path):
    output = run_functional_to_arch(
        tmp_path,
        """
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static buf: [u32; 4];

rtype full_copy_out(rs1: u5, rd: u5) {
  let addr: u32 = _irf[rs1];
  _mem[addr +: 4] = buf[0 +: ];
  _irf[rd] = 0;
}
""",
    )

    assert output.count("aps.globalload") == 4
    assert output.count("aps.store_by") == 4
    assert "aps.copy_by" not in output
    assert "aps.copy " not in output


def test_partial_partition_copy_remains_arch_copy(tmp_path: Path):
    output = run_functional_to_arch(
        tmp_path,
        """
#[partition_dim_array([0])]
#[partition_factor_array([2])]
#[partition_cyclic_array([1])]
static buf: [u32; 4];

rtype partial_copy(rs1: u5, rd: u5) {
  let addr: u32 = _irf[rs1];
  buf[0 +: ] = _mem[addr +: 4];
  _irf[rd] = 0;
}
""",
    )

    assert output.count("aps.copy_by") == 1
    assert "aps.copy " not in output
