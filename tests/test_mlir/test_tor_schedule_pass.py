from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
APS_OPT = REPO_ROOT / "build" / "tools" / "aps-opt" / "aps-opt"


def run_schedule_tor(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    if not APS_OPT.exists():
        pytest.skip(f"{APS_OPT} is not built")

    input_file = tmp_path / "input.mlir"
    output_file = tmp_path / "output.mlir"
    input_file.write_text(source)

    return subprocess.run(
        [
            str(APS_OPT),
            str(input_file),
            "--allow-unregistered-dialect",
            "--schedule-tor",
            "-o",
            str(output_file),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def test_schedule_tor_rejects_multicycle_op_under_unhandled_region(
    tmp_path: Path,
):
    result = run_schedule_tor(
        tmp_path,
        """
module {
  tor.design @aps_isaxes {
    %true = arith.constant true
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    tor.func @flow_nested_region(%arg0: i5, %arg1: i5, ...) attributes {clock = 4.000000e+00 : f32, funct7 = 1 : i32, opcode = 91 : i32, resource = "examples/resource_ihp130.json"} {
      scf.if %true {
        %0 = arith.divsi %c10_i32, %c2_i32 : i32
      }
      tor.return
    }
  }
}
""",
    )

    assert result.returncode != 0
    assert "multi-cycle operation missing scheduling info" in result.stderr
    assert "nested under a control-flow op not handled by schedule-tor" in result.stderr


def test_schedule_tor_assigns_timing_to_multicycle_op_under_tor_if(
    tmp_path: Path,
):
    result = run_schedule_tor(
        tmp_path,
        """
module {
  tor.design @aps_isaxes {
    %true = arith.constant true
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    tor.func @flow_tor_if_nested(%arg0: i5, %arg1: i5, ...) attributes {clock = 4.000000e+00 : f32, funct7 = 2 : i32, opcode = 91 : i32, resource = "examples/resource_ihp130.json"} {
      tor.if %true on (0 to 0) then {
        %0 = arith.divsi %c10_i32, %c2_i32 : i32
        tor.yield
      }
      tor.return
    }
  }
}
""",
    )

    assert result.returncode == 0, result.stderr
    output = (tmp_path / "output.mlir").read_text()
    assert "arith.divsi" in output
    assert "ref_starttime" in output
    assert "ref_endtime" in output
