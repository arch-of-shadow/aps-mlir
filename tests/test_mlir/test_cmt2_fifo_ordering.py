import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "build/tools/aps-opt/aps-e2e"


def test_fifo2_preserves_order_under_backpressure(tmp_path):
    if not E2E.exists() or not shutil.which("verilator"):
        pytest.skip("aps-e2e and Verilator are required")
    rtl = tmp_path / "fifo_source.sv"
    result = subprocess.run(
        [str(E2E), "-i", str(ROOT / "examples/scratchpad_elision.mlir"),
         "-o", str(rtl), "--export", "sv"],
        cwd=ROOT, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    model = tmp_path / "model"
    fixture = ROOT / "tests/test_mlir/fixtures/fifo2_order.cpp"
    compile_result = subprocess.run(
        ["verilator", "--cc", "--exe", "--build", "-j", "2", "-Wno-fatal",
         "--top-module", "FIFO2_I_w32", "-Mdir", str(model), str(rtl), str(fixture)],
        cwd=ROOT, text=True, capture_output=True, timeout=120,
    )
    assert compile_result.returncode == 0, compile_result.stderr[-4000:]
    run = subprocess.run([str(model / "VFIFO2_I_w32")], text=True, capture_output=True, timeout=10)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "PASS" in run.stdout
