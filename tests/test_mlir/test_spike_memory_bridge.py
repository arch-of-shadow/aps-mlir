"""DMA accesses use buffer alignment rather than CPU load/store alignment."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPIKE = ROOT / "thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim"


def test_dma_alignment_and_masks(tmp_path):
    libraries = [SPIKE / "build" / f"lib{name}.a"
                 for name in ("riscv", "disasm", "softfloat", "fesvr", "fdt")]
    if not shutil.which("g++") or not all(p.exists() for p in libraries):
        pytest.skip("C++ compiler and built Spike libraries are required")
    binary = tmp_path / "alignment"
    result = subprocess.run([
        "g++", "-std=c++17", f"-I{SPIKE}", f"-I{SPIKE / 'build'}",
        f"-I{ROOT / 'tools/aps-cosim'}",
        ROOT / "tests/test_mlir/fixtures/spike_dma_alignment.cpp",
        ROOT / "tools/aps-cosim/spike_memory_bridge.cc",
        *libraries, "-ldl", "-lpthread", "-lz", "-o", binary,
    ], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-6000:]
    result = subprocess.run([binary], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS DMA alignment and masked stores" in result.stdout
