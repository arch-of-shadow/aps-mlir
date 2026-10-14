"""Execute optimized RTL with real RV32 custom instructions and shared memory."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COSIM = ROOT / "tools/aps-cosim"
SPIKE = ROOT / "thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim"
E2E = ROOT / "build/tools/aps-opt/aps-e2e"


@pytest.mark.parametrize("case", ["hello", "vgemv3d", "vgemv3d_dma"])
def test_optimized_storage_spike_cosim(tmp_path, case):
    compiler = shutil.which("riscv32-unknown-elf-gcc")
    if not compiler or not shutil.which("verilator") or not E2E.exists():
        pytest.skip("APS, Verilator, and RV32 toolchain are required")
    libraries = [SPIKE / "build" / f"lib{name}.a"
                 for name in ("riscv", "disasm", "softfloat", "fesvr", "fdt")]
    if not all(path.exists() for path in libraries):
        pytest.skip("Spike static libraries are not built")

    def run(command, cwd=ROOT, timeout=180):
        result = subprocess.run([str(arg) for arg in command], cwd=cwd,
                                capture_output=True, text=True, timeout=timeout)
        assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
        return result.stdout

    source = tmp_path / "source.mlir"
    cadl = ROOT / f"tutorial/cadl/{case}.cadl"
    if case == "vgemv3d_dma":
        # Keep the tutorial's scalar loads and computation; use a waited DMA
        # output as a separate completion-path regression.
        cadl = tmp_path / "dma_output.cadl"
        original = (ROOT / "tutorial/cadl/vgemv3d.cadl").read_text()
        original = original.replace("_mem[out_addr] = result[0];\n  _mem[out_addr + 4] = result[1];\n  _mem[out_addr + 8] = result[2];\n  _mem[out_addr + 12] = result[3];",
                                    "_mem[out_addr+:4] = result[0+:];")
        assert "_mem[out_addr+:4]" in original
        cadl.write_text(original)
    source.write_text(run([ROOT / "tools/aps-frontend", "mlir",
                           cadl]))
    rtl = tmp_path / "optimized.sv"
    run([E2E, "-i", source, "-o", rtl, "--export", "sv"])
    text = rtl.read_text()
    if case.startswith("vgemv3d"):
        assert "hellacache_adapter" in text  # Scalar memory must not be omitted.
        assert "module TokenFIFO" in text
    else:
        assert "Reg_width37_init0" in text  # Dynamic response payload retained.
        assert "hellacache_adapter" not in text

    if case == "vgemv3d":
        adapter = tmp_path / "adapter"
        run(["verilator", "--cc", "--exe", "--build", "-j", "2", "-Wno-fatal",
             "--top-module", "MemoryTranslator", "--prefix", "Vmemory",
             "-Mdir", adapter, rtl,
             ROOT / "tests/test_mlir/fixtures/scalar_completion.cpp"])
        output = run([adapter / "Vmemory"])
        (tmp_path / "adapter.log").write_text(output)
        assert "PASS adapter" in output

    elf = tmp_path / "program.riscv"
    command = [compiler, "-std=gnu99", "-O2", "-Wall", "-Wextra",
               "-fno-common", "-fno-builtin-printf",
               "-march=rv32imaf_zicsr_zifencei", "-mabi=ilp32f",
               "-mcmodel=medany", "-specs=htif_nano.specs", "-static",
               "-T", "htif.ld", ROOT / "tests/test_mlir/fixtures/storage_cosim.c",
               "-o", elf]
    if case.startswith("vgemv3d"):
        command.append("-DSTORAGE_GEMV")
    run(command)
    model = tmp_path / "model"
    run(["make", "-j2", "spike", f"RTL={rtl}", f"SPIKE_OBJ_DIR={model}",
         "VERILATE_JOBS=2", "BUILD_JOBS=2",
         f"SPIKE_CFLAGS=-std=c++17 -I{COSIM} -I{SPIKE} -I{SPIKE / 'build'}",
         "SPIKE_LIBS=" + " ".join(map(str, libraries)) + " -ldl -lpthread -lz"],
        cwd=COSIM)
    result = subprocess.run([str(model / "Vmain"), str(elf)], cwd=ROOT,
                            capture_output=True, text=True, timeout=60)
    output = result.stdout
    (tmp_path / "cosim.log").write_text(output)
    assert result.returncode == 0, output + result.stderr
    expected = "PASS gemv trials=64 checked_outputs=256" if case.startswith("vgemv3d") else "PASS hello trials=64 checked_responses=64"
    assert expected in output
