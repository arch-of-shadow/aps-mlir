"""Execute optimized RTL with real RV32 custom instructions and shared memory."""
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COSIM = ROOT / "tools/aps-cosim"
SPIKE = ROOT / "thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim"
E2E = ROOT / "build/tools/aps-opt/aps-e2e"


@pytest.mark.parametrize("dma", [False, True])
@pytest.mark.parametrize("has_result", [False, True])
def test_storage_direct_cosim(tmp_path, dma, has_result):
    """Commands must finish their loops and writes before execute returns."""
    if not shutil.which("verilator") or not E2E.exists():
        pytest.skip("APS and Verilator are required")

    def run(command, cwd=ROOT):
        result = subprocess.run(list(map(str, command)), cwd=cwd,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]
        return result.stdout

    source = (ROOT / "tutorial/cadl/vgemv3d.cadl").read_text()
    if not has_result:
        source = source.replace("_irf[rd] = 0;", "")
    if dma:
        source = source.replace(
            "_mem[out_addr] = result[0];\n  _mem[out_addr + 4] = result[1];\n  _mem[out_addr + 8] = result[2];\n  _mem[out_addr + 12] = result[3];",
            "_mem[out_addr+:4] = result[0+:];")
        assert "_mem[out_addr+:4]" in source
    cadl = tmp_path / "void.cadl"
    cadl.write_text(source)
    mlir = tmp_path / "void.mlir"
    mlir.write_text(run([ROOT / "tools/aps-frontend", "mlir", cadl]))
    rtl = tmp_path / "void.sv"
    run([E2E, "-i", mlir, "-o", rtl, "--export", "sv"])
    model = tmp_path / "model"
    run(["make", "direct", f"RTL={rtl}", f"OBJ_DIR={model}",
         "VERILATE_JOBS=2", "BUILD_JOBS=2"], cwd=COSIM)

    inputs, expected, commands = [], [], []
    for trial in range(64):
        values = [((trial + 3) * (i + 7) * 0x1234567) & 0xffffffff
                  for i in range(20)]
        inputs.extend(values)
        expected.extend([0x12345678] + [
            sum(values[row * 4 + col] * values[16 + col]
                for col in range(4)) & 0xffffffff for row in range(4)
        ] + [0x89abcdef])
        commands.extend(["--cmd", "0x7b", "0x31", "7",
                         hex(0x80000000 + trial * 80),
                         hex(0x80010000 + trial * 24 + 4), str(int(has_result))])
    input_bin, initial_bin, output_bin = [tmp_path / name for name in
                                        ("input.bin", "initial.bin", "output.bin")]
    input_bin.write_bytes(struct.pack(f"<{len(inputs)}I", *inputs))
    initial = [value if i % 6 in (0, 5) else 0xdeadbeef
               for i, value in enumerate(expected)]
    initial_bin.write_bytes(struct.pack(f"<{len(initial)}I", *initial))
    output = run([model / "Vmain", "--load-bin", "0x80000000", input_bin,
                  "--load-bin", "0x80010000", initial_bin, *commands,
                  "--dump-bin", "0x80010000", len(expected) * 4, output_bin])
    assert output.count("cmd opcode=") == 64
    if has_result:
        assert output.count(" rd=x7 data=0x0") == 64
    else:
        assert " rd=" not in output  # No fabricated register response for xd=0.
    assert output_bin.read_bytes() == struct.pack(f"<{len(expected)}I", *expected)


@pytest.mark.parametrize("case", ["hello", "vgemv3d", "vgemv3d_dma",
                                 "vgemv3d_void", "vgemv3d_dma_void"])
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
    if case.startswith("vgemv3d") and ("dma" in case or "void" in case):
        # Keep the tutorial's scalar loads and computation; use a waited DMA
        # output as a separate completion-path regression.
        cadl = tmp_path / "dma_output.cadl"
        original = (ROOT / "tutorial/cadl/vgemv3d.cadl").read_text()
        if "dma" in case:
            original = original.replace("_mem[out_addr] = result[0];\n  _mem[out_addr + 4] = result[1];\n  _mem[out_addr + 8] = result[2];\n  _mem[out_addr + 12] = result[3];",
                                        "_mem[out_addr+:4] = result[0+:];")
            assert "_mem[out_addr+:4]" in original
        if "void" in case:
            assert "_irf[rd] = 0;" in original
            original = original.replace("_irf[rd] = 0;", "")
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
    if "void" in case:
        command.append("-DSTORAGE_VOID")
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
