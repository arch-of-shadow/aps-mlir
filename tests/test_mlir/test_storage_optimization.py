import subprocess
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPT = ROOT / "build/tools/aps-opt/aps-opt"
E2E = ROOT / "build/tools/aps-opt/aps-e2e"


def optimize(tmp_path, text):
    if not OPT.exists():
        pytest.skip("aps-opt is not built")
    source = tmp_path / "input.mlir"
    source.write_text(text)
    result = subprocess.run(
        [str(OPT), str(source), "--aps-optimize-storage"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_widening_is_rematerialized_at_later_consumers(tmp_path):
    result = optimize(tmp_path, """module {
      func.func @test(%arg: i3) -> i32 {
        %wide = arith.extui %arg {starttime = 0 : i32, endtime = 1 : i32} : i3 to i32
        %one = arith.constant 1 : i32
        %sum = arith.addi %wide, %one {starttime = 4 : i32, endtime = 5 : i32} : i32
        func.return %sum : i32
      }
    }""")
    assert "aps.scalar_memory_used = false" in result
    casts = [line for line in result.splitlines() if "arith.extui" in line]
    assert len(casts) == 2
    assert "aps.storage_anchor" in casts[0] and "starttime = 0 : i32" in casts[0]
    cast = casts[1]
    assert "starttime = 4 : i32" in cast and "endtime = 4 : i32" in cast
    # Re-running planning must not create extra casts or move them again.
    assert optimize(tmp_path, result) == result


def test_does_not_move_widening_through_control_flow(tmp_path):
    result = optimize(tmp_path, """module {
      func.func @test(%arg: i3, %condition: i1) -> i32 {
        %wide = arith.extui %arg {starttime = 0 : i32, endtime = 1 : i32} : i3 to i32
        %one = arith.constant 1 : i32
        %out = scf.if %condition -> i32 {
          %sum = arith.addi %wide, %one {starttime = 4 : i32, endtime = 5 : i32} : i32
          scf.yield %sum : i32
        } else {
          scf.yield %one : i32
        }
        func.return %out : i32
      }
    }""")
    cast = next(line for line in result.splitlines() if "arith.extui" in line)
    assert "starttime = 0 : i32" in cast


@pytest.mark.parametrize("other_use", [False, True])
def test_square_storage_packing_is_exclusive_and_idempotent(tmp_path, other_use):
    extra = "%extra = arith.addi %x, %arg {starttime = 7 : i32, endtime = 7 : i32} : i32" if other_use else ""
    result = optimize(tmp_path, f"""module {{
      func.func @test(%arg: i32) -> i32 {{
        %x = arith.subi %arg, %arg {{starttime = 0 : i32, endtime = 1 : i32}} : i32
        %square = tor.muli %x %x on (4 to 5) : (i32, i32) -> i32
        {extra}
        %sum = arith.addi %square, %arg {{starttime = 8 : i32, endtime = 8 : i32}} : i32
        func.return %sum : i32
      }}
    }}""")
    truncations = [line for line in result.splitlines() if "arith.trunci" in line]
    assert len(truncations) == (1 if other_use else 2)
    assert all("i32 to i31" in line for line in truncations)
    assert "arith.shrui" in result and "arith.shli" in result
    assert optimize(tmp_path, result) == result


@pytest.mark.parametrize("width", range(3, 9))
def test_square_codec_exact_for_all_small_bit_patterns(width):
    mask = (1 << width) - 1
    for x in range(1 << width):
        square = x * x & mask
        narrowed = x & (mask >> 1)
        assert narrowed * narrowed & mask == square
        packed = (square >> 1) | (square & 1)
        restored = ((packed >> 1) << 2) | (packed & 1)
        assert restored == square


def test_square_codec_32_bit_edges():
    mask = (1 << 32) - 1
    values = [0, 1, 2, 3, mask, mask - 1, 1 << 31, (1 << 31) - 1]
    values += [(index * 0x9E3779B9) & mask for index in range(1000)]
    for x in values:
        square = x * x & mask
        assert (x & (mask >> 1)) ** 2 & mask == square
        packed = (square >> 1) | (square & 1)
        assert ((packed >> 1) << 2) | (packed & 1) == square


def test_square_storage_does_not_move_across_regions_or_same_cycle(tmp_path):
    result = optimize(tmp_path, """module {
      func.func @test(%arg: i32, %condition: i1) -> i32 {
        %x = arith.subi %arg, %arg {starttime = 0 : i32, endtime = 1 : i32} : i32
        %out = scf.if %condition -> i32 {
          %square = tor.muli %x %x on (4 to 5) : (i32, i32) -> i32
          %sum = arith.addi %square, %arg {starttime = 4 : i32, endtime = 5 : i32} : i32
          scf.yield %sum : i32
        } else {
          scf.yield %arg : i32
        }
        func.return %out : i32
      }
    }""")
    assert "arith.trunci" not in result


@pytest.mark.parametrize("kind", ["load", "store"])
def test_scalar_adapter_usage_includes_reads_and_writes(tmp_path, kind):
    access = "%value = aps.load %addr : i32 -> i32" if kind == "load" else "aps.store %addr, %addr : i32, i32"
    result = optimize(tmp_path, f"""module {{
      func.func @test(%addr: i32) {{
        {access}
        func.return
      }}
    }}""")
    assert "aps.scalar_memory_used = true" in result


@pytest.mark.parametrize("value, zero", [(0, True), (7, False), (-1, False)])
def test_only_proven_zero_responses_can_omit_response_data(tmp_path, value, zero):
    result = optimize(tmp_path, f"""module {{
      tor.design @design {{
        tor.func @test(%rs1: i5, %rs2: i5, %rd: i5, ...) attributes {{
          clock = 4.0 : f32, resource = "examples/resource_ihp130.json"
        }} {{
          %zero = arith.constant {value} : i32
          aps.write_irf %rd, %zero : i5, i32
          tor.return
        }}
      }}
    }}""")
    assert f"aps.zero_rocc_response = {str(zero).lower()}" in result
    assert "aps.compact_rocc_command = true" in result


def test_unknown_command_argument_use_disables_payload_compaction(tmp_path):
    result = optimize(tmp_path, """module {
      tor.design @design {
        tor.func @test(%rs1: i5, %rs2: i5, %rd: i5, ...) attributes {
          clock = 4.0 : f32, resource = "examples/resource_ihp130.json"
        } {
          %sum = arith.addi %rs1, %rs2 : i5
          tor.return
        }
      }
    }""")
    assert "aps.compact_rocc_command = false" in result


def test_v3ddist_does_not_generate_unused_scalar_adapter(tmp_path):
    if not E2E.exists():
        pytest.skip("aps-e2e is not built")
    source = tmp_path / "input.mlir"
    frontend = subprocess.run(
        [str(ROOT / "tools/aps-frontend"), "mlir", str(ROOT / "tutorial/cadl/v3ddist_vv.cadl")],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert frontend.returncode == 0, frontend.stderr
    source.write_text(frontend.stdout)
    result = subprocess.run(
        [str(E2E), "-i", str(source), "-o", str(tmp_path / "output.sv"), "--export", "sv"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    rtl = (tmp_path / "output.sv").read_text()
    assert "hellacache_adapter" not in rtl
    assert "hella_resp_ready" in rtl  # External ports remain available.
    assert "hella_cmd_hella_cmd_to_bus_enable" in rtl
    assert "Reg_width69_init0" in rtl  # rd + rs1data + rs2data, not routing fields.
    assert "Reg_width96_init0" not in rtl
    assert "Reg_width37_init0" not in rtl  # Zero response data is not stored.
    assert "module TokenFIFO1_PUSH(" in rtl
    assert "module TokenFIFO2_I(" in rtl
    assert "Reg_width31_init0" in rtl  # Packed square operands/results.
    # Prove both emitted token queue implementations against occupancy semantics
    # for arbitrary request/reset sequences, not just accepted-transfer traces.
    yosys = shutil.which("yosys")
    if yosys:
        for depth, name in ((1, "TokenFIFO1_PUSH"), (2, "TokenFIFO2_I")):
            script = (
                f"read_verilog -formal -sv -DSYNTHESIS -DDEPTH={depth} -DTOKEN_DUT={name} "
                f"{tmp_path / 'output.sv'} {ROOT / 'tests/test_mlir/fixtures/token_fifo_check.sv'}; "
                "prep -top token_check -flatten; chformal -lower; opt; "
                "sat -seq 4 -tempinduct -maxsteps 16 -prove-asserts -verify"
            )
            proof = subprocess.run([yosys, "-Q", "-T", "-p", script], cwd=ROOT,
                                   capture_output=True, text=True, timeout=30)
            assert proof.returncode == 0, (proof.stdout + proof.stderr)[-4000:]
            assert "SUCCESS" in proof.stdout


def test_v3ddist_external_behavior_matches_before_storage_planning(tmp_path):
    firtool = ROOT / "circt/build/bin/firtool"
    if not all(path.exists() for path in (OPT, E2E, firtool)) or not shutil.which("verilator") or not shutil.which("g++"):
        pytest.skip("APS/CIRCT tools, Verilator, and g++ are required")

    def run(command, timeout=120):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        assert result.returncode == 0, (result.stdout + result.stderr)[-5000:]
        return result.stdout

    source = tmp_path / "source.mlir"
    source.write_text(run([str(ROOT / "tools/aps-frontend"), "mlir",
                           str(ROOT / "tutorial/cadl/v3ddist_vv.cadl")]))
    scheduled = tmp_path / "scheduled.mlir"
    run([str(E2E), "-i", str(source), "-o", str(scheduled), "--export", "scheduled-tor"])
    # The same temporal/CMT2 pipeline, deliberately without storage planning.
    baseline_firrtl = tmp_path / "baseline.mlir"
    pipeline = "builtin.module(tor.design(lower-aps-mem-to-req-collect,tor-time-graph),tor.func(duplicate-memloads),canonicalize,aps-to-cmt2,lower-cmt2-to-firrtl)"
    run([str(OPT), str(scheduled), f"--pass-pipeline={pipeline}", "-o", str(baseline_firrtl)])
    baseline = tmp_path / "baseline.sv"
    run([str(firtool), "--format=mlir", str(baseline_firrtl), "-o", str(baseline)])
    optimized = tmp_path / "optimized.sv"
    run([str(E2E), "-i", str(source), "-o", str(optimized), "--export", "sv"])

    models = []
    for prefix, rtl in (("Vgold", baseline), ("Vgate", optimized)):
        model = tmp_path / prefix
        run(["verilator", "--cc", "--build", "-j", "2", "-Wno-fatal", "--prefix", prefix,
             "--top-module", "main", "-Mdir", str(model), str(rtl)])
        models.append(model)
    runtime = Path(run(["verilator", "--getenv", "VERILATOR_ROOT"]).strip()) / "include"
    executable = tmp_path / "compare"
    run(["g++", "-O2", "-std=c++17", "-pthread", "-I", str(models[0]), "-I", str(models[1]),
         "-I", str(runtime), "-I", str(runtime / "vltstd"),
         str(ROOT / "tests/test_mlir/fixtures/storage_compare.cpp"),
         str(models[0] / "Vgold__ALL.a"), str(models[1] / "Vgate__ALL.a"),
         str(runtime / "verilated.cpp"), str(runtime / "verilated_threads.cpp"),
         "-o", str(executable)])
    assert "all_valid_output_ports_match" in run([str(executable)], timeout=30)
