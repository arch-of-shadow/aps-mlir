"""Fast frontend/compiler regressions; run with pixi run python -m pytest."""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "build/tools/aps-opt/aps-e2e"


def run(*args):
    return subprocess.run(
        [str(a) for a in args], cwd=ROOT, capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def resource(tmp_path_factory):
    directory = tmp_path_factory.mktemp("hardfloat")
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "hardware/hardfloat/build-hardfloat-ip.sh"),
            f"--resource-output={directory / 'resource.json'}",
        ],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    config = json.loads((directory / "resource.json").read_text())
    for name in ("addf", "subf", "mulf"):
        assert config[name]["latency"][5] == 3
        assert config[name]["II"] == 1
        assert config[name]["amount"] == -1
    return directory / "resource.json"


@pytest.mark.parametrize(
    "example,ops",
    [("add", ("tor.addf",)), ("add_mul", ("tor.addf", "tor.mulf", "tor.subf"))],
)
def test_frontend_and_two_phase(tmp_path, resource, example, ops):
    frontend = run(
        ROOT / "tools/aps-frontend", "mlir", ROOT / f"examples/hardfloat/{example}.cadl"
    )
    assert frontend.returncode == 0, frontend.stderr
    source = tmp_path / "input.mlir"
    source.write_text(frontend.stdout)
    output = tmp_path / "output.mlir"
    result = run(E2E, "-i", source, "--resource", resource, "-o", output)
    assert result.returncode == 0, result.stderr
    text = output.read_text()
    assert text.count("cmt2.bind.method @issue") == len(ops)
    assert text.count("cmt2.bind.method @collect") == len(ops)
    assert "collect_flags" in text
    assert "_fp_busy" in text
    assert "fp_context_" in text
    assert "_admit" in text and "_release" in text
    assert "conflictFree = [[@issue, @collect]]" in text
    assert "firrtl.module @HardFloat" in text
    assert "firrtl.extmodule" not in text
    assert "tor." not in text
    # Calls are in distinct rules; collect is a consuming method, not a value.
    assert "@issue(" in text and "@collect(" in text
    assert "cmt2.bind.value @collect" not in text


def test_float_constants_and_live_through(tmp_path, resource):
    source = (ROOT / "examples/hardfloat/add.mlir").read_text()
    source = source.replace(
        "%sum = arith.addf %a, %b : f32",
        """
    %minus_zero = arith.constant -0.0 : f32
    %adjusted = arith.mulf %a, %minus_zero : f32
    %sum = arith.addf %adjusted, %b : f32""",
    )
    path = tmp_path / "constants.mlir"
    path.write_text(source)
    output = tmp_path / "constants.cmt2.mlir"
    result = run(E2E, "-i", path, "--resource", resource, "-o", output)
    assert result.returncode == 0, result.stderr
    text = output.read_text()
    assert "2147483648" in text or "-2147483648" in text or "80000000" in text
    assert "Reg_width32" in text


@pytest.mark.parametrize(
    "replacement,diagnostic",
    [
        ("arith.remf", "unsupported operation"),
    ],
)
def test_unsupported_operation_is_diagnostic(
    tmp_path, resource, replacement, diagnostic
):
    path = tmp_path / "unsupported.mlir"
    path.write_text(
        (ROOT / "examples/hardfloat/add.mlir")
        .read_text()
        .replace("arith.addf", replacement)
    )
    result = run(E2E, "-i", path, "--resource", resource)
    assert result.returncode == 1, result.stderr
    assert diagnostic in result.stderr
    assert "LLVM ERROR" not in result.stderr


def test_fast_math_is_not_silently_dropped(tmp_path, resource):
    source = (ROOT / "examples/hardfloat/add.mlir").read_text()
    path = tmp_path / "fastmath.mlir"
    path.write_text(
        source.replace(
            "arith.addf %a, %b : f32", "arith.addf %a, %b fastmath<reassoc> : f32"
        )
    )
    result = run(E2E, "-i", path, "--resource", resource)
    assert result.returncode == 1, result.stderr
    assert "strict floating-point semantics" in result.stderr
    assert "LLVM ERROR" not in result.stderr


def test_unsupported_fp_memory_format_is_explicitly_rejected(tmp_path):
    path = tmp_path / "memory.mlir"
    path.write_text("module { memref.global @fp : memref<1xf64> = dense<0.0> }")
    result = run(ROOT / "build/tools/aps-opt/aps-opt", path, "--aps-to-cmt2")
    assert result.returncode == 1, result.stderr
    assert "floating storage supports only f32/bf16" in result.stderr
    assert "LLVM ERROR" not in result.stderr


def test_fp64_is_diagnostic(tmp_path, resource):
    source = (ROOT / "examples/hardfloat/add.mlir").read_text()
    source = source.replace("f32", "f64").replace(
        "%a = arith.bitcast %a_bits : i32 to f64",
        """
    %wide_a = arith.extui %a_bits : i32 to i64
    %a = arith.bitcast %wide_a : i64 to f64""",
    )
    source = source.replace(
        "%b = arith.bitcast %b_bits : i32 to f64", "%b = arith.constant 2.0 : f64"
    )
    source = source.replace(
        "%result = arith.bitcast %sum : f64 to i32", "%result = arith.constant 0 : i32"
    )
    # Preserve the otherwise-unused computation so canonicalization cannot erase it.
    source = source.replace(
        "%result = arith.constant 0 : i32",
        """
    %wide = arith.bitcast %sum : f64 to i64
    %result = arith.trunci %wide : i64 to i32""",
    )
    path = tmp_path / "fp64.mlir"
    path.write_text(source)
    result = run(E2E, "-i", path, "--resource", resource)
    assert result.returncode == 1, result.stderr
    assert "only scalar f32" in result.stderr
    assert "LLVM ERROR" not in result.stderr


@pytest.mark.parametrize("operation", ["sitofp", "uitofp", "fptosi", "fptoui"])
def test_wide_integer_conversion_is_diagnostic(tmp_path, resource, operation):
    source = (ROOT / "examples/hardfloat/add.mlir").read_text()
    if operation in ("sitofp", "uitofp"):
        source = source.replace(
            "%sum = arith.addf %a, %b : f32",
            f"%wide = arith.extui %a_bits : i32 to i64\n"
            f"    %sum = arith.{operation} %wide : i64 to f32",
        )
    else:
        source = source.replace("arith.addf %a, %b : f32",
                                f"arith.{operation} %a : f32 to i64")
        source = source.replace("arith.bitcast %sum : f32 to i32",
                                "arith.trunci %sum : i64 to i32")
    path = tmp_path / "wide.mlir"
    path.write_text(source)
    result = run(E2E, "-i", path, "--resource", resource)
    assert result.returncode == 1, result.stderr
    assert "integers up to 32 bits" in result.stderr
    assert "LLVM ERROR" not in result.stderr


@pytest.mark.parametrize("flag", ["nnan", "reassoc", "fast"])
def test_comparison_fast_math_is_rejected(tmp_path, resource, flag):
    source = (ROOT / "examples/hardfloat/add.mlir").read_text()
    source = source.replace("arith.addf %a, %b : f32",
                            f"arith.cmpf olt, %a, %b fastmath<{flag}> : f32")
    source = source.replace("arith.bitcast %sum : f32 to i32",
                            "arith.extui %sum : i1 to i32")
    path = tmp_path / "cmp.mlir"
    path.write_text(source)
    result = run(E2E, "-i", path, "--resource", resource)
    assert result.returncode == 1, result.stderr
    assert "strict floating-point semantics" in result.stderr
    assert "LLVM ERROR" not in result.stderr


@pytest.mark.parametrize("kind,name", [(3, "divf"), (4, "sqrt")])
@pytest.mark.parametrize("named", [False, True])
def test_iterative_scheduling_uses_native_bound(tmp_path, resource, kind, name, named):
    config = json.loads(resource.read_text())[name]
    assert config["latency"][4] == (11 if kind == 3 else 10)
    assert config["latency"][5] == (27 if kind == 3 else 26)
    assert config["II"] == (27 if kind == 3 else 26)
    source = (ROOT / "examples/hardfloat/add.mlir").read_text()
    if named:
        args, types = ("%a, %b", "f32, f32") if kind == 3 else ("%a", "f32")
        replacement = (f'%sum, %flags = "aps.fp"({args}) '
                       f'{{kind = {kind} : i32, width = 32 : i32, rm = 0 : i32, predicate = 0 : i32}} '
                       f': ({types}) -> (f32, i5)')
    else:
        replacement = "%sum = " + ("arith.divf %a, %b : f32" if kind == 3 else "math.sqrt %a : f32")
    source = source.replace("%sum = arith.addf %a, %b : f32", replacement)
    path, output = tmp_path / "iterative.mlir", tmp_path / "result.mlir"
    path.write_text(source)
    result = run(E2E, "-i", path, "--resource", resource, "-o", output,
                 "--print-ir-after-all")
    assert result.returncode == 0, result.stderr
    assert f"HardFloat_w32_o{kind}_p0_l{27 if kind == 3 else 26}_q1" in output.read_text()
    # Inspect the real scheduler output, including the named intrinsic path.
    scheduled = [line for line in result.stderr.splitlines()
                 if ("aps.fp" if named else "tor.divf" if kind == 3 else "math.sqrt") in line
                 and "ref_endtime" in line]
    assert scheduled, result.stderr
    intervals = [(int(re.search(r"ref_starttime = (\d+)", line)[1]),
                  int(re.search(r"ref_endtime = (\d+)", line)[1])) for line in scheduled]
    assert any(end - start == (27 if kind == 3 else 26) for start, end in intervals), scheduled


@pytest.mark.parametrize("kind", [3, 4])
@pytest.mark.parametrize("width", [16, 32])
def test_dedicated_iterative_unit_limits_loop_ii(tmp_path, resource, kind, width):
    # No loop-carried FP dependency: only the dedicated unit's occupancy
    # prevents the scheduler from incorrectly assigning II=1.
    operands, types = ("%a, %b", "f32, f32") if kind == 3 else ("%a", "f32")
    source = f'''module {{ tor.design @d {{
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %c4 = arith.constant 4 : i32
      tor.func @flow(%a: f32, %b: f32, ...) attributes {{
        clock = 4.0 : f32, opcode = 11 : i32, funct7 = 0 : i32,
        resource = "{resource}"}} {{
        %r = tor.for %i = (%c0 : i32) to (%c4 : i32) step (%c1 : i32)
          on (0 to 0) iter_args(%acc = %a) -> (f32) {{
          %v, %flags = "aps.fp"({operands}) {{kind = {kind} : i32,
            width = 32 : i32, rm = 0 : i32, predicate = 0 : i32}}
            : ({types}) -> (f32, i5)
          tor.yield %acc : f32
        }} {{pipeline = 1 : i32, II = 1 : i32}}
        tor.return
      }}
    }}}}'''
    if width == 16:
        source = source.replace("f32", "bf16").replace("clock = 4.0 : bf16", "clock = 4.0 : f32").replace("width = 32", "width = 16")
    path, output = tmp_path / "loop.mlir", tmp_path / "scheduled.mlir"
    path.write_text(source)
    result = run(ROOT / "build/tools/aps-opt/aps-opt", path, "--schedule-tor", "-o", output)
    assert result.returncode == 0, result.stderr
    assert f"II = {width - 5 if kind == 3 else width - 6} : i32" in output.read_text()


def test_reject_stale_padding_contract(tmp_path):
    result = subprocess.run(["bash", str(ROOT / "hardware/hardfloat/build-hardfloat-ip.sh"),
                             "width=16", "op=0", "latency=33"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "native contract requires 3" in result.stderr


def test_reject_stale_scheduler_resources(tmp_path, resource):
    config = json.loads(resource.read_text())
    config["addf_fp32"]["latency"] = [33] * 7
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(config))
    result = run(E2E, "-i", ROOT / "examples/hardfloat/add.mlir", "--resource", stale)
    assert result.returncode != 0
    assert "native operation latency" in result.stderr
