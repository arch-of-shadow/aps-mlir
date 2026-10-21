"""Exercise the actual Yosys -> FIRRTL -> CMT2 -> RTL library path."""

import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from ieee_oracle import IEEE


def run(args, cwd, log):
    with log.open("w") as output:
        result = subprocess.run(list(map(str, args)), cwd=cwd, stdout=output,
                                stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, log.read_text()[-6000:]


@pytest.mark.parametrize("width,kind,operation", [
    (16, 0, "add"), (32, 5, "fma"), (16, 3, "div"), (32, 4, "sqrt"),
    (16, 9, "to_i32"), (32, 10, "to_u32"),
])
def test_imported_library_numerics_and_protocol(tmp_path, width, kind, operation):
    run([sys.executable, ROOT / "hardware/hardfloat/build-hardfloat-ip.py",
         f"width={width}", f"op={kind}"], tmp_path, tmp_path / "import.log")
    module = next(tmp_path.glob("HardFloat*.mlir"))
    name = module.stem
    text = module.read_text()
    assert f"firrtl.module @{name}" in text
    assert "firrtl.extmodule" not in text
    assert "BlackBox" not in text
    assert "in %clock: !firrtl.clock" in text
    assert max(map(int, re.findall(r"!firrtl\.(?:u|s)int<(\d+)>", text))) <= 1024
    metadata = json.loads(module.with_suffix(".json").read_text())

    # Bind the generated implementation using the existing FModuleOp path.
    # No result-name fallback or external FIRRTL module support is needed.
    cmt2 = tmp_path / "bound.mlir"
    cmt2.write_text(text.rstrip().removesuffix("}") + f"""
  cmt2.circuit {{
    cmt2.module.extern.firrtl @FP : @{name}(%clk: !firrtl.clock, %rst: !firrtl.uint<1>) {{
      cmt2.bind.bare %clk, @clock : !firrtl.clock
      cmt2.bind.bare %rst, @reset : !firrtl.uint<1>
      cmt2.bind.method @issue : (!firrtl.uint<32>, !firrtl.uint<32>, !firrtl.uint<32>, !firrtl.uint<3>) -> () [enable = "issue_enable", ready = "issue_ready", arguments = ["operand0", "operand1", "operand2", "rounding_mode"], results = []]
      cmt2.bind.method @collect : () -> (!firrtl.uint<32>, !firrtl.uint<5>) [enable = "collect_enable", ready = "collect_ready", arguments = [], results = ["collect_data", "collect_flags"]]
    }} {{conflict = [[@issue, @issue], [@collect, @collect]], conflictFree = [[@issue, @collect]]}}
    cmt2.module @top(%clk: !firrtl.clock, %rst: !firrtl.uint<1>) {{
      cmt2.instance @fp = @FP(%clk, %rst) : !firrtl.clock, !firrtl.uint<1>
      cmt2.method @issue(%a: !firrtl.uint<32>, %b: !firrtl.uint<32>, %c: !firrtl.uint<32>, %rm: !firrtl.uint<3>) -> () {{
        cmt2.return
      }} {{
        cmt2.call @fp @issue(%a, %b, %c, %rm) : (!firrtl.uint<32>, !firrtl.uint<32>, !firrtl.uint<32>, !firrtl.uint<3>) -> ()
        cmt2.return
      }}
      "cmt2.method"() <{{argNames = [], bodyResNames = ["data", "flags"], function_type = () -> (!firrtl.uint<32>, !firrtl.uint<5>), sym_name = "collect"}}> ({{
        cmt2.return
      }}, {{
        %r:2 = cmt2.call @fp @collect() : () -> (!firrtl.uint<32>, !firrtl.uint<5>)
        cmt2.return %r#0, %r#1 : !firrtl.uint<32>, !firrtl.uint<5>
      }}) : () -> ()
    }}
  }}
}}
""")
    run([ROOT / "circt/build/bin/circt-opt", cmt2, "--lower-cmt2-to-firrtl",
         "-o", tmp_path / "lowered.mlir"], tmp_path, tmp_path / "binding.log")
    assert "firrtl.extmodule" not in (tmp_path / "lowered.mlir").read_text()

    # Simulate the imported implementation, not the original Verilog bundle.
    rtl = tmp_path / "imported.sv"
    run([os.environ.get("FIRTOOL", ROOT / "circt/build/bin/firtool"), module,
         "--format=mlir", "--disable-all-randomization", "-o", rtl],
        tmp_path, tmp_path / "export.log")
    verilator = shutil.which("verilator") or ROOT / ".pixi/envs/default/bin/verilator"
    run([verilator, "--cc", "--exe", "--build", "--assert", "-Wno-fatal",
         "--top-module", name, "--prefix", "VHFUnit", "--Mdir", tmp_path / "obj",
         "-j", "2", "-CFLAGS", "-O1", rtl,
         ROOT / "tests/hardfloat/acceptance/unit_driver.cpp"],
        tmp_path, tmp_path / "verilator.log")
    oracle = IEEE(width)
    directed = [0, oracle.signbit, 1, oracle.max, oracle.inf,
                oracle.inf | oracle.signbit, oracle.nan, oracle.inf | 1,
                127 << oracle.f]
    rng = random.Random(0xC172 + width + kind)
    vectors = []
    for index in range(768):
        operands = [rng.choice(directed) if index < 256 else rng.getrandbits(width)
                    for _ in range(3)]
        rm = index % 5
        data, flags = oracle.operation(operation, *operands, rm=rm)
        vectors.append(" ".join(f"{value:x}" for value in (*operands, rm, data, flags)))
    path = tmp_path / "vectors.txt"
    path.write_text("\n".join(vectors) + "\n")
    run([tmp_path / "obj/VHFUnit", path, metadata["latency"],
         int(kind in (3, 4)), metadata["capacity"]], tmp_path, tmp_path / "simulation.log")


def test_legacy_library_uses_regular_firrtl(tmp_path):
    run([sys.executable, ROOT / "hardware/hardfloat/build-hardfloat-ip.py", "op=2"],
        tmp_path, tmp_path / "import.log")
    text = next(tmp_path.glob("HardFloat32*.mlir")).read_text()
    assert "firrtl.module @HardFloat32_op2_l3_q5" in text
    assert "firrtl.extmodule" not in text
    assert "BlackBox" not in text
