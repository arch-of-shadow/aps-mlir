"""One model drives architectural lowering and Chipyard packaging."""
import copy
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "build/tools/aps-opt/aps-e2e"
MODEL = json.loads((ROOT / "examples/interface.json").read_text())
SOURCE = """module {
  memref.global @gain_cfg : memref<i32> {csr_address = 2049 : i32}
  memref.global @bias_cfg : memref<i32> {csr_address = 2050 : i32}
  func.func @flow_config(%rs1: i5, %rd: i5) attributes {funct7 = 30 : i32, opcode = 43 : i32} {
    %a = aps.read_irf %rs1 : i5 -> i32
    %g = aps.read_csr @gain_cfg : i32
    %b = aps.read_csr @bias_cfg : i32
    %v = arith.addi %a, %g : i32
    %w = arith.addi %v, %b : i32
    aps.write_irf %rd, %w : i5, i32
    return
  }
}
"""


def run(tmp_path, model=None, source=SOURCE, extra=()):
    if not E2E.exists():
        pytest.skip("aps-e2e is not built")
    src, config, rtl = [tmp_path / name for name in ("input.mlir", "interface.json", "output.sv")]
    src.write_text(source)
    config.write_text(json.dumps(model if model is not None else MODEL))
    result = subprocess.run(
        [str(E2E), "-i", str(src), "-o", str(rtl), "--export", "sv",
         "--interface-config", str(config), *extra],
        cwd=ROOT, text=True, capture_output=True,
    )
    return result, rtl, tmp_path / "output.chipyard.yaml"


def test_chipyard_yaml_and_csrs(tmp_path):
    result, rtl, output = run(tmp_path, extra=["--print-ir-after-all"])
    assert result.returncode == 0, result.stderr
    assert "module main(" in rtl.read_text()
    assert re.search(r"aps.memitfc @cpuitfc[^\n]*inflight = 2 : i64", result.stderr)
    config = yaml.safe_load(output.read_text())
    assert config["vsrc"] == [str(rtl)]
    assert config["maxBurstBytes"] == 128
    assert config["nXacts"] == 2
    assert config["arch"] == "rv32"
    assert config["csrs"] == [
        dict(id="0x801", name="gain_cfg", mask="0xffffffff", init="0x0"),
        dict(id="0x802", name="bias_cfg", mask="0xffffffff", init="0x0"),
    ]


def test_explicit_output_and_csr_overrides(tmp_path):
    model = copy.deepcopy(MODEL)
    model["interfaces"][1]["max_beats"] = 8
    model["csrs"] = [dict(name="gain_cfg", id="0x801", mask="0xff", init=7)]
    path = tmp_path / "custom.yaml"
    result, _, automatic = run(tmp_path, model, extra=["--chipyard-config", str(path)])
    assert result.returncode == 0, result.stderr
    assert not automatic.exists()
    config = yaml.safe_load(path.read_text())
    assert config["maxBurstBytes"] == 64
    assert config["csrs"][0] == dict(id="0x801", name="gain_cfg", mask="0xff", init="0x7")


@pytest.mark.parametrize("field,value,message", [
    ("width", 4, "native DMA"),
    ("max_beats", 32, "native DMA"),
    ("max_beats", 3, "beat bounds"),
    ("min_write_beats", 1, "native DMA"),
    ("inflight", 3, "two channels"),
    ("read_latency", 0, "positive model field"),
    ("cache_hint", "typo", "cache_hint"),
])
def test_invalid_hardware_model(tmp_path, field, value, message):
    model = copy.deepcopy(MODEL)
    model["interfaces"][1][field] = value
    result, rtl, config = run(tmp_path, model)
    assert result.returncode != 0 and message in result.stderr
    assert not rtl.exists() and not config.exists()


@pytest.mark.parametrize("overrides,message", [
    ([dict(name="missing", init=0)], "does not refer"),
    ([dict(name="gain_cfg", id=0x803)], "address conflicts"),
    ([dict(name="gain_cfg", mask=1 << 32)], "exceeds rv32"),
    ([dict(name="gain_cfg"), dict(name="gain_cfg")], "duplicate CSR"),
])
def test_invalid_csr_overrides(tmp_path, overrides, message):
    model = copy.deepcopy(MODEL)
    model["csrs"] = overrides
    result, _, output = run(tmp_path, model)
    assert result.returncode != 0 and message in result.stderr
    assert not output.exists()


def test_ir_interface_conflict(tmp_path):
    source = SOURCE.replace("module {", "module {\n aps.memitfc @busitfc {max_beats = 8 : i64}", 1)
    result, _, output = run(tmp_path, source=source)
    assert result.returncode != 0 and "IR/config conflict" in result.stderr
    assert not output.exists()


def test_unknown_key(tmp_path):
    model = copy.deepcopy(MODEL)
    model["interfaces"][1]["maxBurstBytes"] = 128
    result, _, _ = run(tmp_path, model)
    assert result.returncode != 0 and "unknown configuration key" in result.stderr


def test_duplicate_csr_ids(tmp_path):
    result, _, _ = run(tmp_path, source=SOURCE.replace("2050 : i32", "2049 : i32"))
    assert result.returncode != 0 and "unique 12-bit" in result.stderr


def test_sidecar_cannot_overwrite_rtl(tmp_path):
    result, rtl, _ = run(tmp_path, extra=["--chipyard-config", str(tmp_path / "output.sv")])
    assert result.returncode != 0 and "must not overwrite" in result.stderr
    assert not rtl.exists()


def test_cpu_inflight_cannot_exceed_two_slots(tmp_path):
    model = copy.deepcopy(MODEL)
    model["interfaces"][0]["inflight"] = 3
    result, _, _ = run(tmp_path, model)
    assert result.returncode != 0 and "inflight<=2" in result.stderr


@pytest.mark.parametrize("max_beats,size", [(8, 64), (16, 128)])
def test_model_controls_architectural_splitting(tmp_path, max_beats, size):
    source = """module {
      memref.global @buf : memref<32xi32> {var_name = "buf"}
      func.func @flow_copy(%rs1: i5, %rd: i5) attributes {funct7 = 30 : i32, opcode = 43 : i32} {
        %addr = aps.read_irf %rs1 : i5 -> i32
        %buf = memref.get_global @buf : memref<32xi32>
        %zero = arith.constant 0 : i32
        %length = arith.constant 32 : i32
        aps.copy %addr, (%buf)[%zero], %length : i32, (memref<32xi32>), i32, i32
        aps.write_irf %rd, %zero : i5, i32
        return
      }
    }"""
    model = copy.deepcopy(MODEL)
    model["interfaces"][1]["max_beats"] = max_beats
    result, _, output = run(tmp_path, model, source=source, extra=["--print-ir-after-all"])
    assert result.returncode == 0, result.stderr
    sizes = {int(x) for x in re.findall(r"aps.copy_by[^\n]*transfer_bytes = (\d+)", result.stderr)}
    assert sizes == {size}
    config = yaml.safe_load(output.read_text())
    assert config["maxBurstBytes"] == size
    assert config["csrs"] == []


def test_failed_lowering_does_not_emit_chipyard_config(tmp_path):
    # Unsupported scalar width fails later than config loading.
    source = """module {
      func.func @flow_bad(%rs1: i5, %rd: i5) attributes {funct7 = 30 : i32, opcode = 43 : i32} {
        %addr = aps.read_irf %rs1 : i5 -> i32
        %v = aps.load %addr : i32 -> i64
        return
      }
    }"""
    result, _, output = run(tmp_path, source=source)
    assert result.returncode != 0
    assert not output.exists()
