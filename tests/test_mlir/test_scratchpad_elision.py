import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "build/tools/aps-opt/aps-e2e"
SOURCE = (ROOT / "examples/scratchpad_elision.mlir").read_text()


def synthesize(tmp_path, source=SOURCE, extra=(), debug=True):
    if not E2E.exists():
        pytest.skip("aps-e2e is not built")
    src = tmp_path / "input.mlir"
    scheduled = tmp_path / "scheduled.mlir"
    src.write_text(source)
    result = subprocess.run(
        [str(E2E), "-i", str(src), "-o", str(tmp_path / "result.mlir"),
         "--dump-scheduled-tor", str(scheduled),
         *(["--debug-only=scratchpad-elision"] if debug else []), *extra],
        cwd=ROOT, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr[-6000:]
    assert "error:" not in result.stderr, result.stderr[-6000:]
    entries = {}
    for buffer, fields, reason in re.findall(
        r"\[scratchpad-elision\] @(\S+) (.*?): (.*)", result.stderr
    ):
        entry = {"reason": reason}
        for key, value in re.findall(r"(\w+)=(\[[^\]]*\]|\d+)", fields):
            entry[key] = json.loads(value)
        entries[buffer] = entry
    if not debug:
        assert "[scratchpad-elision]" not in result.stderr
    return entries, scheduled.read_text()


def test_elision_removes_allocation_and_staging_after_real_rescheduling(tmp_path):
    entries, scheduled = synthesize(tmp_path)
    decision = entries["bias"]
    assert decision["eligible"] and decision["accepted"]
    assert decision["trial_cycles"] <= decision["baseline_cycles"]
    assert decision["trial_ii"] == decision["baseline_ii"]
    assert "aps.load_by @cpuitfc" in scheduled
    assert 'elided_scratchpad = "bias"' in scheduled
    assert "@bias" not in scheduled
    assert "@output" in scheduled
    assert not entries["output"]["accepted"]
    # Output writeback strictly follows the loop, so source/dest aliasing cannot
    # change reads during the original staged-data snapshot.
    assert "aps.copy_by" in scheduled
    assert "tor.guarded_store" in scheduled
    # No inactive-bank read/modify/write that can replay stale data when the
    # preceding scalar request stalls. This was verified in the RTL smoke test.
    assert "aps.read_smem" not in scheduled
    cmt2 = (tmp_path / "result.mlir").read_text()
    commit = cmt2.split("cmt2.rule @commit_cmd", 1)[1].split("cmt2.return", 1)[0]
    assert "@slot0_txd_reg @read" in commit
    assert "@slot1_txd_reg @read" in commit
    assert "@newer_slot_reg @read" in commit
    assert "firrtl.or" in commit  # A real availability guard, not constant true.


def test_elision_is_automatic_and_diagnostics_are_debug_only(tmp_path):
    entries, scheduled = synthesize(tmp_path, debug=False)
    assert not entries
    assert "@bias" not in scheduled and "aps.load_by @cpuitfc" in scheduled


@pytest.mark.parametrize("option", ["--elision-report", "--scratchpad-elision"])
def test_removed_elision_options_are_not_advertised(option):
    if not E2E.exists():
        pytest.skip("aps-e2e is not built")
    result = subprocess.run(
        [str(E2E), "--help-hidden"], cwd=ROOT, text=True,
        capture_output=True, timeout=10,
    )
    assert result.returncode == 0
    assert option not in result.stdout


def test_latency_regression_rolls_back_without_partial_ir_mutation(tmp_path):
    source = SOURCE.replace("module {", "module {\n aps.memitfc @cpuitfc {read_latency = 40 : i64}\n aps.memitfc @busitfc", 1)
    entries, scheduled = synthesize(tmp_path, source)
    decision = entries["bias"]
    assert decision["eligible"] and not decision["accepted"]
    assert "trial_cycles" in decision
    assert "retained baseline" in decision["reason"]
    assert "@bias" in scheduled and "elided_scratchpad" not in scheduled


@pytest.mark.parametrize("before,after,reason", [
    ('cache_hint = "warm"', 'cache_hint = "cold"', "not explicitly warm"),
    ('cache_hint = "warm"', 'cache_hint = "warm", no_elide = true', "explicitly retained"),
    ('pipeline = 1 : i32', 'pipeline = 0 : i32', "explicitly pipelined"),
    ('pipeline = 1 : i32', 'pipeline = 1 : i32, unroll = 4 : i32', "unrolled"),
    ('%next = arith.addi', 'aps.write_smem %value, %buffer[%i] : i32, memref<32xi32>, i32\n      %next = arith.addi', "scratchpad write"),
    ('%next = arith.addi', 'aps.store %value, %base : i32, i32\n      %next = arith.addi', "main-memory mutation"),
])
def test_conservative_legality_guards(tmp_path, before, after, reason):
    entries, scheduled = synthesize(tmp_path, SOURCE.replace(before, after))
    decision = entries["bias"]
    assert not decision["eligible"] and not decision["accepted"]
    assert reason in decision["reason"]
    assert "@bias" in scheduled and "elided_scratchpad" not in scheduled


def test_staged_range_must_cover_all_reads(tmp_path):
    source = SOURCE.replace("%one = arith.constant 1 : i32", "%one = arith.constant 1 : i32\n    %half = arith.constant 16 : i32")
    source = source.replace("aps.copy %base, (%buffer)[%zero], %count", "aps.copy %base, (%buffer)[%zero], %half")
    entries, _ = synthesize(tmp_path, source)
    assert not entries["bias"]["eligible"]
    assert "staged range" in entries["bias"]["reason"]


def test_noncontiguous_affine_access_is_not_elided(tmp_path):
    source = SOURCE.replace("%one = arith.constant 1 : i32", "%one = arith.constant 1 : i32\n    %half = arith.constant 16 : i32")
    source = source.replace("to %count step %one", "to %half step %one")
    source = source.replace("%value = aps.read_smem %buffer[%i]", "%two = arith.constant 2 : i32\n      %index = arith.muli %i, %two : i32\n      %value = aps.read_smem %buffer[%index]")
    entries, _ = synthesize(tmp_path, source)
    assert not entries["bias"]["eligible"]
    assert "not contiguous" in entries["bias"]["reason"]


def test_unknown_trip_count_keeps_staging(tmp_path):
    source = SOURCE.replace("%dest = aps.read_irf", "%trip = aps.read_irf %rs2 : i5 -> i32\n    %dest = aps.read_irf")
    source = source.replace("to %count step %one", "to %trip step %one")
    entries, _ = synthesize(tmp_path, source, extra=["--export", "scheduled-tor"])
    assert not entries["bias"]["eligible"]
    assert "constant-bounded" in entries["bias"]["reason"]


def test_multiple_live_read_sites_are_retained(tmp_path):
    source = SOURCE.replace("%value = aps.read_smem", "%other = aps.read_smem %buffer[%i] : memref<32xi32>, i32 -> i32\n      %value = aps.read_smem")
    source = source.replace("arith.addi %value, %seven", "arith.addi %value, %other")
    entries, _ = synthesize(tmp_path, source)
    assert not entries["bias"]["eligible"]
    assert "multiple read sites" in entries["bias"]["reason"]


def test_contradictory_access_hint_retains_staging(tmp_path):
    source = SOURCE.replace("aps.copy %base, (%buffer)[%zero], %count :", 'aps.copy %base, (%buffer)[%zero], %count {cache_hint = "cold"} :')
    entries, _ = synthesize(tmp_path, source)
    assert not entries["bias"]["eligible"]
    assert "contradicts" in entries["bias"]["reason"]
