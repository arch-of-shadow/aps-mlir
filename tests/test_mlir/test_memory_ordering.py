from __future__ import annotations

import itertools
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APS_OPT = REPO_ROOT / "build/tools/aps-opt/aps-opt"


def run_pass(tmp_path, source, pipeline="aps-functional-to-arch,aps-schedule-memory"):
    if not APS_OPT.exists():
        pytest.skip("aps-opt is not built")
    source_file = tmp_path / "ordering.mlir"
    source_file.write_text(source)
    result = subprocess.run(
        [str(APS_OPT), str(source_file), f"-p=builtin.module({pipeline})"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30,
    )
    return result.stdout


def copy_module(sizes, *, write=False, hints=None, inflight=2, tor=False):
    # Abstract ordering tests may model an adapter with single-beat writes.
    # Native Chipyard integration uses busitfc, whose writes need two beats.
    interface = "dma_model" if write and not tor else "busitfc"
    declarations = []
    setup = ["%c0 = arith.constant 0 : i32"]
    operations = []
    for q, size in enumerate(sizes):
        declarations.append(f'memref.global "private" @buf{q} : memref<{size}xi8>')
        setup.extend([
            f"%buf{q} = memref.get_global @buf{q} : memref<{size}xi8>",
            f"%len{q} = arith.constant {size} : i32",
            f"%addr{q} = arith.constant {q * 4096} : i64",
        ])
        hint = f' {{cache_hint = "{hints[q]}"}}' if hints else ""
        if write:
            operations.append(
                f"aps.copy (%buf{q})[%c0], %addr{q}, %len{q}{hint} "
                f": (memref<{size}xi8>), i32, i64, i32"
            )
        else:
            operations.append(
                f"aps.copy %addr{q}, (%buf{q})[%c0], %len{q}{hint} "
                f": i64, (memref<{size}xi8>), i32, i32"
            )
    body = "\n".join(setup + operations)
    if tor:
        function = "tor.design @design {" + "\n".join(declarations) + """
          tor.func @order(%rs: i5, %rd: i5, ...) attributes {
            clock = 4.0 : f32, resource = "examples/resource_ihp130.json"
          } {""" + body + "\n tor.return\n} }"
    else:
        function = "func.func @order() {" + body + "\nreturn\n}"
    return "module {\n" + f"""
      aps.memitfc @{interface} {{width = 8 : i64, max_beats = 8 : i64,
        inflight = {inflight} : i64, read_latency = 10 : i64,
        write_completion = 2 : i64, cache_line = 64 : i64}}
    """ + ("" if tor else "\n".join(declarations)) + function + "\n}"


def events(output, mnemonic="aps.copy_by"):
    rows = []
    for line in output.splitlines():
        if mnemonic not in line or "transaction_id" not in line:
            continue
        def number(name):
            return int(re.search(rf"\b{name} = (-?\d+) : i64", line)[1])
        rows.append({name: number(name) for name in (
            "transaction_id", "transfer_bytes", "memory_issue_cycle",
            "memory_completion_cycle", "memory_search_states",
            "memory_schedule_latency",
        )} | {"line": line})
    return sorted(rows, key=lambda row: row["transaction_id"])


def recurrence(sizes, write=False, inflight=2):
    issue, complete = [], []
    for j, size in enumerate(sizes):
        a = 1 + max(issue[-1] if j else -1, complete[j-inflight] if j >= inflight else -1)
        previous = complete[-1] if j else -1
        b = (size + 7) // 8 + (
            2 + max(previous, a - 1) if write else max(previous, a + 9)
        )
        issue.append(a)
        complete.append(b)
    return complete[-1]


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("inflight", [1, 2, 3])
def test_memoized_order_matches_exhaustive_recurrence(tmp_path, write, inflight):
    sizes = [8, 8, 64, 16]
    output = run_pass(tmp_path, copy_module(sizes, write=write, inflight=inflight))
    rows = events(output)
    actual = [row["transfer_bytes"] for row in rows]
    optimum = min(recurrence(p, write, inflight) for p in itertools.permutations(sizes))
    assert recurrence(actual, write, inflight) == optimum
    assert rows[-1]["memory_completion_cycle"] == optimum
    assert all(row["memory_schedule_latency"] == optimum for row in rows)
    assert 'memory_order_solver = "memoized-relative-window"' in output
    for j, row in enumerate(rows):
        if j:
            assert row["memory_issue_cycle"] >= rows[j-1]["memory_issue_cycle"] + 1
        if j >= inflight:
            assert row["memory_issue_cycle"] > rows[j-inflight]["memory_completion_cycle"]


@pytest.mark.parametrize("write", [False, True])
def test_hierarchy_order_and_contiguous_segments(tmp_path, write):
    sizes = [104, 8, 16]
    hints = ["cold", "warm", "cold"]
    output = run_pass(tmp_path, copy_module(sizes, hints=hints, write=write))
    rows = events(output)
    ordered_hints = [re.search(r'cache_hint = "(\w+)"', row["line"])[1] for row in rows]
    assert ordered_hints == sorted(ordered_hints, key=lambda h: h == "cold", reverse=write)
    segments = [j for j, row in enumerate(rows) if "transaction_group" in row["line"]]
    assert segments == list(range(segments[0], segments[0]+3))
    assert sorted(rows[j]["transfer_bytes"] for j in segments) == [8, 32, 64]
    assert [int(re.search(r"issue_segment = (\d+) : i64", rows[j]["line"])[1])
            for j in segments] == list(range(3))


@pytest.mark.parametrize("count", [100, 1000])
def test_repeated_groups_use_counts_instead_of_permutations(tmp_path, count):
    output = run_pass(tmp_path, copy_module([8] * count + [64]))
    rows = events(output)
    assert len(rows) == count + 1
    assert max(row["memory_search_states"] for row in rows) < count * 50


def test_aliasing_scratchpads_preserve_completion_dependency(tmp_path):
    source = copy_module([8, 8, 8]).replace(
        "%buf1 = memref.get_global @buf1", "%buf1 = memref.get_global @buf0"
    ).replace("%buf2 = memref.get_global @buf2", "%buf2 = memref.get_global @buf0")
    rows = events(run_pass(tmp_path, source))
    assert len(rows) == 3
    for j in range(1, 3):
        assert rows[j]["memory_issue_cycle"] > rows[j-1]["memory_completion_cycle"]


def test_opaque_side_effects_stop_reordering(tmp_path):
    source = copy_module([8, 8, 64]).replace(
        "func.func @order() {", "func.func private @barrier()\nfunc.func @order() {"
    ).replace("aps.copy %addr1", "func.call @barrier() : () -> ()\naps.copy %addr1")
    rows = events(run_pass(tmp_path, source))
    # The first copy stays in a separate window. The remaining two may reorder.
    assert rows[0]["transfer_bytes"] == 8
    assert rows[0]["memory_search_states"] == 3
    assert sorted(row["transfer_bytes"] for row in rows[1:]) == [8, 64]


def test_existing_split_pass_preserves_issue_order(tmp_path):
    output = run_pass(
        tmp_path, copy_module([8, 8, 64], tor=True),
        "aps-functional-to-arch,aps-schedule-memory,"
        "tor.design(schedule-tor,lower-aps-mem-to-req-collect)",
    )
    rows = events(output, "aps.copy_issue")
    assert [row["transfer_bytes"] for row in rows] == [8, 64, 8]
    assert len(rows) == output.count("aps.copy_wait") == 3
    times = {}
    for row in rows:
        line = row["line"]
        start = int(re.search(r"ref_starttime = (\d+) : i32", line)[1])
        end = int(re.search(r"ref_endtime = (\d+) : i32", line)[1])
        assert end - start == row["memory_completion_cycle"] - row["memory_issue_cycle"]
        predecessors = re.search(r"after = array<i64(?:: ([^>]+))?>", line)[1]
        gaps = re.search(r"after_gap = array<i64(?:: ([^>]+))?>", line)[1]
        for predecessor, gap in zip(
            map(int, predecessors.split(",")) if predecessors else [],
            map(int, gaps.split(",")) if gaps else [],
        ):
            assert start >= times[predecessor] + gap
        times[row["transaction_id"]] = start
    for line in output.splitlines():
        if "aps.copy_wait" in line:
            assert "transaction_id" not in line
            assert "after =" not in line


@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("inflight", [1, 2, 3])
def test_segments_reorder_within_a_single_transfer(tmp_path, write, inflight):
    rows = events(run_pass(tmp_path, copy_module([104], write=write, inflight=inflight)))
    canonical = [64, 32, 8]
    actual = [row["transfer_bytes"] for row in rows]
    optimum = min(recurrence(p, write, inflight) for p in itertools.permutations(canonical))
    assert sorted(actual) == sorted(canonical)
    assert rows[-1]["memory_completion_cycle"] == optimum
    if not write and inflight == 2:
        assert optimum < recurrence(canonical, write, inflight)
        assert actual != canonical
    previous = None
    for position, row in enumerate(rows):
        line = row["line"]
        logical_segment = int(re.search(r"\bsegment = (\d+) : i64", line)[1])
        offset = int(re.search(r"transfer_offset = (\d+) : i64", line)[1])
        # Reordering must preserve the original address and scratchpad slices.
        assert offset == [0, 64, 96][logical_segment]
        assert row["transfer_bytes"] == canonical[logical_segment]
        assert f"issue_segment = {position} : i64" in line
        if previous is None:
            assert "after_segment" not in line
        else:
            assert f"after_segment = {previous} : i64" in line
        previous = logical_segment


@pytest.mark.parametrize("write", [False, True])
def test_split_pass_accepts_reordered_segments_of_one_transfer(tmp_path, write):
    total = 112 if write else 104
    segments = [64, 32, 16] if write else [64, 32, 8]
    output = run_pass(
        tmp_path, copy_module([total], write=write, tor=True),
        "aps-functional-to-arch,aps-schedule-memory,"
        "tor.design(schedule-tor,lower-aps-mem-to-req-collect)",
    )
    rows = events(output, "aps.copy_issue")
    assert len(rows) == output.count("aps.copy_wait") == 3
    assert sorted(row["transfer_bytes"] for row in rows) == sorted(segments)
    assert rows[-1]["memory_completion_cycle"] == min(
        recurrence(p, write) for p in itertools.permutations(segments)
    )
    starts = [int(re.search(r"ref_starttime = (\d+) : i32", row["line"])[1])
              for row in rows]
    assert starts == sorted(starts)


def test_large_transfer_counts_identical_segments(tmp_path):
    rows = events(run_pass(tmp_path, copy_module([2064])))
    assert len(rows) == 33
    assert sorted(row["transfer_bytes"] for row in rows) == [16] + [64] * 32
    assert max(row["memory_search_states"] for row in rows) < 10000


@pytest.mark.parametrize("write", [False, True])
def test_segment_and_group_orders_are_optimized_jointly(tmp_path, write):
    groups = [[64, 32, 8], [16, 8]]
    rows = events(run_pass(tmp_path, copy_module([104, 24], write=write)))
    optimum = min(
        recurrence(first + second, write)
        for order in itertools.permutations(groups)
        for first in itertools.permutations(order[0])
        for second in itertools.permutations(order[1])
    )
    assert rows[-1]["memory_completion_cycle"] == optimum
    group_ids = [int(re.search(r"transaction_group = (\d+) : i64", row["line"])[1])
                 for row in rows]
    assert len([key for key, _ in itertools.groupby(group_ids)]) == 2


def test_temporal_ordering_rejects_an_uncanonicalized_subbeat_copy(tmp_path):
    source = copy_module([8]).replace("aps.copy %addr0", "aps.copy_by @busitfc, %addr0")
    source = source.replace("%len0 : i64", "%len0 {transfer_bytes = 4 : i64} : i64")
    source_file = tmp_path / "illegal_arch.mlir"
    source_file.write_text(source)
    result = subprocess.run([str(APS_OPT), str(source_file),
        "-p=builtin.module(aps-schedule-memory)"], cwd=REPO_ROOT,
        capture_output=True, text=True)
    assert result.returncode != 0
    assert "canonical whole-beat transaction" in result.stderr
