from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
APS_OPT = REPO_ROOT / "build" / "tools" / "aps-opt" / "aps-opt"

FUNCTIONAL_TO_ARCH_PIPELINE = (
    "builtin.module("
    "func.func(place-readrf-at-entry),"
    "auto-burst-partition,"
    "aps-memory-map,"
    "aps-raise-scf-to-affine,"
    "canonicalize,"
    "func.func(raise-memref-to-affine),"
    "raise-memref-to-affine,"
    "canonicalize,"
    "hls-unroll,"
    "cse,"
    "canonicalize,"
    "func.func(affine-loop-normalize),"
    "canonicalize,"
    "new-array-partition,"
    "canonicalize,"
    "func.func(affine-mem-to-aps,memref-to-aps),"
    "aps-functional-to-arch,"
    "func.func(promote-singleton-memref-to-global)"
    ")"
)


def run_functional_to_arch(tmp_path: Path, source: str) -> str:
    if not APS_OPT.exists():
        pytest.skip(f"{APS_OPT} is not built")
    if shutil.which("pixi") is None:
        pytest.skip("pixi is not available")

    cadl_file = tmp_path / "input.cadl"
    mlir_file = tmp_path / "input.mlir"
    output_file = tmp_path / "output.mlir"
    cadl_file.write_text(source)

    subprocess.run(
        ["pixi", "run", "mlir", str(cadl_file), str(mlir_file)],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            str(APS_OPT),
            "-p",
            FUNCTIONAL_TO_ARCH_PIPELINE,
            str(mlir_file),
            "-o",
            str(output_file),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return output_file.read_text()


def run_arch_pass_on_mlir(tmp_path: Path, source: str) -> str:
    if not APS_OPT.exists():
        pytest.skip(f"{APS_OPT} is not built")
    mlir_file = tmp_path / "interface_model.mlir"
    mlir_file.write_text(source)
    result = subprocess.run(
        [
            str(APS_OPT),
            "-p=builtin.module(aps-functional-to-arch)",
            str(mlir_file),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def test_full_partition_copy_in_lowers_to_scalar_loads(tmp_path: Path):
    output = run_functional_to_arch(
        tmp_path,
        """
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static buf: [u32; 4];

rtype full_copy(rs1: u5, rd: u5) {
  let addr: u32 = _irf[rs1];
  buf[0 +: ] = _mem[addr +: 4];
  _irf[rd] = 0;
}
""",
    )

    assert output.count("aps.load_by") == 4
    assert output.count("aps.globalstore") == 4
    assert "aps.copy_by" not in output
    assert "aps.copy " not in output


def test_full_partition_copy_out_lowers_to_scalar_stores(tmp_path: Path):
    output = run_functional_to_arch(
        tmp_path,
        """
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static buf: [u32; 4];

rtype full_copy_out(rs1: u5, rd: u5) {
  let addr: u32 = _irf[rs1];
  _mem[addr +: 4] = buf[0 +: ];
  _irf[rd] = 0;
}
""",
    )

    assert output.count("aps.globalload") == 4
    assert output.count("aps.store_by") == 4
    assert "aps.copy_by" not in output
    assert "aps.copy " not in output


def test_partial_partition_copy_remains_arch_copy(tmp_path: Path):
    output = run_functional_to_arch(
        tmp_path,
        """
#[partition_dim_array([0])]
#[partition_factor_array([2])]
#[partition_cyclic_array([1])]
static buf: [u32; 4];

rtype partial_copy(rs1: u5, rd: u5) {
  let addr: u32 = _irf[rs1];
  buf[0 +: ] = _mem[addr +: 4];
  _irf[rd] = 0;
}
""",
    )

    assert output.count("aps.copy_by") == 1
    assert "aps.copy " not in output


def test_interface_model_selects_and_canonicalizes_copy(tmp_path: Path):
    output = run_arch_pass_on_mlir(
        tmp_path,
        """
module {
  aps.memitfc @narrow {width = 4 : i64, max_beats = 1 : i64,
    inflight = 1 : i64, read_latency = 2 : i64,
    write_completion = 2 : i64, cache_line = 64 : i64,
    cache_hint = "warm"}
  aps.memitfc @wide {width = 8 : i64, max_beats = 8 : i64,
    inflight = 2 : i64, read_latency = 5 : i64,
    write_completion = 2 : i64, cache_line = 64 : i64,
    cache_hint = "cold"}
  memref.global "private" @buf : memref<104xi8>
  func.func @copy(%addr: i64) {
    %buf = memref.get_global @buf : memref<104xi8>
    %c0 = arith.constant 0 : i32
    %c104 = arith.constant 104 : i32
    aps.copy %addr, (%buf)[%c0], %c104 {cache_hint = "cold"}
      : i64, (memref<104xi8>), i32, i32
    return
  }
}
""",
    )

    assert output.count("aps.copy_by @wide") == 3
    assert "aps.memitfc @cpuitfc" not in output
    assert "aps.memitfc @busitfc" not in output
    for size in (64, 32, 8):
        assert f"transfer_bytes = {size} : i64" in output
    assert "after_segment = 0 : i64" in output
    assert "after_segment = 1 : i64" in output


SOLVER_INTERFACES = """
  aps.memitfc @narrow {width = 4 : i64, max_beats = 2 : i64,
    inflight = 1 : i64, read_latency = 1 : i64,
    write_completion = 2 : i64, cache_line = 64 : i64}
  aps.memitfc @wide {width = 8 : i64, max_beats = 8 : i64,
    inflight = 2 : i64, read_latency = 10 : i64,
    write_completion = 2 : i64, cache_line = 64 : i64}
"""


def test_region_solver_amortizes_interface_startup(tmp_path: Path):
    output = run_arch_pass_on_mlir(
        tmp_path,
        "module {" + SOLVER_INTERFACES + """
  func.func @joint(%addr: i64, %buf: memref<64xi8>) -> i64 {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %x = aps.load %addr : i64 -> i64
    aps.copy %addr, (%buf)[%c0], %c64
      : i64, (memref<64xi8>), i32, i32
    return %x : i64
  }
  func.func @scalar_alone(%addr: i64) -> i64 {
    %x = aps.load %addr : i64 -> i64
    return %x : i64
  }
}
""",
    )
    joint, alone = output.split("func.func @scalar_alone")
    # Independent minimization picks narrow for the scalar (18 vs 22).
    # Joint costs: narrow/narrow=50, narrow/wide=43,
    # wide/narrow=54, wide/wide=38, paying wide startup only once.
    assert "aps.load_by @wide" in joint
    assert "aps.copy_by @wide" in joint
    assert 'interface_solver = "active-set-dp"' in joint
    assert "solver_objective = 3.800000e+01 : f64" in joint
    # A separate region must not share that startup cost.
    assert "aps.load_by @narrow" in alone
    assert "solver_objective = 1.800000e+01 : f64" in alone


def test_solver_separates_reads_and_writes_and_disables_unused_interfaces(
    tmp_path: Path,
):
    output = run_arch_pass_on_mlir(
        tmp_path,
        "module {" + SOLVER_INTERFACES + """
  func.func @directions(%addr: i64, %value: i64) -> i64 {
    %x = aps.load %addr : i64 -> i64
    aps.store %value, %addr : i64, i64
    return %x : i64
  }
}
""",
    )
    assert "aps.load_by @narrow" in output
    assert "aps.store_by @wide" in output
    assert "solver_objective = 1.800000e+01 : f64" in output
    # wide store: -1 + 8/8 + 2 + 64/8 = 10. An unused narrow
    # interface must not contribute its negative activation cost (-1).
    assert "solver_objective = 1.000000e+01 : f64" in output


def test_assignment_scales_with_request_count(tmp_path: Path):
    requests = "\n".join(
        f"%x{i} = aps.load %addr : i64 -> i64" for i in range(1000)
    )
    output = run_arch_pass_on_mlir(
        tmp_path,
        "module {" + SOLVER_INTERFACES + "func.func @many(%addr: i64) {"
        + requests + "\nreturn\n}\n}",
    )
    assert output.count("aps.load_by @wide") == 1000
    assert "solver_objective = 1.300900e+04 : f64" in output


@pytest.mark.parametrize("size,expected", [(8, [8]), (24, [16, 8]), (64, [64]),
                                          (72, [64, 8]), (128, [64, 64]),
                                          (200, [64, 64, 64, 8]), (104, [64, 32, 8])])
def test_binding_emits_only_single_transaction_copies(tmp_path, size, expected):
    output = run_arch_pass_on_mlir(tmp_path, "module { aps.memitfc @busitfc\n" + f"""
      func.func @large(%addr: i64, %buf: memref<{size}xi8>) {{
        %c0 = arith.constant 0 : i32
        %length = arith.constant {size} : i32
        aps.copy %addr, (%buf)[%c0], %length : i64, (memref<{size}xi8>), i32, i32
        return
      }}
    }}""")
    parts = [line for line in output.splitlines() if "aps.copy_by" in line]
    sizes = [int(re.search(r"transfer_bytes = (\d+) : i64", line)[1]) for line in parts]
    assert sizes == expected
    assert all(8 <= size <= 64 and size & (size - 1) == 0 for size in sizes)
    assert sum(sizes) == size
    offset = 0
    for line, count in zip(parts, sizes):
        assert f"transfer_offset = {offset} : i64" in line
        offset += count


def test_binding_rejects_unknown_transaction_length(tmp_path):
    source = tmp_path / "dynamic.mlir"
    source.write_text("""
      module {
        func.func @dynamic(%addr: i64, %buf: memref<256xi8>, %length: i32) {
          %c0 = arith.constant 0 : i32
          aps.copy %addr, (%buf)[%c0], %length : i64, (memref<256xi8>), i32, i32
          return
        }
      }
    """)
    result = subprocess.run([str(APS_OPT), str(source),
        "-p=builtin.module(aps-functional-to-arch)"], cwd=REPO_ROOT,
        text=True, capture_output=True)
    assert result.returncode != 0
    assert "canonicalization requires a constant transfer length" in result.stderr


def bulk_request(size, *, interfaces="", address=None, minimum=1):
    if not interfaces:
        interfaces = f"""
          aps.memitfc @busitfc {{min_beats = {minimum} : i64}}
        """
    addr_setup = f"%address = arith.constant {address} : i64" if address is not None else ""
    addr = "%address" if address is not None else "%addr"
    return "module {" + interfaces + f"""
      func.func @request(%addr: i64, %buf: memref<256xi8>) {{
        {addr_setup}
        %c0 = arith.constant 0 : i32
        %length = arith.constant {size} : i32
        aps.copy {addr}, (%buf)[%c0], %length : i64, (memref<256xi8>), i32, i32
        return
      }}
    }}"""


def arch_error(tmp_path, source):
    input_file = tmp_path / "illegal.mlir"
    input_file.write_text(source)
    result = subprocess.run([str(APS_OPT), str(input_file),
        "-p=builtin.module(aps-functional-to-arch)"], cwd=REPO_ROOT,
        text=True, capture_output=True)
    assert result.returncode != 0
    return result.stderr


@pytest.mark.parametrize("size", [4, 12, 108])
def test_bus_rejects_unrepresentable_subbeat_tails(tmp_path, size):
    error = arch_error(tmp_path, bulk_request(size))
    assert f"no legal memory interface for {size}-byte bulk transfer" in error
    assert "sub-beat tails cannot be rounded up" in error


def test_native_cpu_is_not_a_bulk_dma_fallback(tmp_path):
    error = arch_error(tmp_path, bulk_request(108, interfaces="""
      aps.memitfc @cpuitfc
      aps.memitfc @busitfc
    """))
    assert "no legal memory interface for 108-byte bulk transfer" in error


def test_minimum_is_independent_of_cache_line(tmp_path):
    output = run_arch_pass_on_mlir(tmp_path, bulk_request(8))
    assert "cache_line = 64 : i64" in output
    assert "min_beats = 1 : i64" in output
    assert "transfer_bytes = 8 : i64" in output
    assert output.count("aps.copy_by") == 1


@pytest.mark.parametrize("minimum", [0, 3, 16])
def test_invalid_minimum_beats_is_rejected(tmp_path, minimum):
    assert "min_beats" in arch_error(tmp_path, bulk_request(64, minimum=minimum))


def test_configured_minimum_rejects_short_tail(tmp_path):
    output = run_arch_pass_on_mlir(tmp_path, bulk_request(96, minimum=4))
    assert output.count("aps.copy_by") == 2
    assert "transfer_bytes = 64 : i64" in output
    assert "transfer_bytes = 32 : i64" in output
    assert "no legal memory interface" in arch_error(tmp_path, bulk_request(104, minimum=4))


def test_known_alignment_can_require_splitting_below_maximum(tmp_path):
    output = run_arch_pass_on_mlir(tmp_path, bulk_request(64, address=8))
    parts = [line for line in output.splitlines() if "aps.copy_by" in line]
    sizes = [int(re.search(r"transfer_bytes = (\d+) : i64", line)[1]) for line in parts]
    assert sizes == [8, 16, 32, 8]
    for line, size in zip(parts, sizes):
        offset = int(re.search(r"transfer_offset = (\d+) : i64", line)[1])
        assert (8 + offset) % size == 0
    assert "no legal memory interface" in arch_error(tmp_path, bulk_request(64, address=4))


def test_generic_narrow_interface_can_legally_cover_108_bytes(tmp_path):
    output = run_arch_pass_on_mlir(tmp_path, bulk_request(108, interfaces=SOLVER_INTERFACES))
    assert "aps.copy_by @wide" not in output
    parts = [line for line in output.splitlines() if "aps.copy_by @narrow" in line]
    assert len(parts) == 14  # Thirteen 8-byte transactions and one full 4-byte beat.
    assert sum(int(re.search(r"transfer_bytes = (\d+) : i64", line)[1])
               for line in parts) == 108


def test_native_dma_write_requires_two_beats(tmp_path):
    def write_source(size):
        return bulk_request(size).replace(
            "aps.copy %addr, (%buf)[%c0], %length : i64, (memref<256xi8>), i32, i32",
            "aps.copy (%buf)[%c0], %addr, %length : (memref<256xi8>), i32, i64, i32",
        )
    assert "no legal memory interface" in arch_error(tmp_path, write_source(8))
    assert "no legal memory interface" in arch_error(tmp_path, write_source(104))
    output = run_arch_pass_on_mlir(tmp_path, write_source(112))
    assert "min_write_beats = 2 : i64" in output
    assert output.count("aps.copy_by") == 3
    for size in [64, 32, 16]:
        assert f"transfer_bytes = {size} : i64" in output
