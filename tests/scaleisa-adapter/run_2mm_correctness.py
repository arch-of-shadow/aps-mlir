"""Test the complete 2mm computation with a selected APS dot-product model.

This explicitly constructs a test-only call-site substitution. It does not test
ScaleISA e-graph extraction, emit CPU instructions, or model hardware timing.
Run run_correctness.py first to generate the budget-1000 functional model.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from run_correctness import ROOT, compile_module, descriptor, ir
from mlir.dialects import func


BRIDGE = """
func.func private @aps_dot_bridge(%a: memref<64x64xf32>, %i: index,
    %b: memref<64x64xf32>, %j: index, %rf: memref<32xi32>,
    %csr: memref<4096xi32>, %cpu: memref<?xi32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %base_a = arith.constant 1024 : index
  %base_b = arith.constant 6144 : index
  %csr_b = arith.constant 2049 : index
  %csr_j = arith.constant 2050 : index
  %addr_a = arith.constant 4096 : i32
  %addr_b = arith.constant 24576 : i32
  // Test-only host staging: all array data originates in CPU memory.
  affine.for %row = 0 to 64 {
    affine.for %col = 0 to 64 {
      %flat_row = arith.muli %row, %c64 : index
      %flat = arith.addi %flat_row, %col : index
      %ia = arith.addi %base_a, %flat : index
      %ib = arith.addi %base_b, %flat : index
      %va = affine.load %a[%row, %col] : memref<64x64xf32>
      %vb = affine.load %b[%row, %col] : memref<64x64xf32>
      %ba = arith.bitcast %va : f32 to i32
      %bb = arith.bitcast %vb : f32 to i32
      memref.store %ba, %cpu[%ia] : memref<?xi32>
      memref.store %bb, %cpu[%ib] : memref<?xi32>
    }
  }
  %i32 = arith.index_cast %i : index to i32
  %j32 = arith.index_cast %j : index to i32
  memref.store %addr_a, %rf[%c0] : memref<32xi32>
  memref.store %i32, %rf[%c1] : memref<32xi32>
  memref.store %addr_b, %csr[%csr_b] : memref<4096xi32>
  memref.store %j32, %csr[%csr_j] : memref<4096xi32>
  func.call @DOT_NAME(%rf, %csr, %cpu) :
      (memref<32xi32>, memref<4096xi32>, memref<?xi32>) -> ()
  %bits = memref.load %rf[%c2] : memref<32xi32>
  %value = arith.bitcast %bits : i32 to f32
  return %value : f32
}
"""


def integration_module(source, model, dot_name):
    """Replace exactly the two known sequential reductions, preserving callers."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        function = next(iter(module.body.operations))
        block = function.regions[0].blocks[0]
        function.attributes["sym_name"] = ir.StringAttr.get("two_stage_mm_aps")
        extra_types = [
            ir.Type.parse("memref<32xi32>"),
            ir.Type.parse("memref<4096xi32>"),
            ir.Type.parse("memref<?xi32>"),
        ]
        states = [block.add_argument(t, ir.Location.unknown()) for t in extra_types]
        function.attributes["function_type"] = ir.TypeAttr.get(
            ir.FunctionType.get([arg.type for arg in block.arguments], [])
        )
        sites = []

        def visit(region):
            for nested_block in region.blocks:
                for view in list(nested_block.operations):
                    op = view.operation
                    if op.name == "affine.for" and len(op.results) == 1:
                        sites.append(op)
                    else:
                        for child in op.regions:
                            visit(child)

        visit(function.regions[0])
        if len(sites) != 2:
            raise AssertionError(f"Expected two dot-product sites, got {len(sites)}")
        for loop in sites:
            body = loop.regions[0].blocks[0]
            operations = list(body.operations)
            if [op.operation.name for op in operations] != [
                "affine.load",
                "affine.load",
                "arith.mulf",
                "arith.addf",
                "affine.yield",
            ]:
                raise AssertionError("Unexpected dot-product body")
            a, b, mul, add, yielded = operations
            iv, acc = body.arguments
            zero = ir.FloatAttr(loop.operands[0].owner.attributes["value"]).value
            checks = [
                str(loop.attributes["lowerBoundMap"]) == "affine_map<() -> (0)>",
                str(loop.attributes["upperBoundMap"]) == "affine_map<() -> (64)>",
                ir.IntegerAttr(loop.attributes["step"]).value == 1,
                zero == 0.0,
                a.operands[2] == iv,
                b.operands[1] == iv,
                list(mul.operands) == [a.results[0], b.results[0]],
                list(add.operands) == [acc, mul.results[0]],
                yielded.operands[0] == add.results[0],
            ]
            if not all(checks):
                raise AssertionError(
                    "Dot-product bounds/dataflow differ from candidate"
                )
            with ir.InsertionPoint(loop):
                call = func.CallOp(
                    [ir.F32Type.get()],
                    "aps_dot_bridge",
                    [a.operands[0], a.operands[1], b.operands[0], b.operands[2]]
                    + states,
                )
            loop.results[0].replace_all_uses_with(call.results[0])
            loop.erase()
        host = str(function)
        modeled = ir.Module.parse(model)
        parts = [str(op) for op in modeled.body.operations]
        text = (
            "module {\n"
            + "\n".join(parts)
            + "\n"
            + BRIDGE.replace("DOT_NAME", dot_name)
            + "\n"
            + host
            + "\n}\n"
        )
        result = ir.Module.parse(text)
        result.operation.verify()
        return str(result), len(sites)


def sequential_mm(a, b):
    result = np.zeros((64, 64), dtype=np.float32)
    for k in range(64):
        # Separate ufuncs preserve the source's multiply-then-add order.
        product = np.multiply(a[:, k, None], b[k, None, :], dtype=np.float32)
        np.add(result, product, out=result)
    return result


def inputs():
    rng = np.random.default_rng(20261015)
    zero = np.zeros((64, 64), dtype=np.float32)
    eye = np.eye(64, dtype=np.float32)
    random = lambda: rng.uniform(-1, 1, (64, 64)).astype(np.float32)
    yield "zero", [zero.copy() for _ in range(4)]
    yield "identity", [eye.copy(), random(), eye.copy(), random()]
    basis = zero.copy()
    basis[0, 63], basis[63, 0] = 2, -3
    yield "boundary-basis", [basis, eye.copy(), eye.copy(), zero.copy()]
    yield "signed-constant", [
        np.full((64, 64), v, dtype=np.float32) for v in (0.25, -0.5, 0.125, -2)
    ]
    for case in range(3):
        yield f"random-{case}", [random() for _ in range(4)]


def assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-run",
        type=Path,
        default=ROOT / "build/scaleisa-adapter-correctness",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "build/scaleisa-adapter-2mm-correctness",
    )
    args = parser.parse_args()
    artifacts, output = args.candidate_run.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((artifacts / "correctness-report.json").read_text())
    source = (ROOT / "scaleisa/examples/2mm_new/2mm_new.mlir").read_text()
    if (
        selection["status"] != "passed"
        or selection["pragma"] != "none"
        or selection["max_extra_csrs"] != 5
        or selection["source_sha256"] != hashlib.sha256(source.encode()).hexdigest()
    ):
        raise AssertionError(
            "Run candidate correctness with the requested settings first"
        )
    point = next(p for p in selection["budgets"] if p["budget"] == 1000)
    if [item["impl_id"] for item in point["selected_impls"]] != ["cand26.direct"]:
        raise AssertionError(
            "This integration test requires the sequential dot candidate"
        )
    name = point["selected_impls"][0]["function_name"]
    model_path = artifacts / "budget-1000/model.mlir"
    model = model_path.read_text()
    integrated, sites = integration_module(source, model, name)
    (output / "integrated.mlir").write_text(integrated)
    report = {
        "status": "running",
        "area_budget": 1000,
        "selected_area": point["selected_area"],
        "max_extra_csrs": 5,
        "actual_extra_csrs": 2,
        "pragma": "none",
        "selected_impl": "cand26.direct",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "model_sha256": hashlib.sha256(model.encode()).hexdigest(),
        "replaced_static_sites": sites,
        "scope": "full 2mm with explicit test-only call-site substitution and functional APS model; excludes e-graph extraction, CPU instruction emission, RTL and timing",
        "area_scope": selection["area_units"],
        "cases": [],
    }
    report_path = output / "correctness-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    original = compile_module(source, output / "original.llvm.mlir")
    modeled = compile_module(integrated, output / "integrated.llvm.mlir")
    rf = np.full(32, 0x11223344, dtype=np.int32)
    csr = np.full(4096, 0x556677, dtype=np.int32)
    cpu = np.full(16384, 0x1234567, dtype=np.int32)
    guard = np.ones(cpu.shape, dtype=bool)
    guard[1024:5120] = False
    guard[6144:10240] = False
    # Reuse all state across cases to expose stale scratchpad/register contents.
    for label, matrices in inputs():
        before = [a.copy() for a in matrices]
        software = np.full((64, 64), np.nan, dtype=np.float32)
        actual = software.copy()
        middle = sequential_mm(matrices[0], matrices[1])
        expected = np.add(sequential_mm(middle, matrices[2]), matrices[3])
        original.invoke("two_stage_mm", *map(descriptor, matrices + [software]))
        rf_before, csr_before, cpu_before = rf.copy(), csr.copy(), cpu.copy()
        modeled.invoke(
            "two_stage_mm_aps",
            *map(descriptor, matrices + [actual, rf, csr, cpu]),
        )
        assert_bits(actual, software)
        assert_bits(actual, expected)
        for a, saved in zip(matrices, before):
            assert_bits(a, saved)
        np.testing.assert_array_equal(rf[3:], rf_before[3:])
        np.testing.assert_array_equal(
            np.delete(csr, [2049, 2050]), np.delete(csr_before, [2049, 2050])
        )
        np.testing.assert_array_equal(cpu[guard], cpu_before[guard])
        assert_bits(cpu[1024:5120].view(np.float32).reshape(64, 64), middle)
        assert_bits(cpu[6144:10240].view(np.float32).reshape(64, 64), matrices[2])
        report["cases"].append(
            {
                "name": label,
                "output_elements": actual.size,
                "aps_invocations": 2 * 64 * 64,
                "bitwise_equal_to_original_and_numpy": True,
                "inputs_and_unrelated_state_preserved": True,
            }
        )
        print(f"{label}: all 4096 outputs match bitwise; 8192 APS calls", flush=True)
    report["status"] = "passed"
    report["total_output_elements"] = sum(c["output_elements"] for c in report["cases"])
    report["total_aps_invocations"] = sum(c["aps_invocations"] for c in report["cases"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
