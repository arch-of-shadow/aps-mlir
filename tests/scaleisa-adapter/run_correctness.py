"""Numerically validate selected sequential ScaleISA candidates through APS.

Reuse a completed multi-level discovery run, filter out all pragma variants,
select at budgets 800 and 1000, and execute both original functions and the
functional APS host model. No physical interface, RTL, or timing is modeled.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "circt/build/tools/mlir/python_packages/mlir_core"))
from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime import get_ranked_memref_descriptor


def command(argv, *, env=None):
    result = subprocess.run(
        list(map(str, argv)), cwd=ROOT, env=env, text=True, capture_output=True
    )
    if result.returncode:
        raise RuntimeError(f"Command failed: {argv}\n{result.stderr}")
    return result


def compile_module(text, destination):
    with ir.Context():
        module = ir.Module.parse(text)
        for function in module.body.operations:
            if function.operation.name == "func.func":
                function.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        pm = PassManager.parse(
            "builtin.module(lower-affine,convert-scf-to-cf,"
            "convert-arith-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
            "convert-cf-to-llvm,reconcile-unrealized-casts)"
        )
        pm.run(module.operation)
        destination.write_text(str(module))
        return ExecutionEngine(
            module,
            opt_level=0,
            shared_libs=[str(ROOT / "circt/build/lib/libmlir_c_runner_utils.so")],
        )


def descriptor(array):
    return ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(array)))


def bits(value):
    return np.asarray(value, dtype=np.float32).view(np.uint32).item()


def invoke_source(engine, name, arguments):
    packed = []
    for argument in arguments:
        if isinstance(argument, np.ndarray):
            packed.append(descriptor(argument))
        elif isinstance(argument, int):
            packed.append(ctypes.pointer(ctypes.c_int64(argument)))
        else:
            packed.append(ctypes.pointer(ctypes.c_float(float(argument))))
    result = ctypes.c_float()
    engine.invoke(name, *packed, ctypes.pointer(result))
    return np.float32(result.value)


def check_candidate(original, modeled, implementation, seed):
    name = implementation["function_name"]
    rng = np.random.default_rng(seed)
    dot = implementation["candidate_id"] == 26
    pairs = []
    if dot:
        pairs = [(0, 0), (0, 63), (63, 0), (63, 63), (7, 23), (42, 17)]
        pairs += [tuple(map(int, rng.integers(0, 64, 2))) for _ in range(18)]
    else:
        edge = np.array(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                1.0e-20,
                1.0e20,
                np.finfo(np.float32).tiny,
                np.nextafter(np.float32(1), np.float32(2)),
            ],
            dtype=np.float32,
        )
        pairs = list(zip(edge, edge[::-1]))
        pairs += [
            tuple(rng.uniform(-1000, 1000, 2).astype(np.float32)) for _ in range(64)
        ]
    for case_index, pair in enumerate(pairs):
        rf = np.full(32, 0x11223344, dtype=np.int32)
        csrs = np.full(4096, 0x556677, dtype=np.int32)
        cpu = np.full(16384, 0x1234567, dtype=np.int32)
        if dot:
            i, j = pair
            a = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
            b = rng.uniform(-1, 1, (64, 64)).astype(np.float32)
            if case_index == 0:
                a.fill(0)
            if case_index == 1:
                a.fill(1)
                b.fill(-1)
            rf[0], rf[1] = 0x1000, i
            csrs[0x801], csrs[0x802] = 0x6000, j
            cpu[0x1000 // 4 : 0x1000 // 4 + a.size] = a.reshape(-1).view(np.int32)
            cpu[0x6000 // 4 : 0x6000 // 4 + b.size] = b.reshape(-1).view(np.int32)
            a_before, b_before = a.copy(), b.copy()
            reference = invoke_source(original, name, [a, i, b, j])
            expected = np.float32(0)
            for k in range(64):
                expected = np.float32(expected + np.float32(a[i, k] * b[k, j]))
            np.testing.assert_array_equal(a, a_before)
            np.testing.assert_array_equal(b, b_before)
        else:
            lhs, rhs = map(np.float32, pair)
            rf[0] = np.asarray(lhs).view(np.int32)
            rf[1] = np.asarray(rhs).view(np.int32)
            reference = invoke_source(original, name, [lhs, rhs])
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                if implementation["candidate_id"] == 30:
                    expected = np.float32(lhs + rhs)
                elif implementation["candidate_id"] == 31:
                    expected = np.float32(lhs * rhs)
                else:
                    raise ValueError(f"No independent oracle for {name}")
        rf_before, csr_before, cpu_before = rf.copy(), csrs.copy(), cpu.copy()
        modeled.invoke(name, descriptor(rf), descriptor(csrs), descriptor(cpu))
        actual = int(rf.view(np.uint32)[2])
        if actual != bits(reference) or actual != bits(expected):
            raise AssertionError(
                f"{name} case {case_index}: model={actual:08x}, source={bits(reference):08x}, oracle={bits(expected):08x}"
            )
        np.testing.assert_array_equal(np.delete(rf, 2), np.delete(rf_before, 2))
        np.testing.assert_array_equal(csrs, csr_before)
        np.testing.assert_array_equal(cpu, cpu_before)
    return {
        "function": name,
        "cases": len(pairs),
        "bitwise_equal": True,
        "reference": "original MLIR JIT and independent sequential float32 oracle",
        "cpu_memory_and_nonresult_registers_preserved": True,
        "actual_extra_csrs": 2 if dot else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-dir", type=Path, default=ROOT / "scaleisa/tmp/scaleisa-test"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "build/scaleisa-adapter-correctness"
    )
    args = parser.parse_args()
    cached, output = args.discovery_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT / ".pixi/envs/default/lib")
    env["PATH"] = (
        str(ROOT / ".pixi/envs/default/bin") + os.pathsep + env.get("PATH", "")
    )
    rules = json.loads((cached / "merged/rewrite-rules.json").read_text())
    rules["rules"] = [
        r
        for r in rules["rules"]
        if r["impl_id"].endswith(".direct") and len(r["input_types"]) <= 7
    ]
    (output / "sequential-rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    report = {
        "pragma": "none",
        "max_extra_csrs": 5,
        "area_units": "Hector-estimated square microns; excludes adapter scratchpad/transfer/interface overhead",
        "discovery_reused_from": str(cached),
        "source_sha256": hashlib.sha256(
            (ROOT / "scaleisa/examples/2mm_new/2mm_new.mlir").read_bytes()
        ).hexdigest(),
        "validation_scope": "functional candidate semantics, not RTL or timing",
        "budgets": [],
        "status": "running",
    }
    report_file = output / "correctness-report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    for budget in (800, 1000):
        directory = output / f"budget-{budget}"
        directory.mkdir(exist_ok=True)
        argv = [
            ROOT / "scaleisa/tools/scaleisa-engine/target/debug/scaleisa-engine",
            cached / "selection-source/selection-scf.egraph.json",
            "--software-baseline-input-json",
            cached / "selection-source/software-baseline/input.egraph.json",
            "--rewrite-rules-json",
            output / "sequential-rules.json",
            "--area-budget",
            str(budget),
            "--selection-method",
            "eboost",
            "--dump-selection-report",
            directory / "selection-report.json",
            "--dump-selected-json",
            directory / "selected.egraph.json",
        ]
        result = command(argv, env=env)
        (directory / "engine.log").write_text(result.stdout + "\n" + result.stderr)
        selection = json.loads((directory / "selection-report.json").read_text())
        selected = selection["selected_impls"]
        if not selected:
            raise RuntimeError(f"No implementation selected at budget {budget}")
        sources = []
        with ir.Context():
            for item in selected:
                stage = "affine" if item["origin_stage"] == "affine" else "scf"
                source = (
                    cached / "uarch" / stage / f'{item["function_name"]}.input.mlir'
                )
                module = ir.Module.parse(source.read_text())
                sources.extend(
                    str(f)
                    for f in module.body.operations
                    if f.operation.name == "func.func"
                )
        source_file = directory / "selected-candidates.mlir"
        source_file.write_text("module {\n" + "\n".join(sources) + "\n}\n")
        planned, functional = directory / "planned.mlir", directory / "functional.mlir"
        command(
            [
                ROOT / "build/tools/scaleisa-adapter-opt/scaleisa-adapter-opt",
                source_file,
                "--scaleisa-adapter=max-extra-csrs=5 pragma=none",
                "-o",
                planned,
            ],
            env=env,
        )
        command(
            [
                ROOT / "build/tools/scaleisa-adapter-opt/scaleisa-adapter-opt",
                planned,
                "--scaleisa-materialize-aps",
                "-o",
                directory / "functional.raw.mlir",
            ],
            env=env,
        )
        command(
            [
                ROOT / "build/tools/aps-opt/aps-opt",
                directory / "functional.raw.mlir",
                "--canonicalize",
                "--cse",
                "--place-readrf-at-entry",
                "-o",
                functional,
            ],
            env=env,
        )
        for path in (source_file, planned, functional):
            if re.search(r"\b(?:pipeline|unroll|II)\s*=", path.read_text()):
                raise AssertionError(f"Unexpected pragma in {path}")
        if "scaleisa.max_extra_csrs = 5 : i32" not in planned.read_text():
            raise AssertionError(
                "Correctness export did not retain the requested CSR limit"
            )
        model_env = env.copy()
        model_env["PYTHONPATH"] = str(ROOT / "build/python_packages/circt_core")
        # The APS Python bindings register SCF but not Affine. Lower control
        # flow with the existing native pass before modeling APS operations.
        command(
            [
                ROOT / "build/tools/aps-opt/aps-opt",
                functional,
                "--lower-affine",
                "-o",
                directory / "functional.scf.mlir",
            ],
            env=env,
        )
        command(
            [
                sys.executable,
                ROOT / "tests/scaleisa-adapter/functional_model.py",
                directory / "functional.scf.mlir",
                directory / "model.mlir",
            ],
            env=model_env,
        )
        original = compile_module(
            source_file.read_text(), directory / "original.llvm.mlir"
        )
        modeled = compile_module(
            (directory / "model.mlir").read_text(), directory / "model.llvm.mlir"
        )
        results = [
            check_candidate(
                original, modeled, item, seed=1700 + budget + item["candidate_id"]
            )
            for item in selected
        ]
        point = {
            "budget": budget,
            "selected_area": selection["selected_area"],
            "selected_impls": selected,
            "checks": results,
        }
        report["budgets"].append(point)
        print(
            f"Budget {budget}: area {selection['selected_area']:.5f}; "
            f"{sum(r['cases'] for r in results)} bitwise checks passed",
            flush=True,
        )
    report["status"] = "passed"
    report["total_cases"] = sum(
        check["cases"] for point in report["budgets"] for check in point["checks"]
    )
    (output / "correctness-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output / 'correctness-report.json'}", flush=True)


if __name__ == "__main__":
    main()
