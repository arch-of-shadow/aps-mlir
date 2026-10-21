"""Select real 2mm CI sites, export APS hardware, and link an RV32 program.

Reuses a completed discovery cache. This verifies emitted ELF instructions;
execution of this floating-point/memory workload against APS RTL is separate.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from test_compiler import ROOT, elf_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-dir", type=Path, default=ROOT / "scaleisa/tmp/scaleisa-test"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "build/scaleisa-compiler-2mm"
    )
    args = parser.parse_args()
    cache, output = args.discovery_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(ROOT / ".pixi/envs/default/lib")
    env["MLIR_BINDINGS"] = str(
        ROOT / "circt/build/tools/mlir/python_packages/mlir_core"
    )
    report = {
        "status": "running",
        "area_budget": 1000,
        "max_extra_csrs": 5,
        "pragma": "none",
        "commands": [],
        "binary_execution": "not run; floating-point APS RTL validation is separate",
    }
    report_path = output / "binary-report.json"

    def run(label, argv):
        result = subprocess.run(
            list(map(str, argv)), cwd=ROOT, env=env, capture_output=True, text=True
        )
        (output / f"{label}.log").write_text(result.stdout + result.stderr)
        report["commands"].append(
            {
                "stage": label,
                "argv": list(map(str, argv)),
                "returncode": result.returncode,
            }
        )
        if result.returncode:
            raise RuntimeError(f"{label} failed:\n{result.stderr}")
        return result

    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        rules = json.loads((cache / "merged/rewrite-rules.json").read_text())
        rules["rules"] = [
            r
            for r in rules["rules"]
            if r["impl_id"].endswith(".direct") and len(r["input_types"]) <= 7
        ]
        (output / "sequential-rules.json").write_text(
            json.dumps(rules, indent=2) + "\n"
        )
        run(
            "selection",
            [
                ROOT / "scaleisa/tools/scaleisa-engine/target/debug/scaleisa-engine",
                cache / "selection-source/selection-scf.egraph.json",
                "--software-baseline-input-json",
                cache / "selection-source/software-baseline/input.egraph.json",
                "--rewrite-rules-json",
                output / "sequential-rules.json",
                "--area-budget",
                "1000",
                "--selection-method",
                "eboost",
                "--dump-selection-report",
                output / "selection-report.json",
                "--dump-selected-json",
                output / "selected.egraph.json",
                "--dump-selected-site-materialized-json",
                output / "sites.egraph.json",
                "--dump-selected-site-materialization-report",
                output / "sites-report.json",
            ],
        )
        run(
            "decode-sites",
            [
                sys.executable,
                ROOT / "scaleisa/scripts/scaleisa.py",
                "to-mlir",
                "--input",
                output / "sites.egraph.json",
                "--output",
                output / "sites.mlir",
            ],
        )
        os.environ["MLIR_BINDINGS"] = env["MLIR_BINDINGS"]
        sys.path.insert(0, str(ROOT / "scaleisa/python"))
        from scaleisa_egraph.selected_lowering import lower_selected_egraph_to_mlir
        from scaleisa_egraph.core.mlir_bindings import setup_mlir_bindings

        setup_mlir_bindings()
        from mlir import ir

        report["selected_materialization"] = lower_selected_egraph_to_mlir(
            output / "selected.egraph.json",
            output / "selection-report.json",
            cache / "selection-source/selection-scf.egraph.json",
            cache / "merged/materialized-candidates.egraph.json",
            cache / "merged/materialized-candidates.mlir",
            output / "sites.mlir",
            output / "sites.egraph.json",
            output / "selected.lowered.egraph.json",
            output / "selected.raw.mlir",
        )
        run(
            "clean-selected",
            [
                ROOT / "circt/build/bin/mlir-opt",
                output / "selected.raw.mlir",
                "--canonicalize",
                "--cse",
                "-o",
                output / "selected.mlir",
            ],
        )
        selected = json.loads((output / "selection-report.json").read_text())
        if [i["impl_id"] for i in selected["selected_impls"]] != ["cand26.direct"]:
            raise RuntimeError(
                "Expected the sequential 64-element dot product at budget 1000"
            )
        functions = []
        with ir.Context():
            for item in selected["selected_impls"]:
                stage = "affine" if item["origin_stage"] == "affine" else "scf"
                module = ir.Module.parse(
                    (
                        cache / "uarch" / stage / f'{item["function_name"]}.input.mlir'
                    ).read_text()
                )
                functions.extend(
                    str(f)
                    for f in module.body.operations
                    if f.operation.name == "func.func"
                )
        (output / "candidates.mlir").write_text(
            "module {\n" + "\n".join(functions) + "\n}\n"
        )
        run(
            "export-hardware",
            [
                ROOT / "build/tools/scaleisa-adapter-opt/scaleisa-adapter-opt",
                output / "candidates.mlir",
                "--scaleisa-adapter=max-extra-csrs=5 pragma=none",
                "--scaleisa-materialize-aps",
                "-o",
                output / "hardware.mlir",
            ],
        )
        clang = ROOT / ".pixi/envs/default/bin/clang"
        common = [
            clang,
            "--target=riscv32-unknown-elf",
            "-march=rv32imaf_zicsr_zifencei",
            "-mabi=ilp32f",
            "-msmall-data-limit=0",
            "-ffreestanding",
            "-fno-builtin",
            "-O2",
            "-c",
        ]
        fixtures = ROOT / "tests/scaleisa-compiler/fixtures"
        run("main-object", [*common, fixtures / "2mm_main.c", "-o", output / "main.o"])
        run("startup-object", [*common, fixtures / "start.S", "-o", output / "start.o"])
        run(
            "compile-binary",
            [
                sys.executable,
                ROOT / "tools/scaleisa-compile",
                output / "selected.mlir",
                "--hardware",
                output / "hardware.mlir",
                "--emit",
                "elf",
                "--entry",
                "_start",
                "--mattr=+m,+a,+f,+zicsr,+zifencei",
                "--abi",
                "ilp32f",
                "--link-object",
                output / "main.o",
                "--link-object",
                output / "start.o",
                f"--link-arg=-T{fixtures / 'ram.ld'}",
                "-o",
                output / "2mm.elf",
            ],
        )
        compiled = json.loads(
            (output / "2mm.elf.scaleisa/compile-report.json").read_text()
        )
        if (compiled["ci_sites"], compiled["ci_instructions"]) != (2, 2):
            raise AssertionError("Expected both 2mm dot-product sites to become CIs")
        entry, words = elf_text(output / "2mm.elf")
        instructions = [
            {
                "pc": pc,
                "word": f"0x{word:08x}",
                "opcode": word & 127,
                "funct3": (word >> 12) & 7,
                "funct7": word >> 25,
                "rd": (word >> 7) & 31,
                "rs1": (word >> 15) & 31,
                "rs2": (word >> 20) & 31,
            }
            for pc, word in words.items()
            if word & 127 == 0x2B
        ]
        if (
            entry not in words
            or len(instructions) != 2
            or any(i["funct3"] != 7 or i["funct7"] != 0 for i in instructions)
        ):
            raise AssertionError("Linked ELF has incorrect CI encodings or entry point")
        report.update(
            status="passed",
            selected_area=selected["selected_area"],
            ci_sites=2,
            custom_machine_instructions=instructions,
            elf=str(output / "2mm.elf"),
        )
        print(f"Real selected 2mm: two CI sites linked into {output / '2mm.elf'}")
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
