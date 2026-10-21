"""Build the real selected 2mm ELF/RTL and try APS Spike+Verilator co-sim."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-dir", type=Path, default=ROOT / "scaleisa/tmp/scaleisa-test"
    )
    parser.add_argument(
        "--binary-dir", type=Path, default=ROOT / "build/scaleisa-compiler-2mm"
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    binary = args.binary_dir.resolve()
    output = binary / "cosim"
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = (
        str(ROOT / ".pixi/envs/default/bin") + os.pathsep + env.get("PATH", "")
    )
    env["LD_LIBRARY_PATH"] = str(ROOT / ".pixi/envs/default/lib")
    report = {
        "status": "running",
        "area_budget": 1000,
        "max_extra_csrs": 5,
        "pragma": "none",
        "commands": [],
        "simulation_started": False,
    }
    report_path = output / "cosim-report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def run(stage, argv):
        report["stage"] = stage
        save()
        command = list(map(str, argv))
        log = output / f"{stage}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
        report["commands"].append(
            {
                "stage": stage,
                "argv": command,
                "returncode": result.returncode,
                "log": str(log),
            }
        )
        if result.returncode:
            diagnostics = [
                line
                for line in log.read_text().splitlines()
                if ": error:" in line or "LLVM ERROR:" in line
            ]
            raise RuntimeError(
                "\n".join(diagnostics[:5])
                or f"{stage} exited with {result.returncode}; see {log}"
            )

    save()
    try:
        if not args.skip_prepare:
            run(
                "prepare-binary",
                [
                    sys.executable,
                    ROOT / "tests/scaleisa-compiler/run_2mm_binary.py",
                    "--discovery-dir",
                    args.discovery_dir.resolve(),
                    "--output-dir",
                    binary,
                ],
            )
        fixtures = ROOT / "tests/scaleisa-compiler/fixtures"
        baseline = output / "2mm-cpu-baseline.elf"
        run(
            "compile-cpu-baseline",
            [
                sys.executable,
                ROOT / "tools/scaleisa-compile",
                ROOT / "scaleisa/examples/2mm_new/2mm_new.mlir",
                "--hardware",
                binary / "hardware.mlir",
                "--allow-no-ci",
                "--emit",
                "elf",
                "--entry",
                "_start",
                "--mattr=+m,+a,+f,+zicsr,+zifencei",
                "--abi",
                "ilp32f",
                "--link-object",
                binary / "main.o",
                "--link-object",
                binary / "start.o",
                f"--link-arg=-T{fixtures / 'ram.ld'}",
                "-o",
                baseline,
            ],
        )
        run(
            "cpu-baseline",
            [
                ROOT
                / "thirdparty/chipyard/toolchains/riscv-tools/riscv-isa-sim/build/spike",
                "--isa=RV32IMAF_zicsr_zifencei",
                baseline,
            ],
        )
        report["cpu_baseline"] = {
            "status": "passed",
            "checked_output_elements": 4096,
            "custom_instructions": 0,
            "elf": str(baseline),
        }
        run(
            "synthesis",
            [
                ROOT / "build/tools/aps-opt/aps-e2e",
                "--clock",
                "4.0",
                "-i",
                binary / "hardware.mlir",
                "-o",
                output / "2mm.sv",
                f"--dump-scheduled-tor={output / '2mm.tor.mlir'}",
                f"--dump-cmt2={output / '2mm.cmt2.mlir'}",
                "--export",
                "sv",
            ],
        )
        run(
            "build-simulator",
            [
                "make",
                "-C",
                ROOT / "tools/aps-cosim",
                "spike",
                f"RTL={output / '2mm.sv'}",
                "TOP=main",
                "SPIKE_OBJ_DIR=obj_dir_scaleisa_2mm",
                "VERILATE_JOBS=4",
                "BUILD_JOBS=4",
            ],
        )
        report["simulation_started"] = True
        run(
            "simulation",
            [
                ROOT / "tools/aps-cosim/obj_dir_scaleisa_2mm/Vmain",
                "--add-csr",
                "au_generated_candidate_26__direct_csr0=0x801",
                "--add-csr",
                "au_generated_candidate_26__direct_csr1=0x802",
                binary / "2mm.elf",
            ],
        )
        report.update(status="passed", checked_output_elements=4096)
        print("2mm APS co-simulation passed: all 4096 output values checked.")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report.update(status="failed", error=str(error))
        print(
            f"2mm APS co-simulation stopped at {report['stage']}: {error}",
            file=sys.stderr,
        )
    finally:
        save()
    print(f"Report: {report_path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
