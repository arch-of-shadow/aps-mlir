"""Compile ScaleISA's exposed CI sites to RV32 objects or linked ELF files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def tool(name: str, override: Path | None = None) -> Path:
    if override:
        path = override.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = (
        ROOT / "circt/build/bin" / name,
        ROOT / ".pixi/envs/default/bin" / name,
    )
    for path in candidates:
        if path.is_file():
            return path
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(f"Build or install the required tool: {name}")


def publish(source: Path, destination: Path) -> None:
    # Never leave a partly written binary at the requested output path.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(source.read_bytes())
            stream.flush()
            temporary.chmod(0o755 if source.suffix == ".elf" else 0o644)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def compile_program(args: argparse.Namespace) -> dict:
    source, hardware, output = (
        args.input.resolve(),
        args.hardware.resolve(),
        args.output.resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    work = (args.work_dir or output.with_name(output.name + ".scaleisa")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "input": str(source),
        "hardware": str(hardware),
        "output": str(output),
        "emit": args.emit,
        "target": "riscv32-unknown-elf",
        "abi": args.abi,
        "mattr": args.mattr,
        "memref_abi": args.memref_abi,
        "max_extra_csrs": args.max_extra_csrs,
        "commands": [],
    }
    report_path = work / "compile-report.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command):
        command = list(map(str, command))
        result = subprocess.run(command, capture_output=True, text=True)
        (work / f"{label}.log").write_text(result.stdout + result.stderr)
        report["commands"].append(
            {"stage": label, "argv": command, "returncode": result.returncode}
        )
        save()
        if result.returncode:
            raise RuntimeError(f"{label} failed:\n{result.stderr.strip()}")
        return result.stdout

    save()
    try:
        for label, path in (("input_sha256", source), ("hardware_sha256", hardware)):
            report[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        compiler = tool(
            "scaleisa-compiler-opt",
            args.compiler
            or ROOT / "build/tools/scaleisa-compiler-opt/scaleisa-compiler-opt",
        )
        mlir_opt, translate, llc = (
            tool(name) for name in ("mlir-opt", "mlir-translate", "llc")
        )
        ci, lowered, llvm_ir = (
            work / name for name in ("ci.mlir", "llvm.mlir", "program.ll")
        )
        quoted_hardware = str(hardware).replace("\\", "\\\\").replace('"', '\\"')
        run(
            "lower-ci",
            [
                compiler,
                source,
                f'--scaleisa-lower-ci=hardware-file="{quoted_hardware}" max-extra-csrs={args.max_extra_csrs}',
                "-o",
                ci,
            ],
        )
        text = ci.read_text()
        for key in ("ci_sites", "ci_instructions"):
            match = re.search(rf"scaleisa\.{key} = (\d+) : i64", text)
            if not match:
                raise RuntimeError(f"Missing compiler count: {key}")
            report[key] = int(match[1])
        if not report["ci_sites"] and not args.allow_no_ci:
            raise RuntimeError(
                "No exposed CI sites were compiled; check the selected MLIR and hardware bindings"
            )
        bare = (
            " use-bare-ptr-memref-call-conv=true"
            if args.memref_abi == "bare-pointer"
            else ""
        )
        pipeline = (
            "builtin.module(lower-affine,convert-scf-to-cf,convert-math-to-llvm,"
            "convert-arith-to-llvm{index-bitwidth=32},"
            "finalize-memref-to-llvm{index-bitwidth=32},"
            f"convert-func-to-llvm{{index-bitwidth=32{bare}}},"
            "convert-cf-to-llvm{index-bitwidth=32},reconcile-unrealized-casts)"
        )
        run("lower-llvm", [mlir_opt, ci, f"--pass-pipeline={pipeline}", "-o", lowered])
        run("translate", [translate, lowered, "--mlir-to-llvmir", "-o", llvm_ir])
        common = [
            llc,
            llvm_ir,
            "-mtriple=riscv32-unknown-elf",
            f"-mattr={args.mattr}",
            f"-target-abi={args.abi}",
            "-relocation-model=static",
            f"-O={args.opt_level}",
        ]
        assembly, obj = work / "program.s", work / "program.o"
        run("assembly", [*common, "-filetype=asm", "-o", assembly])
        run("object", [*common, "-filetype=obj", "-o", obj])
        final = obj
        if args.emit == "elf":
            final = work / "program.elf"
            run(
                "link",
                [
                    tool("ld.lld", args.linker),
                    "-m",
                    "elf32lriscv",
                    f"--entry={args.entry}",
                    "--fatal-warnings",
                    obj,
                    *[p.resolve() for p in args.link_object],
                    *args.link_arg,
                    "-o",
                    final,
                ],
            )
        disassembly = run("disassembly", [tool("llvm-objdump"), "-d", final])
        (work / "program.disasm").write_text(disassembly)
        publish(final, output)
        report["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        report["status"] = "passed"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        raise
    print(
        f"Compiled {report['ci_sites']} CI sites into {report['ci_instructions']} custom instructions: {output}"
    )
    print(f"Report: {report_path}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="selected.mlir with exposed CI helper calls"
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        required=True,
        help="matching scaleisa-materialize-aps export",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--emit", choices=("object", "elf"), default="object")
    parser.add_argument(
        "--max-extra-csrs", type=int, choices=range(4097), default=5, metavar="N"
    )
    parser.add_argument(
        "--memref-abi", choices=("bare-pointer", "descriptor"), default="bare-pointer"
    )
    parser.add_argument("--mattr", default="+m,+a,+zicsr,+zifencei")
    parser.add_argument("--abi", choices=("ilp32", "ilp32f", "ilp32d"), default="ilp32")
    parser.add_argument("--opt-level", choices=(0, 1, 2, 3), type=int, default=2)
    parser.add_argument(
        "--entry",
        default="main",
        help="ELF entry symbol; supply platform startup via --link-object",
    )
    parser.add_argument("--link-object", type=Path, action="append", default=[])
    parser.add_argument("--link-arg", action="append", default=[])
    parser.add_argument("--linker", type=Path)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--allow-no-ci", action="store_true")
    args = parser.parse_args(argv)
    try:
        compile_program(args)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"scaleisa-compile: {error}\n")


if __name__ == "__main__":
    main()
