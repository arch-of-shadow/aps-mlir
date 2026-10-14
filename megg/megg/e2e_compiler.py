"""End-to-End C Compiler for Megg"""

import argparse
import dataclasses
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from megg.backend.llvm_backend import MLIRToLLVMBackend
from megg.compiler import Compiler
from megg.frontend.pragma_parser import PragmaParser
from megg.rewrites.match_rewrites import build_ruleset_from_module
from megg.utils import MModule, get_temp_dir
from megg.utils.loop_analyzer import extract_loop_lengths

logger = logging.getLogger(__name__)

@dataclass
class CompilationPaths:
    """Paths for intermediate files during compilation."""
    source_c: Path
    target_c: Path
    rest_c: Path
    target_mlir: Path
    optimized_mlir: Path
    target_ll: Path
    rest_ll: Path
    target_o: Path
    rest_o: Path
    executable: Path
    temp_dir: Path

class E2ECompiler:
    """End-to-end C code compiler with Megg optimization."""

    def __init__(self, source_file: str, custom_instr_mlir: Optional[str] = None,
                 output: str = None, encoding_json: Optional[str] = None,
                 external: bool = True,
                 keep_intermediate: bool = False,
                 verbose: bool = False,
                 handson: bool = False):
        """Initialize E2E compiler."""
        self.handson = handson
        self.snapshots = []  # Collect snapshots in handson mode
        self.source_file = Path(source_file).resolve()
        self.custom_instr_mlir = Path(custom_instr_mlir).resolve() if custom_instr_mlir else None
        self.output = Path(output).resolve() if output else None
        self.encoding_json = Path(encoding_json).resolve() if encoding_json else None
        self.external = external
        self.keep_intermediate = keep_intermediate
        self.verbose = verbose
        self.riscv = self.find_riscv()
        self.megg_include = self.find_megg_include_path()

        if not self.source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        if self.custom_instr_mlir is not None and not self.custom_instr_mlir.exists():
            raise FileNotFoundError(f"Custom instruction file not found: {custom_instr_mlir}")
        if self.encoding_json and not self.encoding_json.exists():
            raise FileNotFoundError(
                f"Encoding JSON file not found: {encoding_json}")

        self.instr_encodings = {}
        if self.encoding_json:
            self._load_encodings()

        project_tmp = get_temp_dir()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.temp_dir = project_tmp / f"megg_e2e_{timestamp}"
        self.temp_dir.mkdir(exist_ok=True)

        logger.info(f"Using temporary directory: {self.temp_dir}")

        self.paths = self._init_paths()

    def find_riscv(self):
        riscv = os.getenv("RISCV")
        if not riscv:
            gcc_path = shutil.which('riscv32-unknown-elf-gcc')
            if gcc_path is None:
                raise FileNotFoundError('Set RISCV or put riscv32-unknown-elf-gcc on PATH')
            riscv = str(Path(gcc_path).resolve().parent.parent)
        return riscv

    def find_megg_include_path(self):
        megg_include_path = os.getenv("MEGGINCLUDE")
        if megg_include_path is None:
            logger.warning("i can't find megg compiler's private include file, set to /usr/include")
            return '/usr/include'
        return megg_include_path

    def _load_encodings(self):
        """Load instruction encodings from JSON file."""

        with open(self.encoding_json, 'r') as f:
            self.instr_encodings = json.load(f)

        logger.info(
            f"Loaded encodings for {len(self.instr_encodings)} instruction(s):")
        for instr_name, encoding in self.instr_encodings.items():
            opcode = encoding.get('opcode', 'N/A')
            funct3 = encoding.get('funct3', '0x0')
            funct7 = encoding.get('funct7', 'N/A')
            logger.info(
                f"  - {instr_name}: opcode={opcode}, funct3={funct3}, funct7={funct7}")

    def _init_paths(self) -> CompilationPaths:
        """Initialize file paths for compilation stages."""
        return CompilationPaths(
            source_c=self.source_file,
            target_c=self.temp_dir / "target.c",
            rest_c=self.temp_dir / "rest.c",
            target_mlir=self.temp_dir / "target.mlir",
            optimized_mlir=self.temp_dir / "optimized.mlir",
            target_ll=self.temp_dir / "target.ll",
            rest_ll=self.temp_dir / "rest.ll",
            target_o=self.temp_dir / "target.o",
            rest_o=self.temp_dir / "rest.o",
            executable=self.output,
            temp_dir=self.temp_dir
        )

    def compile(self) -> bool:
        """Execute complete compilation pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("Step 1: Parsing pragmas and splitting code")
            logger.info("=" * 60)
            marked_funcs = self._parse_pragmas()

            if not marked_funcs:
                logger.info("No functions marked with #pragma megg optimize")
                logger.info("Compiling entire file without Megg optimization")
                return self._compile_direct()

            self._split_code(marked_funcs)

            logger.info("\n" + "=" * 60)
            logger.info("Step 2: Compiling target functions (Megg path)")
            logger.info("=" * 60)
            if not self._compile_target_path():
                logger.error("Target path compilation failed")
                return False

            logger.info("\n" + "=" * 60)
            logger.info("Step 3: Compiling rest functions (LLVM path)")
            logger.info("=" * 60)
            if not self._compile_rest_path():
                logger.error("Rest path compilation failed")
                return False

            logger.info("\n" + "=" * 60)
            logger.info("Step 4: Linking")
            logger.info("=" * 60)
            if not self._link():
                logger.error("Linking failed")
                return False

            logger.info("\n" + "=" * 60)
            logger.info(f"✓ Successfully built: {self.output}")
            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            logger.error(traceback.format_exc())
            return False

        finally:
            if not self.keep_intermediate:
                self._cleanup()

    def _parse_pragmas(self) -> List:
        """Parse pragma directives and identify marked functions."""
        parser = PragmaParser(str(self.source_file))
        marked_funcs = parser.find_marked_functions()

        logger.info(f"Found {len(marked_funcs)} marked function(s):")
        for func_name, start, end in marked_funcs:
            logger.info(f"  - {func_name} (lines {start+1}-{end+1})")

        return marked_funcs

    def _split_code(self, marked_funcs: List):
        """Split source code into target.c and rest.c."""
        parser = PragmaParser(str(self.source_file))
        target_code, rest_code = parser.split_code(marked_funcs)

        with open(self.paths.target_c, 'w') as f:
            f.write(target_code)
        logger.info(f"Created: {self.paths.target_c}")

        with open(self.paths.rest_c, 'w') as f:
            f.write(rest_code)
        logger.info(f"Created: {self.paths.rest_c}")

        if self.verbose:
            logger.info(f"\n--- target.c ---\n{target_code}\n")
            logger.info(f"\n--- rest.c ---\n{rest_code}\n")

    def _compile_target_path(self) -> bool:
        """Compile target functions through Megg pipeline:"""
        logger.info("  2.1: Converting C to MLIR (Polygeist)...")
        if not self._run_polygeist():
            return False

        logger.info("  2.2: Optimizing MLIR (Megg)...")
        if not self._run_megg_optimizer():
            return False

        logger.info("  2.3: Lowering MLIR to LLVM IR...")
        if not self._run_mlir_lower_via_backend():
            return False

        logger.info("  2.4: Generating object file (LLC)...")
        if not self._run_llc(self.paths.target_ll, self.paths.target_o):
            return False

        logger.info(f"  ✓ Target path complete: {self.paths.target_o}")
        return True

    def _compile_rest_path(self) -> bool:
        """Compile rest functions through LLVM pipeline:"""
        logger.info("  3.1: Compiling C to LLVM IR (Clang for RISC-V)...")
        cmd = [
            'clang',
            '--target=riscv32-unknown-linux-elf',
            '-march=rv32imac_zicsr_zifencei', # Chipyard: with RVC compression
            '-mabi=ilp32',      # Soft-float ABI (must match GCC)
            '-mcmodel=medany',  # Chipyard memory model
            '-I' + self.riscv + '/riscv32-unknown-elf/include',
            '-I' + self.riscv + '/include',
            '-I' + self.megg_include,
            '-emit-llvm',
            '-S',  # Generate .ll (text format)
            '-O2',
            str(self.paths.rest_c),
            '-o', str(self.paths.rest_ll)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Clang failed:")
            logger.error(result.stderr)
            return False

        logger.info(f"    Created: {self.paths.rest_ll}")

        logger.info("  3.2: Generating object file (LLC)...")
        if not self._run_llc(self.paths.rest_ll, self.paths.rest_o):
            return False

        logger.info(f"  ✓ Rest path complete: {self.paths.rest_o}")
        return True

    def _strip_module_attributes(self, mlir_path: Path) -> None:
        """Remove top-level module attributes to avoid incompatible layouts."""
        try:
            mlir_text = mlir_path.read_text(encoding='utf-8')
        except OSError as exc:
            logger.warning(
                f"    Failed to read MLIR file for attribute cleanup: {exc}")
            return

        pattern = re.compile(
            r'^(\s*module)\s+attributes\s*\{.*?\}\s*\{', re.DOTALL | re.MULTILINE)
        cleaned_text, count = pattern.subn(r'\1 {', mlir_text, count=1)

        if count == 0:
            logger.debug(
                "    No module-level attributes detected; skipping cleanup")
            return

        try:
            mlir_path.write_text(cleaned_text, encoding='utf-8')
            logger.info(
                f"    Removed module-level attributes from: {mlir_path}")
        except OSError as exc:
            logger.warning(f"    Failed to write cleaned MLIR file: {exc}")

    def _run_polygeist(self) -> bool:
        """Convert C to MLIR using cgeist."""
        cmd = [
            'cgeist', "-c", "-S",
            str(self.paths.target_c),
            "--raise-scf-to-affine", "-O3", "--memref-fullrank",
            '-o', str(self.paths.target_mlir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Polygeist conversion failed:")
            logger.error(result.stderr)
            return False

        logger.info(f"    Created: {self.paths.target_mlir}")
        self._strip_module_attributes(self.paths.target_mlir)
        self._lower_affine_with_polygeist_opt(self.paths.target_mlir)
        return True

    def _lower_affine_with_polygeist_opt(self, mlir_path: Path):
        cmd = [
            'mlir-opt',
            '--lower-affine',
            str(mlir_path),
            '-o', str(mlir_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("    lowering affine to scf failed:")
            logger.error(result.stderr)
            return

    def _run_megg_optimizer(self) -> bool:
        """Optimize MLIR using Megg compiler."""
        try:

            logger.info(f"    Loading target MLIR: {self.paths.target_mlir}")
            target_module = MModule.parse_from_file(str(self.paths.target_mlir))
            custom_ruleset = None
            skeletons = None
            loop_hints = None

            if self.custom_instr_mlir:

                logger.info(f"    Loading pattern MLIR: {self.custom_instr_mlir}")
                pattern_module = MModule.parse_from_file(str(self.custom_instr_mlir))

                logger.info("    Building custom instruction ruleset...")
                custom_ruleset, skeletons = build_ruleset_from_module(
                    pattern_module, instr_encodings=self.instr_encodings)
                logger.info(f"    Found {len(skeletons)} skeleton(s)")

                loop_hints = extract_loop_lengths(pattern_module)
                logger.info(f"    Extracted loop hints: {loop_hints}")
            else:
                logger.info("    No custom instructions provided, running basic optimization only")

            debug_callback = None
            if self.handson:
                def debug_callback(snapshot):
                    self.snapshots.append(snapshot)
                logger.info("    Handson mode enabled - collecting phase snapshots")

            compiler = Compiler(
                module=target_module,
                match_ruleset=custom_ruleset,
                skeletons=skeletons,
                loop_hints=loop_hints,  # Pass loop hints for external passes
                instr_encodings=self.instr_encodings if hasattr(self, 'instr_encodings') else None,  # Pass encodings
                debug_callback=debug_callback  # Pass handson callback
            )

            logger.info("    Running Megg optimization...")
            result = compiler.schedule(
                internal_rewrites=True,
                external_passes=self.external,
                custom_rewrites=(custom_ruleset is not None),
                output_path=str(self.paths.optimized_mlir)
            )

            if self.handson and self.snapshots:
                self._save_handson_snapshots()

            if not result.success:
                logger.error(
                    f"    Megg optimization failed: {result.error_message}")
                return False

            logger.info(f"    ✓ Optimization complete:")
            logger.info(
                f"      - Internal rewrites: {result.internal_rewrites}")
            logger.info(f"      - Custom rewrites: {result.custom_rewrites}")
            logger.info(f"      - Time: {result.time_elapsed:.2f}s")
            logger.info(f"    Created: {self.paths.optimized_mlir}")

            self._save_statistics_report(compiler)

            return True

        except Exception as e:
            logger.error(f"Megg optimization failed: {e}")
            logger.error(traceback.format_exc())
            return False

    def _run_mlir_lower_via_backend(self) -> bool:
        """Lower MLIR to LLVM IR using existing backend."""
        logger.info("    >>> Entering _run_mlir_lower_via_backend")
        try:

            logger.info(
                f"    Loading optimized MLIR from: {self.paths.optimized_mlir}")
            optimized_module = MModule.parse_from_file(
                str(self.paths.optimized_mlir))
            logger.info(
                f"    Module loaded, has {len(list(optimized_module.get_functions()))} function(s)")

            logger.info("    Creating backend...")
            backend = MLIRToLLVMBackend(verbose=self.verbose)
            logger.info("    Calling emit_llvm_ir...")
            llvm_ir = backend.emit_llvm_ir(
                optimized_module, output_path=str(self.paths.target_ll))

            logger.info(
                f"    emit_llvm_ir returned {len(llvm_ir) if llvm_ir else 0} chars")

            if llvm_ir:
                logger.info(f"    Created: {self.paths.target_ll}")
                return True
            else:
                logger.error("    Failed to generate LLVM IR")
                return False

        except Exception as e:
            logger.error(f"MLIR lowering failed: {e}")
            logger.error(traceback.format_exc())
            return False

    def _run_llc(self, input_ll: Path, output_o: Path) -> bool:
        """Compile LLVM IR to object file using LLC (RISC-V target)."""
        llc = shutil.which('llc')
        if not llc:
            logger.error("    I can't find llc! quit now!")
            return False

        target = 'riscv32-unknown-linux-elf'
        # Use soft-float ABI (ilp32) for Chipyard compatibility
        mattr = '+m,+a,+zicsr,+zifencei'  # Integer multiply, atomics, no RVC

        cmd = [
            'llc',
            '-mtriple=' + target,
            '-mattr=' + mattr,
            '-target-abi=ilp32',  # Soft-float ABI
            '-filetype=obj',
            str(input_ll),
            '-o', str(output_o)
        ]

        logger.info(
            f"    Compiling for RISC-V 32-bit (soft-float ABI): {target}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"LLC failed:")
            logger.error(result.stderr)
            logger.error(f"Command: {' '.join(cmd)}")
            return False

        logger.info(f"    Created: {output_o}")
        return True

    def _compile_with_clang(self, input_ll: Path, output_o: Path) -> bool:
        """Compile LLVM IR using Clang."""
        cmd = [
            'clang',
            '-c',  # Compile to object file only
            '-O2',
            str(input_ll),
            '-o', str(output_o)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Clang compilation failed:")
            logger.error(result.stderr)
            return False

        logger.info(f"    Created: {output_o} (via Clang)")
        return True

    def _link(self) -> bool:
        """Link object files into RISC-V executable."""
        gcc_path = self.riscv + '/bin/riscv32-unknown-elf-gcc'
        cmd = [
            gcc_path,
            '-march=rv32imac_zicsr_zifencei', # Chipyard: with RVC compression
            '-mabi=ilp32',                     # Soft-float ABI
            '-specs=htif_nano.specs',          # HTIF for Chipyard/Spike
            '-static',                         # Static linking for baremetal
            str(self.paths.target_o),
            str(self.paths.rest_o),
            '-o', str(self.paths.executable)
        ]
        logger.info(f"  Using htif_nano specs for Chipyard/Spike")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Linking failed:")
            logger.error(result.stderr)
            logger.error(f"Command: {' '.join(cmd)}")
            return False

        logger.info(f"  Created RISC-V executable: {self.paths.executable}")

        self._generate_disassembly()

        return True

    def _generate_disassembly(self):
        """Generate disassembly files for debugging."""
        try:
            objdump = self.riscv + '/bin/riscv32-unknown-elf-objdump'
            asm_path = self.paths.executable.with_suffix('.asm')
            cmd = [
                objdump,
                '-d',           # Disassemble
                '-S',           # Interleave source code if available
                '--wide',       # Don't wrap long lines
                str(self.paths.executable)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                with open(asm_path, 'w') as f:
                    f.write(result.stdout)
                logger.info(f"  Generated disassembly: {asm_path}")
            else:
                logger.warning(
                    f"  Failed to generate disassembly: {result.stderr}")

            if self.keep_intermediate:
                target_asm = self.paths.temp_dir / "target.asm"
                cmd = [
                    objdump,
                    '-d',
                    '-S',
                    '--wide',
                    str(self.paths.target_o)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    with open(target_asm, 'w') as f:
                        f.write(result.stdout)
                    logger.info(
                        f"  Generated target disassembly: {target_asm}")

        except Exception as e:
            logger.warning(f"  Failed to generate disassembly: {e}")

    def _compile_direct(self) -> bool:
        """Compile entire file directly using RISC-V GCC without Megg optimization."""
        logger.info("\n" + "=" * 60)
        logger.info("Direct compilation (no Megg optimization)")
        logger.info("=" * 60)

        riscv_gcc = self.riscv + '/bin/riscv32-unknown-elf-gcc'
        cmd = [
            riscv_gcc,
            '-march=rv32imac_zicsr_zifencei', # Chipyard: with RVC compression
            '-mabi=ilp32',                     # Soft-float ABI
            '-specs=htif_nano.specs',          # HTIF for Chipyard/Spike
            '-static',                         # Static linking for baremetal
            '-O2',                             # Basic optimization
            '-I' + self.riscv + '/riscv32-unknown-elf/include',
            '-I' + self.riscv + '/include',
            '-I' + self.megg_include,
            str(self.source_file),
            '-o', str(self.output)
        ]

        logger.info(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Direct compilation failed:")
            logger.error(result.stderr)
            return False

        logger.info(f"✓ Successfully compiled: {self.output}")

        self._generate_disassembly()

        return True

    def _save_statistics_report(self, compiler):
        """Save compilation statistics to JSON file."""
        try:

            stats_report = compiler.generate_statistics_report()

            stats_path = self.output.with_suffix('.stats.json')
            with open(stats_path, 'w') as f:
                json.dump(stats_report, f, indent=2)

            logger.info(f"    Generated statistics report: {stats_path}")

            internal = stats_report.get('internal_rewrites', 0)
            external = stats_report.get('external_rewrites', [])
            custom = stats_report.get('custom_rewrites', 0)
            egraph = stats_report.get('egraph_statistics', {})

            logger.info(f"    Statistics summary:")
            logger.info(f"      - Internal rewrites used: {internal}")
            logger.info(f"      - External passes applied: {len(external)}")
            logger.info(f"      - Custom patterns matched: {custom}")
            logger.info(f"      - E-graph size: {egraph.get('num_eclasses', 0)} eclasses, {egraph.get('num_enodes', 0)} enodes")

        except Exception as e:
            logger.warning(f"Failed to save statistics report: {e}")

    def _save_handson_snapshots(self):
        """Save phase snapshots to JSON file for hands-on visualization."""

        def _json_default(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return asdict(obj)
            return str(obj)

        try:
            snapshots_data = []
            for snapshot in self.snapshots:
                snapshot_dict = {
                    'phase_name': snapshot.phase_name,
                    'phase_index': snapshot.phase_index,
                    'egraph_stats': snapshot.egraph_stats,
                    'cumulative_stats': snapshot.cumulative_stats,
                    'details': snapshot.details,
                    'timestamp': snapshot.timestamp
                }
                snapshots_data.append(snapshot_dict)

            snapshots_path = self.output.with_suffix('.snapshots.json')
            with open(snapshots_path, 'w') as f:
                json.dump({
                    'source_file': str(self.source_file),
                    'pattern_file': str(self.custom_instr_mlir) if self.custom_instr_mlir else None,
                    'total_phases': len(snapshots_data),
                    'snapshots': snapshots_data
                }, f, indent=2, default=_json_default)

            logger.info(f"    Handson snapshots saved: {snapshots_path}")
            logger.info(f"    Total phases captured: {len(snapshots_data)}")

            print("\n" + "=" * 60)
            print("HANDSON MODE: Phase Snapshots")
            print("=" * 60)
            for i, snapshot in enumerate(snapshots_data):
                print(f"\n[Phase {i}] {snapshot['phase_name']}")
                stats = snapshot['cumulative_stats']
                print(f"  Internal rewrites: {stats.get('internal_rewrites', 0)}")
                print(f"  External rewrites: {stats.get('external_rewrites', 0)}")
                print(f"  Custom rewrites: {stats.get('custom_rewrites', 0)}")
                egraph = snapshot['egraph_stats']
                if egraph:
                    for func_name, func_stats in egraph.items():
                        print(f"  E-graph ({func_name}): {func_stats.get('total_eclasses', 0)} eclasses, {func_stats.get('total_nodes', 0)} enodes")
            print("=" * 60 + "\n")

        except Exception as e:
            logger.warning(f"Failed to save handson snapshots: {e}")
            logger.warning(traceback.format_exc())

    def _cleanup(self):
        """Remove temporary directory and intermediate files."""
        if not self.keep_intermediate and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        elif self.keep_intermediate:
            logger.info(f"Intermediate files preserved in: {self.temp_dir}")

def main():
    """Simple test of E2E compiler."""

    parser = argparse.ArgumentParser(description='Megg E2E Compiler')
    parser.add_argument('input', help='Input C source file')
    parser.add_argument('--custom-instructions',
                        help='Optional custom instruction MLIR file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output executable')
    parser.add_argument('--encoding-json', type=str,
                        help='JSON file with instruction encodings (opcode/funct3/funct7)')
    parser.add_argument('--mlir-passes', type=str,
                        help='Semicolon-separated MLIR passes (e.g., "canonicalize;cse")')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='Keep intermediate files')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--handson', action='store_true',
                        help='Enable handson mode - saves phase snapshots for tutorial visualization')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(message)s'
    )

    compiler = E2ECompiler(
        source_file=args.input,
        custom_instr_mlir=args.custom_instructions,
        output=args.output,
        encoding_json=args.encoding_json,
        keep_intermediate=args.keep_intermediate,
        verbose=args.verbose,
        handson=args.handson
    )

    success = compiler.compile()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
