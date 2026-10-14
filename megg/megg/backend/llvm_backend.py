"""Lower MLIR through LLVM tools and emit IR, objects, or executables."""

import argparse
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from megg.utils import MModule

logger = logging.getLogger(__name__)

def rtype_asm(encoding: dict, num_inputs: int, has_result: bool) -> str:
    """Map matched operands back to the CADL signature and emit CSR setup."""
    indices = encoding.get('operand_indices', list(range(num_inputs)))
    if len(indices) != num_inputs or len(set(indices)) != num_inputs:
        raise ValueError('Invalid custom instruction operand_indices')
    offset = int(has_result)
    operands = {arg: f'${i + offset}' for i, arg in enumerate(indices)}
    csr_inputs = encoding.get('csr_inputs', [])
    csr_indices = set()
    addresses = set()
    lines = []
    for csr in csr_inputs:
        arg = csr['arg_index']
        address = csr['address']
        if isinstance(address, str):
            address = int(address, 0)
        if not isinstance(address, int) or not 0 <= address <= 0xFFF or address & 0xC00 == 0xC00:
            raise ValueError('CSR setup requires a writable 12-bit address')
        if arg < 2 or arg in csr_indices or address in addresses:
            raise ValueError('CSR inputs require distinct arguments and addresses')
        csr_indices.add(arg)
        addresses.add(address)
        if arg in operands:
            lines.append(f'csrrw x0, {address:#x}, {operands[arg]}')
    if set(indices) - {0, 1} - csr_indices:
        raise ValueError('R-type inputs beyond rs1/rs2 require CSR bindings')
    rd = '$0' if has_result else 'x0'
    rs1, rs2 = operands.get(0, 'x0'), operands.get(1, 'x0')
    opcode = encoding.get('opcode', '0x0')
    funct3 = encoding.get('funct3', '0x7')
    funct3 = int(funct3, 0) if isinstance(funct3, str) else int(funct3)
    if not 0 <= funct3 <= 7:
        raise ValueError('R-type funct3 must fit in three bits')
    # RoCC funct3 is {xd, xs1, xs2}; only value-producing calls request rd.
    funct3 = (funct3 & 0x3) | (int(has_result) << 2)
    funct7 = encoding.get('funct7', '0x0')
    lines.append(f'.insn r {opcode}, {funct3:#x}, {funct7}, {rd}, {rs1}, {rs2}')
    return '\n'.join(lines)

class MLIRToLLVMBackend:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @staticmethod
    def _run(tool: str, *args: str) -> str:
        executable = shutil.which(tool)
        if executable is None:
            raise FileNotFoundError(f"Required LLVM tool not found: {tool}")
        command = [executable, *map(str, args)]
        logger.debug("Running %s", command)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{tool} failed ({result.returncode}): {result.stderr.strip()}")
        return result.stdout

    @staticmethod
    def emit_custom_instruction(builder, operands, result_type, name: str, properties: dict):
        encoding = properties.get('encoding')
        if encoding is None:
            raise ValueError(f"Instruction {name} has no RISC-V encoding")
        asm = rtype_asm(encoding, len(operands), result_type is not None)
        if result_type is not None and str(result_type) == 'index':
            result_type = builder.i32()
        constraints = ['=r'] if result_type is not None else []
        constraints.extend(['r'] * len(operands))
        clobbers = list(dict.fromkeys([*properties.get('clobbers', []), 'memory']))
        constraints.extend(f'~{{{clobber}}}' for clobber in clobbers)
        converted = []
        # RV32 custom operands carry pointer addresses, not memref descriptors.
        for value in operands:
            kind = str(value.type)
            if kind == '!llvm.ptr':
                value = builder.ptrtoint(value, builder.i32())
            elif kind.startswith('memref<'):
                value = builder.extract_aligned_pointer_as_index(value)
                value = builder.index_cast(value, builder.i32())
            elif kind == 'index':
                value = builder.index_cast(value, builder.i32())
            converted.append(value)
        return builder.inline_asm(
            asm_string=asm, constraints=','.join(constraints), operands=converted,
            result_type=result_type, has_side_effects=True)

    def emit_llvm_ir(self, module, output_path: Optional[str] = None) -> str:
        with tempfile.TemporaryDirectory(prefix="megg_llvm_") as directory:
            source = Path(directory) / "input.mlir"
            lowered = Path(directory) / "llvm.mlir"
            source.write_text(module.to_string())
            self._run("mlir-opt", "--test-lower-to-llvm", source, "-o", lowered)
            llvm_ir = self._run("mlir-translate", "--mlir-to-llvmir", lowered)
        if output_path:
            Path(output_path).write_text(llvm_ir)
        return llvm_ir

    def compile_to_object(self, llvm_ir: str, output_path: str,
                          optimization_level: str = "O2",
                          target: Optional[str] = None) -> bool:
        with tempfile.TemporaryDirectory(prefix="megg_object_") as directory:
            source = Path(directory) / "input.ll"
            source.write_text(llvm_ir)
            arguments = [f"-{optimization_level}", "-filetype=obj", "-o", output_path]
            if target:
                arguments.append(f"-mtriple={target}")
            self._run("llc", *arguments, source)
        return True

    def compile_to_executable(self, llvm_ir: str, output_path: str,
                             optimization_level: str = "O2",
                             link_libs: Optional[List[str]] = None) -> bool:
        with tempfile.TemporaryDirectory(prefix="megg_executable_") as directory:
            source = Path(directory) / "input.ll"
            source.write_text(llvm_ir)
            arguments = [f"-{optimization_level}", "-o", output_path, str(source)]
            arguments.extend(f"-l{library}" for library in link_libs or [])
            self._run("clang", *arguments)
        return True

    def process(self, module, output_format: str = "llvm-ir",
                output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        if output_format not in {"llvm-ir", "object", "executable"}:
            raise ValueError(f"Unknown output format: {output_format}")
        if output_format != "llvm-ir" and not output_path:
            raise ValueError(f"output_path required for {output_format} generation")
        llvm_ir = self.emit_llvm_ir(module, output_path if output_format == "llvm-ir" else None)
        if output_format == "llvm-ir":
            return llvm_ir
        options = {"optimization_level": kwargs.get("optimization_level", "O2")}
        if output_format == "object":
            self.compile_to_object(llvm_ir, output_path, target=kwargs.get("target"), **options)
        else:
            self.compile_to_executable(llvm_ir, output_path, link_libs=kwargs.get("link_libs"), **options)
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    args = parser.parse_args()
    llvm_ir = MLIRToLLVMBackend().emit_llvm_ir(MModule.parse_from_file(args.input), args.output)
    if args.output is None:
        print(llvm_ir)
