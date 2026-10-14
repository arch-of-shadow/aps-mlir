"""CSR transport must preserve the original rs1/rs2 argument positions."""
import json
import runpy
import subprocess
from pathlib import Path

import pytest

from cadl_frontend.to_c import CTranspiler
from megg.backend.llvm_backend import rtype_asm
from megg.compiler import Compiler
from megg.rewrites.match_rewrites import build_ruleset_from_module
from megg.utils.mlir_utils import MModule


def encoding(**extra):
    return {"opcode": "0x2b", "funct3": "0x7", "funct7": "0x1e", **extra}

def test_csr_setup_uses_compacted_operand_positions():
    asm = rtype_asm(encoding(
        operand_indices=[0, 2, 3],
        csr_inputs=[
            {"arg_index": 2, "address": 0x801},
            {"arg_index": 3, "address": 0x802},
        ],
    ), 3, True)
    assert asm == (
        "csrrw x0, 0x801, $2\n"
        "csrrw x0, 0x802, $3\n"
        ".insn r 0x2b, 0x7, 0x1e, $0, $1, x0"
    )

def test_void_memory_instruction_with_csr():
    asm = rtype_asm(encoding(csr_inputs=[{"arg_index": 2, "address": 0x801}]), 3, False)
    assert asm == "csrrw x0, 0x801, $2\n.insn r 0x2b, 0x3, 0x1e, x0, $0, $1"

def test_unused_csr_does_not_require_a_bound_operand():
    asm = rtype_asm(encoding(
        operand_indices=[1], csr_inputs=[{"arg_index": 2, "address": 0x801}],
    ), 1, True)
    assert asm == ".insn r 0x2b, 0x7, 0x1e, $0, x0, $1"

@pytest.mark.parametrize("num_inputs, has_result, registers", [
    (0, False, "x0, x0, x0"),
    (1, True, "$0, $1, x0"),
    (2, True, "$0, $1, $2"),
])
def test_rtype_without_csr(num_inputs, has_result, registers):
    assert rtype_asm(encoding(), num_inputs, has_result).endswith(registers)

def test_unbound_extra_inputs_rejected():
    with pytest.raises(ValueError, match="CSR bindings"):
        rtype_asm(encoding(), 3, True)

def test_extract_cadl_encoding_includes_csr_bindings(tmp_path):
    cli = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/aps-megg"))
    cadl = tmp_path / "adjust.cadl"
    cadl.write_text("""
#[csr_address(0x801)]
register cfg: u32;
#[opcode(7'b0101011)]
#[funct7(7'b0011110)]
rtype adjust(rs1: u5, rd: u5) {
    _irf[rd] = _irf[rs1] + _csr[cfg];
}
""")
    output = tmp_path / "encoding.json"
    cli['extract_encoding'](cadl, output)
    assert json.loads(output.read_text()) == {"adjust": {
        "opcode": "0x2B", "funct7": "0x1E",
        "csr_inputs": [{"name": "cfg", "address": 0x801, "arg_index": 2}],
    }}

def test_csr_matching_and_reconstruction(tmp_path):

    pattern = MModule.parse("""
module {
  func.func @adjust(%rs1: i32, %rs2: i32, %cfg: i32) -> i32 {
    %0 = arith.muli %rs1, %cfg : i32
    func.return %0 : i32
  }
}
""")
    program = MModule.parse("""
module {
  func.func @ordinary_c(%x: i32, %gain: i32) -> i32 {
    %0 = arith.muli %x, %gain : i32
    func.return %0 : i32
  }
}
""")
    encodings = {"adjust": encoding(csr_inputs=[{"arg_index": 2, "address": 0x801}])}
    rules, skeletons = build_ruleset_from_module(pattern, instr_encodings=encodings)
    assert encodings["adjust"]["operand_indices"] == [0, 2]
    compiler = Compiler(program, match_ruleset=rules, skeletons=skeletons, instr_encodings=encodings)
    result = compiler.schedule(external_passes=False)
    assert result.success, result.error_message
    mlir = result.optimized_module.to_string()
    assert "csrrw x0, 0x801, $2" in mlir
    assert ".insn r 0x2b, 0x7, 0x1e, $0, $1, x0" in mlir
    assert '"=r,r,r,~{memory}"' in mlir
    assert "has_side_effects" in mlir

    # Exercise the real lowering and assembler: a valid MLIR template alone
    # does not guarantee a legal R-type instruction in the target object.
    root = Path(__file__).resolve().parents[1]
    binaries = root / "circt/build/bin"
    optimized = tmp_path / "optimized.mlir"
    lowered = tmp_path / "lowered.mlir"
    llvm = tmp_path / "target.ll"
    obj = tmp_path / "target.o"
    optimized.write_text(mlir)
    subprocess.run([str(binaries / "mlir-opt"), "--test-lower-to-llvm", str(optimized), "-o", str(lowered)], check=True, capture_output=True, text=True)
    translation = subprocess.run([str(binaries / "mlir-translate"), "--mlir-to-llvmir", str(lowered)], check=True, capture_output=True, text=True)
    llvm.write_text(translation.stdout)
    subprocess.run([str(binaries / "llc"), "-mtriple=riscv32-unknown-elf", "-mattr=+m,+zicsr", "-target-abi=ilp32", "-filetype=obj", str(llvm), "-o", str(obj)], check=True, capture_output=True, text=True)
    disassembly = subprocess.run([str(binaries / "llvm-objdump"), "-d", "-M", "no-aliases", str(obj)], check=True, capture_output=True, text=True)
    assert "csrrw" in disassembly.stdout
    assert "0x801" in disassembly.stdout or "2049" in disassembly.stdout

@pytest.mark.parametrize("constant_gain", [False, True])
def test_csr_memory_loop_from_cadl_and_ordinary_c(tmp_path, constant_gain):

    root = Path(__file__).resolve().parents[1]
    cli = runpy.run_path(str(root / "tools/aps-megg"))
    cadl = tmp_path / "scale.cadl"
    cadl.write_text("""
#[csr_address(0x801)]
register gain: u32;
static input: [u32; 4];
static output: [u32; 4];
#[opcode(7'b0101011)]
#[funct7(7'b0011110)]
rtype scale(rs1: u5, rs2: u5) {
    let src: u32 = _irf[rs1];
    let dst: u32 = _irf[rs2];
    let g: u32 = _csr[gain];
    input[0 +: ] = _mem[src +: 4];
    with i: u32 = (0, i_) do {
        output[i] = input[i] * g;
        let i_: u32 = i + 1;
    } while (i_ < 4);
    _mem[dst +: 4] = output[0 +: ];
}
""")
    pattern_c = tmp_path / "pattern.c"
    pattern_c.write_text(CTranspiler().transpile(cadl))
    program_c = tmp_path / "program.c"
    program_source = """
#include <stdint.h>
void ordinary_c(uint32_t gain, uint32_t *src, uint32_t *dst) {
    for (uint32_t i = 0; i < 4; ++i) dst[i] = src[i] * (gain + 1);
}
"""
    if constant_gain:
        program_source = program_source.replace("uint32_t gain, ", "").replace("(gain + 1)", "7")
    program_c.write_text(program_source)
    modules = []
    for source in (pattern_c, program_c):
        mlir = source.with_suffix('.mlir')
        subprocess.run([str(root / 'thirdparty/polygeist/cgeist'), '-S', '-O3', str(source), '-o', str(mlir)], check=True, capture_output=True, text=True)
        cli['strip_module_attrs'](mlir)
        modules.append(MModule.parse_from_file(mlir))
    encoding_json = tmp_path / "scale.json"
    cli['extract_encoding'](cadl, encoding_json)
    encodings = json.loads(encoding_json.read_text())
    rules, skeletons = build_ruleset_from_module(modules[0], instr_encodings=encodings)
    assert skeletons
    compiler = Compiler(modules[1], match_ruleset=rules, skeletons=skeletons, instr_encodings=encodings)
    result = compiler.schedule(external_passes=False)
    assert result.success, result.error_message
    mlir = result.optimized_module.to_string()
    assert "csrrw x0, 0x801, $2" in mlir
    assert ".insn r 0x2B, 0x3, 0x1E, x0, $0, $1" in mlir
    if not constant_gain:
        assert "arith.addi" in mlir  # CSR setup receives the matched gain + 1.
    assert '"r,r,r,~{memory}"' in mlir
