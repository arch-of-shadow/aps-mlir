"""Adversarial binding tests, with executable host models of selected ISAXs.

The host model replaces only emitted inline assembly, using its actual CSR
addresses and operand placeholders. The remaining LLVM program runs natively.
This tests software binding semantics, not the implementation of CSR hardware.
"""
import ctypes
import itertools
import json
import random
import re
import runpy
import shutil
import subprocess
from pathlib import Path

import egglog
import pytest

from cadl_frontend.to_c import CTranspiler
from megg.backend.llvm_backend import rtype_asm
from megg.compiler import Compiler
from megg.egraph.megg_egraph import MeggEClass, MeggEGraph, MeggENode, SkeletonMatcher
from megg.egraph.term import LitTerm, Term
from megg.egraph.terms_to_func import ExprTreeToMLIR
from megg.rewrites.internal_rewrites import constant_folding_laws
from megg.rewrites.match_rewrites import build_ruleset_from_module
from megg.utils.mlir_utils import MModule

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "circt/build/bin"
MASK = 0xFFFFFFFF

def run(*command):
    result = subprocess.run([str(item) for item in command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout

def compile_pair(tmp_path, cadl_source, program_source):

    cli = runpy.run_path(str(ROOT / "tools/aps-megg"))
    cadl = tmp_path / "pattern.cadl"
    cadl.write_text(cadl_source)
    pattern_c = tmp_path / "pattern.c"
    pattern_c.write_text(CTranspiler().transpile(cadl))
    program_c = tmp_path / "program.c"
    program_c.write_text("#include <stdint.h>\n" + program_source)
    modules = []
    for source in (pattern_c, program_c):
        output = source.with_suffix(".mlir")
        run(ROOT / "thirdparty/polygeist/cgeist", "-S", "-O3", source, "-o", output)
        cli["strip_module_attrs"](output)
        modules.append(MModule.parse_from_file(output))
    encoding_path = tmp_path / "encoding.json"
    cli["extract_encoding"](cadl, encoding_path)
    encodings = json.loads(encoding_path.read_text())
    rules, skeletons = build_ruleset_from_module(modules[0], instr_encodings=encodings)
    result = Compiler(modules[1], match_ruleset=rules, skeletons=skeletons,
                      instr_encodings=encodings).schedule(external_passes=False)
    assert result.success, result.error_message
    mlir = result.optimized_module.to_string()
    (tmp_path / "optimized.mlir").write_text(mlir)
    assert "ordinary" in [func.symbol_name for func in result.optimized_module.get_functions()], "Reconstruction silently dropped the function"
    return mlir

def lower(tmp_path, mlir):
    source = tmp_path / "optimized.mlir"
    lowered = tmp_path / "lowered.mlir"
    source.write_text(mlir)
    run(BIN / "mlir-opt", "--test-lower-to-llvm", source, "-o", lowered)
    return run(BIN / "mlir-translate", "--mlir-to-llvmir", lowered)

def host_scalar_model(tmp_path, llvm, argc):
    """Execute x * CSR[0x801] - CSR[0x802] with the emitted operand map."""
    helpers = []

    def replace(match):
        asm, constraints = match.groups()
        assert constraints.startswith("=r,"), constraints
        inputs = constraints.split(",").count("r")

        def value(placeholder):
            if placeholder == "x0":
                return "0"
            index = int(placeholder.removeprefix("$")) - 1
            assert 0 <= index < inputs
            return f"%p{index}"

        writes = re.findall(r"csrrw x0, (0x[0-9a-f]+), (\$\d+)", asm)
        assert len(writes) == 2, asm
        csrs = {int(address, 16): value(operand) for address, operand in writes}
        assert set(csrs) == {0x801, 0x802}
        insn = re.search(r"\.insn r [^,]+, [^,]+, [^,]+, \$0, (\$\d+|x0), (\$\d+|x0)", asm)
        assert insn, asm
        # All test instructions consume rs1, except the rs2-only variant.
        x = value(insn.group(1) if insn.group(1) != "x0" else insn.group(2))
        name = f"csr_model_{len(helpers)}"
        params = ", ".join(f"i32 %p{i}" for i in range(inputs))
        helpers.append(f"""
define private i32 @{name}({params}) {{
  %product = mul i32 {x}, {csrs[0x801]}
  %result = sub i32 %product, {csrs[0x802]}
  ret i32 %result
}}
""")
        return f"@{name}"

    llvm = re.sub(r'asm sideeffect "([^"]+)", "([^"]+)"', replace, llvm)
    assert helpers, "No ISAX was selected"
    source = tmp_path / "host_model.ll"
    library = tmp_path / "host_model.so"
    source.write_text(llvm + "\n".join(helpers))
    run(shutil.which("clang"), "-shared", "-fPIC", "-Wno-override-module", source, "-o", library)
    dll = ctypes.CDLL(str(library))
    function = dll.ordinary
    function.argtypes = [ctypes.c_uint32] * argc
    function.restype = ctypes.c_uint32
    return function, len(helpers)

def scalar_pattern(register="rs1"):
    return f"""
#[csr_address(0x801)]
register gain: u32;
#[csr_address(0x802)]
register bias: u32;
#[opcode(7'b0101011)]
#[funct7(7'b0011110)]
rtype adjust(rs1: u5, rs2: u5, rd: u5) {{
    _irf[rd] = _irf[{register}] * _csr[gain] - _csr[bias];
}}
"""

@pytest.mark.parametrize("register", ["rs1", "rs2"])
def test_permuted_computed_scalar_inputs_execute_correctly(tmp_path, register):
    mlir = compile_pair(tmp_path, scalar_pattern(register), """
uint32_t ordinary(uint32_t bias, uint32_t x, uint32_t gain) {
    return x * ((gain << 1) + 3) - (bias ^ 0x80000000u);
}
""")
    function, count = host_scalar_model(tmp_path, lower(tmp_path, mlir), 3)
    # Algebraic expansion may also offload gain * 2 + 3. Check the actual
    # numerical result of every selected instruction through the host model.
    assert count >= 1
    rng = random.Random(0xC5A)
    boundaries = [0, 1, 0x7FFFFFFF, 0x80000000, MASK]
    samples = list(itertools.product(boundaries, repeat=3))
    samples += [tuple(rng.getrandbits(32) for _ in range(3)) for _ in range(300)]
    for bias, x, gain in samples:
        expected = (x * (((gain << 1) + 3) & MASK) - (bias ^ 0x80000000)) & MASK
        assert function(bias, x, gain) == expected, (bias, x, gain)

def test_constant_csr_inputs_preserve_high_bits(tmp_path):
    mlir = compile_pair(tmp_path, scalar_pattern(), """
uint32_t ordinary(uint32_t x, uint32_t bias) {
    return x * 0x80000001u - bias;
}
""")
    function, count = host_scalar_model(tmp_path, lower(tmp_path, mlir), 2)
    assert count == 1
    rng = random.Random(801)
    for x in [0, 1, 0x7FFFFFFF, 0x80000000, MASK] + [rng.getrandbits(32) for _ in range(300)]:
        bias = rng.getrandbits(32)
        assert function(x, bias) == (x * 0x80000001 - bias) & MASK

def test_folded_negative_csr_bias_still_offloads(tmp_path):
    # Polygeist changes subtraction of UINT32_MAX into addition of 1.
    # This is equivalent modulo 2^32 and should remain matchable.
    mlir = compile_pair(tmp_path, scalar_pattern(), """
uint32_t ordinary(uint32_t x) {
    return x * 0x80000001u - 0xffffffffu;
}
""")
    function, count = host_scalar_model(tmp_path, lower(tmp_path, mlir), 1)
    assert count == 1
    for x in [0, 1, 0x7FFFFFFF, 0x80000000, MASK]:
        assert function(x) == (x * 0x80000001 - MASK) & MASK

def test_repeated_invocations_keep_distinct_csr_values(tmp_path):
    mlir = compile_pair(tmp_path, scalar_pattern(), """
uint32_t ordinary(uint32_t x, uint32_t y, uint32_t g1, uint32_t b1,
                  uint32_t g2, uint32_t b2) {
    uint32_t first = x * (g1 + 1) - b1;
    uint32_t second = y * (g2 + 3) - b2;
    return first ^ second;
}
""")
    function, count = host_scalar_model(tmp_path, lower(tmp_path, mlir), 6)
    assert count == 2, "Both independent matches must survive extraction"
    rng = random.Random(802)
    for _ in range(400):
        x, y, g1, b1, g2, b2 = [rng.getrandbits(32) for _ in range(6)]
        expected = ((x * (g1 + 1) - b1) ^ (y * (g2 + 3) - b2)) & MASK
        assert function(x, y, g1, b1, g2, b2) == expected

def memory_pattern():
    return """
#[csr_address(0x801)]
register gain: u32;
static input: [u32; 8];
static output: [u32; 8];
#[opcode(7'b0101011)]
#[funct7(7'b0011110)]
rtype paired(rs1: u5, rs2: u5) {
    let src: u32 = _irf[rs1];
    let dst: u32 = _irf[rs2];
    let g: u32 = _csr[gain];
    input[0 +: ] = _mem[src +: 8];
    with i: u32 = (0, i_) do {
        output[i] = input[i] * g;
        output[i + 4] = input[i + 4] * g + 1;
        let i_: u32 = i + 1;
    } while (i_ < 4);
    _mem[dst +: 8] = output[0 +: ];
}
"""

@pytest.mark.parametrize("same_gain", [True, False])
def test_multiple_components_require_consistent_csr_binding(tmp_path, same_gain):
    expression = "(g1 + 3)" if same_gain else "(g2 + 5)"
    mlir = compile_pair(tmp_path, memory_pattern(), f"""
void ordinary(uint32_t g1, uint32_t *dst, uint32_t g2, uint32_t *src) {{
    for (uint32_t i = 0; i < 4; ++i) {{
        dst[i] = src[i] * (g1 + 3);
        dst[i + 4] = src[i + 4] * {expression} + 1;
    }}
}}
""")
    if same_gain:
        assert mlir.count("csrrw x0, 0x801") == 1
        assert "scf.for" not in mlir
    else:
        assert "csrrw" not in mlir, "Different component gains must prevent offload"
        assert "scf.for" in mlir
    lower(tmp_path, mlir)  # Also require legal IR in the rejected-match path.

def test_iteration_dependent_csr_value_must_not_offload(tmp_path):
    # One CSR setup before the instruction cannot represent a distinct gain
    # at each iteration of the loop that the instruction replaces.
    mlir = compile_pair(tmp_path, memory_pattern(), """
void ordinary(uint32_t gain, uint32_t *src, uint32_t *dst) {
    for (uint32_t i = 0; i < 4; ++i) {
        dst[i] = src[i] * (gain + i);
        dst[i + 4] = src[i + 4] * (gain + i) + 1;
    }
}
""")
    assert "csrrw" not in mlir, "Loop-varying values cannot be hoisted into CSR setup"
    assert "scf.for" in mlir
    lower(tmp_path, mlir)

def test_outer_iteration_value_can_configure_an_inner_loop(tmp_path):
    mlir = compile_pair(tmp_path, memory_pattern(), """
void ordinary(uint32_t gain, uint32_t *src, uint32_t *dst) {
    for (uint32_t j = 0; j < 3; ++j) {
        for (uint32_t i = 0; i < 4; ++i) {
            dst[i] = src[i] * (gain + j);
            dst[i + 4] = src[i + 4] * (gain + j) + 1;
        }
    }
}
""")
    assert mlir.count("csrrw x0, 0x801") == 1
    assert mlir.count("scf.for") == 1  # The outer loop must remain.
    lower(tmp_path, mlir)

def test_large_operand_map_assembles_with_exact_csr_addresses(tmp_path):
    # Cross the arg9/arg10 and $9/$10 boundaries, and scramble marker order.
    order = [11, 1, 8, 3, 0, 10, 4, 9, 2, 7, 5, 6]
    csrs = [{"arg_index": i, "address": 0x800 + i} for i in range(2, 12)]
    enc = {"opcode": "0x2b", "funct3": "0x7", "funct7": "0x1e",
           "operand_indices": order, "csr_inputs": csrs}
    asm = rtype_asm(enc, len(order), True)
    # Instantiate distinct physical registers, then inspect actual machine code.
    for i in reversed(range(13)):
        asm = asm.replace(f"${i}", f"x{i + 5}")
    source = tmp_path / "many.s"
    obj = tmp_path / "many.o"
    source.write_text(asm + "\n")
    run(BIN / "llvm-mc", "-triple=riscv32", "-mattr=+zicsr", "-filetype=obj", source, "-o", obj)
    binary = tmp_path / "many.bin"
    run(BIN / "llvm-objcopy", "-O", "binary", "--only-section=.text", obj, binary)
    words = [int.from_bytes(binary.read_bytes()[i:i + 4], "little") for i in range(0, 44, 4)]
    for csr, word in zip(csrs, words[:10]):
        assert word & 0x7F == 0x73
        assert (word >> 12) & 7 == 1  # CSRRW, not an alias with different semantics.
        assert (word >> 7) & 31 == 0  # Suppress the CSR read result.
        assert word >> 20 == csr["address"]
        assert (word >> 15) & 31 == order.index(csr["arg_index"]) + 6
    insn = words[-1]
    assert (insn >> 7) & 31 == 5
    assert (insn >> 15) & 31 == order.index(0) + 6
    assert (insn >> 20) & 31 == order.index(1) + 6

@pytest.mark.parametrize('has_result', [False, True])
@pytest.mark.parametrize('funct3', [0, 3, 7, '0x7'])
@pytest.mark.parametrize('num_inputs', [0, 1, 2, 3])
def test_xd_machine_bit_follows_readback(tmp_path, has_result, funct3, num_inputs):
    enc = {'opcode': '0x2b', 'funct7': '0x1e', 'funct3': funct3}
    if num_inputs == 3:
        enc['csr_inputs'] = [{'arg_index': 2, 'address': 0x801}]
    asm = rtype_asm(enc, num_inputs, has_result)
    for i in reversed(range(num_inputs + int(has_result))):
        asm = asm.replace(f'${i}', f'x{i + 5}')
    source = tmp_path / 'xd.s'
    obj = tmp_path / 'xd.o'
    binary = tmp_path / 'xd.bin'
    source.write_text('.text\n' + asm + '\n')
    run(BIN / 'llvm-mc', '-triple=riscv32', '-mattr=+zicsr', '-filetype=obj', source, '-o', obj)
    run(BIN / 'llvm-objcopy', '-O', 'binary', '--only-section=.text', obj, binary)
    word = int.from_bytes(binary.read_bytes()[-4:], 'little')
    assert word & 0x7f == 0x2b
    assert bool(word & (1 << 14)) == has_result
    assert (word >> 7) & 31 == (5 if has_result else 0)
    original = int(funct3, 0) if isinstance(funct3, str) else funct3
    assert (word >> 12) & 3 == original & 3

@pytest.mark.parametrize("overrides", [
    {"operand_indices": [0, 0]},
    {"operand_indices": [0]},
    {"csr_inputs": [{"arg_index": 0, "address": 0x801}]},
    {"csr_inputs": [{"arg_index": 2, "address": -1}]},
    {"csr_inputs": [{"arg_index": 2, "address": 0x1000}]},
    {"csr_inputs": [{"arg_index": 2, "address": 0xC01}]},
    {"csr_inputs": [{"arg_index": 2, "address": 0x801}, {"arg_index": 3, "address": 0x801}]},
    {"csr_inputs": [{"arg_index": 2, "address": 0x801}, {"arg_index": 2, "address": 0x802}]},
])
def test_corrupt_encoding_metadata_rejected(overrides):
    with pytest.raises(ValueError):
        rtype_asm({"opcode": "0x2b", **overrides}, 2, True)

@pytest.mark.parametrize("width", [8, 16, 32])
@pytest.mark.parametrize("kind", ["one", "minimum", "maximum"])
def test_inverse_folded_subtraction_wraps_at_operand_width(width, kind):

    value = {"one": 1, "minimum": -(1 << (width - 1)),
             "maximum": (1 << (width - 1)) - 1}[kind]
    negative = -value & ((1 << width) - 1)
    if negative >= 1 << (width - 1):
        negative -= 1 << width
    ty = egglog.String(f"i{width}")
    lhs = Term.arg(egglog.i64(0), ty)
    addition = Term.add(lhs, Term.lit(LitTerm.int(egglog.i64(value)), ty), ty)
    subtraction = Term.sub(lhs, Term.lit(LitTerm.int(egglog.i64(negative)), ty), ty)
    graph = egglog.EGraph()
    graph.register(addition)
    graph.run(constant_folding_laws().saturate())
    assert graph.check_bool(egglog.eq(addition).to(subtraction))

def test_region_availability_keeps_outer_binders_and_rejects_inner_binders():

    # An inner-loop offload may use the outer induction variable, but cannot
    # use the inner induction variable or a loop-carried value it removes.
    nodes = [
        MeggENode("outer", "Term.loop_index", [], "outer", "arg", 10),
        MeggENode("inner", "Term.loop_index", [], "inner", "arg", 20),
        MeggENode("carry", "Term.loop_iter_arg", [], "carry", "arg", (20, 0)),
        MeggENode("condition_arg", "Term.block_arg", [], "condition_arg", "arg", (30, 0)),
        MeggENode("lit", "Term.lit", [], "lit", "lit", 1),
        MeggENode("outer_expr", "Term.add", ["outer", "lit"], "outer_expr", "impl"),
        MeggENode("inner_expr", "Term.add", ["inner", "lit"], "inner_expr", "impl"),
        MeggENode("vec", "Vec", [], "vec", "impl"),
        MeggENode("block", "Term.block", ["id", "vec"], "block", "impl", 30),
        MeggENode("loop", "Term.for_with_carry", ["lit", "lit", "lit", "inner", "vec", "block"], "loop", "impl"),
    ]
    graph = MeggEGraph({n.node_id: n for n in nodes},
                      {n.eclass: MeggEClass(n.eclass, [n]) for n in nodes}, [])
    matcher = SkeletonMatcher(graph)
    candidate = nodes[-1]
    assert matcher._inputs_available_before_candidate(candidate, ["outer_expr"])
    for forbidden in ("inner_expr", "carry", "condition_arg"):
        assert not matcher._inputs_available_before_candidate(candidate, [forbidden])

def test_reconstruction_failure_is_a_failed_compilation(monkeypatch):

    module = MModule.parse("""
module {
  func.func @ordinary(%x: i32, %y: i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    func.return %sum : i32
  }
}
""")
    compiler = Compiler(module)

    def fail(*args, **kwargs):
        raise ValueError("injected invalid CSR operands")

    monkeypatch.setattr(ExprTreeToMLIR, "reconstruct", fail)
    result = compiler.schedule(external_passes=False, custom_rewrites=False)
    assert not result.success
    assert "ordinary" in result.error_message
    assert "injected invalid CSR operands" in result.error_message
    assert result.optimized_module is module
