"""Verify exposed CI lowering, real RV32 encodings, and split execution."""

import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "build/tools/aps-scaleisa-opt/scaleisa-adapter-opt"
COMPILER = ROOT / "build/tools/aps-scaleisa-opt/scaleisa-compiler-opt"
DRIVER = ROOT / "build/tools/aps-scaleisa-opt/aps-scaleisa-opt"

TREE = """func.func @expr(%a: i32, %x: i32, %w: i32, %y: i32, %z: i32) -> i32
  attributes {au.generated = true} {
  %t = arith.muli %a, %x : i32
  %u = arith.muli %t, %w : i32
  %v = arith.muli %y, %z : i32
  %r = arith.addi %u, %v : i32
  return %r : i32
}"""
PROGRAM = """module {
  func.func private @expr(i32, i32, i32, i32, i32) -> i32
  func.func @main() -> i32 {
    %a = arith.constant 2 : i32
    %x = arith.constant 3 : i32
    %w = arith.constant 4 : i32
    %y = arith.constant 5 : i32
    %z = arith.constant 6 : i32
    %r = func.call @expr(%a, %x, %w, %y, %z) : (i32, i32, i32, i32, i32) -> i32
    return %r : i32
  }
}"""


def elf_text(path):
    """Read actual linked machine words, without depending on objdump aliases."""
    data = path.read_bytes()
    if (
        data[:7] != b"\x7fELF\x01\x01\x01"
        or struct.unpack_from("<H", data, 18)[0] != 243
    ):
        raise AssertionError("Expected a little-endian RV32 ELF")
    entry, shoff = (
        struct.unpack_from("<I", data, 24)[0],
        struct.unpack_from("<I", data, 32)[0],
    )
    size, count, strings = struct.unpack_from("<HHH", data, 46)
    headers = [struct.unpack_from("<10I", data, shoff + i * size) for i in range(count)]
    names = data[headers[strings][4] : headers[strings][4] + headers[strings][5]]
    for header in headers:
        name = names[header[0] :].split(b"\0", 1)[0]
        if name == b".text":
            raw = data[header[4] : header[4] + header[5]]
            return entry, {
                header[3] + i: struct.unpack_from("<I", raw, i)[0]
                for i in range(0, len(raw), 4)
            }
    raise AssertionError("No ELF text section")


def run_scalar_binary(path, split):
    """Execute the emitted scalar smoke program with a test CI semantic hook.

    This checks decoded operand registers/CSR addresses in the final binary;
    it is not an RTL model or a general RISC-V simulator.
    """
    pc, words = elf_text(path)
    regs, csrs, seen = [0] * 32, {}, []
    for _ in range(100):
        word = words[pc]
        opcode, rd, funct3 = word & 127, (word >> 7) & 31, (word >> 12) & 7
        rs1, rs2, funct7 = (word >> 15) & 31, (word >> 20) & 31, word >> 25
        next_pc = pc + 4
        if opcode == 0x13 and funct3 == 0:
            imm = word >> 20
            imm -= 4096 if imm & 2048 else 0
            regs[rd] = (regs[rs1] + imm) & 0xFFFFFFFF
        elif opcode == 0x37:
            regs[rd] = word & 0xFFFFF000
        elif opcode == 0x73 and funct3 == 1:
            address = word >> 20
            previous = csrs.get(address, 0)
            csrs[address] = regs[rs1]
            regs[rd] = previous
        elif opcode == 0x2B:
            if funct3 != 7:
                raise AssertionError("Wrong RoCC funct3")
            a, b = regs[rs1], regs[rs2]
            if split:
                result = (a * b * csrs[0x801], a * b, a + b)[funct7]
            else:
                if funct7 != 0:
                    raise AssertionError("Wrong single-CI funct7")
                result = a * b * csrs[0x801] + csrs[0x802] * csrs[0x803]
            regs[rd] = result & 0xFFFFFFFF
            seen.append(funct7)
        elif (
            opcode == 0x67 and funct3 == 0 and rd == 0 and rs1 == 1 and word >> 20 == 0
        ):
            return regs[10], csrs, seen
        else:
            raise AssertionError(f"Unexpected smoke-program opcode: {word:08x}")
        regs[0] = 0
        pc = next_pc
    raise AssertionError("Scalar smoke program did not return")


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="scaleisa-compiler-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.hardware = self.directory / "hardware.mlir"

    def run_command(self, command, text=None, success=True):
        result = subprocess.run(
            list(map(str, command)),
            input=text,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def export(self, source=TREE, csrs=5):
        result = self.run_command(
            [
                ADAPTER,
                f"--scaleisa-adapter=max-extra-csrs={csrs} pragma=none",
                "--scaleisa-materialize-aps",
            ],
            "module {" + source + "}",
        )
        self.hardware.write_text(result.stdout)
        return result.stdout

    def lower(self, program=PROGRAM, success=True, csrs=5):
        return self.run_command(
            [
                COMPILER,
                f'--scaleisa-lower-ci=hardware-file="{self.hardware}" max-extra-csrs={csrs}',
            ],
            program,
            success,
        )

    def binary(self, program=PROGRAM, emit="elf", success=True):
        source = self.directory / "program.mlir"
        source.write_text(program)
        binary = self.directory / f"program.{emit}"
        result = self.run_command(
            [
                DRIVER,
                "--compile-only",
                source,
                "--hardware",
                self.hardware,
                "--emit",
                emit,
                "-o",
                binary,
            ],
            success=success,
        )
        return binary, result

    def test_one_ci_with_five_csr_cap_executes_from_linked_binary(self):
        self.export()
        binary, _ = self.binary()
        result, csrs, instructions = run_scalar_binary(binary, split=False)
        self.assertEqual(result, 54)
        self.assertEqual(csrs, {0x801: 4, 0x802: 5, 0x803: 6})
        self.assertEqual(instructions, [0])
        report = json.loads(
            (Path(str(binary) + ".scaleisa") / "compile-report.json").read_text()
        )
        self.assertEqual(report["ci_sites"], 1)
        self.assertEqual(report["ci_instructions"], 1)
        self.assertEqual(report["status"], "passed")

    def test_three_split_cis_execute_from_linked_binary(self):
        self.export(csrs=1)
        binary, _ = self.binary()
        result, csrs, instructions = run_scalar_binary(binary, split=True)
        self.assertEqual(result, 54)
        self.assertEqual(csrs, {0x801: 4})
        self.assertEqual(instructions, [0, 1, 2])
        text = (Path(str(binary) + ".scaleisa") / "ci.mlir").read_text()
        self.assertEqual(text.count("llvm.inline_asm"), 3)
        self.assertNotIn("func.call", text)
        self.assertNotIn("scaleisa.sequences", text)

    def test_native_selected_site_identity(self):
        self.export()
        result = self.lower(PROGRAM.replace("@expr", "@expr_site_123"))
        self.assertIn(".insn r 43, 7, 0", result.stdout)
        self.assertNotIn("expr_site_123", result.stdout)

    def test_explicit_binding_for_renamed_helper(self):
        self.export()
        program = PROGRAM.replace("@expr", "@renamed").replace(
            "-> i32\n  func.func @main",
            "-> i32 attributes {scaleisa.ci = @expr}\n  func.func @main",
        )
        self.assertIn("llvm.inline_asm", self.lower(program).stdout)

    def test_explicit_call_binding_for_marked_helper(self):
        self.export()
        program = (
            PROGRAM.replace("@expr", "@renamed")
            .replace(
                "-> i32\n  func.func @main",
                "-> i32 attributes {au.generated = true}\n  func.func @main",
            )
            .replace(
                ": (i32, i32, i32, i32, i32) -> i32",
                "{scaleisa.ci = @expr} : (i32, i32, i32, i32, i32) -> i32",
            )
        )
        output = self.lower(program).stdout
        self.assertIn("llvm.inline_asm", output)
        self.assertNotIn("@renamed", output)

    def test_memref_aligned_pointer_and_f32_bits(self):
        source = """func.func @dot(%a: memref<4xf32>, %i: index, %b: memref<4xf32>, %j: index) -> f32 attributes {au.generated = true} {
          %x = memref.load %a[%i] : memref<4xf32>
          %y = memref.load %b[%j] : memref<4xf32>
          %r = arith.mulf %x, %y : f32
          return %r : f32
        }"""
        self.export(source)
        program = """module {
          func.func private @dot_site_9(memref<4xf32>, index, memref<4xf32>, index) -> f32
          func.func @compute(%a: memref<4xf32>, %i: index, %b: memref<4xf32>, %j: index) -> f32 {
            %r = func.call @dot_site_9(%a, %i, %b, %j) : (memref<4xf32>, index, memref<4xf32>, index) -> f32
            return %r : f32
          }
        }"""
        result = self.lower(program).stdout
        self.assertEqual(result.count("memref.extract_aligned_pointer_as_index"), 2)
        self.assertIn("arith.bitcast", result)
        self.assertIn("fence rw, rw", result)
        binary, _ = self.binary(program, emit="object")
        self.assertTrue(binary.is_file())

    def test_dma_alignment_of_owned_buffers_through_software_helper(self):
        self.export("""func.func @read(%a: memref<32xi32>) -> i32 attributes {au.generated = true} {
          %c0 = arith.constant 0 : index
          %x = memref.load %a[%c0] : memref<32xi32>
          return %x : i32
        }""")
        for existing, expected in ((4, 64), (128, 128)):
            program = """module {
              func.func private @read_site_1(memref<32xi32>) -> i32
              func.func @forward(%a: memref<32xi32>) -> i32 {
                %r = func.call @read_site_1(%a) : (memref<32xi32>) -> i32
                return %r : i32
              }
              func.func @main() -> i32 {
                %a = memref.alloca() {alignment = EXISTING : i64} : memref<32xi32>
                %r = func.call @forward(%a) : (memref<32xi32>) -> i32
                return %r : i32
              }
            }""".replace("EXISTING", str(existing))
            result = self.lower(program).stdout
            self.assertIn(f"alignment = {expected} : i64", result)
            binary, _ = self.binary(program, emit="object")
            self.assertTrue(binary.is_file())
            llvm_ir = (Path(str(binary)+".scaleisa")/"program.ll").read_text()
            self.assertRegex(llvm_ir, rf"alloca .*align {expected}\b")

    def test_f32_and_narrow_scalar_transport(self):
        for typ, op in (("f32", "arith.addf"), ("i8", "arith.addi")):
            self.export(
                f"""func.func @sum(%a: {typ}, %b: {typ}) -> {typ} attributes {{au.generated = true}} {{
              %r = {op} %a, %b : {typ}
              return %r : {typ}
            }}"""
            )
            program = f"""module {{
              func.func private @sum({typ}, {typ}) -> {typ}
              func.func @compute(%a: {typ}, %b: {typ}) -> {typ} {{
                %r = func.call @sum(%a, %b) : ({typ}, {typ}) -> {typ}
                return %r : {typ}
              }}
            }}"""
            output = self.lower(program).stdout
            self.assertIn("arith.bitcast" if typ == "f32" else "arith.extui", output)
            self.binary(program, emit="object")

    def test_void_and_zero_operand_side_effects_survive_codegen(self):
        self.export("func.func @tick() attributes {au.generated = true} { return }")
        program = """module {
          func.func private @tick()
          func.func @main() -> i32 {
            func.call @tick() : () -> ()
            func.call @tick() : () -> ()
            %zero = arith.constant 0 : i32
            return %zero : i32
          }
        }"""
        binary, _ = self.binary(program)
        _, words = elf_text(binary)
        custom = [word for word in words.values() if word & 127 == 0x2B]
        self.assertEqual(len(custom), 2)
        self.assertTrue(all((word >> 7) == 0 for word in custom))

    def test_reject_signature_mismatch(self):
        self.export()
        self.assertIn(
            "signature", self.lower(PROGRAM.replace("i32", "i64"), success=False).stderr
        )

    def test_reject_unbound_selected_helper(self):
        self.export()
        program = """module {
          func.func @missing(%a: i32) -> i32 attributes {au.generated = true} { return %a : i32 }
        }"""
        self.assertIn("no matching hardware", self.lower(program, success=False).stderr)

    def test_reject_readonly_csr_and_csr_limit(self):
        hardware = self.export()
        self.assertIn("max-extra-csrs", self.lower(success=False, csrs=2).stderr)
        self.hardware.write_text(hardware.replace("address = 2049", "address = 3072"))
        self.assertIn("writable", self.lower(success=False).stderr)

    def test_reject_sequence_forward_reference(self):
        hardware = self.export(csrs=1)
        self.hardware.write_text(
            hardware.replace(
                "inputs = [{argument = 0 : i32}", "inputs = [{step = 2 : i32}"
            )
        )
        self.assertIn("forward reference", self.lower(success=False).stderr)

    def test_reject_indirect_helper_use(self):
        self.export()
        program = PROGRAM.replace(
            "%a = arith.constant",
            "%fn = func.constant @expr : (i32, i32, i32, i32, i32) -> i32\n    %a = arith.constant",
        )
        self.assertIn("non-call use", self.lower(program, success=False).stderr)

    def test_no_silent_success_without_ci(self):
        self.export()
        program = "module { func.func @main() -> i32 { %z = arith.constant 0 : i32 return %z : i32 } }"
        binary, result = self.binary(program, success=False)
        self.assertFalse(binary.exists())
        self.assertIn("No exposed CI sites", result.stderr)
        report = json.loads(
            (Path(str(binary) + ".scaleisa") / "compile-report.json").read_text()
        )
        self.assertEqual(report["status"], "failed")

    def test_hardware_path_with_spaces(self):
        self.hardware = self.directory / "hardware with spaces.mlir"
        self.export()
        self.binary()


if __name__ == "__main__":
    unittest.main()
