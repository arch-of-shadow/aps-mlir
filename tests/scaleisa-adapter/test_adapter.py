"""Exercise the adapter through its dedicated executable and MLIR verification."""

import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOL = Path(
    os.environ.get(
        "SCALEISA_ADAPTER_OPT",
        ROOT / "build/tools/aps-scaleisa-opt/scaleisa-adapter-opt",
    )
)
MLIR_OPT = ROOT / "circt/build/bin/mlir-opt"
sys.path.insert(0, str(ROOT / "circt/build/tools/mlir/python_packages/mlir_core"))
from mlir import ir
from mlir.dialects import arith, func


def sequence_reference(text):
    """Build a test-only software reference from non-executable CI metadata."""
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.parse(text)
        if "scaleisa.sequences" not in module.operation.attributes:
            return text
        for sequence in module.operation.attributes["scaleisa.sequences"]:
            typ = ir.FunctionType(ir.TypeAttr(sequence["source_type"]).value)
            with ir.InsertionPoint(module.body):
                wrapper = func.FuncOp(ir.StringAttr(sequence["source"]).value, typ)
            block = wrapper.add_entry_block()
            results = []
            with ir.InsertionPoint(block):

                def resolve(binding):
                    if "argument" in binding:
                        return block.arguments[
                            ir.IntegerAttr(binding["argument"]).value
                        ]
                    if "step" in binding:
                        return results[ir.IntegerAttr(binding["step"]).value]
                    value = ir.IntegerAttr(binding["constant"])
                    return arith.ConstantOp(value.type, value).result

                for step in sequence["steps"]:
                    call = func.CallOp(
                        list(typ.results),
                        ir.FlatSymbolRefAttr(step["callee"]).value,
                        [resolve(binding) for binding in step["inputs"]],
                    )
                    results.append(call.results[0])
                func.ReturnOp([resolve(sequence["result"])])
        module.operation.verify()
        return str(module)


TREE = """
func.func @kernel(%a: i32, %x: i32, %w: i32, %y: i32, %z: i32) -> i32
    attributes {au.generated = true} {
  %t = arith.muli %a, %x : i32
  %u = arith.muli %t, %w : i32
  %v = arith.muli %y, %z : i32
  %r = arith.addi %u, %v : i32
  return %r : i32
}
"""

DOT = """
func.func @dot(%a: memref<8xi32>, %b: memref<8xi32>) -> i32
    attributes {au.generated = true} {
  %zero = arith.constant 0 : i32
  %r = affine.for %i = 0 to 8 iter_args(%acc = %zero) -> i32 {
    %x = affine.load %a[%i] : memref<8xi32>
    %y = affine.load %b[%i] : memref<8xi32>
    %v = arith.muli %x, %y : i32
    %s = arith.addi %acc, %v : i32
    affine.yield %s : i32
  } {pipeline = 7 : i64, unroll = 2 : i64}
  return %r : i32
}
"""


class AdapterTests(unittest.TestCase):
    def test_partition_plan_reaches_flat_scratchpad(self):
        text = '''module {func.func @banked(%a:memref<8xf32>)->f32 attributes {
          au.generated=true, partition_dim_array_0=[0:i64],
          partition_factor_array_0=[2:i64], partition_cyclic_array_0=[true]} {
          %c0=arith.constant 0:index
          %x=memref.load %a[%c0]:memref<8xf32>
          return %x:f32
        }}'''
        passes = ('--scaleisa-convert=preserve-implementation-pragmas=true',
                  '--scaleisa-split=max-extra-csrs=5', '--scaleisa-materialize-aps')
        output = self.run_adapter(text, *passes)
        global_line = next(line for line in output.splitlines() if 'memref.global @banked_buffer' in line)
        self.assertIn('partition_factor_array = [2]', global_line)
        self.assertIn('partition_cyclic_array = [1]', global_line)
        invalid = text.replace('memref<8xf32>', 'memref<2x4xf32>').replace('%a[%c0]', '%a[%c0,%c0]')
        error = self.run_adapter(invalid, *passes, success=False)
        self.assertIn('innermost extent', error)

    def run_adapter(self, text, *passes, success=True):
        result = subprocess.run(
            [str(TOOL), *passes],
            input=text,
            text=True,
            capture_output=True,
            cwd=ROOT,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result.stdout if success else result.stderr

    def planned(self, text=TREE, csrs=1, pragma="none"):
        return self.run_adapter(
            "module {" + text + "}",
            f"--scaleisa-adapter=max-extra-csrs={csrs} pragma={pragma}",
        )

    def test_csr_limit_changes_partition(self):
        with_csrs = self.planned(csrs=1)
        without_csrs = self.planned(csrs=0)
        self.assertEqual(with_csrs.count("scaleisa.instruction"), 3)
        self.assertEqual(without_csrs.count("scaleisa.instruction"), 4)
        self.assertIn("scaleisa.sequences", with_csrs)
        self.assertNotRegex(with_csrs, r"\b(?:func\.)?call\s+@")
        self.assertTrue(
            all(
                int(n) <= 3
                for n in re.findall(r"scaleisa.operand_count = (\d+)", with_csrs)
            )
        )
        self.assertTrue(
            all(
                int(n) <= 2
                for n in re.findall(r"scaleisa.operand_count = (\d+)", without_csrs)
            )
        )

    def assert_semantics(self, candidate, expected):
        rng = random.Random(17)
        calls = []
        for index in range(12):
            values = [rng.randint(-12, 12) for _ in range(5)]
            constants = "\n".join(
                f"%c{i} = arith.constant {v} : i32" for i, v in enumerate(values)
            )
            calls.append(
                f"""
              func.func @check{index}() -> i32 {{
                {constants}
                %r = func.call @kernel(%c0, %c1, %c2, %c3, %c4)
                  : (i32, i32, i32, i32, i32) -> i32
                return %r : i32
              }}
            """
            )
        for csrs in (0, 1, 3):
            output = sequence_reference(self.planned(candidate, csrs=csrs))
            output = output.rstrip().removesuffix("}") + "\n".join(calls) + "\n}"
            result = subprocess.run(
                [str(MLIR_OPT), "--inline", "--canonicalize"],
                input=output,
                text=True,
                capture_output=True,
                cwd=ROOT,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rng.seed(17)
            for index in range(12):
                values = [rng.randint(-12, 12) for _ in range(5)]
                body = re.search(
                    rf"func.func @check{index}\(\).*?\{{(.*?)\}}",
                    result.stdout,
                    re.S,
                ).group(1)
                self.assertIn(f"arith.constant {expected(*values)} : i32", body)

    def test_split_keeps_branch_results_in_register_operands(self):
        with ir.Context():
            module = ir.Module.parse(self.planned())
            sequence = module.operation.attributes["scaleisa.sequences"][0]
            steps = sequence["steps"]
            self.assertEqual(len(steps), 3)
            # CI0(a,x,w), CI1(y,z), CI2(result0,result1).
            self.assertEqual(
                [
                    [
                        ir.IntegerAttr(binding["argument"]).value
                        for binding in step["inputs"]
                    ]
                    for step in list(steps)[:2]
                ],
                [[0, 1, 2], [3, 4]],
            )
            self.assertEqual(
                [
                    ir.IntegerAttr(binding["step"]).value
                    for binding in steps[2]["inputs"]
                ],
                [0, 1],
            )
            self.assertEqual(ir.IntegerAttr(sequence["result"]["step"]).value, 2)

    def test_split_rejects_existing_callers_instead_of_leaving_dangling_calls(self):
        caller = """func.func @caller(%a: i32) -> i32 {
          %r = func.call @kernel(%a, %a, %a, %a, %a)
            : (i32, i32, i32, i32, i32) -> i32
          return %r : i32
        }"""
        error = self.run_adapter(
            "module {" + TREE + caller + "}",
            "--scaleisa-adapter=max-extra-csrs=1 pragma=none",
            success=False,
        )
        self.assertIn("extract the candidate first", error)

    def test_tree_semantics_after_splitting(self):
        self.assert_semantics(TREE, lambda a, x, w, y, z: a * x * w + y * z)

    def test_shared_dag_semantics_and_constants(self):
        dag = TREE.replace(
            "%v = arith.muli %y, %z : i32",
            "%v = arith.muli %t, %z : i32",
        ).replace(
            "return %r : i32",
            "%c = arith.constant 7 : i32\n"
            "%q = arith.addi %r, %c : i32\nreturn %q : i32",
        )
        self.assert_semantics(dag, lambda a, x, w, y, z: a * x * w + a * x * z + 7)

    def test_conversion_precedes_pragmas(self):
        converted = self.run_adapter("module {" + DOT + "}", "--scaleisa-convert")
        self.assertIn("aps.read_smem", converted)
        self.assertNotIn("affine.load", converted)
        self.assertNotIn("pipeline =", converted)
        self.assertNotIn("unroll =", converted)
        planned = self.planned(DOT, pragma="pipeline")
        self.assertIn("pipeline = 1 : i64", planned)
        self.assertIn("scaleisa.instruction", planned)

    def test_correctness_defaults_do_not_insert_pragmas(self):
        output = self.run_adapter(
            "module {" + DOT + "}", "--scaleisa-adapter=max-extra-csrs=5"
        )
        for attr in ("pipeline =", "unroll =", "II ="):
            self.assertNotIn(attr, output)
        self.assertIn("scaleisa.max_extra_csrs = 5 : i32", output)

    def test_selected_loop_choices_survive_conversion(self):
        source = DOT.replace("pipeline = 7", "ref_starttime = 3 : i64, pipeline = 7")
        output = self.run_adapter(
            "module {" + source + "}",
            "--scaleisa-convert=preserve-implementation-pragmas=true",
        )
        self.assertIn("pipeline = 1 : i64", output)
        self.assertIn("II = 7 : i64", output)
        self.assertIn("unroll = 2 : i64", output)
        self.assertNotIn("ref_starttime", output)

    def test_automatic_pipeline_is_enabled_for_aps(self):
        source = DOT.replace('pipeline = 7', 'pipeline = 0')
        output = self.run_adapter('module {' + source + '}',
            '--scaleisa-convert=preserve-implementation-pragmas=true')
        self.assertIn('pipeline = 1 : i64', output)
        self.assertNotIn('II = 0', output)

    def test_old_schedule_costs_are_invalidated(self):
        source = TREE.replace(
            "au.generated = true",
            "au.generated = true, schedule = true, "
            "latency_cycles = 5 : i64, area_square_microns = 12.0 : f64",
        )
        output = self.planned(source)
        self.assertNotIn("latency_cycles", output)
        self.assertNotIn("area_square_microns", output)

    def test_non_candidates_are_untouched(self):
        output = self.run_adapter(
            "module {" + TREE.replace("au.generated = true", "ordinary") + "}",
            "--scaleisa-adapter",
        )
        self.assertNotIn("scaleisa.instruction", output)
        self.assertNotIn("func.call", output)

    def test_static_unroll_admission_and_size_limit(self):
        planned = self.planned(DOT, pragma="unroll")
        self.assertIn("unroll = 0 : i64", planned)
        rejected = self.run_adapter(
            self.planned(DOT),
            "--scaleisa-insert-pragmas=mode=unroll max-expanded-ops=1",
        )
        self.assertIn("scaleisa.pragma_rejected", rejected)
        self.assertNotIn("unroll =", rejected)
        dynamic = (
            DOT.replace("memref<8xi32>", "memref<?xi32>")
            .replace(
                "%zero =",
                "%n = memref.dim %a, %c0 : memref<?xi32>\n%zero =",
            )
            .replace("%n =", "%c0 = arith.constant 0 : index\n%n =")
            .replace(
                "0 to 8",
                "0 to %n",
            )
        )
        self.assertIn(
            "static affine or SCF trip count", self.planned(dynamic, pragma="unroll")
        )

    def test_constant_scf_unroll_reaches_the_native_backend(self):
        source = '''func.func @fill(%out:memref<10xi32>,%v:i32)
          attributes {au.generated=true} {
          %lo=arith.constant -3:index
          %hi=arith.constant 6:index
          %step=arith.constant 2:index
          %offset=arith.constant 3:index
          scf.for %i=%lo to %hi step %step {
            %at=arith.addi %i,%offset:index
            memref.store %v,%out[%at]:memref<10xi32>
          }
          return
        }'''
        planned = self.planned(source, pragma='unroll')
        self.assertNotIn('scaleisa.pragma_rejected', planned)
        self.assertIn('unroll = 0 : i64', planned)
        result = subprocess.run([str(ROOT/'build/tools/aps-opt/aps-opt'),
            '-p=builtin.module(hls-unroll,canonicalize)'], input=planned,
            text=True, capture_output=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('scf.for', result.stdout)
        self.assertEqual(result.stdout.count('aps.write_smem '), 5)

    def test_linalg_buffer_conversion(self):
        source = """module {
          func.func @matmul(%a: memref<4x4xf32>, %b: memref<4x4xf32>,
                           %c: memref<4x4xf32>) attributes {au.generated = true} {
            linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
                          outs(%c : memref<4x4xf32>)
            return
          }
        }"""
        output = self.run_adapter(
            source, "--scaleisa-adapter=max-extra-csrs=1 pragma=pipeline"
        )
        self.assertNotIn("linalg.matmul", output)
        self.assertIn("aps.read_smem", output)
        self.assertIn("aps.write_smem", output)
        self.assertEqual(output.count("pipeline ="), 1)

    def test_oversized_memory_graph_rejected(self):
        source = DOT.replace("%b: memref<8xi32>", "%b: memref<8xi32>, %extra: i32")
        error = self.run_adapter(
            "module {" + source + "}", "--scaleisa-adapter", success=False
        )
        self.assertIn("cannot split memory effects", error)

    def test_scalar_hardware_export(self):
        output = self.run_adapter(self.planned(), "--scaleisa-materialize-aps")
        self.assertEqual(output.count("scaleisa.hardware"), 3)
        self.assertEqual(output.count("aps.read_csr"), 1)
        self.assertEqual(output.count("aps.write_irf"), 3)
        self.assertIn("csr_address = 2049", output)
        self.assertNotIn("csr_address = 2050", output)
        self.assertIn("funct7 = 0", output)
        self.assertIn("funct7 = 1", output)
        self.assertNotIn("func.call", output)
        self.assertNotIn("scaleisa.orchestration", output)

    def test_split_helpers_lower_through_aps_to_systemverilog(self):
        # Verify the hardware boundary, rather than merely verifying the
        # arithmetic planning IR and its non-executable sequencing metadata.
        output = self.run_adapter(self.planned(), "--scaleisa-materialize-aps")
        with tempfile.TemporaryDirectory(prefix="scaleisa-split-") as tmp:
            directory = Path(tmp)
            source = directory / "split.aps.mlir"
            scheduled = directory / "split.tor.mlir"
            rtl = directory / "split.sv"
            source.write_text(output)
            result = subprocess.run(
                [
                    str(ROOT / "build/tools/aps-opt/aps-e2e"),
                    "--clock=4.0",
                    "-i",
                    str(source),
                    "-o",
                    str(rtl),
                    "--export=sv",
                    f"--dump-scheduled-tor={scheduled}",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            tor = scheduled.read_text()
            self.assertEqual(tor.count("tor.func @"), 3)
            self.assertNotIn("func.func", tor)
            self.assertNotRegex(tor, r"\b(?:func\.)?call\s+@kernel")
            self.assertIn("module main(", rtl.read_text())

    def test_float_and_narrow_integer_transport(self):
        for typ, op in (("f32", "arith.addf"), ("i8", "arith.addi")):
            source = f"""func.func @sum(%a: {typ}, %b: {typ}) -> {typ}
              attributes {{au.generated = true}} {{
                %r = {op} %a, %b : {typ}
                return %r : {typ}
              }}"""
            output = self.run_adapter(
                self.planned(source), "--scaleisa-materialize-aps"
            )
            self.assertIn("arith.bitcast" if typ == "f32" else "arith.trunci", output)

    def test_invalid_exports_and_pass_order(self):
        for option in ("opcode=3", "funct7-base=127", "csr-base=3072"):
            self.run_adapter(
                self.planned(), f"--scaleisa-materialize-aps={option}", success=False
            )
        wide = TREE.replace("i32", "i64")
        self.run_adapter(
            self.planned(wide), "--scaleisa-materialize-aps", success=False
        )
        error = self.run_adapter(
            "module {" + TREE + "}", "--scaleisa-split", success=False
        )
        self.assertIn("run scaleisa-convert", error)
        error = self.run_adapter(
            "module {" + TREE + "}",
            "--scaleisa-insert-pragmas",
            success=False,
        )
        self.assertIn("run scaleisa-split", error)

    def test_export_reserves_function_and_csr_names(self):
        # One planned instruction's function name deliberately collides with
        # another instruction's preferred CSR global name.
        small = """func.func @f(%a: i32, %b: i32, %c: i32) -> i32
            attributes {au.generated = true} {
          %v = arith.addi %a, %b : i32
          %r = arith.addi %v, %c : i32
          return %r : i32
        }"""
        output = self.run_adapter(
            self.planned(small + small.replace("@f(", "@f_csr0(")),
            "--scaleisa-materialize-aps",
        )
        self.assertIn("func.func @f_csr0(", output)
        self.assertEqual(output.count("aps.read_csr"), 2)

    def eval_memory_export(self, output, carriers, memory):
        """Execute the emitted straight-line integer APS body against CPU memory."""
        values = {}
        buffers = {}
        element_bytes = {}
        global_buffers = {}
        global_bytes = {}
        for name, count, width in re.findall(
            r"memref.global @(\w+) : memref<(\d+)xi(\d+)>", output
        ):
            global_buffers[name] = [None] * int(count)
            global_bytes[name] = int(width) // 8
        result = None
        csr_symbols = dict(
            re.findall(
                r'memref.global "private" @([\w]+).*csr_address = (\d+)',
                output,
            )
        )
        for line in output.splitlines():
            line = line.strip()
            if match := re.fullmatch(
                r"(%\w+) = aps.read_irf %arg(\d+) : i5 -> i32", line
            ):
                values[match[1]] = carriers[int(match[2])]
            elif match := re.fullmatch(r"(%\w+) = aps.read_csr @(\w+) : i32", line):
                values[match[1]] = carriers[int(csr_symbols[match[2]]) - 2049 + 2]
            elif match := re.fullmatch(
                r"(%\w+) = arith.constant (-?\d+) : (?:i32|index)", line
            ):
                values[match[1]] = int(match[2])
            elif match := re.fullmatch(
                r"(%\w+) = arith.(?:index_cast(?:ui)?|extui|extsi|trunci) (%\w+) : .*",
                line,
            ):
                values[match[1]] = values[match[2]]
            elif match := re.fullmatch(
                r"(%\w+) = arith.(addi|muli) (%\w+), (%\w+) : i32", line
            ):
                lhs, rhs = values[match[3]], values[match[4]]
                values[match[1]] = (
                    lhs + rhs if match[2] == "addi" else lhs * rhs
                ) & 0xFFFFFFFF
            elif match := re.fullmatch(r"(%\w+) = aps.load (%\w+) : i32 -> i\d+", line):
                values[match[1]] = memory[values[match[2]]]
            elif match := re.fullmatch(r"aps.store (%\w+), (%\w+) : i\d+, i32", line):
                memory[values[match[2]]] = values[match[1]]
            elif match := re.fullmatch(r"aps.write_irf %arg2, (%\w+) : i5, i32", line):
                result = values[match[1]]
            elif match := re.fullmatch(
                r"(%\w+) = memref.get_global @(\w+) : memref<\d+xi\d+>", line
            ):
                buffers[match[1]] = global_buffers[match[2]]
                element_bytes[match[1]] = global_bytes[match[2]]
            elif match := re.fullmatch(
                r"aps.copy (%\w+), \((%\w+)\)\[(%\w+)\], (%\w+) : .*", line
            ):
                address, buffer, start, length = match.groups()
                for i in range(values[length]):
                    buffers[buffer][values[start] + i] = memory[
                        values[address] + i * element_bytes[buffer]
                    ]
            elif match := re.fullmatch(
                r"aps.copy \((%\w+)\)\[(%\w+)\], (%\w+), (%\w+) : .*", line
            ):
                buffer, start, address, length = match.groups()
                for i in range(values[length]):
                    memory[values[address] + i * element_bytes[buffer]] = buffers[
                        buffer
                    ][values[start] + i]
            elif match := re.fullmatch(
                r"(%\w+) = aps.read_smem (%\w+)\[(%\w+)\] : .*", line
            ):
                values[match[1]] = buffers[match[2]][values[match[3]]]
            elif match := re.fullmatch(
                r"aps.write_smem (%\w+), (%\w+)\[(%\w+)\] : .*", line
            ):
                buffers[match[2]][values[match[3]]] = values[match[1]]
            elif line.startswith("%") or line.startswith("aps."):
                self.fail(f"Unexpected operation in memory test: {line}")
        return result

    def test_external_arrays_are_copied_into_scratchpads(self):
        output = self.run_adapter(self.planned(DOT), "--scaleisa-materialize-aps")
        body = output.split("func.func @", 1)[1].splitlines()[1:]
        self.assertTrue(all("aps.read_irf" in line for line in body[:2]))
        self.assertEqual(output.count("aps.copy "), 2)
        self.assertEqual(output.count("scaleisa.scratchpad"), 2)
        self.assertEqual(output.count("memref.get_global"), 2)
        for name in re.findall(r"memref.global @(\w+) : memref<8xi32>", output):
            self.assertIn(f'var_name = "{name}"', output)
        self.assertNotIn("memref.alloca", output)
        self.assertNotIn("memref.alloc ", output)
        self.assertEqual(output.count("aps.read_smem"), 2)
        self.assertNotIn("aps.load ", output)
        self.assertNotIn("aps.store ", output)
        self.assertNotIn("aps.copy (", output)  # Read-only inputs need no writeback.
        self.assertNotIn("unrealized_conversion_cast", output)
        self.assertIn('value_kind = "cpu_aligned_base"', output)
        for physical in ("aps.memitfc", "aps.load_by", "aps.copy_by", "aps.load_issue"):
            self.assertNotIn(physical, output)

    def test_buffered_addresses_include_static_offset_and_flatten_indices(self):
        source = """func.func @load(%a: memref<4x8xi32, strided<[8, 1], offset: 3>>,
                                   %i: index, %j: index) -> i32
            attributes {au.generated = true} {
          %v = memref.load %a[%i, %j] : memref<4x8xi32, strided<[8, 1], offset: 3>>
          return %v : i32
        }"""
        output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
        for i, j in ((0, 0), (1, 2), (3, 7)):
            memory = {0x1000 + 4 * (3 + n): n + 97 for n in range(32)}
            self.assertEqual(
                self.eval_memory_export(output, [0x1000, i, j], memory), 97 + 8 * i + j
            )

    def test_writable_aliases_need_explicit_disjointness_contract(self):
        source = """func.func @alias(%a: memref<4xi32>, %b: memref<4xi32>, %v: i32) -> i32
            attributes {au.generated = true} {
          %zero = arith.constant 0 : index
          memref.store %v, %a[%zero] : memref<4xi32>
          %r = memref.load %b[%zero] : memref<4xi32>
          return %r : i32
        }"""
        error = self.run_adapter(
            self.planned(source), "--scaleisa-materialize-aps", success=False
        )
        self.assertIn("scaleisa.noalias", error)
        source = source.replace(
            "%a: memref<4xi32>", "%a: memref<4xi32> {scaleisa.noalias}"
        )
        output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
        memory = {0x2000 + 4 * i: i for i in range(4)}
        memory.update({0x3000 + 4 * i: 5 + i for i in range(4)})
        self.assertEqual(
            self.eval_memory_export(output, [0x2000, 0x3000, 91], memory), 5
        )
        self.assertEqual(memory[0x2000], 91)
        self.assertEqual(memory[0x2004], 1)
        self.assertIn("noalias", output)
        self.assertEqual(output.count("aps.copy "), 3)
        self.assertLess(output.index("aps.copy ("), output.index("aps.write_irf"))

    def test_subword_cpu_address_scaling(self):
        for width, size in ((8, 1), (16, 2), (32, 4)):
            source = f"""func.func @load(%a: memref<8xi{width}>, %i: index) -> i{width}
                attributes {{au.generated = true}} {{
              %v = memref.load %a[%i] : memref<8xi{width}>
              return %v : i{width}
            }}"""
            output = self.run_adapter(
                self.planned(source), "--scaleisa-materialize-aps"
            )
            self.assertEqual(
                self.eval_memory_export(
                    output, [0x3000, 3], {0x3000 + size * i: 14 + i for i in range(8)}
                ),
                17,
            )

    def test_dynamic_descriptor_and_view_uses_rejected(self):
        source = """func.func @dim(%a: memref<?xi32>) -> index
            attributes {au.generated = true} {
          %zero = arith.constant 0 : index
          %n = memref.dim %a, %zero : memref<?xi32>
          return %n : index
        }"""
        error = self.run_adapter(
            self.planned(source), "--scaleisa-materialize-aps", success=False
        )
        self.assertIn("dynamic descriptor fields", error)
        view = """func.func @view(%a: memref<8xi32>) -> i32
            attributes {au.generated = true} {
          %b = memref.subview %a[1] [4] [1]
            : memref<8xi32> to memref<4xi32, strided<[1], offset: 1>>
          %zero = arith.constant 0 : index
          %r = memref.load %b[%zero] : memref<4xi32, strided<[1], offset: 1>>
          return %r : i32
        }"""
        self.run_adapter(
            self.planned(view), "--scaleisa-materialize-aps", success=False
        )

    def test_static_dim_is_materialized_without_descriptor(self):
        source = """func.func @dim(%a: memref<8xi32>) -> index
            attributes {au.generated = true} {
          %zero = arith.constant 0 : index
          %n = memref.dim %a, %zero : memref<8xi32>
          return %n : index
        }"""
        output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
        self.assertNotIn("memref.dim", output)
        self.assertEqual(self.eval_memory_export(output, [0x4000], {}), 8)

    def test_existing_internal_buffers_are_preserved(self):
        source = """func.func @buffer(%a: memref<4xi32>) -> i32
            attributes {au.generated = true} {
          %tmp = memref.alloca() : memref<4xi32>
          %zero = arith.constant 0 : index
          %v = memref.load %a[%zero] : memref<4xi32>
          memref.store %v, %tmp[%zero] : memref<4xi32>
          %r = memref.load %tmp[%zero] : memref<4xi32>
          return %r : i32
        }"""
        output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
        self.assertIn("aps.copy ", output)
        self.assertNotIn("aps.load ", output)
        self.assertIn("aps.write_smem", output)
        self.assertIn("aps.read_smem", output)
        self.assertNotIn("unrealized_conversion_cast", output)

    def test_complete_overwrite_omits_input_copy(self):
        affine = """func.func @clear(%a: memref<2x4xi32>) attributes {au.generated = true} {
          %v = arith.constant 7 : i32
          affine.for %i = 0 to 2 {
            affine.for %j = 0 to 4 {
              affine.store %v, %a[%i, %j] : memref<2x4xi32>
            }
          }
          return
        }"""
        scf = """func.func @clear(%a: memref<2x4xi32>) attributes {au.generated = true} {
          %v = arith.constant 7 : i32
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c4 = arith.constant 4 : index
          scf.for %i = %c0 to %c2 step %c1 {
            scf.for %j = %c0 to %c4 step %c1 {
              memref.store %v, %a[%i, %j] : memref<2x4xi32>
            }
          }
          return
        }"""
        for source in (affine, scf):
            output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
            self.assertEqual(output.count("aps.copy "), 1)
            self.assertIn("aps.copy (", output)
        for source in (affine.replace("%j = 0 to 4", "%j = 0 to 3"),
                       affine.replace("%j = 0 to 4", "%j = 0 to 4 step 2"),
                       affine.replace("[%i, %j]", "[%i, %i]"),
                       affine.replace("%a: memref<2x4xi32>)", "%a: memref<2x4xi32>, %n: index)").replace(
                           "affine.for %i", "%c0 = arith.constant 0 : index\n"
                           "          %c1 = arith.constant 1 : index\n"
                           "          scf.for %extra = %c0 to %n step %c1 { affine.for %i", 1).replace(
                           "          return", "          }\n          return", 1),
                       scf.replace("memref.store %v", "scf.if %flag { memref.store %v").replace(
                           " : memref<2x4xi32>\n", " : memref<2x4xi32> }\n").replace(
                           "%a: memref<2x4xi32>)", "%a: memref<2x4xi32>, %flag: i1)")):
            output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
            self.assertEqual(output.count("aps.copy "), 2)

    def test_partial_output_preserves_contents_and_unrounded_length(self):
        source = """func.func @update(%a: memref<5xi32>, %v: i32) -> i32
            attributes {au.generated = true} {
          %two = arith.constant 2 : index
          memref.store %v, %a[%two] : memref<5xi32>
          %r = memref.load %a[%two] : memref<5xi32>
          return %r : i32
        }"""
        output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
        self.assertIn("memref<5xi32>", output)
        self.assertEqual(output.count("aps.copy "), 2)
        lengths = re.findall(r"aps.copy .*?, (%\w+) :", output)
        for length in lengths:
            self.assertRegex(output, re.escape(length) + r" = arith.constant 4 : i32")
        self.assertEqual(output.count("aps.load "), 1)
        self.assertEqual(output.count("aps.store "), 1)
        for forbidden in (
            "transfer_bytes",
            "segment_count",
            "aps.copy_by",
        ):
            self.assertNotIn(forbidden, output)
        memory = {0x5000 + 4 * i: i + 10 for i in range(5)}
        self.assertEqual(self.eval_memory_export(output, [0x5000, 77], memory), 77)
        self.assertEqual(list(memory.values()), [10, 11, 77, 13, 14])

    def test_globals_are_discovered_by_existing_aps_memory_map(self):
        tool = ROOT / "build/tools/aps-opt/aps-opt"
        if not tool.exists():
            self.skipTest("APS optimizer is not built")
        output = self.run_adapter(self.planned(DOT), "--scaleisa-materialize-aps")
        result = subprocess.run(
            [str(tool), "--aps-memory-map"],
            input=output,
            text=True,
            capture_output=True,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("aps.memorymap", result.stdout)
        for name in re.findall(r"memref.global @(\w+) : memref<8xi32>", output):
            self.assertIn(f'aps.mem_entry "{name}"', result.stdout)
        self.assertNotIn("aps.memitfc", result.stdout)

    def test_scalar_local_buffer_uses_aps_global_register_ops(self):
        source = """func.func @acc(%v: i32) -> i32
            attributes {au.generated = true} {
          %tmp = memref.alloca() : memref<i32>
          memref.store %v, %tmp[] : memref<i32>
          %r = memref.load %tmp[] : memref<i32>
          return %r : i32
        }"""
        output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
        self.assertIn("memref.global @acc_local : memref<i32>", output)
        self.assertIn("aps.globalstore", output)
        self.assertIn("aps.globalload", output)
        self.assertNotIn("memref.alloca", output)

    def test_rank_zero_scalar_reference_is_buffered_once(self):
        source = """func.func @scalar(%a: memref<i32>) -> i32
            attributes {au.generated = true} {
          %v = memref.load %a[] : memref<i32>
          return %v : i32
        }"""
        output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
        self.assertIn("aps.read_smem ", output)
        self.assertEqual(output.count("aps.load "), 1)
        self.assertEqual(output.count("aps.copy "), 0)
        self.assertEqual(self.eval_memory_export(output, [0x6000], {0x6000: 41}), 41)

    def test_rank_zero_update_preserves_alias_contract_and_writes_back(self):
        source = """func.func @scalar(%a: memref<i32>) -> i32
            attributes {au.generated = true} {
          %v = memref.load %a[] : memref<i32>
          %one = arith.constant 1 : i32
          %r = arith.addi %v, %one : i32
          memref.store %r, %a[] : memref<i32>
          return %r : i32
        }"""
        output = self.run_adapter(self.planned(source), "--scaleisa-materialize-aps")
        self.assertEqual(output.count("aps.load "), 1)
        self.assertEqual(output.count("aps.store "), 1)
        self.assertEqual(output.count("aps.copy "), 0)
        self.assertEqual(self.eval_memory_export(output, [0x6000], {0x6000: 41}), 42)

    def test_noncontiguous_arrays_require_layout_legalization(self):
        source = """func.func @strided(%a: memref<4xi32, strided<[2]>>) -> i32
            attributes {au.generated = true} {
          %zero = arith.constant 0 : index
          %v = memref.load %a[%zero] : memref<4xi32, strided<[2]>>
          return %v : i32
        }"""
        error = self.run_adapter(
            self.planned(source), "--scaleisa-materialize-aps", success=False
        )
        self.assertIn("contiguous row-major", error)


if __name__ == "__main__":
    unittest.main()
