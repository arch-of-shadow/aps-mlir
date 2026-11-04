"""Failures must remain failures throughout MEGG's compilation pipeline."""
import gc
import subprocess
from types import SimpleNamespace

import pytest

from megg.backend.llvm_backend import MLIRToLLVMBackend
from megg.compiler import Compiler
from megg.e2e_compiler import E2ECompiler
from megg.egraph.extract import Extractor
from megg.egraph.func_to_terms import (
    FuncToTerms,
    megg_to_mlir_type,
    mlir_type_to_megg_type,
)
from megg.rewrites.external_loop_pass import _apply_mlir_pass_to_loop
from megg.utils.ir_builder import IRBuilder
from megg.utils.mlir_utils import MLIRPassManager, MModule


def test_missing_riscv_compiler_is_reported(monkeypatch):
    monkeypatch.delenv('RISCV', raising=False)
    monkeypatch.setattr('megg.e2e_compiler.shutil.which', lambda _: None)
    with pytest.raises(FileNotFoundError, match='Set RISCV'):
        E2ECompiler.find_riscv(None)

def test_extraction_does_not_silently_drop_children():
    extractor = object.__new__(Extractor)
    extractor.costs = {}
    with pytest.raises(ValueError, match='Child e-class missing'):
        extractor.build_recexpr('root', SimpleNamespace(children=['missing']))

def test_backend_missing_tool_does_not_fabricate_llvm(tmp_path, monkeypatch):
    monkeypatch.setattr('megg.backend.llvm_backend.shutil.which', lambda name: None)
    output = tmp_path / 'output.ll'
    with pytest.raises(FileNotFoundError, match='mlir-opt'):
        MLIRToLLVMBackend().emit_llvm_ir(MModule.parse('module {}'), str(output))
    assert not output.exists()

def test_backend_lowering_failure_does_not_try_other_paths(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('megg.backend.llvm_backend.shutil.which', lambda name: name)

    def fail(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, '', 'invalid source operation')

    monkeypatch.setattr('megg.backend.llvm_backend.subprocess.run', fail)
    with pytest.raises(RuntimeError, match='invalid source operation'):
        MLIRToLLVMBackend().emit_llvm_ir(MModule.parse('module {}'), str(tmp_path / 'output.ll'))
    assert len(calls) == 1
    assert calls[0][0] == 'mlir-opt'
    assert not (tmp_path / 'output.ll').exists()

@pytest.mark.parametrize('from_file', [False, True])
def test_invalid_mlir_is_rejected(tmp_path, from_file):
    source = 'module { func.func @broken('
    with pytest.raises(ValueError, match='Invalid MLIR'):
        if from_file:
            path = tmp_path / 'broken.mlir'
            path.write_text(source)
            MModule.parse_from_file(path)
        else:
            MModule.parse(source)

def test_unsupported_types_are_not_guessed():
    with pytest.raises(ValueError, match='Unsupported MLIR type'):
        mlir_type_to_megg_type('!unknown.type')

def test_dynamic_memref_type_is_preserved():
    dtype = mlir_type_to_megg_type('memref<?x4xf32>')
    assert dtype.shape == (-1, 4)
    assert str(dtype) == 'memref<?x4xf32>'
    module = MModule.parse('module {}')
    with IRBuilder().set_context(module.get_context()):
        assert str(megg_to_mlir_type(dtype)) == 'memref<?x4xf32>'

def test_malformed_constant_is_not_replaced_with_zero():
    operation = SimpleNamespace(has_attr=lambda _: False, results=[])
    with pytest.raises(ValueError, match='Malformed constant'):
        FuncToTerms._constant_to_term(None, operation)

def test_extraction_preserves_unselected_functions_and_globals(monkeypatch):
    def unexpected_visualization(*args, **kwargs):
        raise AssertionError('Compilation must not invoke Graphviz')

    monkeypatch.setattr(Compiler, 'visualize_egraph', unexpected_visualization)
    source = MModule.parse('''
module {
  memref.global "private" @data : memref<4xi32>
  func.func @selected(%x: i32, %y: i32) -> i32 {
    %sum = arith.addi %x, %y : i32
    func.return %sum : i32
  }
  func.func @untouched(%x: i32) -> i32 {
    func.return %x : i32
  }
}
''')
    result = Compiler(source, target_functions=['selected']).schedule(
        external_passes=False, custom_rewrites=False)
    assert result.success, result.error_message
    assert {f.symbol_name for f in result.optimized_module.get_functions()} == {'selected', 'untouched'}
    assert result.optimized_module.to_string().count('memref.global') == 1
    assert MLIRPassManager(result.optimized_module).verify_module()

def loop_module():
    return MModule.parse('''
module {
  func.func @copy(%src: memref<4xi32>, %dst: memref<4xi32>) {
    %zero = arith.constant 0 : index
    %four = arith.constant 4 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %four step %one {
      %v = memref.load %src[%i] : memref<4xi32>
      memref.store %v, %dst[%i] : memref<4xi32>
    }
    func.return
  }
}
''')

def test_external_unroll_uses_clone_mapping_and_valid_lowering():
    source = loop_module()
    function = source.get_functions()[0]
    loop = next(op for op in source.get_operations() if op.name == 'scf.for')
    transformed, _ = _apply_mlir_pass_to_loop(function, loop, 0, 'loop-unroll', 2, 4)
    assert MLIRPassManager(transformed).verify_module()
    assert source.to_string().count('memref.store') == 1
    assert transformed.to_string().count('memref.store') == 2

def test_compiler_external_rewrites_preserve_valid_module():
    source = loop_module()
    hints = {'copy': {'lower': 0, 'upper': 2, 'step': 1,
                      'nest_level': 0, 'nested_loop_count': 0}}
    result = Compiler(source, loop_hints=hints).schedule(custom_rewrites=False)
    assert result.success, result.error_message
    assert result.external_rewrites > 0
    assert MLIRPassManager(result.optimized_module).verify_module()
    assert source.to_string().count('memref.store') == 1

def test_external_pass_failure_does_not_return_untransformed_code(monkeypatch):
    source = loop_module()
    loop = next(op for op in source.get_operations() if op.name == 'scf.for')
    monkeypatch.setattr(MLIRPassManager, 'run_pass', lambda self, name: False)
    with pytest.raises(RuntimeError, match='loop-unroll failed'):
        _apply_mlir_pass_to_loop(source.get_functions()[0], loop, 0, 'loop-unroll', 2, 4)

def test_cloned_module_keeps_original_context_alive():
    source = loop_module()
    cloned, _ = source.clone_with_mapping()
    del source
    gc.collect()
    assert MLIRPassManager(cloned).verify_module()
    assert 'memref.store' in cloned.to_string()

@pytest.mark.parametrize('has_result', [False, True])
def test_custom_instruction_is_emitted_as_llvm_inline_asm(has_result):
    builder = IRBuilder()
    module = MModule.parse('module {}')
    with builder.set_context(module.get_context()):
        result_type = builder.i32() if has_result else None
        function, block = builder.create_function(
            'emit', [builder.i32(), builder.i32()], [result_type] if has_result else [])
        with builder.set_insertion_point_to_end(block):
            result = MLIRToLLVMBackend.emit_custom_instruction(
                builder, block.arguments, result_type, 'custom',
                {'encoding': {'opcode': '0x2b', 'funct3': '0x7', 'funct7': '0x31'}})
            builder.func_return([result] if has_result else [])
        module.append_to_module(function)
    text = module.to_string()
    assert 'llvm.inline_asm has_side_effects' in text
    assert f'.insn r 0x2b, {"0x7" if has_result else "0x3"}, 0x31' in text
    assert '~{memory}' in text
    assert '-> i1' not in text
    assert MLIRPassManager(module).verify_module()
