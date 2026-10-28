"""Singleton promotion must keep shared conditional storage in one protocol."""
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_shared_conditional_global_preserves_all_memory_accesses(tmp_path):
    source = '''module {
      memref.global "private" @shared : memref<1xi32>
      memref.global "private" @plain : memref<1xi32>
      func.func @conditional(%condition:i1,%value:i32) {
        %m=memref.get_global @shared:memref<1xi32>
        %zero=arith.constant 0:i32
        aps.write_smem_if %condition,%value,%m[%zero]:i32,memref<1xi32>,i32
        return
      }
      func.func @reader()->i32 {
        %m=memref.get_global @shared:memref<1xi32>
        %zero=arith.constant 0:i32
        %v=aps.read_smem %m[%zero]:memref<1xi32>,i32 -> i32
        return %v:i32
      }
      func.func @writer(%value:i32) {
        %m=memref.get_global @shared:memref<1xi32>
        %zero=arith.constant 0:i32
        aps.write_smem %value,%m[%zero]:i32,memref<1xi32>,i32
        return
      }
      func.func @unconditional(%value:i32) {
        %m=memref.get_global @plain:memref<1xi32>
        %zero=arith.constant 0:i32
        aps.write_smem %value,%m[%zero]:i32,memref<1xi32>,i32
        return
      }
    }'''
    path = tmp_path/'input.mlir'
    path.write_text(source)
    result = subprocess.run([
        str(ROOT/'build/tools/aps-opt/aps-opt'),
        '-p=builtin.module(promote-singleton-memref-to-global)', str(path)],
        text=True, capture_output=True, check=True, cwd=ROOT)
    assert 'aps.write_smem_if' in result.stdout
    assert 'aps.read_smem' in result.stdout
    assert 'aps.write_smem ' in result.stdout
    assert 'aps.globalload' not in result.stdout
    assert result.stdout.count('aps.globalstore') == 1
    assert '@plain' in result.stdout.split('aps.globalstore')[1]


def test_pipelined_singleton_bank_preserves_reads_and_writes(tmp_path):
    source = '''module {
      memref.global @bank : memref<1xi32> = uninitialized
      func.func @kernel(%value:i32) {
        %m=memref.get_global @bank:memref<1xi32>
        %zero=arith.constant 0:i32
        affine.for %i=0 to 8 {
          aps.write_smem %value,%m[%zero]:i32,memref<1xi32>,i32
        } {pipeline=0:i64}
        %v=aps.read_smem %m[%zero]:memref<1xi32>,i32 -> i32
        return
      }
    }'''
    path = tmp_path/'input.mlir'
    path.write_text(source)
    result = subprocess.run([
        str(ROOT/'build/tools/aps-opt/aps-opt'),
        '-p=builtin.module(promote-singleton-memref-to-global)', str(path)],
        text=True, capture_output=True, check=True, cwd=ROOT)
    assert 'aps.write_smem ' in result.stdout
    assert 'aps.read_smem ' in result.stdout
    assert 'aps.globalstore' not in result.stdout
    assert 'aps.globalload' not in result.stdout
