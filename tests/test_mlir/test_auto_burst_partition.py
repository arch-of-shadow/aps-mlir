"""Explicit logical bank hints must retain the physical DMA lane count."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('element,count,requested,expected',
                         [('f32', 24, 1, 2), ('i16', 24, 2, 4),
                          ('i8', 32, 2, 8), ('f32', 24, 4, 4)])
def test_dma_bank_floor(tmp_path, element, count, requested, expected):
    source = f'''module {{
      memref.global @buf:memref<{count}x{element}> = uninitialized {{
        partition_dim_array=[0:i32],partition_factor_array=[{requested}:i32],
        partition_cyclic_array=[1:i32]}}
      func.func @copy(%addr:i32) {{
        %buf=memref.get_global @buf:memref<{count}x{element}>
        %zero=arith.constant 0:i32
        %count=arith.constant {count}:i32
        aps.copy %addr,(%buf)[%zero],%count:i32,(memref<{count}x{element}>),i32,i32
        return
      }}
    }}'''
    path = tmp_path/'input.mlir'
    path.write_text(source)
    result = subprocess.run([str(ROOT/'build/tools/aps-opt/aps-opt'),
        '-p=builtin.module(auto-burst-partition)', str(path)],
        text=True, capture_output=True, check=True, cwd=ROOT)
    assert f'partition_factor_array = [{expected} : i32]' in result.stdout


def test_unsupported_dma_bank_width_fails_synthesis(tmp_path):
    source = '''module {func.func @load(%a:memref<24xi8>)->i32 attributes {
      au.generated=true, partition_dim_array_0=[0:i64],
      partition_factor_array_0=[3:i64],partition_cyclic_array_0=[true]} {
      %zero=arith.constant 0:index
      %v=memref.load %a[%zero]:memref<24xi8>
      %r=arith.extui %v:i8 to i32
      return %r:i32
    }}'''
    path, hardware, rtl = tmp_path/'input.mlir', tmp_path/'hardware.mlir', tmp_path/'design.sv'
    path.write_text(source)
    subprocess.run([str(ROOT/'build/tools/aps-scaleisa-opt/scaleisa-adapter-opt'),
        str(path), '--scaleisa-convert', '--scaleisa-split=max-extra-csrs=5',
        '--scaleisa-materialize-aps', '-o', str(hardware)],
        check=True, capture_output=True, text=True, cwd=ROOT)
    result = subprocess.run([str(ROOT/'build/tools/aps-opt/aps-e2e'),
        '-i', str(hardware), '--clock', '8', '--resource', str(ROOT/'examples/resource_ihp130.json'),
        '--export', 'sv', '-o', str(rtl)], text=True, capture_output=True, cwd=ROOT)
    assert result.returncode != 0
    assert 'DMA scratchpad banks must provide at least 64 bits per beat' in result.stderr
    assert not rtl.exists() or not rtl.stat().st_size
