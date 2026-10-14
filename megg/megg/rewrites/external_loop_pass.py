"""Apply loop transformations to cloned modules and retain equivalent terms."""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import egglog

from megg.egraph.func_to_terms import FuncToTerms
from megg.utils import MLIRPassManager, MModule, MOperation


def apply_loop_pass_by_name(func: MOperation, loop_index: int, pass_name: str,
                            func_to_terms: FuncToTerms, egraph,
                            unroll_factor: int = 4, tile_size: int = 4):
    loop, _ = func_to_terms.loop_to_term[loop_index]
    result = _apply_mlir_pass_to_loop(
        func, loop, loop_index, pass_name, unroll_factor, tile_size)
    if result is None:
        return None
    module, function = result
    transformer = FuncToTerms.transform(function, egraph)
    egraph.register(egglog.union(func_to_terms.top_block).with_(transformer.top_block))
    return {"success": True, "module": module, "function": function,
            "transformer": transformer}, egraph

def _apply_mlir_pass_to_loop(func: MOperation, loop_op: MOperation, loop_index: int,
                             pass_name: str, unroll_factor: int,
                             tile_size: int) -> Optional[tuple]:
    """Transform a cloned loop and retain its owning module."""
    if (pass_name == 'loop-tile' and tile_size == 1) or (
        pass_name in {'loop-unroll', 'loop-unroll-jam'} and unroll_factor == 1
    ):
        return None
    module, mapping = func._module_owner.clone_with_mapping()
    function = MOperation(mapping[func._op])
    loop = MOperation(mapping[loop_op._op])
    manager = MLIRPassManager(module)
    if pass_name == 'loop-unroll':
        manager.register_affine_loop_unroll_passes(
            function=function, loop_op=loop, unroll_factor=unroll_factor)
        name = 'LoopUnroll'
    elif pass_name == 'loop-tile':
        manager.register_affine_loop_tiling_pass(
            function=function, loop_op=loop, default_tile_size=tile_size)
        name = 'LoopTiling'
    elif pass_name == 'loop-unroll-jam':
        manager.register_affine_loop_unroll_jam_passes(
            function=function, loop_op=loop, unroll_jam_factor=unroll_factor)
        name = 'LoopUnrollJam'
    else:
        raise ValueError(f"Unsupported external loop pass: {pass_name}")
    if not manager.run_pass(name) or not manager.verify_module():
        raise RuntimeError(f"{pass_name} failed or produced invalid MLIR")
    with tempfile.TemporaryDirectory(prefix='megg_loop_') as directory:
        source = Path(directory) / 'input.mlir'
        output = Path(directory) / 'lowered.mlir'
        source.write_text(module.to_string())
        result = subprocess.run(
            ['mlir-opt', '--lower-affine', '--canonicalize', str(source), '-o', str(output)],
            capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"External loop lowering failed: {result.stderr}")
        lowered = MModule.parse_from_file(output)
    transformed = next(f for f in lowered.get_functions() if f.symbol_name == func.symbol_name)
    return lowered, transformed
