"""Loop Analysis Utilities"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from megg.utils.mlir_utils import MModule, MOperation, OperationType

logger = logging.getLogger(__name__)

def extract_loop_lengths(module: MModule) -> Dict[str, Dict[str, int]]:
    """Extract loop bounds from all SCF for loops in the module."""
    loop_bounds: Dict[str, Dict[str, int]] = {}
    loop_counter = {'value': 0}
    
    for func in module.get_functions():
        _analyze_function(func, loop_bounds, loop_counter, depth=0)
    
    logger.info(f"Extracted loop bounds for {len(loop_bounds)} loops")    
    return loop_bounds

def _analyze_function(
    func: MOperation,
    loop_bounds: Dict[str, Dict[str, int]],
    loop_counter: Dict[str, int],
    depth: int,
) -> None:
    """Recursively analyze SCF loops in a function."""
    regions = func.get_regions()
    if not regions:
        return
    
    for region in regions:
        blocks = region.get_blocks()
        if not blocks:
            continue
        
        for block in blocks:
            ops = block.get_operations()
            for op in ops:
                if hasattr(op, 'type') and op.type == OperationType.SCF_FOR:
                    loop_name = _get_loop_name(op, loop_counter['value'])
                    
                    bounds = _extract_scf_bounds(op)
                    
                    if bounds is not None:
                        bounds['nest_level'] = depth
                        bounds['nested_loop_count'] = _count_nested_loops(op)
                        loop_bounds[loop_name] = bounds
                        logger.debug(
                            "Found loop '%s' with bounds %s at depth %d (nested=%d)",
                            loop_name,
                            bounds,
                            depth,
                            bounds['nested_loop_count'],
                        )
                    else:
                        logger.debug(
                            "Found loop '%s' but could not determine bounds at depth %d",
                            loop_name,
                            depth,
                        )
                    
                    loop_counter['value'] += 1
                    
                    _analyze_function(op, loop_bounds, loop_counter, depth + 1)

def _get_loop_name(loop_op: MOperation, counter: int) -> str:
    """Get loop name from attribute or generate one."""
    if loop_op.has_attr("name"):
        try:
            return str(loop_op.get_attr("name"))
        except Exception:
            pass
    
    return f"loop_{counter}"

def _extract_scf_bounds(loop_op: MOperation) -> Optional[Dict[str, int]]:
    """Extract loop bounds from SCF for loop using MLIR API."""
    try:
        # Use the C++ binding method to get bounds
        bounds = loop_op._op.get_scf_loop_bounds()
        if bounds is not None:
            return {
                'lower': int(bounds['lower']),
                'upper': int(bounds['upper']),
                'step': int(bounds['step'])
            }
    except Exception as e:
        logger.debug(f"Failed to extract loop bounds via MLIR API: {e}")
    
    return None

def _count_nested_loops(loop_op: MOperation) -> int:
    """Count the number of loop operations contained within the given loop."""
    loop_kinds = {
        OperationType.SCF_FOR,
        OperationType.AFFINE_FOR,
        OperationType.SCF_WHILE,
    }

    def _walk(op: MOperation, is_root: bool = False) -> int:
        total = 0
        try:
            op_type = op.type
        except Exception:
            op_type = None

        if not is_root and op_type in loop_kinds:
            total += 1

        if getattr(op, 'has_regions', False):
            try:
                regions = op.get_regions()
            except Exception:
                regions = []
            for region in regions:
                for block in region.get_blocks():
                    for child in block.get_operations():
                        total += _walk(child)

        return total

    return _walk(loop_op, is_root=True)


