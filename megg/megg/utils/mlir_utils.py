"""MLIR Utilities for Megg"""
from __future__ import annotations

import logging
import os
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Union

from megg.mlir_utils import (
    MLIRBlock,
    MLIRFunctionType,
    MLIRModule,
    MLIROperation,
    MLIRRegion,
    MLIRType,
    MLIRValue,
    PassManager,
)

logger = logging.getLogger(__name__)

class OperationType(IntEnum):
    """Operation type enumeration matching C++ enum"""
    MODULE = 0

    FUNC_FUNC = 1
    FUNC_CALL = 2
    FUNC_RETURN = 3

    ARITH_ADD = 4
    ARITH_SUB = 5
    ARITH_MUL = 6
    ARITH_DIV = 7
    ARITH_REM = 8
    ARITH_ADDF = 9
    ARITH_SUBF = 10
    ARITH_MULF = 11
    ARITH_DIVF = 12
    ARITH_REMF = 13
    ARITH_CONSTANT = 14
    ARITH_CMPI = 15
    ARITH_CMPF = 16
    ARITH_NEG = 17
    ARITH_NEGF = 18
    ARITH_SELECT = 19

    ARITH_ANDI = 20
    ARITH_ORI = 21
    ARITH_XORI = 22

    ARITH_SHLI = 60
    ARITH_SHRSI = 61
    ARITH_SHRUI = 62

    ARITH_INDEX_CAST = 23
    ARITH_SITOFP = 24
    ARITH_UITOFP = 25
    ARITH_FPTOSI = 26
    ARITH_FPTOUI = 27
    ARITH_EXTSI = 28
    ARITH_EXTUI = 29
    ARITH_TRUNCI = 30
    ARITH_BITCAST = 31

    AFFINE_FOR = 32
    AFFINE_IF = 33
    AFFINE_LOAD = 34
    AFFINE_STORE = 35
    AFFINE_YIELD = 36
    AFFINE_APPLY = 37

    SCF_FOR = 38
    SCF_IF = 39
    SCF_WHILE = 40
    SCF_YIELD = 41
    SCF_CONDITION = 42

    MEMREF_ALLOC = 43
    MEMREF_ALLOCA = 52
    MEMREF_DEALLOC = 44
    MEMREF_LOAD = 45
    MEMREF_STORE = 46
    MEMREF_CAST = 47
    MEMREF_GET_GLOBAL = 48
    MEMREF_GLOBAL = 49

    LLVM_CALL = 50
    LLVM_RETURN = 51

    UNKNOWN = 99

class MType:
    """Wrapper for MLIR Type"""

    def __init__(self, type_obj: MLIRType):
        self._type = type_obj

    def __str__(self) -> str:
        """Get string representation of the type"""
        return str(self._type)

class MFunctionType(MType):
    """Wrapper for MLIR FunctionType"""

    def __init__(self, func_type: MLIRFunctionType):
        super().__init__(func_type)
        self._func_type = func_type

    def get_inputs(self) -> List[MType]:
        """Get input types"""
        return [MType(t) for t in self._func_type.get_inputs()]

    def get_results(self) -> List[MType]:
        """Get result types"""
        return [MType(t) for t in self._func_type.get_results()]

class MValue:
    """Wrapper for MLIR Value"""

    def __init__(self, value: MLIRValue):
        self._value = value

    def __eq__(self, other):
        """Compare MValue objects based on underlying MLIR value pointer"""
        if not isinstance(other, MValue):
            return False
        return self._value.get_ptr() == other._value.get_ptr()

    def __hash__(self):
        """Hash based on underlying MLIR value pointer"""
        return hash(self._value.get_ptr())

    @property
    def type(self) -> str:
        """Get the type of this value as string"""
        return self._value.get_type_str()

    def get_type(self) -> MType:
        """Get the type of this value as MType object"""
        return MType(self._value.get_type())

    def get_defining_op(self) -> Optional[MOperation]:
        """Get the operation that defines this value"""
        op = self._value.get_defining_op()
        return MOperation(op) if op else None

    @property
    def owner(self) -> Optional[MOperation]:
        """Get the operation that owns/defines this value (alias for get_defining_op)"""
        return self.get_defining_op()

    def __str__(self) -> str:
        """Get string representation of the value"""
        return str(self._value)

class MModule:
    """MLIRModule cpp wrapper"""

    def __init__(self, source: Optional[str] = None):
        self._module = MLIRModule()
        if source is not None:
            if not self._module.parse_from_string(source):
                raise ValueError("Invalid MLIR module")

    @classmethod
    def parse(cls, mlir_text: str) -> MModule:
        if not mlir_text:
            raise ValueError("MLIR text cannot be empty")
        return cls(mlir_text)

    @classmethod
    def parse_from_file(cls, filepath: Union[str, Path]) -> MModule:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"MLIR file not found: {filepath}")
        instance = cls()
        success = instance._module.parse_from_file(str(filepath))
        if not success:
            raise ValueError(f"Invalid MLIR file: {filepath}")
        return instance

    @staticmethod
    def save_mlir(mod: MModule, file: str):
        if not os.path.isdir(os.path.dirname(file)):
            raise RuntimeError(f"{file} is not a valid path")
        with open(file, "w") as f:
            f.write(mod.to_string())

    def get_context(self):
        """Get the MLIR context of this module"""
        return self._module.get_context()

    def to_string(self) -> str:
        return self._module.to_string()

    def get_operations(self) -> List[MOperation]:
        """Get all operations in the module"""
        ops = self._module.get_operations()
        return [self._owned_operation(op) for op in ops]

    def get_functions(self) -> List[MOperation]:
        """Get all function operations in the module"""
        funcs = self._module.get_functions()
        return [self._owned_operation(func) for func in funcs]

    def _owned_operation(self, operation):
        result = MOperation(operation)
        result._module_owner = self
        return result

    def empty_like(self):
        result = MModule()
        result._module = self._module.empty_like()
        return result

    def append_to_module(self, op):
        if hasattr(op, '_op'):
            self._module.append_to_module(op._op)
        else:
            self._module.append_to_module(op)

    def clone_with_mapping(self):
        result = self._module.clone_with_mapping()
        if result is None:
            return None

        cloned_cpp_module, cpp_op_map = result
        cloned_module = MModule()
        cloned_module._module = cloned_cpp_module
        return cloned_module, cpp_op_map

    def has_unsupported_ops(self, func_op: MOperation) -> bool:
        """Check if function contains unsupported operations."""
        if not func_op.has_regions:
            return False
        regions = func_op.get_regions()
        if not regions:
            return False

        for region in regions:
            blocks = region.get_blocks()
            for block in blocks:
                operations = block.get_operations()
                for op in operations:
                    if 'call' in op.name:
                        return True
        return False

    def validate(self) -> bool:
        return self._module.validate()

    def __str__(self) -> str:
        return self.to_string()

    def __iter__(self):
        """Iterate over all operations in the module"""
        return iter(self.get_operations())

class MRegion:
    """Wrapper for MLIR Region"""

    def __init__(self, region: MLIRRegion):
        self._region = region

    @property
    def num_blocks(self) -> int:
        """Get number of blocks in the region"""
        return len(self._region)

    @property
    def empty(self) -> bool:
        """Check if region is empty"""
        return self._region.empty()

    def get_blocks(self) -> List['MBlock']:
        """Get all blocks in the region"""
        blocks = self._region.get_blocks()
        return [MBlock(block) for block in blocks]

    def __len__(self) -> int:
        return self.num_blocks

    def __iter__(self):
        """Iterate over blocks in the region"""
        return iter(self.get_blocks())

class MBlock:
    """Wrapper for MLIR Block"""

    def __init__(self, block: MLIRBlock):
        if isinstance(block, MBlock):
            block = block._block
        self._block = block

    @property
    def empty(self) -> bool:
        """Check if block is empty"""
        return self._block.empty()

    def get_operations(self) -> List['MOperation']:
        """Get all operations in the block"""
        ops = self._block.get_operations()
        return [MOperation(op) for op in ops]

    @property
    def arguments(self) -> List[MValue]:
        """Get block arguments"""
        args = self._block.get_arguments()
        return [MValue(arg) for arg in args]

    @property
    def operations(self):
        """Alias for get_operations()"""
        return self.get_operations()

    def __iter__(self):
        """Iterate over operations in the block"""
        return iter(self.get_operations())

    def __str__(self):
        return str(self._block)

    def get_terminator(self) -> MOperation:
        termi = self._block.get_terminator()
        if not isinstance(termi, MOperation):
            return MOperation(termi)
        else:
            return termi

    def get_parent_op(self):
        return self._block.get_parent_op()

class MOperation:
    """Wrapper for MLIR Operation"""

    def __init__(self, operation: MLIROperation):
        self._op = operation

    @property
    def name(self) -> str:
        """Get operation name (e.g., 'func.func', 'affine.for')"""
        return self._op.get_name()

    @property
    def symbol_name(self) -> str:
        """Get symbol name (e.g., 'matmul' from 'func.func @matmul')"""
        return self._op.get_symbol_name()

    @property
    def type(self) -> OperationType:
        """Get operation type as enum"""
        return OperationType(self._op.get_type())

    @property
    def get_function_type(self) -> Optional[MFunctionType]:
        func_type = self._op.get_function_type()
        return MFunctionType(func_type) if func_type else None

    @property
    def num_operands(self) -> int:
        """Get number of operands"""
        return self._op.get_num_operands()

    @property
    def num_results(self) -> int:
        """Get number of results"""
        return self._op.get_num_results()

    @property
    def operands(self) -> List[MValue]:
        """Get operands"""
        operands = self._op.get_operands()
        return [MValue(op) for op in operands]

    @property
    def results(self) -> List[MValue]:
        """Get results"""
        results = self._op.get_results()
        return [MValue(res) for res in results]

    @property
    def num_regions(self) -> int:
        """Get number of regions"""
        return self._op.get_num_regions()

    @property
    def has_regions(self) -> bool:
        """Check if operation has regions"""
        return self._op.has_regions()

    def get_regions(self) -> List[MRegion]:
        """Get all regions of the operation"""
        regions = self._op.get_regions()
        return [MRegion(region) for region in regions]

    @property
    def get_first_region(self) -> MRegion:
        return self.get_regions()[0]

    @property
    def regions(self) -> List[MRegion]:
        """Alias for get_regions()"""
        return self.get_regions()

    @property
    def is_terminator(self) -> bool:
        """Check if operation is a terminator"""
        return self._op.is_terminator()

    def get_parent_op(self) -> Optional[MOperation]:
        """Get parent operation if exists"""
        parent = self._op.get_parent_op()
        return MOperation(parent) if parent else None

    def get_block(self) -> Optional[MBlock]:
        """Get containing block"""
        block = self._op.get_block()
        return MBlock(block) if block else None

    def get_attr(self, name: str):
        """Get attribute value by name"""
        return self._op.get_attr(name)

    def has_attr(self, name: str) -> bool:
        """Check if operation has an attribute"""
        return self._op.has_attr(name)

    @property
    def attr_names(self) -> List[str]:
        """Get all attribute names"""
        return self._op.get_attr_names()

    def erase(self):
        """Erase this operation from its parent block"""
        self._op.erase()

    def __str__(self) -> str:
        """Get string representation of the operation"""
        return str(self._op)

class MLIRPassManager:
    """MLIRPassManager cpp wrapper"""

    def __init__(self, module: MModule):
        self._pass_manager = PassManager(module._module)
        self._module = module

    @staticmethod
    def _normalize_function_target(function: MOperation):
        """Convert function target to Operation pointer."""
        if function is None:
            return None
        if isinstance(function, MOperation):
            return function._op
        raise TypeError(
            "function must be None or an MOperation representing a func.func")

    def register_affine_loop_unroll_passes(
        self,
        function: MOperation,
        loop_op: MOperation,
        unroll_factor: int = 4,
    ) -> None:
        """Register affine loop unrolling passes for a specific loop operation."""
        func_ref = function._op if function is not None else None
        loop_ref = loop_op._op if loop_op is not None else None
        self._pass_manager.register_affine_loop_unroll_passes(
            func_ref, loop_ref, unroll_factor
        )

    def register_affine_loop_unroll_jam_passes(
        self,
        function: MOperation,
        loop_op: MOperation,
        unroll_jam_factor: int = 4,
    ) -> None:
        """Register affine loop unroll-and-jam passes for a specific loop operation."""
        func_ref = function._op if function is not None else None
        loop_ref = loop_op._op if loop_op is not None else None
        self._pass_manager.register_affine_loop_unroll_jam_passes(
            func_ref, loop_ref, unroll_jam_factor
        )

    def register_affine_loop_tiling_pass(
        self,
        function: MOperation,
        loop_op: MOperation,
        default_tile_size: int = 32,
    ) -> None:
        """Register affine loop tiling pass for a specific loop operation."""
        func_ref = function._op if function is not None else None
        loop_ref = loop_op._op if loop_op is not None else None
        self._pass_manager.register_affine_loop_tiling_pass(
            func_ref, loop_ref, default_tile_size
        )

    def register_affine_loop_unroll_jam_pass(
        self,
        function: Union[str, MOperation, None] = None,
        loop_name: str = "",
        unroll_jam_factor: int = 4,
    ) -> None:
        """Register affine loop unroll-and-jam pass for a specific loop by name."""
        target_function = self._normalize_function_target(function)
        self._pass_manager.register_affine_loop_unroll_jam_pass(
            target_function, loop_name, unroll_jam_factor
        )

    def run_pass(self, pass_name: str) -> bool:
        """Run a specific registered pass by name"""
        return self._pass_manager.run_pass(pass_name)

    def run_all_pass(self) -> bool:
        return self._pass_manager.run_all_pass()

    def verify_module(self) -> bool:
        """Verify the MLIR module is valid"""
        return self._pass_manager.verify_module()

    def clear_custom_passes(self) -> None:
        """Clear all registered custom passes"""
        self._pass_manager.clear_custom_passes()
