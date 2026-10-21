#ifndef SCALEISA_ADAPTER_MEMORY_ABI_H
#define SCALEISA_ADAPTER_MEMORY_ABI_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class SymbolTable;
namespace func { class FuncOp; }
namespace scaleisa_adapter {
/// External memrefs are CPU memory. Their RV32 carriers contain the aligned
/// base pointer; the static memref offset and strides define element addresses.
LogicalResult validateCPUMemoryArgument(BlockArgument argument);

/// Stage array operands through scratchpads using functional aps.copy, then
/// retain scratchpad computation. Only rank-zero scalar references use
/// aps.load/aps.store. Physical interfaces and transfer splitting are downstream.
LogicalResult lowerCPUMemoryArgument(Value memref, Value cpuBase,
                                     SymbolTable &symbols);

/// Hoist static local buffers to the same module-global scratchpad convention.
LogicalResult hoistScratchpadAllocations(func::FuncOp function,
                                         SymbolTable &symbols);

/// Independent buffers require writable operands to be disjoint from the other
/// external operands. Validate the explicit scaleisa.noalias argument contract.
LogicalResult validateBufferedMemoryAliasing(Operation *function);
} // namespace scaleisa_adapter
} // namespace mlir

#endif
