#include "scaleisa-adapter/MemoryABI.h"
#include "APS/APSOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>

namespace mlir::scaleisa_adapter {
namespace {
struct Layout {
  SmallVector<int64_t> strides;
  int64_t offset;
  unsigned elementBytes;
};

FailureOr<Layout> getLayout(MemRefType type) {
  Layout layout;
  Type element = type.getElementType();
  auto integer = dyn_cast<IntegerType>(element);
  if ((!integer || !integer.isSignless() ||
       (integer.getWidth() != 8 && integer.getWidth() != 16 &&
        integer.getWidth() != 32)) && !element.isF32())
    return failure();
  if (Attribute space = type.getMemorySpace()) {
    auto integerSpace = dyn_cast<IntegerAttr>(space);
    if (!integerSpace || integerSpace.getInt() != 0)
      return failure();
  }
  if (failed(type.getStridesAndOffset(layout.strides, layout.offset)) ||
      ShapedType::isDynamic(layout.offset) ||
      llvm::any_of(layout.strides, ShapedType::isDynamic))
    return failure();
  layout.elementBytes = element.getIntOrFloatBitWidth() / 8;
  if (type.getRank()) {
    if (!type.hasStaticShape())
      return failure();
    int64_t stride = 1;
    for (int64_t dim = type.getRank(); dim-- > 0;) {
      if (type.getDimSize(dim) <= 0 || layout.strides[dim] != stride ||
          type.getDimSize(dim) > INT32_MAX / stride)
        return failure();
      stride *= type.getDimSize(dim);
    }
    if (stride > INT32_MAX / layout.elementBytes)
      return failure();
  }
  return layout;
}

Value constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIntOp>(loc, value, 32);
}

Value byteAddress(OpBuilder &builder, Location loc, Value base,
                  ValueRange indices, const Layout &layout) {
  Value linear = constant(builder, loc, layout.offset);
  for (auto [index, stride] : llvm::zip(indices, layout.strides)) {
    Value value = index;
    if (value.getType().isIndex())
      value = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), value);
    else if (value.getType().getIntOrFloatBitWidth() < 32)
      value = builder.create<arith::ExtSIOp>(loc, builder.getI32Type(), value);
    else if (value.getType().getIntOrFloatBitWidth() > 32)
      value = builder.create<arith::TruncIOp>(loc, builder.getI32Type(), value);
    if (stride != 1)
      value = builder.create<arith::MulIOp>(loc, value, constant(builder, loc, stride));
    linear = builder.create<arith::AddIOp>(loc, linear, value);
  }
  if (layout.elementBytes != 1)
    linear = builder.create<arith::MulIOp>(
        loc, linear, constant(builder, loc, layout.elementBytes));
  return builder.create<arith::AddIOp>(loc, base, linear);
}

Value createScratchpad(OpBuilder &builder, Location loc, MemRefType type,
                        StringRef name, ModuleOp module, SymbolTable &symbols) {
  // Match CADL frontend output: globals with var_name are discovered by
  // APSMemoryMap, and computations access them with memref.get_global.
  OpBuilder globals(module.getContext());
  globals.setInsertionPointToStart(module.getBody());
  auto global = globals.create<memref::GlobalOp>(
      loc, globals.getStringAttr(name), StringAttr(), TypeAttr::get(type),
      Attribute(), UnitAttr(), IntegerAttr());
  symbols.insert(global);
  global->setAttr("var_name", globals.getStringAttr(global.getSymName()));
  global->setAttr("scaleisa.scratchpad", globals.getUnitAttr());
  return builder.create<memref::GetGlobalOp>(loc, type, global.getSymName());
}

LogicalResult lowerScratchpadAccesses(Value memory, Value scratchpad,
                                      const Layout &layout) {
  auto type = cast<MemRefType>(memory.getType());
  SmallVector<Operation *> users(memory.getUsers());
  auto global = scratchpad.getDefiningOp<memref::GetGlobalOp>();
  for (Operation *user : users) {
    OpBuilder builder(user);
    Location loc = user->getLoc();
    ValueRange indices;
    Value stored;
    Type resultType;
    if (auto load = dyn_cast<aps::ReadSmem>(user)) {
      indices = load.getIndices();
      resultType = load.getType();
    } else if (auto load = dyn_cast<memref::LoadOp>(user)) {
      indices = load.getIndices();
      resultType = load.getType();
    } else if (auto store = dyn_cast<aps::WriteSmem>(user)) {
      indices = store.getIndices();
      stored = store.getValue();
    } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
      indices = store.getIndices();
      stored = store.getValue();
    } else if (auto dim = dyn_cast<memref::DimOp>(user)) {
      auto index = dim.getConstantIndex();
      if (!index || *index < 0 || *index >= type.getRank())
        return dim.emitError("scratchpad dimensions must be statically known");
      auto value = builder.create<arith::ConstantIndexOp>(loc, type.getDimSize(*index));
      dim.getResult().replaceAllUsesWith(value.getResult());
      dim.erase();
      continue;
    } else {
      return user->emitError("unsupported scratchpad alias or escaping local buffer");
    }
    Value result;
    if (!type.getRank()) {
      if (stored)
        builder.create<aps::GlobalStore>(loc, stored, global.getName());
      else
        result = builder.create<aps::GlobalLoad>(loc, resultType, global.getName());
    } else {
      Layout local = layout;
      local.offset = 0;
      local.elementBytes = 1;
      Value index = byteAddress(builder, loc, constant(builder, loc, 0), indices, local);
      if (stored)
        builder.create<aps::WriteSmem>(loc, stored, scratchpad, ValueRange{index});
      else
        result = builder.create<aps::ReadSmem>(loc, resultType, scratchpad, ValueRange{index});
    }
    if (result)
      user->getResult(0).replaceAllUsesWith(result);
    user->erase();
  }
  return success();
}
} // namespace

LogicalResult validateCPUMemoryArgument(BlockArgument argument) {
  auto type = dyn_cast<MemRefType>(argument.getType());
  if (!type || failed(getLayout(type)))
    return argument.getOwner()->getParentOp()->emitError(
        "buffered CPU arrays require a static contiguous row-major layout, positive extents, and i8/i16/i32/f32 elements in default memory space; dynamic descriptor fields need explicit legalization");
  for (Operation *user : argument.getUsers()) {
    if (!type.getRank() && isa<memref::LoadOp, memref::StoreOp>(user))
      continue;
    if (auto load = dyn_cast<aps::ReadSmem>(user)) {
      if (load.getMemref() == argument)
        continue;
    }
    if (auto store = dyn_cast<aps::WriteSmem>(user)) {
      if (store.getMemref() == argument)
        continue;
    }
    if (auto dim = dyn_cast<memref::DimOp>(user)) {
      auto index = dim.getConstantIndex();
      if (index && *index >= 0 && *index < type.getRank() &&
          !type.isDynamicDim(*index))
        continue;
    }
    return user->emitError(
        "CPU memref argument supports direct APS reads/writes and static memref.dim only; aliases, views, and dynamic descriptor fields need explicit legalization");
  }
  return success();
}

LogicalResult validateBufferedMemoryAliasing(Operation *operation) {
  auto function = cast<func::FuncOp>(operation);
  SmallVector<BlockArgument> active;
  SmallVector<BlockArgument> written;
  bool hasArray = false;
  for (BlockArgument argument : function.getArguments()) {
    auto type = dyn_cast<MemRefType>(argument.getType());
    if (!type)
      continue;
    bool accessed = false, writes = false;
    for (Operation *user : argument.getUsers()) {
      accessed |= isa<aps::ReadSmem, aps::WriteSmem,
                      memref::LoadOp, memref::StoreOp>(user);
      writes |= isa<aps::WriteSmem, memref::StoreOp>(user);
    }
    if (accessed) {
      active.push_back(argument);
      hasArray |= type.getRank() != 0;
    }
    if (writes)
      written.push_back(argument);
  }
  if (!hasArray)
    return success();
  if (active.size() > 1)
    for (BlockArgument argument : written)
      if (!isa_and_nonnull<UnitAttr>(
              function.getArgAttr(argument.getArgNumber(), "scaleisa.noalias")))
        return function.emitError(
            "buffering multiple external operands requires scaleisa.noalias on each writable memref argument; overlapping writable buffers need explicit alias legalization");
  bool unmodeledCPUAccess = false;
  function.walk([&](Operation *op) {
    unmodeledCPUAccess |= isa<aps::Load, aps::Store, aps::Copy>(op);
  });
  if (unmodeledCPUAccess)
    return function.emitError(
        "cannot buffer external arrays around existing CPU-memory effects without explicit alias legalization");
  return success();
}

LogicalResult lowerCPUMemoryArgument(Value memref, Value cpuBase,
                                     SymbolTable &symbols) {
  auto type = cast<MemRefType>(memref.getType());
  auto layout = getLayout(type);
  if (failed(layout))
    return failure();
  SmallVector<Operation *> users(memref.getUsers());
  bool read = false, written = false;
  for (Operation *user : users) {
    read |= isa<aps::ReadSmem>(user);
    written |= isa<aps::WriteSmem>(user);
  }
  Value scratchpad;
  if (type.getRank() && (read || written)) {
    OpBuilder builder(memref.getDefiningOp());
    Location loc = memref.getLoc();
    auto bufferType = MemRefType::get({type.getNumElements()}, type.getElementType());
    auto function = memref.getDefiningOp()->getParentOfType<func::FuncOp>();
    scratchpad = createScratchpad(builder, loc, bufferType,
        function.getName().str() + "_buffer", function->getParentOfType<ModuleOp>(), symbols);
    Value address = cpuBase;
    if (layout->offset)
      address = builder.create<arith::AddIOp>(
          loc, cpuBase, constant(builder, loc, static_cast<uint32_t>(
              static_cast<uint64_t>(layout->offset) * layout->elementBytes)));
    Value start = constant(builder, loc, 0);
    Value length = constant(builder, loc, type.getNumElements());
    // Import the original contents even for write-only arguments: a partial
    // store must preserve untouched elements when the buffer is written back.
    builder.create<aps::Copy>(loc, aps::CopyDirection::In, address,
                              ValueRange{scratchpad}, start, length);
    if (written) {
      auto function = memref.getDefiningOp()->getParentOfType<func::FuncOp>();
      Operation *completion = function.front().getTerminator();
      for (Operation &op : function.front())
        if (isa<aps::WriteIRF>(op)) {
          completion = &op;
          break;
        }
      builder.setInsertionPoint(completion);
      builder.create<aps::Copy>(loc, aps::CopyDirection::Out, address,
                                ValueRange{scratchpad}, start, length);
    }
  }
  if (scratchpad)
    return lowerScratchpadAccesses(memref, scratchpad, *layout);
  for (Operation *user : users) {
    OpBuilder builder(user);
    Location loc = user->getLoc();
    if (auto load = dyn_cast<memref::LoadOp>(user)) {
      Value address = byteAddress(builder, loc, cpuBase, {}, *layout);
      auto value = builder.create<aps::Load>(loc, load.getType(), address);
      load.getResult().replaceAllUsesWith(value.getResult());
      load.erase();
    } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
      Value address = byteAddress(builder, loc, cpuBase, {}, *layout);
      builder.create<aps::Store>(loc, store.getValue(), address);
      store.erase();
    } else if (auto load = dyn_cast<aps::ReadSmem>(user)) {
      Value address = byteAddress(builder, loc, cpuBase, load.getIndices(), *layout);
      auto value = builder.create<aps::Load>(loc, load.getType(), address);
      value->setAttrs(load->getAttrs());
      load.getResult().replaceAllUsesWith(value.getResult());
      load.erase();
    } else if (auto store = dyn_cast<aps::WriteSmem>(user)) {
      Value address = byteAddress(builder, loc, cpuBase, store.getIndices(), *layout);
      auto write = builder.create<aps::Store>(loc, store.getValue(), address);
      write->setAttrs(store->getAttrs());
      store.erase();
    } else if (auto dim = dyn_cast<memref::DimOp>(user)) {
      auto value = builder.create<arith::ConstantIndexOp>(
          loc, type.getDimSize(*dim.getConstantIndex()));
      dim.getResult().replaceAllUsesWith(value.getResult());
      dim.erase();
    } else {
      return user->emitError("unlegalized use of an external CPU memref");
    }
  }
  return success();
}

LogicalResult hoistScratchpadAllocations(func::FuncOp function,
                                         SymbolTable &symbols) {
  SmallVector<Operation *> allocations;
  function.walk([&](Operation *op) {
    if (isa<memref::AllocaOp, memref::AllocOp>(op))
      allocations.push_back(op);
  });
  for (Operation *allocation : allocations) {
    Value memory = allocation->getResult(0);
    auto type = cast<MemRefType>(memory.getType());
    auto layout = getLayout(type);
    if (failed(layout) || allocation->getNumOperands())
      return allocation->emitError("local scratchpads require positive static contiguous shapes");
    auto globalType = type.getRank()
        ? MemRefType::get({type.getNumElements()}, type.getElementType()) : type;
    OpBuilder builder(allocation);
    Value scratchpad = createScratchpad(builder, allocation->getLoc(), globalType,
        function.getName().str() + "_local", function->getParentOfType<ModuleOp>(), symbols);
    if (failed(lowerScratchpadAccesses(memory, scratchpad, *layout)))
      return failure();
    allocation->erase();
  }
  return success();
}
} // namespace mlir::scaleisa_adapter
