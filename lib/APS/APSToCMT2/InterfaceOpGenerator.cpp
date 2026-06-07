//===- InterfaceOpGenerator.cpp - Interface Operation Generator ------------===//
//
// This file implements the interface operation generator for TOR functions
//
//===----------------------------------------------------------------------===//

#include "APS/BBHandler.h"
#include "circt/Dialect/Cmt2/ECMT2/Signal.h"
#include "APS/APSOps.h"

namespace mlir {

using namespace mlir;
using namespace mlir::tor;
using namespace circt::cmt2::ecmt2;
using namespace circt::cmt2::ecmt2::stl;
using namespace circt::firrtl;

LogicalResult InterfaceOpGenerator::generateRule(Operation *op, mlir::OpBuilder &b,
                                               Location loc, int64_t slot,
                                               llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  if (auto itfcLoadReq = dyn_cast<aps::LoadIssue>(op)) {
    return generateItfcLoadReq(itfcLoadReq, b, loc, slot, localMap);
  } else if (auto itfcLoadCollect = dyn_cast<aps::LoadWait>(op)) {
    return generateItfcLoadCollect(itfcLoadCollect, b, loc, slot, localMap);
  } else if (auto itfcStoreReq = dyn_cast<aps::StoreIssue>(op)) {
    return generateItfcStoreReq(itfcStoreReq, b, loc, slot, localMap);
  } else if (auto itfcStoreCollect = dyn_cast<aps::StoreWait>(op)) {
    return generateItfcStoreCollect(itfcStoreCollect, b, loc, slot, localMap);
  }

  return op->emitError(
      "internal error: unsupported op reached interface generator");
}

bool InterfaceOpGenerator::canHandle(Operation *op) const {
  return isa<aps::LoadIssue, aps::LoadWait, aps::StoreIssue, aps::StoreWait>(op);
}

LogicalResult InterfaceOpGenerator::generateItfcLoadReq(aps::LoadIssue op, mlir::OpBuilder &b,
                                                      Location loc, int64_t slot,
                                                      llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Handle CPU interface load request operations
  // This sends a load request to memory and returns a request token
  auto context = b.getContext();

  auto userCmdBundleType = BundleType::get(
      context,
      {BundleType::BundleElement{b.getStringAttr("addr"), false,
                                 UIntType::get(context, 32)},
       BundleType::BundleElement{b.getStringAttr("cmd"), false,
                                 UIntType::get(context, 1)},
       BundleType::BundleElement{b.getStringAttr("size"), false,
                                 UIntType::get(context, 2)},
       BundleType::BundleElement{b.getStringAttr("data"), false,
                                 UIntType::get(context, 32)},
       BundleType::BundleElement{b.getStringAttr("mask"), false,
                                 UIntType::get(context, 4)},
       BundleType::BundleElement{b.getStringAttr("tag"), false,
                                 UIntType::get(context, 8)}});

  auto addr = getValueInRule(op.getCpuAddr(), op.getOperation(), b, localMap, loc);
  if (failed(addr)) {
    op.emitError("Failed to get address for interface load request");
    return failure();
  }
  auto addrSignal = Signal(*addr, &b, loc);
  auto readCmd = UInt::constant(0, 1, b, loc);
  auto data = UInt::constant(0, 32, b, loc);
  auto size = UInt::constant(2, 2, b, loc);
  auto mask = UInt::constant(0, 4, b, loc);
  auto tagConst = UInt::constant(0x3f, 8, b, loc);
  auto tag = (addrSignal & tagConst).bits(7, 0);

  llvm::SmallVector<mlir::Value> bundleFields = {
      *addr, readCmd.getValue(), size.getValue(),
      data.getValue(), mask.getValue(), tag.getValue()};
  auto bundleValue = b.create<BundleCreateOp>(loc, userCmdBundleType, bundleFields);

  auto hellaMemInstance = bbHandler->getHellaMemInstance();
  if (!hellaMemInstance) {
    op.emitError("HellaMem instance not available");
    return failure();
  }

  hellaMemInstance->callMethod("cmd_from_user", {bundleValue}, b);

  localMap[op.getResult()] = UInt::constant(1, 1, b, loc).getValue();
  return success();
}

LogicalResult InterfaceOpGenerator::generateItfcLoadCollect(aps::LoadWait op, mlir::OpBuilder &b,
                                                          Location loc, int64_t slot,
                                                          llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  auto hellaMemInstance = bbHandler->getHellaMemInstance();
  if (!hellaMemInstance) {
    op.emitError("HellaMem instance not available");
    return failure();
  }

  auto resp = hellaMemInstance->callMethod("resp_to_user", {}, b)[0];
  auto respBundle = Bundle(resp, &b, loc);
  auto loadResult = respBundle["data"];

  localMap[op.getResult()] = loadResult.getValue();
  return success();
}

LogicalResult InterfaceOpGenerator::generateItfcStoreReq(aps::StoreIssue op, mlir::OpBuilder &b,
                                                       Location loc, int64_t slot,
                                                       llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Handle CPU interface store request operations
  // This sends a store request to memory and returns a request token

  auto context = b.getContext();

  auto value = getValueInRule(op.getValue(), op.getOperation(), b, localMap, loc);
  auto addr = getValueInRule(op.getCpuAddr(), op.getOperation(), b, localMap, loc);
  if (failed(value) || failed(addr)) {
    op.emitError("Failed to get value or address for interface store request");
    return failure();
  }

  auto userCmdBundleType = BundleType::get(
      context,
      {BundleType::BundleElement{b.getStringAttr("addr"), false,
                                 UIntType::get(context, 32)},
       BundleType::BundleElement{b.getStringAttr("cmd"), false,
                                 UIntType::get(context, 1)},
       BundleType::BundleElement{b.getStringAttr("size"), false,
                                 UIntType::get(context, 2)},
       BundleType::BundleElement{b.getStringAttr("data"), false,
                                 UIntType::get(context, 32)},
       BundleType::BundleElement{b.getStringAttr("mask"), false,
                                 UIntType::get(context, 4)},
       BundleType::BundleElement{b.getStringAttr("tag"), false,
                                 UIntType::get(context, 8)}});

  auto addrSignal = Signal(*addr, &b, loc);
  auto WriteCmd = UInt::constant(1, 1, b, loc);
  auto size = UInt::constant(2, 2, b, loc);
  auto mask = UInt::constant(0, 4, b, loc);
  auto tagConst = UInt::constant(0x3f, 8, b, loc);
  auto tag = (addrSignal & tagConst).bits(7, 0);

  llvm::SmallVector<mlir::Value> bundleFields = {
      *addr, WriteCmd.getValue(), size.getValue(),
      *value, mask.getValue(), tag.getValue()};
  auto bundleValue = b.create<BundleCreateOp>(loc, userCmdBundleType, bundleFields);

  auto hellaMemInstance = bbHandler->getHellaMemInstance();
  if (!hellaMemInstance) {
    op.emitError("HellaMem instance not available");
    return failure();
  }

  hellaMemInstance->callMethod("cmd_from_user", {bundleValue}, b);

  localMap[op.getResult()] = UInt::constant(1, 1, b, loc).getValue();
  return success();
}

LogicalResult InterfaceOpGenerator::generateItfcStoreCollect(aps::StoreWait op, mlir::OpBuilder &b,
                                                           Location loc, int64_t slot,
                                                           llvm::DenseMap<mlir::Value, mlir::Value> &localMap) {
  // Do nothing here, don't reply...
  return success();
}

} // namespace mlir
