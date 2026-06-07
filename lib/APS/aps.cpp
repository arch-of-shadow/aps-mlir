#include "APS/APSDialect.h"
#include "APS/APSOps.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/STLExtras.h"

#include "APS/APSDialect.cpp.inc"
#include "APS/APSEnums.cpp.inc"
#define GET_OP_CLASSES
#include "APS/APS.cpp.inc"

using namespace mlir;
using namespace aps;

void APSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "APS/APS.cpp.inc"
  >();
}

static Attribute getCopyDirectionAttr(MLIRContext *ctx,
                                      aps::CopyDirection direction) {
  return IntegerAttr::get(IntegerType::get(ctx, 32),
                          static_cast<int32_t>(direction));
}

static void printCommaSeparatedValues(OpAsmPrinter &p, ValueRange values) {
  llvm::interleaveComma(values, p, [&](Value value) { p << value; });
}

static void printCommaSeparatedTypes(OpAsmPrinter &p, TypeRange types) {
  llvm::interleaveComma(types, p, [&](Type type) { p << type; });
}

static ParseResult parseCopyPayload(OpAsmParser &parser,
                                    OperationState &result,
                                    StringRef directionAttrName) {
  OpAsmParser::UnresolvedOperand cpuAddr;
  SmallVector<OpAsmParser::UnresolvedOperand> memrefs;
  OpAsmParser::UnresolvedOperand start;
  OpAsmParser::UnresolvedOperand length;
  Type cpuAddrType;
  SmallVector<Type> memrefTypes;
  Type startType;
  Type lengthType;

  bool isOut = succeeded(parser.parseOptionalLParen());
  if (isOut) {
    if (parser.parseOperandList(memrefs) || parser.parseRParen() ||
        parser.parseLSquare() || parser.parseOperand(start) ||
        parser.parseRSquare() || parser.parseComma() ||
        parser.parseOperand(cpuAddr) || parser.parseComma() ||
        parser.parseOperand(length))
      return failure();
  } else {
    if (parser.parseOperand(cpuAddr) || parser.parseComma() ||
        parser.parseLParen() || parser.parseOperandList(memrefs) ||
        parser.parseRParen() || parser.parseLSquare() ||
        parser.parseOperand(start) || parser.parseRSquare() ||
        parser.parseComma() || parser.parseOperand(length))
      return failure();
  }

  result.addAttribute(directionAttrName,
                      getCopyDirectionAttr(parser.getContext(),
                                           isOut ? aps::CopyDirection::Out
                                                 : aps::CopyDirection::In));

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon())
    return failure();

  if (isOut) {
    if (parser.parseLParen() || parser.parseTypeList(memrefTypes) ||
        parser.parseRParen() || parser.parseComma() ||
        parser.parseType(startType) || parser.parseComma() ||
        parser.parseType(cpuAddrType) || parser.parseComma() ||
        parser.parseType(lengthType))
      return failure();
  } else {
    if (parser.parseType(cpuAddrType) || parser.parseComma() ||
        parser.parseLParen() || parser.parseTypeList(memrefTypes) ||
        parser.parseRParen() || parser.parseComma() ||
        parser.parseType(startType) || parser.parseComma() ||
        parser.parseType(lengthType))
      return failure();
  }

  if (parser.resolveOperand(cpuAddr, cpuAddrType, result.operands) ||
      parser.resolveOperands(memrefs, memrefTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperand(start, startType, result.operands) ||
      parser.resolveOperand(length, lengthType, result.operands))
    return failure();

  return success();
}

ParseResult Copy::parse(OpAsmParser &parser, OperationState &result) {
  return parseCopyPayload(parser, result, getDirectionAttrName(result.name));
}

void Copy::print(OpAsmPrinter &p) {
  bool isOut = getDirection() == aps::CopyDirection::Out;
  p << " ";
  if (isOut) {
    p << "(";
    printCommaSeparatedValues(p, getMemrefs());
    p << ")[" << getStart() << "], " << getCpuAddr() << ", " << getLength();
  } else {
    p << getCpuAddr() << ", (";
    printCommaSeparatedValues(p, getMemrefs());
    p << ")[" << getStart() << "], " << getLength();
  }

  p.printOptionalAttrDict((*this)->getAttrs(), {getDirectionAttrName()});
  p << " : ";
  if (isOut) {
    p << "(";
    printCommaSeparatedTypes(p, getMemrefs().getTypes());
    p << "), " << getStart().getType() << ", " << getCpuAddr().getType()
      << ", " << getLength().getType();
  } else {
    p << getCpuAddr().getType() << ", (";
    printCommaSeparatedTypes(p, getMemrefs().getTypes());
    p << "), " << getStart().getType() << ", " << getLength().getType();
  }
}

ParseResult CopyBy::parse(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr itfcAttr;
  if (parser.parseAttribute(itfcAttr, getItfcAttrName(result.name),
                            result.attributes) ||
      parser.parseComma())
    return failure();

  return parseCopyPayload(parser, result, getDirectionAttrName(result.name));
}

void CopyBy::print(OpAsmPrinter &p) {
  bool isOut = getDirection() == aps::CopyDirection::Out;
  p << " " << getItfcAttr() << ", ";
  if (isOut) {
    p << "(";
    printCommaSeparatedValues(p, getMemrefs());
    p << ")[" << getStart() << "], " << getCpuAddr() << ", " << getLength();
  } else {
    p << getCpuAddr() << ", (";
    printCommaSeparatedValues(p, getMemrefs());
    p << ")[" << getStart() << "], " << getLength();
  }

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getDirectionAttrName(), getItfcAttrName()});
  p << " : ";
  if (isOut) {
    p << "(";
    printCommaSeparatedTypes(p, getMemrefs().getTypes());
    p << "), " << getStart().getType() << ", " << getCpuAddr().getType()
      << ", " << getLength().getType();
  } else {
    p << getCpuAddr().getType() << ", (";
    printCommaSeparatedTypes(p, getMemrefs().getTypes());
    p << "), " << getStart().getType() << ", " << getLength().getType();
  }
}

ParseResult CopyIssue::parse(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr itfcAttr;
  if (parser.parseAttribute(itfcAttr, getItfcAttrName(result.name),
                            result.attributes) ||
      parser.parseComma())
    return failure();

  if (parseCopyPayload(parser, result, getDirectionAttrName(result.name)))
    return failure();

  Type requestType;
  if (parser.parseArrow() || parser.parseType(requestType))
    return failure();
  result.addTypes(requestType);
  return success();
}

void CopyIssue::print(OpAsmPrinter &p) {
  bool isOut = getDirection() == aps::CopyDirection::Out;
  p << " " << getItfcAttr() << ", ";
  if (isOut) {
    p << "(";
    printCommaSeparatedValues(p, getMemrefs());
    p << ")[" << getStart() << "], " << getCpuAddr() << ", " << getLength();
  } else {
    p << getCpuAddr() << ", (";
    printCommaSeparatedValues(p, getMemrefs());
    p << ")[" << getStart() << "], " << getLength();
  }

  p.printOptionalAttrDict((*this)->getAttrs(),
                          {getDirectionAttrName(), getItfcAttrName()});
  p << " : ";
  if (isOut) {
    p << "(";
    printCommaSeparatedTypes(p, getMemrefs().getTypes());
    p << "), " << getStart().getType() << ", " << getCpuAddr().getType()
      << ", " << getLength().getType();
  } else {
    p << getCpuAddr().getType() << ", (";
    printCommaSeparatedTypes(p, getMemrefs().getTypes());
    p << "), " << getStart().getType() << ", " << getLength().getType();
  }
  p << " -> " << getRequest().getType();
}

//===----------------------------------------------------------------------===//
// MemoryMapOp
//===----------------------------------------------------------------------===//

ParseResult MemoryMapOp::parse(OpAsmParser &parser, OperationState &result) {
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  return success();
}

void MemoryMapOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// GlobalLoadOp Canonicalization
//===----------------------------------------------------------------------===//

namespace {
static bool isSameGlobalMemref(StringRef globalName, Value memref) {
  auto globalOp = memref.getDefiningOp<memref::GetGlobalOp>();
  return globalOp && globalOp.getName() == globalName;
}

static bool containsGlobalMemref(ValueRange memrefs, StringRef globalName) {
  return llvm::any_of(memrefs, [&](Value memref) {
    return isSameGlobalMemref(globalName, memref);
  });
}

static bool sameGlobal(GlobalLoad loadOp, GlobalStore storeOp) {
  return loadOp.getGlobalName() == storeOp.getGlobalName();
}

struct FoldGlobalLoadAfterStore : public OpRewritePattern<GlobalLoad> {
  using OpRewritePattern<GlobalLoad>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalLoad loadOp,
                                 PatternRewriter &rewriter) const override {
    for (Operation *prevOp = loadOp->getPrevNode(); prevOp;
         prevOp = prevOp->getPrevNode()) {
      if (prevOp->getNumRegions() > 0)
        return failure();

      auto storeOp = dyn_cast<GlobalStore>(prevOp);
      if (storeOp && sameGlobal(loadOp, storeOp)) {
        rewriter.replaceOp(loadOp, storeOp.getValue());
        return success();
      }

      if (auto copyOp = dyn_cast<Copy>(prevOp)) {
        if (copyOp.getDirection() == CopyDirection::In &&
            containsGlobalMemref(copyOp.getMemrefs(), loadOp.getGlobalName()))
          return failure();
      }
    }

    return failure();
  }
};

struct RemoveDeadGlobalStore : public OpRewritePattern<GlobalStore> {
  using OpRewritePattern<GlobalStore>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStore storeOp,
                                 PatternRewriter &rewriter) const override {
    for (Operation *nextOp = storeOp->getNextNode(); nextOp;
         nextOp = nextOp->getNextNode()) {
      if (nextOp->getNumRegions() > 0)
        return failure();

      if (auto loadOp = dyn_cast<GlobalLoad>(nextOp)) {
        if (loadOp.getGlobalName() == storeOp.getGlobalName())
          return failure();
      }

      if (auto nextStoreOp = dyn_cast<GlobalStore>(nextOp)) {
        if (nextStoreOp.getGlobalName() == storeOp.getGlobalName()) {
          rewriter.eraseOp(storeOp);
          return success();
        }
      }
    }

    return failure();
  }
};
} // namespace

void aps::GlobalLoad::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
  results.add<FoldGlobalLoadAfterStore>(context);
}

void aps::GlobalStore::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.add<RemoveDeadGlobalStore>(context);
}

//===----------------------------------------------------------------------===//
// ReadSmem Canonicalization
//===----------------------------------------------------------------------===//

namespace {
static bool isSameMemref(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;

  auto lhsGlobal = lhs.getDefiningOp<memref::GetGlobalOp>();
  auto rhsGlobal = rhs.getDefiningOp<memref::GetGlobalOp>();
  return lhsGlobal && rhsGlobal && lhsGlobal.getName() == rhsGlobal.getName();
}

static bool containsMemref(ValueRange memrefs, Value memref) {
  return llvm::any_of(memrefs, [&](Value candidate) {
    return isSameMemref(candidate, memref);
  });
}

static bool isSameMemoryLocation(Value memref1, ValueRange indices1,
                                  Value memref2, ValueRange indices2) {
  return isSameMemref(memref1, memref2) && llvm::equal(indices1, indices2);
}

static bool mayReadMemref(Operation *op, Value memref) {
  if (auto loadOp = dyn_cast<ReadSmem>(op))
    return isSameMemref(loadOp.getMemref(), memref);
  if (auto copyOp = dyn_cast<Copy>(op))
    return copyOp.getDirection() == CopyDirection::Out &&
           containsMemref(copyOp.getMemrefs(), memref);
  return false;
}

static bool mayWriteMemref(Operation *op, Value memref) {
  if (auto storeOp = dyn_cast<WriteSmem>(op))
    return isSameMemref(storeOp.getMemref(), memref);
  if (auto copyOp = dyn_cast<Copy>(op))
    return copyOp.getDirection() == CopyDirection::In &&
           containsMemref(copyOp.getMemrefs(), memref);
  return false;
}

struct FoldMemLoadAfterStore : public OpRewritePattern<ReadSmem> {
  using OpRewritePattern<ReadSmem>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReadSmem loadOp,
                                 PatternRewriter &rewriter) const override {
    Value loadMemref = loadOp.getMemref();
    ValueRange loadIndices = loadOp.getIndices();

    for (Operation *prevOp = loadOp->getPrevNode(); prevOp;
         prevOp = prevOp->getPrevNode()) {
      if (prevOp->getNumRegions() > 0)
        return failure();

      if (auto storeOp = dyn_cast<WriteSmem>(prevOp)) {
        Value storeMemref = storeOp.getMemref();
        ValueRange storeIndices = storeOp.getIndices();

        if (isSameMemoryLocation(loadMemref, loadIndices, storeMemref,
                                  storeIndices)) {
          rewriter.replaceOp(loadOp, storeOp.getValue());
          return success();
        }

        if (isSameMemref(storeMemref, loadMemref))
          return failure();
      }

      if (mayWriteMemref(prevOp, loadMemref))
        return failure();
    }

    return failure();
  }
};

struct RemoveDeadMemStore : public OpRewritePattern<WriteSmem> {
  using OpRewritePattern<WriteSmem>::OpRewritePattern;

  LogicalResult matchAndRewrite(WriteSmem storeOp,
                                 PatternRewriter &rewriter) const override {
    Value storeMemref = storeOp.getMemref();
    ValueRange storeIndices = storeOp.getIndices();

    for (Operation *nextOp = storeOp->getNextNode(); nextOp;
         nextOp = nextOp->getNextNode()) {
      if (nextOp->getNumRegions() > 0)
        return failure();

      if (auto nextStoreOp = dyn_cast<WriteSmem>(nextOp)) {
        Value nextMemref = nextStoreOp.getMemref();
        ValueRange nextIndices = nextStoreOp.getIndices();

        if (isSameMemoryLocation(storeMemref, storeIndices, nextMemref,
                                  nextIndices)) {
          rewriter.eraseOp(storeOp);
          return success();
        }

        if (isSameMemref(nextMemref, storeMemref))
          return failure();
      }

      if (auto nextLoadOp = dyn_cast<ReadSmem>(nextOp)) {
        Value nextMemref = nextLoadOp.getMemref();
        ValueRange nextIndices = nextLoadOp.getIndices();

        if (isSameMemoryLocation(storeMemref, storeIndices, nextMemref,
                                  nextIndices))
          return failure();

        if (isSameMemref(nextMemref, storeMemref))
          return failure();
      }

      if (mayReadMemref(nextOp, storeMemref))
        return failure();
    }

    return failure();
  }
};
} // namespace

void aps::ReadSmem::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<FoldMemLoadAfterStore>(context);
}

void aps::WriteSmem::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<RemoveDeadMemStore>(context);
}

// Force template instantiation for TypeID
namespace mlir::detail {
template struct TypeIDResolver<aps::APSDialect, void>;
}
