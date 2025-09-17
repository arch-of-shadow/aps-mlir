#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Casting.h"
#include <iomanip>

static inline mlir::Operation *getDefiningOpByValue(mlir::Value val) {
  if (auto blockArg = val.dyn_cast<mlir::BlockArgument>()) {
    return blockArg.getOwner()->getParentOp();
  }
  return val.getDefiningOp();
}

static inline mlir::Operation *getAncestorOp(mlir::Operation *op) {
  mlir::Operation *iterationOp = op;
  while (iterationOp && !mlir::isa<mlir::ModuleOp>(iterationOp))
    iterationOp = iterationOp->getParentOp();
  return iterationOp;
}

static inline unsigned getLineAttrInterger(mlir::Operation *op,
                                           std::string type) {
  if (!op->hasAttr(type + "-line")) {
    auto line = llvm::dyn_cast<mlir::FileLineColLoc>(op->getLoc()).getLine();
    return line;
  }
  return op->getAttr(type + "-line")
      .cast<mlir::IntegerAttr>()
      .getValue()
      .getSExtValue();
}

static inline void setPragmaStructureAttrInvalid(
    mlir::SmallVector<mlir::Attribute, 4> &pragmaStructureAttr,
    mlir::MLIRContext *ctx) {
  pragmaStructureAttr[1] =
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 2);
}

static inline void setPragmaStructureAttrValid(
    mlir::SmallVector<mlir::Attribute, 4> &pragmaStructureAttr,
    mlir::MLIRContext *ctx) {
  pragmaStructureAttr[1] =
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), 1);
}

static inline void setPragmaStructureAttrNewValue(
    mlir::SmallVector<mlir::Attribute, 4> &pragmaStructureAttr,
    mlir::MLIRContext *ctx, int newValue) {
  pragmaStructureAttr[3] =
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), newValue);
}

static inline void
setPragmaStructureAttrStatusByModuleOp(mlir::Operation *moduleOp,
                                       int lineNumber, bool isValid = true) {
  auto ctx = moduleOp->getContext();
  if (auto lineAttr = moduleOp->getAttr("hls.pragma_line_" +
                                        llvm::Twine(lineNumber).str())) {
    auto pragmaStructureAttr =
        llvm::to_vector<4>(lineAttr.dyn_cast<mlir::ArrayAttr>());
    if (isValid) {
      setPragmaStructureAttrValid(pragmaStructureAttr, ctx);
    } else {
      setPragmaStructureAttrInvalid(pragmaStructureAttr, ctx);
    }
    moduleOp->setAttr("hls.pragma_line_" + llvm::Twine(lineNumber).str(),
                      mlir::ArrayAttr::get(ctx, pragmaStructureAttr));
  }
}

static inline void setPragmaStructureAttrStatusByOp(mlir::Operation *op,
                                                    std::string type,
                                                    bool isValid = true) {
  auto *moduleOp = getAncestorOp(op);
  if (mlir::isa<mlir::ModuleOp>(moduleOp)) {
    int lineNumber = getLineAttrInterger(op, type);
    setPragmaStructureAttrStatusByModuleOp(moduleOp, lineNumber, isValid);
  }
}

static inline void setPragmaStructureAttrNewValueByOp(mlir::Operation *op,
                                                      std::string type,
                                                      int newValue) {
  setPragmaStructureAttrStatusByOp(op, type);
  auto *moduleOp = getAncestorOp(op);
  if (mlir::isa<mlir::ModuleOp>(moduleOp)) {
    auto ctx = moduleOp->getContext();
    int lineNumber = getLineAttrInterger(op, type);
    if (auto lineAttr = moduleOp->getAttr("hls.pragma_line_" +
                                          llvm::Twine(lineNumber).str())) {
      auto pragmaStructureAttr =
          llvm::to_vector<4>(lineAttr.dyn_cast<mlir::ArrayAttr>());
      setPragmaStructureAttrNewValue(pragmaStructureAttr, ctx, newValue);
      moduleOp->setAttr("hls.pragma_line_" + llvm::Twine(lineNumber).str(),
                        mlir::ArrayAttr::get(ctx, pragmaStructureAttr));
    }
  }
}

static inline void setPragmaStructureAttrStatusByValue(mlir::Value val,
                                                       std::string type,
                                                       bool isValid = true) {
  auto *op = getDefiningOpByValue(val);
  setPragmaStructureAttrStatusByOp(op, type, isValid);
}

template <typename T>
static inline std::string getFormatWidthStr(int width, char fill, T Value) {
  std::ostringstream ss;
  ss << std::setw(width) << std::setfill(fill) << Value;
  return ss.str();
}

template <typename T>
static inline std::string getFormatWidthLeftStr(int width, char fill, T Value) {
  std::ostringstream ss;
  ss << std::left << std::setw(width) << std::setfill(fill) << Value;
  return ss.str();
}

static inline std::string getStatusString(int status) {
  if (status == 1) {
    return "Valid";
  }
  if (status == 0) {
    return "Init";
  }
  if (status == 2) {
    return "Invalid";
  }
  return "";
}

static inline void printPragmaReport(mlir::ModuleOp moduleOp) {
  if (!moduleOp->hasAttr("hls.pragma_line_list")) {
    return;
  }
  llvm::outs() << getFormatWidthStr(110, '=', "") << "\n";
  llvm::outs() << "== " << "Pragma Report\n";
  llvm::outs() << getFormatWidthStr(110, '=', "") << "\n";
  llvm::outs() << "+" << getFormatWidthStr(108, '-', "") << "+\n";
  llvm::outs() << "|" << getFormatWidthStr(22, ' ', "Type") << "     |"
               << getFormatWidthStr(46, ' ', "Options") << "     |"
               << getFormatWidthStr(9, ' ', "Status") << "  |"
               << getFormatWidthStr(12, ' ', "Line number") << "    |\n";

  mlir::SmallVector<mlir::Attribute> pragmaLineListAttr = llvm::to_vector<4>(
      moduleOp->getAttr("hls.pragma_line_list").dyn_cast<mlir::ArrayAttr>());
  auto cmp = [&](mlir::Attribute a, mlir::Attribute b) {
    return a.cast<mlir::IntegerAttr>().getValue().getSExtValue() <
           b.cast<mlir::IntegerAttr>().getValue().getSExtValue();
  };
  std::stable_sort(pragmaLineListAttr.begin(), pragmaLineListAttr.end(), cmp);

  for (auto lineAttr : pragmaLineListAttr) {
    int lineNumber =
        lineAttr.cast<mlir::IntegerAttr>().getValue().getSExtValue();
    auto pragmaStructureAttr =
        moduleOp->getAttr("hls.pragma_line_" + llvm::Twine(lineNumber).str())
            .dyn_cast<mlir::ArrayAttr>();
    std::string type =
        pragmaStructureAttr[0].cast<mlir::StringAttr>().getValue().str();
    std::string option =
        pragmaStructureAttr[4].cast<mlir::StringAttr>().getValue().str();
    int oldValue = pragmaStructureAttr[2]
                       .cast<mlir::IntegerAttr>()
                       .getValue()
                       .getSExtValue();
    int newValue = pragmaStructureAttr[3]
                       .cast<mlir::IntegerAttr>()
                       .getValue()
                       .getSExtValue();
    auto statusStr = getStatusString(pragmaStructureAttr[1]
                                         .cast<mlir::IntegerAttr>()
                                         .getValue()
                                         .getSExtValue());
    if ((type == "pipeline" || type == "unroll") && oldValue != newValue &&
        statusStr == "Valid") {
      llvm::outs() << "|" << getFormatWidthStr(22, ' ', type) << "     |"
                   << getFormatWidthStr(23, ' ', option)
                   << getFormatWidthLeftStr(23, ' ',
                                            ", effective value = " +
                                                llvm::Twine(newValue).str())
                   << "     |" << getFormatWidthStr(9, ' ', statusStr) << "  |"
                   << getFormatWidthStr(12, ' ', lineNumber) << "    |\n";
    } else {
      llvm::outs() << "|" << getFormatWidthStr(22, ' ', type) << "     |"
                   << getFormatWidthStr(46, ' ', option) << "     |"
                   << getFormatWidthStr(9, ' ', statusStr) << "  |"
                   << getFormatWidthStr(12, ' ', lineNumber) << "    |\n";
    }
  }
  llvm::outs() << "+" << getFormatWidthStr(108, '-', "") << "+\n\n";
}
