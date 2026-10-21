add_mlir_library(ScaleISACompiler
  LowerCI.cpp
  LINK_LIBS PUBLIC
  APS MLIRAffineDialect MLIRArithDialect MLIRFuncDialect MLIRLLVMDialect
  MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect
  MLIRIR MLIRParser MLIRPass MLIRSupport
)
