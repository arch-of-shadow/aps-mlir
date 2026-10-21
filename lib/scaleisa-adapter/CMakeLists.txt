add_mlir_library(ScaleISAAdapter
  Convert.cpp
  Split.cpp
  InsertPragmas.cpp
  MaterializeAPS.cpp
  MemoryABI.cpp
  Pipelines.cpp
  DEPENDS ScaleISAAdapterPassIncGen
  LINK_LIBS PUBLIC
  APS MLIRAffineDialect MLIRAffineAnalysis MLIRArithDialect MLIRFuncDialect
  MLIRLinalgDialect MLIRLinalgTransforms MLIRMemRefDialect MLIRSCFDialect
  MLIRPass MLIRTransforms
)
