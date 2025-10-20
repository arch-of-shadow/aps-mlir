add_mlir_library(APS
  aps.cpp
  SCFForIndexCast.cpp
  APSMemToMemRef.cpp
  AffineMemToAPSMem.cpp
  MemoryMapPass.cpp
  InferAffineMemAccess.cpp
  APSSplitMemoryOpsPass.cpp
  APSMemoryPoolGenPass.cpp
  Transforms/APSToStandard.cpp
  Transforms/CombExtractToArithTrunc.cpp
  Transforms/ArithSelectToSCFIf.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/APS

  DEPENDS
  MLIRAPSIncGen
  MLIRAPSPassIncGen
  MLIRTORIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  CIRCTECMT2
  CIRCTCmt2
  CIRCTFIRRTL
)

add_mlir_public_c_api_library(APSCAPI
  CAPI/APS.cpp
  aps.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR

  LINK_COMPONENTS
  Support
)
