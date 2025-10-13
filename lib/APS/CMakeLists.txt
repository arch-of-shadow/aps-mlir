add_mlir_library(APS
  aps.cpp
  SCFForIndexCast.cpp
  APSMemToMemRef.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/APS

  DEPENDS
  MLIRAPSIncGen
  MLIRAPSPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
)

add_mlir_public_c_api_library(APSCAPI
  CAPI/APS.cpp
  aps.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR

  LINK_COMPONENTS
  Support
)