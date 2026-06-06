add_mlir_library(APS
  aps.cpp
  NormalizeSCFForIndices.cpp
  APSRaiseSCFToAffine.cpp
  APSMemToMemRef.cpp
  MemoryIndexUtils.cpp
  AffineMemToAPS.cpp
  MemRefToAPS.cpp
  AutoBurstPartition.cpp
  APSMemoryMap.cpp
  RaiseMemRefToAffine.cpp
  DuplicateMemLoads.cpp
  LowerAPSMemToReqCollect.cpp
  PlaceReadRFAtEntry.cpp
  PromoteSingletonMemRefToGlobal.cpp
  ArithMulDivToShift.cpp
  APSToCMT2/APSToCMT2.cpp
  APSToCMT2/OpRuleGenerator.cpp
  APSToCMT2/BlockHandler.cpp
  APSToCMT2/BBHandler.cpp
  APSToCMT2/LoopHandler.cpp
  APSToCMT2/IfHandler.cpp
  APSToCMT2/ArithmeticOpGenerator.cpp
  APSToCMT2/RegisterOpGenerator.cpp
  APSToCMT2/MemoryOpGenerator.cpp
  APSToCMT2/InterfaceOpGenerator.cpp
  APSToCMT2/InterfaceGenerator.cpp
  APSToCMT2/MemoryGenerator.cpp
  APSToCMT2/GlobalRegisterGenerator.cpp

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
