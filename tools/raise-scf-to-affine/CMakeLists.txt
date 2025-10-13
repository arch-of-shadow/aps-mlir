add_llvm_executable(raise-scf-to-affine
  raise-scf-to-affine.cpp
)

target_link_libraries(raise-scf-to-affine PRIVATE
  APS
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTOR
  MLIRTransforms
)

llvm_update_compile_flags(raise-scf-to-affine)
