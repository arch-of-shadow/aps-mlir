add_llvm_executable(aps-opt
  aps-opt.cpp
)

target_link_libraries(aps-opt PRIVATE
  APS

  ${LP_SOLVE_LIB}
  MLIRTOR
  MLIROptLib
  MLIRSCFTransforms
  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenMPDialect
  MLIRGPUTransforms
  MLIRTransforms
  MLIRSCFToControlFlow
  MLIRFuncToLLVM

  MLIRAffineTransforms
  MLIRAffineToStandard
  MLIRMathToLLVM
  MLIRTargetLLVMIRImport
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToOpenMP
  MLIROpenMPToLLVM
  MLIROpenMPToLLVMIRTranslation
  MLIRGPUToNVVMTransforms
  MLIRGPUToGPURuntimeTransforms
  MLIRFuncAllExtensions
  MLIRLLVMIRToLLVMTranslation
  MLIRFromLLVMIRTranslationRegistration
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRCAPIRegisterEverything
  MLIRBuiltinToLLVMIRTranslation
)

llvm_update_compile_flags(aps-opt)
