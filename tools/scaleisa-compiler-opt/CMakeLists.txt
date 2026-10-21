add_executable(scaleisa-compiler-opt main.cpp)
target_compile_options(scaleisa-compiler-opt PRIVATE -fno-rtti)
target_link_libraries(scaleisa-compiler-opt PRIVATE ScaleISACompiler APS MLIROptLib
  MLIRAffineDialect MLIRArithDialect MLIRFuncDialect MLIRLLVMDialect
  MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect)

add_custom_target(check-scaleisa-compiler
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${PROJECT_SOURCE_DIR}/tests/scaleisa-compiler" -p "test_*.py" -v
  DEPENDS scaleisa-compiler-opt scaleisa-adapter-opt
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  USES_TERMINAL)
