add_executable(scaleisa-adapter-opt scaleisa-adapter-opt.cpp)
target_compile_options(scaleisa-adapter-opt PRIVATE -fno-rtti)
target_link_libraries(scaleisa-adapter-opt PRIVATE
  ScaleISAAdapter MLIROptLib MLIRMathDialect MLIRTensorDialect MLIRLLVMDialect
)

add_custom_target(check-scaleisa-adapter
  COMMAND "${CMAKE_COMMAND}" -E env
    "SCALEISA_ADAPTER_OPT=$<TARGET_FILE:scaleisa-adapter-opt>"
    "${Python3_EXECUTABLE}" -m unittest discover
    -s "${PROJECT_SOURCE_DIR}/tests/scaleisa-adapter" -p "test_*.py" -v
  DEPENDS scaleisa-adapter-opt aps-e2e
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  USES_TERMINAL
)

add_executable(scaleisa-compiler-opt scaleisa-compiler-opt.cpp)
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

add_executable(aps-scaleisa-opt main.cpp)
target_compile_options(aps-scaleisa-opt PRIVATE -fno-rtti)
target_compile_definitions(aps-scaleisa-opt PRIVATE
  APS_SOURCE_ROOT="${PROJECT_SOURCE_DIR}" APS_BINARY_ROOT="${PROJECT_BINARY_DIR}")
target_link_libraries(aps-scaleisa-opt PRIVATE APS MLIRTOR MLIRParser MLIRIR MLIRSupport
  MLIRAffineDialect MLIRArithDialect MLIRFuncDialect MLIRLLVMDialect
  MLIRLinalgDialect MLIRMathDialect MLIRMemRefDialect MLIRSCFDialect MLIRTensorDialect
  nlohmann_json::nlohmann_json)
add_dependencies(aps-scaleisa-opt scaleisa-adapter-opt scaleisa-compiler-opt aps-e2e scaleisa-opt)
add_dependencies(check-scaleisa-compiler aps-scaleisa-opt)
add_custom_target(check-aps-scaleisa
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${PROJECT_SOURCE_DIR}/tests/aps-scaleisa" -p "test_*.py" -v
  DEPENDS aps-scaleisa-opt
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" USES_TERMINAL)
