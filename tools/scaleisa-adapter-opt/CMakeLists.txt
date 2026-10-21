add_executable(scaleisa-adapter-opt main.cpp)
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
