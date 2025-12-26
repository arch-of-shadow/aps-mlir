# pixi run opt examples/floatv2/test_float.cadl /tmp/1/test_float.mlir && pixi run sv /tmp/1/test_float.mlir /home/xys/aps-mlir/yosys/sv/test_float.sv
# pixi run opt examples/floatv2/test_float_comprehensive.cadl /tmp/1/test_float_comprehensive.mlir && pixi run sv /tmp/1/test_float_comprehensive.mlir /home/xys/aps-mlir/yosys/sv/test_float_comprehensive.sv
pixi run opt examples/floatv2/softmax.cadl /tmp/1/softmax.mlir && pixi run sv /tmp/1/softmax.mlir /home/xys/aps-mlir/yosys/sv/softmax.sv
pixi run opt examples/floatv2/sigmoid.cadl /tmp/1/sigmoid.mlir && pixi run sv /tmp/1/sigmoid.mlir /home/xys/aps-mlir/yosys/sv/sigmoid.sv
pixi run opt examples/floatv2/softmax_gemm.cadl /tmp/1/softmax_gemm.mlir && pixi run sv /tmp/1/softmax_gemm.mlir /home/xys/aps-mlir/yosys/sv/softmax_gemm.sv
