# ./build/tools/aps-opt/aps-opt scalar.mlir --aps-to-cmt2-gen --mlir-print-ir-after-failure | sed -n '/^module {$/,/^}$/p' | sed '/tor.design @flow_scalar_math {/,/^  }/d' > scalar_cmt.mlir
# ./circt/build/bin/circt-opt scalar_cmt.mlir --lower-cmt2-to-firrtl --mlir-print-ir-after-failure | ./circt/build/bin/firtool -format=mlir -o scalar.sv

./build/tools/aps-opt/aps-opt rocc_mem_smoke.mlir --aps-to-cmt2-gen --mlir-print-ir-after-failure | sed -n '/^module {$/,/^}$/p' | sed '/tor.design @aps_isaxes {/,/^  }/d' > rocc_mem_smoke_cmt.mlir
./circt/build/bin/circt-opt rocc_mem_smoke_cmt.mlir --lower-cmt2-to-firrtl --mlir-print-ir-after-failure | ./circt/build/bin/firtool -format=mlir -o rocc_mem_smoke_cmt.sv