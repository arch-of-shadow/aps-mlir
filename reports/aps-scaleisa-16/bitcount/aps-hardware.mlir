#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_9__l0u"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "affine.for"(%2) <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 1>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg3: index, %arg4: i32):
      %4 = "arith.index_cast"(%arg3) : (index) -> i32
      %5 = "arith.shrui"(%0, %4) : (i32, i32) -> i32
      %6 = "arith.andi"(%5, %1) : (i32, i32) -> i32
      %7 = "arith.addi"(%arg4, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.yield"(%7) : (i32) -> ()
    }) {unroll = 0 : i64} : (i32) -> i32
    "aps.write_irf"(%arg2, %3) : (i5, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 6 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1"}], scaleisa.source_type = (i32) -> i32} : () -> ()
}) : () -> ()

