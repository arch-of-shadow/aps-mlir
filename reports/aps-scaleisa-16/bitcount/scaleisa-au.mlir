#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() <{function_type = (i32) -> i32, sym_name = "au_generated_candidate_9__l0u"}> ({
  ^bb0(%arg0: i32):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "affine.for"(%0) <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 1>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg1: index, %arg2: i32):
      %2 = "arith.index_cast"(%arg1) : (index) -> i32
      %3 = "arith.shrui"(%arg0, %2) : (i32, i32) -> i32
      %4 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %5 = "arith.andi"(%3, %4) : (i32, i32) -> i32
      %6 = "arith.addi"(%arg2, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.yield"(%6) : (i32) -> ()
    }) {unroll = 0 : i64} : (i32) -> i32
    "func.return"(%1) : (i32) -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
