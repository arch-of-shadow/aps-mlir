#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (4096)>
#map3 = affine_map<() -> (4)>
"builtin.module"() ({
  "func.func"() <{function_type = (memref<4x4096xi32>) -> (), sym_name = "au_generated_candidate_53__l1p_u1x8"}> ({
  ^bb0(%arg0: memref<4x4096xi32>):
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg1: index):
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg2: index):
        %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        "affine.store"(%0, %arg0, %arg1, %arg2) <{map = #map}> : (i32, memref<4x4096xi32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {pipeline = 0 : i64, unroll = 8 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [8]} : () -> ()
}) : () -> ()
