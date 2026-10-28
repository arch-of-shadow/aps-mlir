#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (4096)>
#map2 = affine_map<() -> (4)>
"builtin.module"() ({
  "func.func"() <{function_type = (memref<4x4096xi32>) -> (), sym_name = "au_generated_candidate_53__l1p_u1x8"}> ({
  ^bb0(%arg0: memref<4x4096xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg1: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg2: index):
        %1 = "arith.index_cast"(%arg1) : (index) -> i32
        %2 = "arith.index_cast"(%arg2) : (index) -> i32
        "aps.write_smem"(%0, %arg0, %1, %2) : (i32, memref<4x4096xi32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {pipeline = 1 : i64, unroll = 8 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [8], scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 1 : i32} : () -> ()
}) : () -> ()

