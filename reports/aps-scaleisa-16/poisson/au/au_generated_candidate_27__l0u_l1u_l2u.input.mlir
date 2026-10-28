#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @au_generated_candidate_27__l0u_l1u_l2u(%arg0: memref<216xf32>, %arg1: memref<216xf32>) attributes {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [0], partition_dim_array_1 = [0], partition_factor_array_0 = [4], partition_factor_array_1 = [4]} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.apply #map(%arg2)
      %1 = arith.index_cast %0 : index to i32
      affine.for %arg3 = 0 to 4 {
        %2 = affine.apply #map(%arg3)
        %3 = arith.index_cast %2 : index to i32
        affine.for %arg4 = 0 to 4 {
          %4 = affine.apply #map(%arg4)
          %5 = arith.index_cast %4 : index to i32
          %c6_i32 = arith.constant 6 : i32
          %6 = arith.muli %1, %c6_i32 : i32
          %7 = arith.addi %6, %3 : i32
          %8 = arith.muli %7, %c6_i32 : i32
          %9 = arith.addi %8, %5 : i32
          %10 = arith.index_cast %9 : i32 to index
          %11 = memref.load %arg0[%10] : memref<216xf32>
          memref.store %11, %arg1[%10] : memref<216xf32>
        } {unroll = 0 : i64}
      } {unroll = 0 : i64}
    } {unroll = 0 : i64}
    return
  }
}

