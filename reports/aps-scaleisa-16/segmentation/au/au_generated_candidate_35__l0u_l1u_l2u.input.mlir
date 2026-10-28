module {
  func.func @au_generated_candidate_35__l0u_l1u_l2u(%arg0: memref<8x8xi32>, %arg1: memref<i32>) attributes {au.generated = true, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [8]} {
    affine.for %arg2 = 0 to 8 {
      affine.for %arg3 = 0 to 8 step 8 {
        affine.for %arg4 = 0 to 8 {
          %0 = affine.load %arg0[%arg2, %arg4] : memref<8x8xi32>
          %c255_i32 = arith.constant 255 : i32
          %1 = arith.andi %0, %c255_i32 : i32
          %2 = affine.load %arg1[] : memref<i32>
          %3 = arith.addi %2, %1 : i32
          affine.store %3, %arg1[] : memref<i32>
        } {unroll = 0 : i64}
      } {unroll = 0 : i64}
    } {unroll = 0 : i64}
    return
  }
}

