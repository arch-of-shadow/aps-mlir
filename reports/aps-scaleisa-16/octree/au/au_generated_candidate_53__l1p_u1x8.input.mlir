module {
  func.func @au_generated_candidate_53__l1p_u1x8(%arg0: memref<4x4096xi32>) attributes {au.generated = true, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [8]} {
    affine.for %arg1 = 0 to 4 {
      affine.for %arg2 = 0 to 4096 {
        %c0_i32 = arith.constant 0 : i32
        affine.store %c0_i32, %arg0[%arg1, %arg2] : memref<4x4096xi32>
      } {pipeline = 0 : i64, unroll = 8 : i64}
    }
    return
  }
}

