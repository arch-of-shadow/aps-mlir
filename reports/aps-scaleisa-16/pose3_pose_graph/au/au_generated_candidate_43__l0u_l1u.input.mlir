module {
  func.func @au_generated_candidate_43__l0u_l1u(%arg0: memref<18x6xf32>, %arg1: memref<18x6xf32>) attributes {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [1], partition_dim_array_1 = [1], partition_factor_array_0 = [2], partition_factor_array_1 = [2]} {
    affine.for %arg2 = 0 to 18 {
      affine.for %arg3 = 0 to 6 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<18x6xf32>
        affine.store %0, %arg1[%arg2, %arg3] : memref<18x6xf32>
      } {unroll = 0 : i64}
    } {unroll = 0 : i64}
    return
  }
}

