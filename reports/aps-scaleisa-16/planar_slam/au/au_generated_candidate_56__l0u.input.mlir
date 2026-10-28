module {
  func.func @au_generated_candidate_56__l0u(%arg0: memref<20x2xf32>) attributes {au.generated = true, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [2]} {
    affine.for %arg1 = 0 to 20 {
      %0 = arith.index_cast %arg1 : index to i32
      %c5_i32 = arith.constant 5 : i32
      %1 = arith.remsi %0, %c5_i32 : i32
      %2 = arith.sitofp %1 : i32 to f32
      %cst = arith.constant 3.800000e-01 : f32
      %3 = arith.mulf %2, %cst : f32
      %cst_0 = arith.constant 1.600000e+00 : f32
      %4 = arith.addf %3, %cst_0 : f32
      affine.store %4, %arg0[%arg1, 0] : memref<20x2xf32>
      %5 = arith.divsi %0, %c5_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %cst_1 = arith.constant 2.700000e-01 : f32
      %7 = arith.mulf %6, %cst_1 : f32
      %cst_2 = arith.constant 1.200000e+00 : f32
      %8 = arith.addf %7, %cst_2 : f32
      affine.store %8, %arg0[%arg1, 1] : memref<20x2xf32>
    } {unroll = 0 : i64}
    return
  }
}

