module {
  func.func @au_generated_candidate_38__l0u(%arg0: memref<24x3xf32>) attributes {au.generated = true} {
    affine.for %arg1 = 0 to 24 {
      %0 = arith.index_cast %arg1 : index to i32
      %c6_i32 = arith.constant 6 : i32
      %1 = arith.remsi %0, %c6_i32 : i32
      %2 = arith.sitofp %1 : i32 to f32
      %cst = arith.constant 4.500000e-01 : f32
      %3 = arith.mulf %2, %cst : f32
      %cst_0 = arith.constant -1.200000e+00 : f32
      %4 = arith.addf %3, %cst_0 : f32
      affine.store %4, %arg0[%arg1, 0] : memref<24x3xf32>
      %5 = arith.divsi %0, %c6_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %cst_1 = arith.constant 6.000000e-01 : f32
      %7 = arith.mulf %6, %cst_1 : f32
      %cst_2 = arith.constant -0.899999976 : f32
      %8 = arith.addf %7, %cst_2 : f32
      affine.store %8, %arg0[%arg1, 1] : memref<24x3xf32>
      %c5_i32 = arith.constant 5 : i32
      %9 = arith.remsi %0, %c5_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %cst_3 = arith.constant 1.800000e-01 : f32
      %11 = arith.mulf %10, %cst_3 : f32
      %cst_4 = arith.constant 2.800000e+00 : f32
      %12 = arith.addf %11, %cst_4 : f32
      affine.store %12, %arg0[%arg1, 2] : memref<24x3xf32>
    } {unroll = 0 : i64}
    return
  }
}

