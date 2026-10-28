module {
  func.func @au_generated_candidate_44__l0u(%arg0: memref<8x3xf32>) attributes {au.generated = true} {
    affine.for %arg1 = 0 to 8 {
      %0 = arith.index_cast %arg1 : index to i32
      %1 = arith.sitofp %0 : i32 to f32
      %cst = arith.constant 5.500000e-01 : f32
      %2 = arith.mulf %1, %cst : f32
      %cst_0 = arith.constant -1.800000e+00 : f32
      %3 = arith.addf %2, %cst_0 : f32
      affine.store %3, %arg0[%arg1, 0] : memref<8x3xf32>
      %c3_i32 = arith.constant 3 : i32
      %4 = arith.remsi %0, %c3_i32 : i32
      %c-1_i32 = arith.constant -1 : i32
      %5 = arith.addi %4, %c-1_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %cst_1 = arith.constant 2.500000e-01 : f32
      %7 = arith.mulf %6, %cst_1 : f32
      affine.store %7, %arg0[%arg1, 1] : memref<8x3xf32>
      %c2_i32 = arith.constant 2 : i32
      %8 = arith.remsi %0, %c2_i32 : i32
      %9 = arith.sitofp %8 : i32 to f32
      %cst_2 = arith.constant 1.200000e-01 : f32
      %10 = arith.mulf %9, %cst_2 : f32
      affine.store %10, %arg0[%arg1, 2] : memref<8x3xf32>
    } {unroll = 0 : i64}
    return
  }
}

