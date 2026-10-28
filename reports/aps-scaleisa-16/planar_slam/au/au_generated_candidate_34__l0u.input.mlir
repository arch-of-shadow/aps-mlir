module {
  func.func @au_generated_candidate_34__l0u(%arg0: memref<16x3xf32>) attributes {au.generated = true} {
    affine.for %arg1 = 0 to 16 {
      %0 = arith.index_cast %arg1 : index to i32
      %1 = arith.sitofp %0 : i32 to f32
      %cst = arith.constant 7.500000e-01 : f32
      %2 = arith.mulf %1, %cst : f32
      %c3_i32 = arith.constant 3 : i32
      %3 = arith.remsi %0, %c3_i32 : i32
      %4 = arith.sitofp %3 : i32 to f32
      %cst_0 = arith.constant 5.000000e-02 : f32
      %5 = arith.mulf %4, %cst_0 : f32
      %6 = arith.addf %2, %5 : f32
      affine.store %6, %arg0[%arg1, 0] : memref<16x3xf32>
      %c2_i32 = arith.constant 2 : i32
      %7 = arith.remsi %0, %c2_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %cst_1 = arith.constant 1.800000e-01 : f32
      %9 = arith.mulf %8, %cst_1 : f32
      affine.store %9, %arg0[%arg1, 1] : memref<16x3xf32>
      %c5_i32 = arith.constant 5 : i32
      %10 = arith.remsi %0, %c5_i32 : i32
      %c-2_i32 = arith.constant -2 : i32
      %11 = arith.addi %10, %c-2_i32 : i32
      %12 = arith.sitofp %11 : i32 to f32
      %cst_2 = arith.constant 1.500000e-02 : f32
      %13 = arith.mulf %12, %cst_2 : f32
      affine.store %13, %arg0[%arg1, 2] : memref<16x3xf32>
    } {unroll = 0 : i64}
    return
  }
}

