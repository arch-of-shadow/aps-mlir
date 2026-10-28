module {
  func.func @au_generated_candidate_8__l0u(%arg0: memref<32x3xf32>, %arg1: memref<32x3xf32>) attributes {au.generated = true} {
    affine.for %arg2 = 0 to 32 {
      %0 = arith.index_cast %arg2 : index to i32
      %c7_i32 = arith.constant 7 : i32
      %1 = arith.muli %0, %c7_i32 : i32
      %c41_i32 = arith.constant 41 : i32
      %2 = arith.remsi %1, %c41_i32 : i32
      %3 = arith.sitofp %2 : i32 to f32
      %cst = arith.constant 3.000000e-02 : f32
      %4 = arith.mulf %3, %cst : f32
      %c11_i32 = arith.constant 11 : i32
      %5 = arith.muli %0, %c11_i32 : i32
      %c37_i32 = arith.constant 37 : i32
      %6 = arith.remsi %5, %c37_i32 : i32
      %7 = arith.sitofp %6 : i32 to f32
      %cst_0 = arith.constant 2.500000e-02 : f32
      %8 = arith.mulf %7, %cst_0 : f32
      %c13_i32 = arith.constant 13 : i32
      %9 = arith.muli %0, %c13_i32 : i32
      %c31_i32 = arith.constant 31 : i32
      %10 = arith.remsi %9, %c31_i32 : i32
      %11 = arith.sitofp %10 : i32 to f32
      %cst_1 = arith.constant 2.000000e-02 : f32
      %12 = arith.mulf %11, %cst_1 : f32
      affine.store %4, %arg0[%arg2, 0] : memref<32x3xf32>
      affine.store %8, %arg0[%arg2, 1] : memref<32x3xf32>
      affine.store %12, %arg0[%arg2, 2] : memref<32x3xf32>
      %cst_2 = arith.constant 8.000000e-02 : f32
      %13 = arith.addf %4, %cst_2 : f32
      affine.store %13, %arg1[%arg2, 0] : memref<32x3xf32>
      %cst_3 = arith.constant 4.000000e-02 : f32
      %14 = arith.subf %8, %cst_3 : f32
      affine.store %14, %arg1[%arg2, 1] : memref<32x3xf32>
      %cst_4 = arith.constant 6.000000e-02 : f32
      %15 = arith.addf %12, %cst_4 : f32
      affine.store %15, %arg1[%arg2, 2] : memref<32x3xf32>
    } {unroll = 0 : i64}
    return
  }
}

