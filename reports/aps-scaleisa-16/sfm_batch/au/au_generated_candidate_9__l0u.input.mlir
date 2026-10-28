module {
  func.func @au_generated_candidate_9__l0u(%arg0: memref<8x3xf32>) attributes {au.generated = true} {
    affine.for %arg1 = 0 to 8 {
      %0 = arith.index_cast %arg1 : index to i32
      %c4_i32 = arith.constant 4 : i32
      %1 = arith.remsi %0, %c4_i32 : i32
      %c-1_i32 = arith.constant -1 : i32
      %2 = arith.addi %1, %c-1_i32 : i32
      %3 = arith.sitofp %2 : i32 to f32
      %cst = arith.constant 1.200000e-01 : f32
      %4 = arith.mulf %3, %cst : f32
      %5 = affine.load %arg0[%arg1, 0] : memref<8x3xf32>
      %6 = arith.addf %5, %4 : f32
      affine.store %6, %arg0[%arg1, 0] : memref<8x3xf32>
      %c2_i32 = arith.constant 2 : i32
      %7 = arith.remsi %0, %c2_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %cst_0 = arith.constant 8.000000e-02 : f32
      %9 = arith.mulf %8, %cst_0 : f32
      %10 = affine.load %arg0[%arg1, 1] : memref<8x3xf32>
      %11 = arith.subf %10, %9 : f32
      affine.store %11, %arg0[%arg1, 1] : memref<8x3xf32>
      %c3_i32 = arith.constant 3 : i32
      %12 = arith.remsi %0, %c3_i32 : i32
      %13 = arith.addi %12, %c-1_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %cst_1 = arith.constant 4.000000e-02 : f32
      %15 = arith.mulf %14, %cst_1 : f32
      %16 = affine.load %arg0[%arg1, 2] : memref<8x3xf32>
      %17 = arith.addf %16, %15 : f32
      affine.store %17, %arg0[%arg1, 2] : memref<8x3xf32>
    } {unroll = 0 : i64}
    return
  }
}

