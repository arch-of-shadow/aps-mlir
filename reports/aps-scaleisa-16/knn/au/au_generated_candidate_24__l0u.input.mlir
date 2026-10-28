module {
  func.func @au_generated_candidate_24__l0u(%arg0: memref<32x3xf32>) attributes {au.generated = true} {
    affine.for %arg1 = 0 to 32 {
      %0 = arith.index_cast %arg1 : index to i32
      %c11_i32 = arith.constant 11 : i32
      %1 = arith.muli %0, %c11_i32 : i32
      %c3_i32 = arith.constant 3 : i32
      %2 = arith.addi %1, %c3_i32 : i32
      %c101_i32 = arith.constant 101 : i32
      %3 = arith.remsi %2, %c101_i32 : i32
      %4 = arith.sitofp %3 : i32 to f32
      %cst = arith.constant 3.100000e-02 : f32
      %5 = arith.mulf %4, %cst : f32
      affine.store %5, %arg0[%arg1, 0] : memref<32x3xf32>
      %c17_i32 = arith.constant 17 : i32
      %6 = arith.muli %0, %c17_i32 : i32
      %c5_i32 = arith.constant 5 : i32
      %7 = arith.addi %6, %c5_i32 : i32
      %c97_i32 = arith.constant 97 : i32
      %8 = arith.remsi %7, %c97_i32 : i32
      %9 = arith.sitofp %8 : i32 to f32
      %cst_0 = arith.constant 2.900000e-02 : f32
      %10 = arith.mulf %9, %cst_0 : f32
      affine.store %10, %arg0[%arg1, 1] : memref<32x3xf32>
      %c19_i32 = arith.constant 19 : i32
      %11 = arith.muli %0, %c19_i32 : i32
      %c7_i32 = arith.constant 7 : i32
      %12 = arith.addi %11, %c7_i32 : i32
      %c89_i32 = arith.constant 89 : i32
      %13 = arith.remsi %12, %c89_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %cst_1 = arith.constant 2.700000e-02 : f32
      %15 = arith.mulf %14, %cst_1 : f32
      affine.store %15, %arg0[%arg1, 2] : memref<32x3xf32>
    } {unroll = 0 : i64}
    return
  }
}

