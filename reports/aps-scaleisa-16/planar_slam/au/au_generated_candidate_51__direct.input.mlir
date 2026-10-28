module {
  func.func @au_generated_candidate_51__direct(%arg0: memref<20x2xf32>, %arg1: memref<i32>) attributes {au.generated = true} {
    affine.for %arg2 = 0 to 20 {
      %0 = affine.load %arg0[%arg2, 0] : memref<20x2xf32>
      %cst = arith.constant 1.510000e+02 : f32
      %1 = arith.mulf %0, %cst : f32
      %2 = arith.fptosi %1 : f32 to i32
      %3 = affine.load %arg1[] : memref<i32>
      %4 = arith.addi %3, %2 : i32
      affine.store %4, %arg1[] : memref<i32>
      %5 = affine.load %arg0[%arg2, 1] : memref<20x2xf32>
      %cst_0 = arith.constant 1.810000e+02 : f32
      %6 = arith.mulf %5, %cst_0 : f32
      %7 = arith.fptosi %6 : f32 to i32
      %8 = arith.addi %4, %7 : i32
      affine.store %8, %arg1[] : memref<i32>
    }
    return
  }
}

