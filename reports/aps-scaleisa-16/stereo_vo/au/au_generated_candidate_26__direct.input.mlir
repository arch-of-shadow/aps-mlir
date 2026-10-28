module {
  func.func @au_generated_candidate_26__direct(%arg0: memref<32x3xf32>, %arg1: memref<i32>) attributes {au.generated = true} {
    affine.for %arg2 = 0 to 32 step 32 {
      affine.for %arg3 = 0 to 32 {
        %0 = affine.load %arg0[%arg3, 0] : memref<32x3xf32>
        %cst = arith.constant 7.900000e+01 : f32
        %1 = arith.mulf %0, %cst : f32
        %2 = arith.fptosi %1 : f32 to i32
        %3 = affine.load %arg1[] : memref<i32>
        %4 = arith.addi %3, %2 : i32
        affine.store %4, %arg1[] : memref<i32>
        %5 = affine.load %arg0[%arg3, 1] : memref<32x3xf32>
        %cst_0 = arith.constant 1.030000e+02 : f32
        %6 = arith.mulf %5, %cst_0 : f32
        %7 = arith.fptosi %6 : f32 to i32
        %8 = arith.addi %4, %7 : i32
        affine.store %8, %arg1[] : memref<i32>
        %9 = affine.load %arg0[%arg3, 2] : memref<32x3xf32>
        %cst_1 = arith.constant 1.270000e+02 : f32
        %10 = arith.mulf %9, %cst_1 : f32
        %11 = arith.fptosi %10 : f32 to i32
        %12 = arith.addi %8, %11 : i32
        affine.store %12, %arg1[] : memref<i32>
      }
    }
    return
  }
}

