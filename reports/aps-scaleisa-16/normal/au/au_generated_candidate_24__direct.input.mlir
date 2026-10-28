module {
  func.func @au_generated_candidate_24__direct(%arg0: memref<12x12x3xf32>, %arg1: memref<12x12xf32>, %arg2: memref<i32>) attributes {au.generated = true} {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %arg0[%arg3 + 1, %arg4 + 1, 0] : memref<12x12x3xf32>
        %cst = arith.constant 1.700000e+01 : f32
        %1 = arith.mulf %0, %cst : f32
        %2 = affine.load %arg0[%arg3 + 1, %arg4 + 1, 1] : memref<12x12x3xf32>
        %cst_0 = arith.constant 2.300000e+01 : f32
        %3 = arith.mulf %2, %cst_0 : f32
        %4 = arith.addf %1, %3 : f32
        %5 = affine.load %arg0[%arg3 + 1, %arg4 + 1, 2] : memref<12x12x3xf32>
        %cst_1 = arith.constant 3.100000e+01 : f32
        %6 = arith.mulf %5, %cst_1 : f32
        %7 = arith.addf %4, %6 : f32
        %8 = affine.load %arg1[%arg3 + 1, %arg4 + 1] : memref<12x12xf32>
        %cst_2 = arith.constant 1.100000e+01 : f32
        %9 = arith.mulf %8, %cst_2 : f32
        %10 = arith.addf %7, %9 : f32
        %cst_3 = arith.constant 6.400000e+01 : f32
        %11 = arith.mulf %10, %cst_3 : f32
        %12 = arith.fptosi %11 : f32 to i32
        %13 = affine.load %arg2[] : memref<i32>
        %14 = arith.addi %13, %12 : i32
        affine.store %14, %arg2[] : memref<i32>
      }
    }
    return
  }
}

