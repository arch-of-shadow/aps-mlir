module {
  func.func @au_generated_candidate_17__direct(%arg0: memref<32x3xf32>, %arg1: memref<32x3xf32>, %arg2: memref<32x3xf32>) attributes {au.generated = true} {
    affine.for %arg3 = 0 to 32 {
      %0 = affine.load %arg0[%arg3, 0] : memref<32x3xf32>
      %cst = arith.constant 1.600000e-01 : f32
      %1 = arith.mulf %0, %cst : f32
      %2 = affine.load %arg1[%arg3, 0] : memref<32x3xf32>
      %3 = arith.divf %1, %2 : f32
      %4 = affine.load %arg2[%arg3, 0] : memref<32x3xf32>
      %5 = arith.subf %4, %3 : f32
      affine.store %5, %arg2[%arg3, 0] : memref<32x3xf32>
      %6 = affine.load %arg0[%arg3, 1] : memref<32x3xf32>
      %7 = arith.mulf %6, %cst : f32
      %8 = affine.load %arg1[%arg3, 1] : memref<32x3xf32>
      %9 = arith.divf %7, %8 : f32
      %10 = affine.load %arg2[%arg3, 1] : memref<32x3xf32>
      %11 = arith.subf %10, %9 : f32
      affine.store %11, %arg2[%arg3, 1] : memref<32x3xf32>
      %12 = affine.load %arg0[%arg3, 2] : memref<32x3xf32>
      %cst_0 = arith.constant 1.200000e-01 : f32
      %13 = arith.mulf %12, %cst_0 : f32
      %14 = affine.load %arg1[%arg3, 2] : memref<32x3xf32>
      %15 = arith.divf %13, %14 : f32
      %16 = affine.load %arg2[%arg3, 2] : memref<32x3xf32>
      %17 = arith.subf %16, %15 : f32
      affine.store %17, %arg2[%arg3, 2] : memref<32x3xf32>
    }
    return
  }
}

