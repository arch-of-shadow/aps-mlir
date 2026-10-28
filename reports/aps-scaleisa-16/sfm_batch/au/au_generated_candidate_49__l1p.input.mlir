module {
  func.func @au_generated_candidate_49__l1p(%arg0: memref<8x3xf32>, %arg1: memref<8x3xf32>, %arg2: memref<8x3xf32>) attributes {au.generated = true} {
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 3 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<8x3xf32>
        %cst = arith.constant 1.800000e-01 : f32
        %1 = arith.mulf %0, %cst : f32
        %2 = affine.load %arg1[%arg3, %arg4] : memref<8x3xf32>
        %3 = arith.divf %1, %2 : f32
        %4 = affine.load %arg2[%arg3, %arg4] : memref<8x3xf32>
        %5 = arith.subf %4, %3 : f32
        affine.store %5, %arg2[%arg3, %arg4] : memref<8x3xf32>
      } {pipeline = 0 : i64}
    }
    return
  }
}

