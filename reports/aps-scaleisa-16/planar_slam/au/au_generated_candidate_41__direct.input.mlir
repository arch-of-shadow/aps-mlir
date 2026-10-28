module {
  func.func @au_generated_candidate_41__direct(%arg0: memref<20x2xf32>, %arg1: memref<20x2xf32>, %arg2: memref<20x2xf32>) attributes {au.generated = true} {
    affine.for %arg3 = 0 to 20 {
      %0 = affine.load %arg0[%arg3, 0] : memref<20x2xf32>
      %cst = arith.constant 2.500000e-01 : f32
      %1 = arith.mulf %0, %cst : f32
      %2 = affine.load %arg1[%arg3, 0] : memref<20x2xf32>
      %3 = arith.divf %1, %2 : f32
      %4 = affine.load %arg2[%arg3, 0] : memref<20x2xf32>
      %5 = arith.subf %4, %3 : f32
      affine.store %5, %arg2[%arg3, 0] : memref<20x2xf32>
      %6 = affine.load %arg0[%arg3, 1] : memref<20x2xf32>
      %7 = arith.mulf %6, %cst : f32
      %8 = affine.load %arg1[%arg3, 1] : memref<20x2xf32>
      %9 = arith.divf %7, %8 : f32
      %10 = affine.load %arg2[%arg3, 1] : memref<20x2xf32>
      %11 = arith.subf %10, %9 : f32
      affine.store %11, %arg2[%arg3, 1] : memref<20x2xf32>
    }
    return
  }
}

