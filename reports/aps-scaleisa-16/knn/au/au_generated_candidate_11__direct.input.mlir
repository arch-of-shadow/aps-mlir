module {
  func.func @au_generated_candidate_11__direct(%arg0: memref<4x8xi32>, %arg1: memref<4x8xf32>, %arg2: memref<i32>) attributes {au.generated = true} {
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 8 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<4x8xi32>
        %c0_i32 = arith.constant 0 : i32
        %1 = arith.cmpi sge, %0, %c0_i32 : i32
        %2 = scf.if %1 -> (i32) {
          %5 = affine.load %arg0[%arg3, %arg4] : memref<4x8xi32>
          %6 = affine.load %arg1[%arg3, %arg4] : memref<4x8xf32>
          %cst = arith.constant 2.560000e+02 : f32
          %7 = arith.mulf %6, %cst : f32
          %8 = arith.fptosi %7 : f32 to i32
          %9 = arith.addi %5, %8 : i32
          scf.yield %9 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %3 = affine.load %arg2[] : memref<i32>
        %4 = arith.addi %3, %2 : i32
        affine.store %4, %arg2[] : memref<i32>
      }
    }
    return
  }
}

