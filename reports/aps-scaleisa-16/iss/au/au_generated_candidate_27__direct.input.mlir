module {
  func.func @au_generated_candidate_27__direct(%arg0: memref<16x16xi32>, %arg1: memref<16x16xf32>, %arg2: memref<i32>) attributes {au.generated = true} {
    affine.for %arg3 = 0 to 16 {
      %0 = arith.index_cast %arg3 : index to i32
      affine.for %arg4 = 0 to 16 {
        %1 = arith.index_cast %arg4 : index to i32
        %2 = affine.load %arg0[%arg3, %arg4] : memref<16x16xi32>
        %c0_i32 = arith.constant 0 : i32
        %3 = arith.cmpi ne, %2, %c0_i32 : i32
        %4 = scf.if %3 -> (i32) {
          %7 = affine.load %arg1[%arg3, %arg4] : memref<16x16xf32>
          %cst = arith.constant 3.200000e+01 : f32
          %8 = arith.mulf %7, %cst : f32
          %9 = arith.fptosi %8 : f32 to i32
          %10 = arith.addi %9, %1 : i32
          %11 = arith.addi %10, %0 : i32
          scf.yield %11 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %5 = affine.load %arg2[] : memref<i32>
        %6 = arith.addi %5, %4 : i32
        affine.store %6, %arg2[] : memref<i32>
      }
    }
    return
  }
}

