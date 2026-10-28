module {
  func.func @au_generated_candidate_32__direct(%arg0: memref<18x6xf32>, %arg1: memref<i32>) attributes {au.generated = true} {
    affine.for %arg2 = 0 to 18 {
      affine.for %arg3 = 0 to 6 {
        %0 = arith.index_cast %arg3 : index to i32
        %1 = affine.load %arg0[%arg2, %arg3] : memref<18x6xf32>
        %2 = arith.sitofp %0 : i32 to f32
        %cst = arith.constant 1.900000e+01 : f32
        %3 = arith.mulf %2, %cst : f32
        %cst_0 = arith.constant 8.300000e+01 : f32
        %4 = arith.addf %3, %cst_0 : f32
        %5 = arith.mulf %1, %4 : f32
        %6 = arith.fptosi %5 : f32 to i32
        %7 = affine.load %arg1[] : memref<i32>
        %8 = arith.addi %6, %7 : i32
        affine.store %8, %arg1[] : memref<i32>
      }
    }
    return
  }
}

