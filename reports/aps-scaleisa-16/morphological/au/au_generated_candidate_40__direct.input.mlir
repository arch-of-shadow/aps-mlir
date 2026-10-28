module {
  func.func @au_generated_candidate_40__direct(%arg0: memref<12x12xf32>, %arg1: memref<i32>) attributes {au.generated = true} {
    affine.for %arg2 = 0 to 12 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 12 {
        %1 = arith.index_cast %arg3 : index to i32
        %2 = affine.load %arg0[%arg2, %arg3] : memref<12x12xf32>
        %cst = arith.constant 1.280000e+02 : f32
        %3 = arith.mulf %2, %cst : f32
        %4 = arith.fptosi %3 : f32 to i32
        %5 = arith.addi %1, %0 : i32
        %c3_i32 = arith.constant 3 : i32
        %6 = arith.andi %5, %c3_i32 : i32
        %7 = arith.addi %4, %6 : i32
        %8 = affine.load %arg1[] : memref<i32>
        %9 = arith.addi %8, %7 : i32
        affine.store %9, %arg1[] : memref<i32>
      }
    }
    return
  }
}

