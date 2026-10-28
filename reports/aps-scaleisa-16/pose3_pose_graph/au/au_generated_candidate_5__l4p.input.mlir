module {
  func.func @au_generated_candidate_5__l4p(%arg0: memref<18x6xf32>, %arg1: memref<18x6xf32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<18x6xf32>, %arg5: memref<30x6xf32>, %arg6: memref<30x6xf32>) attributes {au.generated = true} {
    affine.for %arg7 = 0 to 7 {
      affine.for %arg8 = 0 to 6 {
        affine.for %arg9 = 0 to 18 {
          %cst = arith.constant 0.000000e+00 : f32
          affine.store %cst, %arg0[%arg9, %arg8] : memref<18x6xf32>
          %cst_0 = arith.constant 1.000000e-03 : f32
          affine.store %cst_0, %arg1[%arg9, %arg8] : memref<18x6xf32>
        }
      }
      affine.for %arg8 = 0 to 30 {
        %0 = affine.load %arg2[%arg8] : memref<30xi32>
        %1 = affine.load %arg3[%arg8] : memref<30xi32>
        affine.for %arg9 = 0 to 6 {
          %2 = arith.index_cast %1 : i32 to index
          %3 = memref.load %arg4[%2, %arg9] : memref<18x6xf32>
          %4 = arith.index_cast %0 : i32 to index
          %5 = memref.load %arg4[%4, %arg9] : memref<18x6xf32>
          %6 = arith.subf %3, %5 : f32
          %7 = affine.load %arg5[%arg8, %arg9] : memref<30x6xf32>
          %8 = arith.subf %6, %7 : f32
          %9 = affine.load %arg6[%arg8, %arg9] : memref<30x6xf32>
          %10 = arith.mulf %9, %8 : f32
          %11 = memref.load %arg0[%4, %arg9] : memref<18x6xf32>
          %12 = arith.addf %11, %10 : f32
          memref.store %12, %arg0[%4, %arg9] : memref<18x6xf32>
          %13 = memref.load %arg0[%2, %arg9] : memref<18x6xf32>
          %14 = arith.subf %13, %10 : f32
          memref.store %14, %arg0[%2, %arg9] : memref<18x6xf32>
          %15 = affine.load %arg6[%arg8, %arg9] : memref<30x6xf32>
          %16 = memref.load %arg1[%4, %arg9] : memref<18x6xf32>
          %17 = arith.addf %16, %15 : f32
          memref.store %17, %arg1[%4, %arg9] : memref<18x6xf32>
          %18 = affine.load %arg6[%arg8, %arg9] : memref<30x6xf32>
          %19 = memref.load %arg1[%2, %arg9] : memref<18x6xf32>
          %20 = arith.addf %19, %18 : f32
          memref.store %20, %arg1[%2, %arg9] : memref<18x6xf32>
        } {pipeline = 0 : i64}
      }
      affine.for %arg8 = 0 to 6 {
        %cst = arith.constant 0.000000e+00 : f32
        affine.store %cst, %arg4[0, %arg8] : memref<18x6xf32>
      }
      affine.for %arg8 = 0 to 17 {
        affine.for %arg9 = 0 to 6 {
          %0 = arith.index_cast %arg9 : index to i32
          %c3_i32 = arith.constant 3 : i32
          %1 = arith.cmpi slt, %0, %c3_i32 : i32
          %cst = arith.constant 2.000000e-01 : f32
          %cst_0 = arith.constant 1.200000e-01 : f32
          %2 = arith.select %1, %cst, %cst_0 : f32
          %3 = affine.load %arg0[%arg8 + 1, %arg9] : memref<18x6xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = affine.load %arg1[%arg8 + 1, %arg9] : memref<18x6xf32>
          %6 = arith.divf %4, %5 : f32
          %7 = affine.load %arg4[%arg8 + 1, %arg9] : memref<18x6xf32>
          %8 = arith.subf %7, %6 : f32
          affine.store %8, %arg4[%arg8 + 1, %arg9] : memref<18x6xf32>
        }
      }
    }
    return
  }
}

