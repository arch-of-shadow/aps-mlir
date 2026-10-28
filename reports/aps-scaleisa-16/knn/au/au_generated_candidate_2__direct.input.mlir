module {
  func.func @au_generated_candidate_2__direct(%arg0: memref<4x8xf32>, %arg1: memref<4x8xi32>, %arg2: memref<32x3xf32>, %arg3: memref<4x3xf32>, %arg4: memref<8xf32>, %arg5: memref<8xi32>) attributes {au.generated = true} {
    affine.for %arg6 = 0 to 4 {
      affine.for %arg7 = 0 to 8 step 8 {
        affine.for %arg8 = 0 to 8 {
          %cst = arith.constant 1.000000e+30 : f32
          affine.store %cst, %arg0[%arg6, %arg8] : memref<4x8xf32>
          %c-1_i32 = arith.constant -1 : i32
          affine.store %c-1_i32, %arg1[%arg6, %arg8] : memref<4x8xi32>
        }
      }
      affine.for %arg7 = 0 to 32 {
        %0 = arith.index_cast %arg7 : index to i32
        %1 = affine.load %arg2[%arg7, 0] : memref<32x3xf32>
        %2 = affine.load %arg3[%arg6, 0] : memref<4x3xf32>
        %3 = arith.subf %1, %2 : f32
        %4 = affine.load %arg2[%arg7, 1] : memref<32x3xf32>
        %5 = affine.load %arg3[%arg6, 1] : memref<4x3xf32>
        %6 = arith.subf %4, %5 : f32
        %7 = affine.load %arg2[%arg7, 2] : memref<32x3xf32>
        %8 = affine.load %arg3[%arg6, 2] : memref<4x3xf32>
        %9 = arith.subf %7, %8 : f32
        %10 = arith.mulf %3, %3 : f32
        %11 = arith.mulf %6, %6 : f32
        %12 = arith.addf %10, %11 : f32
        %13 = arith.mulf %9, %9 : f32
        %14 = arith.addf %12, %13 : f32
        %alloca = memref.alloca() : memref<i32>
        %c0_i32 = arith.constant 0 : i32
        affine.store %c0_i32, %alloca[] : memref<i32>
        affine.for %arg8 = 0 to 8 {
          %15 = arith.index_cast %arg8 : index to i32
          %16 = affine.load %alloca[] : memref<i32>
          %17 = arith.cmpi eq, %16, %c0_i32 : i32
          %18 = scf.if %17 -> (i1) {
            %19 = affine.load %arg0[%arg6, %arg8] : memref<4x8xf32>
            %20 = arith.cmpf olt, %14, %19 : f32
            scf.yield %20 : i1
          } else {
            %false = arith.constant false
            scf.yield %false : i1
          }
          scf.if %18 {
            affine.store %14, %arg4[%arg8] : memref<8xf32>
            affine.store %0, %arg5[%arg8] : memref<8xi32>
            %c1_i32 = arith.constant 1 : i32
            affine.store %c1_i32, %alloca[] : memref<i32>
          } else {
            %19 = arith.subi %15, %16 : i32
            %20 = arith.index_cast %19 : i32 to index
            %21 = memref.load %arg0[%arg6, %20] : memref<4x8xf32>
            affine.store %21, %arg4[%arg8] : memref<8xf32>
            %22 = memref.load %arg1[%arg6, %20] : memref<4x8xi32>
            affine.store %22, %arg5[%arg8] : memref<8xi32>
          }
        }
        affine.for %arg8 = 0 to 8 {
          %15 = affine.load %arg4[%arg8] : memref<8xf32>
          affine.store %15, %arg0[%arg6, %arg8] : memref<4x8xf32>
          %16 = affine.load %arg5[%arg8] : memref<8xi32>
          affine.store %16, %arg1[%arg6, %arg8] : memref<4x8xi32>
        }
      }
    }
    return
  }
}

