module {
  func.func @au_generated_candidate_0__direct(%arg0: memref<32x3xf32>, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: memref<i32>, %arg6: memref<f32>) attributes {au.generated = true} {
    affine.for %arg7 = 0 to 32 {
      %0 = affine.load %arg0[%arg7, 0] : memref<32x3xf32>
      %1 = arith.subf %0, %arg1 : f32
      %2 = affine.load %arg0[%arg7, 1] : memref<32x3xf32>
      %3 = arith.subf %2, %arg2 : f32
      %4 = affine.load %arg0[%arg7, 2] : memref<32x3xf32>
      %5 = arith.subf %4, %arg3 : f32
      %cst = arith.constant 0.000000e+00 : f32
      %6 = arith.cmpf olt, %1, %cst : f32
      %7 = scf.if %6 -> (f32) {
        %20 = arith.negf %1 : f32
        scf.yield %20 : f32
      } else {
        scf.yield %1 : f32
      }
      %8 = arith.cmpf olt, %3, %cst : f32
      %9 = scf.if %8 -> (f32) {
        %20 = arith.negf %3 : f32
        scf.yield %20 : f32
      } else {
        scf.yield %3 : f32
      }
      %10 = arith.addf %7, %9 : f32
      %11 = arith.cmpf olt, %5, %cst : f32
      %12 = scf.if %11 -> (f32) {
        %20 = arith.negf %5 : f32
        scf.yield %20 : f32
      } else {
        scf.yield %5 : f32
      }
      %13 = arith.addf %10, %12 : f32
      %14 = arith.subf %13, %arg4 : f32
      %15 = arith.cmpf olt, %14, %cst : f32
      %16 = scf.if %15 -> (f32) {
        %20 = arith.negf %14 : f32
        scf.yield %20 : f32
      } else {
        scf.yield %14 : f32
      }
      %cst_0 = arith.constant 8.000000e-02 : f32
      %17 = arith.cmpf olt, %16, %cst_0 : f32
      scf.if %17 {
        %20 = affine.load %arg5[] : memref<i32>
        %c1_i32 = arith.constant 1 : i32
        %21 = arith.addi %20, %c1_i32 : i32
        affine.store %21, %arg5[] : memref<i32>
      }
      %18 = affine.load %arg6[] : memref<f32>
      %19 = arith.addf %18, %16 : f32
      affine.store %19, %arg6[] : memref<f32>
    }
    return
  }
}

