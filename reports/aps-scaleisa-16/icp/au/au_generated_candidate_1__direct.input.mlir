module {
  func.func @au_generated_candidate_1__direct(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<32x3xf32>, %arg4: memref<32x3xf32>, %arg5: memref<f32>, %arg6: memref<f32>, %arg7: memref<f32>, %arg8: memref<f32>) attributes {au.generated = true} {
    affine.for %arg9 = 0 to 32 {
      %alloca = memref.alloca() : memref<f32>
      %cst = arith.constant 1.000000e+30 : f32
      affine.store %cst, %alloca[] : memref<f32>
      %alloca_0 = memref.alloca() : memref<i32>
      %c0_i32 = arith.constant 0 : i32
      affine.store %c0_i32, %alloca_0[] : memref<i32>
      affine.for %arg10 = 0 to 32 {
        %26 = arith.index_cast %arg10 : index to i32
        %27 = affine.load %arg0[] : memref<f32>
        %28 = affine.load %arg1[] : memref<f32>
        %29 = affine.load %arg2[] : memref<f32>
        %30 = affine.load %arg3[%arg9, 0] : memref<32x3xf32>
        %31 = arith.addf %30, %27 : f32
        %32 = affine.load %arg4[%arg10, 0] : memref<32x3xf32>
        %33 = arith.subf %31, %32 : f32
        %34 = affine.load %arg3[%arg9, 1] : memref<32x3xf32>
        %35 = arith.addf %34, %28 : f32
        %36 = affine.load %arg4[%arg10, 1] : memref<32x3xf32>
        %37 = arith.subf %35, %36 : f32
        %38 = affine.load %arg3[%arg9, 2] : memref<32x3xf32>
        %39 = arith.addf %38, %29 : f32
        %40 = affine.load %arg4[%arg10, 2] : memref<32x3xf32>
        %41 = arith.subf %39, %40 : f32
        %42 = arith.mulf %33, %33 : f32
        %43 = arith.mulf %37, %37 : f32
        %44 = arith.addf %42, %43 : f32
        %45 = arith.mulf %41, %41 : f32
        %46 = arith.addf %44, %45 : f32
        %47 = affine.load %alloca[] : memref<f32>
        %48 = arith.cmpf olt, %46, %47 : f32
        scf.if %48 {
          affine.store %46, %alloca[] : memref<f32>
          affine.store %26, %alloca_0[] : memref<i32>
        }
      }
      %0 = affine.load %alloca_0[] : memref<i32>
      %1 = arith.index_cast %0 : i32 to index
      %c0 = arith.constant 0 : index
      %2 = memref.load %arg4[%1, %c0] : memref<32x3xf32>
      %3 = affine.load %arg3[%arg9, 0] : memref<32x3xf32>
      %4 = affine.load %arg0[] : memref<f32>
      %5 = arith.addf %3, %4 : f32
      %6 = arith.subf %2, %5 : f32
      %7 = affine.load %arg5[] : memref<f32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %arg5[] : memref<f32>
      %c1 = arith.constant 1 : index
      %9 = memref.load %arg4[%1, %c1] : memref<32x3xf32>
      %10 = affine.load %arg3[%arg9, 1] : memref<32x3xf32>
      %11 = affine.load %arg1[] : memref<f32>
      %12 = arith.addf %10, %11 : f32
      %13 = arith.subf %9, %12 : f32
      %14 = affine.load %arg6[] : memref<f32>
      %15 = arith.addf %14, %13 : f32
      affine.store %15, %arg6[] : memref<f32>
      %c2 = arith.constant 2 : index
      %16 = memref.load %arg4[%1, %c2] : memref<32x3xf32>
      %17 = affine.load %arg3[%arg9, 2] : memref<32x3xf32>
      %18 = affine.load %arg2[] : memref<f32>
      %19 = arith.addf %17, %18 : f32
      %20 = arith.subf %16, %19 : f32
      %21 = affine.load %arg7[] : memref<f32>
      %22 = arith.addf %21, %20 : f32
      affine.store %22, %arg7[] : memref<f32>
      %23 = affine.load %alloca[] : memref<f32>
      %24 = affine.load %arg8[] : memref<f32>
      %25 = arith.addf %24, %23 : f32
      affine.store %25, %arg8[] : memref<f32>
    }
    return
  }
}

