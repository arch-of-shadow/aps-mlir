module {
  func.func @au_generated_candidate_12__direct(%arg0: i32, %arg1: memref<12x12x3xf32>, %arg2: memref<12x12x3xf32>, %arg3: index) attributes {au.generated = true} {
    affine.for %arg4 = 0 to 12 {
      %0 = arith.index_cast %arg4 : index to i32
      %alloca = memref.alloca() : memref<f32>
      %cst = arith.constant 0.000000e+00 : f32
      affine.store %cst, %alloca[] : memref<f32>
      %alloca_0 = memref.alloca() : memref<f32>
      affine.store %cst, %alloca_0[] : memref<f32>
      %alloca_1 = memref.alloca() : memref<f32>
      affine.store %cst, %alloca_1[] : memref<f32>
      affine.for %arg5 = 0 to 3 {
        %7 = arith.index_cast %arg5 : index to i32
        %c-1_i32 = arith.constant -1 : i32
        %8 = arith.addi %7, %c-1_i32 : i32
        affine.for %arg6 = 0 to 3 {
          %9 = arith.index_cast %arg6 : index to i32
          %10 = arith.addi %9, %c-1_i32 : i32
          %11 = arith.addi %arg0, %8 : i32
          %c0_i32 = arith.constant 0 : i32
          %12 = arith.cmpi sge, %11, %c0_i32 : i32
          %c11_i32 = arith.constant 11 : i32
          %13 = arith.cmpi sgt, %11, %c11_i32 : i32
          %14 = arith.andi %12, %13 : i1
          %true = arith.constant true
          %15 = arith.xori %14, %true : i1
          %16 = arith.andi %15, %12 : i1
          %17 = arith.select %14, %c11_i32, %c0_i32 : i32
          %18 = arith.select %16, %11, %17 : i32
          %19 = arith.addi %0, %10 : i32
          %20 = arith.cmpi sge, %19, %c0_i32 : i32
          %21 = arith.cmpi sgt, %19, %c11_i32 : i32
          %22 = arith.andi %20, %21 : i1
          %23 = arith.xori %22, %true : i1
          %24 = arith.andi %23, %20 : i1
          %25 = arith.select %22, %c11_i32, %c0_i32 : i32
          %26 = arith.select %24, %19, %25 : i32
          %27 = arith.index_cast %18 : i32 to index
          %28 = arith.index_cast %26 : i32 to index
          %c0 = arith.constant 0 : index
          %29 = memref.load %arg1[%27, %28, %c0] : memref<12x12x3xf32>
          %30 = affine.load %alloca[] : memref<f32>
          %31 = arith.addf %30, %29 : f32
          affine.store %31, %alloca[] : memref<f32>
          %c1 = arith.constant 1 : index
          %32 = memref.load %arg1[%27, %28, %c1] : memref<12x12x3xf32>
          %33 = affine.load %alloca_0[] : memref<f32>
          %34 = arith.addf %33, %32 : f32
          affine.store %34, %alloca_0[] : memref<f32>
          %c2 = arith.constant 2 : index
          %35 = memref.load %arg1[%27, %28, %c2] : memref<12x12x3xf32>
          %36 = affine.load %alloca_1[] : memref<f32>
          %37 = arith.addf %36, %35 : f32
          affine.store %37, %alloca_1[] : memref<f32>
        }
      }
      %1 = affine.load %alloca[] : memref<f32>
      %cst_2 = arith.constant 9.000000e+00 : f32
      %2 = arith.divf %1, %cst_2 : f32
      affine.store %2, %arg2[%arg3, %arg4, 0] : memref<12x12x3xf32>
      %3 = affine.load %alloca_0[] : memref<f32>
      %4 = arith.divf %3, %cst_2 : f32
      affine.store %4, %arg2[%arg3, %arg4, 1] : memref<12x12x3xf32>
      %5 = affine.load %alloca_1[] : memref<f32>
      %6 = arith.divf %5, %cst_2 : f32
      affine.store %6, %arg2[%arg3, %arg4, 2] : memref<12x12x3xf32>
    }
    return
  }
}

