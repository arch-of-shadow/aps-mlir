module {
  func.func @au_generated_candidate_10__direct(%arg0: memref<16x16x3xf32>, %arg1: index, %arg2: i32, %arg3: memref<16x16xf32>) attributes {au.generated = true} {
    affine.for %arg4 = 0 to 16 {
      %0 = arith.index_cast %arg4 : index to i32
      %alloca = memref.alloca() : memref<f32>
      %cst = arith.constant 0.000000e+00 : f32
      affine.store %cst, %alloca[] : memref<f32>
      %alloca_0 = memref.alloca() : memref<f32>
      affine.store %cst, %alloca_0[] : memref<f32>
      %alloca_1 = memref.alloca() : memref<f32>
      affine.store %cst, %alloca_1[] : memref<f32>
      %alloca_2 = memref.alloca() : memref<f32>
      affine.store %cst, %alloca_2[] : memref<f32>
      %alloca_3 = memref.alloca() : memref<f32>
      affine.store %cst, %alloca_3[] : memref<f32>
      %alloca_4 = memref.alloca() : memref<f32>
      affine.store %cst, %alloca_4[] : memref<f32>
      %1 = affine.load %arg0[%arg1, %arg4, 0] : memref<16x16x3xf32>
      %2 = affine.load %arg0[%arg1, %arg4, 1] : memref<16x16x3xf32>
      %3 = affine.load %arg0[%arg1, %arg4, 2] : memref<16x16x3xf32>
      affine.for %arg5 = 0 to 5 {
        %22 = arith.index_cast %arg5 : index to i32
        %c-2_i32 = arith.constant -2 : i32
        %23 = arith.addi %22, %c-2_i32 : i32
        affine.for %arg6 = 0 to 5 {
          %24 = arith.index_cast %arg6 : index to i32
          %25 = arith.addi %24, %c-2_i32 : i32
          %26 = arith.addi %arg2, %23 : i32
          %c0_i32 = arith.constant 0 : i32
          %27 = arith.cmpi sge, %26, %c0_i32 : i32
          %c15_i32 = arith.constant 15 : i32
          %28 = arith.cmpi sgt, %26, %c15_i32 : i32
          %29 = arith.andi %27, %28 : i1
          %true = arith.constant true
          %30 = arith.xori %29, %true : i1
          %31 = arith.andi %30, %27 : i1
          %32 = arith.select %29, %c15_i32, %c0_i32 : i32
          %33 = arith.select %31, %26, %32 : i32
          %34 = arith.addi %0, %25 : i32
          %35 = arith.cmpi sge, %34, %c0_i32 : i32
          %36 = arith.cmpi sgt, %34, %c15_i32 : i32
          %37 = arith.andi %35, %36 : i1
          %38 = arith.subi %true, %37 : i1
          %39 = arith.andi %38, %35 : i1
          %40 = arith.select %37, %c15_i32, %c0_i32 : i32
          %41 = arith.select %39, %34, %40 : i32
          %42 = arith.index_cast %33 : i32 to index
          %43 = arith.index_cast %41 : i32 to index
          %c0 = arith.constant 0 : index
          %44 = memref.load %arg0[%42, %43, %c0] : memref<16x16x3xf32>
          %45 = arith.subf %44, %1 : f32
          %c1 = arith.constant 1 : index
          %46 = memref.load %arg0[%42, %43, %c1] : memref<16x16x3xf32>
          %47 = arith.subf %46, %2 : f32
          %c2 = arith.constant 2 : index
          %48 = memref.load %arg0[%42, %43, %c2] : memref<16x16x3xf32>
          %49 = arith.subf %48, %3 : f32
          %50 = arith.mulf %45, %45 : f32
          %51 = affine.load %alloca[] : memref<f32>
          %52 = arith.addf %51, %50 : f32
          affine.store %52, %alloca[] : memref<f32>
          %53 = arith.mulf %47, %47 : f32
          %54 = affine.load %alloca_0[] : memref<f32>
          %55 = arith.addf %54, %53 : f32
          affine.store %55, %alloca_0[] : memref<f32>
          %56 = arith.mulf %49, %49 : f32
          %57 = affine.load %alloca_1[] : memref<f32>
          %58 = arith.addf %57, %56 : f32
          affine.store %58, %alloca_1[] : memref<f32>
          %59 = arith.mulf %45, %47 : f32
          %60 = affine.load %alloca_2[] : memref<f32>
          %61 = arith.addf %60, %59 : f32
          affine.store %61, %alloca_2[] : memref<f32>
          %62 = arith.mulf %45, %49 : f32
          %63 = affine.load %alloca_3[] : memref<f32>
          %64 = arith.addf %63, %62 : f32
          affine.store %64, %alloca_3[] : memref<f32>
          %65 = arith.mulf %47, %49 : f32
          %66 = affine.load %alloca_4[] : memref<f32>
          %67 = arith.addf %66, %65 : f32
          affine.store %67, %alloca_4[] : memref<f32>
        }
      }
      %4 = affine.load %alloca[] : memref<f32>
      %5 = affine.load %alloca_0[] : memref<f32>
      %6 = arith.addf %4, %5 : f32
      %7 = affine.load %alloca_1[] : memref<f32>
      %8 = arith.addf %6, %7 : f32
      %9 = affine.load %alloca_2[] : memref<f32>
      %10 = arith.cmpf olt, %9, %cst : f32
      %11 = scf.if %10 -> (f32) {
        %22 = arith.negf %9 : f32
        scf.yield %22 : f32
      } else {
        scf.yield %9 : f32
      }
      %12 = affine.load %alloca_3[] : memref<f32>
      %13 = arith.cmpf olt, %12, %cst : f32
      %14 = scf.if %13 -> (f32) {
        %22 = arith.negf %12 : f32
        scf.yield %22 : f32
      } else {
        scf.yield %12 : f32
      }
      %15 = arith.addf %11, %14 : f32
      %16 = affine.load %alloca_4[] : memref<f32>
      %17 = arith.cmpf olt, %16, %cst : f32
      %18 = scf.if %17 -> (f32) {
        %22 = arith.negf %16 : f32
        scf.yield %22 : f32
      } else {
        scf.yield %16 : f32
      }
      %19 = arith.addf %15, %18 : f32
      %cst_5 = arith.constant 5.000000e-01 : f32
      %20 = arith.mulf %19, %cst_5 : f32
      %21 = arith.subf %8, %20 : f32
      affine.store %21, %arg3[%arg1, %arg4] : memref<16x16xf32>
    }
    return
  }
}

