module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<4xf32>
    %alloca_0 = memref.alloca() : memref<3xf32>
    %alloca_1 = memref.alloca() : memref<32x3xf32>
    %alloca_2 = memref.alloca() : memref<32x3xf32>
    %cast = memref.cast %alloca_2 : memref<32x3xf32> to memref<?x3xf32>
    %cast_3 = memref.cast %alloca_1 : memref<32x3xf32> to memref<?x3xf32>
    call @_ZL11init_cloudsPA3_fS0_(%cast, %cast_3) : (memref<?x3xf32>, memref<?x3xf32>) -> ()
    %cast_4 = memref.cast %alloca_0 : memref<3xf32> to memref<?xf32>
    %cast_5 = memref.cast %alloca : memref<4xf32> to memref<?xf32>
    call @_ZL9icp_alignPA3_fS0_PfS1_(%cast, %cast_3, %cast_4, %cast_5) : (memref<?x3xf32>, memref<?x3xf32>, memref<?xf32>, memref<?xf32>) -> ()
    %0 = call @_ZL9summarizePfS_(%cast_4, %cast_5) : (memref<?xf32>, memref<?xf32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL11init_cloudsPA3_fS0_(%arg0: memref<?x3xf32>, %arg1: memref<?x3xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 6.000000e-02 : f32
    %cst_0 = arith.constant 4.000000e-02 : f32
    %cst_1 = arith.constant 8.000000e-02 : f32
    %cst_2 = arith.constant 2.000000e-02 : f32
    %c31_i32 = arith.constant 31 : i32
    %c13_i32 = arith.constant 13 : i32
    %cst_3 = arith.constant 2.500000e-02 : f32
    %c37_i32 = arith.constant 37 : i32
    %c11_i32 = arith.constant 11 : i32
    %cst_4 = arith.constant 3.000000e-02 : f32
    %c41_i32 = arith.constant 41 : i32
    %c7_i32 = arith.constant 7 : i32
    affine.for %arg2 = 0 to 32 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c7_i32 : i32
      %2 = arith.remsi %1, %c41_i32 : i32
      %3 = arith.sitofp %2 : i32 to f32
      %4 = arith.mulf %3, %cst_4 : f32
      %5 = arith.muli %0, %c11_i32 : i32
      %6 = arith.remsi %5, %c37_i32 : i32
      %7 = arith.sitofp %6 : i32 to f32
      %8 = arith.mulf %7, %cst_3 : f32
      %9 = arith.muli %0, %c13_i32 : i32
      %10 = arith.remsi %9, %c31_i32 : i32
      %11 = arith.sitofp %10 : i32 to f32
      %12 = arith.mulf %11, %cst_2 : f32
      affine.store %4, %arg0[%arg2, 0] : memref<?x3xf32>
      affine.store %8, %arg0[%arg2, 1] : memref<?x3xf32>
      affine.store %12, %arg0[%arg2, 2] : memref<?x3xf32>
      %13 = arith.addf %4, %cst_1 : f32
      affine.store %13, %arg1[%arg2, 0] : memref<?x3xf32>
      %14 = arith.subf %8, %cst_0 : f32
      affine.store %14, %arg1[%arg2, 1] : memref<?x3xf32>
      %15 = arith.addf %12, %cst : f32
      affine.store %15, %arg1[%arg2, 2] : memref<?x3xf32>
    }
    return
  }
  func.func private @_ZL9icp_alignPA3_fS0_PfS1_(%arg0: memref<?x3xf32>, %arg1: memref<?x3xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 3.200000e+01 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 1.000000e+30 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %alloca_3 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_5[] : memref<f32>
    %alloca_6 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_6[] : memref<f32>
    %alloca_7 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_7[] : memref<f32>
    %alloca_8 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_8[] : memref<f32>
    %alloca_9 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_9[] : memref<f32>
    %alloca_10 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_10[] : memref<f32>
    affine.store %cst_2, %alloca_10[] : memref<f32>
    affine.store %cst_2, %alloca_9[] : memref<f32>
    affine.store %cst_2, %alloca_8[] : memref<f32>
    affine.for %arg4 = 0 to 4 {
      affine.store %cst_2, %alloca_7[] : memref<f32>
      affine.store %cst_2, %alloca_6[] : memref<f32>
      affine.store %cst_2, %alloca_5[] : memref<f32>
      affine.store %cst_2, %alloca_4[] : memref<f32>
      affine.for %arg5 = 0 to 32 {
        affine.store %cst_1, %alloca_3[] : memref<f32>
        affine.store %c0_i32, %alloca[] : memref<i32>
        affine.for %arg6 = 0 to 32 {
          %46 = arith.index_cast %arg6 : index to i32
          %47 = "polygeist.subindex"(%arg0, %arg5) : (memref<?x3xf32>, index) -> memref<?xf32>
          %48 = "polygeist.subindex"(%arg1, %arg6) : (memref<?x3xf32>, index) -> memref<?xf32>
          %49 = affine.load %alloca_10[] : memref<f32>
          %50 = affine.load %alloca_9[] : memref<f32>
          %51 = affine.load %alloca_8[] : memref<f32>
          %52 = func.call @_ZL7sq_distPfS_fff(%47, %48, %49, %50, %51) : (memref<?xf32>, memref<?xf32>, f32, f32, f32) -> f32
          %53 = affine.load %alloca_3[] : memref<f32>
          %54 = arith.cmpf olt, %52, %53 : f32
          scf.if %54 {
            affine.store %52, %alloca_3[] : memref<f32>
            affine.store %46, %alloca[] : memref<i32>
          }
        }
        %20 = affine.load %alloca[] : memref<i32>
        %21 = arith.index_cast %20 : i32 to index
        %22 = memref.load %arg1[%21, %c0] : memref<?x3xf32>
        %23 = affine.load %arg0[%arg5, 0] : memref<?x3xf32>
        %24 = affine.load %alloca_10[] : memref<f32>
        %25 = arith.addf %23, %24 : f32
        %26 = arith.subf %22, %25 : f32
        %27 = affine.load %alloca_7[] : memref<f32>
        %28 = arith.addf %27, %26 : f32
        affine.store %28, %alloca_7[] : memref<f32>
        %29 = memref.load %arg1[%21, %c1] : memref<?x3xf32>
        %30 = affine.load %arg0[%arg5, 1] : memref<?x3xf32>
        %31 = affine.load %alloca_9[] : memref<f32>
        %32 = arith.addf %30, %31 : f32
        %33 = arith.subf %29, %32 : f32
        %34 = affine.load %alloca_6[] : memref<f32>
        %35 = arith.addf %34, %33 : f32
        affine.store %35, %alloca_6[] : memref<f32>
        %36 = memref.load %arg1[%21, %c2] : memref<?x3xf32>
        %37 = affine.load %arg0[%arg5, 2] : memref<?x3xf32>
        %38 = affine.load %alloca_8[] : memref<f32>
        %39 = arith.addf %37, %38 : f32
        %40 = arith.subf %36, %39 : f32
        %41 = affine.load %alloca_5[] : memref<f32>
        %42 = arith.addf %41, %40 : f32
        affine.store %42, %alloca_5[] : memref<f32>
        %43 = affine.load %alloca_3[] : memref<f32>
        %44 = affine.load %alloca_4[] : memref<f32>
        %45 = arith.addf %44, %43 : f32
        affine.store %45, %alloca_4[] : memref<f32>
      }
      %3 = affine.load %alloca_7[] : memref<f32>
      %4 = arith.mulf %3, %cst_0 : f32
      %5 = arith.divf %4, %cst : f32
      %6 = affine.load %alloca_10[] : memref<f32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %alloca_10[] : memref<f32>
      %8 = affine.load %alloca_6[] : memref<f32>
      %9 = arith.mulf %8, %cst_0 : f32
      %10 = arith.divf %9, %cst : f32
      %11 = affine.load %alloca_9[] : memref<f32>
      %12 = arith.addf %11, %10 : f32
      affine.store %12, %alloca_9[] : memref<f32>
      %13 = affine.load %alloca_5[] : memref<f32>
      %14 = arith.mulf %13, %cst_0 : f32
      %15 = arith.divf %14, %cst : f32
      %16 = affine.load %alloca_8[] : memref<f32>
      %17 = arith.addf %16, %15 : f32
      affine.store %17, %alloca_8[] : memref<f32>
      %18 = affine.load %alloca_4[] : memref<f32>
      %19 = arith.divf %18, %cst : f32
      affine.store %19, %arg3[%arg4] : memref<?xf32>
    }
    %0 = affine.load %alloca_10[] : memref<f32>
    affine.store %0, %arg2[0] : memref<?xf32>
    %1 = affine.load %alloca_9[] : memref<f32>
    affine.store %1, %arg2[1] : memref<?xf32>
    %2 = affine.load %alloca_8[] : memref<f32>
    affine.store %2, %arg2[2] : memref<?xf32>
    return
  }
  func.func private @_ZL9summarizePfS_(%arg0: memref<?xf32>, %arg1: memref<?xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e+02 : f32
    %cst_1 = arith.constant 1.400000e+03 : f32
    %cst_2 = arith.constant 1.200000e+03 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant 1.000000e+03 : f32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %0 = affine.load %arg0[0] : memref<?xf32>
    %1 = arith.mulf %0, %cst_3 : f32
    %2 = arith.fptosi %1 : f32 to i32
    affine.store %2, %alloca[] : memref<i32>
    %3 = affine.load %arg0[1] : memref<?xf32>
    %4 = arith.mulf %3, %cst_2 : f32
    %5 = arith.fptosi %4 : f32 to i32
    %6 = arith.addi %2, %5 : i32
    affine.store %6, %alloca[] : memref<i32>
    %7 = affine.load %arg0[2] : memref<?xf32>
    %8 = arith.mulf %7, %cst_1 : f32
    %9 = arith.fptosi %8 : f32 to i32
    %10 = arith.addi %6, %9 : i32
    affine.store %10, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 4 {
      %16 = affine.load %arg1[%arg2] : memref<?xf32>
      %17 = arith.mulf %16, %cst_0 : f32
      %18 = arith.fptosi %17 : f32 to i32
      %19 = affine.load %alloca[] : memref<i32>
      %20 = arith.addi %19, %18 : i32
      affine.store %20, %alloca[] : memref<i32>
    }
    %11 = affine.load %alloca[] : memref<i32>
    %12 = arith.sitofp %11 : i32 to f32
    %13 = call @_ZL5abs_ff(%12) : (f32) -> f32
    %14 = arith.cmpf ogt, %13, %cst : f32
    %15 = arith.select %14, %11, %c1_i32 : i32
    return %15 : i32
  }
  func.func private @_ZL7sq_distPfS_fff(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: f32, %arg4: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = affine.load %arg0[0] : memref<?xf32>
    %1 = arith.addf %0, %arg2 : f32
    %2 = affine.load %arg1[0] : memref<?xf32>
    %3 = arith.subf %1, %2 : f32
    %4 = affine.load %arg0[1] : memref<?xf32>
    %5 = arith.addf %4, %arg3 : f32
    %6 = affine.load %arg1[1] : memref<?xf32>
    %7 = arith.subf %5, %6 : f32
    %8 = affine.load %arg0[2] : memref<?xf32>
    %9 = arith.addf %8, %arg4 : f32
    %10 = affine.load %arg1[2] : memref<?xf32>
    %11 = arith.subf %9, %10 : f32
    %12 = arith.mulf %3, %3 : f32
    %13 = arith.mulf %7, %7 : f32
    %14 = arith.addf %12, %13 : f32
    %15 = arith.mulf %11, %11 : f32
    %16 = arith.addf %14, %15 : f32
    return %16 : f32
  }
  func.func private @_ZL5abs_ff(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.cmpf olt, %arg0, %cst : f32
    %1 = scf.if %0 -> (f32) {
      %2 = arith.negf %arg0 : f32
      scf.yield %2 : f32
    } else {
      scf.yield %arg0 : f32
    }
    return %1 : f32
  }
}

