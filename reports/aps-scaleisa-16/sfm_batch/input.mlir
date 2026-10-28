module {
  memref.global "private" @_ZL10DEPTH_BIAS : memref<1xf32> = dense<6.000000e+00>
  memref.global "private" @_ZL2FY : memref<1xf32> = dense<2.160000e+02>
  memref.global "private" @_ZL2FX : memref<1xf32> = dense<2.200000e+02>
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<192x2xf32>
    %alloca_0 = memref.alloca() : memref<192xi32>
    %alloca_1 = memref.alloca() : memref<192xi32>
    %alloca_2 = memref.alloca() : memref<24x3xf32>
    %alloca_3 = memref.alloca() : memref<8x3xf32>
    %cast = memref.cast %alloca_3 : memref<8x3xf32> to memref<?x3xf32>
    %cast_4 = memref.cast %alloca_2 : memref<24x3xf32> to memref<?x3xf32>
    %cast_5 = memref.cast %alloca_1 : memref<192xi32> to memref<?xi32>
    %cast_6 = memref.cast %alloca_0 : memref<192xi32> to memref<?xi32>
    %cast_7 = memref.cast %alloca : memref<192x2xf32> to memref<?x2xf32>
    call @_ZL16init_sfm_problemPA3_fS0_PiS1_PA2_f(%cast, %cast_4, %cast_5, %cast_6, %cast_7) : (memref<?x3xf32>, memref<?x3xf32>, memref<?xi32>, memref<?xi32>, memref<?x2xf32>) -> ()
    call @_ZL12optimize_sfmPA3_fS0_PiS1_PA2_f(%cast, %cast_4, %cast_5, %cast_6, %cast_7) : (memref<?x3xf32>, memref<?x3xf32>, memref<?xi32>, memref<?xi32>, memref<?x2xf32>) -> ()
    %0 = call @_ZL9summarizePA3_fS0_(%cast, %cast_4) : (memref<?x3xf32>, memref<?x3xf32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL16init_sfm_problemPA3_fS0_PiS1_PA2_f(%arg0: memref<?x3xf32>, %arg1: memref<?x3xf32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?x2xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-2_i32 = arith.constant -2 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 5.000000e-02 : f32
    %cst_0 = arith.constant 7.000000e-02 : f32
    %cst_1 = arith.constant 1.000000e-01 : f32
    %cst_2 = arith.constant 4.000000e-02 : f32
    %cst_3 = arith.constant 8.000000e-02 : f32
    %c4_i32 = arith.constant 4 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst_4 = arith.constant 2.000000e-02 : f32
    %cst_5 = arith.constant 3.000000e-02 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_6 = arith.constant 1.800000e-01 : f32
    %cst_7 = arith.constant 2.800000e+00 : f32
    %cst_8 = arith.constant 6.000000e-01 : f32
    %cst_9 = arith.constant -0.899999976 : f32
    %c6_i32 = arith.constant 6 : i32
    %cst_10 = arith.constant 4.500000e-01 : f32
    %cst_11 = arith.constant -1.200000e+00 : f32
    %cst_12 = arith.constant 1.200000e-01 : f32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %cst_13 = arith.constant 2.500000e-01 : f32
    %cst_14 = arith.constant 5.500000e-01 : f32
    %cst_15 = arith.constant -1.800000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %cst_16 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<1xf32>
    affine.store %cst_16, %alloca[0] : memref<1xf32>
    %alloca_17 = memref.alloca() : memref<1xf32>
    affine.store %cst_16, %alloca_17[0] : memref<1xf32>
    %alloca_18 = memref.alloca() : memref<1xf32>
    affine.store %cst_16, %alloca_18[0] : memref<1xf32>
    %alloca_19 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_19[] : memref<i32>
    affine.for %arg5 = 0 to 8 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = arith.sitofp %0 : i32 to f32
      %2 = arith.mulf %1, %cst_14 : f32
      %3 = arith.addf %2, %cst_15 : f32
      affine.store %3, %arg0[%arg5, 0] : memref<?x3xf32>
      %4 = arith.remsi %0, %c3_i32 : i32
      %5 = arith.addi %4, %c-1_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.mulf %6, %cst_13 : f32
      affine.store %7, %arg0[%arg5, 1] : memref<?x3xf32>
      %8 = arith.remsi %0, %c2_i32 : i32
      %9 = arith.sitofp %8 : i32 to f32
      %10 = arith.mulf %9, %cst_12 : f32
      affine.store %10, %arg0[%arg5, 2] : memref<?x3xf32>
    }
    affine.for %arg5 = 0 to 24 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = arith.remsi %0, %c6_i32 : i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.mulf %2, %cst_10 : f32
      %4 = arith.addf %3, %cst_11 : f32
      affine.store %4, %arg1[%arg5, 0] : memref<?x3xf32>
      %5 = arith.divsi %0, %c6_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.mulf %6, %cst_8 : f32
      %8 = arith.addf %7, %cst_9 : f32
      affine.store %8, %arg1[%arg5, 1] : memref<?x3xf32>
      %9 = arith.remsi %0, %c5_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_6 : f32
      %12 = arith.addf %11, %cst_7 : f32
      affine.store %12, %arg1[%arg5, 2] : memref<?x3xf32>
    }
    affine.store %c0_i32, %alloca_19[] : memref<i32>
    affine.for %arg5 = 0 to 8 {
      %0 = arith.index_cast %arg5 : index to i32
      affine.for %arg6 = 0 to 24 {
        %1 = arith.index_cast %arg6 : index to i32
        %2 = affine.load %alloca_19[] : memref<i32>
        %3 = arith.index_cast %2 : i32 to index
        memref.store %0, %arg2[%3] : memref<?xi32>
        memref.store %1, %arg3[%3] : memref<?xi32>
        %4 = "polygeist.subindex"(%arg0, %arg5) : (memref<?x3xf32>, index) -> memref<?xf32>
        %5 = "polygeist.subindex"(%arg1, %arg6) : (memref<?x3xf32>, index) -> memref<?xf32>
        %cast = memref.cast %alloca_18 : memref<1xf32> to memref<?xf32>
        %cast_20 = memref.cast %alloca_17 : memref<1xf32> to memref<?xf32>
        %cast_21 = memref.cast %alloca : memref<1xf32> to memref<?xf32>
        func.call @_ZL13project_pointPfS_S_S_S_(%4, %5, %cast, %cast_20, %cast_21) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
        %6 = affine.load %alloca_18[0] : memref<1xf32>
        %7 = arith.remsi %2, %c5_i32 : i32
        %8 = arith.addi %7, %c-2_i32 : i32
        %9 = arith.sitofp %8 : i32 to f32
        %10 = arith.mulf %9, %cst_5 : f32
        %11 = arith.addf %6, %10 : f32
        memref.store %11, %arg4[%3, %c0] : memref<?x2xf32>
        %12 = affine.load %alloca_17[0] : memref<1xf32>
        %13 = arith.remsi %2, %c7_i32 : i32
        %14 = arith.addi %13, %c-3_i32 : i32
        %15 = arith.sitofp %14 : i32 to f32
        %16 = arith.mulf %15, %cst_4 : f32
        %17 = arith.subf %12, %16 : f32
        memref.store %17, %arg4[%3, %c1] : memref<?x2xf32>
        %18 = arith.addi %2, %c1_i32 : i32
        affine.store %18, %alloca_19[] : memref<i32>
      }
    }
    affine.for %arg5 = 0 to 8 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = arith.remsi %0, %c4_i32 : i32
      %2 = arith.addi %1, %c-1_i32 : i32
      %3 = arith.sitofp %2 : i32 to f32
      %4 = arith.mulf %3, %cst_12 : f32
      %5 = affine.load %arg0[%arg5, 0] : memref<?x3xf32>
      %6 = arith.addf %5, %4 : f32
      affine.store %6, %arg0[%arg5, 0] : memref<?x3xf32>
      %7 = arith.remsi %0, %c2_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %9 = arith.mulf %8, %cst_3 : f32
      %10 = affine.load %arg0[%arg5, 1] : memref<?x3xf32>
      %11 = arith.subf %10, %9 : f32
      affine.store %11, %arg0[%arg5, 1] : memref<?x3xf32>
      %12 = arith.remsi %0, %c3_i32 : i32
      %13 = arith.addi %12, %c-1_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %15 = arith.mulf %14, %cst_2 : f32
      %16 = affine.load %arg0[%arg5, 2] : memref<?x3xf32>
      %17 = arith.addf %16, %15 : f32
      affine.store %17, %arg0[%arg5, 2] : memref<?x3xf32>
    }
    affine.for %arg5 = 0 to 24 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = arith.remsi %0, %c3_i32 : i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.mulf %2, %cst_1 : f32
      %4 = affine.load %arg1[%arg5, 0] : memref<?x3xf32>
      %5 = arith.subf %4, %3 : f32
      affine.store %5, %arg1[%arg5, 0] : memref<?x3xf32>
      %6 = arith.remsi %0, %c2_i32 : i32
      %7 = arith.sitofp %6 : i32 to f32
      %8 = arith.mulf %7, %cst_0 : f32
      %9 = affine.load %arg1[%arg5, 1] : memref<?x3xf32>
      %10 = arith.addf %9, %8 : f32
      affine.store %10, %arg1[%arg5, 1] : memref<?x3xf32>
      %11 = arith.remsi %0, %c4_i32 : i32
      %12 = arith.addi %11, %c-1_i32 : i32
      %13 = arith.sitofp %12 : i32 to f32
      %14 = arith.mulf %13, %cst : f32
      %15 = affine.load %arg1[%arg5, 2] : memref<?x3xf32>
      %16 = arith.addf %15, %14 : f32
      affine.store %16, %arg1[%arg5, 2] : memref<?x3xf32>
    }
    return
  }
  func.func private @_ZL12optimize_sfmPA3_fS0_PiS1_PA2_f(%arg0: memref<?x3xf32>, %arg1: memref<?x3xf32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?x2xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 2.000000e-01 : f32
    %cst_0 = arith.constant 1.800000e-01 : f32
    %cst_1 = arith.constant 1.000000e-03 : f32
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<1xf32>
    affine.store %cst_2, %alloca[0] : memref<1xf32>
    %alloca_3 = memref.alloca() : memref<1xf32>
    affine.store %cst_2, %alloca_3[0] : memref<1xf32>
    %alloca_4 = memref.alloca() : memref<1xf32>
    affine.store %cst_2, %alloca_4[0] : memref<1xf32>
    %alloca_5 = memref.alloca() : memref<24x3xf32>
    %alloca_6 = memref.alloca() : memref<24x3xf32>
    %alloca_7 = memref.alloca() : memref<8x3xf32>
    %alloca_8 = memref.alloca() : memref<8x3xf32>
    affine.for %arg5 = 0 to 6 {
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 3 {
          affine.store %cst_2, %alloca_8[%arg6, %arg7] : memref<8x3xf32>
          affine.store %cst_1, %alloca_7[%arg6, %arg7] : memref<8x3xf32>
        }
      }
      affine.for %arg6 = 0 to 24 {
        affine.for %arg7 = 0 to 3 {
          affine.store %cst_2, %alloca_6[%arg6, %arg7] : memref<24x3xf32>
          affine.store %cst_1, %alloca_5[%arg6, %arg7] : memref<24x3xf32>
        }
      }
      affine.for %arg6 = 0 to 192 {
        %0 = affine.load %arg2[%arg6] : memref<?xi32>
        %1 = affine.load %arg3[%arg6] : memref<?xi32>
        %2 = arith.index_cast %0 : i32 to index
        %3 = "polygeist.subindex"(%arg0, %2) : (memref<?x3xf32>, index) -> memref<?xf32>
        %4 = arith.index_cast %1 : i32 to index
        %5 = "polygeist.subindex"(%arg1, %4) : (memref<?x3xf32>, index) -> memref<?xf32>
        %cast = memref.cast %alloca_4 : memref<1xf32> to memref<?xf32>
        %cast_9 = memref.cast %alloca_3 : memref<1xf32> to memref<?xf32>
        %cast_10 = memref.cast %alloca : memref<1xf32> to memref<?xf32>
        func.call @_ZL13project_pointPfS_S_S_S_(%3, %5, %cast, %cast_9, %cast_10) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
        %6 = memref.load %arg1[%4, %c0] : memref<?x3xf32>
        %7 = memref.load %arg0[%2, %c0] : memref<?x3xf32>
        %8 = arith.subf %6, %7 : f32
        %9 = memref.load %arg1[%4, %c1] : memref<?x3xf32>
        %10 = memref.load %arg0[%2, %c1] : memref<?x3xf32>
        %11 = arith.subf %9, %10 : f32
        %12 = affine.load %alloca_4[0] : memref<1xf32>
        %13 = affine.load %arg4[%arg6, 0] : memref<?x2xf32>
        %14 = arith.subf %12, %13 : f32
        %15 = affine.load %alloca_3[0] : memref<1xf32>
        %16 = affine.load %arg4[%arg6, 1] : memref<?x2xf32>
        %17 = arith.subf %15, %16 : f32
        %18 = memref.get_global @_ZL2FX : memref<1xf32>
        %19 = affine.load %18[0] : memref<1xf32>
        %20 = affine.load %alloca[0] : memref<1xf32>
        %21 = arith.mulf %19, %20 : f32
        %22 = memref.get_global @_ZL2FY : memref<1xf32>
        %23 = affine.load %22[0] : memref<1xf32>
        %24 = arith.mulf %23, %20 : f32
        %25 = arith.negf %19 : f32
        %26 = arith.mulf %25, %8 : f32
        %27 = arith.mulf %26, %20 : f32
        %28 = arith.mulf %27, %20 : f32
        %29 = arith.negf %23 : f32
        %30 = arith.mulf %29, %11 : f32
        %31 = arith.mulf %30, %20 : f32
        %32 = arith.mulf %31, %20 : f32
        %33 = arith.negf %14 : f32
        %34 = arith.mulf %33, %21 : f32
        %35 = memref.load %alloca_8[%2, %c0] : memref<8x3xf32>
        %36 = arith.addf %35, %34 : f32
        memref.store %36, %alloca_8[%2, %c0] : memref<8x3xf32>
        %37 = arith.negf %17 : f32
        %38 = arith.mulf %37, %24 : f32
        %39 = memref.load %alloca_8[%2, %c1] : memref<8x3xf32>
        %40 = arith.addf %39, %38 : f32
        memref.store %40, %alloca_8[%2, %c1] : memref<8x3xf32>
        %41 = arith.mulf %14, %28 : f32
        %42 = arith.mulf %17, %32 : f32
        %43 = arith.addf %41, %42 : f32
        %44 = arith.negf %43 : f32
        %45 = memref.load %alloca_8[%2, %c2] : memref<8x3xf32>
        %46 = arith.addf %45, %44 : f32
        memref.store %46, %alloca_8[%2, %c2] : memref<8x3xf32>
        %47 = arith.mulf %21, %21 : f32
        %48 = memref.load %alloca_7[%2, %c0] : memref<8x3xf32>
        %49 = arith.addf %48, %47 : f32
        memref.store %49, %alloca_7[%2, %c0] : memref<8x3xf32>
        %50 = arith.mulf %24, %24 : f32
        %51 = memref.load %alloca_7[%2, %c1] : memref<8x3xf32>
        %52 = arith.addf %51, %50 : f32
        memref.store %52, %alloca_7[%2, %c1] : memref<8x3xf32>
        %53 = arith.mulf %28, %28 : f32
        %54 = arith.mulf %32, %32 : f32
        %55 = arith.addf %53, %54 : f32
        %56 = memref.load %alloca_7[%2, %c2] : memref<8x3xf32>
        %57 = arith.addf %56, %55 : f32
        memref.store %57, %alloca_7[%2, %c2] : memref<8x3xf32>
        %58 = arith.mulf %14, %21 : f32
        %59 = memref.load %alloca_6[%4, %c0] : memref<24x3xf32>
        %60 = arith.addf %59, %58 : f32
        memref.store %60, %alloca_6[%4, %c0] : memref<24x3xf32>
        %61 = arith.mulf %17, %24 : f32
        %62 = memref.load %alloca_6[%4, %c1] : memref<24x3xf32>
        %63 = arith.addf %62, %61 : f32
        memref.store %63, %alloca_6[%4, %c1] : memref<24x3xf32>
        %64 = memref.load %alloca_6[%4, %c2] : memref<24x3xf32>
        %65 = arith.addf %64, %43 : f32
        memref.store %65, %alloca_6[%4, %c2] : memref<24x3xf32>
        %66 = memref.load %alloca_5[%4, %c0] : memref<24x3xf32>
        %67 = arith.addf %66, %47 : f32
        memref.store %67, %alloca_5[%4, %c0] : memref<24x3xf32>
        %68 = memref.load %alloca_5[%4, %c1] : memref<24x3xf32>
        %69 = arith.addf %68, %50 : f32
        memref.store %69, %alloca_5[%4, %c1] : memref<24x3xf32>
        %70 = memref.load %alloca_5[%4, %c2] : memref<24x3xf32>
        %71 = arith.addf %70, %55 : f32
        memref.store %71, %alloca_5[%4, %c2] : memref<24x3xf32>
      }
      affine.for %arg6 = 0 to 8 {
        affine.for %arg7 = 0 to 3 {
          %0 = affine.load %alloca_8[%arg6, %arg7] : memref<8x3xf32>
          %1 = arith.mulf %0, %cst_0 : f32
          %2 = affine.load %alloca_7[%arg6, %arg7] : memref<8x3xf32>
          %3 = arith.divf %1, %2 : f32
          %4 = affine.load %arg0[%arg6, %arg7] : memref<?x3xf32>
          %5 = arith.subf %4, %3 : f32
          affine.store %5, %arg0[%arg6, %arg7] : memref<?x3xf32>
        }
      }
      affine.for %arg6 = 0 to 24 {
        affine.for %arg7 = 0 to 3 {
          %0 = affine.load %alloca_6[%arg6, %arg7] : memref<24x3xf32>
          %1 = arith.mulf %0, %cst : f32
          %2 = affine.load %alloca_5[%arg6, %arg7] : memref<24x3xf32>
          %3 = arith.divf %1, %2 : f32
          %4 = affine.load %arg1[%arg6, %arg7] : memref<?x3xf32>
          %5 = arith.subf %4, %3 : f32
          affine.store %5, %arg1[%arg6, %arg7] : memref<?x3xf32>
        }
      }
    }
    return
  }
  func.func private @_ZL9summarizePA3_fS0_(%arg0: memref<?x3xf32>, %arg1: memref<?x3xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.270000e+02 : f32
    %cst_1 = arith.constant 1.090000e+02 : f32
    %cst_2 = arith.constant 8.300000e+01 : f32
    %cst_3 = arith.constant 1.490000e+02 : f32
    %cst_4 = arith.constant 1.310000e+02 : f32
    %cst_5 = arith.constant 9.700000e+01 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 8 {
      %5 = affine.load %arg0[%arg2, 0] : memref<?x3xf32>
      %6 = arith.mulf %5, %cst_5 : f32
      %7 = arith.fptosi %6 : f32 to i32
      %8 = affine.load %alloca[] : memref<i32>
      %9 = arith.addi %8, %7 : i32
      affine.store %9, %alloca[] : memref<i32>
      %10 = affine.load %arg0[%arg2, 1] : memref<?x3xf32>
      %11 = arith.mulf %10, %cst_4 : f32
      %12 = arith.fptosi %11 : f32 to i32
      %13 = arith.addi %9, %12 : i32
      affine.store %13, %alloca[] : memref<i32>
      %14 = affine.load %arg0[%arg2, 2] : memref<?x3xf32>
      %15 = arith.mulf %14, %cst_3 : f32
      %16 = arith.fptosi %15 : f32 to i32
      %17 = arith.addi %13, %16 : i32
      affine.store %17, %alloca[] : memref<i32>
    }
    affine.for %arg2 = 0 to 24 {
      %5 = affine.load %arg1[%arg2, 0] : memref<?x3xf32>
      %6 = arith.mulf %5, %cst_2 : f32
      %7 = arith.fptosi %6 : f32 to i32
      %8 = affine.load %alloca[] : memref<i32>
      %9 = arith.addi %8, %7 : i32
      affine.store %9, %alloca[] : memref<i32>
      %10 = affine.load %arg1[%arg2, 1] : memref<?x3xf32>
      %11 = arith.mulf %10, %cst_1 : f32
      %12 = arith.fptosi %11 : f32 to i32
      %13 = arith.addi %9, %12 : i32
      affine.store %13, %alloca[] : memref<i32>
      %14 = affine.load %arg1[%arg2, 2] : memref<?x3xf32>
      %15 = arith.mulf %14, %cst_0 : f32
      %16 = arith.fptosi %15 : f32 to i32
      %17 = arith.addi %13, %16 : i32
      affine.store %17, %alloca[] : memref<i32>
    }
    %0 = affine.load %alloca[] : memref<i32>
    %1 = arith.sitofp %0 : i32 to f32
    %2 = call @_ZL5abs_ff(%1) : (f32) -> f32
    %3 = arith.cmpf ogt, %2, %cst : f32
    %4 = arith.select %3, %0, %c1_i32 : i32
    return %4 : i32
  }
  func.func private @_ZL13project_pointPfS_S_S_S_(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = affine.load %arg1[0] : memref<?xf32>
    %1 = affine.load %arg0[0] : memref<?xf32>
    %2 = arith.subf %0, %1 : f32
    %3 = affine.load %arg1[1] : memref<?xf32>
    %4 = affine.load %arg0[1] : memref<?xf32>
    %5 = arith.subf %3, %4 : f32
    %6 = affine.load %arg1[2] : memref<?xf32>
    %7 = affine.load %arg0[2] : memref<?xf32>
    %8 = arith.subf %6, %7 : f32
    %9 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
    %10 = affine.load %9[0] : memref<1xf32>
    %11 = arith.addf %8, %10 : f32
    %12 = arith.divf %cst, %11 : f32
    affine.store %12, %arg4[0] : memref<?xf32>
    %13 = memref.get_global @_ZL2FX : memref<1xf32>
    %14 = affine.load %13[0] : memref<1xf32>
    %15 = arith.mulf %14, %2 : f32
    %16 = affine.load %arg4[0] : memref<?xf32>
    %17 = arith.mulf %15, %16 : f32
    affine.store %17, %arg2[0] : memref<?xf32>
    %18 = memref.get_global @_ZL2FY : memref<1xf32>
    %19 = affine.load %18[0] : memref<1xf32>
    %20 = arith.mulf %19, %5 : f32
    %21 = affine.load %arg4[0] : memref<?xf32>
    %22 = arith.mulf %20, %21 : f32
    affine.store %22, %arg3[0] : memref<?xf32>
    return
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

