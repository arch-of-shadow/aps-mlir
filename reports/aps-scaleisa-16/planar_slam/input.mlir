module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<64x2xf32>
    %alloca_0 = memref.alloca() : memref<64xi32>
    %alloca_1 = memref.alloca() : memref<64xi32>
    %alloca_2 = memref.alloca() : memref<20x2xf32>
    %alloca_3 = memref.alloca() : memref<16x3xf32>
    %cast = memref.cast %alloca_3 : memref<16x3xf32> to memref<?x3xf32>
    %cast_4 = memref.cast %alloca_2 : memref<20x2xf32> to memref<?x2xf32>
    %cast_5 = memref.cast %alloca_1 : memref<64xi32> to memref<?xi32>
    %cast_6 = memref.cast %alloca_0 : memref<64xi32> to memref<?xi32>
    %cast_7 = memref.cast %alloca : memref<64x2xf32> to memref<?x2xf32>
    call @_ZL12init_problemPA3_fPA2_fPiS3_S2_(%cast, %cast_4, %cast_5, %cast_6, %cast_7) : (memref<?x3xf32>, memref<?x2xf32>, memref<?xi32>, memref<?xi32>, memref<?x2xf32>) -> ()
    call @_ZL15optimize_planarPA3_fPA2_fPiS3_S2_(%cast, %cast_4, %cast_5, %cast_6, %cast_7) : (memref<?x3xf32>, memref<?x2xf32>, memref<?xi32>, memref<?xi32>, memref<?x2xf32>) -> ()
    %0 = call @_ZL9summarizePA3_fPA2_f(%cast, %cast_4) : (memref<?x3xf32>, memref<?x2xf32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL12init_problemPA3_fPA2_fPiS3_S2_(%arg0: memref<?x3xf32>, %arg1: memref<?x2xf32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?x2xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-2_i32 = arith.constant -2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 6.000000e-02 : f32
    %cst_0 = arith.constant 9.000000e-02 : f32
    %cst_1 = arith.constant 0.00999999977 : f32
    %cst_2 = arith.constant 7.000000e-02 : f32
    %cst_3 = arith.constant 1.200000e-01 : f32
    %c4_i32 = arith.constant 4 : i32
    %cst_4 = arith.constant 2.000000e-02 : f32
    %c7_i32 = arith.constant 7 : i32
    %cst_5 = arith.constant 2.700000e-01 : f32
    %cst_6 = arith.constant 1.200000e+00 : f32
    %cst_7 = arith.constant 3.800000e-01 : f32
    %cst_8 = arith.constant 1.600000e+00 : f32
    %c20_i32 = arith.constant 20 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_9 = arith.constant 1.500000e-02 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_10 = arith.constant 1.800000e-01 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_11 = arith.constant 5.000000e-02 : f32
    %cst_12 = arith.constant 7.500000e-01 : f32
    %c16_i32 = arith.constant 16 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    affine.for %arg5 = 0 to 16 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = arith.sitofp %0 : i32 to f32
      %2 = arith.mulf %1, %cst_12 : f32
      %3 = arith.remsi %0, %c3_i32 : i32
      %4 = arith.sitofp %3 : i32 to f32
      %5 = arith.mulf %4, %cst_11 : f32
      %6 = arith.addf %2, %5 : f32
      affine.store %6, %arg0[%arg5, 0] : memref<?x3xf32>
      %7 = arith.remsi %0, %c2_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %9 = arith.mulf %8, %cst_10 : f32
      affine.store %9, %arg0[%arg5, 1] : memref<?x3xf32>
      %10 = arith.remsi %0, %c5_i32 : i32
      %11 = arith.addi %10, %c-2_i32 : i32
      %12 = arith.sitofp %11 : i32 to f32
      %13 = arith.mulf %12, %cst_9 : f32
      affine.store %13, %arg0[%arg5, 2] : memref<?x3xf32>
    }
    affine.for %arg5 = 0 to 20 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = arith.remsi %0, %c5_i32 : i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.mulf %2, %cst_7 : f32
      %4 = arith.addf %3, %cst_8 : f32
      affine.store %4, %arg1[%arg5, 0] : memref<?x2xf32>
      %5 = arith.divsi %0, %c5_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.mulf %6, %cst_5 : f32
      %8 = arith.addf %7, %cst_6 : f32
      affine.store %8, %arg1[%arg5, 1] : memref<?x2xf32>
    }
    affine.for %arg5 = 0 to 64 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = arith.remsi %0, %c16_i32 : i32
      %2 = arith.muli %0, %c3_i32 : i32
      %3 = arith.divsi %0, %c7_i32 : i32
      %4 = arith.addi %2, %3 : i32
      %5 = arith.remsi %4, %c20_i32 : i32
      %6 = arith.index_cast %5 : i32 to index
      %7 = memref.load %arg1[%6, %c0] : memref<?x2xf32>
      %8 = arith.index_cast %1 : i32 to index
      %9 = memref.load %arg0[%8, %c0] : memref<?x3xf32>
      %10 = arith.subf %7, %9 : f32
      %11 = memref.load %arg1[%6, %c1] : memref<?x2xf32>
      %12 = memref.load %arg0[%8, %c1] : memref<?x3xf32>
      %13 = arith.subf %11, %12 : f32
      %14 = memref.load %arg0[%8, %c2] : memref<?x3xf32>
      affine.store %1, %arg2[%arg5] : memref<?xi32>
      affine.store %5, %arg3[%arg5] : memref<?xi32>
      %15 = arith.mulf %14, %13 : f32
      %16 = arith.addf %10, %15 : f32
      %17 = arith.remsi %0, %c3_i32 : i32
      %18 = arith.sitofp %17 : i32 to f32
      %19 = arith.mulf %18, %cst_4 : f32
      %20 = arith.addf %16, %19 : f32
      affine.store %20, %arg4[%arg5, 0] : memref<?x2xf32>
      %21 = arith.mulf %14, %10 : f32
      %22 = arith.subf %13, %21 : f32
      %23 = arith.remsi %0, %c4_i32 : i32
      %24 = arith.sitofp %23 : i32 to f32
      %25 = arith.mulf %24, %cst_9 : f32
      %26 = arith.subf %22, %25 : f32
      affine.store %26, %arg4[%arg5, 1] : memref<?x2xf32>
    }
    affine.for %arg5 = 0 to 16 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = arith.remsi %0, %c4_i32 : i32
      %2 = arith.addi %1, %c-1_i32 : i32
      %3 = arith.sitofp %2 : i32 to f32
      %4 = arith.mulf %3, %cst_3 : f32
      %5 = affine.load %arg0[%arg5, 0] : memref<?x3xf32>
      %6 = arith.addf %5, %4 : f32
      affine.store %6, %arg0[%arg5, 0] : memref<?x3xf32>
      %7 = arith.remsi %0, %c3_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %9 = arith.mulf %8, %cst_2 : f32
      %10 = affine.load %arg0[%arg5, 1] : memref<?x3xf32>
      %11 = arith.subf %10, %9 : f32
      affine.store %11, %arg0[%arg5, 1] : memref<?x3xf32>
      %12 = arith.remsi %0, %c2_i32 : i32
      %13 = arith.sitofp %12 : i32 to f32
      %14 = arith.mulf %13, %cst_1 : f32
      %15 = affine.load %arg0[%arg5, 2] : memref<?x3xf32>
      %16 = arith.addf %15, %14 : f32
      affine.store %16, %arg0[%arg5, 2] : memref<?x3xf32>
    }
    affine.for %arg5 = 0 to 20 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = arith.remsi %0, %c3_i32 : i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.mulf %2, %cst_0 : f32
      %4 = affine.load %arg1[%arg5, 0] : memref<?x2xf32>
      %5 = arith.subf %4, %3 : f32
      affine.store %5, %arg1[%arg5, 0] : memref<?x2xf32>
      %6 = arith.remsi %0, %c2_i32 : i32
      %7 = arith.sitofp %6 : i32 to f32
      %8 = arith.mulf %7, %cst : f32
      %9 = affine.load %arg1[%arg5, 1] : memref<?x2xf32>
      %10 = arith.addf %9, %8 : f32
      affine.store %10, %arg1[%arg5, 1] : memref<?x2xf32>
    }
    return
  }
  func.func private @_ZL15optimize_planarPA3_fPA2_fPiS3_S2_(%arg0: memref<?x3xf32>, %arg1: memref<?x2xf32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?x2xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 2.500000e-01 : f32
    %cst_0 = arith.constant 1.800000e-01 : f32
    %cst_1 = arith.constant 2.200000e-01 : f32
    %cst_2 = arith.constant 1.000000e-01 : f32
    %cst_3 = arith.constant 4.000000e-01 : f32
    %cst_4 = arith.constant 1.000000e+00 : f32
    %cst_5 = arith.constant 2.000000e-01 : f32
    %cst_6 = arith.constant -0.899999976 : f32
    %cst_7 = arith.constant 1.000000e-03 : f32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst_8 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<20x2xf32>
    %alloca_9 = memref.alloca() : memref<20x2xf32>
    %alloca_10 = memref.alloca() : memref<16x3xf32>
    %alloca_11 = memref.alloca() : memref<16x3xf32>
    affine.for %arg5 = 0 to 7 {
      affine.for %arg6 = 0 to 16 {
        affine.store %cst_8, %alloca_11[%arg6, 0] : memref<16x3xf32>
        affine.store %cst_8, %alloca_11[%arg6, 1] : memref<16x3xf32>
        affine.store %cst_8, %alloca_11[%arg6, 2] : memref<16x3xf32>
        affine.store %cst_7, %alloca_10[%arg6, 0] : memref<16x3xf32>
        affine.store %cst_7, %alloca_10[%arg6, 1] : memref<16x3xf32>
        affine.store %cst_7, %alloca_10[%arg6, 2] : memref<16x3xf32>
      }
      affine.for %arg6 = 0 to 20 {
        affine.store %cst_8, %alloca_9[%arg6, 0] : memref<20x2xf32>
        affine.store %cst_8, %alloca_9[%arg6, 1] : memref<20x2xf32>
        affine.store %cst_7, %alloca[%arg6, 0] : memref<20x2xf32>
        affine.store %cst_7, %alloca[%arg6, 1] : memref<20x2xf32>
      }
      affine.for %arg6 = 0 to 64 {
        %0 = affine.load %arg2[%arg6] : memref<?xi32>
        %1 = affine.load %arg3[%arg6] : memref<?xi32>
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg1[%2, %c0] : memref<?x2xf32>
        %4 = arith.index_cast %0 : i32 to index
        %5 = memref.load %arg0[%4, %c0] : memref<?x3xf32>
        %6 = arith.subf %3, %5 : f32
        %7 = memref.load %arg1[%2, %c1] : memref<?x2xf32>
        %8 = memref.load %arg0[%4, %c1] : memref<?x3xf32>
        %9 = arith.subf %7, %8 : f32
        %10 = memref.load %arg0[%4, %c2] : memref<?x3xf32>
        %11 = arith.mulf %10, %9 : f32
        %12 = arith.addf %6, %11 : f32
        %13 = arith.mulf %10, %6 : f32
        %14 = arith.subf %9, %13 : f32
        %15 = affine.load %arg4[%arg6, 0] : memref<?x2xf32>
        %16 = arith.subf %12, %15 : f32
        %17 = affine.load %arg4[%arg6, 1] : memref<?x2xf32>
        %18 = arith.subf %14, %17 : f32
        %19 = arith.mulf %16, %cst_6 : f32
        %20 = arith.mulf %10, %cst_5 : f32
        %21 = arith.mulf %20, %18 : f32
        %22 = arith.addf %19, %21 : f32
        %23 = memref.load %alloca_11[%4, %c0] : memref<16x3xf32>
        %24 = arith.addf %23, %22 : f32
        memref.store %24, %alloca_11[%4, %c0] : memref<16x3xf32>
        %25 = arith.mulf %18, %cst_6 : f32
        %26 = arith.mulf %20, %16 : f32
        %27 = arith.subf %25, %26 : f32
        %28 = memref.load %alloca_11[%4, %c1] : memref<16x3xf32>
        %29 = arith.addf %28, %27 : f32
        memref.store %29, %alloca_11[%4, %c1] : memref<16x3xf32>
        %30 = arith.mulf %9, %16 : f32
        %31 = arith.mulf %6, %18 : f32
        %32 = arith.subf %30, %31 : f32
        %33 = memref.load %alloca_11[%4, %c2] : memref<16x3xf32>
        %34 = arith.addf %33, %32 : f32
        memref.store %34, %alloca_11[%4, %c2] : memref<16x3xf32>
        %35 = memref.load %alloca_10[%4, %c0] : memref<16x3xf32>
        %36 = arith.addf %35, %cst_4 : f32
        memref.store %36, %alloca_10[%4, %c0] : memref<16x3xf32>
        %37 = memref.load %alloca_10[%4, %c1] : memref<16x3xf32>
        %38 = arith.addf %37, %cst_4 : f32
        memref.store %38, %alloca_10[%4, %c1] : memref<16x3xf32>
        %39 = arith.mulf %6, %6 : f32
        %40 = arith.mulf %9, %9 : f32
        %41 = arith.addf %39, %40 : f32
        %42 = arith.mulf %41, %cst_3 : f32
        %43 = arith.addf %42, %cst_4 : f32
        %44 = memref.load %alloca_10[%4, %c2] : memref<16x3xf32>
        %45 = arith.addf %44, %43 : f32
        memref.store %45, %alloca_10[%4, %c2] : memref<16x3xf32>
        %46 = arith.mulf %10, %cst_2 : f32
        %47 = arith.mulf %46, %18 : f32
        %48 = arith.subf %16, %47 : f32
        %49 = memref.load %alloca_9[%2, %c0] : memref<20x2xf32>
        %50 = arith.addf %49, %48 : f32
        memref.store %50, %alloca_9[%2, %c0] : memref<20x2xf32>
        %51 = arith.mulf %46, %16 : f32
        %52 = arith.addf %18, %51 : f32
        %53 = memref.load %alloca_9[%2, %c1] : memref<20x2xf32>
        %54 = arith.addf %53, %52 : f32
        memref.store %54, %alloca_9[%2, %c1] : memref<20x2xf32>
        %55 = memref.load %alloca[%2, %c0] : memref<20x2xf32>
        %56 = arith.addf %55, %cst_4 : f32
        memref.store %56, %alloca[%2, %c0] : memref<20x2xf32>
        %57 = memref.load %alloca[%2, %c1] : memref<20x2xf32>
        %58 = arith.addf %57, %cst_4 : f32
        memref.store %58, %alloca[%2, %c1] : memref<20x2xf32>
      }
      affine.store %cst_8, %arg0[0, 0] : memref<?x3xf32>
      affine.store %cst_8, %arg0[0, 1] : memref<?x3xf32>
      affine.store %cst_8, %arg0[0, 2] : memref<?x3xf32>
      affine.for %arg6 = 0 to 15 {
        %0 = affine.load %alloca_11[%arg6 + 1, 0] : memref<16x3xf32>
        %1 = arith.mulf %0, %cst_1 : f32
        %2 = affine.load %alloca_10[%arg6 + 1, 0] : memref<16x3xf32>
        %3 = arith.divf %1, %2 : f32
        %4 = affine.load %arg0[%arg6 + 1, 0] : memref<?x3xf32>
        %5 = arith.subf %4, %3 : f32
        affine.store %5, %arg0[%arg6 + 1, 0] : memref<?x3xf32>
        %6 = affine.load %alloca_11[%arg6 + 1, 1] : memref<16x3xf32>
        %7 = arith.mulf %6, %cst_1 : f32
        %8 = affine.load %alloca_10[%arg6 + 1, 1] : memref<16x3xf32>
        %9 = arith.divf %7, %8 : f32
        %10 = affine.load %arg0[%arg6 + 1, 1] : memref<?x3xf32>
        %11 = arith.subf %10, %9 : f32
        affine.store %11, %arg0[%arg6 + 1, 1] : memref<?x3xf32>
        %12 = affine.load %alloca_11[%arg6 + 1, 2] : memref<16x3xf32>
        %13 = arith.mulf %12, %cst_0 : f32
        %14 = affine.load %alloca_10[%arg6 + 1, 2] : memref<16x3xf32>
        %15 = arith.divf %13, %14 : f32
        %16 = affine.load %arg0[%arg6 + 1, 2] : memref<?x3xf32>
        %17 = arith.subf %16, %15 : f32
        affine.store %17, %arg0[%arg6 + 1, 2] : memref<?x3xf32>
      }
      affine.for %arg6 = 0 to 20 {
        %0 = affine.load %alloca_9[%arg6, 0] : memref<20x2xf32>
        %1 = arith.mulf %0, %cst : f32
        %2 = affine.load %alloca[%arg6, 0] : memref<20x2xf32>
        %3 = arith.divf %1, %2 : f32
        %4 = affine.load %arg1[%arg6, 0] : memref<?x2xf32>
        %5 = arith.subf %4, %3 : f32
        affine.store %5, %arg1[%arg6, 0] : memref<?x2xf32>
        %6 = affine.load %alloca_9[%arg6, 1] : memref<20x2xf32>
        %7 = arith.mulf %6, %cst : f32
        %8 = affine.load %alloca[%arg6, 1] : memref<20x2xf32>
        %9 = arith.divf %7, %8 : f32
        %10 = affine.load %arg1[%arg6, 1] : memref<?x2xf32>
        %11 = arith.subf %10, %9 : f32
        affine.store %11, %arg1[%arg6, 1] : memref<?x2xf32>
      }
    }
    return
  }
  func.func private @_ZL9summarizePA3_fPA2_f(%arg0: memref<?x3xf32>, %arg1: memref<?x2xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.810000e+02 : f32
    %cst_1 = arith.constant 1.510000e+02 : f32
    %cst_2 = arith.constant 1.970000e+02 : f32
    %cst_3 = arith.constant 1.310000e+02 : f32
    %cst_4 = arith.constant 9.700000e+01 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 16 {
      %5 = affine.load %arg0[%arg2, 0] : memref<?x3xf32>
      %6 = arith.mulf %5, %cst_4 : f32
      %7 = arith.fptosi %6 : f32 to i32
      %8 = affine.load %alloca[] : memref<i32>
      %9 = arith.addi %8, %7 : i32
      affine.store %9, %alloca[] : memref<i32>
      %10 = affine.load %arg0[%arg2, 1] : memref<?x3xf32>
      %11 = arith.mulf %10, %cst_3 : f32
      %12 = arith.fptosi %11 : f32 to i32
      %13 = arith.addi %9, %12 : i32
      affine.store %13, %alloca[] : memref<i32>
      %14 = affine.load %arg0[%arg2, 2] : memref<?x3xf32>
      %15 = arith.mulf %14, %cst_2 : f32
      %16 = arith.fptosi %15 : f32 to i32
      %17 = arith.addi %13, %16 : i32
      affine.store %17, %alloca[] : memref<i32>
    }
    affine.for %arg2 = 0 to 20 {
      %5 = affine.load %arg1[%arg2, 0] : memref<?x2xf32>
      %6 = arith.mulf %5, %cst_1 : f32
      %7 = arith.fptosi %6 : f32 to i32
      %8 = affine.load %alloca[] : memref<i32>
      %9 = arith.addi %8, %7 : i32
      affine.store %9, %alloca[] : memref<i32>
      %10 = affine.load %arg1[%arg2, 1] : memref<?x2xf32>
      %11 = arith.mulf %10, %cst_0 : f32
      %12 = arith.fptosi %11 : f32 to i32
      %13 = arith.addi %9, %12 : i32
      affine.store %13, %alloca[] : memref<i32>
    }
    %0 = affine.load %alloca[] : memref<i32>
    %1 = arith.sitofp %0 : i32 to f32
    %2 = call @_ZL5abs_ff(%1) : (f32) -> f32
    %3 = arith.cmpf ogt, %2, %cst : f32
    %4 = arith.select %3, %0, %c1_i32 : i32
    return %4 : i32
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

