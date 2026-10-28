module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 9.700000e+01 : f32
    %cst_0 = arith.constant 1.310000e+02 : f32
    %cst_1 = arith.constant 1.970000e+02 : f32
    %cst_2 = arith.constant 1.510000e+02 : f32
    %cst_3 = arith.constant 1.810000e+02 : f32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant 1.000000e-03 : f32
    %cst_6 = arith.constant -0.899999976 : f32
    %cst_7 = arith.constant 2.000000e-01 : f32
    %cst_8 = arith.constant 1.000000e+00 : f32
    %cst_9 = arith.constant 4.000000e-01 : f32
    %cst_10 = arith.constant 1.000000e-01 : f32
    %cst_11 = arith.constant 2.200000e-01 : f32
    %cst_12 = arith.constant 2.500000e-01 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c16_i32 = arith.constant 16 : i32
    %cst_13 = arith.constant 7.500000e-01 : f32
    %cst_14 = arith.constant 5.000000e-02 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_15 = arith.constant 1.800000e-01 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_16 = arith.constant 1.500000e-02 : f32
    %c5_i32 = arith.constant 5 : i32
    %c20_i32 = arith.constant 20 : i32
    %cst_17 = arith.constant 1.600000e+00 : f32
    %cst_18 = arith.constant 3.800000e-01 : f32
    %cst_19 = arith.constant 1.200000e+00 : f32
    %cst_20 = arith.constant 2.700000e-01 : f32
    %c7_i32 = arith.constant 7 : i32
    %cst_21 = arith.constant 2.000000e-02 : f32
    %c4_i32 = arith.constant 4 : i32
    %cst_22 = arith.constant 1.200000e-01 : f32
    %cst_23 = arith.constant 7.000000e-02 : f32
    %cst_24 = arith.constant 0.00999999977 : f32
    %cst_25 = arith.constant 9.000000e-02 : f32
    %cst_26 = arith.constant 6.000000e-02 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c-2_i32 = arith.constant -2 : i32
    %alloca = memref.alloca() : memref<64x2xf32>
    %alloca_27 = memref.alloca() : memref<64xi32>
    %alloca_28 = memref.alloca() : memref<64xi32>
    %alloca_29 = memref.alloca() : memref<20x2xf32>
    %alloca_30 = memref.alloca() : memref<16x3xf32>
    affine.for %arg0 = 0 to 16 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.sitofp %6 : i32 to f32
      %8 = arith.mulf %7, %cst_13 : f32
      %9 = arith.remsi %6, %c3_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_14 : f32
      %12 = arith.addf %8, %11 : f32
      affine.store %12, %alloca_30[%arg0, 0] : memref<16x3xf32>
      %13 = arith.remsi %6, %c2_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %15 = arith.mulf %14, %cst_15 : f32
      affine.store %15, %alloca_30[%arg0, 1] : memref<16x3xf32>
      %16 = arith.remsi %6, %c5_i32 : i32
      %17 = arith.addi %16, %c-2_i32 : i32
      %18 = arith.sitofp %17 : i32 to f32
      %19 = arith.mulf %18, %cst_16 : f32
      affine.store %19, %alloca_30[%arg0, 2] : memref<16x3xf32>
    }
    affine.for %arg0 = 0 to 20 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.remsi %6, %c5_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %9 = arith.mulf %8, %cst_18 : f32
      %10 = arith.addf %9, %cst_17 : f32
      affine.store %10, %alloca_29[%arg0, 0] : memref<20x2xf32>
      %11 = arith.divsi %6, %c5_i32 : i32
      %12 = arith.sitofp %11 : i32 to f32
      %13 = arith.mulf %12, %cst_20 : f32
      %14 = arith.addf %13, %cst_19 : f32
      affine.store %14, %alloca_29[%arg0, 1] : memref<20x2xf32>
    }
    affine.for %arg0 = 0 to 64 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.remsi %6, %c16_i32 : i32
      %8 = arith.muli %6, %c3_i32 : i32
      %9 = arith.divsi %6, %c7_i32 : i32
      %10 = arith.addi %8, %9 : i32
      %11 = arith.remsi %10, %c20_i32 : i32
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %alloca_29[%12, %c0] : memref<20x2xf32>
      %14 = arith.index_cast %7 : i32 to index
      %15 = memref.load %alloca_30[%14, %c0] : memref<16x3xf32>
      %16 = arith.subf %13, %15 : f32
      %17 = memref.load %alloca_29[%12, %c1] : memref<20x2xf32>
      %18 = memref.load %alloca_30[%14, %c1] : memref<16x3xf32>
      %19 = arith.subf %17, %18 : f32
      %20 = memref.load %alloca_30[%14, %c2] : memref<16x3xf32>
      affine.store %7, %alloca_28[%arg0] : memref<64xi32>
      affine.store %11, %alloca_27[%arg0] : memref<64xi32>
      %21 = arith.mulf %20, %19 : f32
      %22 = arith.addf %16, %21 : f32
      %23 = arith.remsi %6, %c3_i32 : i32
      %24 = arith.sitofp %23 : i32 to f32
      %25 = arith.mulf %24, %cst_21 : f32
      %26 = arith.addf %22, %25 : f32
      affine.store %26, %alloca[%arg0, 0] : memref<64x2xf32>
      %27 = arith.mulf %20, %16 : f32
      %28 = arith.subf %19, %27 : f32
      %29 = arith.remsi %6, %c4_i32 : i32
      %30 = arith.sitofp %29 : i32 to f32
      %31 = arith.mulf %30, %cst_16 : f32
      %32 = arith.subf %28, %31 : f32
      affine.store %32, %alloca[%arg0, 1] : memref<64x2xf32>
    }
    affine.for %arg0 = 0 to 16 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.remsi %6, %c4_i32 : i32
      %8 = arith.addi %7, %c-1_i32 : i32
      %9 = arith.sitofp %8 : i32 to f32
      %10 = arith.mulf %9, %cst_22 : f32
      %11 = affine.load %alloca_30[%arg0, 0] : memref<16x3xf32>
      %12 = arith.addf %11, %10 : f32
      affine.store %12, %alloca_30[%arg0, 0] : memref<16x3xf32>
      %13 = arith.remsi %6, %c3_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %15 = arith.mulf %14, %cst_23 : f32
      %16 = affine.load %alloca_30[%arg0, 1] : memref<16x3xf32>
      %17 = arith.subf %16, %15 : f32
      affine.store %17, %alloca_30[%arg0, 1] : memref<16x3xf32>
      %18 = arith.remsi %6, %c2_i32 : i32
      %19 = arith.sitofp %18 : i32 to f32
      %20 = arith.mulf %19, %cst_24 : f32
      %21 = affine.load %alloca_30[%arg0, 2] : memref<16x3xf32>
      %22 = arith.addf %21, %20 : f32
      affine.store %22, %alloca_30[%arg0, 2] : memref<16x3xf32>
    }
    affine.for %arg0 = 0 to 20 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.remsi %6, %c3_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %9 = arith.mulf %8, %cst_25 : f32
      %10 = affine.load %alloca_29[%arg0, 0] : memref<20x2xf32>
      %11 = arith.subf %10, %9 : f32
      affine.store %11, %alloca_29[%arg0, 0] : memref<20x2xf32>
      %12 = arith.remsi %6, %c2_i32 : i32
      %13 = arith.sitofp %12 : i32 to f32
      %14 = arith.mulf %13, %cst_26 : f32
      %15 = affine.load %alloca_29[%arg0, 1] : memref<20x2xf32>
      %16 = arith.addf %15, %14 : f32
      affine.store %16, %alloca_29[%arg0, 1] : memref<20x2xf32>
    }
    %alloca_31 = memref.alloca() : memref<20x2xf32>
    %alloca_32 = memref.alloca() : memref<20x2xf32>
    %alloca_33 = memref.alloca() : memref<16x3xf32>
    %alloca_34 = memref.alloca() : memref<16x3xf32>
    affine.for %arg0 = 0 to 7 {
      affine.for %arg1 = 0 to 16 {
        affine.store %cst_4, %alloca_34[%arg1, 0] : memref<16x3xf32>
        affine.store %cst_4, %alloca_34[%arg1, 1] : memref<16x3xf32>
        affine.store %cst_4, %alloca_34[%arg1, 2] : memref<16x3xf32>
        affine.store %cst_5, %alloca_33[%arg1, 0] : memref<16x3xf32>
        affine.store %cst_5, %alloca_33[%arg1, 1] : memref<16x3xf32>
        affine.store %cst_5, %alloca_33[%arg1, 2] : memref<16x3xf32>
      }
      affine.for %arg1 = 0 to 20 {
        affine.store %cst_4, %alloca_32[%arg1, 0] : memref<20x2xf32>
        affine.store %cst_4, %alloca_32[%arg1, 1] : memref<20x2xf32>
        affine.store %cst_5, %alloca_31[%arg1, 0] : memref<20x2xf32>
        affine.store %cst_5, %alloca_31[%arg1, 1] : memref<20x2xf32>
      }
      affine.for %arg1 = 0 to 64 {
        %6 = affine.load %alloca_28[%arg1] : memref<64xi32>
        %7 = affine.load %alloca_27[%arg1] : memref<64xi32>
        %8 = arith.index_cast %7 : i32 to index
        %9 = memref.load %alloca_29[%8, %c0] : memref<20x2xf32>
        %10 = arith.index_cast %6 : i32 to index
        %11 = memref.load %alloca_30[%10, %c0] : memref<16x3xf32>
        %12 = arith.subf %9, %11 : f32
        %13 = memref.load %alloca_29[%8, %c1] : memref<20x2xf32>
        %14 = memref.load %alloca_30[%10, %c1] : memref<16x3xf32>
        %15 = arith.subf %13, %14 : f32
        %16 = memref.load %alloca_30[%10, %c2] : memref<16x3xf32>
        %17 = arith.mulf %16, %15 : f32
        %18 = arith.addf %12, %17 : f32
        %19 = arith.mulf %16, %12 : f32
        %20 = arith.subf %15, %19 : f32
        %21 = affine.load %alloca[%arg1, 0] : memref<64x2xf32>
        %22 = arith.subf %18, %21 : f32
        %23 = affine.load %alloca[%arg1, 1] : memref<64x2xf32>
        %24 = arith.subf %20, %23 : f32
        %25 = arith.mulf %22, %cst_6 : f32
        %26 = arith.mulf %16, %cst_7 : f32
        %27 = arith.mulf %26, %24 : f32
        %28 = arith.addf %25, %27 : f32
        %29 = memref.load %alloca_34[%10, %c0] : memref<16x3xf32>
        %30 = arith.addf %29, %28 : f32
        memref.store %30, %alloca_34[%10, %c0] : memref<16x3xf32>
        %31 = arith.mulf %24, %cst_6 : f32
        %32 = arith.mulf %26, %22 : f32
        %33 = arith.subf %31, %32 : f32
        %34 = memref.load %alloca_34[%10, %c1] : memref<16x3xf32>
        %35 = arith.addf %34, %33 : f32
        memref.store %35, %alloca_34[%10, %c1] : memref<16x3xf32>
        %36 = arith.mulf %15, %22 : f32
        %37 = arith.mulf %12, %24 : f32
        %38 = arith.subf %36, %37 : f32
        %39 = memref.load %alloca_34[%10, %c2] : memref<16x3xf32>
        %40 = arith.addf %39, %38 : f32
        memref.store %40, %alloca_34[%10, %c2] : memref<16x3xf32>
        %41 = memref.load %alloca_33[%10, %c0] : memref<16x3xf32>
        %42 = arith.addf %41, %cst_8 : f32
        memref.store %42, %alloca_33[%10, %c0] : memref<16x3xf32>
        %43 = memref.load %alloca_33[%10, %c1] : memref<16x3xf32>
        %44 = arith.addf %43, %cst_8 : f32
        memref.store %44, %alloca_33[%10, %c1] : memref<16x3xf32>
        %45 = arith.mulf %12, %12 : f32
        %46 = arith.mulf %15, %15 : f32
        %47 = arith.addf %45, %46 : f32
        %48 = arith.mulf %47, %cst_9 : f32
        %49 = arith.addf %48, %cst_8 : f32
        %50 = memref.load %alloca_33[%10, %c2] : memref<16x3xf32>
        %51 = arith.addf %50, %49 : f32
        memref.store %51, %alloca_33[%10, %c2] : memref<16x3xf32>
        %52 = arith.mulf %16, %cst_10 : f32
        %53 = arith.mulf %52, %24 : f32
        %54 = arith.subf %22, %53 : f32
        %55 = memref.load %alloca_32[%8, %c0] : memref<20x2xf32>
        %56 = arith.addf %55, %54 : f32
        memref.store %56, %alloca_32[%8, %c0] : memref<20x2xf32>
        %57 = arith.mulf %52, %22 : f32
        %58 = arith.addf %24, %57 : f32
        %59 = memref.load %alloca_32[%8, %c1] : memref<20x2xf32>
        %60 = arith.addf %59, %58 : f32
        memref.store %60, %alloca_32[%8, %c1] : memref<20x2xf32>
        %61 = memref.load %alloca_31[%8, %c0] : memref<20x2xf32>
        %62 = arith.addf %61, %cst_8 : f32
        memref.store %62, %alloca_31[%8, %c0] : memref<20x2xf32>
        %63 = memref.load %alloca_31[%8, %c1] : memref<20x2xf32>
        %64 = arith.addf %63, %cst_8 : f32
        memref.store %64, %alloca_31[%8, %c1] : memref<20x2xf32>
      }
      affine.store %cst_4, %alloca_30[0, 0] : memref<16x3xf32>
      affine.store %cst_4, %alloca_30[0, 1] : memref<16x3xf32>
      affine.store %cst_4, %alloca_30[0, 2] : memref<16x3xf32>
      affine.for %arg1 = 0 to 15 {
        %6 = affine.load %alloca_34[%arg1 + 1, 0] : memref<16x3xf32>
        %7 = arith.mulf %6, %cst_11 : f32
        %8 = affine.load %alloca_33[%arg1 + 1, 0] : memref<16x3xf32>
        %9 = arith.divf %7, %8 : f32
        %10 = affine.load %alloca_30[%arg1 + 1, 0] : memref<16x3xf32>
        %11 = arith.subf %10, %9 : f32
        affine.store %11, %alloca_30[%arg1 + 1, 0] : memref<16x3xf32>
        %12 = affine.load %alloca_34[%arg1 + 1, 1] : memref<16x3xf32>
        %13 = arith.mulf %12, %cst_11 : f32
        %14 = affine.load %alloca_33[%arg1 + 1, 1] : memref<16x3xf32>
        %15 = arith.divf %13, %14 : f32
        %16 = affine.load %alloca_30[%arg1 + 1, 1] : memref<16x3xf32>
        %17 = arith.subf %16, %15 : f32
        affine.store %17, %alloca_30[%arg1 + 1, 1] : memref<16x3xf32>
        %18 = affine.load %alloca_34[%arg1 + 1, 2] : memref<16x3xf32>
        %19 = arith.mulf %18, %cst_15 : f32
        %20 = affine.load %alloca_33[%arg1 + 1, 2] : memref<16x3xf32>
        %21 = arith.divf %19, %20 : f32
        %22 = affine.load %alloca_30[%arg1 + 1, 2] : memref<16x3xf32>
        %23 = arith.subf %22, %21 : f32
        affine.store %23, %alloca_30[%arg1 + 1, 2] : memref<16x3xf32>
      }
      affine.for %arg1 = 0 to 20 {
        %6 = affine.load %alloca_32[%arg1, 0] : memref<20x2xf32>
        %7 = arith.mulf %6, %cst_12 : f32
        %8 = affine.load %alloca_31[%arg1, 0] : memref<20x2xf32>
        %9 = arith.divf %7, %8 : f32
        %10 = affine.load %alloca_29[%arg1, 0] : memref<20x2xf32>
        %11 = arith.subf %10, %9 : f32
        affine.store %11, %alloca_29[%arg1, 0] : memref<20x2xf32>
        %12 = affine.load %alloca_32[%arg1, 1] : memref<20x2xf32>
        %13 = arith.mulf %12, %cst_12 : f32
        %14 = affine.load %alloca_31[%arg1, 1] : memref<20x2xf32>
        %15 = arith.divf %13, %14 : f32
        %16 = affine.load %alloca_29[%arg1, 1] : memref<20x2xf32>
        %17 = arith.subf %16, %15 : f32
        affine.store %17, %alloca_29[%arg1, 1] : memref<20x2xf32>
      }
    }
    %alloca_35 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_35[] : memref<i32>
    affine.store %c0_i32, %alloca_35[] : memref<i32>
    affine.for %arg0 = 0 to 16 {
      %6 = affine.load %alloca_30[%arg0, 0] : memref<16x3xf32>
      %7 = arith.mulf %6, %cst : f32
      %8 = arith.fptosi %7 : f32 to i32
      %9 = affine.load %alloca_35[] : memref<i32>
      %10 = arith.addi %9, %8 : i32
      affine.store %10, %alloca_35[] : memref<i32>
      %11 = affine.load %alloca_30[%arg0, 1] : memref<16x3xf32>
      %12 = arith.mulf %11, %cst_0 : f32
      %13 = arith.fptosi %12 : f32 to i32
      %14 = arith.addi %10, %13 : i32
      affine.store %14, %alloca_35[] : memref<i32>
      %15 = affine.load %alloca_30[%arg0, 2] : memref<16x3xf32>
      %16 = arith.mulf %15, %cst_1 : f32
      %17 = arith.fptosi %16 : f32 to i32
      %18 = arith.addi %14, %17 : i32
      affine.store %18, %alloca_35[] : memref<i32>
    }
    affine.for %arg0 = 0 to 20 {
      %6 = affine.load %alloca_29[%arg0, 0] : memref<20x2xf32>
      %7 = arith.mulf %6, %cst_2 : f32
      %8 = arith.fptosi %7 : f32 to i32
      %9 = affine.load %alloca_35[] : memref<i32>
      %10 = arith.addi %9, %8 : i32
      affine.store %10, %alloca_35[] : memref<i32>
      %11 = affine.load %alloca_29[%arg0, 1] : memref<20x2xf32>
      %12 = arith.mulf %11, %cst_3 : f32
      %13 = arith.fptosi %12 : f32 to i32
      %14 = arith.addi %10, %13 : i32
      affine.store %14, %alloca_35[] : memref<i32>
    }
    %0 = affine.load %alloca_35[] : memref<i32>
    %1 = arith.sitofp %0 : i32 to f32
    %2 = arith.cmpi slt, %0, %c0_i32 : i32
    %3 = scf.if %2 -> (f32) {
      %6 = arith.negf %1 : f32
      scf.yield %6 : f32
    } else {
      scf.yield %1 : f32
    }
    %4 = arith.cmpf ogt, %3, %cst_4 : f32
    %5 = arith.select %4, %0, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_30, %alloca_29, %alloca_28, %alloca_27, %alloca) : (memref<16x3xf32>, memref<20x2xf32>, memref<64xi32>, memref<64xi32>, memref<64x2xf32>) -> ()
    return %5 : i32
  }
  func.func private @pcl_validate_outputs(memref<16x3xf32>, memref<20x2xf32>, memref<64xi32>, memref<64xi32>, memref<64x2xf32>)
}

