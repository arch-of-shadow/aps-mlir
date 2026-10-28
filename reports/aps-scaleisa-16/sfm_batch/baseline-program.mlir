module {
  memref.global "private" @_ZL10DEPTH_BIAS : memref<1xf32> = dense<6.000000e+00>
  memref.global "private" @_ZL2FY : memref<1xf32> = dense<2.160000e+02>
  memref.global "private" @_ZL2FX : memref<1xf32> = dense<2.200000e+02>
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 9.700000e+01 : f32
    %cst_0 = arith.constant 1.310000e+02 : f32
    %cst_1 = arith.constant 1.490000e+02 : f32
    %cst_2 = arith.constant 8.300000e+01 : f32
    %cst_3 = arith.constant 1.090000e+02 : f32
    %cst_4 = arith.constant 1.270000e+02 : f32
    %c2 = arith.constant 2 : index
    %cst_5 = arith.constant 1.000000e-03 : f32
    %cst_6 = arith.constant 2.000000e-01 : f32
    %cst_7 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_8 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %cst_9 = arith.constant -1.800000e+00 : f32
    %cst_10 = arith.constant 5.500000e-01 : f32
    %cst_11 = arith.constant 2.500000e-01 : f32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_12 = arith.constant 1.200000e-01 : f32
    %cst_13 = arith.constant -1.200000e+00 : f32
    %cst_14 = arith.constant 4.500000e-01 : f32
    %c6_i32 = arith.constant 6 : i32
    %cst_15 = arith.constant -0.899999976 : f32
    %cst_16 = arith.constant 6.000000e-01 : f32
    %cst_17 = arith.constant 2.800000e+00 : f32
    %cst_18 = arith.constant 1.800000e-01 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_19 = arith.constant 3.000000e-02 : f32
    %cst_20 = arith.constant 2.000000e-02 : f32
    %c7_i32 = arith.constant 7 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_21 = arith.constant 8.000000e-02 : f32
    %cst_22 = arith.constant 4.000000e-02 : f32
    %cst_23 = arith.constant 1.000000e-01 : f32
    %cst_24 = arith.constant 7.000000e-02 : f32
    %cst_25 = arith.constant 5.000000e-02 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-2_i32 = arith.constant -2 : i32
    %alloca = memref.alloca() : memref<192x2xf32>
    %alloca_26 = memref.alloca() : memref<192xi32>
    %alloca_27 = memref.alloca() : memref<192xi32>
    %alloca_28 = memref.alloca() : memref<24x3xf32>
    %alloca_29 = memref.alloca() : memref<8x3xf32>
    %alloca_30 = memref.alloca() : memref<1xf32>
    affine.store %cst_8, %alloca_30[0] : memref<1xf32>
    %alloca_31 = memref.alloca() : memref<1xf32>
    affine.store %cst_8, %alloca_31[0] : memref<1xf32>
    %alloca_32 = memref.alloca() : memref<1xf32>
    affine.store %cst_8, %alloca_32[0] : memref<1xf32>
    %alloca_33 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_33[] : memref<i32>
    affine.for %arg0 = 0 to 8 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.sitofp %6 : i32 to f32
      %8 = arith.mulf %7, %cst_10 : f32
      %9 = arith.addf %8, %cst_9 : f32
      affine.store %9, %alloca_29[%arg0, 0] : memref<8x3xf32>
      %10 = arith.remsi %6, %c3_i32 : i32
      %11 = arith.addi %10, %c-1_i32 : i32
      %12 = arith.sitofp %11 : i32 to f32
      %13 = arith.mulf %12, %cst_11 : f32
      affine.store %13, %alloca_29[%arg0, 1] : memref<8x3xf32>
      %14 = arith.remsi %6, %c2_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.mulf %15, %cst_12 : f32
      affine.store %16, %alloca_29[%arg0, 2] : memref<8x3xf32>
    }
    affine.for %arg0 = 0 to 24 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.remsi %6, %c6_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %9 = arith.mulf %8, %cst_14 : f32
      %10 = arith.addf %9, %cst_13 : f32
      affine.store %10, %alloca_28[%arg0, 0] : memref<24x3xf32>
      %11 = arith.divsi %6, %c6_i32 : i32
      %12 = arith.sitofp %11 : i32 to f32
      %13 = arith.mulf %12, %cst_16 : f32
      %14 = arith.addf %13, %cst_15 : f32
      affine.store %14, %alloca_28[%arg0, 1] : memref<24x3xf32>
      %15 = arith.remsi %6, %c5_i32 : i32
      %16 = arith.sitofp %15 : i32 to f32
      %17 = arith.mulf %16, %cst_18 : f32
      %18 = arith.addf %17, %cst_17 : f32
      affine.store %18, %alloca_28[%arg0, 2] : memref<24x3xf32>
    }
    affine.store %c0_i32, %alloca_33[] : memref<i32>
    affine.for %arg0 = 0 to 8 {
      %6 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 24 {
        %7 = arith.index_cast %arg1 : index to i32
        %8 = affine.load %alloca_33[] : memref<i32>
        %9 = arith.index_cast %8 : i32 to index
        memref.store %6, %alloca_27[%9] : memref<192xi32>
        memref.store %7, %alloca_26[%9] : memref<192xi32>
        %10 = affine.load %alloca_28[%arg1, 0] : memref<24x3xf32>
        %11 = affine.load %alloca_29[%arg0, 0] : memref<8x3xf32>
        %12 = arith.subf %10, %11 : f32
        %13 = affine.load %alloca_28[%arg1, 1] : memref<24x3xf32>
        %14 = affine.load %alloca_29[%arg0, 1] : memref<8x3xf32>
        %15 = arith.subf %13, %14 : f32
        %16 = affine.load %alloca_28[%arg1, 2] : memref<24x3xf32>
        %17 = affine.load %alloca_29[%arg0, 2] : memref<8x3xf32>
        %18 = arith.subf %16, %17 : f32
        %19 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
        %20 = affine.load %19[0] : memref<1xf32>
        %21 = arith.addf %18, %20 : f32
        %22 = arith.divf %cst_7, %21 : f32
        affine.store %22, %alloca_30[0] : memref<1xf32>
        %23 = memref.get_global @_ZL2FX : memref<1xf32>
        %24 = affine.load %23[0] : memref<1xf32>
        %25 = arith.mulf %24, %12 : f32
        %26 = affine.load %alloca_30[0] : memref<1xf32>
        %27 = arith.mulf %25, %26 : f32
        affine.store %27, %alloca_32[0] : memref<1xf32>
        %28 = memref.get_global @_ZL2FY : memref<1xf32>
        %29 = affine.load %28[0] : memref<1xf32>
        %30 = arith.mulf %29, %15 : f32
        %31 = affine.load %alloca_30[0] : memref<1xf32>
        %32 = arith.mulf %30, %31 : f32
        affine.store %32, %alloca_31[0] : memref<1xf32>
        %33 = affine.load %alloca_32[0] : memref<1xf32>
        %34 = arith.remsi %8, %c5_i32 : i32
        %35 = arith.addi %34, %c-2_i32 : i32
        %36 = arith.sitofp %35 : i32 to f32
        %37 = arith.mulf %36, %cst_19 : f32
        %38 = arith.addf %33, %37 : f32
        memref.store %38, %alloca[%9, %c0] : memref<192x2xf32>
        %39 = affine.load %alloca_31[0] : memref<1xf32>
        %40 = arith.remsi %8, %c7_i32 : i32
        %41 = arith.addi %40, %c-3_i32 : i32
        %42 = arith.sitofp %41 : i32 to f32
        %43 = arith.mulf %42, %cst_20 : f32
        %44 = arith.subf %39, %43 : f32
        memref.store %44, %alloca[%9, %c1] : memref<192x2xf32>
        %45 = arith.addi %8, %c1_i32 : i32
        affine.store %45, %alloca_33[] : memref<i32>
      }
    }
    affine.for %arg0 = 0 to 8 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.remsi %6, %c4_i32 : i32
      %8 = arith.addi %7, %c-1_i32 : i32
      %9 = arith.sitofp %8 : i32 to f32
      %10 = arith.mulf %9, %cst_12 : f32
      %11 = affine.load %alloca_29[%arg0, 0] : memref<8x3xf32>
      %12 = arith.addf %11, %10 : f32
      affine.store %12, %alloca_29[%arg0, 0] : memref<8x3xf32>
      %13 = arith.remsi %6, %c2_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %15 = arith.mulf %14, %cst_21 : f32
      %16 = affine.load %alloca_29[%arg0, 1] : memref<8x3xf32>
      %17 = arith.subf %16, %15 : f32
      affine.store %17, %alloca_29[%arg0, 1] : memref<8x3xf32>
      %18 = arith.remsi %6, %c3_i32 : i32
      %19 = arith.addi %18, %c-1_i32 : i32
      %20 = arith.sitofp %19 : i32 to f32
      %21 = arith.mulf %20, %cst_22 : f32
      %22 = affine.load %alloca_29[%arg0, 2] : memref<8x3xf32>
      %23 = arith.addf %22, %21 : f32
      affine.store %23, %alloca_29[%arg0, 2] : memref<8x3xf32>
    }
    affine.for %arg0 = 0 to 24 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.remsi %6, %c3_i32 : i32
      %8 = arith.sitofp %7 : i32 to f32
      %9 = arith.mulf %8, %cst_23 : f32
      %10 = affine.load %alloca_28[%arg0, 0] : memref<24x3xf32>
      %11 = arith.subf %10, %9 : f32
      affine.store %11, %alloca_28[%arg0, 0] : memref<24x3xf32>
      %12 = arith.remsi %6, %c2_i32 : i32
      %13 = arith.sitofp %12 : i32 to f32
      %14 = arith.mulf %13, %cst_24 : f32
      %15 = affine.load %alloca_28[%arg0, 1] : memref<24x3xf32>
      %16 = arith.addf %15, %14 : f32
      affine.store %16, %alloca_28[%arg0, 1] : memref<24x3xf32>
      %17 = arith.remsi %6, %c4_i32 : i32
      %18 = arith.addi %17, %c-1_i32 : i32
      %19 = arith.sitofp %18 : i32 to f32
      %20 = arith.mulf %19, %cst_25 : f32
      %21 = affine.load %alloca_28[%arg0, 2] : memref<24x3xf32>
      %22 = arith.addf %21, %20 : f32
      affine.store %22, %alloca_28[%arg0, 2] : memref<24x3xf32>
    }
    %alloca_34 = memref.alloca() : memref<1xf32>
    affine.store %cst_8, %alloca_34[0] : memref<1xf32>
    %alloca_35 = memref.alloca() : memref<1xf32>
    affine.store %cst_8, %alloca_35[0] : memref<1xf32>
    %alloca_36 = memref.alloca() : memref<1xf32>
    affine.store %cst_8, %alloca_36[0] : memref<1xf32>
    %alloca_37 = memref.alloca() : memref<24x3xf32>
    %alloca_38 = memref.alloca() : memref<24x3xf32>
    %alloca_39 = memref.alloca() : memref<8x3xf32>
    %alloca_40 = memref.alloca() : memref<8x3xf32>
    affine.for %arg0 = 0 to 6 {
      affine.for %arg1 = 0 to 8 {
        affine.for %arg2 = 0 to 3 {
          affine.store %cst_8, %alloca_40[%arg1, %arg2] : memref<8x3xf32>
          affine.store %cst_5, %alloca_39[%arg1, %arg2] : memref<8x3xf32>
        }
      }
      affine.for %arg1 = 0 to 24 {
        affine.for %arg2 = 0 to 3 {
          affine.store %cst_8, %alloca_38[%arg1, %arg2] : memref<24x3xf32>
          affine.store %cst_5, %alloca_37[%arg1, %arg2] : memref<24x3xf32>
        }
      }
      affine.for %arg1 = 0 to 192 {
        %6 = affine.load %alloca_27[%arg1] : memref<192xi32>
        %7 = affine.load %alloca_26[%arg1] : memref<192xi32>
        %8 = arith.index_cast %6 : i32 to index
        %9 = arith.index_cast %7 : i32 to index
        %10 = memref.load %alloca_28[%9, %c0] : memref<24x3xf32>
        %11 = memref.load %alloca_29[%8, %c0] : memref<8x3xf32>
        %12 = arith.subf %10, %11 : f32
        %13 = memref.load %alloca_28[%9, %c1] : memref<24x3xf32>
        %14 = memref.load %alloca_29[%8, %c1] : memref<8x3xf32>
        %15 = arith.subf %13, %14 : f32
        %16 = memref.load %alloca_28[%9, %c2] : memref<24x3xf32>
        %17 = memref.load %alloca_29[%8, %c2] : memref<8x3xf32>
        %18 = arith.subf %16, %17 : f32
        %19 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
        %20 = affine.load %19[0] : memref<1xf32>
        %21 = arith.addf %18, %20 : f32
        %22 = arith.divf %cst_7, %21 : f32
        affine.store %22, %alloca_34[0] : memref<1xf32>
        %23 = memref.get_global @_ZL2FX : memref<1xf32>
        %24 = affine.load %23[0] : memref<1xf32>
        %25 = arith.mulf %24, %12 : f32
        %26 = affine.load %alloca_34[0] : memref<1xf32>
        %27 = arith.mulf %25, %26 : f32
        affine.store %27, %alloca_36[0] : memref<1xf32>
        %28 = memref.get_global @_ZL2FY : memref<1xf32>
        %29 = affine.load %28[0] : memref<1xf32>
        %30 = arith.mulf %29, %15 : f32
        %31 = affine.load %alloca_34[0] : memref<1xf32>
        %32 = arith.mulf %30, %31 : f32
        affine.store %32, %alloca_35[0] : memref<1xf32>
        %33 = memref.load %alloca_28[%9, %c0] : memref<24x3xf32>
        %34 = memref.load %alloca_29[%8, %c0] : memref<8x3xf32>
        %35 = arith.subf %33, %34 : f32
        %36 = memref.load %alloca_28[%9, %c1] : memref<24x3xf32>
        %37 = memref.load %alloca_29[%8, %c1] : memref<8x3xf32>
        %38 = arith.subf %36, %37 : f32
        %39 = affine.load %alloca_36[0] : memref<1xf32>
        %40 = affine.load %alloca[%arg1, 0] : memref<192x2xf32>
        %41 = arith.subf %39, %40 : f32
        %42 = affine.load %alloca_35[0] : memref<1xf32>
        %43 = affine.load %alloca[%arg1, 1] : memref<192x2xf32>
        %44 = arith.subf %42, %43 : f32
        %45 = affine.load %23[0] : memref<1xf32>
        %46 = affine.load %alloca_34[0] : memref<1xf32>
        %47 = arith.mulf %45, %46 : f32
        %48 = affine.load %28[0] : memref<1xf32>
        %49 = arith.mulf %48, %46 : f32
        %50 = arith.negf %45 : f32
        %51 = arith.mulf %50, %35 : f32
        %52 = arith.mulf %51, %46 : f32
        %53 = arith.mulf %52, %46 : f32
        %54 = arith.negf %48 : f32
        %55 = arith.mulf %54, %38 : f32
        %56 = arith.mulf %55, %46 : f32
        %57 = arith.mulf %56, %46 : f32
        %58 = arith.negf %41 : f32
        %59 = arith.mulf %58, %47 : f32
        %60 = memref.load %alloca_40[%8, %c0] : memref<8x3xf32>
        %61 = arith.addf %60, %59 : f32
        memref.store %61, %alloca_40[%8, %c0] : memref<8x3xf32>
        %62 = arith.negf %44 : f32
        %63 = arith.mulf %62, %49 : f32
        %64 = memref.load %alloca_40[%8, %c1] : memref<8x3xf32>
        %65 = arith.addf %64, %63 : f32
        memref.store %65, %alloca_40[%8, %c1] : memref<8x3xf32>
        %66 = arith.mulf %41, %53 : f32
        %67 = arith.mulf %44, %57 : f32
        %68 = arith.addf %66, %67 : f32
        %69 = arith.negf %68 : f32
        %70 = memref.load %alloca_40[%8, %c2] : memref<8x3xf32>
        %71 = arith.addf %70, %69 : f32
        memref.store %71, %alloca_40[%8, %c2] : memref<8x3xf32>
        %72 = arith.mulf %47, %47 : f32
        %73 = memref.load %alloca_39[%8, %c0] : memref<8x3xf32>
        %74 = arith.addf %73, %72 : f32
        memref.store %74, %alloca_39[%8, %c0] : memref<8x3xf32>
        %75 = arith.mulf %49, %49 : f32
        %76 = memref.load %alloca_39[%8, %c1] : memref<8x3xf32>
        %77 = arith.addf %76, %75 : f32
        memref.store %77, %alloca_39[%8, %c1] : memref<8x3xf32>
        %78 = arith.mulf %53, %53 : f32
        %79 = arith.mulf %57, %57 : f32
        %80 = arith.addf %78, %79 : f32
        %81 = memref.load %alloca_39[%8, %c2] : memref<8x3xf32>
        %82 = arith.addf %81, %80 : f32
        memref.store %82, %alloca_39[%8, %c2] : memref<8x3xf32>
        %83 = arith.mulf %41, %47 : f32
        %84 = memref.load %alloca_38[%9, %c0] : memref<24x3xf32>
        %85 = arith.addf %84, %83 : f32
        memref.store %85, %alloca_38[%9, %c0] : memref<24x3xf32>
        %86 = arith.mulf %44, %49 : f32
        %87 = memref.load %alloca_38[%9, %c1] : memref<24x3xf32>
        %88 = arith.addf %87, %86 : f32
        memref.store %88, %alloca_38[%9, %c1] : memref<24x3xf32>
        %89 = memref.load %alloca_38[%9, %c2] : memref<24x3xf32>
        %90 = arith.addf %89, %68 : f32
        memref.store %90, %alloca_38[%9, %c2] : memref<24x3xf32>
        %91 = memref.load %alloca_37[%9, %c0] : memref<24x3xf32>
        %92 = arith.addf %91, %72 : f32
        memref.store %92, %alloca_37[%9, %c0] : memref<24x3xf32>
        %93 = memref.load %alloca_37[%9, %c1] : memref<24x3xf32>
        %94 = arith.addf %93, %75 : f32
        memref.store %94, %alloca_37[%9, %c1] : memref<24x3xf32>
        %95 = memref.load %alloca_37[%9, %c2] : memref<24x3xf32>
        %96 = arith.addf %95, %80 : f32
        memref.store %96, %alloca_37[%9, %c2] : memref<24x3xf32>
      }
      affine.for %arg1 = 0 to 8 {
        affine.for %arg2 = 0 to 3 {
          %6 = affine.load %alloca_40[%arg1, %arg2] : memref<8x3xf32>
          %7 = arith.mulf %6, %cst_18 : f32
          %8 = affine.load %alloca_39[%arg1, %arg2] : memref<8x3xf32>
          %9 = arith.divf %7, %8 : f32
          %10 = affine.load %alloca_29[%arg1, %arg2] : memref<8x3xf32>
          %11 = arith.subf %10, %9 : f32
          affine.store %11, %alloca_29[%arg1, %arg2] : memref<8x3xf32>
        }
      }
      affine.for %arg1 = 0 to 24 {
        affine.for %arg2 = 0 to 3 {
          %6 = affine.load %alloca_38[%arg1, %arg2] : memref<24x3xf32>
          %7 = arith.mulf %6, %cst_6 : f32
          %8 = affine.load %alloca_37[%arg1, %arg2] : memref<24x3xf32>
          %9 = arith.divf %7, %8 : f32
          %10 = affine.load %alloca_28[%arg1, %arg2] : memref<24x3xf32>
          %11 = arith.subf %10, %9 : f32
          affine.store %11, %alloca_28[%arg1, %arg2] : memref<24x3xf32>
        }
      }
    }
    %alloca_41 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_41[] : memref<i32>
    affine.store %c0_i32, %alloca_41[] : memref<i32>
    affine.for %arg0 = 0 to 8 {
      %6 = affine.load %alloca_29[%arg0, 0] : memref<8x3xf32>
      %7 = arith.mulf %6, %cst : f32
      %8 = arith.fptosi %7 : f32 to i32
      %9 = affine.load %alloca_41[] : memref<i32>
      %10 = arith.addi %9, %8 : i32
      affine.store %10, %alloca_41[] : memref<i32>
      %11 = affine.load %alloca_29[%arg0, 1] : memref<8x3xf32>
      %12 = arith.mulf %11, %cst_0 : f32
      %13 = arith.fptosi %12 : f32 to i32
      %14 = arith.addi %10, %13 : i32
      affine.store %14, %alloca_41[] : memref<i32>
      %15 = affine.load %alloca_29[%arg0, 2] : memref<8x3xf32>
      %16 = arith.mulf %15, %cst_1 : f32
      %17 = arith.fptosi %16 : f32 to i32
      %18 = arith.addi %14, %17 : i32
      affine.store %18, %alloca_41[] : memref<i32>
    }
    affine.for %arg0 = 0 to 24 {
      %6 = affine.load %alloca_28[%arg0, 0] : memref<24x3xf32>
      %7 = arith.mulf %6, %cst_2 : f32
      %8 = arith.fptosi %7 : f32 to i32
      %9 = affine.load %alloca_41[] : memref<i32>
      %10 = arith.addi %9, %8 : i32
      affine.store %10, %alloca_41[] : memref<i32>
      %11 = affine.load %alloca_28[%arg0, 1] : memref<24x3xf32>
      %12 = arith.mulf %11, %cst_3 : f32
      %13 = arith.fptosi %12 : f32 to i32
      %14 = arith.addi %10, %13 : i32
      affine.store %14, %alloca_41[] : memref<i32>
      %15 = affine.load %alloca_28[%arg0, 2] : memref<24x3xf32>
      %16 = arith.mulf %15, %cst_4 : f32
      %17 = arith.fptosi %16 : f32 to i32
      %18 = arith.addi %14, %17 : i32
      affine.store %18, %alloca_41[] : memref<i32>
    }
    %0 = affine.load %alloca_41[] : memref<i32>
    %1 = arith.sitofp %0 : i32 to f32
    %2 = arith.cmpi slt, %0, %c0_i32 : i32
    %3 = scf.if %2 -> (f32) {
      %6 = arith.negf %1 : f32
      scf.yield %6 : f32
    } else {
      scf.yield %1 : f32
    }
    %4 = arith.cmpf ogt, %3, %cst_8 : f32
    %5 = arith.select %4, %0, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_29, %alloca_28, %alloca_27, %alloca_26, %alloca) : (memref<8x3xf32>, memref<24x3xf32>, memref<192xi32>, memref<192xi32>, memref<192x2xf32>) -> ()
    return %5 : i32
  }
  func.func private @pcl_validate_outputs(memref<8x3xf32>, memref<24x3xf32>, memref<192xi32>, memref<192xi32>, memref<192x2xf32>)
}

