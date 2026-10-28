module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+03 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 1.200000e+03 : f32
    %cst_1 = arith.constant 1.400000e+03 : f32
    %cst_2 = arith.constant 5.000000e+02 : f32
    %c0 = arith.constant 0 : index
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_4 = arith.constant 1.000000e+30 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %cst_6 = arith.constant 3.200000e+01 : f32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c7_i32 = arith.constant 7 : i32
    %c41_i32 = arith.constant 41 : i32
    %cst_7 = arith.constant 3.000000e-02 : f32
    %c11_i32 = arith.constant 11 : i32
    %c37_i32 = arith.constant 37 : i32
    %cst_8 = arith.constant 2.500000e-02 : f32
    %c13_i32 = arith.constant 13 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_9 = arith.constant 2.000000e-02 : f32
    %cst_10 = arith.constant 8.000000e-02 : f32
    %cst_11 = arith.constant 4.000000e-02 : f32
    %cst_12 = arith.constant 6.000000e-02 : f32
    %alloca = memref.alloca() : memref<4xf32>
    %alloca_13 = memref.alloca() : memref<3xf32>
    %alloca_14 = memref.alloca() : memref<32x3xf32>
    %alloca_15 = memref.alloca() : memref<32x3xf32>
    affine.for %arg0 = 0 to 32 {
      %20 = arith.index_cast %arg0 : index to i32
      %21 = arith.muli %20, %c7_i32 : i32
      %22 = arith.remsi %21, %c41_i32 : i32
      %23 = arith.sitofp %22 : i32 to f32
      %24 = arith.mulf %23, %cst_7 : f32
      %25 = arith.muli %20, %c11_i32 : i32
      %26 = arith.remsi %25, %c37_i32 : i32
      %27 = arith.sitofp %26 : i32 to f32
      %28 = arith.mulf %27, %cst_8 : f32
      %29 = arith.muli %20, %c13_i32 : i32
      %30 = arith.remsi %29, %c31_i32 : i32
      %31 = arith.sitofp %30 : i32 to f32
      %32 = arith.mulf %31, %cst_9 : f32
      affine.store %24, %alloca_15[%arg0, 0] : memref<32x3xf32>
      affine.store %28, %alloca_15[%arg0, 1] : memref<32x3xf32>
      affine.store %32, %alloca_15[%arg0, 2] : memref<32x3xf32>
      %33 = arith.addf %24, %cst_10 : f32
      affine.store %33, %alloca_14[%arg0, 0] : memref<32x3xf32>
      %34 = arith.subf %28, %cst_11 : f32
      affine.store %34, %alloca_14[%arg0, 1] : memref<32x3xf32>
      %35 = arith.addf %32, %cst_12 : f32
      affine.store %35, %alloca_14[%arg0, 2] : memref<32x3xf32>
    }
    %alloca_16 = memref.alloca() : memref<f32>
    affine.store %cst_3, %alloca_16[] : memref<f32>
    %alloca_17 = memref.alloca() : memref<f32>
    affine.store %cst_3, %alloca_17[] : memref<f32>
    %alloca_18 = memref.alloca() : memref<f32>
    affine.store %cst_3, %alloca_18[] : memref<f32>
    affine.store %cst_3, %alloca_18[] : memref<f32>
    affine.store %cst_3, %alloca_17[] : memref<f32>
    affine.store %cst_3, %alloca_16[] : memref<f32>
    affine.for %arg0 = 0 to 4 {
      %alloca_20 = memref.alloca() : memref<f32>
      affine.store %cst_3, %alloca_20[] : memref<f32>
      %alloca_21 = memref.alloca() : memref<f32>
      affine.store %cst_3, %alloca_21[] : memref<f32>
      %alloca_22 = memref.alloca() : memref<f32>
      affine.store %cst_3, %alloca_22[] : memref<f32>
      %alloca_23 = memref.alloca() : memref<f32>
      affine.store %cst_3, %alloca_23[] : memref<f32>
      affine.for %arg1 = 0 to 32 {
        %alloca_24 = memref.alloca() : memref<f32>
        affine.store %cst_4, %alloca_24[] : memref<f32>
        %alloca_25 = memref.alloca() : memref<i32>
        affine.store %c0_i32, %alloca_25[] : memref<i32>
        affine.for %arg2 = 0 to 32 {
          %63 = arith.index_cast %arg2 : index to i32
          %64 = affine.load %alloca_18[] : memref<f32>
          %65 = affine.load %alloca_17[] : memref<f32>
          %66 = affine.load %alloca_16[] : memref<f32>
          %67 = affine.load %alloca_15[%arg1, 0] : memref<32x3xf32>
          %68 = arith.addf %67, %64 : f32
          %69 = affine.load %alloca_14[%arg2, 0] : memref<32x3xf32>
          %70 = arith.subf %68, %69 : f32
          %71 = affine.load %alloca_15[%arg1, 1] : memref<32x3xf32>
          %72 = arith.addf %71, %65 : f32
          %73 = affine.load %alloca_14[%arg2, 1] : memref<32x3xf32>
          %74 = arith.subf %72, %73 : f32
          %75 = affine.load %alloca_15[%arg1, 2] : memref<32x3xf32>
          %76 = arith.addf %75, %66 : f32
          %77 = affine.load %alloca_14[%arg2, 2] : memref<32x3xf32>
          %78 = arith.subf %76, %77 : f32
          %79 = arith.mulf %70, %70 : f32
          %80 = arith.mulf %74, %74 : f32
          %81 = arith.addf %79, %80 : f32
          %82 = arith.mulf %78, %78 : f32
          %83 = arith.addf %81, %82 : f32
          %84 = affine.load %alloca_24[] : memref<f32>
          %85 = arith.cmpf olt, %83, %84 : f32
          scf.if %85 {
            affine.store %83, %alloca_24[] : memref<f32>
            affine.store %63, %alloca_25[] : memref<i32>
          }
        }
        %37 = affine.load %alloca_25[] : memref<i32>
        %38 = arith.index_cast %37 : i32 to index
        %39 = memref.load %alloca_14[%38, %c0] : memref<32x3xf32>
        %40 = affine.load %alloca_15[%arg1, 0] : memref<32x3xf32>
        %41 = affine.load %alloca_18[] : memref<f32>
        %42 = arith.addf %40, %41 : f32
        %43 = arith.subf %39, %42 : f32
        %44 = affine.load %alloca_20[] : memref<f32>
        %45 = arith.addf %44, %43 : f32
        affine.store %45, %alloca_20[] : memref<f32>
        %46 = memref.load %alloca_14[%38, %c1] : memref<32x3xf32>
        %47 = affine.load %alloca_15[%arg1, 1] : memref<32x3xf32>
        %48 = affine.load %alloca_17[] : memref<f32>
        %49 = arith.addf %47, %48 : f32
        %50 = arith.subf %46, %49 : f32
        %51 = affine.load %alloca_21[] : memref<f32>
        %52 = arith.addf %51, %50 : f32
        affine.store %52, %alloca_21[] : memref<f32>
        %53 = memref.load %alloca_14[%38, %c2] : memref<32x3xf32>
        %54 = affine.load %alloca_15[%arg1, 2] : memref<32x3xf32>
        %55 = affine.load %alloca_16[] : memref<f32>
        %56 = arith.addf %54, %55 : f32
        %57 = arith.subf %53, %56 : f32
        %58 = affine.load %alloca_22[] : memref<f32>
        %59 = arith.addf %58, %57 : f32
        affine.store %59, %alloca_22[] : memref<f32>
        %60 = affine.load %alloca_24[] : memref<f32>
        %61 = affine.load %alloca_23[] : memref<f32>
        %62 = arith.addf %61, %60 : f32
        affine.store %62, %alloca_23[] : memref<f32>
      }
      %20 = affine.load %alloca_20[] : memref<f32>
      %21 = arith.mulf %20, %cst_5 : f32
      %22 = arith.divf %21, %cst_6 : f32
      %23 = affine.load %alloca_18[] : memref<f32>
      %24 = arith.addf %23, %22 : f32
      affine.store %24, %alloca_18[] : memref<f32>
      %25 = affine.load %alloca_21[] : memref<f32>
      %26 = arith.mulf %25, %cst_5 : f32
      %27 = arith.divf %26, %cst_6 : f32
      %28 = affine.load %alloca_17[] : memref<f32>
      %29 = arith.addf %28, %27 : f32
      affine.store %29, %alloca_17[] : memref<f32>
      %30 = affine.load %alloca_22[] : memref<f32>
      %31 = arith.mulf %30, %cst_5 : f32
      %32 = arith.divf %31, %cst_6 : f32
      %33 = affine.load %alloca_16[] : memref<f32>
      %34 = arith.addf %33, %32 : f32
      affine.store %34, %alloca_16[] : memref<f32>
      %35 = affine.load %alloca_23[] : memref<f32>
      %36 = arith.divf %35, %cst_6 : f32
      affine.store %36, %alloca[%arg0] : memref<4xf32>
    }
    %0 = affine.load %alloca_18[] : memref<f32>
    affine.store %0, %alloca_13[0] : memref<3xf32>
    %1 = affine.load %alloca_17[] : memref<f32>
    affine.store %1, %alloca_13[1] : memref<3xf32>
    %2 = affine.load %alloca_16[] : memref<f32>
    affine.store %2, %alloca_13[2] : memref<3xf32>
    %alloca_19 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_19[] : memref<i32>
    affine.store %c0_i32, %alloca_19[] : memref<i32>
    %3 = affine.load %alloca_13[0] : memref<3xf32>
    %4 = arith.mulf %3, %cst : f32
    %5 = arith.fptosi %4 : f32 to i32
    affine.store %5, %alloca_19[] : memref<i32>
    %6 = affine.load %alloca_13[1] : memref<3xf32>
    %7 = arith.mulf %6, %cst_0 : f32
    %8 = arith.fptosi %7 : f32 to i32
    %9 = arith.addi %5, %8 : i32
    affine.store %9, %alloca_19[] : memref<i32>
    %10 = affine.load %alloca_13[2] : memref<3xf32>
    %11 = arith.mulf %10, %cst_1 : f32
    %12 = arith.fptosi %11 : f32 to i32
    %13 = arith.addi %9, %12 : i32
    affine.store %13, %alloca_19[] : memref<i32>
    affine.for %arg0 = 0 to 4 {
      %20 = affine.load %alloca[%arg0] : memref<4xf32>
      %21 = arith.mulf %20, %cst_2 : f32
      %22 = arith.fptosi %21 : f32 to i32
      %23 = affine.load %alloca_19[] : memref<i32>
      %24 = arith.addi %23, %22 : i32
      affine.store %24, %alloca_19[] : memref<i32>
    }
    %14 = affine.load %alloca_19[] : memref<i32>
    %15 = arith.sitofp %14 : i32 to f32
    %16 = arith.cmpi slt, %14, %c0_i32 : i32
    %17 = scf.if %16 -> (f32) {
      %20 = arith.negf %15 : f32
      scf.yield %20 : f32
    } else {
      scf.yield %15 : f32
    }
    %18 = arith.cmpf ogt, %17, %cst_3 : f32
    %19 = arith.select %18, %14, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_15, %alloca_14, %alloca_13, %alloca) : (memref<32x3xf32>, memref<32x3xf32>, memref<3xf32>, memref<4xf32>) -> ()
    return %19 : i32
  }
  func.func private @pcl_validate_outputs(memref<32x3xf32>, memref<32x3xf32>, memref<3xf32>, memref<4xf32>)
}

