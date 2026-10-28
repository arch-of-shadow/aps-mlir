module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+03 : f32
    %cst_0 = arith.constant 1.100000e+03 : f32
    %cst_1 = arith.constant 1.200000e+03 : f32
    %cst_2 = arith.constant 1.300000e+03 : f32
    %c0 = arith.constant 0 : index
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_4 = arith.constant 1.000000e+30 : f32
    %c11_i32 = arith.constant 11 : i32
    %c32_i32 = arith.constant 32 : i32
    %c23_i32 = arith.constant 23 : i32
    %c29_i32 = arith.constant 29 : i32
    %cst_5 = arith.constant 2.500000e-01 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_6 = arith.constant 8.000000e-02 : f32
    %false = arith.constant false
    %true = arith.constant true
    %c768_i32 = arith.constant 768 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_7 = arith.constant 6.000000e-01 : f32
    %cst_8 = arith.constant 2.000000e-01 : f32
    %cst_9 = arith.constant 6.400000e+01 : f32
    %cst_10 = arith.constant 4.000000e-01 : f32
    %cst_11 = arith.constant 1.200000e+01 : f32
    %cst_12 = arith.constant 3.000000e-01 : f32
    %c31_i32 = arith.constant 31 : i32
    %cst_13 = arith.constant 3.100000e+01 : f32
    %c13_i32 = arith.constant 13 : i32
    %c3_i32 = arith.constant 3 : i32
    %c101_i32 = arith.constant 101 : i32
    %cst_14 = arith.constant 1.010000e+02 : f32
    %c17_i32 = arith.constant 17 : i32
    %c5_i32 = arith.constant 5 : i32
    %c97_i32 = arith.constant 97 : i32
    %cst_15 = arith.constant 9.700000e+01 : f32
    %c19_i32 = arith.constant 19 : i32
    %c7_i32 = arith.constant 7 : i32
    %c89_i32 = arith.constant 89 : i32
    %cst_16 = arith.constant 8.900000e+01 : f32
    %alloca = memref.alloca() : memref<4xi32>
    %alloca_17 = memref.alloca() : memref<4xf32>
    %alloca_18 = memref.alloca() : memref<32x3xf32>
    affine.for %arg0 = 0 to 32 {
      %16 = arith.index_cast %arg0 : index to i32
      %17 = arith.cmpi slt, %16, %c768_i32 : i32
      scf.if %17 {
        %18 = arith.remsi %16, %c64_i32 : i32
        %19 = arith.divsi %16, %c64_i32 : i32
        %20 = arith.sitofp %18 : i32 to f32
        %21 = arith.mulf %20, %cst_8 : f32
        %22 = arith.divf %21, %cst_9 : f32
        %23 = arith.addf %22, %cst_7 : f32
        affine.store %23, %alloca_18[%arg0, 0] : memref<32x3xf32>
        %24 = arith.sitofp %19 : i32 to f32
        %25 = arith.mulf %24, %cst_8 : f32
        %26 = arith.divf %25, %cst_11 : f32
        %27 = arith.addf %26, %cst_10 : f32
        affine.store %27, %alloca_18[%arg0, 1] : memref<32x3xf32>
        %28 = arith.addi %18, %19 : i32
        %29 = arith.remsi %28, %c31_i32 : i32
        %30 = arith.sitofp %29 : i32 to f32
        %31 = arith.mulf %30, %cst_8 : f32
        %32 = arith.divf %31, %cst_13 : f32
        %33 = arith.addf %32, %cst_12 : f32
        affine.store %33, %alloca_18[%arg0, 2] : memref<32x3xf32>
      } else {
        %18 = arith.muli %16, %c13_i32 : i32
        %19 = arith.addi %18, %c3_i32 : i32
        %20 = arith.remsi %19, %c101_i32 : i32
        %21 = arith.sitofp %20 : i32 to f32
        %22 = arith.divf %21, %cst_14 : f32
        affine.store %22, %alloca_18[%arg0, 0] : memref<32x3xf32>
        %23 = arith.muli %16, %c17_i32 : i32
        %24 = arith.addi %23, %c5_i32 : i32
        %25 = arith.remsi %24, %c97_i32 : i32
        %26 = arith.sitofp %25 : i32 to f32
        %27 = arith.divf %26, %cst_15 : f32
        affine.store %27, %alloca_18[%arg0, 1] : memref<32x3xf32>
        %28 = arith.muli %16, %c19_i32 : i32
        %29 = arith.addi %28, %c7_i32 : i32
        %30 = arith.remsi %29, %c89_i32 : i32
        %31 = arith.sitofp %30 : i32 to f32
        %32 = arith.divf %31, %cst_16 : f32
        affine.store %32, %alloca_18[%arg0, 2] : memref<32x3xf32>
      }
    }
    %alloca_19 = memref.alloca() : memref<1xi32>
    affine.store %c0_i32, %alloca_19[0] : memref<1xi32>
    %alloca_20 = memref.alloca() : memref<1xi32>
    affine.store %c0_i32, %alloca_20[0] : memref<1xi32>
    %alloca_21 = memref.alloca() : memref<1xi32>
    %cast = memref.cast %alloca_21 : memref<1xi32> to memref<?xi32>
    affine.store %c0_i32, %alloca_21[0] : memref<1xi32>
    %alloca_22 = memref.alloca() : memref<1xi32>
    %cast_23 = memref.cast %alloca_22 : memref<1xi32> to memref<?xi32>
    affine.store %c0_i32, %alloca_22[0] : memref<1xi32>
    %alloca_24 = memref.alloca() : memref<f32>
    affine.store %cst_3, %alloca_24[] : memref<f32>
    %alloca_25 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_25[] : memref<i32>
    affine.store %c-1_i32, %alloca_25[] : memref<i32>
    affine.store %cst_4, %alloca_24[] : memref<f32>
    affine.for %arg0 = 0 to 4 {
      %16 = arith.index_cast %arg0 : index to i32
      %17 = arith.muli %16, %c7_i32 : i32
      %18 = arith.addi %17, %c11_i32 : i32
      %19 = arith.remsi %18, %c32_i32 : i32
      affine.store %19, %alloca_22[0] : memref<1xi32>
      %20 = arith.muli %16, %c13_i32 : i32
      %21 = arith.addi %20, %c19_i32 : i32
      %22 = arith.remsi %21, %c32_i32 : i32
      affine.store %22, %alloca_21[0] : memref<1xi32>
      %23 = arith.muli %16, %c17_i32 : i32
      %24 = arith.addi %23, %c23_i32 : i32
      %25 = arith.remsi %24, %c32_i32 : i32
      affine.store %25, %alloca_20[0] : memref<1xi32>
      %26 = arith.muli %16, %c29_i32 : i32
      %27 = arith.addi %26, %c31_i32 : i32
      %28 = arith.remsi %27, %c32_i32 : i32
      affine.store %28, %alloca_19[0] : memref<1xi32>
      %29 = affine.load %alloca_22[0] : memref<1xi32>
      %30 = arith.index_cast %29 : i32 to index
      %31 = memref.load %alloca_18[%30, %c0] : memref<32x3xf32>
      %32 = affine.load %alloca_21[0] : memref<1xi32>
      %33 = arith.index_cast %32 : i32 to index
      %34 = memref.load %alloca_18[%33, %c0] : memref<32x3xf32>
      %35 = arith.addf %31, %34 : f32
      %36 = affine.load %alloca_20[0] : memref<1xi32>
      %37 = arith.index_cast %36 : i32 to index
      %38 = memref.load %alloca_18[%37, %c0] : memref<32x3xf32>
      %39 = arith.addf %35, %38 : f32
      %40 = affine.load %alloca_19[0] : memref<1xi32>
      %41 = arith.index_cast %40 : i32 to index
      %42 = memref.load %alloca_18[%41, %c0] : memref<32x3xf32>
      %43 = arith.addf %39, %42 : f32
      %44 = arith.mulf %43, %cst_5 : f32
      %45 = memref.load %alloca_18[%30, %c1] : memref<32x3xf32>
      %46 = memref.load %alloca_18[%33, %c1] : memref<32x3xf32>
      %47 = arith.addf %45, %46 : f32
      %48 = memref.load %alloca_18[%37, %c1] : memref<32x3xf32>
      %49 = arith.addf %47, %48 : f32
      %50 = memref.load %alloca_18[%41, %c1] : memref<32x3xf32>
      %51 = arith.addf %49, %50 : f32
      %52 = arith.mulf %51, %cst_5 : f32
      %53 = memref.load %alloca_18[%30, %c2] : memref<32x3xf32>
      %54 = memref.load %alloca_18[%33, %c2] : memref<32x3xf32>
      %55 = arith.addf %53, %54 : f32
      %56 = memref.load %alloca_18[%37, %c2] : memref<32x3xf32>
      %57 = arith.addf %55, %56 : f32
      %58 = memref.load %alloca_18[%41, %c2] : memref<32x3xf32>
      %59 = arith.addf %57, %58 : f32
      %60 = arith.mulf %59, %cst_5 : f32
      %alloca_27 = memref.alloca() : memref<f32>
      affine.store %cst_3, %alloca_27[] : memref<f32>
      %alloca_28 = memref.alloca() : memref<i32>
      affine.store %c0_i32, %alloca_28[] : memref<i32>
      %alloca_29 = memref.alloca() : memref<f32>
      affine.store %cst_3, %alloca_29[] : memref<f32>
      affine.for %arg1 = 0 to 4 {
        %67 = arith.index_cast %arg1 : index to i32
        %68 = arith.cmpi eq, %67, %c0_i32 : i32
        %69 = scf.if %68 -> (memref<?xi32>) {
          scf.yield %cast_23 : memref<?xi32>
        } else {
          %88 = arith.cmpi eq, %67, %c1_i32 : i32
          %89 = scf.if %88 -> (memref<?xi32>) {
            scf.yield %cast : memref<?xi32>
          } else {
            %90 = arith.cmpi eq, %67, %c2_i32 : i32
            %91 = arith.select %90, %alloca_20, %alloca_19 : memref<1xi32>
            %cast_30 = memref.cast %91 : memref<1xi32> to memref<?xi32>
            scf.yield %cast_30 : memref<?xi32>
          }
          scf.yield %89 : memref<?xi32>
        }
        %70 = affine.load %69[0] : memref<?xi32>
        %71 = arith.index_cast %70 : i32 to index
        %72 = memref.load %alloca_18[%71, %c0] : memref<32x3xf32>
        %73 = arith.subf %72, %44 : f32
        %74 = memref.load %alloca_18[%71, %c1] : memref<32x3xf32>
        %75 = arith.subf %74, %52 : f32
        %76 = memref.load %alloca_18[%71, %c2] : memref<32x3xf32>
        %77 = arith.subf %76, %60 : f32
        %78 = arith.cmpf olt, %73, %cst_3 : f32
        %79 = scf.if %78 -> (f32) {
          %88 = arith.negf %73 : f32
          scf.yield %88 : f32
        } else {
          scf.yield %73 : f32
        }
        %80 = arith.cmpf olt, %75, %cst_3 : f32
        %81 = scf.if %80 -> (f32) {
          %88 = arith.negf %75 : f32
          scf.yield %88 : f32
        } else {
          scf.yield %75 : f32
        }
        %82 = arith.addf %79, %81 : f32
        %83 = arith.cmpf olt, %77, %cst_3 : f32
        %84 = scf.if %83 -> (f32) {
          %88 = arith.negf %77 : f32
          scf.yield %88 : f32
        } else {
          scf.yield %77 : f32
        }
        %85 = arith.addf %82, %84 : f32
        %86 = affine.load %alloca_27[] : memref<f32>
        %87 = arith.addf %86, %85 : f32
        affine.store %87, %alloca_27[] : memref<f32>
      }
      %61 = affine.load %alloca_27[] : memref<f32>
      %62 = arith.mulf %61, %cst_5 : f32
      affine.store %62, %alloca_27[] : memref<f32>
      affine.for %arg1 = 0 to 32 {
        %67 = affine.load %alloca_18[%arg1, 0] : memref<32x3xf32>
        %68 = arith.subf %67, %44 : f32
        %69 = affine.load %alloca_18[%arg1, 1] : memref<32x3xf32>
        %70 = arith.subf %69, %52 : f32
        %71 = affine.load %alloca_18[%arg1, 2] : memref<32x3xf32>
        %72 = arith.subf %71, %60 : f32
        %73 = arith.cmpf olt, %68, %cst_3 : f32
        %74 = scf.if %73 -> (f32) {
          %87 = arith.negf %68 : f32
          scf.yield %87 : f32
        } else {
          scf.yield %68 : f32
        }
        %75 = arith.cmpf olt, %70, %cst_3 : f32
        %76 = scf.if %75 -> (f32) {
          %87 = arith.negf %70 : f32
          scf.yield %87 : f32
        } else {
          scf.yield %70 : f32
        }
        %77 = arith.addf %74, %76 : f32
        %78 = arith.cmpf olt, %72, %cst_3 : f32
        %79 = scf.if %78 -> (f32) {
          %87 = arith.negf %72 : f32
          scf.yield %87 : f32
        } else {
          scf.yield %72 : f32
        }
        %80 = arith.addf %77, %79 : f32
        %81 = arith.subf %80, %62 : f32
        %82 = arith.cmpf olt, %81, %cst_3 : f32
        %83 = scf.if %82 -> (f32) {
          %87 = arith.negf %81 : f32
          scf.yield %87 : f32
        } else {
          scf.yield %81 : f32
        }
        %84 = arith.cmpf olt, %83, %cst_6 : f32
        scf.if %84 {
          %87 = affine.load %alloca_28[] : memref<i32>
          %88 = arith.addi %87, %c1_i32 : i32
          affine.store %88, %alloca_28[] : memref<i32>
        }
        %85 = affine.load %alloca_29[] : memref<f32>
        %86 = arith.addf %85, %83 : f32
        affine.store %86, %alloca_29[] : memref<f32>
      }
      %63 = affine.load %alloca_28[] : memref<i32>
      affine.store %63, %alloca[%arg0] : memref<4xi32>
      %64 = affine.load %alloca_25[] : memref<i32>
      %65 = arith.cmpi sgt, %63, %64 : i32
      %66 = scf.if %65 -> (i1) {
        scf.yield %true : i1
      } else {
        %67 = arith.cmpi eq, %63, %64 : i32
        %68 = scf.if %67 -> (i1) {
          %69 = affine.load %alloca_29[] : memref<f32>
          %70 = affine.load %alloca_24[] : memref<f32>
          %71 = arith.cmpf olt, %69, %70 : f32
          scf.yield %71 : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %68 : i1
      }
      scf.if %66 {
        affine.store %63, %alloca_25[] : memref<i32>
        %67 = affine.load %alloca_29[] : memref<f32>
        affine.store %67, %alloca_24[] : memref<f32>
        affine.store %44, %alloca_17[0] : memref<4xf32>
        affine.store %52, %alloca_17[1] : memref<4xf32>
        affine.store %60, %alloca_17[2] : memref<4xf32>
        affine.store %62, %alloca_17[3] : memref<4xf32>
      }
    }
    %alloca_26 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_26[] : memref<i32>
    %0 = affine.load %alloca_17[0] : memref<4xf32>
    %1 = arith.mulf %0, %cst : f32
    %2 = arith.fptosi %1 : f32 to i32
    %3 = affine.load %alloca_17[1] : memref<4xf32>
    %4 = arith.mulf %3, %cst_0 : f32
    %5 = arith.fptosi %4 : f32 to i32
    %6 = arith.addi %2, %5 : i32
    %7 = affine.load %alloca_17[2] : memref<4xf32>
    %8 = arith.mulf %7, %cst_1 : f32
    %9 = arith.fptosi %8 : f32 to i32
    %10 = arith.addi %6, %9 : i32
    %11 = affine.load %alloca_17[3] : memref<4xf32>
    %12 = arith.mulf %11, %cst_2 : f32
    %13 = arith.fptosi %12 : f32 to i32
    %14 = arith.addi %10, %13 : i32
    affine.store %14, %alloca_26[] : memref<i32>
    affine.for %arg0 = 0 to 4 {
      %16 = affine.load %alloca[%arg0] : memref<4xi32>
      %17 = affine.load %alloca_26[] : memref<i32>
      %18 = arith.addi %17, %16 : i32
      affine.store %18, %alloca_26[] : memref<i32>
    }
    %15 = affine.load %alloca_26[] : memref<i32>
    call @pcl_validate_outputs(%alloca_18, %alloca_17, %alloca) : (memref<32x3xf32>, memref<4xf32>, memref<4xi32>) -> ()
    return %15 : i32
  }
  func.func private @pcl_validate_outputs(memref<32x3xf32>, memref<4xf32>, memref<4xi32>)
}

