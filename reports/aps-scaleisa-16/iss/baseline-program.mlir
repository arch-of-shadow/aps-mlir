module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.200000e+01 : f32
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant 5.000000e-01 : f32
    %c15_i32 = arith.constant 15 : i32
    %c-2_i32 = arith.constant -2 : i32
    %c19_i32 = arith.constant 19 : i32
    %c9_i32 = arith.constant 9 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_2 = arith.constant 2.000000e-02 : f32
    %cst_3 = arith.constant 3.000000e-02 : f32
    %alloca = memref.alloca() : memref<16x16xi32>
    %alloca_4 = memref.alloca() : memref<16x16xf32>
    %alloca_5 = memref.alloca() : memref<16x16x3xf32>
    affine.for %arg0 = 0 to 16 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 16 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.sitofp %2 : i32 to f32
        %4 = arith.mulf %3, %cst_3 : f32
        affine.store %4, %alloca_5[%arg0, %arg1, 0] : memref<16x16x3xf32>
        %5 = arith.sitofp %1 : i32 to f32
        %6 = arith.mulf %5, %cst_3 : f32
        affine.store %6, %alloca_5[%arg0, %arg1, 1] : memref<16x16x3xf32>
        %7 = arith.muli %2, %c5_i32 : i32
        %8 = arith.muli %1, %c9_i32 : i32
        %9 = arith.addi %7, %8 : i32
        %10 = arith.xori %2, %1 : i32
        %11 = arith.addi %9, %10 : i32
        %12 = arith.remsi %11, %c19_i32 : i32
        %13 = arith.sitofp %12 : i32 to f32
        %14 = arith.mulf %13, %cst_2 : f32
        affine.store %14, %alloca_5[%arg0, %arg1, 2] : memref<16x16x3xf32>
      }
    }
    affine.for %arg0 = 0 to 16 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 16 {
        %2 = arith.index_cast %arg1 : index to i32
        %alloca_7 = memref.alloca() : memref<f32>
        affine.store %cst_0, %alloca_7[] : memref<f32>
        %alloca_8 = memref.alloca() : memref<f32>
        affine.store %cst_0, %alloca_8[] : memref<f32>
        %alloca_9 = memref.alloca() : memref<f32>
        affine.store %cst_0, %alloca_9[] : memref<f32>
        %alloca_10 = memref.alloca() : memref<f32>
        affine.store %cst_0, %alloca_10[] : memref<f32>
        %alloca_11 = memref.alloca() : memref<f32>
        affine.store %cst_0, %alloca_11[] : memref<f32>
        %alloca_12 = memref.alloca() : memref<f32>
        affine.store %cst_0, %alloca_12[] : memref<f32>
        %3 = affine.load %alloca_5[%arg0, %arg1, 0] : memref<16x16x3xf32>
        %4 = affine.load %alloca_5[%arg0, %arg1, 1] : memref<16x16x3xf32>
        %5 = affine.load %alloca_5[%arg0, %arg1, 2] : memref<16x16x3xf32>
        affine.for %arg2 = 0 to 5 {
          %24 = arith.index_cast %arg2 : index to i32
          %25 = arith.addi %24, %c-2_i32 : i32
          affine.for %arg3 = 0 to 5 {
            %26 = arith.index_cast %arg3 : index to i32
            %27 = arith.addi %26, %c-2_i32 : i32
            %28 = arith.addi %1, %25 : i32
            %29 = arith.cmpi sge, %28, %c0_i32 : i32
            %30 = arith.cmpi sgt, %28, %c15_i32 : i32
            %31 = arith.andi %29, %30 : i1
            %32 = arith.xori %31, %true : i1
            %33 = arith.andi %32, %29 : i1
            %34 = arith.select %31, %c15_i32, %c0_i32 : i32
            %35 = arith.select %33, %28, %34 : i32
            %36 = arith.addi %2, %27 : i32
            %37 = arith.cmpi sge, %36, %c0_i32 : i32
            %38 = arith.cmpi sgt, %36, %c15_i32 : i32
            %39 = arith.andi %37, %38 : i1
            %40 = arith.xori %39, %true : i1
            %41 = arith.andi %40, %37 : i1
            %42 = arith.select %39, %c15_i32, %c0_i32 : i32
            %43 = arith.select %41, %36, %42 : i32
            %44 = arith.index_cast %35 : i32 to index
            %45 = arith.index_cast %43 : i32 to index
            %46 = memref.load %alloca_5[%44, %45, %c0] : memref<16x16x3xf32>
            %47 = arith.subf %46, %3 : f32
            %48 = memref.load %alloca_5[%44, %45, %c1] : memref<16x16x3xf32>
            %49 = arith.subf %48, %4 : f32
            %50 = memref.load %alloca_5[%44, %45, %c2] : memref<16x16x3xf32>
            %51 = arith.subf %50, %5 : f32
            %52 = arith.mulf %47, %47 : f32
            %53 = affine.load %alloca_7[] : memref<f32>
            %54 = arith.addf %53, %52 : f32
            affine.store %54, %alloca_7[] : memref<f32>
            %55 = arith.mulf %49, %49 : f32
            %56 = affine.load %alloca_8[] : memref<f32>
            %57 = arith.addf %56, %55 : f32
            affine.store %57, %alloca_8[] : memref<f32>
            %58 = arith.mulf %51, %51 : f32
            %59 = affine.load %alloca_9[] : memref<f32>
            %60 = arith.addf %59, %58 : f32
            affine.store %60, %alloca_9[] : memref<f32>
            %61 = arith.mulf %47, %49 : f32
            %62 = affine.load %alloca_10[] : memref<f32>
            %63 = arith.addf %62, %61 : f32
            affine.store %63, %alloca_10[] : memref<f32>
            %64 = arith.mulf %47, %51 : f32
            %65 = affine.load %alloca_11[] : memref<f32>
            %66 = arith.addf %65, %64 : f32
            affine.store %66, %alloca_11[] : memref<f32>
            %67 = arith.mulf %49, %51 : f32
            %68 = affine.load %alloca_12[] : memref<f32>
            %69 = arith.addf %68, %67 : f32
            affine.store %69, %alloca_12[] : memref<f32>
          }
        }
        %6 = affine.load %alloca_7[] : memref<f32>
        %7 = affine.load %alloca_8[] : memref<f32>
        %8 = arith.addf %6, %7 : f32
        %9 = affine.load %alloca_9[] : memref<f32>
        %10 = arith.addf %8, %9 : f32
        %11 = affine.load %alloca_10[] : memref<f32>
        %12 = arith.cmpf olt, %11, %cst_0 : f32
        %13 = scf.if %12 -> (f32) {
          %24 = arith.negf %11 : f32
          scf.yield %24 : f32
        } else {
          scf.yield %11 : f32
        }
        %14 = affine.load %alloca_11[] : memref<f32>
        %15 = arith.cmpf olt, %14, %cst_0 : f32
        %16 = scf.if %15 -> (f32) {
          %24 = arith.negf %14 : f32
          scf.yield %24 : f32
        } else {
          scf.yield %14 : f32
        }
        %17 = arith.addf %13, %16 : f32
        %18 = affine.load %alloca_12[] : memref<f32>
        %19 = arith.cmpf olt, %18, %cst_0 : f32
        %20 = scf.if %19 -> (f32) {
          %24 = arith.negf %18 : f32
          scf.yield %24 : f32
        } else {
          scf.yield %18 : f32
        }
        %21 = arith.addf %17, %20 : f32
        %22 = arith.mulf %21, %cst_1 : f32
        %23 = arith.subf %10, %22 : f32
        affine.store %23, %alloca_4[%arg0, %arg1] : memref<16x16xf32>
      }
    }
    affine.for %arg0 = 0 to 16 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 16 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = affine.load %alloca_4[%arg0, %arg1] : memref<16x16xf32>
        %4 = arith.cmpf ogt, %3, %cst_2 : f32
        %5 = arith.extui %4 : i1 to i32
        %alloca_7 = memref.alloca() : memref<i32>
        affine.store %5, %alloca_7[] : memref<i32>
        affine.for %arg2 = 0 to 3 {
          %7 = arith.index_cast %arg2 : index to i32
          %8 = arith.addi %7, %c-1_i32 : i32
          affine.for %arg3 = 0 to 3 {
            %9 = arith.index_cast %arg3 : index to i32
            %10 = arith.addi %9, %c-1_i32 : i32
            %11 = arith.addi %1, %8 : i32
            %12 = arith.cmpi sge, %11, %c0_i32 : i32
            %13 = arith.cmpi sgt, %11, %c15_i32 : i32
            %14 = arith.andi %12, %13 : i1
            %15 = arith.xori %14, %true : i1
            %16 = arith.andi %15, %12 : i1
            %17 = arith.select %14, %c15_i32, %c0_i32 : i32
            %18 = arith.select %16, %11, %17 : i32
            %19 = arith.addi %2, %10 : i32
            %20 = arith.cmpi sge, %19, %c0_i32 : i32
            %21 = arith.cmpi sgt, %19, %c15_i32 : i32
            %22 = arith.andi %20, %21 : i1
            %23 = arith.xori %22, %true : i1
            %24 = arith.andi %23, %20 : i1
            %25 = arith.select %22, %c15_i32, %c0_i32 : i32
            %26 = arith.select %24, %19, %25 : i32
            %27 = arith.cmpi ne, %8, %c0_i32 : i32
            %28 = scf.if %27 -> (i1) {
              scf.yield %true : i1
            } else {
              %29 = arith.cmpi ne, %10, %c0_i32 : i32
              scf.yield %29 : i1
            }
            scf.if %28 {
              %29 = arith.index_cast %18 : i32 to index
              %30 = arith.index_cast %26 : i32 to index
              %31 = memref.load %alloca_4[%29, %30] : memref<16x16xf32>
              %32 = arith.cmpf ogt, %31, %3 : f32
              scf.if %32 {
                affine.store %c0_i32, %alloca_7[] : memref<i32>
              }
            }
          }
        }
        %6 = affine.load %alloca_7[] : memref<i32>
        affine.store %6, %alloca[%arg0, %arg1] : memref<16x16xi32>
      }
    }
    %alloca_6 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_6[] : memref<i32>
    affine.store %c0_i32, %alloca_6[] : memref<i32>
    affine.for %arg0 = 0 to 16 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 16 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = affine.load %alloca[%arg0, %arg1] : memref<16x16xi32>
        %4 = arith.cmpi ne, %3, %c0_i32 : i32
        %5 = scf.if %4 -> (i32) {
          %8 = affine.load %alloca_4[%arg0, %arg1] : memref<16x16xf32>
          %9 = arith.mulf %8, %cst : f32
          %10 = arith.fptosi %9 : f32 to i32
          %11 = arith.addi %10, %2 : i32
          %12 = arith.addi %11, %1 : i32
          scf.yield %12 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %6 = affine.load %alloca_6[] : memref<i32>
        %7 = arith.addi %6, %5 : i32
        affine.store %7, %alloca_6[] : memref<i32>
      }
    }
    %0 = affine.load %alloca_6[] : memref<i32>
    call @pcl_validate_outputs(%alloca_5, %alloca_4, %alloca) : (memref<16x16x3xf32>, memref<16x16xf32>, memref<16x16xi32>) -> ()
    return %0 : i32
  }
  func.func private @pcl_validate_outputs(memref<16x16x3xf32>, memref<16x16xf32>, memref<16x16xi32>)
}

