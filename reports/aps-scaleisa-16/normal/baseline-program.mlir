module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.700000e+01 : f32
    %cst_0 = arith.constant 2.300000e+01 : f32
    %cst_1 = arith.constant 3.100000e+01 : f32
    %cst_2 = arith.constant 1.100000e+01 : f32
    %cst_3 = arith.constant 6.400000e+01 : f32
    %cst_4 = arith.constant 9.000000e+00 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 1.000000e-03 : f32
    %cst_8 = arith.constant 2.000000e+00 : f32
    %c11_i32 = arith.constant 11 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c9_i32 = arith.constant 9 : i32
    %cst_9 = arith.constant 0.00999999977 : f32
    %cst_10 = arith.constant 7.000000e-02 : f32
    %cst_11 = arith.constant 1.500000e-01 : f32
    %cst_12 = arith.constant 4.000000e-02 : f32
    %cst_13 = arith.constant 5.000000e-02 : f32
    %alloca = memref.alloca() : memref<12x12xf32>
    %alloca_14 = memref.alloca() : memref<12x12x3xf32>
    %alloca_15 = memref.alloca() : memref<12x12x3xf32>
    %alloca_16 = memref.alloca() : memref<12x12x3xf32>
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.sitofp %2 : i32 to f32
        %4 = arith.mulf %3, %cst_13 : f32
        %5 = arith.sitofp %1 : i32 to f32
        %6 = arith.mulf %5, %cst_12 : f32
        affine.store %4, %alloca_16[%arg0, %arg1, 0] : memref<12x12x3xf32>
        affine.store %6, %alloca_16[%arg0, %arg1, 1] : memref<12x12x3xf32>
        %7 = arith.mulf %4, %cst_11 : f32
        %8 = arith.mulf %6, %cst_10 : f32
        %9 = arith.addf %7, %8 : f32
        %10 = arith.muli %2, %1 : i32
        %11 = arith.remsi %10, %c9_i32 : i32
        %12 = arith.sitofp %11 : i32 to f32
        %13 = arith.mulf %12, %cst_9 : f32
        %14 = arith.addf %9, %13 : f32
        affine.store %14, %alloca_16[%arg0, %arg1, 2] : memref<12x12x3xf32>
      }
    }
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.addi %2, %c-1_i32 : i32
        %4 = arith.cmpi sge, %3, %c0_i32 : i32
        %5 = arith.cmpi sgt, %3, %c11_i32 : i32
        %6 = arith.andi %4, %5 : i1
        %7 = arith.xori %6, %true : i1
        %8 = arith.andi %7, %4 : i1
        %9 = arith.select %6, %c11_i32, %c0_i32 : i32
        %10 = arith.select %8, %3, %9 : i32
        %11 = arith.addi %2, %c1_i32 : i32
        %12 = arith.cmpi sge, %11, %c0_i32 : i32
        %13 = arith.cmpi sgt, %11, %c11_i32 : i32
        %14 = arith.andi %12, %13 : i1
        %15 = arith.xori %14, %true : i1
        %16 = arith.andi %15, %12 : i1
        %17 = arith.select %14, %c11_i32, %c0_i32 : i32
        %18 = arith.select %16, %11, %17 : i32
        %19 = arith.addi %1, %c-1_i32 : i32
        %20 = arith.cmpi sge, %19, %c0_i32 : i32
        %21 = arith.cmpi sgt, %19, %c11_i32 : i32
        %22 = arith.andi %20, %21 : i1
        %23 = arith.xori %22, %true : i1
        %24 = arith.andi %23, %20 : i1
        %25 = arith.select %22, %c11_i32, %c0_i32 : i32
        %26 = arith.select %24, %19, %25 : i32
        %27 = arith.addi %1, %c1_i32 : i32
        %28 = arith.cmpi sge, %27, %c0_i32 : i32
        %29 = arith.cmpi sgt, %27, %c11_i32 : i32
        %30 = arith.andi %28, %29 : i1
        %31 = arith.xori %30, %true : i1
        %32 = arith.andi %31, %28 : i1
        %33 = arith.select %30, %c11_i32, %c0_i32 : i32
        %34 = arith.select %32, %27, %33 : i32
        %35 = arith.index_cast %18 : i32 to index
        %36 = memref.load %alloca_16[%arg0, %35, %c0] : memref<12x12x3xf32>
        %37 = arith.index_cast %10 : i32 to index
        %38 = memref.load %alloca_16[%arg0, %37, %c0] : memref<12x12x3xf32>
        %39 = arith.subf %36, %38 : f32
        %40 = memref.load %alloca_16[%arg0, %35, %c1] : memref<12x12x3xf32>
        %41 = memref.load %alloca_16[%arg0, %37, %c1] : memref<12x12x3xf32>
        %42 = arith.subf %40, %41 : f32
        %43 = memref.load %alloca_16[%arg0, %35, %c2] : memref<12x12x3xf32>
        %44 = memref.load %alloca_16[%arg0, %37, %c2] : memref<12x12x3xf32>
        %45 = arith.subf %43, %44 : f32
        %46 = arith.index_cast %34 : i32 to index
        %47 = memref.load %alloca_16[%46, %arg1, %c0] : memref<12x12x3xf32>
        %48 = arith.index_cast %26 : i32 to index
        %49 = memref.load %alloca_16[%48, %arg1, %c0] : memref<12x12x3xf32>
        %50 = arith.subf %47, %49 : f32
        %51 = memref.load %alloca_16[%46, %arg1, %c1] : memref<12x12x3xf32>
        %52 = memref.load %alloca_16[%48, %arg1, %c1] : memref<12x12x3xf32>
        %53 = arith.subf %51, %52 : f32
        %54 = memref.load %alloca_16[%46, %arg1, %c2] : memref<12x12x3xf32>
        %55 = memref.load %alloca_16[%48, %arg1, %c2] : memref<12x12x3xf32>
        %56 = arith.subf %54, %55 : f32
        %57 = arith.mulf %42, %56 : f32
        %58 = arith.mulf %45, %53 : f32
        %59 = arith.subf %57, %58 : f32
        %60 = arith.mulf %45, %50 : f32
        %61 = arith.mulf %39, %56 : f32
        %62 = arith.subf %60, %61 : f32
        %63 = arith.mulf %39, %53 : f32
        %64 = arith.mulf %42, %50 : f32
        %65 = arith.subf %63, %64 : f32
        %66 = arith.cmpf olt, %59, %cst_5 : f32
        %67 = scf.if %66 -> (f32) {
          %94 = arith.negf %59 : f32
          scf.yield %94 : f32
        } else {
          scf.yield %59 : f32
        }
        %68 = arith.cmpf olt, %62, %cst_5 : f32
        %69 = scf.if %68 -> (f32) {
          %94 = arith.negf %62 : f32
          scf.yield %94 : f32
        } else {
          scf.yield %62 : f32
        }
        %70 = arith.addf %67, %69 : f32
        %71 = arith.cmpf olt, %65, %cst_5 : f32
        %72 = scf.if %71 -> (f32) {
          %94 = arith.negf %65 : f32
          scf.yield %94 : f32
        } else {
          scf.yield %65 : f32
        }
        %73 = arith.addf %70, %72 : f32
        %74 = arith.addf %73, %cst_7 : f32
        %75 = arith.divf %cst_6, %74 : f32
        %76 = arith.mulf %59, %75 : f32
        affine.store %76, %alloca_15[%arg0, %arg1, 0] : memref<12x12x3xf32>
        %77 = arith.mulf %62, %75 : f32
        affine.store %77, %alloca_15[%arg0, %arg1, 1] : memref<12x12x3xf32>
        %78 = arith.mulf %65, %75 : f32
        affine.store %78, %alloca_15[%arg0, %arg1, 2] : memref<12x12x3xf32>
        %79 = memref.load %alloca_16[%arg0, %35, %c2] : memref<12x12x3xf32>
        %80 = affine.load %alloca_16[%arg0, %arg1, 2] : memref<12x12x3xf32>
        %81 = arith.mulf %80, %cst_8 : f32
        %82 = arith.subf %79, %81 : f32
        %83 = memref.load %alloca_16[%arg0, %37, %c2] : memref<12x12x3xf32>
        %84 = arith.addf %82, %83 : f32
        %85 = arith.cmpf olt, %84, %cst_5 : f32
        %86 = scf.if %85 -> (f32) {
          %94 = arith.negf %84 : f32
          scf.yield %94 : f32
        } else {
          scf.yield %84 : f32
        }
        %87 = memref.load %alloca_16[%46, %arg1, %c2] : memref<12x12x3xf32>
        %88 = arith.subf %87, %81 : f32
        %89 = memref.load %alloca_16[%48, %arg1, %c2] : memref<12x12x3xf32>
        %90 = arith.addf %88, %89 : f32
        %91 = arith.cmpf olt, %90, %cst_5 : f32
        %92 = scf.if %91 -> (f32) {
          %94 = arith.negf %90 : f32
          scf.yield %94 : f32
        } else {
          scf.yield %90 : f32
        }
        %93 = arith.addf %86, %92 : f32
        affine.store %93, %alloca[%arg0, %arg1] : memref<12x12xf32>
      }
    }
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %alloca_18 = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca_18[] : memref<f32>
        %alloca_19 = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca_19[] : memref<f32>
        %alloca_20 = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca_20[] : memref<f32>
        affine.for %arg2 = 0 to 3 {
          %9 = arith.index_cast %arg2 : index to i32
          %10 = arith.addi %9, %c-1_i32 : i32
          affine.for %arg3 = 0 to 3 {
            %11 = arith.index_cast %arg3 : index to i32
            %12 = arith.addi %11, %c-1_i32 : i32
            %13 = arith.addi %1, %10 : i32
            %14 = arith.cmpi sge, %13, %c0_i32 : i32
            %15 = arith.cmpi sgt, %13, %c11_i32 : i32
            %16 = arith.andi %14, %15 : i1
            %17 = arith.xori %16, %true : i1
            %18 = arith.andi %17, %14 : i1
            %19 = arith.select %16, %c11_i32, %c0_i32 : i32
            %20 = arith.select %18, %13, %19 : i32
            %21 = arith.addi %2, %12 : i32
            %22 = arith.cmpi sge, %21, %c0_i32 : i32
            %23 = arith.cmpi sgt, %21, %c11_i32 : i32
            %24 = arith.andi %22, %23 : i1
            %25 = arith.xori %24, %true : i1
            %26 = arith.andi %25, %22 : i1
            %27 = arith.select %24, %c11_i32, %c0_i32 : i32
            %28 = arith.select %26, %21, %27 : i32
            %29 = arith.index_cast %20 : i32 to index
            %30 = arith.index_cast %28 : i32 to index
            %31 = memref.load %alloca_15[%29, %30, %c0] : memref<12x12x3xf32>
            %32 = affine.load %alloca_18[] : memref<f32>
            %33 = arith.addf %32, %31 : f32
            affine.store %33, %alloca_18[] : memref<f32>
            %34 = memref.load %alloca_15[%29, %30, %c1] : memref<12x12x3xf32>
            %35 = affine.load %alloca_19[] : memref<f32>
            %36 = arith.addf %35, %34 : f32
            affine.store %36, %alloca_19[] : memref<f32>
            %37 = memref.load %alloca_15[%29, %30, %c2] : memref<12x12x3xf32>
            %38 = affine.load %alloca_20[] : memref<f32>
            %39 = arith.addf %38, %37 : f32
            affine.store %39, %alloca_20[] : memref<f32>
          }
        }
        %3 = affine.load %alloca_18[] : memref<f32>
        %4 = arith.divf %3, %cst_4 : f32
        affine.store %4, %alloca_14[%arg0, %arg1, 0] : memref<12x12x3xf32>
        %5 = affine.load %alloca_19[] : memref<f32>
        %6 = arith.divf %5, %cst_4 : f32
        affine.store %6, %alloca_14[%arg0, %arg1, 1] : memref<12x12x3xf32>
        %7 = affine.load %alloca_20[] : memref<f32>
        %8 = arith.divf %7, %cst_4 : f32
        affine.store %8, %alloca_14[%arg0, %arg1, 2] : memref<12x12x3xf32>
      }
    }
    %alloca_17 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_17[] : memref<i32>
    affine.store %c0_i32, %alloca_17[] : memref<i32>
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        %1 = affine.load %alloca_14[%arg0 + 1, %arg1 + 1, 0] : memref<12x12x3xf32>
        %2 = arith.mulf %1, %cst : f32
        %3 = affine.load %alloca_14[%arg0 + 1, %arg1 + 1, 1] : memref<12x12x3xf32>
        %4 = arith.mulf %3, %cst_0 : f32
        %5 = arith.addf %2, %4 : f32
        %6 = affine.load %alloca_14[%arg0 + 1, %arg1 + 1, 2] : memref<12x12x3xf32>
        %7 = arith.mulf %6, %cst_1 : f32
        %8 = arith.addf %5, %7 : f32
        %9 = affine.load %alloca[%arg0 + 1, %arg1 + 1] : memref<12x12xf32>
        %10 = arith.mulf %9, %cst_2 : f32
        %11 = arith.addf %8, %10 : f32
        %12 = arith.mulf %11, %cst_3 : f32
        %13 = arith.fptosi %12 : f32 to i32
        %14 = affine.load %alloca_17[] : memref<i32>
        %15 = arith.addi %14, %13 : i32
        affine.store %15, %alloca_17[] : memref<i32>
      }
    }
    %0 = affine.load %alloca_17[] : memref<i32>
    call @pcl_validate_outputs(%alloca_16, %alloca_15, %alloca_14, %alloca) : (memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12xf32>) -> ()
    return %0 : i32
  }
  func.func private @pcl_validate_outputs(memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12xf32>)
}

