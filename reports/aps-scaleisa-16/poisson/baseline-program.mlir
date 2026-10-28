#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst = arith.constant 6.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 8.300000e+01 : f32
    %c83_i32 = arith.constant 83 : i32
    %c13_i32 = arith.constant 13 : i32
    %cst_2 = arith.constant 8.900000e+01 : f32
    %c89_i32 = arith.constant 89 : i32
    %c5_i32 = arith.constant 5 : i32
    %c11_i32 = arith.constant 11 : i32
    %cst_3 = arith.constant 9.700000e+01 : f32
    %c97_i32 = arith.constant 97 : i32
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<216xf32>
    %alloca_5 = memref.alloca() : memref<216xf32>
    %alloca_6 = memref.alloca() : memref<216xf32>
    %alloca_7 = memref.alloca() : memref<24x6xf32>
    affine.for %arg0 = 0 to 216 {
      affine.store %cst_4, %alloca_6[%arg0] : memref<216xf32>
      affine.store %cst_4, %alloca_5[%arg0] : memref<216xf32>
      affine.store %cst_4, %alloca[%arg0] : memref<216xf32>
    }
    affine.for %arg0 = 0 to 24 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.muli %1, %c7_i32 : i32
      %3 = arith.addi %2, %c3_i32 : i32
      %4 = arith.remsi %3, %c97_i32 : i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.divf %5, %cst_3 : f32
      affine.store %6, %alloca_7[%arg0, 0] : memref<24x6xf32>
      %7 = arith.muli %1, %c11_i32 : i32
      %8 = arith.addi %7, %c5_i32 : i32
      %9 = arith.remsi %8, %c89_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.divf %10, %cst_2 : f32
      affine.store %11, %alloca_7[%arg0, 1] : memref<24x6xf32>
      %12 = arith.muli %1, %c13_i32 : i32
      %13 = arith.addi %12, %c7_i32 : i32
      %14 = arith.remsi %13, %c83_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.divf %15, %cst_1 : f32
      affine.store %16, %alloca_7[%arg0, 2] : memref<24x6xf32>
      %17 = affine.load %alloca_7[%arg0, 0] : memref<24x6xf32>
      %18 = arith.subf %cst_0, %17 : f32
      affine.store %18, %alloca_7[%arg0, 3] : memref<24x6xf32>
      %19 = affine.load %alloca_7[%arg0, 1] : memref<24x6xf32>
      %20 = arith.subf %cst_0, %19 : f32
      affine.store %20, %alloca_7[%arg0, 4] : memref<24x6xf32>
      %21 = affine.load %alloca_7[%arg0, 2] : memref<24x6xf32>
      %22 = arith.subf %cst_0, %21 : f32
      affine.store %22, %alloca_7[%arg0, 5] : memref<24x6xf32>
    }
    affine.for %arg0 = 0 to 24 {
      %1 = affine.load %alloca_7[%arg0, 0] : memref<24x6xf32>
      %2 = arith.mulf %1, %cst : f32
      %3 = arith.fptosi %2 : f32 to i32
      %4 = arith.cmpi sge, %3, %c0_i32 : i32
      %5 = arith.cmpi sgt, %3, %c5_i32 : i32
      %6 = arith.andi %4, %5 : i1
      %7 = arith.xori %6, %true : i1
      %8 = arith.andi %7, %4 : i1
      %9 = arith.select %6, %c5_i32, %c0_i32 : i32
      %10 = arith.select %8, %3, %9 : i32
      %11 = affine.load %alloca_7[%arg0, 1] : memref<24x6xf32>
      %12 = arith.mulf %11, %cst : f32
      %13 = arith.fptosi %12 : f32 to i32
      %14 = arith.cmpi sge, %13, %c0_i32 : i32
      %15 = arith.cmpi sgt, %13, %c5_i32 : i32
      %16 = arith.andi %14, %15 : i1
      %17 = arith.xori %16, %true : i1
      %18 = arith.andi %17, %14 : i1
      %19 = arith.select %16, %c5_i32, %c0_i32 : i32
      %20 = arith.select %18, %13, %19 : i32
      %21 = affine.load %alloca_7[%arg0, 2] : memref<24x6xf32>
      %22 = arith.mulf %21, %cst : f32
      %23 = arith.fptosi %22 : f32 to i32
      %24 = arith.cmpi sge, %23, %c0_i32 : i32
      %25 = arith.cmpi sgt, %23, %c5_i32 : i32
      %26 = arith.andi %24, %25 : i1
      %27 = arith.xori %26, %true : i1
      %28 = arith.andi %27, %24 : i1
      %29 = arith.select %26, %c5_i32, %c0_i32 : i32
      %30 = arith.select %28, %23, %29 : i32
      %31 = arith.muli %10, %c6_i32 : i32
      %32 = arith.addi %31, %20 : i32
      %33 = arith.muli %32, %c6_i32 : i32
      %34 = arith.addi %33, %30 : i32
      %35 = arith.index_cast %34 : i32 to index
      %36 = affine.load %alloca_7[%arg0, 3] : memref<24x6xf32>
      %37 = affine.load %alloca_7[%arg0, 4] : memref<24x6xf32>
      %38 = arith.addf %36, %37 : f32
      %39 = affine.load %alloca_7[%arg0, 5] : memref<24x6xf32>
      %40 = arith.addf %38, %39 : f32
      %41 = memref.load %alloca_6[%35] : memref<216xf32>
      %42 = arith.addf %41, %40 : f32
      memref.store %42, %alloca_6[%35] : memref<216xf32>
    }
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 4 {
        %1 = affine.apply #map(%arg1)
        %2 = arith.index_cast %1 : index to i32
        affine.for %arg2 = 0 to 4 {
          %3 = affine.apply #map(%arg2)
          %4 = arith.index_cast %3 : index to i32
          affine.for %arg3 = 0 to 4 {
            %5 = affine.apply #map(%arg3)
            %6 = arith.index_cast %5 : index to i32
            %7 = arith.muli %2, %c6_i32 : i32
            %8 = arith.addi %7, %4 : i32
            %9 = arith.muli %8, %c6_i32 : i32
            %10 = arith.addi %9, %6 : i32
            %11 = arith.addi %2, %c-1_i32 : i32
            %12 = arith.muli %11, %c6_i32 : i32
            %13 = arith.addi %12, %4 : i32
            %14 = arith.muli %13, %c6_i32 : i32
            %15 = arith.addi %14, %6 : i32
            %16 = arith.addi %2, %c1_i32 : i32
            %17 = arith.muli %16, %c6_i32 : i32
            %18 = arith.addi %17, %4 : i32
            %19 = arith.muli %18, %c6_i32 : i32
            %20 = arith.addi %19, %6 : i32
            %21 = arith.addi %4, %c-1_i32 : i32
            %22 = arith.addi %7, %21 : i32
            %23 = arith.muli %22, %c6_i32 : i32
            %24 = arith.addi %23, %6 : i32
            %25 = arith.addi %4, %c1_i32 : i32
            %26 = arith.addi %7, %25 : i32
            %27 = arith.muli %26, %c6_i32 : i32
            %28 = arith.addi %27, %6 : i32
            %29 = arith.addi %6, %c-1_i32 : i32
            %30 = arith.addi %9, %29 : i32
            %31 = arith.addi %6, %c1_i32 : i32
            %32 = arith.addi %9, %31 : i32
            %33 = arith.index_cast %15 : i32 to index
            %34 = memref.load %alloca_5[%33] : memref<216xf32>
            %35 = arith.index_cast %20 : i32 to index
            %36 = memref.load %alloca_5[%35] : memref<216xf32>
            %37 = arith.addf %34, %36 : f32
            %38 = arith.index_cast %24 : i32 to index
            %39 = memref.load %alloca_5[%38] : memref<216xf32>
            %40 = arith.addf %37, %39 : f32
            %41 = arith.index_cast %28 : i32 to index
            %42 = memref.load %alloca_5[%41] : memref<216xf32>
            %43 = arith.addf %40, %42 : f32
            %44 = arith.index_cast %30 : i32 to index
            %45 = memref.load %alloca_5[%44] : memref<216xf32>
            %46 = arith.addf %43, %45 : f32
            %47 = arith.index_cast %32 : i32 to index
            %48 = memref.load %alloca_5[%47] : memref<216xf32>
            %49 = arith.addf %46, %48 : f32
            %50 = arith.index_cast %10 : i32 to index
            %51 = memref.load %alloca_6[%50] : memref<216xf32>
            %52 = arith.addf %51, %49 : f32
            %53 = arith.divf %52, %cst : f32
            memref.store %53, %alloca[%50] : memref<216xf32>
          }
        }
      }
      affine.for %arg1 = 0 to 4 {
        %1 = affine.apply #map(%arg1)
        %2 = arith.index_cast %1 : index to i32
        affine.for %arg2 = 0 to 4 {
          %3 = affine.apply #map(%arg2)
          %4 = arith.index_cast %3 : index to i32
          affine.for %arg3 = 0 to 4 {
            %5 = affine.apply #map(%arg3)
            %6 = arith.index_cast %5 : index to i32
            %7 = arith.muli %2, %c6_i32 : i32
            %8 = arith.addi %7, %4 : i32
            %9 = arith.muli %8, %c6_i32 : i32
            %10 = arith.addi %9, %6 : i32
            %11 = arith.index_cast %10 : i32 to index
            %12 = memref.load %alloca[%11] : memref<216xf32>
            memref.store %12, %alloca_5[%11] : memref<216xf32>
          }
        }
      }
    }
    %alloca_8 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_8[] : memref<i32>
    affine.store %c0_i32, %alloca_8[] : memref<i32>
    affine.for %arg0 = 0 to 4 {
      %1 = affine.apply #map(%arg0)
      %2 = arith.index_cast %1 : index to i32
      affine.for %arg1 = 0 to 4 {
        %3 = affine.apply #map(%arg1)
        %4 = arith.index_cast %3 : index to i32
        affine.for %arg2 = 0 to 4 {
          %5 = affine.apply #map(%arg2)
          %6 = arith.index_cast %5 : index to i32
          %7 = arith.muli %2, %c6_i32 : i32
          %8 = arith.addi %7, %4 : i32
          %9 = arith.muli %8, %c6_i32 : i32
          %10 = arith.addi %9, %6 : i32
          %11 = arith.addi %2, %c1_i32 : i32
          %12 = arith.muli %11, %c6_i32 : i32
          %13 = arith.addi %12, %4 : i32
          %14 = arith.muli %13, %c6_i32 : i32
          %15 = arith.addi %14, %6 : i32
          %16 = arith.addi %4, %c1_i32 : i32
          %17 = arith.addi %7, %16 : i32
          %18 = arith.muli %17, %c6_i32 : i32
          %19 = arith.addi %18, %6 : i32
          %20 = arith.addi %6, %c1_i32 : i32
          %21 = arith.addi %9, %20 : i32
          %22 = arith.index_cast %10 : i32 to index
          %23 = memref.load %alloca_5[%22] : memref<216xf32>
          %24 = arith.index_cast %15 : i32 to index
          %25 = memref.load %alloca_5[%24] : memref<216xf32>
          %26 = arith.mulf %23, %25 : f32
          %27 = arith.cmpf olt, %26, %cst_4 : f32
          %28 = arith.extui %27 : i1 to i32
          %29 = affine.load %alloca_8[] : memref<i32>
          %30 = arith.addi %29, %28 : i32
          affine.store %30, %alloca_8[] : memref<i32>
          %31 = arith.index_cast %19 : i32 to index
          %32 = memref.load %alloca_5[%31] : memref<216xf32>
          %33 = arith.mulf %23, %32 : f32
          %34 = arith.cmpf olt, %33, %cst_4 : f32
          %35 = arith.extui %34 : i1 to i32
          %36 = arith.addi %30, %35 : i32
          affine.store %36, %alloca_8[] : memref<i32>
          %37 = arith.index_cast %21 : i32 to index
          %38 = memref.load %alloca_5[%37] : memref<216xf32>
          %39 = arith.mulf %23, %38 : f32
          %40 = arith.cmpf olt, %39, %cst_4 : f32
          %41 = arith.extui %40 : i1 to i32
          %42 = arith.addi %36, %41 : i32
          affine.store %42, %alloca_8[] : memref<i32>
        }
      }
    }
    %0 = affine.load %alloca_8[] : memref<i32>
    call @pcl_validate_outputs(%alloca_7, %alloca_6, %alloca_5, %alloca) : (memref<24x6xf32>, memref<216xf32>, memref<216xf32>, memref<216xf32>) -> ()
    return %0 : i32
  }
  func.func private @pcl_validate_outputs(memref<24x6xf32>, memref<216xf32>, memref<216xf32>, memref<216xf32>)
}

