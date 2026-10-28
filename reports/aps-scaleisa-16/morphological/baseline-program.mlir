module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.280000e+02 : f32
    %c3_i32 = arith.constant 3 : i32
    %true = arith.constant true
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c11_i32 = arith.constant 11 : i32
    %c-2_i32 = arith.constant -2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c17_i32 = arith.constant 17 : i32
    %cst_1 = arith.constant 2.000000e-02 : f32
    %c7_i32 = arith.constant 7 : i32
    %c13_i32 = arith.constant 13 : i32
    %c51_i32 = arith.constant 51 : i32
    %c-53_i32 = arith.constant -53 : i32
    %c-31_i32 = arith.constant -31 : i32
    %alloca = memref.alloca() : memref<12x12xf32>
    %alloca_2 = memref.alloca() : memref<12x12xf32>
    %alloca_3 = memref.alloca() : memref<12x12xf32>
    %alloca_4 = memref.alloca() : memref<12x12xf32>
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.addi %2, %c-31_i32 : i32
        %4 = arith.muli %3, %3 : i32
        %5 = arith.addi %1, %c-53_i32 : i32
        %6 = arith.muli %5, %5 : i32
        %7 = arith.addi %4, %6 : i32
        %8 = arith.remsi %7, %c17_i32 : i32
        %9 = arith.muli %2, %c7_i32 : i32
        %10 = arith.muli %1, %c13_i32 : i32
        %11 = arith.addi %9, %10 : i32
        %12 = arith.addi %11, %8 : i32
        %13 = arith.remsi %12, %c51_i32 : i32
        %14 = arith.sitofp %13 : i32 to f32
        %15 = arith.mulf %14, %cst_1 : f32
        affine.store %15, %alloca_4[%arg0, %arg1] : memref<12x12xf32>
      }
    }
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = affine.load %alloca_4[%arg0, %arg1] : memref<12x12xf32>
        %alloca_6 = memref.alloca() : memref<f32>
        affine.store %3, %alloca_6[] : memref<f32>
        affine.for %arg2 = 0 to 5 {
          %5 = arith.index_cast %arg2 : index to i32
          %6 = arith.addi %5, %c-2_i32 : i32
          affine.for %arg3 = 0 to 5 {
            %7 = arith.index_cast %arg3 : index to i32
            %8 = arith.addi %7, %c-2_i32 : i32
            %9 = arith.addi %1, %6 : i32
            %10 = arith.cmpi sge, %9, %c0_i32 : i32
            %11 = arith.cmpi sgt, %9, %c11_i32 : i32
            %12 = arith.andi %10, %11 : i1
            %13 = arith.xori %12, %true : i1
            %14 = arith.andi %13, %10 : i1
            %15 = arith.select %12, %c11_i32, %c0_i32 : i32
            %16 = arith.select %14, %9, %15 : i32
            %17 = arith.addi %2, %8 : i32
            %18 = arith.cmpi sge, %17, %c0_i32 : i32
            %19 = arith.cmpi sgt, %17, %c11_i32 : i32
            %20 = arith.andi %18, %19 : i1
            %21 = arith.xori %20, %true : i1
            %22 = arith.andi %21, %18 : i1
            %23 = arith.select %20, %c11_i32, %c0_i32 : i32
            %24 = arith.select %22, %17, %23 : i32
            %25 = arith.index_cast %16 : i32 to index
            %26 = arith.index_cast %24 : i32 to index
            %27 = memref.load %alloca_4[%25, %26] : memref<12x12xf32>
            %28 = affine.load %alloca_6[] : memref<f32>
            %29 = arith.cmpf olt, %27, %28 : f32
            scf.if %29 {
              affine.store %27, %alloca_6[] : memref<f32>
            }
          }
        }
        %4 = affine.load %alloca_6[] : memref<f32>
        affine.store %4, %alloca_3[%arg0, %arg1] : memref<12x12xf32>
      }
    }
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = affine.load %alloca_3[%arg0, %arg1] : memref<12x12xf32>
        %alloca_6 = memref.alloca() : memref<f32>
        affine.store %3, %alloca_6[] : memref<f32>
        affine.for %arg2 = 0 to 5 {
          %5 = arith.index_cast %arg2 : index to i32
          %6 = arith.addi %5, %c-2_i32 : i32
          affine.for %arg3 = 0 to 5 {
            %7 = arith.index_cast %arg3 : index to i32
            %8 = arith.addi %7, %c-2_i32 : i32
            %9 = arith.addi %1, %6 : i32
            %10 = arith.cmpi sge, %9, %c0_i32 : i32
            %11 = arith.cmpi sgt, %9, %c11_i32 : i32
            %12 = arith.andi %10, %11 : i1
            %13 = arith.xori %12, %true : i1
            %14 = arith.andi %13, %10 : i1
            %15 = arith.select %12, %c11_i32, %c0_i32 : i32
            %16 = arith.select %14, %9, %15 : i32
            %17 = arith.addi %2, %8 : i32
            %18 = arith.cmpi sge, %17, %c0_i32 : i32
            %19 = arith.cmpi sgt, %17, %c11_i32 : i32
            %20 = arith.andi %18, %19 : i1
            %21 = arith.xori %20, %true : i1
            %22 = arith.andi %21, %18 : i1
            %23 = arith.select %20, %c11_i32, %c0_i32 : i32
            %24 = arith.select %22, %17, %23 : i32
            %25 = arith.index_cast %16 : i32 to index
            %26 = arith.index_cast %24 : i32 to index
            %27 = memref.load %alloca_3[%25, %26] : memref<12x12xf32>
            %28 = affine.load %alloca_6[] : memref<f32>
            %29 = arith.cmpf ogt, %27, %28 : f32
            scf.if %29 {
              affine.store %27, %alloca_6[] : memref<f32>
            }
          }
        }
        %4 = affine.load %alloca_6[] : memref<f32>
        affine.store %4, %alloca_2[%arg0, %arg1] : memref<12x12xf32>
      }
    }
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = affine.load %alloca_2[%arg0, %arg1] : memref<12x12xf32>
        %alloca_6 = memref.alloca() : memref<f32>
        affine.store %3, %alloca_6[] : memref<f32>
        affine.for %arg2 = 0 to 5 {
          %5 = arith.index_cast %arg2 : index to i32
          %6 = arith.addi %5, %c-2_i32 : i32
          affine.for %arg3 = 0 to 5 {
            %7 = arith.index_cast %arg3 : index to i32
            %8 = arith.addi %7, %c-2_i32 : i32
            %9 = arith.addi %1, %6 : i32
            %10 = arith.cmpi sge, %9, %c0_i32 : i32
            %11 = arith.cmpi sgt, %9, %c11_i32 : i32
            %12 = arith.andi %10, %11 : i1
            %13 = arith.xori %12, %true : i1
            %14 = arith.andi %13, %10 : i1
            %15 = arith.select %12, %c11_i32, %c0_i32 : i32
            %16 = arith.select %14, %9, %15 : i32
            %17 = arith.addi %2, %8 : i32
            %18 = arith.cmpi sge, %17, %c0_i32 : i32
            %19 = arith.cmpi sgt, %17, %c11_i32 : i32
            %20 = arith.andi %18, %19 : i1
            %21 = arith.xori %20, %true : i1
            %22 = arith.andi %21, %18 : i1
            %23 = arith.select %20, %c11_i32, %c0_i32 : i32
            %24 = arith.select %22, %17, %23 : i32
            %25 = arith.index_cast %16 : i32 to index
            %26 = arith.index_cast %24 : i32 to index
            %27 = memref.load %alloca_2[%25, %26] : memref<12x12xf32>
            %28 = affine.load %alloca_6[] : memref<f32>
            %29 = arith.cmpf ogt, %27, %28 : f32
            scf.if %29 {
              affine.store %27, %alloca_6[] : memref<f32>
            }
          }
        }
        %4 = affine.load %alloca_6[] : memref<f32>
        affine.store %4, %alloca_3[%arg0, %arg1] : memref<12x12xf32>
      }
    }
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = affine.load %alloca_3[%arg0, %arg1] : memref<12x12xf32>
        %alloca_6 = memref.alloca() : memref<f32>
        affine.store %3, %alloca_6[] : memref<f32>
        affine.for %arg2 = 0 to 5 {
          %5 = arith.index_cast %arg2 : index to i32
          %6 = arith.addi %5, %c-2_i32 : i32
          affine.for %arg3 = 0 to 5 {
            %7 = arith.index_cast %arg3 : index to i32
            %8 = arith.addi %7, %c-2_i32 : i32
            %9 = arith.addi %1, %6 : i32
            %10 = arith.cmpi sge, %9, %c0_i32 : i32
            %11 = arith.cmpi sgt, %9, %c11_i32 : i32
            %12 = arith.andi %10, %11 : i1
            %13 = arith.xori %12, %true : i1
            %14 = arith.andi %13, %10 : i1
            %15 = arith.select %12, %c11_i32, %c0_i32 : i32
            %16 = arith.select %14, %9, %15 : i32
            %17 = arith.addi %2, %8 : i32
            %18 = arith.cmpi sge, %17, %c0_i32 : i32
            %19 = arith.cmpi sgt, %17, %c11_i32 : i32
            %20 = arith.andi %18, %19 : i1
            %21 = arith.xori %20, %true : i1
            %22 = arith.andi %21, %18 : i1
            %23 = arith.select %20, %c11_i32, %c0_i32 : i32
            %24 = arith.select %22, %17, %23 : i32
            %25 = arith.index_cast %16 : i32 to index
            %26 = arith.index_cast %24 : i32 to index
            %27 = memref.load %alloca_3[%25, %26] : memref<12x12xf32>
            %28 = affine.load %alloca_6[] : memref<f32>
            %29 = arith.cmpf olt, %27, %28 : f32
            scf.if %29 {
              affine.store %27, %alloca_6[] : memref<f32>
            }
          }
        }
        %4 = affine.load %alloca_6[] : memref<f32>
        affine.store %4, %alloca[%arg0, %arg1] : memref<12x12xf32>
      }
    }
    %alloca_5 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_5[] : memref<i32>
    affine.store %c0_i32, %alloca_5[] : memref<i32>
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = affine.load %alloca[%arg0, %arg1] : memref<12x12xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.fptosi %4 : f32 to i32
        %6 = arith.addi %2, %1 : i32
        %7 = arith.andi %6, %c3_i32 : i32
        %8 = arith.addi %5, %7 : i32
        %9 = affine.load %alloca_5[] : memref<i32>
        %10 = arith.addi %9, %8 : i32
        affine.store %10, %alloca_5[] : memref<i32>
      }
    }
    %0 = affine.load %alloca_5[] : memref<i32>
    call @pcl_validate_outputs(%alloca_4, %alloca_3, %alloca_2, %alloca) : (memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>) -> ()
    return %0 : i32
  }
  func.func private @pcl_validate_outputs(memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>)
}

