module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant 1.000000e+30 : f32
    %false = arith.constant false
    %cst_0 = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c24_i32 = arith.constant 24 : i32
    %c56_i32 = arith.constant 56 : i32
    %c30_i32 = arith.constant 30 : i32
    %cst_1 = arith.constant 1.000000e-01 : f32
    %cst_2 = arith.constant 5.000000e-03 : f32
    %c7_i32 = arith.constant 7 : i32
    %c5_i32 = arith.constant 5 : i32
    %c9_i32 = arith.constant 9 : i32
    %alloca = memref.alloca() : memref<8x8xi32>
    %alloca_3 = memref.alloca() : memref<8x8xf32>
    %alloca_4 = memref.alloca() : memref<8x8xf32>
    affine.for %arg0 = 0 to 8 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 8 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.cmpi sgt, %2, %c24_i32 : i32
        %4 = arith.extui %3 : i1 to i32
        %5 = arith.cmpi sgt, %2, %c56_i32 : i32
        %6 = arith.extui %5 : i1 to i32
        %7 = arith.addi %4, %6 : i32
        %8 = arith.cmpi sgt, %1, %c30_i32 : i32
        %9 = arith.extui %8 : i1 to i32
        %10 = arith.addi %7, %9 : i32
        %11 = arith.sitofp %10 : i32 to f32
        %12 = arith.mulf %11, %cst_1 : f32
        %13 = arith.muli %2, %c7_i32 : i32
        %14 = arith.muli %1, %c5_i32 : i32
        %15 = arith.addi %13, %14 : i32
        %16 = arith.remsi %15, %c9_i32 : i32
        %17 = arith.sitofp %16 : i32 to f32
        %18 = arith.mulf %17, %cst_2 : f32
        %19 = arith.addf %12, %18 : f32
        affine.store %19, %alloca_4[%arg0, %arg1] : memref<8x8xf32>
        %20 = arith.muli %1, %c8_i32 : i32
        %21 = arith.addi %20, %2 : i32
        affine.store %21, %alloca[%arg0, %arg1] : memref<8x8xi32>
      }
    }
    affine.for %arg0 = 0 to 8 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 8 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.addi %2, %c-1_i32 : i32
        %4 = arith.cmpi sge, %3, %c0_i32 : i32
        %5 = arith.cmpi sgt, %3, %c7_i32 : i32
        %6 = arith.andi %4, %5 : i1
        %7 = arith.xori %6, %true : i1
        %8 = arith.andi %7, %4 : i1
        %9 = arith.select %6, %c7_i32, %c0_i32 : i32
        %10 = arith.select %8, %3, %9 : i32
        %11 = arith.addi %2, %c1_i32 : i32
        %12 = arith.cmpi sge, %11, %c0_i32 : i32
        %13 = arith.cmpi sgt, %11, %c7_i32 : i32
        %14 = arith.andi %12, %13 : i1
        %15 = arith.xori %14, %true : i1
        %16 = arith.andi %15, %12 : i1
        %17 = arith.select %14, %c7_i32, %c0_i32 : i32
        %18 = arith.select %16, %11, %17 : i32
        %19 = arith.addi %1, %c-1_i32 : i32
        %20 = arith.cmpi sge, %19, %c0_i32 : i32
        %21 = arith.cmpi sgt, %19, %c7_i32 : i32
        %22 = arith.andi %20, %21 : i1
        %23 = arith.xori %22, %true : i1
        %24 = arith.andi %23, %20 : i1
        %25 = arith.select %22, %c7_i32, %c0_i32 : i32
        %26 = arith.select %24, %19, %25 : i32
        %27 = arith.addi %1, %c1_i32 : i32
        %28 = arith.cmpi sge, %27, %c0_i32 : i32
        %29 = arith.cmpi sgt, %27, %c7_i32 : i32
        %30 = arith.andi %28, %29 : i1
        %31 = arith.xori %30, %true : i1
        %32 = arith.andi %31, %28 : i1
        %33 = arith.select %30, %c7_i32, %c0_i32 : i32
        %34 = arith.select %32, %27, %33 : i32
        %35 = arith.index_cast %18 : i32 to index
        %36 = memref.load %alloca_4[%arg0, %35] : memref<8x8xf32>
        %37 = arith.index_cast %10 : i32 to index
        %38 = memref.load %alloca_4[%arg0, %37] : memref<8x8xf32>
        %39 = arith.subf %36, %38 : f32
        %40 = arith.cmpf olt, %39, %cst_0 : f32
        %41 = scf.if %40 -> (f32) {
          %50 = arith.negf %39 : f32
          scf.yield %50 : f32
        } else {
          scf.yield %39 : f32
        }
        %42 = arith.index_cast %34 : i32 to index
        %43 = memref.load %alloca_4[%42, %arg1] : memref<8x8xf32>
        %44 = arith.index_cast %26 : i32 to index
        %45 = memref.load %alloca_4[%44, %arg1] : memref<8x8xf32>
        %46 = arith.subf %43, %45 : f32
        %47 = arith.cmpf olt, %46, %cst_0 : f32
        %48 = scf.if %47 -> (f32) {
          %50 = arith.negf %46 : f32
          scf.yield %50 : f32
        } else {
          scf.yield %46 : f32
        }
        %49 = arith.addf %41, %48 : f32
        affine.store %49, %alloca_3[%arg0, %arg1] : memref<8x8xf32>
      }
    }
    affine.for %arg0 = 0 to 2 {
      affine.for %arg1 = 0 to 8 {
        %1 = arith.index_cast %arg1 : index to i32
        affine.for %arg2 = 0 to 8 {
          %2 = arith.index_cast %arg2 : index to i32
          %3 = affine.load %alloca[%arg1, %arg2] : memref<8x8xi32>
          %alloca_6 = memref.alloca() : memref<i32>
          affine.store %3, %alloca_6[] : memref<i32>
          %alloca_7 = memref.alloca() : memref<f32>
          affine.store %cst, %alloca_7[] : memref<f32>
          affine.for %arg3 = 0 to 3 {
            %5 = arith.index_cast %arg3 : index to i32
            %6 = arith.addi %5, %c-1_i32 : i32
            affine.for %arg4 = 0 to 3 {
              %7 = arith.index_cast %arg4 : index to i32
              %8 = arith.addi %7, %c-1_i32 : i32
              %9 = arith.addi %1, %6 : i32
              %10 = arith.cmpi sge, %9, %c0_i32 : i32
              %11 = arith.cmpi sgt, %9, %c7_i32 : i32
              %12 = arith.andi %10, %11 : i1
              %13 = arith.xori %12, %true : i1
              %14 = arith.andi %13, %10 : i1
              %15 = arith.select %12, %c7_i32, %c0_i32 : i32
              %16 = arith.select %14, %9, %15 : i32
              %17 = arith.addi %2, %8 : i32
              %18 = arith.cmpi sge, %17, %c0_i32 : i32
              %19 = arith.cmpi sgt, %17, %c7_i32 : i32
              %20 = arith.andi %18, %19 : i1
              %21 = arith.xori %20, %true : i1
              %22 = arith.andi %21, %18 : i1
              %23 = arith.select %20, %c7_i32, %c0_i32 : i32
              %24 = arith.select %22, %17, %23 : i32
              %25 = affine.load %alloca_4[%arg1, %arg2] : memref<8x8xf32>
              %26 = arith.index_cast %16 : i32 to index
              %27 = arith.index_cast %24 : i32 to index
              %28 = memref.load %alloca_4[%26, %27] : memref<8x8xf32>
              %29 = arith.subf %25, %28 : f32
              %30 = arith.cmpf olt, %29, %cst_0 : f32
              %31 = scf.if %30 -> (f32) {
                %37 = arith.negf %29 : f32
                scf.yield %37 : f32
              } else {
                scf.yield %29 : f32
              }
              %32 = memref.load %alloca_3[%26, %27] : memref<8x8xf32>
              %33 = arith.addf %31, %32 : f32
              %34 = affine.load %alloca_7[] : memref<f32>
              %35 = arith.cmpf olt, %33, %34 : f32
              %36 = scf.if %35 -> (i1) {
                scf.yield %true : i1
              } else {
                %37 = arith.cmpf oeq, %33, %34 : f32
                %38 = scf.if %37 -> (i1) {
                  %39 = memref.load %alloca[%26, %27] : memref<8x8xi32>
                  %40 = affine.load %alloca_6[] : memref<i32>
                  %41 = arith.cmpi slt, %39, %40 : i32
                  scf.yield %41 : i1
                } else {
                  scf.yield %false : i1
                }
                scf.yield %38 : i1
              }
              scf.if %36 {
                affine.store %33, %alloca_7[] : memref<f32>
                %37 = memref.load %alloca[%26, %27] : memref<8x8xi32>
                affine.store %37, %alloca_6[] : memref<i32>
              }
            }
          }
          %4 = affine.load %alloca_6[] : memref<i32>
          affine.store %4, %alloca[%arg1, %arg2] : memref<8x8xi32>
        }
      }
    }
    %alloca_5 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_5[] : memref<i32>
    affine.store %c0_i32, %alloca_5[] : memref<i32>
    affine.for %arg0 = 0 to 8 {
      affine.for %arg1 = 0 to 8 {
        %1 = affine.load %alloca[%arg0, %arg1] : memref<8x8xi32>
        %2 = arith.andi %1, %c255_i32 : i32
        %3 = affine.load %alloca_5[] : memref<i32>
        %4 = arith.addi %3, %2 : i32
        affine.store %4, %alloca_5[] : memref<i32>
      }
    }
    %0 = affine.load %alloca_5[] : memref<i32>
    call @pcl_validate_outputs(%alloca_4, %alloca_3, %alloca) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xi32>) -> ()
    return %0 : i32
  }
  func.func private @pcl_validate_outputs(memref<8x8xf32>, memref<8x8xf32>, memref<8x8xi32>)
}

