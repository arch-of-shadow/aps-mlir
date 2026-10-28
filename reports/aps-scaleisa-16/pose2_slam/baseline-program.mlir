module {
  memref.global @bench_input_0 : memref<40xi32> = uninitialized
  memref.global @bench_input_1 : memref<40xi32> = uninitialized
  memref.global @bench_input_2 : memref<40x3xf32> = uninitialized
  memref.global @bench_input_3 : memref<40x3xf32> = uninitialized
  memref.global @bench_input_4 : memref<24x3xf32> = uninitialized
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.010000e+02 : f32
    %cst_0 = arith.constant 1.730000e+02 : f32
    %cst_1 = arith.constant 2.570000e+02 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant 1.000000e-03 : f32
    %cst_4 = arith.constant 2.800000e-01 : f32
    %alloca = memref.alloca() : memref<24x3xf32>
    %alloca_5 = memref.alloca() : memref<40x3xf32>
    %alloca_6 = memref.alloca() : memref<40x3xf32>
    %alloca_7 = memref.alloca() : memref<40xi32>
    %alloca_8 = memref.alloca() : memref<40xi32>
    %0 = memref.get_global @bench_input_0 : memref<40xi32>
    affine.for %arg0 = 0 to 40 {
      %11 = affine.load %0[%arg0] : memref<40xi32>
      affine.store %11, %alloca_8[%arg0] : memref<40xi32>
    }
    %1 = memref.get_global @bench_input_1 : memref<40xi32>
    affine.for %arg0 = 0 to 40 {
      %11 = affine.load %1[%arg0] : memref<40xi32>
      affine.store %11, %alloca_7[%arg0] : memref<40xi32>
    }
    %2 = memref.get_global @bench_input_2 : memref<40x3xf32>
    affine.for %arg0 = 0 to 40 {
      affine.for %arg1 = 0 to 3 {
        %11 = affine.load %2[%arg0, %arg1] : memref<40x3xf32>
        affine.store %11, %alloca_6[%arg0, %arg1] : memref<40x3xf32>
      }
    }
    %3 = memref.get_global @bench_input_3 : memref<40x3xf32>
    affine.for %arg0 = 0 to 40 {
      affine.for %arg1 = 0 to 3 {
        %11 = affine.load %3[%arg0, %arg1] : memref<40x3xf32>
        affine.store %11, %alloca_5[%arg0, %arg1] : memref<40x3xf32>
      }
    }
    %4 = memref.get_global @bench_input_4 : memref<24x3xf32>
    affine.for %arg0 = 0 to 24 {
      affine.for %arg1 = 0 to 3 {
        %11 = affine.load %4[%arg0, %arg1] : memref<24x3xf32>
        affine.store %11, %alloca[%arg0, %arg1] : memref<24x3xf32>
      }
    }
    %alloca_9 = memref.alloca() : memref<24x3xf32>
    %alloca_10 = memref.alloca() : memref<24x3xf32>
    affine.for %arg0 = 0 to 8 {
      affine.for %arg1 = 0 to 24 {
        affine.for %arg2 = 0 to 3 {
          affine.store %cst_2, %alloca_10[%arg1, %arg2] : memref<24x3xf32>
          affine.store %cst_3, %alloca_9[%arg1, %arg2] : memref<24x3xf32>
        }
      }
      affine.for %arg1 = 0 to 40 {
        %11 = affine.load %alloca_8[%arg1] : memref<40xi32>
        %12 = affine.load %alloca_7[%arg1] : memref<40xi32>
        affine.for %arg2 = 0 to 3 {
          %13 = arith.index_cast %12 : i32 to index
          %14 = memref.load %alloca[%13, %arg2] : memref<24x3xf32>
          %15 = arith.index_cast %11 : i32 to index
          %16 = memref.load %alloca[%15, %arg2] : memref<24x3xf32>
          %17 = arith.subf %14, %16 : f32
          %18 = affine.load %alloca_6[%arg1, %arg2] : memref<40x3xf32>
          %19 = arith.subf %17, %18 : f32
          %20 = affine.load %alloca_5[%arg1, %arg2] : memref<40x3xf32>
          %21 = arith.mulf %20, %19 : f32
          %22 = memref.load %alloca_10[%15, %arg2] : memref<24x3xf32>
          %23 = arith.addf %22, %21 : f32
          memref.store %23, %alloca_10[%15, %arg2] : memref<24x3xf32>
          %24 = memref.load %alloca_10[%13, %arg2] : memref<24x3xf32>
          %25 = arith.subf %24, %21 : f32
          memref.store %25, %alloca_10[%13, %arg2] : memref<24x3xf32>
          %26 = affine.load %alloca_5[%arg1, %arg2] : memref<40x3xf32>
          %27 = memref.load %alloca_9[%15, %arg2] : memref<24x3xf32>
          %28 = arith.addf %27, %26 : f32
          memref.store %28, %alloca_9[%15, %arg2] : memref<24x3xf32>
          %29 = affine.load %alloca_5[%arg1, %arg2] : memref<40x3xf32>
          %30 = memref.load %alloca_9[%13, %arg2] : memref<24x3xf32>
          %31 = arith.addf %30, %29 : f32
          memref.store %31, %alloca_9[%13, %arg2] : memref<24x3xf32>
        }
      }
      affine.store %cst_2, %alloca[0, 0] : memref<24x3xf32>
      affine.store %cst_2, %alloca[0, 1] : memref<24x3xf32>
      affine.store %cst_2, %alloca[0, 2] : memref<24x3xf32>
      affine.for %arg1 = 0 to 23 {
        affine.for %arg2 = 0 to 3 {
          %11 = affine.load %alloca_10[%arg1 + 1, %arg2] : memref<24x3xf32>
          %12 = arith.mulf %11, %cst_4 : f32
          %13 = affine.load %alloca_9[%arg1 + 1, %arg2] : memref<24x3xf32>
          %14 = arith.divf %12, %13 : f32
          %15 = affine.load %alloca[%arg1 + 1, %arg2] : memref<24x3xf32>
          %16 = arith.subf %15, %14 : f32
          affine.store %16, %alloca[%arg1 + 1, %arg2] : memref<24x3xf32>
        }
      }
    }
    %alloca_11 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_11[] : memref<i32>
    affine.store %c0_i32, %alloca_11[] : memref<i32>
    affine.for %arg0 = 0 to 24 {
      %11 = affine.load %alloca[%arg0, 0] : memref<24x3xf32>
      %12 = arith.mulf %11, %cst : f32
      %13 = arith.fptosi %12 : f32 to i32
      %14 = affine.load %alloca_11[] : memref<i32>
      %15 = arith.addi %14, %13 : i32
      affine.store %15, %alloca_11[] : memref<i32>
      %16 = affine.load %alloca[%arg0, 1] : memref<24x3xf32>
      %17 = arith.mulf %16, %cst_0 : f32
      %18 = arith.fptosi %17 : f32 to i32
      %19 = arith.addi %15, %18 : i32
      affine.store %19, %alloca_11[] : memref<i32>
      %20 = affine.load %alloca[%arg0, 2] : memref<24x3xf32>
      %21 = arith.mulf %20, %cst_1 : f32
      %22 = arith.fptosi %21 : f32 to i32
      %23 = arith.addi %19, %22 : i32
      affine.store %23, %alloca_11[] : memref<i32>
    }
    %5 = affine.load %alloca_11[] : memref<i32>
    %6 = arith.sitofp %5 : i32 to f32
    %7 = arith.cmpi slt, %5, %c0_i32 : i32
    %8 = scf.if %7 -> (f32) {
      %11 = arith.negf %6 : f32
      scf.yield %11 : f32
    } else {
      scf.yield %6 : f32
    }
    %9 = arith.cmpf ogt, %8, %cst_2 : f32
    %10 = arith.select %9, %5, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_8, %alloca_7, %alloca_6, %alloca_5, %alloca) : (memref<40xi32>, memref<40xi32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>) -> ()
    return %10 : i32
  }
  func.func private @pcl_validate_outputs(memref<40xi32>, memref<40xi32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>)
}

