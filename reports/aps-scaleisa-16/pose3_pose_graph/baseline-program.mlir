module {
  memref.global @bench_input_0 : memref<18x6xf32> = uninitialized
  memref.global @bench_input_1 : memref<30xi32> = uninitialized
  memref.global @bench_input_2 : memref<30xi32> = uninitialized
  memref.global @bench_input_3 : memref<30x6xf32> = uninitialized
  memref.global @bench_input_4 : memref<30x6xf32> = uninitialized
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 8.300000e+01 : f32
    %cst_0 = arith.constant 1.900000e+01 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e-03 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_3 = arith.constant 2.000000e-01 : f32
    %cst_4 = arith.constant 1.200000e-01 : f32
    %alloca = memref.alloca() : memref<30x6xf32>
    %alloca_5 = memref.alloca() : memref<30x6xf32>
    %alloca_6 = memref.alloca() : memref<30xi32>
    %alloca_7 = memref.alloca() : memref<30xi32>
    %alloca_8 = memref.alloca() : memref<18x6xf32>
    %0 = memref.get_global @bench_input_0 : memref<18x6xf32>
    affine.for %arg0 = 0 to 18 {
      affine.for %arg1 = 0 to 6 {
        %11 = affine.load %0[%arg0, %arg1] : memref<18x6xf32>
        affine.store %11, %alloca_8[%arg0, %arg1] : memref<18x6xf32>
      }
    }
    %1 = memref.get_global @bench_input_1 : memref<30xi32>
    affine.for %arg0 = 0 to 30 {
      %11 = affine.load %1[%arg0] : memref<30xi32>
      affine.store %11, %alloca_7[%arg0] : memref<30xi32>
    }
    %2 = memref.get_global @bench_input_2 : memref<30xi32>
    affine.for %arg0 = 0 to 30 {
      %11 = affine.load %2[%arg0] : memref<30xi32>
      affine.store %11, %alloca_6[%arg0] : memref<30xi32>
    }
    %3 = memref.get_global @bench_input_3 : memref<30x6xf32>
    affine.for %arg0 = 0 to 30 {
      affine.for %arg1 = 0 to 6 {
        %11 = affine.load %3[%arg0, %arg1] : memref<30x6xf32>
        affine.store %11, %alloca_5[%arg0, %arg1] : memref<30x6xf32>
      }
    }
    %4 = memref.get_global @bench_input_4 : memref<30x6xf32>
    affine.for %arg0 = 0 to 30 {
      affine.for %arg1 = 0 to 6 {
        %11 = affine.load %4[%arg0, %arg1] : memref<30x6xf32>
        affine.store %11, %alloca[%arg0, %arg1] : memref<30x6xf32>
      }
    }
    %alloca_9 = memref.alloca() : memref<18x6xf32>
    %alloca_10 = memref.alloca() : memref<18x6xf32>
    affine.for %arg0 = 0 to 7 {
      affine.for %arg1 = 0 to 18 {
        affine.for %arg2 = 0 to 6 {
          affine.store %cst_1, %alloca_10[%arg1, %arg2] : memref<18x6xf32>
          affine.store %cst_2, %alloca_9[%arg1, %arg2] : memref<18x6xf32>
        }
      }
      affine.for %arg1 = 0 to 30 {
        %11 = affine.load %alloca_7[%arg1] : memref<30xi32>
        %12 = affine.load %alloca_6[%arg1] : memref<30xi32>
        affine.for %arg2 = 0 to 6 {
          %13 = arith.index_cast %12 : i32 to index
          %14 = memref.load %alloca_8[%13, %arg2] : memref<18x6xf32>
          %15 = arith.index_cast %11 : i32 to index
          %16 = memref.load %alloca_8[%15, %arg2] : memref<18x6xf32>
          %17 = arith.subf %14, %16 : f32
          %18 = affine.load %alloca_5[%arg1, %arg2] : memref<30x6xf32>
          %19 = arith.subf %17, %18 : f32
          %20 = affine.load %alloca[%arg1, %arg2] : memref<30x6xf32>
          %21 = arith.mulf %20, %19 : f32
          %22 = memref.load %alloca_10[%15, %arg2] : memref<18x6xf32>
          %23 = arith.addf %22, %21 : f32
          memref.store %23, %alloca_10[%15, %arg2] : memref<18x6xf32>
          %24 = memref.load %alloca_10[%13, %arg2] : memref<18x6xf32>
          %25 = arith.subf %24, %21 : f32
          memref.store %25, %alloca_10[%13, %arg2] : memref<18x6xf32>
          %26 = affine.load %alloca[%arg1, %arg2] : memref<30x6xf32>
          %27 = memref.load %alloca_9[%15, %arg2] : memref<18x6xf32>
          %28 = arith.addf %27, %26 : f32
          memref.store %28, %alloca_9[%15, %arg2] : memref<18x6xf32>
          %29 = affine.load %alloca[%arg1, %arg2] : memref<30x6xf32>
          %30 = memref.load %alloca_9[%13, %arg2] : memref<18x6xf32>
          %31 = arith.addf %30, %29 : f32
          memref.store %31, %alloca_9[%13, %arg2] : memref<18x6xf32>
        }
      }
      affine.for %arg1 = 0 to 6 {
        affine.store %cst_1, %alloca_8[0, %arg1] : memref<18x6xf32>
      }
      affine.for %arg1 = 0 to 17 {
        affine.for %arg2 = 0 to 6 {
          %11 = arith.index_cast %arg2 : index to i32
          %12 = arith.cmpi slt, %11, %c3_i32 : i32
          %13 = arith.select %12, %cst_3, %cst_4 : f32
          %14 = affine.load %alloca_10[%arg1 + 1, %arg2] : memref<18x6xf32>
          %15 = arith.mulf %13, %14 : f32
          %16 = affine.load %alloca_9[%arg1 + 1, %arg2] : memref<18x6xf32>
          %17 = arith.divf %15, %16 : f32
          %18 = affine.load %alloca_8[%arg1 + 1, %arg2] : memref<18x6xf32>
          %19 = arith.subf %18, %17 : f32
          affine.store %19, %alloca_8[%arg1 + 1, %arg2] : memref<18x6xf32>
        }
      }
    }
    %alloca_11 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_11[] : memref<i32>
    affine.store %c0_i32, %alloca_11[] : memref<i32>
    affine.for %arg0 = 0 to 18 {
      affine.for %arg1 = 0 to 6 {
        %11 = arith.index_cast %arg1 : index to i32
        %12 = affine.load %alloca_8[%arg0, %arg1] : memref<18x6xf32>
        %13 = arith.sitofp %11 : i32 to f32
        %14 = arith.mulf %13, %cst_0 : f32
        %15 = arith.addf %14, %cst : f32
        %16 = arith.mulf %12, %15 : f32
        %17 = arith.fptosi %16 : f32 to i32
        %18 = affine.load %alloca_11[] : memref<i32>
        %19 = arith.addi %18, %17 : i32
        affine.store %19, %alloca_11[] : memref<i32>
      }
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
    %9 = arith.cmpf ogt, %8, %cst_1 : f32
    %10 = arith.select %9, %5, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_8, %alloca_7, %alloca_6, %alloca_5, %alloca) : (memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<30x6xf32>, memref<30x6xf32>) -> ()
    return %10 : i32
  }
  func.func private @pcl_validate_outputs(memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<30x6xf32>, memref<30x6xf32>)
}

