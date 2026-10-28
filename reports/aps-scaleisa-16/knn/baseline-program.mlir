module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.560000e+02 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant 1.000000e+30 : f32
    %c-1_i32 = arith.constant -1 : i32
    %false = arith.constant false
    %c31_i32 = arith.constant 31 : i32
    %c13_i32 = arith.constant 13 : i32
    %c29_i32 = arith.constant 29 : i32
    %c9_i32 = arith.constant 9 : i32
    %c23_i32 = arith.constant 23 : i32
    %cst_2 = arith.constant 2.700000e-02 : f32
    %c89_i32 = arith.constant 89 : i32
    %c7_i32 = arith.constant 7 : i32
    %c19_i32 = arith.constant 19 : i32
    %cst_3 = arith.constant 2.900000e-02 : f32
    %c97_i32 = arith.constant 97 : i32
    %c5_i32 = arith.constant 5 : i32
    %c17_i32 = arith.constant 17 : i32
    %cst_4 = arith.constant 3.100000e-02 : f32
    %c101_i32 = arith.constant 101 : i32
    %c3_i32 = arith.constant 3 : i32
    %c11_i32 = arith.constant 11 : i32
    %alloca = memref.alloca() : memref<4x8xi32>
    %alloca_5 = memref.alloca() : memref<4x8xf32>
    %alloca_6 = memref.alloca() : memref<4x3xf32>
    %alloca_7 = memref.alloca() : memref<32x3xf32>
    affine.for %arg0 = 0 to 32 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.muli %6, %c11_i32 : i32
      %8 = arith.addi %7, %c3_i32 : i32
      %9 = arith.remsi %8, %c101_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_4 : f32
      affine.store %11, %alloca_7[%arg0, 0] : memref<32x3xf32>
      %12 = arith.muli %6, %c17_i32 : i32
      %13 = arith.addi %12, %c5_i32 : i32
      %14 = arith.remsi %13, %c97_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.mulf %15, %cst_3 : f32
      affine.store %16, %alloca_7[%arg0, 1] : memref<32x3xf32>
      %17 = arith.muli %6, %c19_i32 : i32
      %18 = arith.addi %17, %c7_i32 : i32
      %19 = arith.remsi %18, %c89_i32 : i32
      %20 = arith.sitofp %19 : i32 to f32
      %21 = arith.mulf %20, %cst_2 : f32
      affine.store %21, %alloca_7[%arg0, 2] : memref<32x3xf32>
    }
    affine.for %arg0 = 0 to 4 {
      %6 = arith.index_cast %arg0 : index to i32
      %7 = arith.muli %6, %c23_i32 : i32
      %8 = arith.addi %7, %c9_i32 : i32
      %9 = arith.remsi %8, %c101_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_4 : f32
      affine.store %11, %alloca_6[%arg0, 0] : memref<4x3xf32>
      %12 = arith.muli %6, %c29_i32 : i32
      %13 = arith.addi %12, %c13_i32 : i32
      %14 = arith.remsi %13, %c97_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.mulf %15, %cst_3 : f32
      affine.store %16, %alloca_6[%arg0, 1] : memref<4x3xf32>
      %17 = arith.muli %6, %c31_i32 : i32
      %18 = arith.addi %17, %c17_i32 : i32
      %19 = arith.remsi %18, %c89_i32 : i32
      %20 = arith.sitofp %19 : i32 to f32
      %21 = arith.mulf %20, %cst_2 : f32
      affine.store %21, %alloca_6[%arg0, 2] : memref<4x3xf32>
    }
    %alloca_8 = memref.alloca() : memref<8xi32>
    %alloca_9 = memref.alloca() : memref<8xf32>
    affine.for %arg0 = 0 to 4 {
      affine.for %arg1 = 0 to 8 {
        affine.store %cst_1, %alloca_5[%arg0, %arg1] : memref<4x8xf32>
        affine.store %c-1_i32, %alloca[%arg0, %arg1] : memref<4x8xi32>
      }
      affine.for %arg1 = 0 to 32 {
        %6 = arith.index_cast %arg1 : index to i32
        %7 = affine.load %alloca_7[%arg1, 0] : memref<32x3xf32>
        %8 = affine.load %alloca_6[%arg0, 0] : memref<4x3xf32>
        %9 = arith.subf %7, %8 : f32
        %10 = affine.load %alloca_7[%arg1, 1] : memref<32x3xf32>
        %11 = affine.load %alloca_6[%arg0, 1] : memref<4x3xf32>
        %12 = arith.subf %10, %11 : f32
        %13 = affine.load %alloca_7[%arg1, 2] : memref<32x3xf32>
        %14 = affine.load %alloca_6[%arg0, 2] : memref<4x3xf32>
        %15 = arith.subf %13, %14 : f32
        %16 = arith.mulf %9, %9 : f32
        %17 = arith.mulf %12, %12 : f32
        %18 = arith.addf %16, %17 : f32
        %19 = arith.mulf %15, %15 : f32
        %20 = arith.addf %18, %19 : f32
        %alloca_11 = memref.alloca() : memref<i32>
        affine.store %c0_i32, %alloca_11[] : memref<i32>
        affine.for %arg2 = 0 to 8 {
          %21 = arith.index_cast %arg2 : index to i32
          %22 = affine.load %alloca_11[] : memref<i32>
          %23 = arith.cmpi eq, %22, %c0_i32 : i32
          %24 = scf.if %23 -> (i1) {
            %25 = affine.load %alloca_5[%arg0, %arg2] : memref<4x8xf32>
            %26 = arith.cmpf olt, %20, %25 : f32
            scf.yield %26 : i1
          } else {
            scf.yield %false : i1
          }
          scf.if %24 {
            affine.store %20, %alloca_9[%arg2] : memref<8xf32>
            affine.store %6, %alloca_8[%arg2] : memref<8xi32>
            affine.store %c1_i32, %alloca_11[] : memref<i32>
          } else {
            %25 = arith.subi %21, %22 : i32
            %26 = arith.index_cast %25 : i32 to index
            %27 = memref.load %alloca_5[%arg0, %26] : memref<4x8xf32>
            affine.store %27, %alloca_9[%arg2] : memref<8xf32>
            %28 = memref.load %alloca[%arg0, %26] : memref<4x8xi32>
            affine.store %28, %alloca_8[%arg2] : memref<8xi32>
          }
        }
        affine.for %arg2 = 0 to 8 {
          %21 = affine.load %alloca_9[%arg2] : memref<8xf32>
          affine.store %21, %alloca_5[%arg0, %arg2] : memref<4x8xf32>
          %22 = affine.load %alloca_8[%arg2] : memref<8xi32>
          affine.store %22, %alloca[%arg0, %arg2] : memref<4x8xi32>
        }
      }
    }
    %alloca_10 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_10[] : memref<i32>
    affine.store %c0_i32, %alloca_10[] : memref<i32>
    affine.for %arg0 = 0 to 4 {
      affine.for %arg1 = 0 to 8 {
        %6 = affine.load %alloca[%arg0, %arg1] : memref<4x8xi32>
        %7 = arith.cmpi sge, %6, %c0_i32 : i32
        %8 = scf.if %7 -> (i32) {
          %11 = affine.load %alloca[%arg0, %arg1] : memref<4x8xi32>
          %12 = affine.load %alloca_5[%arg0, %arg1] : memref<4x8xf32>
          %13 = arith.mulf %12, %cst : f32
          %14 = arith.fptosi %13 : f32 to i32
          %15 = arith.addi %11, %14 : i32
          scf.yield %15 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %9 = affine.load %alloca_10[] : memref<i32>
        %10 = arith.addi %9, %8 : i32
        affine.store %10, %alloca_10[] : memref<i32>
      }
    }
    %0 = affine.load %alloca_10[] : memref<i32>
    %1 = arith.sitofp %0 : i32 to f32
    %2 = arith.cmpi slt, %0, %c0_i32 : i32
    %3 = scf.if %2 -> (f32) {
      %6 = arith.negf %1 : f32
      scf.yield %6 : f32
    } else {
      scf.yield %1 : f32
    }
    %4 = arith.cmpf ogt, %3, %cst_0 : f32
    %5 = arith.select %4, %0, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_7, %alloca_6, %alloca_5, %alloca) : (memref<32x3xf32>, memref<4x3xf32>, memref<4x8xf32>, memref<4x8xi32>) -> ()
    return %5 : i32
  }
  func.func private @pcl_validate_outputs(memref<32x3xf32>, memref<4x3xf32>, memref<4x8xf32>, memref<4x8xi32>)
}

