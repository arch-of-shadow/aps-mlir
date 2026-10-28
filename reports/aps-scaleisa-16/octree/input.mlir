module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.370000e+02 : f32
    %c137_i32 = arith.constant 137 : i32
    %c5_i32 = arith.constant 5 : i32
    %c19_i32 = arith.constant 19 : i32
    %cst_0 = arith.constant 1.310000e+02 : f32
    %c131_i32 = arith.constant 131 : i32
    %c11_i32 = arith.constant 11 : i32
    %c17_i32 = arith.constant 17 : i32
    %cst_1 = arith.constant 1.270000e+02 : f32
    %c127_i32 = arith.constant 127 : i32
    %c7_i32 = arith.constant 7 : i32
    %c13_i32 = arith.constant 13 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %alloca_2 = memref.alloca() : memref<128xi32>
    %alloca_3 = memref.alloca() : memref<4x4096xi32>
    %alloca_4 = memref.alloca() : memref<2048x3xf32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg0 = 0 to 4 {
      affine.for %arg1 = 0 to 4096 {
        affine.store %c0_i32, %alloca_3[%arg0, %arg1] : memref<4x4096xi32>
      }
    }
    affine.for %arg0 = 0 to 2048 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.muli %1, %c13_i32 : i32
      %3 = arith.addi %2, %c7_i32 : i32
      %4 = arith.remsi %3, %c127_i32 : i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.divf %5, %cst_1 : f32
      affine.store %6, %alloca_4[%arg0, 0] : memref<2048x3xf32>
      %7 = arith.muli %1, %c17_i32 : i32
      %8 = arith.addi %7, %c11_i32 : i32
      %9 = arith.remsi %8, %c131_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.divf %10, %cst_0 : f32
      affine.store %11, %alloca_4[%arg0, 1] : memref<2048x3xf32>
      %12 = arith.muli %1, %c19_i32 : i32
      %13 = arith.addi %12, %c5_i32 : i32
      %14 = arith.remsi %13, %c137_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.divf %15, %cst : f32
      affine.store %16, %alloca_4[%arg0, 2] : memref<2048x3xf32>
    }
    %cast = memref.cast %alloca_4 : memref<2048x3xf32> to memref<?x3xf32>
    %cast_5 = memref.cast %alloca_3 : memref<4x4096xi32> to memref<?x4096xi32>
    call @_ZL18build_level_countsPA3_fPA4096_i(%cast, %cast_5) : (memref<?x3xf32>, memref<?x4096xi32>) -> ()
    %cast_6 = memref.cast %alloca_2 : memref<128xi32> to memref<?xi32>
    call @_ZL15query_neighborsPA3_fPA4096_iPi(%cast, %cast_5, %cast_6) : (memref<?x3xf32>, memref<?x4096xi32>, memref<?xi32>) -> ()
    affine.for %arg0 = 0 to 128 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = affine.load %alloca_2[%arg0] : memref<128xi32>
      %3 = arith.andi %1, %c7_i32 : i32
      %4 = arith.addi %3, %c1_i32 : i32
      %5 = arith.muli %2, %4 : i32
      %6 = affine.load %alloca[] : memref<i32>
      %7 = arith.addi %6, %5 : i32
      affine.store %7, %alloca[] : memref<i32>
    }
    %0 = affine.load %alloca[] : memref<i32>
    return %0 : i32
  }
  func.func private @_ZL18build_level_countsPA3_fPA4096_i(%arg0: memref<?x3xf32>, %arg1: memref<?x4096xi32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 4 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.shrsi %c16_i32, %0 : i32
      affine.for %arg3 = 0 to 2048 {
        %2 = affine.load %arg0[%arg3, 0] : memref<?x3xf32>
        %3 = arith.sitofp %1 : i32 to f32
        %4 = arith.mulf %2, %3 : f32
        %5 = arith.fptosi %4 : f32 to i32
        %6 = arith.addi %1, %c-1_i32 : i32
        %7 = func.call @_ZL7clamp_iiii(%5, %c0_i32, %6) : (i32, i32, i32) -> i32
        %8 = affine.load %arg0[%arg3, 1] : memref<?x3xf32>
        %9 = arith.mulf %8, %3 : f32
        %10 = arith.fptosi %9 : f32 to i32
        %11 = func.call @_ZL7clamp_iiii(%10, %c0_i32, %6) : (i32, i32, i32) -> i32
        %12 = affine.load %arg0[%arg3, 2] : memref<?x3xf32>
        %13 = arith.mulf %12, %3 : f32
        %14 = arith.fptosi %13 : f32 to i32
        %15 = func.call @_ZL7clamp_iiii(%14, %c0_i32, %6) : (i32, i32, i32) -> i32
        %16 = arith.muli %7, %c16_i32 : i32
        %17 = arith.addi %16, %11 : i32
        %18 = arith.muli %17, %c16_i32 : i32
        %19 = arith.addi %18, %15 : i32
        %20 = arith.index_cast %19 : i32 to index
        %21 = memref.load %arg1[%arg2, %20] : memref<?x4096xi32>
        %22 = arith.addi %21, %c1_i32 : i32
        memref.store %22, %arg1[%arg2, %20] : memref<?x4096xi32>
      }
    }
    return
  }
  func.func private @_ZL15query_neighborsPA3_fPA4096_iPi(%arg0: memref<?x3xf32>, %arg1: memref<?x4096xi32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg3 = 0 to 128 {
      affine.store %c0_i32, %alloca[] : memref<i32>
      %0 = affine.load %arg0[%arg3, 0] : memref<?x3xf32>
      %1 = affine.load %arg0[%arg3, 1] : memref<?x3xf32>
      %2 = affine.load %arg0[%arg3, 2] : memref<?x3xf32>
      affine.for %arg4 = 0 to 4 {
        %4 = arith.index_cast %arg4 : index to i32
        %5 = arith.shrsi %c16_i32, %4 : i32
        %6 = arith.sitofp %5 : i32 to f32
        %7 = arith.mulf %0, %6 : f32
        %8 = arith.fptosi %7 : f32 to i32
        %9 = arith.addi %5, %c-1_i32 : i32
        %10 = func.call @_ZL7clamp_iiii(%8, %c0_i32, %9) : (i32, i32, i32) -> i32
        %11 = arith.mulf %1, %6 : f32
        %12 = arith.fptosi %11 : f32 to i32
        %13 = func.call @_ZL7clamp_iiii(%12, %c0_i32, %9) : (i32, i32, i32) -> i32
        %14 = arith.mulf %2, %6 : f32
        %15 = arith.fptosi %14 : f32 to i32
        %16 = func.call @_ZL7clamp_iiii(%15, %c0_i32, %9) : (i32, i32, i32) -> i32
        affine.for %arg5 = 0 to 3 {
          %17 = arith.index_cast %arg5 : index to i32
          %18 = arith.addi %17, %c-1_i32 : i32
          affine.for %arg6 = 0 to 3 {
            %19 = arith.index_cast %arg6 : index to i32
            %20 = arith.addi %19, %c-1_i32 : i32
            affine.for %arg7 = 0 to 3 {
              %21 = arith.index_cast %arg7 : index to i32
              %22 = arith.addi %21, %c-1_i32 : i32
              %23 = arith.addi %10, %22 : i32
              %24 = func.call @_ZL7clamp_iiii(%23, %c0_i32, %9) : (i32, i32, i32) -> i32
              %25 = arith.addi %13, %20 : i32
              %26 = func.call @_ZL7clamp_iiii(%25, %c0_i32, %9) : (i32, i32, i32) -> i32
              %27 = arith.addi %16, %18 : i32
              %28 = func.call @_ZL7clamp_iiii(%27, %c0_i32, %9) : (i32, i32, i32) -> i32
              %29 = arith.muli %24, %c16_i32 : i32
              %30 = arith.addi %29, %26 : i32
              %31 = arith.muli %30, %c16_i32 : i32
              %32 = arith.addi %31, %28 : i32
              %33 = arith.index_cast %32 : i32 to index
              %34 = memref.load %arg1[%arg4, %33] : memref<?x4096xi32>
              %35 = arith.addi %4, %c1_i32 : i32
              %36 = arith.muli %34, %35 : i32
              %37 = affine.load %alloca[] : memref<i32>
              %38 = arith.addi %37, %36 : i32
              affine.store %38, %alloca[] : memref<i32>
            }
          }
        }
      }
      %3 = affine.load %alloca[] : memref<i32>
      affine.store %3, %arg2[%arg3] : memref<?xi32>
    }
    return
  }
  func.func private @_ZL7clamp_iiii(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi slt, %arg0, %arg1 : i32
    %1 = arith.cmpi sge, %arg0, %arg1 : i32
    %2 = arith.select %0, %arg1, %c0_i32 : i32
    %3 = arith.cmpi sgt, %arg0, %arg2 : i32
    %4 = arith.andi %1, %3 : i1
    %5 = arith.xori %4, %true : i1
    %6 = arith.andi %5, %1 : i1
    %7 = arith.select %4, %arg2, %2 : i32
    %8 = arith.select %6, %arg0, %7 : i32
    return %8 : i32
  }
}

