module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c31_i32 = arith.constant 31 : i32
    %c13_i32 = arith.constant 13 : i32
    %c29_i32 = arith.constant 29 : i32
    %c9_i32 = arith.constant 9 : i32
    %c23_i32 = arith.constant 23 : i32
    %cst = arith.constant 2.700000e-02 : f32
    %c89_i32 = arith.constant 89 : i32
    %c7_i32 = arith.constant 7 : i32
    %c19_i32 = arith.constant 19 : i32
    %cst_0 = arith.constant 2.900000e-02 : f32
    %c97_i32 = arith.constant 97 : i32
    %c5_i32 = arith.constant 5 : i32
    %c17_i32 = arith.constant 17 : i32
    %cst_1 = arith.constant 3.100000e-02 : f32
    %c101_i32 = arith.constant 101 : i32
    %c3_i32 = arith.constant 3 : i32
    %c11_i32 = arith.constant 11 : i32
    %alloca = memref.alloca() : memref<4x8xi32>
    %alloca_2 = memref.alloca() : memref<4x8xf32>
    %alloca_3 = memref.alloca() : memref<4x3xf32>
    %alloca_4 = memref.alloca() : memref<32x3xf32>
    affine.for %arg0 = 0 to 32 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.muli %1, %c11_i32 : i32
      %3 = arith.addi %2, %c3_i32 : i32
      %4 = arith.remsi %3, %c101_i32 : i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.mulf %5, %cst_1 : f32
      affine.store %6, %alloca_4[%arg0, 0] : memref<32x3xf32>
      %7 = arith.muli %1, %c17_i32 : i32
      %8 = arith.addi %7, %c5_i32 : i32
      %9 = arith.remsi %8, %c97_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_0 : f32
      affine.store %11, %alloca_4[%arg0, 1] : memref<32x3xf32>
      %12 = arith.muli %1, %c19_i32 : i32
      %13 = arith.addi %12, %c7_i32 : i32
      %14 = arith.remsi %13, %c89_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.mulf %15, %cst : f32
      affine.store %16, %alloca_4[%arg0, 2] : memref<32x3xf32>
    }
    affine.for %arg0 = 0 to 4 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.muli %1, %c23_i32 : i32
      %3 = arith.addi %2, %c9_i32 : i32
      %4 = arith.remsi %3, %c101_i32 : i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.mulf %5, %cst_1 : f32
      affine.store %6, %alloca_3[%arg0, 0] : memref<4x3xf32>
      %7 = arith.muli %1, %c29_i32 : i32
      %8 = arith.addi %7, %c13_i32 : i32
      %9 = arith.remsi %8, %c97_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_0 : f32
      affine.store %11, %alloca_3[%arg0, 1] : memref<4x3xf32>
      %12 = arith.muli %1, %c31_i32 : i32
      %13 = arith.addi %12, %c17_i32 : i32
      %14 = arith.remsi %13, %c89_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.mulf %15, %cst : f32
      affine.store %16, %alloca_3[%arg0, 2] : memref<4x3xf32>
    }
    %cast = memref.cast %alloca_4 : memref<32x3xf32> to memref<?x3xf32>
    %cast_5 = memref.cast %alloca_3 : memref<4x3xf32> to memref<?x3xf32>
    %cast_6 = memref.cast %alloca_2 : memref<4x8xf32> to memref<?x8xf32>
    %cast_7 = memref.cast %alloca : memref<4x8xi32> to memref<?x8xi32>
    call @_ZL10knn_searchPA3_fS0_PA8_fPA8_i(%cast, %cast_5, %cast_6, %cast_7) : (memref<?x3xf32>, memref<?x3xf32>, memref<?x8xf32>, memref<?x8xi32>) -> ()
    %0 = call @_ZL9summarizePA8_fPA8_i(%cast_6, %cast_7) : (memref<?x8xf32>, memref<?x8xi32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL10knn_searchPA3_fS0_PA8_fPA8_i(%arg0: memref<?x3xf32>, %arg1: memref<?x3xf32>, %arg2: memref<?x8xf32>, %arg3: memref<?x8xi32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %false = arith.constant false
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 1.000000e+30 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %alloca_0 = memref.alloca() : memref<8xi32>
    %alloca_1 = memref.alloca() : memref<8xf32>
    affine.for %arg4 = 0 to 4 {
      affine.for %arg5 = 0 to 8 {
        affine.store %cst, %arg2[%arg4, %arg5] : memref<?x8xf32>
        affine.store %c-1_i32, %arg3[%arg4, %arg5] : memref<?x8xi32>
      }
      affine.for %arg5 = 0 to 32 {
        %0 = arith.index_cast %arg5 : index to i32
        %1 = "polygeist.subindex"(%arg0, %arg5) : (memref<?x3xf32>, index) -> memref<?xf32>
        %2 = "polygeist.subindex"(%arg1, %arg4) : (memref<?x3xf32>, index) -> memref<?xf32>
        %3 = func.call @_ZL7sq_distPfS_(%1, %2) : (memref<?xf32>, memref<?xf32>) -> f32
        affine.store %c0_i32, %alloca[] : memref<i32>
        affine.for %arg6 = 0 to 8 {
          %4 = arith.index_cast %arg6 : index to i32
          %5 = affine.load %alloca[] : memref<i32>
          %6 = arith.cmpi eq, %5, %c0_i32 : i32
          %7 = scf.if %6 -> (i1) {
            %8 = affine.load %arg2[%arg4, %arg6] : memref<?x8xf32>
            %9 = arith.cmpf olt, %3, %8 : f32
            scf.yield %9 : i1
          } else {
            scf.yield %false : i1
          }
          scf.if %7 {
            affine.store %3, %alloca_1[%arg6] : memref<8xf32>
            affine.store %0, %alloca_0[%arg6] : memref<8xi32>
            affine.store %c1_i32, %alloca[] : memref<i32>
          } else {
            %8 = arith.subi %4, %5 : i32
            %9 = arith.index_cast %8 : i32 to index
            %10 = memref.load %arg2[%arg4, %9] : memref<?x8xf32>
            affine.store %10, %alloca_1[%arg6] : memref<8xf32>
            %11 = memref.load %arg3[%arg4, %9] : memref<?x8xi32>
            affine.store %11, %alloca_0[%arg6] : memref<8xi32>
          }
        }
        affine.for %arg6 = 0 to 8 {
          %4 = affine.load %alloca_1[%arg6] : memref<8xf32>
          affine.store %4, %arg2[%arg4, %arg6] : memref<?x8xf32>
          %5 = affine.load %alloca_0[%arg6] : memref<8xi32>
          affine.store %5, %arg3[%arg4, %arg6] : memref<?x8xi32>
        }
      }
    }
    return
  }
  func.func private @_ZL9summarizePA8_fPA8_i(%arg0: memref<?x8xf32>, %arg1: memref<?x8xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.560000e+02 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 4 {
      affine.for %arg3 = 0 to 8 {
        %5 = affine.load %arg1[%arg2, %arg3] : memref<?x8xi32>
        %6 = arith.cmpi sge, %5, %c0_i32 : i32
        %7 = scf.if %6 -> (i32) {
          %10 = affine.load %arg1[%arg2, %arg3] : memref<?x8xi32>
          %11 = affine.load %arg0[%arg2, %arg3] : memref<?x8xf32>
          %12 = arith.mulf %11, %cst_0 : f32
          %13 = arith.fptosi %12 : f32 to i32
          %14 = arith.addi %10, %13 : i32
          scf.yield %14 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %8 = affine.load %alloca[] : memref<i32>
        %9 = arith.addi %8, %7 : i32
        affine.store %9, %alloca[] : memref<i32>
      }
    }
    %0 = affine.load %alloca[] : memref<i32>
    %1 = arith.sitofp %0 : i32 to f32
    %2 = call @_ZL5abs_ff(%1) : (f32) -> f32
    %3 = arith.cmpf ogt, %2, %cst : f32
    %4 = arith.select %3, %0, %c1_i32 : i32
    return %4 : i32
  }
  func.func private @_ZL7sq_distPfS_(%arg0: memref<?xf32>, %arg1: memref<?xf32>) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = affine.load %arg0[0] : memref<?xf32>
    %1 = affine.load %arg1[0] : memref<?xf32>
    %2 = arith.subf %0, %1 : f32
    %3 = affine.load %arg0[1] : memref<?xf32>
    %4 = affine.load %arg1[1] : memref<?xf32>
    %5 = arith.subf %3, %4 : f32
    %6 = affine.load %arg0[2] : memref<?xf32>
    %7 = affine.load %arg1[2] : memref<?xf32>
    %8 = arith.subf %6, %7 : f32
    %9 = arith.mulf %2, %2 : f32
    %10 = arith.mulf %5, %5 : f32
    %11 = arith.addf %9, %10 : f32
    %12 = arith.mulf %8, %8 : f32
    %13 = arith.addf %11, %12 : f32
    return %13 : f32
  }
  func.func private @_ZL5abs_ff(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.cmpf olt, %arg0, %cst : f32
    %1 = scf.if %0 -> (f32) {
      %2 = arith.negf %arg0 : f32
      scf.yield %2 : f32
    } else {
      scf.yield %arg0 : f32
    }
    return %1 : f32
  }
}

