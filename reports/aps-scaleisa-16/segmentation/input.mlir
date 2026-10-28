module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<8x8xi32>
    %alloca_0 = memref.alloca() : memref<8x8xf32>
    %alloca_1 = memref.alloca() : memref<8x8xf32>
    %cast = memref.cast %alloca_1 : memref<8x8xf32> to memref<?x8xf32>
    %cast_2 = memref.cast %alloca : memref<8x8xi32> to memref<?x8xi32>
    call @_ZL10init_scenePA8_fPA8_i(%cast, %cast_2) : (memref<?x8xf32>, memref<?x8xi32>) -> ()
    %cast_3 = memref.cast %alloca_0 : memref<8x8xf32> to memref<?x8xf32>
    call @_ZL16compute_gradientPA8_fS0_(%cast, %cast_3) : (memref<?x8xf32>, memref<?x8xf32>) -> ()
    call @_ZL11region_growPA8_fS0_PA8_i(%cast, %cast_3, %cast_2) : (memref<?x8xf32>, memref<?x8xf32>, memref<?x8xi32>) -> ()
    %0 = call @_ZL9summarizePA8_i(%cast_2) : (memref<?x8xi32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL10init_scenePA8_fPA8_i(%arg0: memref<?x8xf32>, %arg1: memref<?x8xi32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c9_i32 = arith.constant 9 : i32
    %c5_i32 = arith.constant 5 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant 5.000000e-03 : f32
    %cst_0 = arith.constant 1.000000e-01 : f32
    %c30_i32 = arith.constant 30 : i32
    %c56_i32 = arith.constant 56 : i32
    %c24_i32 = arith.constant 24 : i32
    %c96_i32 = arith.constant 8 : i32
    affine.for %arg2 = 0 to 8 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 8 {
        %1 = arith.index_cast %arg3 : index to i32
        %2 = arith.cmpi sgt, %1, %c24_i32 : i32
        %3 = arith.extui %2 : i1 to i32
        %4 = arith.cmpi sgt, %1, %c56_i32 : i32
        %5 = arith.extui %4 : i1 to i32
        %6 = arith.addi %3, %5 : i32
        %7 = arith.cmpi sgt, %0, %c30_i32 : i32
        %8 = arith.extui %7 : i1 to i32
        %9 = arith.addi %6, %8 : i32
        %10 = arith.sitofp %9 : i32 to f32
        %11 = arith.mulf %10, %cst_0 : f32
        %12 = arith.muli %1, %c7_i32 : i32
        %13 = arith.muli %0, %c5_i32 : i32
        %14 = arith.addi %12, %13 : i32
        %15 = arith.remsi %14, %c9_i32 : i32
        %16 = arith.sitofp %15 : i32 to f32
        %17 = arith.mulf %16, %cst : f32
        %18 = arith.addf %11, %17 : f32
        affine.store %18, %arg0[%arg2, %arg3] : memref<?x8xf32>
        %19 = arith.muli %0, %c96_i32 : i32
        %20 = arith.addi %19, %1 : i32
        affine.store %20, %arg1[%arg2, %arg3] : memref<?x8xi32>
      }
    }
    return
  }
  func.func private @_ZL16compute_gradientPA8_fS0_(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-1_i32 = arith.constant -1 : i32
    %c95_i32 = arith.constant 7 : i32
    %c71_i32 = arith.constant 7 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 8 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 8 {
        %1 = arith.index_cast %arg3 : index to i32
        %2 = arith.addi %1, %c-1_i32 : i32
        %3 = func.call @_ZL7clamp_iiii(%2, %c0_i32, %c95_i32) : (i32, i32, i32) -> i32
        %4 = arith.addi %1, %c1_i32 : i32
        %5 = func.call @_ZL7clamp_iiii(%4, %c0_i32, %c95_i32) : (i32, i32, i32) -> i32
        %6 = arith.addi %0, %c-1_i32 : i32
        %7 = func.call @_ZL7clamp_iiii(%6, %c0_i32, %c71_i32) : (i32, i32, i32) -> i32
        %8 = arith.addi %0, %c1_i32 : i32
        %9 = func.call @_ZL7clamp_iiii(%8, %c0_i32, %c71_i32) : (i32, i32, i32) -> i32
        %10 = arith.index_cast %5 : i32 to index
        %11 = memref.load %arg0[%arg2, %10] : memref<?x8xf32>
        %12 = arith.index_cast %3 : i32 to index
        %13 = memref.load %arg0[%arg2, %12] : memref<?x8xf32>
        %14 = arith.subf %11, %13 : f32
        %15 = func.call @_ZL5abs_ff(%14) : (f32) -> f32
        %16 = arith.index_cast %9 : i32 to index
        %17 = memref.load %arg0[%16, %arg3] : memref<?x8xf32>
        %18 = arith.index_cast %7 : i32 to index
        %19 = memref.load %arg0[%18, %arg3] : memref<?x8xf32>
        %20 = arith.subf %17, %19 : f32
        %21 = func.call @_ZL5abs_ff(%20) : (f32) -> f32
        %22 = arith.addf %15, %21 : f32
        affine.store %22, %arg1[%arg2, %arg3] : memref<?x8xf32>
      }
    }
    return
  }
  func.func private @_ZL11region_growPA8_fS0_PA8_i(%arg0: memref<?x8xf32>, %arg1: memref<?x8xf32>, %arg2: memref<?x8xi32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %c71_i32 = arith.constant 7 : i32
    %c95_i32 = arith.constant 7 : i32
    %false = arith.constant false
    %cst = arith.constant 1.000000e+30 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_1[] : memref<i32>
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 8 {
        %0 = arith.index_cast %arg4 : index to i32
        affine.for %arg5 = 0 to 8 {
          %1 = arith.index_cast %arg5 : index to i32
          %2 = affine.load %arg2[%arg4, %arg5] : memref<?x8xi32>
          affine.store %2, %alloca_1[] : memref<i32>
          affine.store %cst, %alloca[] : memref<f32>
          affine.for %arg6 = 0 to 3 {
            %4 = arith.index_cast %arg6 : index to i32
            %5 = arith.addi %4, %c-1_i32 : i32
            affine.for %arg7 = 0 to 3 {
              %6 = arith.index_cast %arg7 : index to i32
              %7 = arith.addi %6, %c-1_i32 : i32
              %8 = arith.addi %0, %5 : i32
              %9 = func.call @_ZL7clamp_iiii(%8, %c0_i32, %c71_i32) : (i32, i32, i32) -> i32
              %10 = arith.addi %1, %7 : i32
              %11 = func.call @_ZL7clamp_iiii(%10, %c0_i32, %c95_i32) : (i32, i32, i32) -> i32
              %12 = affine.load %arg0[%arg4, %arg5] : memref<?x8xf32>
              %13 = arith.index_cast %9 : i32 to index
              %14 = arith.index_cast %11 : i32 to index
              %15 = memref.load %arg0[%13, %14] : memref<?x8xf32>
              %16 = arith.subf %12, %15 : f32
              %17 = func.call @_ZL5abs_ff(%16) : (f32) -> f32
              %18 = memref.load %arg1[%13, %14] : memref<?x8xf32>
              %19 = arith.addf %17, %18 : f32
              %20 = affine.load %alloca[] : memref<f32>
              %21 = arith.cmpf olt, %19, %20 : f32
              %22 = scf.if %21 -> (i1) {
                scf.yield %true : i1
              } else {
                %23 = arith.cmpf oeq, %19, %20 : f32
                %24 = scf.if %23 -> (i1) {
                  %25 = memref.load %arg2[%13, %14] : memref<?x8xi32>
                  %26 = affine.load %alloca_1[] : memref<i32>
                  %27 = arith.cmpi slt, %25, %26 : i32
                  scf.yield %27 : i1
                } else {
                  scf.yield %false : i1
                }
                scf.yield %24 : i1
              }
              scf.if %22 {
                affine.store %19, %alloca[] : memref<f32>
                %23 = memref.load %arg2[%13, %14] : memref<?x8xi32>
                affine.store %23, %alloca_1[] : memref<i32>
              }
            }
          }
          %3 = affine.load %alloca_1[] : memref<i32>
          affine.store %3, %arg2[%arg4, %arg5] : memref<?x8xi32>
        }
      }
    }
    return
  }
  func.func private @_ZL9summarizePA8_i(%arg0: memref<?x8xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c255_i32 = arith.constant 255 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg1 = 0 to 8 {
      affine.for %arg2 = 0 to 8 {
        %1 = affine.load %arg0[%arg1, %arg2] : memref<?x8xi32>
        %2 = arith.andi %1, %c255_i32 : i32
        %3 = affine.load %alloca[] : memref<i32>
        %4 = arith.addi %3, %2 : i32
        affine.store %4, %alloca[] : memref<i32>
      }
    }
    %0 = affine.load %alloca[] : memref<i32>
    return %0 : i32
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

