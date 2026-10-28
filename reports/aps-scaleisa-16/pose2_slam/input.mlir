module {
  memref.global @bench_input_0 : memref<40xi32> = uninitialized
  memref.global @bench_input_1 : memref<40xi32> = uninitialized
  memref.global @bench_input_2 : memref<40x3xf32> = uninitialized
  memref.global @bench_input_3 : memref<40x3xf32> = uninitialized
  memref.global @bench_input_4 : memref<24x3xf32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<24x3xf32>
    %alloca_0 = memref.alloca() : memref<40x3xf32>
    %alloca_1 = memref.alloca() : memref<40x3xf32>
    %alloca_2 = memref.alloca() : memref<40xi32>
    %alloca_3 = memref.alloca() : memref<40xi32>
    %cast = memref.cast %alloca_3 : memref<40xi32> to memref<?xi32>
    %cast_4 = memref.cast %alloca_2 : memref<40xi32> to memref<?xi32>
    %cast_5 = memref.cast %alloca_1 : memref<40x3xf32> to memref<?x3xf32>
    %cast_6 = memref.cast %alloca_0 : memref<40x3xf32> to memref<?x3xf32>
    %initial_0=memref.get_global @bench_input_0:memref<40xi32>
affine.for %stage_0_0=0 to 40 {
%initial_value_0=affine.load %initial_0[%stage_0_0]:memref<40xi32>
affine.store %initial_value_0,%alloca_3[%stage_0_0]:memref<40xi32>
}
    %initial_1=memref.get_global @bench_input_1:memref<40xi32>
affine.for %stage_1_0=0 to 40 {
%initial_value_1=affine.load %initial_1[%stage_1_0]:memref<40xi32>
affine.store %initial_value_1,%alloca_2[%stage_1_0]:memref<40xi32>
}
    %initial_2=memref.get_global @bench_input_2:memref<40x3xf32>
affine.for %stage_2_0=0 to 40 {
affine.for %stage_2_1=0 to 3 {
%initial_value_2=affine.load %initial_2[%stage_2_0,%stage_2_1]:memref<40x3xf32>
affine.store %initial_value_2,%alloca_1[%stage_2_0,%stage_2_1]:memref<40x3xf32>
}
}
    %initial_3=memref.get_global @bench_input_3:memref<40x3xf32>
affine.for %stage_3_0=0 to 40 {
affine.for %stage_3_1=0 to 3 {
%initial_value_3=affine.load %initial_3[%stage_3_0,%stage_3_1]:memref<40x3xf32>
affine.store %initial_value_3,%alloca_0[%stage_3_0,%stage_3_1]:memref<40x3xf32>
}
}
    %initial_4=memref.get_global @bench_input_4:memref<24x3xf32>
affine.for %stage_4_0=0 to 24 {
affine.for %stage_4_1=0 to 3 {
%initial_value_4=affine.load %initial_4[%stage_4_0,%stage_4_1]:memref<24x3xf32>
affine.store %initial_value_4,%alloca[%stage_4_0,%stage_4_1]:memref<24x3xf32>
}
}
    %cast_7 = memref.cast %alloca : memref<24x3xf32> to memref<?x3xf32>

    call @_ZL16solve_pose_graphPA3_fPiS1_S0_S0_(%cast_7, %cast, %cast_4, %cast_5, %cast_6) : (memref<?x3xf32>, memref<?xi32>, memref<?xi32>, memref<?x3xf32>, memref<?x3xf32>) -> ()
    %0 = call @_ZL9summarizePA3_f(%cast_7) : (memref<?x3xf32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL16build_pose_graphPiS_PA3_fS1_(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?x3xf32>, %arg3: memref<?x3xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-2_i32 = arith.constant -2 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c18_i32 = arith.constant 18 : i32
    %cst = arith.constant 2.200000e+00 : f32
    %cst_0 = arith.constant 1.800000e+00 : f32
    %cst_1 = arith.constant 1.700000e+00 : f32
    %c7_i32 = arith.constant 7 : i32
    %cst_2 = arith.constant 5.000000e-02 : f32
    %cst_3 = arith.constant 7.000000e-02 : f32
    %cst_4 = arith.constant -2.000000e-01 : f32
    %cst_5 = arith.constant 1.500000e-01 : f32
    %cst_6 = arith.constant 4.000000e+00 : f32
    %c4_i32 = arith.constant 4 : i32
    %c40_i32 = arith.constant 40 : i32
    %cst_7 = arith.constant 1.300000e+00 : f32
    %cst_8 = arith.constant 1.100000e+00 : f32
    %cst_9 = arith.constant 1.000000e+00 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_10 = arith.constant 3.000000e-02 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_11 = arith.constant 1.000000e-01 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_12 = arith.constant 2.000000e-02 : f32
    %cst_13 = arith.constant 0.899999976 : f32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %alloca_14 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_14[] : memref<i32>
    affine.store %c0_i32, %alloca_14[] : memref<i32>
    affine.for %arg4 = 0 to 23 {
      %1 = arith.index_cast %arg4 : index to i32
      %2 = affine.load %alloca_14[] : memref<i32>
      %3 = arith.index_cast %2 : i32 to index
      memref.store %1, %arg0[%3] : memref<?xi32>
      %4 = arith.addi %1, %c1_i32 : i32
      memref.store %4, %arg1[%3] : memref<?xi32>
      %5 = arith.remsi %1, %c3_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.mulf %6, %cst_12 : f32
      %8 = arith.addf %7, %cst_13 : f32
      memref.store %8, %arg2[%3, %c0] : memref<?x3xf32>
      %9 = arith.remsi %1, %c2_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_11 : f32
      memref.store %11, %arg2[%3, %c1] : memref<?x3xf32>
      %12 = arith.remsi %1, %c5_i32 : i32
      %13 = arith.addi %12, %c-2_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %15 = arith.mulf %14, %cst_10 : f32
      memref.store %15, %arg2[%3, %c2] : memref<?x3xf32>
      memref.store %cst_9, %arg3[%3, %c0] : memref<?x3xf32>
      memref.store %cst_8, %arg3[%3, %c1] : memref<?x3xf32>
      memref.store %cst_7, %arg3[%3, %c2] : memref<?x3xf32>
      %16 = arith.addi %2, %c1_i32 : i32
      affine.store %16, %alloca_14[] : memref<i32>
    }
    affine.store %c0_i32, %alloca[] : memref<i32>
    %0 = scf.while : () -> i32 {
      %1 = affine.load %alloca_14[] : memref<i32>
      %2 = arith.cmpi slt, %1, %c40_i32 : i32
      scf.condition(%2) %1 : i32
    } do {
    ^bb0(%arg4: i32):
      %1 = affine.load %alloca[] : memref<i32>
      %2 = arith.muli %1, %c5_i32 : i32
      %3 = arith.addi %2, %c3_i32 : i32
      %4 = arith.remsi %3, %c18_i32 : i32
      %5 = arith.index_cast %arg4 : i32 to index
      memref.store %4, %arg0[%5] : memref<?xi32>
      %6 = arith.addi %4, %c4_i32 : i32
      %7 = arith.remsi %1, %c3_i32 : i32
      %8 = arith.addi %6, %7 : i32
      memref.store %8, %arg1[%5] : memref<?xi32>
      %9 = arith.remsi %1, %c4_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_5 : f32
      %12 = arith.addf %11, %cst_6 : f32
      memref.store %12, %arg2[%5, %c0] : memref<?x3xf32>
      %13 = arith.remsi %1, %c5_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %15 = arith.mulf %14, %cst_3 : f32
      %16 = arith.addf %15, %cst_4 : f32
      memref.store %16, %arg2[%5, %c1] : memref<?x3xf32>
      %17 = arith.remsi %1, %c7_i32 : i32
      %18 = arith.addi %17, %c-3_i32 : i32
      %19 = arith.sitofp %18 : i32 to f32
      %20 = arith.mulf %19, %cst_2 : f32
      memref.store %20, %arg2[%5, %c2] : memref<?x3xf32>
      memref.store %cst_1, %arg3[%5, %c0] : memref<?x3xf32>
      memref.store %cst_0, %arg3[%5, %c1] : memref<?x3xf32>
      memref.store %cst, %arg3[%5, %c2] : memref<?x3xf32>
      %21 = arith.addi %1, %c1_i32 : i32
      affine.store %21, %alloca[] : memref<i32>
      %22 = arith.addi %arg4, %c1_i32 : i32
      affine.store %22, %alloca_14[] : memref<i32>
      scf.yield
    }
    return
  }
  func.func private @_ZL10init_statePA3_f(%arg0: memref<?x3xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-3_i32 = arith.constant -3 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst = arith.constant 2.000000e-02 : f32
    %cst_0 = arith.constant 0.00999999977 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_1 = arith.constant 1.800000e-01 : f32
    %c4_i32 = arith.constant 4 : i32
    %cst_2 = arith.constant 4.000000e-02 : f32
    %cst_3 = arith.constant 0.879999995 : f32
    %c24_i32 = arith.constant 24 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %0 = scf.while : () -> i32 {
      %1 = affine.load %alloca[] : memref<i32>
      %2 = arith.cmpi slt, %1, %c24_i32 : i32
      scf.condition(%2) %1 : i32
    } do {
    ^bb0(%arg1: i32):
      %1 = arith.index_cast %arg1 : i32 to index
      %2 = arith.sitofp %arg1 : i32 to f32
      %3 = arith.mulf %2, %cst_3 : f32
      %4 = arith.remsi %arg1, %c4_i32 : i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.mulf %5, %cst_2 : f32
      %7 = arith.addf %3, %6 : f32
      memref.store %7, %arg0[%1, %c0] : memref<?x3xf32>
      %8 = arith.remsi %arg1, %c3_i32 : i32
      %9 = arith.sitofp %8 : i32 to f32
      %10 = arith.mulf %9, %cst_1 : f32
      %11 = arith.mulf %2, %cst_0 : f32
      %12 = arith.subf %10, %11 : f32
      memref.store %12, %arg0[%1, %c1] : memref<?x3xf32>
      %13 = arith.remsi %arg1, %c6_i32 : i32
      %14 = arith.addi %13, %c-3_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.mulf %15, %cst : f32
      memref.store %16, %arg0[%1, %c2] : memref<?x3xf32>
      %17 = arith.addi %arg1, %c1_i32 : i32
      affine.store %17, %alloca[] : memref<i32>
      scf.yield
    }
    return
  }
  func.func private @_ZL16solve_pose_graphPA3_fPiS1_S0_S0_(%arg0: memref<?x3xf32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?x3xf32>, %arg4: memref<?x3xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 2.800000e-01 : f32
    %cst_0 = arith.constant 1.000000e-03 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<24x3xf32>
    %alloca_2 = memref.alloca() : memref<24x3xf32>
    affine.for %arg5 = 0 to 8 {
      affine.for %arg6 = 0 to 24 {
        affine.for %arg7 = 0 to 3 {
          affine.store %cst_1, %alloca_2[%arg6, %arg7] : memref<24x3xf32>
          affine.store %cst_0, %alloca[%arg6, %arg7] : memref<24x3xf32>
        }
      }
      affine.for %arg6 = 0 to 40 {
        %0 = affine.load %arg1[%arg6] : memref<?xi32>
        %1 = affine.load %arg2[%arg6] : memref<?xi32>
        affine.for %arg7 = 0 to 3 {
          %2 = arith.index_cast %1 : i32 to index
          %3 = memref.load %arg0[%2, %arg7] : memref<?x3xf32>
          %4 = arith.index_cast %0 : i32 to index
          %5 = memref.load %arg0[%4, %arg7] : memref<?x3xf32>
          %6 = arith.subf %3, %5 : f32
          %7 = affine.load %arg3[%arg6, %arg7] : memref<?x3xf32>
          %8 = arith.subf %6, %7 : f32
          %9 = affine.load %arg4[%arg6, %arg7] : memref<?x3xf32>
          %10 = arith.mulf %9, %8 : f32
          %11 = memref.load %alloca_2[%4, %arg7] : memref<24x3xf32>
          %12 = arith.addf %11, %10 : f32
          memref.store %12, %alloca_2[%4, %arg7] : memref<24x3xf32>
          %13 = memref.load %alloca_2[%2, %arg7] : memref<24x3xf32>
          %14 = arith.subf %13, %10 : f32
          memref.store %14, %alloca_2[%2, %arg7] : memref<24x3xf32>
          %15 = affine.load %arg4[%arg6, %arg7] : memref<?x3xf32>
          %16 = memref.load %alloca[%4, %arg7] : memref<24x3xf32>
          %17 = arith.addf %16, %15 : f32
          memref.store %17, %alloca[%4, %arg7] : memref<24x3xf32>
          %18 = affine.load %arg4[%arg6, %arg7] : memref<?x3xf32>
          %19 = memref.load %alloca[%2, %arg7] : memref<24x3xf32>
          %20 = arith.addf %19, %18 : f32
          memref.store %20, %alloca[%2, %arg7] : memref<24x3xf32>
        }
      }
      affine.store %cst_1, %arg0[0, 0] : memref<?x3xf32>
      affine.store %cst_1, %arg0[0, 1] : memref<?x3xf32>
      affine.store %cst_1, %arg0[0, 2] : memref<?x3xf32>
      affine.for %arg6 = 0 to 23 {
        affine.for %arg7 = 0 to 3 {
          %0 = affine.load %alloca_2[%arg6 + 1, %arg7] : memref<24x3xf32>
          %1 = arith.mulf %0, %cst : f32
          %2 = affine.load %alloca[%arg6 + 1, %arg7] : memref<24x3xf32>
          %3 = arith.divf %1, %2 : f32
          %4 = affine.load %arg0[%arg6 + 1, %arg7] : memref<?x3xf32>
          %5 = arith.subf %4, %3 : f32
          affine.store %5, %arg0[%arg6 + 1, %arg7] : memref<?x3xf32>
        }
      }
    }
    return
  }
  func.func private @_ZL9summarizePA3_f(%arg0: memref<?x3xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.570000e+02 : f32
    %cst_1 = arith.constant 1.730000e+02 : f32
    %cst_2 = arith.constant 1.010000e+02 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg1 = 0 to 24 {
      %5 = affine.load %arg0[%arg1, 0] : memref<?x3xf32>
      %6 = arith.mulf %5, %cst_2 : f32
      %7 = arith.fptosi %6 : f32 to i32
      %8 = affine.load %alloca[] : memref<i32>
      %9 = arith.addi %8, %7 : i32
      affine.store %9, %alloca[] : memref<i32>
      %10 = affine.load %arg0[%arg1, 1] : memref<?x3xf32>
      %11 = arith.mulf %10, %cst_1 : f32
      %12 = arith.fptosi %11 : f32 to i32
      %13 = arith.addi %9, %12 : i32
      affine.store %13, %alloca[] : memref<i32>
      %14 = affine.load %arg0[%arg1, 2] : memref<?x3xf32>
      %15 = arith.mulf %14, %cst_0 : f32
      %16 = arith.fptosi %15 : f32 to i32
      %17 = arith.addi %13, %16 : i32
      affine.store %17, %alloca[] : memref<i32>
    }
    %0 = affine.load %alloca[] : memref<i32>
    %1 = arith.sitofp %0 : i32 to f32
    %2 = call @_ZL5abs_ff(%1) : (f32) -> f32
    %3 = arith.cmpf ogt, %2, %cst : f32
    %4 = arith.select %3, %0, %c1_i32 : i32
    return %4 : i32
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

