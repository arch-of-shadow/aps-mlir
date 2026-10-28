module {
  memref.global @bench_input_0 : memref<18x6xf32> = uninitialized
  memref.global @bench_input_1 : memref<30xi32> = uninitialized
  memref.global @bench_input_2 : memref<30xi32> = uninitialized
  memref.global @bench_input_3 : memref<30x6xf32> = uninitialized
  memref.global @bench_input_4 : memref<30x6xf32> = uninitialized
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<30x6xf32>
    %alloca_0 = memref.alloca() : memref<30x6xf32>
    %alloca_1 = memref.alloca() : memref<30xi32>
    %alloca_2 = memref.alloca() : memref<30xi32>
    %alloca_3 = memref.alloca() : memref<18x6xf32>
    %cast = memref.cast %alloca_3 : memref<18x6xf32> to memref<?x6xf32>
    %cast_4 = memref.cast %alloca_2 : memref<30xi32> to memref<?xi32>
    %cast_5 = memref.cast %alloca_1 : memref<30xi32> to memref<?xi32>
    %cast_6 = memref.cast %alloca_0 : memref<30x6xf32> to memref<?x6xf32>
    %cast_7 = memref.cast %alloca : memref<30x6xf32> to memref<?x6xf32>
    %initial_0=memref.get_global @bench_input_0:memref<18x6xf32>
affine.for %stage_0_0=0 to 18 {
affine.for %stage_0_1=0 to 6 {
%initial_value_0=affine.load %initial_0[%stage_0_0,%stage_0_1]:memref<18x6xf32>
affine.store %initial_value_0,%alloca_3[%stage_0_0,%stage_0_1]:memref<18x6xf32>
}
}
    %initial_1=memref.get_global @bench_input_1:memref<30xi32>
affine.for %stage_1_0=0 to 30 {
%initial_value_1=affine.load %initial_1[%stage_1_0]:memref<30xi32>
affine.store %initial_value_1,%alloca_2[%stage_1_0]:memref<30xi32>
}
    %initial_2=memref.get_global @bench_input_2:memref<30xi32>
affine.for %stage_2_0=0 to 30 {
%initial_value_2=affine.load %initial_2[%stage_2_0]:memref<30xi32>
affine.store %initial_value_2,%alloca_1[%stage_2_0]:memref<30xi32>
}
    %initial_3=memref.get_global @bench_input_3:memref<30x6xf32>
affine.for %stage_3_0=0 to 30 {
affine.for %stage_3_1=0 to 6 {
%initial_value_3=affine.load %initial_3[%stage_3_0,%stage_3_1]:memref<30x6xf32>
affine.store %initial_value_3,%alloca_0[%stage_3_0,%stage_3_1]:memref<30x6xf32>
}
}
    %initial_4=memref.get_global @bench_input_4:memref<30x6xf32>
affine.for %stage_4_0=0 to 30 {
affine.for %stage_4_1=0 to 6 {
%initial_value_4=affine.load %initial_4[%stage_4_0,%stage_4_1]:memref<30x6xf32>
affine.store %initial_value_4,%alloca[%stage_4_0,%stage_4_1]:memref<30x6xf32>
}
}
    call @_ZL11solve_pose3PA6_fPiS1_S0_S0_(%cast, %cast_4, %cast_5, %cast_6, %cast_7) : (memref<?x6xf32>, memref<?xi32>, memref<?xi32>, memref<?x6xf32>, memref<?x6xf32>) -> ()
    %0 = call @_ZL9summarizePA6_f(%cast) : (memref<?x6xf32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL18init_pose3_problemPA6_fPiS1_S0_S0_(%arg0: memref<?x6xf32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?x6xf32>, %arg4: memref<?x6xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-2_i32 = arith.constant -2 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c13_i32 = arith.constant 13 : i32
    %cst = arith.constant 2.000000e-03 : f32
    %cst_0 = arith.constant 3.000000e-03 : f32
    %cst_1 = arith.constant 6.000000e-02 : f32
    %cst_2 = arith.constant 2.600000e+00 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %cst_4 = arith.constant 0.00899999961 : f32
    %cst_5 = arith.constant 1.100000e-01 : f32
    %cst_6 = arith.constant 1.800000e-01 : f32
    %cst_7 = arith.constant 1.750000e+00 : f32
    %c30_i32 = arith.constant 30 : i32
    %cst_8 = arith.constant 1.800000e+00 : f32
    %cst_9 = arith.constant 1.300000e+00 : f32
    %c6_i32 = arith.constant 6 : i32
    %cst_10 = arith.constant 6.000000e-03 : f32
    %cst_11 = arith.constant 4.000000e-03 : f32
    %cst_12 = arith.constant 5.000000e-03 : f32
    %cst_13 = arith.constant 3.000000e-02 : f32
    %cst_14 = arith.constant 5.000000e-02 : f32
    %cst_15 = arith.constant 1.200000e-02 : f32
    %cst_16 = arith.constant 8.000000e-03 : f32
    %cst_17 = arith.constant 0.00999999977 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_18 = arith.constant 8.000000e-02 : f32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_19 = arith.constant 1.500000e-01 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_20 = arith.constant 4.000000e-02 : f32
    %cst_21 = arith.constant 5.500000e-01 : f32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %alloca_22 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_22[] : memref<i32>
    %alloca_23 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_23[] : memref<i32>
    affine.for %arg5 = 0 to 18 {
      %1 = arith.index_cast %arg5 : index to i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.mulf %2, %cst_21 : f32
      %4 = arith.remsi %1, %c3_i32 : i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.mulf %5, %cst_20 : f32
      %7 = arith.addf %3, %6 : f32
      affine.store %7, %arg0[%arg5, 0] : memref<?x6xf32>
      %8 = arith.remsi %1, %c4_i32 : i32
      %9 = arith.sitofp %8 : i32 to f32
      %10 = arith.mulf %9, %cst_19 : f32
      affine.store %10, %arg0[%arg5, 1] : memref<?x6xf32>
      %11 = arith.remsi %1, %c5_i32 : i32
      %12 = arith.sitofp %11 : i32 to f32
      %13 = arith.mulf %12, %cst_18 : f32
      affine.store %13, %arg0[%arg5, 2] : memref<?x6xf32>
      %14 = arith.addi %11, %c-2_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.mulf %15, %cst_17 : f32
      affine.store %16, %arg0[%arg5, 3] : memref<?x6xf32>
      %17 = arith.addi %8, %c-1_i32 : i32
      %18 = arith.sitofp %17 : i32 to f32
      %19 = arith.mulf %18, %cst_16 : f32
      affine.store %19, %arg0[%arg5, 4] : memref<?x6xf32>
      %20 = arith.addi %4, %c-1_i32 : i32
      %21 = arith.sitofp %20 : i32 to f32
      %22 = arith.mulf %21, %cst_15 : f32
      affine.store %22, %arg0[%arg5, 5] : memref<?x6xf32>
    }
    affine.store %c0_i32, %alloca_23[] : memref<i32>
    affine.for %arg5 = 0 to 17 {
      %1 = arith.index_cast %arg5 : index to i32
      %2 = affine.load %alloca_23[] : memref<i32>
      %3 = arith.index_cast %2 : i32 to index
      memref.store %1, %arg1[%3] : memref<?xi32>
      %4 = arith.addi %1, %c1_i32 : i32
      memref.store %4, %arg2[%3] : memref<?xi32>
      memref.store %cst_21, %arg3[%3, %c0] : memref<?x6xf32>
      %5 = arith.remsi %1, %c2_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.mulf %6, %cst_14 : f32
      memref.store %7, %arg3[%3, %c1] : memref<?x6xf32>
      %8 = arith.remsi %1, %c3_i32 : i32
      %9 = arith.sitofp %8 : i32 to f32
      %10 = arith.mulf %9, %cst_13 : f32
      memref.store %10, %arg3[%3, %c2] : memref<?x6xf32>
      %11 = arith.addi %8, %c-1_i32 : i32
      %12 = arith.sitofp %11 : i32 to f32
      %13 = arith.mulf %12, %cst_12 : f32
      memref.store %13, %arg3[%3, %c3] : memref<?x6xf32>
      %14 = arith.remsi %1, %c4_i32 : i32
      %15 = arith.addi %14, %c-2_i32 : i32
      %16 = arith.sitofp %15 : i32 to f32
      %17 = arith.mulf %16, %cst_11 : f32
      memref.store %17, %arg3[%3, %c4] : memref<?x6xf32>
      %18 = arith.remsi %1, %c5_i32 : i32
      %19 = arith.addi %18, %c-2_i32 : i32
      %20 = arith.sitofp %19 : i32 to f32
      %21 = arith.mulf %20, %cst_10 : f32
      memref.store %21, %arg3[%3, %c5] : memref<?x6xf32>
      affine.for %arg6 = 0 to 6 {
        %23 = arith.index_cast %arg6 : index to i32
        %24 = arith.cmpi slt, %23, %c3_i32 : i32
        %25 = arith.select %24, %cst_9, %cst_8 : f32
        memref.store %25, %arg4[%3, %arg6] : memref<?x6xf32>
      }
      %22 = arith.addi %2, %c1_i32 : i32
      affine.store %22, %alloca_23[] : memref<i32>
    }
    affine.store %c0_i32, %alloca_22[] : memref<i32>
    %0 = scf.while : () -> i32 {
      %1 = affine.load %alloca_23[] : memref<i32>
      %2 = arith.cmpi slt, %1, %c30_i32 : i32
      scf.condition(%2) %1 : i32
    } do {
    ^bb0(%arg5: i32):
      %1 = affine.load %alloca_22[] : memref<i32>
      %2 = arith.muli %1, %c4_i32 : i32
      %3 = arith.addi %2, %c2_i32 : i32
      %4 = arith.remsi %3, %c13_i32 : i32
      %5 = arith.index_cast %arg5 : i32 to index
      memref.store %4, %arg1[%5] : memref<?xi32>
      %6 = arith.addi %4, %c3_i32 : i32
      %7 = arith.remsi %1, %c2_i32 : i32
      %8 = arith.addi %6, %7 : i32
      memref.store %8, %arg2[%5] : memref<?xi32>
      %9 = arith.remsi %1, %c3_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_14 : f32
      %12 = arith.addf %11, %cst_7 : f32
      memref.store %12, %arg3[%5, %c0] : memref<?x6xf32>
      %13 = arith.sitofp %7 : i32 to f32
      %14 = arith.mulf %13, %cst_6 : f32
      memref.store %14, %arg3[%5, %c1] : memref<?x6xf32>
      %15 = arith.remsi %1, %c4_i32 : i32
      %16 = arith.sitofp %15 : i32 to f32
      %17 = arith.mulf %16, %cst_5 : f32
      memref.store %17, %arg3[%5, %c2] : memref<?x6xf32>
      %18 = arith.addi %15, %c-2_i32 : i32
      %19 = arith.sitofp %18 : i32 to f32
      %20 = arith.mulf %19, %cst_17 : f32
      memref.store %20, %arg3[%5, %c3] : memref<?x6xf32>
      %21 = arith.remsi %1, %c5_i32 : i32
      %22 = arith.addi %21, %c-2_i32 : i32
      %23 = arith.sitofp %22 : i32 to f32
      %24 = arith.mulf %23, %cst_4 : f32
      memref.store %24, %arg3[%5, %c4] : memref<?x6xf32>
      %25 = arith.remsi %1, %c6_i32 : i32
      %26 = arith.addi %25, %c-3_i32 : i32
      %27 = arith.sitofp %26 : i32 to f32
      %28 = arith.mulf %27, %cst_15 : f32
      memref.store %28, %arg3[%5, %c5] : memref<?x6xf32>
      affine.store %c0_i32, %alloca[] : memref<i32>
      %29 = scf.while : () -> i32 {
        %32 = affine.load %alloca[] : memref<i32>
        %33 = arith.cmpi slt, %32, %c6_i32 : i32
        scf.condition(%33) %32 : i32
      } do {
      ^bb0(%arg6: i32):
        %32 = arith.index_cast %arg6 : i32 to index
        %33 = arith.cmpi slt, %arg6, %c3_i32 : i32
        %34 = arith.select %33, %cst_3, %cst_2 : f32
        memref.store %34, %arg4[%5, %32] : memref<?x6xf32>
        %35 = arith.addi %arg6, %c1_i32 : i32
        affine.store %35, %alloca[] : memref<i32>
        scf.yield
      }
      %30 = arith.addi %1, %c1_i32 : i32
      affine.store %30, %alloca_22[] : memref<i32>
      %31 = arith.addi %arg5, %c1_i32 : i32
      affine.store %31, %alloca_23[] : memref<i32>
      scf.yield
    }
    affine.for %arg5 = 0 to 18 {
      %1 = arith.index_cast %arg5 : index to i32
      %2 = arith.remsi %1, %c4_i32 : i32
      %3 = arith.addi %2, %c-1_i32 : i32
      %4 = arith.sitofp %3 : i32 to f32
      %5 = arith.mulf %4, %cst_1 : f32
      %6 = affine.load %arg0[%arg5, 0] : memref<?x6xf32>
      %7 = arith.addf %6, %5 : f32
      affine.store %7, %arg0[%arg5, 0] : memref<?x6xf32>
      %8 = arith.remsi %1, %c3_i32 : i32
      %9 = arith.sitofp %8 : i32 to f32
      %10 = arith.mulf %9, %cst_20 : f32
      %11 = affine.load %arg0[%arg5, 1] : memref<?x6xf32>
      %12 = arith.subf %11, %10 : f32
      affine.store %12, %arg0[%arg5, 1] : memref<?x6xf32>
      %13 = arith.remsi %1, %c2_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %15 = arith.mulf %14, %cst_13 : f32
      %16 = affine.load %arg0[%arg5, 2] : memref<?x6xf32>
      %17 = arith.addf %16, %15 : f32
      affine.store %17, %arg0[%arg5, 2] : memref<?x6xf32>
      %18 = arith.mulf %14, %cst_0 : f32
      %19 = affine.load %arg0[%arg5, 3] : memref<?x6xf32>
      %20 = arith.addf %19, %18 : f32
      affine.store %20, %arg0[%arg5, 3] : memref<?x6xf32>
      %21 = arith.mulf %9, %cst : f32
      %22 = affine.load %arg0[%arg5, 4] : memref<?x6xf32>
      %23 = arith.subf %22, %21 : f32
      affine.store %23, %arg0[%arg5, 4] : memref<?x6xf32>
      %24 = arith.sitofp %2 : i32 to f32
      %25 = arith.mulf %24, %cst_11 : f32
      %26 = affine.load %arg0[%arg5, 5] : memref<?x6xf32>
      %27 = arith.addf %26, %25 : f32
      affine.store %27, %arg0[%arg5, 5] : memref<?x6xf32>
    }
    return
  }
  func.func private @_ZL11solve_pose3PA6_fPiS1_S0_S0_(%arg0: memref<?x6xf32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?x6xf32>, %arg4: memref<?x6xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 1.200000e-01 : f32
    %cst_0 = arith.constant 2.000000e-01 : f32
    %c3_i32 = arith.constant 3 : i32
    %cst_1 = arith.constant 1.000000e-03 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<18x6xf32>
    %alloca_3 = memref.alloca() : memref<18x6xf32>
    affine.for %arg5 = 0 to 7 {
      affine.for %arg6 = 0 to 18 {
        affine.for %arg7 = 0 to 6 {
          affine.store %cst_2, %alloca_3[%arg6, %arg7] : memref<18x6xf32>
          affine.store %cst_1, %alloca[%arg6, %arg7] : memref<18x6xf32>
        }
      }
      affine.for %arg6 = 0 to 30 {
        %0 = affine.load %arg1[%arg6] : memref<?xi32>
        %1 = affine.load %arg2[%arg6] : memref<?xi32>
        affine.for %arg7 = 0 to 6 {
          %2 = arith.index_cast %1 : i32 to index
          %3 = memref.load %arg0[%2, %arg7] : memref<?x6xf32>
          %4 = arith.index_cast %0 : i32 to index
          %5 = memref.load %arg0[%4, %arg7] : memref<?x6xf32>
          %6 = arith.subf %3, %5 : f32
          %7 = affine.load %arg3[%arg6, %arg7] : memref<?x6xf32>
          %8 = arith.subf %6, %7 : f32
          %9 = affine.load %arg4[%arg6, %arg7] : memref<?x6xf32>
          %10 = arith.mulf %9, %8 : f32
          %11 = memref.load %alloca_3[%4, %arg7] : memref<18x6xf32>
          %12 = arith.addf %11, %10 : f32
          memref.store %12, %alloca_3[%4, %arg7] : memref<18x6xf32>
          %13 = memref.load %alloca_3[%2, %arg7] : memref<18x6xf32>
          %14 = arith.subf %13, %10 : f32
          memref.store %14, %alloca_3[%2, %arg7] : memref<18x6xf32>
          %15 = affine.load %arg4[%arg6, %arg7] : memref<?x6xf32>
          %16 = memref.load %alloca[%4, %arg7] : memref<18x6xf32>
          %17 = arith.addf %16, %15 : f32
          memref.store %17, %alloca[%4, %arg7] : memref<18x6xf32>
          %18 = affine.load %arg4[%arg6, %arg7] : memref<?x6xf32>
          %19 = memref.load %alloca[%2, %arg7] : memref<18x6xf32>
          %20 = arith.addf %19, %18 : f32
          memref.store %20, %alloca[%2, %arg7] : memref<18x6xf32>
        }
      }
      affine.for %arg6 = 0 to 6 {
        affine.store %cst_2, %arg0[0, %arg6] : memref<?x6xf32>
      }
      affine.for %arg6 = 0 to 17 {
        affine.for %arg7 = 0 to 6 {
          %0 = arith.index_cast %arg7 : index to i32
          %1 = arith.cmpi slt, %0, %c3_i32 : i32
          %2 = arith.select %1, %cst_0, %cst : f32
          %3 = affine.load %alloca_3[%arg6 + 1, %arg7] : memref<18x6xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = affine.load %alloca[%arg6 + 1, %arg7] : memref<18x6xf32>
          %6 = arith.divf %4, %5 : f32
          %7 = affine.load %arg0[%arg6 + 1, %arg7] : memref<?x6xf32>
          %8 = arith.subf %7, %6 : f32
          affine.store %8, %arg0[%arg6 + 1, %arg7] : memref<?x6xf32>
        }
      }
    }
    return
  }
  func.func private @_ZL9summarizePA6_f(%arg0: memref<?x6xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.900000e+01 : f32
    %cst_1 = arith.constant 8.300000e+01 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg1 = 0 to 18 {
      affine.for %arg2 = 0 to 6 {
        %5 = arith.index_cast %arg2 : index to i32
        %6 = affine.load %arg0[%arg1, %arg2] : memref<?x6xf32>
        %7 = arith.sitofp %5 : i32 to f32
        %8 = arith.mulf %7, %cst_0 : f32
        %9 = arith.addf %8, %cst_1 : f32
        %10 = arith.mulf %6, %9 : f32
        %11 = arith.fptosi %10 : f32 to i32
        %12 = affine.load %alloca[] : memref<i32>
        %13 = arith.addi %12, %11 : i32
        affine.store %13, %alloca[] : memref<i32>
      }
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

