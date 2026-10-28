module {
  func.func @au_generated_candidate_21__l0u_l1u(%arg0: memref<8x8xf32>, %arg1: memref<8x8xi32>) attributes {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [1], partition_dim_array_1 = [1], partition_factor_array_0 = [8], partition_factor_array_1 = [8]} {
    affine.for %arg2 = 0 to 8 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 8 {
        %1 = arith.index_cast %arg3 : index to i32
        %c24_i32 = arith.constant 24 : i32
        %2 = arith.cmpi sgt, %1, %c24_i32 : i32
        %3 = arith.extui %2 : i1 to i32
        %c56_i32 = arith.constant 56 : i32
        %4 = arith.cmpi sgt, %1, %c56_i32 : i32
        %5 = arith.extui %4 : i1 to i32
        %6 = arith.addi %3, %5 : i32
        %c30_i32 = arith.constant 30 : i32
        %7 = arith.cmpi sgt, %0, %c30_i32 : i32
        %8 = arith.extui %7 : i1 to i32
        %9 = arith.addi %6, %8 : i32
        %10 = arith.sitofp %9 : i32 to f32
        %cst = arith.constant 1.000000e-01 : f32
        %11 = arith.mulf %10, %cst : f32
        %c7_i32 = arith.constant 7 : i32
        %12 = arith.muli %1, %c7_i32 : i32
        %c5_i32 = arith.constant 5 : i32
        %13 = arith.muli %0, %c5_i32 : i32
        %14 = arith.addi %12, %13 : i32
        %c9_i32 = arith.constant 9 : i32
        %15 = arith.remsi %14, %c9_i32 : i32
        %16 = arith.sitofp %15 : i32 to f32
        %cst_0 = arith.constant 5.000000e-03 : f32
        %17 = arith.mulf %16, %cst_0 : f32
        %18 = arith.addf %11, %17 : f32
        affine.store %18, %arg0[%arg2, %arg3] : memref<8x8xf32>
        %c8_i32 = arith.constant 8 : i32
        %19 = arith.muli %0, %c8_i32 : i32
        %20 = arith.addi %19, %1 : i32
        affine.store %20, %arg1[%arg2, %arg3] : memref<8x8xi32>
      } {unroll = 0 : i64}
    } {unroll = 0 : i64}
    return
  }
}

