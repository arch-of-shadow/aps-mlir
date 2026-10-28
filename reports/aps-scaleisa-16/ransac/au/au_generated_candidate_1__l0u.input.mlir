module {
  func.func @au_generated_candidate_1__l0u(%arg0: memref<32x3xf32>) attributes {au.generated = true} {
    affine.for %arg1 = 0 to 32 {
      %0 = arith.index_cast %arg1 : index to i32
      %c768_i32 = arith.constant 768 : i32
      %1 = arith.cmpi slt, %0, %c768_i32 : i32
      scf.if %1 {
        %c64_i32 = arith.constant 64 : i32
        %2 = arith.remsi %0, %c64_i32 : i32
        %3 = arith.divsi %0, %c64_i32 : i32
        %4 = arith.sitofp %2 : i32 to f32
        %cst = arith.constant 2.000000e-01 : f32
        %5 = arith.mulf %4, %cst : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %6 = arith.divf %5, %cst_0 : f32
        %cst_1 = arith.constant 6.000000e-01 : f32
        %7 = arith.addf %6, %cst_1 : f32
        affine.store %7, %arg0[%arg1, 0] : memref<32x3xf32>
        %8 = arith.sitofp %3 : i32 to f32
        %9 = arith.mulf %8, %cst : f32
        %cst_2 = arith.constant 1.200000e+01 : f32
        %10 = arith.divf %9, %cst_2 : f32
        %cst_3 = arith.constant 4.000000e-01 : f32
        %11 = arith.addf %10, %cst_3 : f32
        affine.store %11, %arg0[%arg1, 1] : memref<32x3xf32>
        %12 = arith.addi %2, %3 : i32
        %c31_i32 = arith.constant 31 : i32
        %13 = arith.remsi %12, %c31_i32 : i32
        %14 = arith.sitofp %13 : i32 to f32
        %15 = arith.mulf %14, %cst : f32
        %cst_4 = arith.constant 3.100000e+01 : f32
        %16 = arith.divf %15, %cst_4 : f32
        %cst_5 = arith.constant 3.000000e-01 : f32
        %17 = arith.addf %16, %cst_5 : f32
        affine.store %17, %arg0[%arg1, 2] : memref<32x3xf32>
      } else {
        %c13_i32 = arith.constant 13 : i32
        %2 = arith.muli %0, %c13_i32 : i32
        %c3_i32 = arith.constant 3 : i32
        %3 = arith.addi %2, %c3_i32 : i32
        %c101_i32 = arith.constant 101 : i32
        %4 = arith.remsi %3, %c101_i32 : i32
        %5 = arith.sitofp %4 : i32 to f32
        %cst = arith.constant 1.010000e+02 : f32
        %6 = arith.divf %5, %cst : f32
        affine.store %6, %arg0[%arg1, 0] : memref<32x3xf32>
        %c17_i32 = arith.constant 17 : i32
        %7 = arith.muli %0, %c17_i32 : i32
        %c5_i32 = arith.constant 5 : i32
        %8 = arith.addi %7, %c5_i32 : i32
        %c97_i32 = arith.constant 97 : i32
        %9 = arith.remsi %8, %c97_i32 : i32
        %10 = arith.sitofp %9 : i32 to f32
        %cst_0 = arith.constant 9.700000e+01 : f32
        %11 = arith.divf %10, %cst_0 : f32
        affine.store %11, %arg0[%arg1, 1] : memref<32x3xf32>
        %c19_i32 = arith.constant 19 : i32
        %12 = arith.muli %0, %c19_i32 : i32
        %c7_i32 = arith.constant 7 : i32
        %13 = arith.addi %12, %c7_i32 : i32
        %c89_i32 = arith.constant 89 : i32
        %14 = arith.remsi %13, %c89_i32 : i32
        %15 = arith.sitofp %14 : i32 to f32
        %cst_1 = arith.constant 8.900000e+01 : f32
        %16 = arith.divf %15, %cst_1 : f32
        affine.store %16, %arg0[%arg1, 2] : memref<32x3xf32>
      }
    } {unroll = 0 : i64}
    return
  }
}

