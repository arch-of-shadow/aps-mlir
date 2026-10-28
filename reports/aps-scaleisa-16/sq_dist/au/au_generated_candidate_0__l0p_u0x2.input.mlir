module {
  func.func @au_generated_candidate_0__l0p_u0x2(%arg0: memref<64x6xf32>, %arg1: memref<3xf32>, %arg2: memref<3xf32>, %arg3: memref<64xf32>) attributes {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_3 = [true], partition_dim_array_0 = [1], partition_dim_array_3 = [0], partition_factor_array_0 = [2], partition_factor_array_3 = [2]} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %0 = memref.load %arg0[%arg4, %c0] : memref<64x6xf32>
      %1 = memref.load %arg0[%arg4, %c1] : memref<64x6xf32>
      %c2 = arith.constant 2 : index
      %2 = memref.load %arg0[%arg4, %c2] : memref<64x6xf32>
      %c3 = arith.constant 3 : index
      %3 = memref.load %arg0[%arg4, %c3] : memref<64x6xf32>
      %c4 = arith.constant 4 : index
      %4 = memref.load %arg0[%arg4, %c4] : memref<64x6xf32>
      %c5 = arith.constant 5 : index
      %5 = memref.load %arg0[%arg4, %c5] : memref<64x6xf32>
      memref.store %0, %arg1[%c0] : memref<3xf32>
      memref.store %1, %arg1[%c1] : memref<3xf32>
      memref.store %2, %arg1[%c2] : memref<3xf32>
      memref.store %3, %arg2[%c0] : memref<3xf32>
      memref.store %4, %arg2[%c1] : memref<3xf32>
      memref.store %5, %arg2[%c2] : memref<3xf32>
      %6 = memref.load %arg1[%c0] : memref<3xf32>
      %7 = memref.load %arg2[%c0] : memref<3xf32>
      %8 = arith.subf %6, %7 : f32
      %9 = memref.load %arg1[%c1] : memref<3xf32>
      %10 = memref.load %arg2[%c1] : memref<3xf32>
      %11 = arith.subf %9, %10 : f32
      %12 = memref.load %arg1[%c2] : memref<3xf32>
      %13 = memref.load %arg2[%c2] : memref<3xf32>
      %14 = arith.subf %12, %13 : f32
      %15 = arith.mulf %8, %8 : f32
      %16 = arith.mulf %11, %11 : f32
      %17 = arith.addf %15, %16 : f32
      %18 = arith.mulf %14, %14 : f32
      %19 = arith.addf %17, %18 : f32
      memref.store %19, %arg3[%arg4] : memref<64xf32>
    } {pipeline = 0 : i64, unroll = 2 : i64}
    return
  }
}

