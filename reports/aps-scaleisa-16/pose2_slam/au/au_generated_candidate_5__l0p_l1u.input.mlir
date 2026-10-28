module {
  func.func @au_generated_candidate_5__l0p_l1u(%arg0: memref<40xi32>, %arg1: memref<40xi32>, %arg2: memref<24x3xf32>, %arg3: memref<40x3xf32>, %arg4: memref<40x3xf32>, %arg5: memref<24x3xf32>, %arg6: memref<24x3xf32>) attributes {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [0], partition_dim_array_1 = [0], partition_factor_array_0 = [2], partition_factor_array_1 = [2]} {
    affine.for %arg7 = 0 to 40 {
      %0 = affine.load %arg0[%arg7] : memref<40xi32>
      %1 = affine.load %arg1[%arg7] : memref<40xi32>
      affine.for %arg8 = 0 to 3 {
        %2 = arith.index_cast %1 : i32 to index
        %3 = memref.load %arg2[%2, %arg8] : memref<24x3xf32>
        %4 = arith.index_cast %0 : i32 to index
        %5 = memref.load %arg2[%4, %arg8] : memref<24x3xf32>
        %6 = arith.subf %3, %5 : f32
        %7 = affine.load %arg3[%arg7, %arg8] : memref<40x3xf32>
        %8 = arith.subf %6, %7 : f32
        %9 = affine.load %arg4[%arg7, %arg8] : memref<40x3xf32>
        %10 = arith.mulf %9, %8 : f32
        %11 = memref.load %arg5[%4, %arg8] : memref<24x3xf32>
        %12 = arith.addf %11, %10 : f32
        memref.store %12, %arg5[%4, %arg8] : memref<24x3xf32>
        %13 = memref.load %arg5[%2, %arg8] : memref<24x3xf32>
        %14 = arith.subf %13, %10 : f32
        memref.store %14, %arg5[%2, %arg8] : memref<24x3xf32>
        %15 = affine.load %arg4[%arg7, %arg8] : memref<40x3xf32>
        %16 = memref.load %arg6[%4, %arg8] : memref<24x3xf32>
        %17 = arith.addf %16, %15 : f32
        memref.store %17, %arg6[%4, %arg8] : memref<24x3xf32>
        %18 = affine.load %arg4[%arg7, %arg8] : memref<40x3xf32>
        %19 = memref.load %arg6[%2, %arg8] : memref<24x3xf32>
        %20 = arith.addf %19, %18 : f32
        memref.store %20, %arg6[%2, %arg8] : memref<24x3xf32>
      } {unroll = 0 : i64}
    } {pipeline = 0 : i64}
    return
  }
}

