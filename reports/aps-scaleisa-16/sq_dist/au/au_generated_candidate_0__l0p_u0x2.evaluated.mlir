module {
  func.func @au_generated_candidate_0__l0p_u0x2(%arg0: memref<64x6xf32>, %arg1: memref<3xf32>, %arg2: memref<3xf32>, %arg3: memref<64xf32>) attributes {area_square_microns = 4846.1149800000003 : f64, "array-partition" = true, au.generated = true, clock = 1.000000e+01 : f64, control_area_square_microns = 167.61168000000001 : f64, cycle = 1665 : i64, datapath_area_square_microns = 4678.5033000000003 : f64, evaluation_status = "ok", fully_unrolled = false, hls.partition_bank_count_0 = 2 : i64, hls.partition_bank_count_3 = 2 : i64, latency_cycles = 1665 : i64, partition_cyclic_array_0 = [true], partition_cyclic_array_3 = [true], partition_dim_array_0 = [1], partition_dim_array_3 = [0], partition_factor_array_0 = [2], partition_factor_array_3 = [2], resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-scalars-sq_dist/target-resource.json", schedule = true} {
    %c5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 5 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c64 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 64 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    scf.for %arg4 = %c0 to %c64 step %c2 {
      %0 = memref.load %arg0[%arg4, %c0] {endtime = 1 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<64x6xf32>
      %1 = memref.load %arg0[%arg4, %c1] {endtime = 1 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<64x6xf32>
      %2 = memref.load %arg0[%arg4, %c2] {endtime = 2 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<64x6xf32>
      %3 = memref.load %arg0[%arg4, %c3] {endtime = 2 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<64x6xf32>
      %4 = memref.load %arg0[%arg4, %c4] {endtime = 3 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<64x6xf32>
      %5 = memref.load %arg0[%arg4, %c5] {endtime = 3 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<64x6xf32>
      memref.store %0, %arg1[%c0] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<3xf32>
      memref.store %1, %arg1[%c1] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<3xf32>
      memref.store %2, %arg1[%c2] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<3xf32>
      memref.store %3, %arg2[%c0] {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<3xf32>
      memref.store %4, %arg2[%c1] {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<3xf32>
      memref.store %5, %arg2[%c2] {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<3xf32>
      %6 = memref.load %arg1[%c0] {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : memref<3xf32>
      %7 = memref.load %arg2[%c0] {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : memref<3xf32>
      %8 = arith.subf %6, %7 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32
      %9 = memref.load %arg1[%c1] {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<3xf32>
      %10 = memref.load %arg2[%c1] {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<3xf32>
      %11 = arith.subf %9, %10 {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : f32
      %12 = memref.load %arg1[%c2] {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : memref<3xf32>
      %13 = memref.load %arg2[%c2] {endtime = 15 : i64, ref_endtime = 15 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : memref<3xf32>
      %14 = arith.subf %12, %13 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : f32
      %15 = arith.mulf %8, %8 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
      %16 = arith.mulf %11, %11 {endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : f32
      %17 = arith.addf %15, %16 {endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : f32
      %18 = arith.mulf %14, %14 {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : f32
      %19 = arith.addf %17, %18 {endtime = 25 : i64, ref_endtime = 25 : i64, ref_starttime = 22 : i64, starttime = 22 : i64} : f32
      memref.store %19, %arg3[%arg4] {endtime = 26 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 26 : i64, ref_starttime = 25 : i64, starttime = 25 : i64} : memref<64xf32>
      %20 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %21 = memref.load %arg0[%20, %c0] {endtime = 27 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 27 : i64, ref_starttime = 26 : i64, starttime = 26 : i64} : memref<64x6xf32>
      %22 = memref.load %arg0[%20, %c1] {endtime = 27 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 27 : i64, ref_starttime = 26 : i64, starttime = 26 : i64} : memref<64x6xf32>
      %23 = memref.load %arg0[%20, %c2] {endtime = 28 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 28 : i64, ref_starttime = 27 : i64, starttime = 27 : i64} : memref<64x6xf32>
      %24 = memref.load %arg0[%20, %c3] {endtime = 28 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 28 : i64, ref_starttime = 27 : i64, starttime = 27 : i64} : memref<64x6xf32>
      %25 = memref.load %arg0[%20, %c4] {endtime = 29 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 29 : i64, ref_starttime = 28 : i64, starttime = 28 : i64} : memref<64x6xf32>
      %26 = memref.load %arg0[%20, %c5] {endtime = 29 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 29 : i64, ref_starttime = 28 : i64, starttime = 28 : i64} : memref<64x6xf32>
      memref.store %21, %arg1[%c0] {endtime = 30 : i64, ref_endtime = 30 : i64, ref_starttime = 29 : i64, starttime = 29 : i64} : memref<3xf32>
      memref.store %22, %arg1[%c1] {endtime = 31 : i64, ref_endtime = 31 : i64, ref_starttime = 30 : i64, starttime = 30 : i64} : memref<3xf32>
      memref.store %23, %arg1[%c2] {endtime = 32 : i64, ref_endtime = 32 : i64, ref_starttime = 31 : i64, starttime = 31 : i64} : memref<3xf32>
      memref.store %24, %arg2[%c0] {endtime = 33 : i64, ref_endtime = 33 : i64, ref_starttime = 32 : i64, starttime = 32 : i64} : memref<3xf32>
      memref.store %25, %arg2[%c1] {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 33 : i64, starttime = 33 : i64} : memref<3xf32>
      memref.store %26, %arg2[%c2] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<3xf32>
      %27 = memref.load %arg1[%c0] {endtime = 36 : i64, ref_endtime = 36 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : memref<3xf32>
      %28 = memref.load %arg2[%c0] {endtime = 37 : i64, ref_endtime = 37 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : memref<3xf32>
      %29 = arith.subf %27, %28 {endtime = 40 : i64, ref_endtime = 40 : i64, ref_starttime = 37 : i64, starttime = 37 : i64} : f32
      %30 = memref.load %arg1[%c1] {endtime = 38 : i64, ref_endtime = 38 : i64, ref_starttime = 37 : i64, starttime = 37 : i64} : memref<3xf32>
      %31 = memref.load %arg2[%c1] {endtime = 39 : i64, ref_endtime = 39 : i64, ref_starttime = 38 : i64, starttime = 38 : i64} : memref<3xf32>
      %32 = arith.subf %30, %31 {endtime = 42 : i64, ref_endtime = 42 : i64, ref_starttime = 39 : i64, starttime = 39 : i64} : f32
      %33 = memref.load %arg1[%c2] {endtime = 40 : i64, ref_endtime = 40 : i64, ref_starttime = 39 : i64, starttime = 39 : i64} : memref<3xf32>
      %34 = memref.load %arg2[%c2] {endtime = 41 : i64, ref_endtime = 41 : i64, ref_starttime = 40 : i64, starttime = 40 : i64} : memref<3xf32>
      %35 = arith.subf %33, %34 {endtime = 44 : i64, ref_endtime = 44 : i64, ref_starttime = 41 : i64, starttime = 41 : i64} : f32
      %36 = arith.mulf %29, %29 {endtime = 43 : i64, ref_endtime = 43 : i64, ref_starttime = 40 : i64, starttime = 40 : i64} : f32
      %37 = arith.mulf %32, %32 {endtime = 45 : i64, ref_endtime = 45 : i64, ref_starttime = 42 : i64, starttime = 42 : i64} : f32
      %38 = arith.addf %36, %37 {endtime = 48 : i64, ref_endtime = 48 : i64, ref_starttime = 45 : i64, starttime = 45 : i64} : f32
      %39 = arith.mulf %35, %35 {endtime = 47 : i64, ref_endtime = 47 : i64, ref_starttime = 44 : i64, starttime = 44 : i64} : f32
      %40 = arith.addf %38, %39 {endtime = 51 : i64, ref_endtime = 51 : i64, ref_starttime = 48 : i64, starttime = 48 : i64} : f32
      memref.store %40, %arg3[%20] {endtime = 52 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 52 : i64, ref_starttime = 51 : i64, starttime = 51 : i64} : memref<64xf32>
    } {II = 52 : i64, cycle = 1665 : i64, endtime = 1665 : i64, pipeline = 0 : i64, ref_endtime = 1665 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 32 : i64, unroll = 2 : i64}
    return
  }
}

