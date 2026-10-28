module {
  func.func @au_generated_candidate_5__l0p_l1u(%arg0: memref<40xi32>, %arg1: memref<40xi32>, %arg2: memref<24x3xf32>, %arg3: memref<40x3xf32>, %arg4: memref<40x3xf32>, %arg5: memref<24x3xf32>, %arg6: memref<24x3xf32>) attributes {area_square_microns = 5092.9398000000001 : f64, "array-partition" = true, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 167.61168000000001 : f64, cycle = 3921 : i64, datapath_area_square_microns = 4925.3281200000001 : f64, evaluation_status = "ok", fully_unrolled = false, hls.partition_bank_count_0 = 2 : i64, hls.partition_bank_count_1 = 2 : i64, latency_cycles = 3921 : i64, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [0], partition_dim_array_1 = [0], partition_factor_array_0 = [2], partition_factor_array_1 = [2], resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-pose2_slam/target-resource.json", schedule = true} {
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c40 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 40 : index
    scf.for %arg7 = %c0 to %c40 step %c1 {
      %0 = memref.load %arg0[%arg7] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<40xi32>
      %1 = memref.load %arg1[%arg7] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<40xi32>
      %2 = arith.index_cast %1 {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : i32 to index
      %3 = memref.load %arg2[%2, %c0] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<24x3xf32>
      %4 = arith.index_cast %0 {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : i32 to index
      %5 = memref.load %arg2[%4, %c0] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<24x3xf32>
      %6 = arith.subf %3, %5 {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32
      %7 = memref.load %arg3[%arg7, %c0] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<40x3xf32>
      %8 = arith.subf %6, %7 {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : f32
      %9 = memref.load %arg4[%arg7, %c0] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<40x3xf32>
      %10 = arith.mulf %9, %8 {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : f32
      %11 = memref.load %arg5[%4, %c0] {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<24x3xf32>
      %12 = arith.addf %11, %10 {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : f32
      memref.store %12, %arg5[%4, %c0] {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : memref<24x3xf32>
      %13 = memref.load %arg5[%2, %c0] {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<24x3xf32>
      %14 = arith.subf %13, %10 {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : f32
      memref.store %14, %arg5[%2, %c0] {endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : memref<24x3xf32>
      %15 = memref.load %arg4[%arg7, %c0] {endtime = 23 : i64, ref_endtime = 23 : i64, ref_starttime = 22 : i64, starttime = 22 : i64} : memref<40x3xf32>
      %16 = memref.load %arg6[%4, %c0] {endtime = 24 : i64, ref_endtime = 24 : i64, ref_starttime = 23 : i64, starttime = 23 : i64} : memref<24x3xf32>
      %17 = arith.addf %16, %15 {endtime = 27 : i64, ref_endtime = 27 : i64, ref_starttime = 24 : i64, starttime = 24 : i64} : f32
      memref.store %17, %arg6[%4, %c0] {endtime = 28 : i64, ref_endtime = 28 : i64, ref_starttime = 27 : i64, starttime = 27 : i64} : memref<24x3xf32>
      %18 = memref.load %arg4[%arg7, %c0] {endtime = 29 : i64, ref_endtime = 29 : i64, ref_starttime = 28 : i64, starttime = 28 : i64} : memref<40x3xf32>
      %19 = memref.load %arg6[%2, %c0] {endtime = 30 : i64, ref_endtime = 30 : i64, ref_starttime = 29 : i64, starttime = 29 : i64} : memref<24x3xf32>
      %20 = arith.addf %19, %18 {endtime = 33 : i64, ref_endtime = 33 : i64, ref_starttime = 30 : i64, starttime = 30 : i64} : f32
      memref.store %20, %arg6[%2, %c0] {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 33 : i64, starttime = 33 : i64} : memref<24x3xf32>
      %21 = arith.index_cast %1 {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : i32 to index
      %22 = memref.load %arg2[%21, %c1] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<24x3xf32>
      %23 = arith.index_cast %0 {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : i32 to index
      %24 = memref.load %arg2[%23, %c1] {endtime = 36 : i64, ref_endtime = 36 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : memref<24x3xf32>
      %25 = arith.subf %22, %24 {endtime = 39 : i64, ref_endtime = 39 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : f32
      %26 = memref.load %arg3[%arg7, %c1] {endtime = 37 : i64, ref_endtime = 37 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : memref<40x3xf32>
      %27 = arith.subf %25, %26 {endtime = 42 : i64, ref_endtime = 42 : i64, ref_starttime = 39 : i64, starttime = 39 : i64} : f32
      %28 = memref.load %arg4[%arg7, %c1] {endtime = 38 : i64, ref_endtime = 38 : i64, ref_starttime = 37 : i64, starttime = 37 : i64} : memref<40x3xf32>
      %29 = arith.mulf %28, %27 {endtime = 45 : i64, ref_endtime = 45 : i64, ref_starttime = 42 : i64, starttime = 42 : i64} : f32
      %30 = memref.load %arg5[%23, %c1] {endtime = 39 : i64, ref_endtime = 39 : i64, ref_starttime = 38 : i64, starttime = 38 : i64} : memref<24x3xf32>
      %31 = arith.addf %30, %29 {endtime = 48 : i64, ref_endtime = 48 : i64, ref_starttime = 45 : i64, starttime = 45 : i64} : f32
      memref.store %31, %arg5[%23, %c1] {endtime = 49 : i64, ref_endtime = 49 : i64, ref_starttime = 48 : i64, starttime = 48 : i64} : memref<24x3xf32>
      %32 = memref.load %arg5[%21, %c1] {endtime = 50 : i64, ref_endtime = 50 : i64, ref_starttime = 49 : i64, starttime = 49 : i64} : memref<24x3xf32>
      %33 = arith.subf %32, %29 {endtime = 53 : i64, ref_endtime = 53 : i64, ref_starttime = 50 : i64, starttime = 50 : i64} : f32
      memref.store %33, %arg5[%21, %c1] {endtime = 54 : i64, ref_endtime = 54 : i64, ref_starttime = 53 : i64, starttime = 53 : i64} : memref<24x3xf32>
      %34 = memref.load %arg4[%arg7, %c1] {endtime = 55 : i64, ref_endtime = 55 : i64, ref_starttime = 54 : i64, starttime = 54 : i64} : memref<40x3xf32>
      %35 = memref.load %arg6[%23, %c1] {endtime = 56 : i64, ref_endtime = 56 : i64, ref_starttime = 55 : i64, starttime = 55 : i64} : memref<24x3xf32>
      %36 = arith.addf %35, %34 {endtime = 59 : i64, ref_endtime = 59 : i64, ref_starttime = 56 : i64, starttime = 56 : i64} : f32
      memref.store %36, %arg6[%23, %c1] {endtime = 60 : i64, ref_endtime = 60 : i64, ref_starttime = 59 : i64, starttime = 59 : i64} : memref<24x3xf32>
      %37 = memref.load %arg4[%arg7, %c1] {endtime = 61 : i64, ref_endtime = 61 : i64, ref_starttime = 60 : i64, starttime = 60 : i64} : memref<40x3xf32>
      %38 = memref.load %arg6[%21, %c1] {endtime = 62 : i64, ref_endtime = 62 : i64, ref_starttime = 61 : i64, starttime = 61 : i64} : memref<24x3xf32>
      %39 = arith.addf %38, %37 {endtime = 65 : i64, ref_endtime = 65 : i64, ref_starttime = 62 : i64, starttime = 62 : i64} : f32
      memref.store %39, %arg6[%21, %c1] {endtime = 66 : i64, ref_endtime = 66 : i64, ref_starttime = 65 : i64, starttime = 65 : i64} : memref<24x3xf32>
      %40 = arith.index_cast %1 {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : i32 to index
      %41 = memref.load %arg2[%40, %c2] {endtime = 67 : i64, ref_endtime = 67 : i64, ref_starttime = 66 : i64, starttime = 66 : i64} : memref<24x3xf32>
      %42 = arith.index_cast %0 {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : i32 to index
      %43 = memref.load %arg2[%42, %c2] {endtime = 68 : i64, ref_endtime = 68 : i64, ref_starttime = 67 : i64, starttime = 67 : i64} : memref<24x3xf32>
      %44 = arith.subf %41, %43 {endtime = 71 : i64, ref_endtime = 71 : i64, ref_starttime = 68 : i64, starttime = 68 : i64} : f32
      %45 = memref.load %arg3[%arg7, %c2] {endtime = 69 : i64, ref_endtime = 69 : i64, ref_starttime = 68 : i64, starttime = 68 : i64} : memref<40x3xf32>
      %46 = arith.subf %44, %45 {endtime = 74 : i64, ref_endtime = 74 : i64, ref_starttime = 71 : i64, starttime = 71 : i64} : f32
      %47 = memref.load %arg4[%arg7, %c2] {endtime = 70 : i64, ref_endtime = 70 : i64, ref_starttime = 69 : i64, starttime = 69 : i64} : memref<40x3xf32>
      %48 = arith.mulf %47, %46 {endtime = 77 : i64, ref_endtime = 77 : i64, ref_starttime = 74 : i64, starttime = 74 : i64} : f32
      %49 = memref.load %arg5[%42, %c2] {endtime = 71 : i64, ref_endtime = 71 : i64, ref_starttime = 70 : i64, starttime = 70 : i64} : memref<24x3xf32>
      %50 = arith.addf %49, %48 {endtime = 80 : i64, ref_endtime = 80 : i64, ref_starttime = 77 : i64, starttime = 77 : i64} : f32
      memref.store %50, %arg5[%42, %c2] {endtime = 81 : i64, ref_endtime = 81 : i64, ref_starttime = 80 : i64, starttime = 80 : i64} : memref<24x3xf32>
      %51 = memref.load %arg5[%40, %c2] {endtime = 82 : i64, ref_endtime = 82 : i64, ref_starttime = 81 : i64, starttime = 81 : i64} : memref<24x3xf32>
      %52 = arith.subf %51, %48 {endtime = 85 : i64, ref_endtime = 85 : i64, ref_starttime = 82 : i64, starttime = 82 : i64} : f32
      memref.store %52, %arg5[%40, %c2] {endtime = 86 : i64, ref_endtime = 86 : i64, ref_starttime = 85 : i64, starttime = 85 : i64} : memref<24x3xf32>
      %53 = memref.load %arg4[%arg7, %c2] {endtime = 87 : i64, ref_endtime = 87 : i64, ref_starttime = 86 : i64, starttime = 86 : i64} : memref<40x3xf32>
      %54 = memref.load %arg6[%42, %c2] {endtime = 88 : i64, ref_endtime = 88 : i64, ref_starttime = 87 : i64, starttime = 87 : i64} : memref<24x3xf32>
      %55 = arith.addf %54, %53 {endtime = 91 : i64, ref_endtime = 91 : i64, ref_starttime = 88 : i64, starttime = 88 : i64} : f32
      memref.store %55, %arg6[%42, %c2] {endtime = 92 : i64, ref_endtime = 92 : i64, ref_starttime = 91 : i64, starttime = 91 : i64} : memref<24x3xf32>
      %56 = memref.load %arg4[%arg7, %c2] {endtime = 93 : i64, ref_endtime = 93 : i64, ref_starttime = 92 : i64, starttime = 92 : i64} : memref<40x3xf32>
      %57 = memref.load %arg6[%40, %c2] {endtime = 94 : i64, ref_endtime = 94 : i64, ref_starttime = 93 : i64, starttime = 93 : i64} : memref<24x3xf32>
      %58 = arith.addf %57, %56 {endtime = 97 : i64, ref_endtime = 97 : i64, ref_starttime = 94 : i64, starttime = 94 : i64} : f32
      memref.store %58, %arg6[%40, %c2] {endtime = 98 : i64, ref_endtime = 98 : i64, ref_starttime = 97 : i64, starttime = 97 : i64} : memref<24x3xf32>
    } {II = 98 : i64, cycle = 3921 : i64, endtime = 3921 : i64, pipeline = 0 : i64, ref_endtime = 3921 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 40 : i64}
    return
  }
}

