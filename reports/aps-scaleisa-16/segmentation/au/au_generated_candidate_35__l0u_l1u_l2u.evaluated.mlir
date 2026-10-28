module {
  func.func @au_generated_candidate_35__l0u_l1u_l2u(%arg0: memref<8x8xi32>, %arg1: memref<i32>) attributes {area_square_microns = 5161.8448799999933 : f64, "array-partition" = true, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 0.000000e+00 : f64, cycle = 192 : i64, datapath_area_square_microns = 5161.8448799999933 : f64, evaluation_status = "ok", fully_unrolled = true, hls.partition_bank_count_0 = 8 : i64, latency_cycles = 192 : i64, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [8], resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-segmentation/target-resource.json", schedule = true} {
    %c7 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 7 : index
    %c6 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6 : index
    %c5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 5 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    %c255_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 255 : i32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %0 = memref.load %arg0[%c0, %c0] {endtime = 1 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 8 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<8x8xi32>
    %1 = arith.andi %0, %c255_i32 {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : i32
    %2 = memref.load %arg1[] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<i32>
    %3 = arith.addi %2, %1 {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : i32
    memref.store %3, %arg1[] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<i32>
    %4 = memref.load %arg0[%c0, %c1] {endtime = 4 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 8 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<8x8xi32>
    %5 = arith.andi %4, %c255_i32 {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : i32
    %6 = memref.load %arg1[] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<i32>
    %7 = arith.addi %6, %5 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : i32
    memref.store %7, %arg1[] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<i32>
    %8 = memref.load %arg0[%c0, %c2] {endtime = 7 : i64, hls.memory_bank = 2 : i64, hls.partition_factor = 8 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<8x8xi32>
    %9 = arith.andi %8, %c255_i32 {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : i32
    %10 = memref.load %arg1[] {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<i32>
    %11 = arith.addi %10, %9 {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : i32
    memref.store %11, %arg1[] {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<i32>
    %12 = memref.load %arg0[%c0, %c3] {endtime = 10 : i64, hls.memory_bank = 3 : i64, hls.partition_factor = 8 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : memref<8x8xi32>
    %13 = arith.andi %12, %c255_i32 {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : i32
    %14 = memref.load %arg1[] {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : memref<i32>
    %15 = arith.addi %14, %13 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : i32
    memref.store %15, %arg1[] {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<i32>
    %16 = memref.load %arg0[%c0, %c4] {endtime = 13 : i64, hls.memory_bank = 4 : i64, hls.partition_factor = 8 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<8x8xi32>
    %17 = arith.andi %16, %c255_i32 {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : i32
    %18 = memref.load %arg1[] {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : memref<i32>
    %19 = arith.addi %18, %17 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : i32
    memref.store %19, %arg1[] {endtime = 15 : i64, ref_endtime = 15 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : memref<i32>
    %20 = memref.load %arg0[%c0, %c5] {endtime = 16 : i64, hls.memory_bank = 5 : i64, hls.partition_factor = 8 : i64, ref_endtime = 16 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : memref<8x8xi32>
    %21 = arith.andi %20, %c255_i32 {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : i32
    %22 = memref.load %arg1[] {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : memref<i32>
    %23 = arith.addi %22, %21 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : i32
    memref.store %23, %arg1[] {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<i32>
    %24 = memref.load %arg0[%c0, %c6] {endtime = 19 : i64, hls.memory_bank = 6 : i64, hls.partition_factor = 8 : i64, ref_endtime = 19 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : memref<8x8xi32>
    %25 = arith.andi %24, %c255_i32 {endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : i32
    %26 = memref.load %arg1[] {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : memref<i32>
    %27 = arith.addi %26, %25 {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : i32
    memref.store %27, %arg1[] {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : memref<i32>
    %28 = memref.load %arg0[%c0, %c7] {endtime = 22 : i64, hls.memory_bank = 7 : i64, hls.partition_factor = 8 : i64, ref_endtime = 22 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : memref<8x8xi32>
    %29 = arith.andi %28, %c255_i32 {endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 22 : i64, starttime = 22 : i64} : i32
    %30 = memref.load %arg1[] {endtime = 23 : i64, ref_endtime = 23 : i64, ref_starttime = 22 : i64, starttime = 22 : i64} : memref<i32>
    %31 = arith.addi %30, %29 {endtime = 23 : i64, ref_endtime = 23 : i64, ref_starttime = 23 : i64, starttime = 23 : i64} : i32
    memref.store %31, %arg1[] {endtime = 24 : i64, ref_endtime = 24 : i64, ref_starttime = 23 : i64, starttime = 23 : i64} : memref<i32>
    %32 = memref.load %arg0[%c1, %c0] {endtime = 25 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 8 : i64, ref_endtime = 25 : i64, ref_starttime = 24 : i64, starttime = 24 : i64} : memref<8x8xi32>
    %33 = arith.andi %32, %c255_i32 {endtime = 25 : i64, ref_endtime = 25 : i64, ref_starttime = 25 : i64, starttime = 25 : i64} : i32
    %34 = memref.load %arg1[] {endtime = 26 : i64, ref_endtime = 26 : i64, ref_starttime = 25 : i64, starttime = 25 : i64} : memref<i32>
    %35 = arith.addi %34, %33 {endtime = 26 : i64, ref_endtime = 26 : i64, ref_starttime = 26 : i64, starttime = 26 : i64} : i32
    memref.store %35, %arg1[] {endtime = 27 : i64, ref_endtime = 27 : i64, ref_starttime = 26 : i64, starttime = 26 : i64} : memref<i32>
    %36 = memref.load %arg0[%c1, %c1] {endtime = 28 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 8 : i64, ref_endtime = 28 : i64, ref_starttime = 27 : i64, starttime = 27 : i64} : memref<8x8xi32>
    %37 = arith.andi %36, %c255_i32 {endtime = 28 : i64, ref_endtime = 28 : i64, ref_starttime = 28 : i64, starttime = 28 : i64} : i32
    %38 = memref.load %arg1[] {endtime = 29 : i64, ref_endtime = 29 : i64, ref_starttime = 28 : i64, starttime = 28 : i64} : memref<i32>
    %39 = arith.addi %38, %37 {endtime = 29 : i64, ref_endtime = 29 : i64, ref_starttime = 29 : i64, starttime = 29 : i64} : i32
    memref.store %39, %arg1[] {endtime = 30 : i64, ref_endtime = 30 : i64, ref_starttime = 29 : i64, starttime = 29 : i64} : memref<i32>
    %40 = memref.load %arg0[%c1, %c2] {endtime = 31 : i64, hls.memory_bank = 2 : i64, hls.partition_factor = 8 : i64, ref_endtime = 31 : i64, ref_starttime = 30 : i64, starttime = 30 : i64} : memref<8x8xi32>
    %41 = arith.andi %40, %c255_i32 {endtime = 31 : i64, ref_endtime = 31 : i64, ref_starttime = 31 : i64, starttime = 31 : i64} : i32
    %42 = memref.load %arg1[] {endtime = 32 : i64, ref_endtime = 32 : i64, ref_starttime = 31 : i64, starttime = 31 : i64} : memref<i32>
    %43 = arith.addi %42, %41 {endtime = 32 : i64, ref_endtime = 32 : i64, ref_starttime = 32 : i64, starttime = 32 : i64} : i32
    memref.store %43, %arg1[] {endtime = 33 : i64, ref_endtime = 33 : i64, ref_starttime = 32 : i64, starttime = 32 : i64} : memref<i32>
    %44 = memref.load %arg0[%c1, %c3] {endtime = 34 : i64, hls.memory_bank = 3 : i64, hls.partition_factor = 8 : i64, ref_endtime = 34 : i64, ref_starttime = 33 : i64, starttime = 33 : i64} : memref<8x8xi32>
    %45 = arith.andi %44, %c255_i32 {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : i32
    %46 = memref.load %arg1[] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<i32>
    %47 = arith.addi %46, %45 {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : i32
    memref.store %47, %arg1[] {endtime = 36 : i64, ref_endtime = 36 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : memref<i32>
    %48 = memref.load %arg0[%c1, %c4] {endtime = 37 : i64, hls.memory_bank = 4 : i64, hls.partition_factor = 8 : i64, ref_endtime = 37 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : memref<8x8xi32>
    %49 = arith.andi %48, %c255_i32 {endtime = 37 : i64, ref_endtime = 37 : i64, ref_starttime = 37 : i64, starttime = 37 : i64} : i32
    %50 = memref.load %arg1[] {endtime = 38 : i64, ref_endtime = 38 : i64, ref_starttime = 37 : i64, starttime = 37 : i64} : memref<i32>
    %51 = arith.addi %50, %49 {endtime = 38 : i64, ref_endtime = 38 : i64, ref_starttime = 38 : i64, starttime = 38 : i64} : i32
    memref.store %51, %arg1[] {endtime = 39 : i64, ref_endtime = 39 : i64, ref_starttime = 38 : i64, starttime = 38 : i64} : memref<i32>
    %52 = memref.load %arg0[%c1, %c5] {endtime = 40 : i64, hls.memory_bank = 5 : i64, hls.partition_factor = 8 : i64, ref_endtime = 40 : i64, ref_starttime = 39 : i64, starttime = 39 : i64} : memref<8x8xi32>
    %53 = arith.andi %52, %c255_i32 {endtime = 40 : i64, ref_endtime = 40 : i64, ref_starttime = 40 : i64, starttime = 40 : i64} : i32
    %54 = memref.load %arg1[] {endtime = 41 : i64, ref_endtime = 41 : i64, ref_starttime = 40 : i64, starttime = 40 : i64} : memref<i32>
    %55 = arith.addi %54, %53 {endtime = 41 : i64, ref_endtime = 41 : i64, ref_starttime = 41 : i64, starttime = 41 : i64} : i32
    memref.store %55, %arg1[] {endtime = 42 : i64, ref_endtime = 42 : i64, ref_starttime = 41 : i64, starttime = 41 : i64} : memref<i32>
    %56 = memref.load %arg0[%c1, %c6] {endtime = 43 : i64, hls.memory_bank = 6 : i64, hls.partition_factor = 8 : i64, ref_endtime = 43 : i64, ref_starttime = 42 : i64, starttime = 42 : i64} : memref<8x8xi32>
    %57 = arith.andi %56, %c255_i32 {endtime = 43 : i64, ref_endtime = 43 : i64, ref_starttime = 43 : i64, starttime = 43 : i64} : i32
    %58 = memref.load %arg1[] {endtime = 44 : i64, ref_endtime = 44 : i64, ref_starttime = 43 : i64, starttime = 43 : i64} : memref<i32>
    %59 = arith.addi %58, %57 {endtime = 44 : i64, ref_endtime = 44 : i64, ref_starttime = 44 : i64, starttime = 44 : i64} : i32
    memref.store %59, %arg1[] {endtime = 45 : i64, ref_endtime = 45 : i64, ref_starttime = 44 : i64, starttime = 44 : i64} : memref<i32>
    %60 = memref.load %arg0[%c1, %c7] {endtime = 46 : i64, hls.memory_bank = 7 : i64, hls.partition_factor = 8 : i64, ref_endtime = 46 : i64, ref_starttime = 45 : i64, starttime = 45 : i64} : memref<8x8xi32>
    %61 = arith.andi %60, %c255_i32 {endtime = 46 : i64, ref_endtime = 46 : i64, ref_starttime = 46 : i64, starttime = 46 : i64} : i32
    %62 = memref.load %arg1[] {endtime = 47 : i64, ref_endtime = 47 : i64, ref_starttime = 46 : i64, starttime = 46 : i64} : memref<i32>
    %63 = arith.addi %62, %61 {endtime = 47 : i64, ref_endtime = 47 : i64, ref_starttime = 47 : i64, starttime = 47 : i64} : i32
    memref.store %63, %arg1[] {endtime = 48 : i64, ref_endtime = 48 : i64, ref_starttime = 47 : i64, starttime = 47 : i64} : memref<i32>
    %64 = memref.load %arg0[%c2, %c0] {endtime = 49 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 8 : i64, ref_endtime = 49 : i64, ref_starttime = 48 : i64, starttime = 48 : i64} : memref<8x8xi32>
    %65 = arith.andi %64, %c255_i32 {endtime = 49 : i64, ref_endtime = 49 : i64, ref_starttime = 49 : i64, starttime = 49 : i64} : i32
    %66 = memref.load %arg1[] {endtime = 50 : i64, ref_endtime = 50 : i64, ref_starttime = 49 : i64, starttime = 49 : i64} : memref<i32>
    %67 = arith.addi %66, %65 {endtime = 50 : i64, ref_endtime = 50 : i64, ref_starttime = 50 : i64, starttime = 50 : i64} : i32
    memref.store %67, %arg1[] {endtime = 51 : i64, ref_endtime = 51 : i64, ref_starttime = 50 : i64, starttime = 50 : i64} : memref<i32>
    %68 = memref.load %arg0[%c2, %c1] {endtime = 52 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 8 : i64, ref_endtime = 52 : i64, ref_starttime = 51 : i64, starttime = 51 : i64} : memref<8x8xi32>
    %69 = arith.andi %68, %c255_i32 {endtime = 52 : i64, ref_endtime = 52 : i64, ref_starttime = 52 : i64, starttime = 52 : i64} : i32
    %70 = memref.load %arg1[] {endtime = 53 : i64, ref_endtime = 53 : i64, ref_starttime = 52 : i64, starttime = 52 : i64} : memref<i32>
    %71 = arith.addi %70, %69 {endtime = 53 : i64, ref_endtime = 53 : i64, ref_starttime = 53 : i64, starttime = 53 : i64} : i32
    memref.store %71, %arg1[] {endtime = 54 : i64, ref_endtime = 54 : i64, ref_starttime = 53 : i64, starttime = 53 : i64} : memref<i32>
    %72 = memref.load %arg0[%c2, %c2] {endtime = 55 : i64, hls.memory_bank = 2 : i64, hls.partition_factor = 8 : i64, ref_endtime = 55 : i64, ref_starttime = 54 : i64, starttime = 54 : i64} : memref<8x8xi32>
    %73 = arith.andi %72, %c255_i32 {endtime = 55 : i64, ref_endtime = 55 : i64, ref_starttime = 55 : i64, starttime = 55 : i64} : i32
    %74 = memref.load %arg1[] {endtime = 56 : i64, ref_endtime = 56 : i64, ref_starttime = 55 : i64, starttime = 55 : i64} : memref<i32>
    %75 = arith.addi %74, %73 {endtime = 56 : i64, ref_endtime = 56 : i64, ref_starttime = 56 : i64, starttime = 56 : i64} : i32
    memref.store %75, %arg1[] {endtime = 57 : i64, ref_endtime = 57 : i64, ref_starttime = 56 : i64, starttime = 56 : i64} : memref<i32>
    %76 = memref.load %arg0[%c2, %c3] {endtime = 58 : i64, hls.memory_bank = 3 : i64, hls.partition_factor = 8 : i64, ref_endtime = 58 : i64, ref_starttime = 57 : i64, starttime = 57 : i64} : memref<8x8xi32>
    %77 = arith.andi %76, %c255_i32 {endtime = 58 : i64, ref_endtime = 58 : i64, ref_starttime = 58 : i64, starttime = 58 : i64} : i32
    %78 = memref.load %arg1[] {endtime = 59 : i64, ref_endtime = 59 : i64, ref_starttime = 58 : i64, starttime = 58 : i64} : memref<i32>
    %79 = arith.addi %78, %77 {endtime = 59 : i64, ref_endtime = 59 : i64, ref_starttime = 59 : i64, starttime = 59 : i64} : i32
    memref.store %79, %arg1[] {endtime = 60 : i64, ref_endtime = 60 : i64, ref_starttime = 59 : i64, starttime = 59 : i64} : memref<i32>
    %80 = memref.load %arg0[%c2, %c4] {endtime = 61 : i64, hls.memory_bank = 4 : i64, hls.partition_factor = 8 : i64, ref_endtime = 61 : i64, ref_starttime = 60 : i64, starttime = 60 : i64} : memref<8x8xi32>
    %81 = arith.andi %80, %c255_i32 {endtime = 61 : i64, ref_endtime = 61 : i64, ref_starttime = 61 : i64, starttime = 61 : i64} : i32
    %82 = memref.load %arg1[] {endtime = 62 : i64, ref_endtime = 62 : i64, ref_starttime = 61 : i64, starttime = 61 : i64} : memref<i32>
    %83 = arith.addi %82, %81 {endtime = 62 : i64, ref_endtime = 62 : i64, ref_starttime = 62 : i64, starttime = 62 : i64} : i32
    memref.store %83, %arg1[] {endtime = 63 : i64, ref_endtime = 63 : i64, ref_starttime = 62 : i64, starttime = 62 : i64} : memref<i32>
    %84 = memref.load %arg0[%c2, %c5] {endtime = 64 : i64, hls.memory_bank = 5 : i64, hls.partition_factor = 8 : i64, ref_endtime = 64 : i64, ref_starttime = 63 : i64, starttime = 63 : i64} : memref<8x8xi32>
    %85 = arith.andi %84, %c255_i32 {endtime = 64 : i64, ref_endtime = 64 : i64, ref_starttime = 64 : i64, starttime = 64 : i64} : i32
    %86 = memref.load %arg1[] {endtime = 65 : i64, ref_endtime = 65 : i64, ref_starttime = 64 : i64, starttime = 64 : i64} : memref<i32>
    %87 = arith.addi %86, %85 {endtime = 65 : i64, ref_endtime = 65 : i64, ref_starttime = 65 : i64, starttime = 65 : i64} : i32
    memref.store %87, %arg1[] {endtime = 66 : i64, ref_endtime = 66 : i64, ref_starttime = 65 : i64, starttime = 65 : i64} : memref<i32>
    %88 = memref.load %arg0[%c2, %c6] {endtime = 67 : i64, hls.memory_bank = 6 : i64, hls.partition_factor = 8 : i64, ref_endtime = 67 : i64, ref_starttime = 66 : i64, starttime = 66 : i64} : memref<8x8xi32>
    %89 = arith.andi %88, %c255_i32 {endtime = 67 : i64, ref_endtime = 67 : i64, ref_starttime = 67 : i64, starttime = 67 : i64} : i32
    %90 = memref.load %arg1[] {endtime = 68 : i64, ref_endtime = 68 : i64, ref_starttime = 67 : i64, starttime = 67 : i64} : memref<i32>
    %91 = arith.addi %90, %89 {endtime = 68 : i64, ref_endtime = 68 : i64, ref_starttime = 68 : i64, starttime = 68 : i64} : i32
    memref.store %91, %arg1[] {endtime = 69 : i64, ref_endtime = 69 : i64, ref_starttime = 68 : i64, starttime = 68 : i64} : memref<i32>
    %92 = memref.load %arg0[%c2, %c7] {endtime = 70 : i64, hls.memory_bank = 7 : i64, hls.partition_factor = 8 : i64, ref_endtime = 70 : i64, ref_starttime = 69 : i64, starttime = 69 : i64} : memref<8x8xi32>
    %93 = arith.andi %92, %c255_i32 {endtime = 70 : i64, ref_endtime = 70 : i64, ref_starttime = 70 : i64, starttime = 70 : i64} : i32
    %94 = memref.load %arg1[] {endtime = 71 : i64, ref_endtime = 71 : i64, ref_starttime = 70 : i64, starttime = 70 : i64} : memref<i32>
    %95 = arith.addi %94, %93 {endtime = 71 : i64, ref_endtime = 71 : i64, ref_starttime = 71 : i64, starttime = 71 : i64} : i32
    memref.store %95, %arg1[] {endtime = 72 : i64, ref_endtime = 72 : i64, ref_starttime = 71 : i64, starttime = 71 : i64} : memref<i32>
    %96 = memref.load %arg0[%c3, %c0] {endtime = 73 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 8 : i64, ref_endtime = 73 : i64, ref_starttime = 72 : i64, starttime = 72 : i64} : memref<8x8xi32>
    %97 = arith.andi %96, %c255_i32 {endtime = 73 : i64, ref_endtime = 73 : i64, ref_starttime = 73 : i64, starttime = 73 : i64} : i32
    %98 = memref.load %arg1[] {endtime = 74 : i64, ref_endtime = 74 : i64, ref_starttime = 73 : i64, starttime = 73 : i64} : memref<i32>
    %99 = arith.addi %98, %97 {endtime = 74 : i64, ref_endtime = 74 : i64, ref_starttime = 74 : i64, starttime = 74 : i64} : i32
    memref.store %99, %arg1[] {endtime = 75 : i64, ref_endtime = 75 : i64, ref_starttime = 74 : i64, starttime = 74 : i64} : memref<i32>
    %100 = memref.load %arg0[%c3, %c1] {endtime = 76 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 8 : i64, ref_endtime = 76 : i64, ref_starttime = 75 : i64, starttime = 75 : i64} : memref<8x8xi32>
    %101 = arith.andi %100, %c255_i32 {endtime = 76 : i64, ref_endtime = 76 : i64, ref_starttime = 76 : i64, starttime = 76 : i64} : i32
    %102 = memref.load %arg1[] {endtime = 77 : i64, ref_endtime = 77 : i64, ref_starttime = 76 : i64, starttime = 76 : i64} : memref<i32>
    %103 = arith.addi %102, %101 {endtime = 77 : i64, ref_endtime = 77 : i64, ref_starttime = 77 : i64, starttime = 77 : i64} : i32
    memref.store %103, %arg1[] {endtime = 78 : i64, ref_endtime = 78 : i64, ref_starttime = 77 : i64, starttime = 77 : i64} : memref<i32>
    %104 = memref.load %arg0[%c3, %c2] {endtime = 79 : i64, hls.memory_bank = 2 : i64, hls.partition_factor = 8 : i64, ref_endtime = 79 : i64, ref_starttime = 78 : i64, starttime = 78 : i64} : memref<8x8xi32>
    %105 = arith.andi %104, %c255_i32 {endtime = 79 : i64, ref_endtime = 79 : i64, ref_starttime = 79 : i64, starttime = 79 : i64} : i32
    %106 = memref.load %arg1[] {endtime = 80 : i64, ref_endtime = 80 : i64, ref_starttime = 79 : i64, starttime = 79 : i64} : memref<i32>
    %107 = arith.addi %106, %105 {endtime = 80 : i64, ref_endtime = 80 : i64, ref_starttime = 80 : i64, starttime = 80 : i64} : i32
    memref.store %107, %arg1[] {endtime = 81 : i64, ref_endtime = 81 : i64, ref_starttime = 80 : i64, starttime = 80 : i64} : memref<i32>
    %108 = memref.load %arg0[%c3, %c3] {endtime = 82 : i64, hls.memory_bank = 3 : i64, hls.partition_factor = 8 : i64, ref_endtime = 82 : i64, ref_starttime = 81 : i64, starttime = 81 : i64} : memref<8x8xi32>
    %109 = arith.andi %108, %c255_i32 {endtime = 82 : i64, ref_endtime = 82 : i64, ref_starttime = 82 : i64, starttime = 82 : i64} : i32
    %110 = memref.load %arg1[] {endtime = 83 : i64, ref_endtime = 83 : i64, ref_starttime = 82 : i64, starttime = 82 : i64} : memref<i32>
    %111 = arith.addi %110, %109 {endtime = 83 : i64, ref_endtime = 83 : i64, ref_starttime = 83 : i64, starttime = 83 : i64} : i32
    memref.store %111, %arg1[] {endtime = 84 : i64, ref_endtime = 84 : i64, ref_starttime = 83 : i64, starttime = 83 : i64} : memref<i32>
    %112 = memref.load %arg0[%c3, %c4] {endtime = 85 : i64, hls.memory_bank = 4 : i64, hls.partition_factor = 8 : i64, ref_endtime = 85 : i64, ref_starttime = 84 : i64, starttime = 84 : i64} : memref<8x8xi32>
    %113 = arith.andi %112, %c255_i32 {endtime = 85 : i64, ref_endtime = 85 : i64, ref_starttime = 85 : i64, starttime = 85 : i64} : i32
    %114 = memref.load %arg1[] {endtime = 86 : i64, ref_endtime = 86 : i64, ref_starttime = 85 : i64, starttime = 85 : i64} : memref<i32>
    %115 = arith.addi %114, %113 {endtime = 86 : i64, ref_endtime = 86 : i64, ref_starttime = 86 : i64, starttime = 86 : i64} : i32
    memref.store %115, %arg1[] {endtime = 87 : i64, ref_endtime = 87 : i64, ref_starttime = 86 : i64, starttime = 86 : i64} : memref<i32>
    %116 = memref.load %arg0[%c3, %c5] {endtime = 88 : i64, hls.memory_bank = 5 : i64, hls.partition_factor = 8 : i64, ref_endtime = 88 : i64, ref_starttime = 87 : i64, starttime = 87 : i64} : memref<8x8xi32>
    %117 = arith.andi %116, %c255_i32 {endtime = 88 : i64, ref_endtime = 88 : i64, ref_starttime = 88 : i64, starttime = 88 : i64} : i32
    %118 = memref.load %arg1[] {endtime = 89 : i64, ref_endtime = 89 : i64, ref_starttime = 88 : i64, starttime = 88 : i64} : memref<i32>
    %119 = arith.addi %118, %117 {endtime = 89 : i64, ref_endtime = 89 : i64, ref_starttime = 89 : i64, starttime = 89 : i64} : i32
    memref.store %119, %arg1[] {endtime = 90 : i64, ref_endtime = 90 : i64, ref_starttime = 89 : i64, starttime = 89 : i64} : memref<i32>
    %120 = memref.load %arg0[%c3, %c6] {endtime = 91 : i64, hls.memory_bank = 6 : i64, hls.partition_factor = 8 : i64, ref_endtime = 91 : i64, ref_starttime = 90 : i64, starttime = 90 : i64} : memref<8x8xi32>
    %121 = arith.andi %120, %c255_i32 {endtime = 91 : i64, ref_endtime = 91 : i64, ref_starttime = 91 : i64, starttime = 91 : i64} : i32
    %122 = memref.load %arg1[] {endtime = 92 : i64, ref_endtime = 92 : i64, ref_starttime = 91 : i64, starttime = 91 : i64} : memref<i32>
    %123 = arith.addi %122, %121 {endtime = 92 : i64, ref_endtime = 92 : i64, ref_starttime = 92 : i64, starttime = 92 : i64} : i32
    memref.store %123, %arg1[] {endtime = 93 : i64, ref_endtime = 93 : i64, ref_starttime = 92 : i64, starttime = 92 : i64} : memref<i32>
    %124 = memref.load %arg0[%c3, %c7] {endtime = 94 : i64, hls.memory_bank = 7 : i64, hls.partition_factor = 8 : i64, ref_endtime = 94 : i64, ref_starttime = 93 : i64, starttime = 93 : i64} : memref<8x8xi32>
    %125 = arith.andi %124, %c255_i32 {endtime = 94 : i64, ref_endtime = 94 : i64, ref_starttime = 94 : i64, starttime = 94 : i64} : i32
    %126 = memref.load %arg1[] {endtime = 95 : i64, ref_endtime = 95 : i64, ref_starttime = 94 : i64, starttime = 94 : i64} : memref<i32>
    %127 = arith.addi %126, %125 {endtime = 95 : i64, ref_endtime = 95 : i64, ref_starttime = 95 : i64, starttime = 95 : i64} : i32
    memref.store %127, %arg1[] {endtime = 96 : i64, ref_endtime = 96 : i64, ref_starttime = 95 : i64, starttime = 95 : i64} : memref<i32>
    %128 = memref.load %arg0[%c4, %c0] {endtime = 97 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 8 : i64, ref_endtime = 97 : i64, ref_starttime = 96 : i64, starttime = 96 : i64} : memref<8x8xi32>
    %129 = arith.andi %128, %c255_i32 {endtime = 97 : i64, ref_endtime = 97 : i64, ref_starttime = 97 : i64, starttime = 97 : i64} : i32
    %130 = memref.load %arg1[] {endtime = 98 : i64, ref_endtime = 98 : i64, ref_starttime = 97 : i64, starttime = 97 : i64} : memref<i32>
    %131 = arith.addi %130, %129 {endtime = 98 : i64, ref_endtime = 98 : i64, ref_starttime = 98 : i64, starttime = 98 : i64} : i32
    memref.store %131, %arg1[] {endtime = 99 : i64, ref_endtime = 99 : i64, ref_starttime = 98 : i64, starttime = 98 : i64} : memref<i32>
    %132 = memref.load %arg0[%c4, %c1] {endtime = 100 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 8 : i64, ref_endtime = 100 : i64, ref_starttime = 99 : i64, starttime = 99 : i64} : memref<8x8xi32>
    %133 = arith.andi %132, %c255_i32 {endtime = 100 : i64, ref_endtime = 100 : i64, ref_starttime = 100 : i64, starttime = 100 : i64} : i32
    %134 = memref.load %arg1[] {endtime = 101 : i64, ref_endtime = 101 : i64, ref_starttime = 100 : i64, starttime = 100 : i64} : memref<i32>
    %135 = arith.addi %134, %133 {endtime = 101 : i64, ref_endtime = 101 : i64, ref_starttime = 101 : i64, starttime = 101 : i64} : i32
    memref.store %135, %arg1[] {endtime = 102 : i64, ref_endtime = 102 : i64, ref_starttime = 101 : i64, starttime = 101 : i64} : memref<i32>
    %136 = memref.load %arg0[%c4, %c2] {endtime = 103 : i64, hls.memory_bank = 2 : i64, hls.partition_factor = 8 : i64, ref_endtime = 103 : i64, ref_starttime = 102 : i64, starttime = 102 : i64} : memref<8x8xi32>
    %137 = arith.andi %136, %c255_i32 {endtime = 103 : i64, ref_endtime = 103 : i64, ref_starttime = 103 : i64, starttime = 103 : i64} : i32
    %138 = memref.load %arg1[] {endtime = 104 : i64, ref_endtime = 104 : i64, ref_starttime = 103 : i64, starttime = 103 : i64} : memref<i32>
    %139 = arith.addi %138, %137 {endtime = 104 : i64, ref_endtime = 104 : i64, ref_starttime = 104 : i64, starttime = 104 : i64} : i32
    memref.store %139, %arg1[] {endtime = 105 : i64, ref_endtime = 105 : i64, ref_starttime = 104 : i64, starttime = 104 : i64} : memref<i32>
    %140 = memref.load %arg0[%c4, %c3] {endtime = 106 : i64, hls.memory_bank = 3 : i64, hls.partition_factor = 8 : i64, ref_endtime = 106 : i64, ref_starttime = 105 : i64, starttime = 105 : i64} : memref<8x8xi32>
    %141 = arith.andi %140, %c255_i32 {endtime = 106 : i64, ref_endtime = 106 : i64, ref_starttime = 106 : i64, starttime = 106 : i64} : i32
    %142 = memref.load %arg1[] {endtime = 107 : i64, ref_endtime = 107 : i64, ref_starttime = 106 : i64, starttime = 106 : i64} : memref<i32>
    %143 = arith.addi %142, %141 {endtime = 107 : i64, ref_endtime = 107 : i64, ref_starttime = 107 : i64, starttime = 107 : i64} : i32
    memref.store %143, %arg1[] {endtime = 108 : i64, ref_endtime = 108 : i64, ref_starttime = 107 : i64, starttime = 107 : i64} : memref<i32>
    %144 = memref.load %arg0[%c4, %c4] {endtime = 109 : i64, hls.memory_bank = 4 : i64, hls.partition_factor = 8 : i64, ref_endtime = 109 : i64, ref_starttime = 108 : i64, starttime = 108 : i64} : memref<8x8xi32>
    %145 = arith.andi %144, %c255_i32 {endtime = 109 : i64, ref_endtime = 109 : i64, ref_starttime = 109 : i64, starttime = 109 : i64} : i32
    %146 = memref.load %arg1[] {endtime = 110 : i64, ref_endtime = 110 : i64, ref_starttime = 109 : i64, starttime = 109 : i64} : memref<i32>
    %147 = arith.addi %146, %145 {endtime = 110 : i64, ref_endtime = 110 : i64, ref_starttime = 110 : i64, starttime = 110 : i64} : i32
    memref.store %147, %arg1[] {endtime = 111 : i64, ref_endtime = 111 : i64, ref_starttime = 110 : i64, starttime = 110 : i64} : memref<i32>
    %148 = memref.load %arg0[%c4, %c5] {endtime = 112 : i64, hls.memory_bank = 5 : i64, hls.partition_factor = 8 : i64, ref_endtime = 112 : i64, ref_starttime = 111 : i64, starttime = 111 : i64} : memref<8x8xi32>
    %149 = arith.andi %148, %c255_i32 {endtime = 112 : i64, ref_endtime = 112 : i64, ref_starttime = 112 : i64, starttime = 112 : i64} : i32
    %150 = memref.load %arg1[] {endtime = 113 : i64, ref_endtime = 113 : i64, ref_starttime = 112 : i64, starttime = 112 : i64} : memref<i32>
    %151 = arith.addi %150, %149 {endtime = 113 : i64, ref_endtime = 113 : i64, ref_starttime = 113 : i64, starttime = 113 : i64} : i32
    memref.store %151, %arg1[] {endtime = 114 : i64, ref_endtime = 114 : i64, ref_starttime = 113 : i64, starttime = 113 : i64} : memref<i32>
    %152 = memref.load %arg0[%c4, %c6] {endtime = 115 : i64, hls.memory_bank = 6 : i64, hls.partition_factor = 8 : i64, ref_endtime = 115 : i64, ref_starttime = 114 : i64, starttime = 114 : i64} : memref<8x8xi32>
    %153 = arith.andi %152, %c255_i32 {endtime = 115 : i64, ref_endtime = 115 : i64, ref_starttime = 115 : i64, starttime = 115 : i64} : i32
    %154 = memref.load %arg1[] {endtime = 116 : i64, ref_endtime = 116 : i64, ref_starttime = 115 : i64, starttime = 115 : i64} : memref<i32>
    %155 = arith.addi %154, %153 {endtime = 116 : i64, ref_endtime = 116 : i64, ref_starttime = 116 : i64, starttime = 116 : i64} : i32
    memref.store %155, %arg1[] {endtime = 117 : i64, ref_endtime = 117 : i64, ref_starttime = 116 : i64, starttime = 116 : i64} : memref<i32>
    %156 = memref.load %arg0[%c4, %c7] {endtime = 118 : i64, hls.memory_bank = 7 : i64, hls.partition_factor = 8 : i64, ref_endtime = 118 : i64, ref_starttime = 117 : i64, starttime = 117 : i64} : memref<8x8xi32>
    %157 = arith.andi %156, %c255_i32 {endtime = 118 : i64, ref_endtime = 118 : i64, ref_starttime = 118 : i64, starttime = 118 : i64} : i32
    %158 = memref.load %arg1[] {endtime = 119 : i64, ref_endtime = 119 : i64, ref_starttime = 118 : i64, starttime = 118 : i64} : memref<i32>
    %159 = arith.addi %158, %157 {endtime = 119 : i64, ref_endtime = 119 : i64, ref_starttime = 119 : i64, starttime = 119 : i64} : i32
    memref.store %159, %arg1[] {endtime = 120 : i64, ref_endtime = 120 : i64, ref_starttime = 119 : i64, starttime = 119 : i64} : memref<i32>
    %160 = memref.load %arg0[%c5, %c0] {endtime = 121 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 8 : i64, ref_endtime = 121 : i64, ref_starttime = 120 : i64, starttime = 120 : i64} : memref<8x8xi32>
    %161 = arith.andi %160, %c255_i32 {endtime = 121 : i64, ref_endtime = 121 : i64, ref_starttime = 121 : i64, starttime = 121 : i64} : i32
    %162 = memref.load %arg1[] {endtime = 122 : i64, ref_endtime = 122 : i64, ref_starttime = 121 : i64, starttime = 121 : i64} : memref<i32>
    %163 = arith.addi %162, %161 {endtime = 122 : i64, ref_endtime = 122 : i64, ref_starttime = 122 : i64, starttime = 122 : i64} : i32
    memref.store %163, %arg1[] {endtime = 123 : i64, ref_endtime = 123 : i64, ref_starttime = 122 : i64, starttime = 122 : i64} : memref<i32>
    %164 = memref.load %arg0[%c5, %c1] {endtime = 124 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 8 : i64, ref_endtime = 124 : i64, ref_starttime = 123 : i64, starttime = 123 : i64} : memref<8x8xi32>
    %165 = arith.andi %164, %c255_i32 {endtime = 124 : i64, ref_endtime = 124 : i64, ref_starttime = 124 : i64, starttime = 124 : i64} : i32
    %166 = memref.load %arg1[] {endtime = 125 : i64, ref_endtime = 125 : i64, ref_starttime = 124 : i64, starttime = 124 : i64} : memref<i32>
    %167 = arith.addi %166, %165 {endtime = 125 : i64, ref_endtime = 125 : i64, ref_starttime = 125 : i64, starttime = 125 : i64} : i32
    memref.store %167, %arg1[] {endtime = 126 : i64, ref_endtime = 126 : i64, ref_starttime = 125 : i64, starttime = 125 : i64} : memref<i32>
    %168 = memref.load %arg0[%c5, %c2] {endtime = 127 : i64, hls.memory_bank = 2 : i64, hls.partition_factor = 8 : i64, ref_endtime = 127 : i64, ref_starttime = 126 : i64, starttime = 126 : i64} : memref<8x8xi32>
    %169 = arith.andi %168, %c255_i32 {endtime = 127 : i64, ref_endtime = 127 : i64, ref_starttime = 127 : i64, starttime = 127 : i64} : i32
    %170 = memref.load %arg1[] {endtime = 128 : i64, ref_endtime = 128 : i64, ref_starttime = 127 : i64, starttime = 127 : i64} : memref<i32>
    %171 = arith.addi %170, %169 {endtime = 128 : i64, ref_endtime = 128 : i64, ref_starttime = 128 : i64, starttime = 128 : i64} : i32
    memref.store %171, %arg1[] {endtime = 129 : i64, ref_endtime = 129 : i64, ref_starttime = 128 : i64, starttime = 128 : i64} : memref<i32>
    %172 = memref.load %arg0[%c5, %c3] {endtime = 130 : i64, hls.memory_bank = 3 : i64, hls.partition_factor = 8 : i64, ref_endtime = 130 : i64, ref_starttime = 129 : i64, starttime = 129 : i64} : memref<8x8xi32>
    %173 = arith.andi %172, %c255_i32 {endtime = 130 : i64, ref_endtime = 130 : i64, ref_starttime = 130 : i64, starttime = 130 : i64} : i32
    %174 = memref.load %arg1[] {endtime = 131 : i64, ref_endtime = 131 : i64, ref_starttime = 130 : i64, starttime = 130 : i64} : memref<i32>
    %175 = arith.addi %174, %173 {endtime = 131 : i64, ref_endtime = 131 : i64, ref_starttime = 131 : i64, starttime = 131 : i64} : i32
    memref.store %175, %arg1[] {endtime = 132 : i64, ref_endtime = 132 : i64, ref_starttime = 131 : i64, starttime = 131 : i64} : memref<i32>
    %176 = memref.load %arg0[%c5, %c4] {endtime = 133 : i64, hls.memory_bank = 4 : i64, hls.partition_factor = 8 : i64, ref_endtime = 133 : i64, ref_starttime = 132 : i64, starttime = 132 : i64} : memref<8x8xi32>
    %177 = arith.andi %176, %c255_i32 {endtime = 133 : i64, ref_endtime = 133 : i64, ref_starttime = 133 : i64, starttime = 133 : i64} : i32
    %178 = memref.load %arg1[] {endtime = 134 : i64, ref_endtime = 134 : i64, ref_starttime = 133 : i64, starttime = 133 : i64} : memref<i32>
    %179 = arith.addi %178, %177 {endtime = 134 : i64, ref_endtime = 134 : i64, ref_starttime = 134 : i64, starttime = 134 : i64} : i32
    memref.store %179, %arg1[] {endtime = 135 : i64, ref_endtime = 135 : i64, ref_starttime = 134 : i64, starttime = 134 : i64} : memref<i32>
    %180 = memref.load %arg0[%c5, %c5] {endtime = 136 : i64, hls.memory_bank = 5 : i64, hls.partition_factor = 8 : i64, ref_endtime = 136 : i64, ref_starttime = 135 : i64, starttime = 135 : i64} : memref<8x8xi32>
    %181 = arith.andi %180, %c255_i32 {endtime = 136 : i64, ref_endtime = 136 : i64, ref_starttime = 136 : i64, starttime = 136 : i64} : i32
    %182 = memref.load %arg1[] {endtime = 137 : i64, ref_endtime = 137 : i64, ref_starttime = 136 : i64, starttime = 136 : i64} : memref<i32>
    %183 = arith.addi %182, %181 {endtime = 137 : i64, ref_endtime = 137 : i64, ref_starttime = 137 : i64, starttime = 137 : i64} : i32
    memref.store %183, %arg1[] {endtime = 138 : i64, ref_endtime = 138 : i64, ref_starttime = 137 : i64, starttime = 137 : i64} : memref<i32>
    %184 = memref.load %arg0[%c5, %c6] {endtime = 139 : i64, hls.memory_bank = 6 : i64, hls.partition_factor = 8 : i64, ref_endtime = 139 : i64, ref_starttime = 138 : i64, starttime = 138 : i64} : memref<8x8xi32>
    %185 = arith.andi %184, %c255_i32 {endtime = 139 : i64, ref_endtime = 139 : i64, ref_starttime = 139 : i64, starttime = 139 : i64} : i32
    %186 = memref.load %arg1[] {endtime = 140 : i64, ref_endtime = 140 : i64, ref_starttime = 139 : i64, starttime = 139 : i64} : memref<i32>
    %187 = arith.addi %186, %185 {endtime = 140 : i64, ref_endtime = 140 : i64, ref_starttime = 140 : i64, starttime = 140 : i64} : i32
    memref.store %187, %arg1[] {endtime = 141 : i64, ref_endtime = 141 : i64, ref_starttime = 140 : i64, starttime = 140 : i64} : memref<i32>
    %188 = memref.load %arg0[%c5, %c7] {endtime = 142 : i64, hls.memory_bank = 7 : i64, hls.partition_factor = 8 : i64, ref_endtime = 142 : i64, ref_starttime = 141 : i64, starttime = 141 : i64} : memref<8x8xi32>
    %189 = arith.andi %188, %c255_i32 {endtime = 142 : i64, ref_endtime = 142 : i64, ref_starttime = 142 : i64, starttime = 142 : i64} : i32
    %190 = memref.load %arg1[] {endtime = 143 : i64, ref_endtime = 143 : i64, ref_starttime = 142 : i64, starttime = 142 : i64} : memref<i32>
    %191 = arith.addi %190, %189 {endtime = 143 : i64, ref_endtime = 143 : i64, ref_starttime = 143 : i64, starttime = 143 : i64} : i32
    memref.store %191, %arg1[] {endtime = 144 : i64, ref_endtime = 144 : i64, ref_starttime = 143 : i64, starttime = 143 : i64} : memref<i32>
    %192 = memref.load %arg0[%c6, %c0] {endtime = 145 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 8 : i64, ref_endtime = 145 : i64, ref_starttime = 144 : i64, starttime = 144 : i64} : memref<8x8xi32>
    %193 = arith.andi %192, %c255_i32 {endtime = 145 : i64, ref_endtime = 145 : i64, ref_starttime = 145 : i64, starttime = 145 : i64} : i32
    %194 = memref.load %arg1[] {endtime = 146 : i64, ref_endtime = 146 : i64, ref_starttime = 145 : i64, starttime = 145 : i64} : memref<i32>
    %195 = arith.addi %194, %193 {endtime = 146 : i64, ref_endtime = 146 : i64, ref_starttime = 146 : i64, starttime = 146 : i64} : i32
    memref.store %195, %arg1[] {endtime = 147 : i64, ref_endtime = 147 : i64, ref_starttime = 146 : i64, starttime = 146 : i64} : memref<i32>
    %196 = memref.load %arg0[%c6, %c1] {endtime = 148 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 8 : i64, ref_endtime = 148 : i64, ref_starttime = 147 : i64, starttime = 147 : i64} : memref<8x8xi32>
    %197 = arith.andi %196, %c255_i32 {endtime = 148 : i64, ref_endtime = 148 : i64, ref_starttime = 148 : i64, starttime = 148 : i64} : i32
    %198 = memref.load %arg1[] {endtime = 149 : i64, ref_endtime = 149 : i64, ref_starttime = 148 : i64, starttime = 148 : i64} : memref<i32>
    %199 = arith.addi %198, %197 {endtime = 149 : i64, ref_endtime = 149 : i64, ref_starttime = 149 : i64, starttime = 149 : i64} : i32
    memref.store %199, %arg1[] {endtime = 150 : i64, ref_endtime = 150 : i64, ref_starttime = 149 : i64, starttime = 149 : i64} : memref<i32>
    %200 = memref.load %arg0[%c6, %c2] {endtime = 151 : i64, hls.memory_bank = 2 : i64, hls.partition_factor = 8 : i64, ref_endtime = 151 : i64, ref_starttime = 150 : i64, starttime = 150 : i64} : memref<8x8xi32>
    %201 = arith.andi %200, %c255_i32 {endtime = 151 : i64, ref_endtime = 151 : i64, ref_starttime = 151 : i64, starttime = 151 : i64} : i32
    %202 = memref.load %arg1[] {endtime = 152 : i64, ref_endtime = 152 : i64, ref_starttime = 151 : i64, starttime = 151 : i64} : memref<i32>
    %203 = arith.addi %202, %201 {endtime = 152 : i64, ref_endtime = 152 : i64, ref_starttime = 152 : i64, starttime = 152 : i64} : i32
    memref.store %203, %arg1[] {endtime = 153 : i64, ref_endtime = 153 : i64, ref_starttime = 152 : i64, starttime = 152 : i64} : memref<i32>
    %204 = memref.load %arg0[%c6, %c3] {endtime = 154 : i64, hls.memory_bank = 3 : i64, hls.partition_factor = 8 : i64, ref_endtime = 154 : i64, ref_starttime = 153 : i64, starttime = 153 : i64} : memref<8x8xi32>
    %205 = arith.andi %204, %c255_i32 {endtime = 154 : i64, ref_endtime = 154 : i64, ref_starttime = 154 : i64, starttime = 154 : i64} : i32
    %206 = memref.load %arg1[] {endtime = 155 : i64, ref_endtime = 155 : i64, ref_starttime = 154 : i64, starttime = 154 : i64} : memref<i32>
    %207 = arith.addi %206, %205 {endtime = 155 : i64, ref_endtime = 155 : i64, ref_starttime = 155 : i64, starttime = 155 : i64} : i32
    memref.store %207, %arg1[] {endtime = 156 : i64, ref_endtime = 156 : i64, ref_starttime = 155 : i64, starttime = 155 : i64} : memref<i32>
    %208 = memref.load %arg0[%c6, %c4] {endtime = 157 : i64, hls.memory_bank = 4 : i64, hls.partition_factor = 8 : i64, ref_endtime = 157 : i64, ref_starttime = 156 : i64, starttime = 156 : i64} : memref<8x8xi32>
    %209 = arith.andi %208, %c255_i32 {endtime = 157 : i64, ref_endtime = 157 : i64, ref_starttime = 157 : i64, starttime = 157 : i64} : i32
    %210 = memref.load %arg1[] {endtime = 158 : i64, ref_endtime = 158 : i64, ref_starttime = 157 : i64, starttime = 157 : i64} : memref<i32>
    %211 = arith.addi %210, %209 {endtime = 158 : i64, ref_endtime = 158 : i64, ref_starttime = 158 : i64, starttime = 158 : i64} : i32
    memref.store %211, %arg1[] {endtime = 159 : i64, ref_endtime = 159 : i64, ref_starttime = 158 : i64, starttime = 158 : i64} : memref<i32>
    %212 = memref.load %arg0[%c6, %c5] {endtime = 160 : i64, hls.memory_bank = 5 : i64, hls.partition_factor = 8 : i64, ref_endtime = 160 : i64, ref_starttime = 159 : i64, starttime = 159 : i64} : memref<8x8xi32>
    %213 = arith.andi %212, %c255_i32 {endtime = 160 : i64, ref_endtime = 160 : i64, ref_starttime = 160 : i64, starttime = 160 : i64} : i32
    %214 = memref.load %arg1[] {endtime = 161 : i64, ref_endtime = 161 : i64, ref_starttime = 160 : i64, starttime = 160 : i64} : memref<i32>
    %215 = arith.addi %214, %213 {endtime = 161 : i64, ref_endtime = 161 : i64, ref_starttime = 161 : i64, starttime = 161 : i64} : i32
    memref.store %215, %arg1[] {endtime = 162 : i64, ref_endtime = 162 : i64, ref_starttime = 161 : i64, starttime = 161 : i64} : memref<i32>
    %216 = memref.load %arg0[%c6, %c6] {endtime = 163 : i64, hls.memory_bank = 6 : i64, hls.partition_factor = 8 : i64, ref_endtime = 163 : i64, ref_starttime = 162 : i64, starttime = 162 : i64} : memref<8x8xi32>
    %217 = arith.andi %216, %c255_i32 {endtime = 163 : i64, ref_endtime = 163 : i64, ref_starttime = 163 : i64, starttime = 163 : i64} : i32
    %218 = memref.load %arg1[] {endtime = 164 : i64, ref_endtime = 164 : i64, ref_starttime = 163 : i64, starttime = 163 : i64} : memref<i32>
    %219 = arith.addi %218, %217 {endtime = 164 : i64, ref_endtime = 164 : i64, ref_starttime = 164 : i64, starttime = 164 : i64} : i32
    memref.store %219, %arg1[] {endtime = 165 : i64, ref_endtime = 165 : i64, ref_starttime = 164 : i64, starttime = 164 : i64} : memref<i32>
    %220 = memref.load %arg0[%c6, %c7] {endtime = 166 : i64, hls.memory_bank = 7 : i64, hls.partition_factor = 8 : i64, ref_endtime = 166 : i64, ref_starttime = 165 : i64, starttime = 165 : i64} : memref<8x8xi32>
    %221 = arith.andi %220, %c255_i32 {endtime = 166 : i64, ref_endtime = 166 : i64, ref_starttime = 166 : i64, starttime = 166 : i64} : i32
    %222 = memref.load %arg1[] {endtime = 167 : i64, ref_endtime = 167 : i64, ref_starttime = 166 : i64, starttime = 166 : i64} : memref<i32>
    %223 = arith.addi %222, %221 {endtime = 167 : i64, ref_endtime = 167 : i64, ref_starttime = 167 : i64, starttime = 167 : i64} : i32
    memref.store %223, %arg1[] {endtime = 168 : i64, ref_endtime = 168 : i64, ref_starttime = 167 : i64, starttime = 167 : i64} : memref<i32>
    %224 = memref.load %arg0[%c7, %c0] {endtime = 169 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 8 : i64, ref_endtime = 169 : i64, ref_starttime = 168 : i64, starttime = 168 : i64} : memref<8x8xi32>
    %225 = arith.andi %224, %c255_i32 {endtime = 169 : i64, ref_endtime = 169 : i64, ref_starttime = 169 : i64, starttime = 169 : i64} : i32
    %226 = memref.load %arg1[] {endtime = 170 : i64, ref_endtime = 170 : i64, ref_starttime = 169 : i64, starttime = 169 : i64} : memref<i32>
    %227 = arith.addi %226, %225 {endtime = 170 : i64, ref_endtime = 170 : i64, ref_starttime = 170 : i64, starttime = 170 : i64} : i32
    memref.store %227, %arg1[] {endtime = 171 : i64, ref_endtime = 171 : i64, ref_starttime = 170 : i64, starttime = 170 : i64} : memref<i32>
    %228 = memref.load %arg0[%c7, %c1] {endtime = 172 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 8 : i64, ref_endtime = 172 : i64, ref_starttime = 171 : i64, starttime = 171 : i64} : memref<8x8xi32>
    %229 = arith.andi %228, %c255_i32 {endtime = 172 : i64, ref_endtime = 172 : i64, ref_starttime = 172 : i64, starttime = 172 : i64} : i32
    %230 = memref.load %arg1[] {endtime = 173 : i64, ref_endtime = 173 : i64, ref_starttime = 172 : i64, starttime = 172 : i64} : memref<i32>
    %231 = arith.addi %230, %229 {endtime = 173 : i64, ref_endtime = 173 : i64, ref_starttime = 173 : i64, starttime = 173 : i64} : i32
    memref.store %231, %arg1[] {endtime = 174 : i64, ref_endtime = 174 : i64, ref_starttime = 173 : i64, starttime = 173 : i64} : memref<i32>
    %232 = memref.load %arg0[%c7, %c2] {endtime = 175 : i64, hls.memory_bank = 2 : i64, hls.partition_factor = 8 : i64, ref_endtime = 175 : i64, ref_starttime = 174 : i64, starttime = 174 : i64} : memref<8x8xi32>
    %233 = arith.andi %232, %c255_i32 {endtime = 175 : i64, ref_endtime = 175 : i64, ref_starttime = 175 : i64, starttime = 175 : i64} : i32
    %234 = memref.load %arg1[] {endtime = 176 : i64, ref_endtime = 176 : i64, ref_starttime = 175 : i64, starttime = 175 : i64} : memref<i32>
    %235 = arith.addi %234, %233 {endtime = 176 : i64, ref_endtime = 176 : i64, ref_starttime = 176 : i64, starttime = 176 : i64} : i32
    memref.store %235, %arg1[] {endtime = 177 : i64, ref_endtime = 177 : i64, ref_starttime = 176 : i64, starttime = 176 : i64} : memref<i32>
    %236 = memref.load %arg0[%c7, %c3] {endtime = 178 : i64, hls.memory_bank = 3 : i64, hls.partition_factor = 8 : i64, ref_endtime = 178 : i64, ref_starttime = 177 : i64, starttime = 177 : i64} : memref<8x8xi32>
    %237 = arith.andi %236, %c255_i32 {endtime = 178 : i64, ref_endtime = 178 : i64, ref_starttime = 178 : i64, starttime = 178 : i64} : i32
    %238 = memref.load %arg1[] {endtime = 179 : i64, ref_endtime = 179 : i64, ref_starttime = 178 : i64, starttime = 178 : i64} : memref<i32>
    %239 = arith.addi %238, %237 {endtime = 179 : i64, ref_endtime = 179 : i64, ref_starttime = 179 : i64, starttime = 179 : i64} : i32
    memref.store %239, %arg1[] {endtime = 180 : i64, ref_endtime = 180 : i64, ref_starttime = 179 : i64, starttime = 179 : i64} : memref<i32>
    %240 = memref.load %arg0[%c7, %c4] {endtime = 181 : i64, hls.memory_bank = 4 : i64, hls.partition_factor = 8 : i64, ref_endtime = 181 : i64, ref_starttime = 180 : i64, starttime = 180 : i64} : memref<8x8xi32>
    %241 = arith.andi %240, %c255_i32 {endtime = 181 : i64, ref_endtime = 181 : i64, ref_starttime = 181 : i64, starttime = 181 : i64} : i32
    %242 = memref.load %arg1[] {endtime = 182 : i64, ref_endtime = 182 : i64, ref_starttime = 181 : i64, starttime = 181 : i64} : memref<i32>
    %243 = arith.addi %242, %241 {endtime = 182 : i64, ref_endtime = 182 : i64, ref_starttime = 182 : i64, starttime = 182 : i64} : i32
    memref.store %243, %arg1[] {endtime = 183 : i64, ref_endtime = 183 : i64, ref_starttime = 182 : i64, starttime = 182 : i64} : memref<i32>
    %244 = memref.load %arg0[%c7, %c5] {endtime = 184 : i64, hls.memory_bank = 5 : i64, hls.partition_factor = 8 : i64, ref_endtime = 184 : i64, ref_starttime = 183 : i64, starttime = 183 : i64} : memref<8x8xi32>
    %245 = arith.andi %244, %c255_i32 {endtime = 184 : i64, ref_endtime = 184 : i64, ref_starttime = 184 : i64, starttime = 184 : i64} : i32
    %246 = memref.load %arg1[] {endtime = 185 : i64, ref_endtime = 185 : i64, ref_starttime = 184 : i64, starttime = 184 : i64} : memref<i32>
    %247 = arith.addi %246, %245 {endtime = 185 : i64, ref_endtime = 185 : i64, ref_starttime = 185 : i64, starttime = 185 : i64} : i32
    memref.store %247, %arg1[] {endtime = 186 : i64, ref_endtime = 186 : i64, ref_starttime = 185 : i64, starttime = 185 : i64} : memref<i32>
    %248 = memref.load %arg0[%c7, %c6] {endtime = 187 : i64, hls.memory_bank = 6 : i64, hls.partition_factor = 8 : i64, ref_endtime = 187 : i64, ref_starttime = 186 : i64, starttime = 186 : i64} : memref<8x8xi32>
    %249 = arith.andi %248, %c255_i32 {endtime = 187 : i64, ref_endtime = 187 : i64, ref_starttime = 187 : i64, starttime = 187 : i64} : i32
    %250 = memref.load %arg1[] {endtime = 188 : i64, ref_endtime = 188 : i64, ref_starttime = 187 : i64, starttime = 187 : i64} : memref<i32>
    %251 = arith.addi %250, %249 {endtime = 188 : i64, ref_endtime = 188 : i64, ref_starttime = 188 : i64, starttime = 188 : i64} : i32
    memref.store %251, %arg1[] {endtime = 189 : i64, ref_endtime = 189 : i64, ref_starttime = 188 : i64, starttime = 188 : i64} : memref<i32>
    %252 = memref.load %arg0[%c7, %c7] {endtime = 190 : i64, hls.memory_bank = 7 : i64, hls.partition_factor = 8 : i64, ref_endtime = 190 : i64, ref_starttime = 189 : i64, starttime = 189 : i64} : memref<8x8xi32>
    %253 = arith.andi %252, %c255_i32 {endtime = 190 : i64, ref_endtime = 190 : i64, ref_starttime = 190 : i64, starttime = 190 : i64} : i32
    %254 = memref.load %arg1[] {endtime = 191 : i64, ref_endtime = 191 : i64, ref_starttime = 190 : i64, starttime = 190 : i64} : memref<i32>
    %255 = arith.addi %254, %253 {endtime = 191 : i64, ref_endtime = 191 : i64, ref_starttime = 191 : i64, starttime = 191 : i64} : i32
    memref.store %255, %arg1[] {endtime = 192 : i64, ref_endtime = 192 : i64, ref_starttime = 191 : i64, starttime = 191 : i64} : memref<i32>
    return
  }
}

