module {
  func.func @au_generated_candidate_56__l0u(%arg0: memref<20x2xf32>) attributes {area_square_microns = 71.500320000000002 : f64, "array-partition" = true, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 0.000000e+00 : f64, cycle = 20 : i64, datapath_area_square_microns = 71.500320000000002 : f64, evaluation_status = "ok", fully_unrolled = true, hls.partition_bank_count_0 = 2 : i64, latency_cycles = 20 : i64, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [2], resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-planar_slam/target-resource.json", schedule = true} {
    %c19 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 19 : index
    %c18 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 18 : index
    %c17 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 17 : index
    %c16 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 16 : index
    %c15 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 15 : index
    %c14 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 14 : index
    %c13 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 13 : index
    %c12 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 12 : index
    %c11 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 11 : index
    %c10 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 10 : index
    %c9 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 9 : index
    %c8 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8 : index
    %c7 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 7 : index
    %c6 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6 : index
    %c5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 5 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.01000023 : f32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.740000e+00 : f32
    %cst_1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.470000e+00 : f32
    %cst_2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3.120000e+00 : f32
    %cst_3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.740000e+00 : f32
    %cst_4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.36000013 : f32
    %cst_5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.980000e+00 : f32
    %cst_6 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.200000e+00 : f32
    %cst_7 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.600000e+00 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    memref.store %cst_7, %arg0[%c0, %c0] {endtime = 1 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<20x2xf32>
    memref.store %cst_6, %arg0[%c0, %c1] {endtime = 1 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<20x2xf32>
    memref.store %cst_5, %arg0[%c1, %c0] {endtime = 2 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<20x2xf32>
    memref.store %cst_6, %arg0[%c1, %c1] {endtime = 2 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<20x2xf32>
    memref.store %cst_4, %arg0[%c2, %c0] {endtime = 3 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<20x2xf32>
    memref.store %cst_6, %arg0[%c2, %c1] {endtime = 3 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<20x2xf32>
    memref.store %cst_3, %arg0[%c3, %c0] {endtime = 4 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<20x2xf32>
    memref.store %cst_6, %arg0[%c3, %c1] {endtime = 4 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<20x2xf32>
    memref.store %cst_2, %arg0[%c4, %c0] {endtime = 5 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<20x2xf32>
    memref.store %cst_6, %arg0[%c4, %c1] {endtime = 5 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<20x2xf32>
    memref.store %cst_7, %arg0[%c5, %c0] {endtime = 6 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<20x2xf32>
    memref.store %cst_1, %arg0[%c5, %c1] {endtime = 6 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<20x2xf32>
    memref.store %cst_5, %arg0[%c6, %c0] {endtime = 7 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<20x2xf32>
    memref.store %cst_1, %arg0[%c6, %c1] {endtime = 7 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<20x2xf32>
    memref.store %cst_4, %arg0[%c7, %c0] {endtime = 8 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<20x2xf32>
    memref.store %cst_1, %arg0[%c7, %c1] {endtime = 8 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<20x2xf32>
    memref.store %cst_3, %arg0[%c8, %c0] {endtime = 9 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<20x2xf32>
    memref.store %cst_1, %arg0[%c8, %c1] {endtime = 9 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<20x2xf32>
    memref.store %cst_2, %arg0[%c9, %c0] {endtime = 10 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : memref<20x2xf32>
    memref.store %cst_1, %arg0[%c9, %c1] {endtime = 10 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : memref<20x2xf32>
    memref.store %cst_7, %arg0[%c10, %c0] {endtime = 11 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : memref<20x2xf32>
    memref.store %cst_0, %arg0[%c10, %c1] {endtime = 11 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : memref<20x2xf32>
    memref.store %cst_5, %arg0[%c11, %c0] {endtime = 12 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<20x2xf32>
    memref.store %cst_0, %arg0[%c11, %c1] {endtime = 12 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<20x2xf32>
    memref.store %cst_4, %arg0[%c12, %c0] {endtime = 13 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<20x2xf32>
    memref.store %cst_0, %arg0[%c12, %c1] {endtime = 13 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<20x2xf32>
    memref.store %cst_3, %arg0[%c13, %c0] {endtime = 14 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 14 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : memref<20x2xf32>
    memref.store %cst_0, %arg0[%c13, %c1] {endtime = 14 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 14 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : memref<20x2xf32>
    memref.store %cst_2, %arg0[%c14, %c0] {endtime = 15 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 15 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : memref<20x2xf32>
    memref.store %cst_0, %arg0[%c14, %c1] {endtime = 15 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 15 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : memref<20x2xf32>
    memref.store %cst_7, %arg0[%c15, %c0] {endtime = 16 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 16 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : memref<20x2xf32>
    memref.store %cst, %arg0[%c15, %c1] {endtime = 16 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 16 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : memref<20x2xf32>
    memref.store %cst_5, %arg0[%c16, %c0] {endtime = 17 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 17 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : memref<20x2xf32>
    memref.store %cst, %arg0[%c16, %c1] {endtime = 17 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 17 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : memref<20x2xf32>
    memref.store %cst_4, %arg0[%c17, %c0] {endtime = 18 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<20x2xf32>
    memref.store %cst, %arg0[%c17, %c1] {endtime = 18 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<20x2xf32>
    memref.store %cst_3, %arg0[%c18, %c0] {endtime = 19 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 19 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : memref<20x2xf32>
    memref.store %cst, %arg0[%c18, %c1] {endtime = 19 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 19 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : memref<20x2xf32>
    memref.store %cst_2, %arg0[%c19, %c0] {endtime = 20 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 2 : i64, ref_endtime = 20 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : memref<20x2xf32>
    memref.store %cst, %arg0[%c19, %c1] {endtime = 20 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 2 : i64, ref_endtime = 20 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : memref<20x2xf32>
    return
  }
}

