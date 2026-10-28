module {
  func.func @au_generated_candidate_44__l0u(%arg0: memref<8x3xf32>) attributes {area_square_microns = 35.750160000000001 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 0.000000e+00 : f64, cycle = 24 : i64, datapath_area_square_microns = 35.750160000000001 : f64, evaluation_status = "ok", fully_unrolled = true, latency_cycles = 24 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-sfm_batch/target-resource.json", schedule = true} {
    %c7 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 7 : index
    %c6 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6 : index
    %c5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 5 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.05000019 : f32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.50000024 : f32
    %cst_1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.950000047 : f32
    %cst_2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.400000095 : f32
    %cst_3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -0.149999857 : f32
    %cst_4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.500000e-01 : f32
    %cst_5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -0.699999928 : f32
    %cst_6 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.200000e-01 : f32
    %cst_7 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -1.250000e+00 : f32
    %cst_8 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.000000e+00 : f32
    %cst_9 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -2.500000e-01 : f32
    %cst_10 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -1.800000e+00 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    memref.store %cst_10, %arg0[%c0, %c0] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<8x3xf32>
    memref.store %cst_9, %arg0[%c0, %c1] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<8x3xf32>
    memref.store %cst_8, %arg0[%c0, %c2] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<8x3xf32>
    memref.store %cst_7, %arg0[%c1, %c0] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<8x3xf32>
    memref.store %cst_8, %arg0[%c1, %c1] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<8x3xf32>
    memref.store %cst_6, %arg0[%c1, %c2] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<8x3xf32>
    memref.store %cst_5, %arg0[%c2, %c0] {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<8x3xf32>
    memref.store %cst_4, %arg0[%c2, %c1] {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<8x3xf32>
    memref.store %cst_8, %arg0[%c2, %c2] {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<8x3xf32>
    memref.store %cst_3, %arg0[%c3, %c0] {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : memref<8x3xf32>
    memref.store %cst_9, %arg0[%c3, %c1] {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : memref<8x3xf32>
    memref.store %cst_6, %arg0[%c3, %c2] {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<8x3xf32>
    memref.store %cst_2, %arg0[%c4, %c0] {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<8x3xf32>
    memref.store %cst_8, %arg0[%c4, %c1] {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : memref<8x3xf32>
    memref.store %cst_8, %arg0[%c4, %c2] {endtime = 15 : i64, ref_endtime = 15 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : memref<8x3xf32>
    memref.store %cst_1, %arg0[%c5, %c0] {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : memref<8x3xf32>
    memref.store %cst_4, %arg0[%c5, %c1] {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : memref<8x3xf32>
    memref.store %cst_6, %arg0[%c5, %c2] {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<8x3xf32>
    memref.store %cst_0, %arg0[%c6, %c0] {endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : memref<8x3xf32>
    memref.store %cst_9, %arg0[%c6, %c1] {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : memref<8x3xf32>
    memref.store %cst_8, %arg0[%c6, %c2] {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : memref<8x3xf32>
    memref.store %cst, %arg0[%c7, %c0] {endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : memref<8x3xf32>
    memref.store %cst_8, %arg0[%c7, %c1] {endtime = 23 : i64, ref_endtime = 23 : i64, ref_starttime = 22 : i64, starttime = 22 : i64} : memref<8x3xf32>
    memref.store %cst_6, %arg0[%c7, %c2] {endtime = 24 : i64, ref_endtime = 24 : i64, ref_starttime = 23 : i64, starttime = 23 : i64} : memref<8x3xf32>
    return
  }
}

