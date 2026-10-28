module {
  func.func @au_generated_candidate_22__direct(%arg0: memref<24x3xf32>, %arg1: memref<i32>) attributes {area_square_microns = 1790.0740800000003 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 167.61168000000001 : f64, cycle = 433 : i64, datapath_area_square_microns = 1622.4624000000003 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 433 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-sfm_batch/target-resource.json", schedule = true} {
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.270000e+02 : f32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.090000e+02 : f32
    %cst_1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8.300000e+01 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c24 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 24 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg2 = %c0 to %c24 step %c1 {
      %0 = memref.load %arg0[%arg2, %c0] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<24x3xf32>
      %1 = arith.mulf %0, %cst_1 {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
      %2 = arith.fptosi %1 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32 to i32
      %3 = memref.load %arg1[] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<i32>
      %4 = arith.addi %3, %2 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : i32
      memref.store %4, %arg1[] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<i32>
      %5 = memref.load %arg0[%arg2, %c1] {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<24x3xf32>
      %6 = arith.mulf %5, %cst_0 {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : f32
      %7 = arith.fptosi %6 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : f32 to i32
      %8 = arith.addi %4, %7 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : i32
      memref.store %8, %arg1[] {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<i32>
      %9 = memref.load %arg0[%arg2, %c2] {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<24x3xf32>
      %10 = arith.mulf %9, %cst {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : f32
      %11 = arith.fptosi %10 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : f32 to i32
      %12 = arith.addi %8, %11 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : i32
      memref.store %12, %arg1[] {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<i32>
    } {cycle = 433 : i64, endtime = 433 : i64, ref_endtime = 433 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 24 : i64}
    return
  }
}

