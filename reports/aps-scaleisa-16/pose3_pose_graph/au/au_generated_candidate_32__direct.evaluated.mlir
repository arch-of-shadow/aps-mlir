module {
  func.func @au_generated_candidate_32__direct(%arg0: memref<18x6xf32>, %arg1: memref<i32>) attributes {area_square_microns = 1557.8875800000003 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 326.35872000000001 : f64, cycle = 1423 : i64, datapath_area_square_microns = 1231.5288600000003 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 1423 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-pose3_pose_graph/target-resource.json", schedule = true} {
    %c6 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8.300000e+01 : f32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.900000e+01 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c18 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 18 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg2 = %c0 to %c18 step %c1 {
      scf.for %arg3 = %c0 to %c6 step %c1 {
        %0 = arith.index_cast %arg3 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %1 = memref.load %arg0[%arg2, %arg3] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<18x6xf32>
        %2 = arith.sitofp %0 {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to f32
        %3 = arith.mulf %2, %cst_0 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : f32
        %4 = arith.addf %3, %cst {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : f32
        %5 = arith.mulf %1, %4 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : f32
        %6 = arith.fptosi %5 {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32 to i32
        %7 = memref.load %arg1[] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<i32>
        %8 = arith.addi %6, %7 {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : i32
        memref.store %8, %arg1[] {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<i32>
      } {cycle = 79 : i64, endtime = 79 : i64, ref_endtime = 79 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 6 : i64}
    } {cycle = 1423 : i64, endtime = 1423 : i64, ref_endtime = 1423 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 18 : i64}
    return
  }
}

