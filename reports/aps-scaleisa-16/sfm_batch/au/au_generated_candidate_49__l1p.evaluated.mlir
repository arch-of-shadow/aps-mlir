module {
  func.func @au_generated_candidate_49__l1p(%arg0: memref<8x3xf32>, %arg1: memref<8x3xf32>, %arg2: memref<8x3xf32>) attributes {area_square_microns = 1154.4444000000001 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 326.35872000000001 : f64, cycle = 849 : i64, datapath_area_square_microns = 828.08568000000002 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 849 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-sfm_batch/target-resource.json", schedule = true} {
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.800000e-01 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c8 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg3 = %c0 to %c8 step %c1 {
      scf.for %arg4 = %c0 to %c3 step %c1 {
        %0 = memref.load %arg0[%arg3, %arg4] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<8x3xf32>
        %1 = arith.mulf %0, %cst {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
        %2 = memref.load %arg1[%arg3, %arg4] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<8x3xf32>
        %3 = arith.divf %1, %2 {endtime = 31 : i64, ref_endtime = 31 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32
        %4 = memref.load %arg2[%arg3, %arg4] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<8x3xf32>
        %5 = arith.subf %4, %3 {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 31 : i64, starttime = 31 : i64} : f32
        memref.store %5, %arg2[%arg3, %arg4] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<8x3xf32>
      } {II = 35 : i64, cycle = 106 : i64, endtime = 106 : i64, pipeline = 0 : i64, ref_endtime = 106 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 3 : i64}
    } {cycle = 849 : i64, endtime = 849 : i64, ref_endtime = 849 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 8 : i64}
    return
  }
}

