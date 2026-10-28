module {
  func.func @au_generated_candidate_31__l1p(%arg0: memref<24x3xf32>, %arg1: memref<24x3xf32>, %arg2: memref<24x3xf32>) attributes {area_square_microns = 1520.28576 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 335.22336000000001 : f64, cycle = 2419 : i64, datapath_area_square_microns = 1185.0624 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 2419 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-pose2_slam/target-resource.json", schedule = true} {
    %c23 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 23 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.800000e-01 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg3 = %c0 to %c3 step %c1 {
      scf.for %arg4 = %c0 to %c23 step %c1 {
        %0 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %1 = memref.load %arg0[%0, %arg3] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<24x3xf32>
        %2 = arith.mulf %1, %cst {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
        %3 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %4 = memref.load %arg1[%3, %arg3] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<24x3xf32>
        %5 = arith.divf %2, %4 {endtime = 31 : i64, ref_endtime = 31 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32
        %6 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %7 = memref.load %arg2[%6, %arg3] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<24x3xf32>
        %8 = arith.subf %7, %5 {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 31 : i64, starttime = 31 : i64} : f32
        %9 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        memref.store %8, %arg2[%9, %arg3] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<24x3xf32>
      } {II = 35 : i64, cycle = 806 : i64, endtime = 806 : i64, pipeline = 0 : i64, ref_endtime = 806 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 23 : i64}
    } {cycle = 2419 : i64, endtime = 2419 : i64, ref_endtime = 2419 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 3 : i64}
    return
  }
}

