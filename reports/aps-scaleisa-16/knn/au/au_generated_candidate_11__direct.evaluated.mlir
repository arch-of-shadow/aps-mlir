module {
  func.func @au_generated_candidate_11__direct(%arg0: memref<4x8xi32>, %arg1: memref<4x8xf32>, %arg2: memref<i32>) attributes {area_square_microns = 967.32467999999994 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 326.35872000000001 : f64, cycle = 325 : i64, datapath_area_square_microns = 640.96596 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 325 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-knn-32/target-resource.json", schedule = true} {
    %c8 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.560000e+02 : f32
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg3 = %c0 to %c4 step %c1 {
      scf.for %arg4 = %c0 to %c8 step %c1 {
        %0 = memref.load %arg0[%arg3, %arg4] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x8xi32>
        %1 = arith.cmpi sge, %0, %c0_i32 {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : i32
        %2 = scf.if %1 -> (i32) {
          %5 = memref.load %arg0[%arg3, %arg4] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<4x8xi32>
          %6 = memref.load %arg1[%arg3, %arg4] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<4x8xf32>
          %7 = arith.mulf %6, %cst {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : f32
          %8 = arith.fptosi %7 {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : f32 to i32
          %9 = arith.addi %5, %8 {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : i32
          scf.yield %9 : i32
        } else {
          scf.yield %c0_i32 : i32
        } {cycle = 7 : i64, endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 1 : i64, starttime = 1 : i64}
        %3 = memref.load %arg2[] {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<i32>
        %4 = arith.addi %3, %2 {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : i32
        memref.store %4, %arg2[] {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : memref<i32>
      } {cycle = 81 : i64, endtime = 81 : i64, ref_endtime = 81 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 8 : i64}
    } {cycle = 325 : i64, endtime = 325 : i64, ref_endtime = 325 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 4 : i64}
    return
  }
}

