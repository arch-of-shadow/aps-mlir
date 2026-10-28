module {
  func.func @au_generated_candidate_27__direct(%arg0: memref<16x16xi32>, %arg1: memref<16x16xf32>, %arg2: memref<i32>) attributes {area_square_microns = 1011.6916200000001 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 326.35872000000001 : f64, cycle = 2321 : i64, datapath_area_square_microns = 685.3329 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 2321 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-iss-16-fixed/target-resource.json", schedule = true} {
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3.200000e+01 : f32
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c16 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 16 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg3 = %c0 to %c16 step %c1 {
      %0 = arith.index_cast %arg3 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      scf.for %arg4 = %c0 to %c16 step %c1 {
        %1 = arith.index_cast %arg4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %2 = memref.load %arg0[%arg3, %arg4] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<16x16xi32>
        %3 = arith.cmpi ne, %2, %c0_i32 {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : i32
        %4 = scf.if %3 -> (i32) {
          %7 = memref.load %arg1[%arg3, %arg4] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<16x16xf32>
          %8 = arith.mulf %7, %cst {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : f32
          %9 = arith.fptosi %8 {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : f32 to i32
          %10 = arith.addi %9, %1 {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : i32
          %11 = arith.addi %10, %0 {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : i32
          scf.yield %11 : i32
        } else {
          scf.yield %c0_i32 : i32
        } {cycle = 6 : i64, endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 1 : i64, starttime = 1 : i64}
        %5 = memref.load %arg2[] {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<i32>
        %6 = arith.addi %5, %4 {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : i32
        memref.store %6, %arg2[] {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<i32>
      } {cycle = 145 : i64, endtime = 145 : i64, ref_endtime = 145 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 16 : i64}
    } {cycle = 2321 : i64, endtime = 2321 : i64, ref_endtime = 2321 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 16 : i64}
    return
  }
}

