module {
  func.func @au_generated_candidate_40__direct(%arg0: memref<12x12xf32>, %arg1: memref<i32>) attributes {area_square_microns = 1011.0063600000001 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 326.35872000000001 : f64, cycle = 877 : i64, datapath_area_square_microns = 684.64764000000002 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 877 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-morphological-12/target-resource.json", schedule = true} {
    %c3_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : i32
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.280000e+02 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c12 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 12 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg2 = %c0 to %c12 step %c1 {
      %0 = arith.index_cast %arg2 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      scf.for %arg3 = %c0 to %c12 step %c1 {
        %1 = arith.index_cast %arg3 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %2 = memref.load %arg0[%arg2, %arg3] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<12x12xf32>
        %3 = arith.mulf %2, %cst {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
        %4 = arith.fptosi %3 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32 to i32
        %5 = arith.addi %1, %0 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %6 = arith.andi %5, %c3_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %7 = arith.addi %4, %6 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : i32
        %8 = memref.load %arg1[] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<i32>
        %9 = arith.addi %8, %7 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : i32
        memref.store %9, %arg1[] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<i32>
      } {cycle = 73 : i64, endtime = 73 : i64, ref_endtime = 73 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 12 : i64}
    } {cycle = 877 : i64, endtime = 877 : i64, ref_endtime = 877 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 12 : i64}
    return
  }
}

