module {
  func.func @au_generated_candidate_24__direct(%arg0: memref<12x12x3xf32>, %arg1: memref<12x12xf32>, %arg2: memref<i32>) attributes {area_square_microns = 3834.0734400000001 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 326.35872000000001 : f64, cycle = 1911 : i64, datapath_area_square_microns = 3507.7147199999999 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 1911 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-normal-12/target-resource.json", schedule = true} {
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6.400000e+01 : f32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.100000e+01 : f32
    %cst_1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3.100000e+01 : f32
    %cst_2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.300000e+01 : f32
    %cst_3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.700000e+01 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c10 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 10 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg3 = %c0 to %c10 step %c1 {
      scf.for %arg4 = %c0 to %c10 step %c1 {
        %0 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %1 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %2 = memref.load %arg0[%0, %1, %c0] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<12x12x3xf32>
        %3 = arith.mulf %2, %cst_3 {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
        %4 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %5 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %6 = memref.load %arg0[%4, %5, %c1] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<12x12x3xf32>
        %7 = arith.mulf %6, %cst_2 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : f32
        %8 = arith.addf %3, %7 {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : f32
        %9 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %10 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %11 = memref.load %arg0[%9, %10, %c2] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<12x12x3xf32>
        %12 = arith.mulf %11, %cst_1 {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : f32
        %13 = arith.addf %8, %12 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : f32
        %14 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %15 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %16 = memref.load %arg1[%14, %15] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<12x12xf32>
        %17 = arith.mulf %16, %cst_0 {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32
        %18 = arith.addf %13, %17 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32
        %19 = arith.mulf %18, %cst {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
        %20 = arith.fptosi %19 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : f32 to i32
        %21 = memref.load %arg2[] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<i32>
        %22 = arith.addi %21, %20 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : i32
        memref.store %22, %arg2[] {endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : memref<i32>
      } {cycle = 191 : i64, endtime = 191 : i64, ref_endtime = 191 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 10 : i64}
    } {cycle = 1911 : i64, endtime = 1911 : i64, ref_endtime = 1911 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 10 : i64}
    return
  }
}

