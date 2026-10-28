module {
  func.func @au_generated_candidate_0__direct(%arg0: memref<32x3xf32>, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: memref<i32>, %arg6: memref<f32>) attributes {area_square_microns = 2066.2921799999999 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 167.61168000000001 : f64, cycle = 865 : i64, datapath_area_square_microns = 1898.6804999999999 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 865 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-ransac/target-resource.json", schedule = true} {
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : i32
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8.000000e-02 : f32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.000000e+00 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 32 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg7 = %c0 to %c32 step %c1 {
      %0 = memref.load %arg0[%arg7, %c0] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<32x3xf32>
      %1 = arith.subf %0, %arg1 {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
      %2 = memref.load %arg0[%arg7, %c1] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<32x3xf32>
      %3 = arith.subf %2, %arg2 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : f32
      %4 = memref.load %arg0[%arg7, %c2] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<32x3xf32>
      %5 = arith.subf %4, %arg3 {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : f32
      %6 = arith.cmpf olt, %1, %cst_0 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32
      %7 = scf.if %6 -> (f32) {
        %20 = arith.negf %1 {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : f32
        scf.yield %20 : f32
      } else {
        scf.yield %1 : f32
      } {cycle = 1 : i64, endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64}
      %8 = arith.cmpf olt, %3, %cst_0 {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : f32
      %9 = scf.if %8 -> (f32) {
        %20 = arith.negf %3 {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : f32
        scf.yield %20 : f32
      } else {
        scf.yield %3 : f32
      } {cycle = 1 : i64, endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64}
      %10 = arith.addf %7, %9 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : f32
      %11 = arith.cmpf olt, %5, %cst_0 {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : f32
      %12 = scf.if %11 -> (f32) {
        %20 = arith.negf %5 {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : f32
        scf.yield %20 : f32
      } else {
        scf.yield %5 : f32
      } {cycle = 1 : i64, endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64}
      %13 = arith.addf %10, %12 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32
      %14 = arith.subf %13, %arg4 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
      %15 = arith.cmpf olt, %14, %cst_0 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : f32
      %16 = scf.if %15 -> (f32) {
        %20 = arith.negf %14 {endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : f32
        scf.yield %20 : f32
      } else {
        scf.yield %14 : f32
      } {cycle = 1 : i64, endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 18 : i64, starttime = 18 : i64}
      %17 = arith.cmpf olt, %16, %cst {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : f32
      scf.if %17 {
        %20 = memref.load %arg5[] {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : memref<i32>
        %21 = arith.addi %20, %c1_i32 {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : i32
        memref.store %21, %arg5[] {endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : memref<i32>
      } {cycle = 2 : i64, endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 20 : i64, starttime = 20 : i64}
      %18 = memref.load %arg6[] {endtime = 23 : i64, ref_endtime = 23 : i64, ref_starttime = 22 : i64, starttime = 22 : i64} : memref<f32>
      %19 = arith.addf %18, %16 {endtime = 26 : i64, ref_endtime = 26 : i64, ref_starttime = 23 : i64, starttime = 23 : i64} : f32
      memref.store %19, %arg6[] {endtime = 27 : i64, ref_endtime = 27 : i64, ref_starttime = 26 : i64, starttime = 26 : i64} : memref<f32>
    } {cycle = 865 : i64, endtime = 865 : i64, ref_endtime = 865 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 32 : i64}
    return
  }
}

