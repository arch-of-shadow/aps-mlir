module {
  func.func @au_generated_candidate_12__direct(%arg0: memref<16x3xf32>, %arg1: memref<16x3xf32>, %arg2: memref<16x3xf32>) attributes {area_square_microns = 3651.2985600000006 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 167.61168000000001 : f64, cycle = 1576 : i64, datapath_area_square_microns = 3483.6868800000007 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 1576 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-planar_slam/target-resource.json", schedule = true} {
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.800000e-01 : f32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.200000e-01 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c15 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 15 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg3 = %c0 to %c15 step %c1 {
      %0 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %1 = memref.load %arg0[%0, %c0] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<16x3xf32>
      %2 = arith.mulf %1, %cst_0 {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
      %3 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %4 = memref.load %arg1[%3, %c0] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<16x3xf32>
      %5 = arith.divf %2, %4 {endtime = 31 : i64, ref_endtime = 31 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32
      %6 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %7 = memref.load %arg2[%6, %c0] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<16x3xf32>
      %8 = arith.subf %7, %5 {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 31 : i64, starttime = 31 : i64} : f32
      %9 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      memref.store %8, %arg2[%9, %c0] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<16x3xf32>
      %10 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %11 = memref.load %arg0[%10, %c1] {endtime = 36 : i64, ref_endtime = 36 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : memref<16x3xf32>
      %12 = arith.mulf %11, %cst_0 {endtime = 39 : i64, ref_endtime = 39 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : f32
      %13 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %14 = memref.load %arg1[%13, %c1] {endtime = 37 : i64, ref_endtime = 37 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : memref<16x3xf32>
      %15 = arith.divf %12, %14 {endtime = 66 : i64, ref_endtime = 66 : i64, ref_starttime = 39 : i64, starttime = 39 : i64} : f32
      %16 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %17 = memref.load %arg2[%16, %c1] {endtime = 38 : i64, ref_endtime = 38 : i64, ref_starttime = 37 : i64, starttime = 37 : i64} : memref<16x3xf32>
      %18 = arith.subf %17, %15 {endtime = 69 : i64, ref_endtime = 69 : i64, ref_starttime = 66 : i64, starttime = 66 : i64} : f32
      %19 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      memref.store %18, %arg2[%19, %c1] {endtime = 70 : i64, ref_endtime = 70 : i64, ref_starttime = 69 : i64, starttime = 69 : i64} : memref<16x3xf32>
      %20 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %21 = memref.load %arg0[%20, %c2] {endtime = 71 : i64, ref_endtime = 71 : i64, ref_starttime = 70 : i64, starttime = 70 : i64} : memref<16x3xf32>
      %22 = arith.mulf %21, %cst {endtime = 74 : i64, ref_endtime = 74 : i64, ref_starttime = 71 : i64, starttime = 71 : i64} : f32
      %23 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %24 = memref.load %arg1[%23, %c2] {endtime = 72 : i64, ref_endtime = 72 : i64, ref_starttime = 71 : i64, starttime = 71 : i64} : memref<16x3xf32>
      %25 = arith.divf %22, %24 {endtime = 101 : i64, ref_endtime = 101 : i64, ref_starttime = 74 : i64, starttime = 74 : i64} : f32
      %26 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %27 = memref.load %arg2[%26, %c2] {endtime = 73 : i64, ref_endtime = 73 : i64, ref_starttime = 72 : i64, starttime = 72 : i64} : memref<16x3xf32>
      %28 = arith.subf %27, %25 {endtime = 104 : i64, ref_endtime = 104 : i64, ref_starttime = 101 : i64, starttime = 101 : i64} : f32
      %29 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      memref.store %28, %arg2[%29, %c2] {endtime = 105 : i64, ref_endtime = 105 : i64, ref_starttime = 104 : i64, starttime = 104 : i64} : memref<16x3xf32>
    } {cycle = 1576 : i64, endtime = 1576 : i64, ref_endtime = 1576 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 15 : i64}
    return
  }
}

