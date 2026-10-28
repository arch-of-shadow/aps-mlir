module {
  func.func @au_generated_candidate_41__direct(%arg0: memref<20x2xf32>, %arg1: memref<20x2xf32>, %arg2: memref<20x2xf32>) attributes {area_square_microns = 1788.03288 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 167.61168000000001 : f64, cycle = 1401 : i64, datapath_area_square_microns = 1620.4212 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 1401 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-planar_slam/target-resource.json", schedule = true} {
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.500000e-01 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c20 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 20 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg3 = %c0 to %c20 step %c1 {
      %0 = memref.load %arg0[%arg3, %c0] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<20x2xf32>
      %1 = arith.mulf %0, %cst {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
      %2 = memref.load %arg1[%arg3, %c0] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<20x2xf32>
      %3 = arith.divf %1, %2 {endtime = 31 : i64, ref_endtime = 31 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32
      %4 = memref.load %arg2[%arg3, %c0] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<20x2xf32>
      %5 = arith.subf %4, %3 {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 31 : i64, starttime = 31 : i64} : f32
      memref.store %5, %arg2[%arg3, %c0] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<20x2xf32>
      %6 = memref.load %arg0[%arg3, %c1] {endtime = 36 : i64, ref_endtime = 36 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : memref<20x2xf32>
      %7 = arith.mulf %6, %cst {endtime = 39 : i64, ref_endtime = 39 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : f32
      %8 = memref.load %arg1[%arg3, %c1] {endtime = 37 : i64, ref_endtime = 37 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : memref<20x2xf32>
      %9 = arith.divf %7, %8 {endtime = 66 : i64, ref_endtime = 66 : i64, ref_starttime = 39 : i64, starttime = 39 : i64} : f32
      %10 = memref.load %arg2[%arg3, %c1] {endtime = 38 : i64, ref_endtime = 38 : i64, ref_starttime = 37 : i64, starttime = 37 : i64} : memref<20x2xf32>
      %11 = arith.subf %10, %9 {endtime = 69 : i64, ref_endtime = 69 : i64, ref_starttime = 66 : i64, starttime = 66 : i64} : f32
      memref.store %11, %arg2[%arg3, %c1] {endtime = 70 : i64, ref_endtime = 70 : i64, ref_starttime = 69 : i64, starttime = 69 : i64} : memref<20x2xf32>
    } {cycle = 1401 : i64, endtime = 1401 : i64, ref_endtime = 1401 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 20 : i64}
    return
  }
}

