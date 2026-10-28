module {
  func.func @au_generated_candidate_11__direct(%arg0: memref<216xf32>, %arg1: memref<i32>) attributes {area_square_microns = 4090.1565599999999 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 493.97040000000004 : f64, cycle = 1237 : i64, datapath_area_square_microns = 3596.1861599999997 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 1237 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-poisson/target-resource.json", schedule = true} {
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.000000e+00 : f32
    %c1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : i32
    %c6_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6 : i32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg2 = %c0 to %c4 step %c1 {
      %0 = arith.addi %arg2, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %1 = arith.index_cast %0 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %2 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %3 = arith.index_cast %2 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        scf.for %arg4 = %c0 to %c4 step %c1 {
          %4 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
          %5 = arith.index_cast %4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
          %6 = arith.muli %1, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %7 = arith.addi %6, %3 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %8 = arith.muli %7, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %9 = arith.addi %8, %5 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %10 = arith.addi %1, %c1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %11 = arith.muli %10, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %12 = arith.addi %11, %3 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %13 = arith.muli %12, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %14 = arith.addi %13, %5 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %15 = arith.addi %3, %c1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %16 = arith.addi %6, %15 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %17 = arith.muli %16, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %18 = arith.addi %17, %5 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %19 = arith.addi %5, %c1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %20 = arith.addi %8, %19 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %21 = arith.index_cast %9 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %22 = memref.load %arg0[%21] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<216xf32>
          %23 = arith.index_cast %14 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %24 = memref.load %arg0[%23] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<216xf32>
          %25 = arith.mulf %22, %24 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : f32
          %26 = arith.cmpf olt, %25, %cst {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : f32
          %27 = arith.extui %26 {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : i1 to i32
          %28 = memref.load %arg1[] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<i32>
          %29 = arith.addi %28, %27 {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : i32
          memref.store %29, %arg1[] {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<i32>
          %30 = arith.index_cast %18 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %31 = memref.load %arg0[%30] {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<216xf32>
          %32 = arith.mulf %22, %31 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : f32
          %33 = arith.cmpf olt, %32, %cst {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32
          %34 = arith.extui %33 {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : i1 to i32
          %35 = arith.addi %29, %34 {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : i32
          memref.store %35, %arg1[] {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<i32>
          %36 = arith.index_cast %20 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %37 = memref.load %arg0[%36] {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : memref<216xf32>
          %38 = arith.mulf %22, %37 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
          %39 = arith.cmpf olt, %38, %cst {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : f32
          %40 = arith.extui %39 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : i1 to i32
          %41 = arith.addi %35, %40 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : i32
          memref.store %41, %arg1[] {endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : memref<i32>
        } {cycle = 77 : i64, endtime = 77 : i64, ref_endtime = 77 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 4 : i64}
      } {cycle = 309 : i64, endtime = 309 : i64, ref_endtime = 309 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 4 : i64}
    } {cycle = 1237 : i64, endtime = 1237 : i64, ref_endtime = 1237 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 4 : i64}
    return
  }
}

