module {
  func.func @au_generated_candidate_10__direct(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>) attributes {area_square_microns = 1867.52304 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 326.35872000000001 : f64, cycle = 1161 : i64, datapath_area_square_microns = 1541.1643200000001 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 1161 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-segmentation/target-resource.json", schedule = true} {
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.000000e+00 : f32
    %c1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : i32
    %true = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} true
    %c7_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 7 : i32
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c-1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -1 : i32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c8 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg2 = %c0 to %c8 step %c1 {
      %0 = arith.index_cast %arg2 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %1 = arith.index_cast %arg3 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %2 = arith.addi %1, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %3 = arith.cmpi sge, %2, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %4 = arith.cmpi sgt, %2, %c7_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %5 = arith.andi %3, %4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %6 = arith.xori %5, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %7 = arith.andi %6, %3 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %8 = arith.select %5, %c7_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %9 = arith.select %7, %2, %8 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %10 = arith.addi %1, %c1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %11 = arith.cmpi sge, %10, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %12 = arith.cmpi sgt, %10, %c7_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %13 = arith.andi %11, %12 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %14 = arith.xori %13, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %15 = arith.andi %14, %11 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %16 = arith.select %13, %c7_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %17 = arith.select %15, %10, %16 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %18 = arith.addi %0, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %19 = arith.cmpi sge, %18, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %20 = arith.cmpi sgt, %18, %c7_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %21 = arith.andi %19, %20 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %22 = arith.xori %21, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %23 = arith.andi %22, %19 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %24 = arith.select %21, %c7_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %25 = arith.select %23, %18, %24 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %26 = arith.addi %0, %c1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %27 = arith.cmpi sge, %26, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %28 = arith.cmpi sgt, %26, %c7_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %29 = arith.andi %27, %28 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %30 = arith.subi %true, %29 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %31 = arith.andi %30, %27 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
        %32 = arith.select %29, %c7_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %33 = arith.select %31, %26, %32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        %34 = arith.index_cast %17 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
        %35 = memref.load %arg0[%arg2, %34] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<8x8xf32>
        %36 = arith.index_cast %9 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
        %37 = memref.load %arg0[%arg2, %36] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<8x8xf32>
        %38 = arith.subf %35, %37 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : f32
        %39 = arith.cmpf olt, %38, %cst {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : f32
        %40 = scf.if %39 -> (f32) {
          %49 = arith.negf %38 {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : f32
          scf.yield %49 : f32
        } else {
          scf.yield %38 : f32
        } {cycle = 1 : i64, endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64}
        %41 = arith.index_cast %33 {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : i32 to index
        %42 = memref.load %arg0[%41, %arg3] {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<8x8xf32>
        %43 = arith.index_cast %25 {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : i32 to index
        %44 = memref.load %arg0[%43, %arg3] {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<8x8xf32>
        %45 = arith.subf %42, %44 {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : f32
        %46 = arith.cmpf olt, %45, %cst {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : f32
        %47 = scf.if %46 -> (f32) {
          %49 = arith.negf %45 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : f32
          scf.yield %49 : f32
        } else {
          scf.yield %45 : f32
        } {cycle = 1 : i64, endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 13 : i64, starttime = 13 : i64}
        %48 = arith.addf %40, %47 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
        memref.store %48, %arg1[%arg2, %arg3] {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<8x8xf32>
      } {cycle = 145 : i64, endtime = 145 : i64, ref_endtime = 145 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 8 : i64}
    } {cycle = 1161 : i64, endtime = 1161 : i64, ref_endtime = 1161 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 8 : i64}
    return
  }
}

