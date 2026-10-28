module {
  func.func @au_generated_candidate_0__direct(%arg0: memref<8x8xi32>, %arg1: index, %arg2: i32, %arg3: memref<8x8xf32>, %arg4: memref<8x8xf32>) attributes {area_square_microns = 1638.9815400000002 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 485.10576000000003 : f64, cycle = 1633 : i64, datapath_area_square_microns = 1153.8757800000001 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 1633 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-segmentation/target-resource.json", schedule = true} {
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %false = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} false
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.000000e+00 : f32
    %true = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} true
    %c7_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 7 : i32
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c-1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -1 : i32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.000000e+30 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c8 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg5 = %c0 to %c8 step %c1 {
      %0 = arith.index_cast %arg5 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      %1 = memref.load %arg0[%arg1, %arg5] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<8x8xi32>
      %alloca = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<i32>
      memref.store %1, %alloca[] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<i32>
      %alloca_1 = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca_1[] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<f32>
      scf.for %arg6 = %c0 to %c3 step %c1 {
        %3 = arith.index_cast %arg6 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %4 = arith.addi %3, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        scf.for %arg7 = %c0 to %c3 step %c1 {
          %5 = arith.index_cast %arg7 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
          %6 = arith.addi %5, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %7 = arith.addi %arg2, %4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %8 = arith.cmpi sge, %7, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %9 = arith.cmpi sgt, %7, %c7_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %10 = arith.andi %8, %9 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %11 = arith.xori %10, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %12 = arith.andi %11, %8 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %13 = arith.select %10, %c7_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %14 = arith.select %12, %7, %13 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %15 = arith.addi %0, %6 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %16 = arith.cmpi sge, %15, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %17 = arith.cmpi sgt, %15, %c7_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %18 = arith.andi %16, %17 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %19 = arith.xori %18, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %20 = arith.andi %19, %16 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %21 = arith.select %18, %c7_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %22 = arith.select %20, %15, %21 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %23 = memref.load %arg3[%arg1, %arg5] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<8x8xf32>
          %24 = arith.index_cast %14 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %25 = arith.index_cast %22 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %26 = memref.load %arg3[%24, %25] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<8x8xf32>
          %27 = arith.subf %23, %26 {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : f32
          %28 = arith.cmpf olt, %27, %cst {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : f32
          %29 = scf.if %28 -> (f32) {
            %35 = arith.negf %27 {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : f32
            scf.yield %35 : f32
          } else {
            scf.yield %27 : f32
          } {cycle = 1 : i64, endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64}
          %30 = memref.load %arg4[%24, %25] {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : memref<8x8xf32>
          %31 = arith.addf %29, %30 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32
          %32 = memref.load %alloca_1[] {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<f32>
          %33 = arith.cmpf olt, %31, %32 {endtime = 15 : i64, ref_endtime = 15 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
          %34 = scf.if %33 -> (i1) {
            scf.yield %true : i1
          } else {
            %35 = arith.cmpf oeq, %31, %32 {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : f32
            %36 = scf.if %35 -> (i1) {
              %37 = memref.load %arg0[%24, %25] {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : memref<8x8xi32>
              %38 = memref.load %alloca[] {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<i32>
              %39 = arith.cmpi slt, %37, %38 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : i32
              scf.yield %39 : i1
            } else {
              scf.yield %false : i1
            } {cycle = 3 : i64, endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 16 : i64, starttime = 16 : i64}
            scf.yield %36 : i1
          } {cycle = 4 : i64, endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 15 : i64, starttime = 15 : i64}
          scf.if %34 {
            memref.store %31, %alloca_1[] {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : memref<f32>
            %35 = memref.load %arg0[%24, %25] {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : memref<8x8xi32>
            memref.store %35, %alloca[] {endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : memref<i32>
          } {cycle = 3 : i64, endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 19 : i64, starttime = 19 : i64}
        } {cycle = 67 : i64, endtime = 67 : i64, ref_endtime = 67 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 3 : i64}
      } {cycle = 202 : i64, endtime = 202 : i64, ref_endtime = 202 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 3 : i64}
      %2 = memref.load %alloca[] {endtime = 203 : i64, ref_endtime = 203 : i64, ref_starttime = 202 : i64, starttime = 202 : i64} : memref<i32>
      memref.store %2, %arg0[%arg1, %arg5] {endtime = 204 : i64, ref_endtime = 204 : i64, ref_starttime = 203 : i64, starttime = 203 : i64} : memref<8x8xi32>
    } {cycle = 1633 : i64, endtime = 1633 : i64, ref_endtime = 1633 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 8 : i64}
    return
  }
}

