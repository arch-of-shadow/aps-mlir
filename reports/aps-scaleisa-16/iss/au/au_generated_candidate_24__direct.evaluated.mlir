module {
  func.func @au_generated_candidate_24__direct(%arg0: memref<16x16xf32>, %arg1: index, %arg2: i32, %arg3: memref<16x16xi32>) attributes {area_square_microns = 1185.44148 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 485.10576000000003 : f64, cycle = 961 : i64, datapath_area_square_microns = 700.33572000000004 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 961 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-iss-16-fixed/target-resource.json", schedule = true} {
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %true = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} true
    %c15_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 15 : i32
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c-1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -1 : i32
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.000000e-02 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c16 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 16 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg4 = %c0 to %c16 step %c1 {
      %0 = arith.index_cast %arg4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      %1 = memref.load %arg0[%arg1, %arg4] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<16x16xf32>
      %2 = arith.cmpf ogt, %1, %cst {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
      %3 = arith.extui %2 {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : i1 to i32
      %alloca = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<i32>
      memref.store %3, %alloca[] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<i32>
      scf.for %arg5 = %c0 to %c3 step %c1 {
        %5 = arith.index_cast %arg5 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %6 = arith.addi %5, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        scf.for %arg6 = %c0 to %c3 step %c1 {
          %7 = arith.index_cast %arg6 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
          %8 = arith.addi %7, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %9 = arith.addi %arg2, %6 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %10 = arith.cmpi sge, %9, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %11 = arith.cmpi sgt, %9, %c15_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %12 = arith.andi %10, %11 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %13 = arith.xori %12, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %14 = arith.andi %13, %10 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %15 = arith.select %12, %c15_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %16 = arith.select %14, %9, %15 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %17 = arith.addi %0, %8 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %18 = arith.cmpi sge, %17, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %19 = arith.cmpi sgt, %17, %c15_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %20 = arith.andi %18, %19 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %21 = arith.xori %20, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %22 = arith.andi %21, %18 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %23 = arith.select %20, %c15_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %24 = arith.select %22, %17, %23 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %25 = arith.cmpi ne, %6, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %26 = scf.if %25 -> (i1) {
            scf.yield %true : i1
          } else {
            %27 = arith.cmpi ne, %8, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
            scf.yield %27 : i1
          } {cycle = 1 : i64, endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64}
          scf.if %26 {
            %27 = arith.index_cast %16 {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : i32 to index
            %28 = arith.index_cast %24 {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : i32 to index
            %29 = memref.load %arg0[%27, %28] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<16x16xf32>
            %30 = arith.cmpf ogt, %29, %1 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32
            scf.if %30 {
              memref.store %c0_i32, %alloca[] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<i32>
            } {cycle = 1 : i64, endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64}
          } {cycle = 5 : i64, endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 1 : i64, starttime = 1 : i64}
        } {cycle = 19 : i64, endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 3 : i64}
      } {cycle = 58 : i64, endtime = 58 : i64, ref_endtime = 58 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 3 : i64}
      %4 = memref.load %alloca[] {endtime = 59 : i64, ref_endtime = 59 : i64, ref_starttime = 58 : i64, starttime = 58 : i64} : memref<i32>
      memref.store %4, %arg3[%arg1, %arg4] {endtime = 60 : i64, ref_endtime = 60 : i64, ref_starttime = 59 : i64, starttime = 59 : i64} : memref<16x16xi32>
    } {cycle = 961 : i64, endtime = 961 : i64, ref_endtime = 961 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 16 : i64}
    return
  }
}

