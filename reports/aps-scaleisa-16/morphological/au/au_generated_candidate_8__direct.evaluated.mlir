module {
  func.func @au_generated_candidate_8__direct(%arg0: memref<12x12xf32>, %arg1: index, %arg2: i32, %arg3: memref<12x12xf32>) attributes {area_square_microns = 1048.33116 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 485.10576000000003 : f64, cycle = 1897 : i64, datapath_area_square_microns = 563.22539999999992 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 1897 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-morphological-12/target-resource.json", schedule = true} {
    %c5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 5 : index
    %true = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} true
    %c11_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 11 : i32
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c-2_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -2 : i32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c12 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 12 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg4 = %c0 to %c12 step %c1 {
      %0 = arith.index_cast %arg4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      %1 = memref.load %arg0[%arg1, %arg4] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<12x12xf32>
      %alloca = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %1, %alloca[] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<f32>
      scf.for %arg5 = %c0 to %c5 step %c1 {
        %3 = arith.index_cast %arg5 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %4 = arith.addi %3, %c-2_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        scf.for %arg6 = %c0 to %c5 step %c1 {
          %5 = arith.index_cast %arg6 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
          %6 = arith.addi %5, %c-2_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %7 = arith.addi %arg2, %4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %8 = arith.cmpi sge, %7, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %9 = arith.cmpi sgt, %7, %c11_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %10 = arith.andi %8, %9 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %11 = arith.xori %10, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %12 = arith.andi %11, %8 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %13 = arith.select %10, %c11_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %14 = arith.select %12, %7, %13 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %15 = arith.addi %0, %6 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %16 = arith.cmpi sge, %15, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %17 = arith.cmpi sgt, %15, %c11_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %18 = arith.andi %16, %17 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %19 = arith.xori %18, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %20 = arith.andi %19, %16 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %21 = arith.select %18, %c11_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %22 = arith.select %20, %15, %21 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %23 = arith.index_cast %14 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %24 = arith.index_cast %22 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %25 = memref.load %arg0[%23, %24] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<12x12xf32>
          %26 = memref.load %alloca[] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<f32>
          %27 = arith.cmpf ogt, %25, %26 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : f32
          scf.if %27 {
            memref.store %25, %alloca[] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<f32>
          } {cycle = 1 : i64, endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64}
        } {cycle = 31 : i64, endtime = 31 : i64, ref_endtime = 31 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 5 : i64}
      } {cycle = 156 : i64, endtime = 156 : i64, ref_endtime = 156 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 5 : i64}
      %2 = memref.load %alloca[] {endtime = 157 : i64, ref_endtime = 157 : i64, ref_starttime = 156 : i64, starttime = 156 : i64} : memref<f32>
      memref.store %2, %arg3[%arg1, %arg4] {endtime = 158 : i64, ref_endtime = 158 : i64, ref_starttime = 157 : i64, starttime = 157 : i64} : memref<12x12xf32>
    } {cycle = 1897 : i64, endtime = 1897 : i64, ref_endtime = 1897 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 12 : i64}
    return
  }
}

