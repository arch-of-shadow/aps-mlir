module {
  func.func @au_generated_candidate_12__direct(%arg0: i32, %arg1: memref<12x12x3xf32>, %arg2: memref<12x12x3xf32>, %arg3: index) attributes {area_square_microns = 2129.8463999999999 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 485.10576000000003 : f64, cycle = 3361 : i64, datapath_area_square_microns = 1644.74064 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 3361 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-normal-12/target-resource.json", schedule = true} {
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 9.000000e+00 : f32
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %true = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} true
    %c11_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 11 : i32
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c-1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -1 : i32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.000000e+00 : f32
    %c12 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 12 : index
    scf.for %arg4 = %c0 to %c12 step %c1 {
      %0 = arith.index_cast %arg4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      %alloca = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca[] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      %alloca_1 = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca_1[] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<f32>
      %alloca_2 = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca_2[] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<f32>
      scf.for %arg5 = %c0 to %c3 step %c1 {
        %7 = arith.index_cast %arg5 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %8 = arith.addi %7, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        scf.for %arg6 = %c0 to %c3 step %c1 {
          %9 = arith.index_cast %arg6 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
          %10 = arith.addi %9, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %11 = arith.addi %arg0, %8 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %12 = arith.cmpi sge, %11, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %13 = arith.cmpi sgt, %11, %c11_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %14 = arith.andi %12, %13 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %15 = arith.xori %14, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %16 = arith.andi %15, %12 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %17 = arith.select %14, %c11_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %18 = arith.select %16, %11, %17 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %19 = arith.addi %0, %10 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %20 = arith.cmpi sge, %19, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %21 = arith.cmpi sgt, %19, %c11_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %22 = arith.andi %20, %21 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %23 = arith.xori %22, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %24 = arith.andi %23, %20 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %25 = arith.select %22, %c11_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %26 = arith.select %24, %19, %25 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %27 = arith.index_cast %18 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %28 = arith.index_cast %26 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %29 = memref.load %arg1[%27, %28, %c0] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<12x12x3xf32>
          %30 = memref.load %alloca[] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<f32>
          %31 = arith.addf %30, %29 {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : f32
          memref.store %31, %alloca[] {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<f32>
          %32 = memref.load %arg1[%27, %28, %c1] {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : memref<12x12x3xf32>
          %33 = memref.load %alloca_1[] {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : memref<f32>
          %34 = arith.addf %33, %32 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32
          memref.store %34, %alloca_1[] {endtime = 15 : i64, ref_endtime = 15 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : memref<f32>
          %35 = memref.load %arg1[%27, %28, %c2] {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : memref<12x12x3xf32>
          %36 = memref.load %alloca_2[] {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : memref<f32>
          %37 = arith.addf %36, %35 {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : f32
          memref.store %37, %alloca_2[] {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : memref<f32>
        } {cycle = 64 : i64, endtime = 64 : i64, ref_endtime = 64 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 3 : i64}
      } {cycle = 193 : i64, endtime = 193 : i64, ref_endtime = 193 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 3 : i64}
      %1 = memref.load %alloca[] {endtime = 194 : i64, ref_endtime = 194 : i64, ref_starttime = 193 : i64, starttime = 193 : i64} : memref<f32>
      %2 = arith.divf %1, %cst {endtime = 221 : i64, ref_endtime = 221 : i64, ref_starttime = 194 : i64, starttime = 194 : i64} : f32
      memref.store %2, %arg2[%arg3, %arg4, %c0] {endtime = 222 : i64, ref_endtime = 222 : i64, ref_starttime = 221 : i64, starttime = 221 : i64} : memref<12x12x3xf32>
      %3 = memref.load %alloca_1[] {endtime = 223 : i64, ref_endtime = 223 : i64, ref_starttime = 222 : i64, starttime = 222 : i64} : memref<f32>
      %4 = arith.divf %3, %cst {endtime = 250 : i64, ref_endtime = 250 : i64, ref_starttime = 223 : i64, starttime = 223 : i64} : f32
      memref.store %4, %arg2[%arg3, %arg4, %c1] {endtime = 251 : i64, ref_endtime = 251 : i64, ref_starttime = 250 : i64, starttime = 250 : i64} : memref<12x12x3xf32>
      %5 = memref.load %alloca_2[] {endtime = 252 : i64, ref_endtime = 252 : i64, ref_starttime = 251 : i64, starttime = 251 : i64} : memref<f32>
      %6 = arith.divf %5, %cst {endtime = 279 : i64, ref_endtime = 279 : i64, ref_starttime = 252 : i64, starttime = 252 : i64} : f32
      memref.store %6, %arg2[%arg3, %arg4, %c2] {endtime = 280 : i64, ref_endtime = 280 : i64, ref_starttime = 279 : i64, starttime = 279 : i64} : memref<12x12x3xf32>
    } {cycle = 3361 : i64, endtime = 3361 : i64, ref_endtime = 3361 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 12 : i64}
    return
  }
}

