module {
  func.func @au_generated_candidate_2__direct(%arg0: memref<4x8xf32>, %arg1: memref<4x8xi32>, %arg2: memref<32x3xf32>, %arg3: memref<4x3xf32>, %arg4: memref<8xf32>, %arg5: memref<8xi32>) attributes {area_square_microns = 3202.4824200000007 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 811.46447999999998 : f64, cycle = 25929 : i64, datapath_area_square_microns = 2391.0179400000006 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 25929 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-knn-32/target-resource.json", schedule = true} {
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 32 : index
    %c8 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8 : index
    %c1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : i32
    %false = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} false
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c-1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -1 : i32
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.000000e+30 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg6 = %c0 to %c4 step %c1 {
      scf.for %arg7 = %c0 to %c8 step %c1 {
        memref.store %cst, %arg0[%arg6, %arg7] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x8xf32>
        memref.store %c-1_i32, %arg1[%arg6, %arg7] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<4x8xi32>
      } {cycle = 17 : i64, endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 8 : i64}
      scf.for %arg7 = %c0 to %c32 step %c1 {
        %0 = arith.index_cast %arg7 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : index to i32
        %1 = memref.load %arg2[%arg7, %c0] {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<32x3xf32>
        %2 = memref.load %arg3[%arg6, %c0] {endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : memref<4x3xf32>
        %3 = arith.subf %1, %2 {endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : f32
        %4 = memref.load %arg2[%arg7, %c1] {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : memref<32x3xf32>
        %5 = memref.load %arg3[%arg6, %c1] {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : memref<4x3xf32>
        %6 = arith.subf %4, %5 {endtime = 24 : i64, ref_endtime = 24 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : f32
        %7 = memref.load %arg2[%arg7, %c2] {endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : memref<32x3xf32>
        %8 = memref.load %arg3[%arg6, %c2] {endtime = 23 : i64, ref_endtime = 23 : i64, ref_starttime = 22 : i64, starttime = 22 : i64} : memref<4x3xf32>
        %9 = arith.subf %7, %8 {endtime = 26 : i64, ref_endtime = 26 : i64, ref_starttime = 23 : i64, starttime = 23 : i64} : f32
        %10 = arith.mulf %3, %3 {endtime = 25 : i64, ref_endtime = 25 : i64, ref_starttime = 22 : i64, starttime = 22 : i64} : f32
        %11 = arith.mulf %6, %6 {endtime = 27 : i64, ref_endtime = 27 : i64, ref_starttime = 24 : i64, starttime = 24 : i64} : f32
        %12 = arith.addf %10, %11 {endtime = 30 : i64, ref_endtime = 30 : i64, ref_starttime = 27 : i64, starttime = 27 : i64} : f32
        %13 = arith.mulf %9, %9 {endtime = 29 : i64, ref_endtime = 29 : i64, ref_starttime = 26 : i64, starttime = 26 : i64} : f32
        %14 = arith.addf %12, %13 {endtime = 33 : i64, ref_endtime = 33 : i64, ref_starttime = 30 : i64, starttime = 30 : i64} : f32
        %alloca = memref.alloca() {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<i32>
        memref.store %c0_i32, %alloca[] {endtime = 24 : i64, ref_endtime = 24 : i64, ref_starttime = 23 : i64, starttime = 23 : i64} : memref<i32>
        scf.for %arg8 = %c0 to %c8 step %c1 {
          %15 = arith.index_cast %arg8 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : index to i32
          %16 = memref.load %alloca[] {endtime = 25 : i64, ref_endtime = 25 : i64, ref_starttime = 24 : i64, starttime = 24 : i64} : memref<i32>
          %17 = arith.cmpi eq, %16, %c0_i32 {endtime = 25 : i64, ref_endtime = 25 : i64, ref_starttime = 25 : i64, starttime = 25 : i64} : i32
          %18 = scf.if %17 -> (i1) {
            %19 = memref.load %arg0[%arg6, %arg8] {endtime = 26 : i64, ref_endtime = 26 : i64, ref_starttime = 25 : i64, starttime = 25 : i64} : memref<4x8xf32>
            %20 = arith.cmpf olt, %14, %19 {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 33 : i64, starttime = 33 : i64} : f32
            scf.yield %20 : i1
          } else {
            scf.yield %false : i1
          } {cycle = 9 : i64, endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 25 : i64, starttime = 25 : i64}
          scf.if %18 {
            memref.store %14, %arg4[%arg8] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<8xf32>
            memref.store %0, %arg5[%arg8] {endtime = 36 : i64, ref_endtime = 36 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : memref<8xi32>
            memref.store %c1_i32, %alloca[] {endtime = 37 : i64, ref_endtime = 37 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : memref<i32>
          } else {
            %19 = arith.subi %15, %16 {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : i32
            %20 = arith.index_cast %19 {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : i32 to index
            %21 = memref.load %arg0[%arg6, %20] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<4x8xf32>
            memref.store %21, %arg4[%arg8] {endtime = 36 : i64, ref_endtime = 36 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : memref<8xf32>
            %22 = memref.load %arg1[%arg6, %20] {endtime = 37 : i64, ref_endtime = 37 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : memref<4x8xi32>
            memref.store %22, %arg5[%arg8] {endtime = 38 : i64, ref_endtime = 38 : i64, ref_starttime = 37 : i64, starttime = 37 : i64} : memref<8xi32>
          } {cycle = 4 : i64, endtime = 38 : i64, ref_endtime = 38 : i64, ref_starttime = 34 : i64, starttime = 34 : i64}
        } {cycle = 169 : i64, endtime = 186 : i64, ref_endtime = 186 : i64, ref_starttime = 17 : i64, starttime = 17 : i64, "tripcount-max" = 8 : i64}
        scf.for %arg8 = %c0 to %c8 step %c1 {
          %15 = memref.load %arg4[%arg8] {endtime = 187 : i64, ref_endtime = 187 : i64, ref_starttime = 186 : i64, starttime = 186 : i64} : memref<8xf32>
          memref.store %15, %arg0[%arg6, %arg8] {endtime = 188 : i64, ref_endtime = 188 : i64, ref_starttime = 187 : i64, starttime = 187 : i64} : memref<4x8xf32>
          %16 = memref.load %arg5[%arg8] {endtime = 189 : i64, ref_endtime = 189 : i64, ref_starttime = 188 : i64, starttime = 188 : i64} : memref<8xi32>
          memref.store %16, %arg1[%arg6, %arg8] {endtime = 190 : i64, ref_endtime = 190 : i64, ref_starttime = 189 : i64, starttime = 189 : i64} : memref<4x8xi32>
        } {cycle = 33 : i64, endtime = 219 : i64, ref_endtime = 219 : i64, ref_starttime = 186 : i64, starttime = 186 : i64, "tripcount-max" = 8 : i64}
      } {cycle = 6465 : i64, endtime = 6482 : i64, ref_endtime = 6482 : i64, ref_starttime = 17 : i64, starttime = 17 : i64, "tripcount-max" = 32 : i64}
    } {cycle = 25929 : i64, endtime = 25929 : i64, ref_endtime = 25929 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 4 : i64}
    return
  }
}

