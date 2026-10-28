module {
  func.func @au_generated_candidate_5__l4p(%arg0: memref<18x6xf32>, %arg1: memref<18x6xf32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<18x6xf32>, %arg5: memref<30x6xf32>, %arg6: memref<30x6xf32>) attributes {area_square_microns = 4144.8169800000005 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 1309.8671999999999 : f64, cycle = 67684 : i64, datapath_area_square_microns = 2834.9497800000004 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 67684 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-pose3_pose_graph/target-resource.json", schedule = true} {
    %c17 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 17 : index
    %c30 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 30 : index
    %c18 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 18 : index
    %c6 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.200000e-01 : f32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.000000e-01 : f32
    %c3_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : i32
    %cst_1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.000000e-03 : f32
    %cst_2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.000000e+00 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c7 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 7 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg7 = %c0 to %c7 step %c1 {
      scf.for %arg8 = %c0 to %c6 step %c1 {
        scf.for %arg9 = %c0 to %c18 step %c1 {
          memref.store %cst_2, %arg0[%arg9, %arg8] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<18x6xf32>
          memref.store %cst_1, %arg1[%arg9, %arg8] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<18x6xf32>
        } {cycle = 37 : i64, endtime = 37 : i64, ref_endtime = 37 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 18 : i64}
      } {cycle = 223 : i64, endtime = 223 : i64, ref_endtime = 223 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 6 : i64}
      scf.for %arg8 = %c0 to %c30 step %c1 {
        %0 = memref.load %arg2[%arg8] {endtime = 224 : i64, ref_endtime = 224 : i64, ref_starttime = 223 : i64, starttime = 223 : i64} : memref<30xi32>
        %1 = memref.load %arg3[%arg8] {endtime = 225 : i64, ref_endtime = 225 : i64, ref_starttime = 224 : i64, starttime = 224 : i64} : memref<30xi32>
        scf.for %arg9 = %c0 to %c6 step %c1 {
          %2 = arith.index_cast %1 {endtime = 225 : i64, ref_endtime = 225 : i64, ref_starttime = 225 : i64, starttime = 225 : i64} : i32 to index
          %3 = memref.load %arg4[%2, %arg9] {endtime = 226 : i64, ref_endtime = 226 : i64, ref_starttime = 225 : i64, starttime = 225 : i64} : memref<18x6xf32>
          %4 = arith.index_cast %0 {endtime = 224 : i64, ref_endtime = 224 : i64, ref_starttime = 224 : i64, starttime = 224 : i64} : i32 to index
          %5 = memref.load %arg4[%4, %arg9] {endtime = 227 : i64, ref_endtime = 227 : i64, ref_starttime = 226 : i64, starttime = 226 : i64} : memref<18x6xf32>
          %6 = arith.subf %3, %5 {endtime = 230 : i64, ref_endtime = 230 : i64, ref_starttime = 227 : i64, starttime = 227 : i64} : f32
          %7 = memref.load %arg5[%arg8, %arg9] {endtime = 228 : i64, ref_endtime = 228 : i64, ref_starttime = 227 : i64, starttime = 227 : i64} : memref<30x6xf32>
          %8 = arith.subf %6, %7 {endtime = 233 : i64, ref_endtime = 233 : i64, ref_starttime = 230 : i64, starttime = 230 : i64} : f32
          %9 = memref.load %arg6[%arg8, %arg9] {endtime = 229 : i64, ref_endtime = 229 : i64, ref_starttime = 228 : i64, starttime = 228 : i64} : memref<30x6xf32>
          %10 = arith.mulf %9, %8 {endtime = 236 : i64, ref_endtime = 236 : i64, ref_starttime = 233 : i64, starttime = 233 : i64} : f32
          %11 = memref.load %arg0[%4, %arg9] {endtime = 230 : i64, ref_endtime = 230 : i64, ref_starttime = 229 : i64, starttime = 229 : i64} : memref<18x6xf32>
          %12 = arith.addf %11, %10 {endtime = 239 : i64, ref_endtime = 239 : i64, ref_starttime = 236 : i64, starttime = 236 : i64} : f32
          memref.store %12, %arg0[%4, %arg9] {endtime = 240 : i64, ref_endtime = 240 : i64, ref_starttime = 239 : i64, starttime = 239 : i64} : memref<18x6xf32>
          %13 = memref.load %arg0[%2, %arg9] {endtime = 241 : i64, ref_endtime = 241 : i64, ref_starttime = 240 : i64, starttime = 240 : i64} : memref<18x6xf32>
          %14 = arith.subf %13, %10 {endtime = 244 : i64, ref_endtime = 244 : i64, ref_starttime = 241 : i64, starttime = 241 : i64} : f32
          memref.store %14, %arg0[%2, %arg9] {endtime = 245 : i64, ref_endtime = 245 : i64, ref_starttime = 244 : i64, starttime = 244 : i64} : memref<18x6xf32>
          %15 = memref.load %arg6[%arg8, %arg9] {endtime = 246 : i64, ref_endtime = 246 : i64, ref_starttime = 245 : i64, starttime = 245 : i64} : memref<30x6xf32>
          %16 = memref.load %arg1[%4, %arg9] {endtime = 247 : i64, ref_endtime = 247 : i64, ref_starttime = 246 : i64, starttime = 246 : i64} : memref<18x6xf32>
          %17 = arith.addf %16, %15 {endtime = 250 : i64, ref_endtime = 250 : i64, ref_starttime = 247 : i64, starttime = 247 : i64} : f32
          memref.store %17, %arg1[%4, %arg9] {endtime = 251 : i64, ref_endtime = 251 : i64, ref_starttime = 250 : i64, starttime = 250 : i64} : memref<18x6xf32>
          %18 = memref.load %arg6[%arg8, %arg9] {endtime = 252 : i64, ref_endtime = 252 : i64, ref_starttime = 251 : i64, starttime = 251 : i64} : memref<30x6xf32>
          %19 = memref.load %arg1[%2, %arg9] {endtime = 253 : i64, ref_endtime = 253 : i64, ref_starttime = 252 : i64, starttime = 252 : i64} : memref<18x6xf32>
          %20 = arith.addf %19, %18 {endtime = 256 : i64, ref_endtime = 256 : i64, ref_starttime = 253 : i64, starttime = 253 : i64} : f32
          memref.store %20, %arg1[%2, %arg9] {endtime = 257 : i64, ref_endtime = 257 : i64, ref_starttime = 256 : i64, starttime = 256 : i64} : memref<18x6xf32>
        } {II = 32 : i64, cycle = 195 : i64, endtime = 418 : i64, pipeline = 0 : i64, ref_endtime = 418 : i64, ref_starttime = 223 : i64, starttime = 223 : i64, "tripcount-max" = 6 : i64}
      } {cycle = 5851 : i64, endtime = 6074 : i64, ref_endtime = 6074 : i64, ref_starttime = 223 : i64, starttime = 223 : i64, "tripcount-max" = 30 : i64}
      scf.for %arg8 = %c0 to %c6 step %c1 {
        memref.store %cst_2, %arg4[%c0, %arg8] {endtime = 6075 : i64, ref_endtime = 6075 : i64, ref_starttime = 6074 : i64, starttime = 6074 : i64} : memref<18x6xf32>
      } {cycle = 7 : i64, endtime = 6081 : i64, ref_endtime = 6081 : i64, ref_starttime = 6074 : i64, starttime = 6074 : i64, "tripcount-max" = 6 : i64}
      scf.for %arg8 = %c0 to %c17 step %c1 {
        scf.for %arg9 = %c0 to %c6 step %c1 {
          %0 = arith.index_cast %arg9 {endtime = 6081 : i64, ref_endtime = 6081 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64} : index to i32
          %1 = arith.cmpi slt, %0, %c3_i32 {endtime = 6081 : i64, ref_endtime = 6081 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64} : i32
          %2 = arith.select %1, %cst_0, %cst {endtime = 6081 : i64, ref_endtime = 6081 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64} : f32
          %3 = arith.addi %arg8, %c1 {endtime = 6081 : i64, ref_endtime = 6081 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64} : index
          %4 = memref.load %arg0[%3, %arg9] {endtime = 6082 : i64, ref_endtime = 6082 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64} : memref<18x6xf32>
          %5 = arith.mulf %2, %4 {endtime = 6085 : i64, ref_endtime = 6085 : i64, ref_starttime = 6082 : i64, starttime = 6082 : i64} : f32
          %6 = arith.addi %arg8, %c1 {endtime = 6081 : i64, ref_endtime = 6081 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64} : index
          %7 = memref.load %arg1[%6, %arg9] {endtime = 6083 : i64, ref_endtime = 6083 : i64, ref_starttime = 6082 : i64, starttime = 6082 : i64} : memref<18x6xf32>
          %8 = arith.divf %5, %7 {endtime = 6112 : i64, ref_endtime = 6112 : i64, ref_starttime = 6085 : i64, starttime = 6085 : i64} : f32
          %9 = arith.addi %arg8, %c1 {endtime = 6081 : i64, ref_endtime = 6081 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64} : index
          %10 = memref.load %arg4[%9, %arg9] {endtime = 6084 : i64, ref_endtime = 6084 : i64, ref_starttime = 6083 : i64, starttime = 6083 : i64} : memref<18x6xf32>
          %11 = arith.subf %10, %8 {endtime = 6115 : i64, ref_endtime = 6115 : i64, ref_starttime = 6112 : i64, starttime = 6112 : i64} : f32
          %12 = arith.addi %arg8, %c1 {endtime = 6081 : i64, ref_endtime = 6081 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64} : index
          memref.store %11, %arg4[%12, %arg9] {endtime = 6116 : i64, ref_endtime = 6116 : i64, ref_starttime = 6115 : i64, starttime = 6115 : i64} : memref<18x6xf32>
        } {cycle = 211 : i64, endtime = 6292 : i64, ref_endtime = 6292 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64, "tripcount-max" = 6 : i64}
      } {cycle = 3588 : i64, endtime = 9669 : i64, ref_endtime = 9669 : i64, ref_starttime = 6081 : i64, starttime = 6081 : i64, "tripcount-max" = 17 : i64}
    } {cycle = 67684 : i64, endtime = 67684 : i64, ref_endtime = 67684 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 7 : i64}
    return
  }
}

