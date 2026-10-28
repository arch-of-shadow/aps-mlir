module {
  func.func @au_generated_candidate_10__direct(%arg0: memref<16x16x3xf32>, %arg1: index, %arg2: i32, %arg3: memref<16x16xf32>) attributes {area_square_microns = 6879.6021600000004 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 493.97040000000004 : f64, cycle = 18449 : i64, datapath_area_square_microns = 6385.6317600000002 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 18449 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-iss-16-fixed/target-resource.json", schedule = true} {
    %c5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 5 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 5.000000e-01 : f32
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %true = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} true
    %c15_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 15 : i32
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c-2_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -2 : i32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.000000e+00 : f32
    %c16 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 16 : index
    scf.for %arg4 = %c0 to %c16 step %c1 {
      %0 = arith.index_cast %arg4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      %alloca = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca[] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      %alloca_1 = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca_1[] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<f32>
      %alloca_2 = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca_2[] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<f32>
      %alloca_3 = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca_3[] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<f32>
      %alloca_4 = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca_4[] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<f32>
      %alloca_5 = memref.alloca() {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<f32>
      memref.store %cst_0, %alloca_5[] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<f32>
      %1 = memref.load %arg0[%arg1, %arg4, %c0] {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<16x16x3xf32>
      %2 = memref.load %arg0[%arg1, %arg4, %c1] {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<16x16x3xf32>
      %3 = memref.load %arg0[%arg1, %arg4, %c2] {endtime = 9 : i64, ref_endtime = 9 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : memref<16x16x3xf32>
      scf.for %arg5 = %c0 to %c5 step %c1 {
        %22 = arith.index_cast %arg5 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        %23 = arith.addi %22, %c-2_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
        scf.for %arg6 = %c0 to %c5 step %c1 {
          %24 = arith.index_cast %arg6 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
          %25 = arith.addi %24, %c-2_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %26 = arith.addi %arg2, %23 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %27 = arith.cmpi sge, %26, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %28 = arith.cmpi sgt, %26, %c15_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %29 = arith.andi %27, %28 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %30 = arith.xori %29, %true {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %31 = arith.andi %30, %27 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %32 = arith.select %29, %c15_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %33 = arith.select %31, %26, %32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %34 = arith.addi %0, %25 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %35 = arith.cmpi sge, %34, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %36 = arith.cmpi sgt, %34, %c15_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %37 = arith.andi %35, %36 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %38 = arith.subi %true, %37 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %39 = arith.andi %38, %35 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i1
          %40 = arith.select %37, %c15_i32, %c0_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %41 = arith.select %39, %34, %40 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %42 = arith.index_cast %33 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %43 = arith.index_cast %41 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %44 = memref.load %arg0[%42, %43, %c0] {endtime = 10 : i64, ref_endtime = 10 : i64, ref_starttime = 9 : i64, starttime = 9 : i64} : memref<16x16x3xf32>
          %45 = arith.subf %44, %1 {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : f32
          %46 = memref.load %arg0[%42, %43, %c1] {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 10 : i64, starttime = 10 : i64} : memref<16x16x3xf32>
          %47 = arith.subf %46, %2 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32
          %48 = memref.load %arg0[%42, %43, %c2] {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<16x16x3xf32>
          %49 = arith.subf %48, %3 {endtime = 15 : i64, ref_endtime = 15 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : f32
          %50 = arith.mulf %45, %45 {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : f32
          %51 = memref.load %alloca[] {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<f32>
          %52 = arith.addf %51, %50 {endtime = 19 : i64, ref_endtime = 19 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : f32
          memref.store %52, %alloca[] {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 19 : i64, starttime = 19 : i64} : memref<f32>
          %53 = arith.mulf %47, %47 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
          %54 = memref.load %alloca_1[] {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : memref<f32>
          %55 = arith.addf %54, %53 {endtime = 24 : i64, ref_endtime = 24 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : f32
          memref.store %55, %alloca_1[] {endtime = 25 : i64, ref_endtime = 25 : i64, ref_starttime = 24 : i64, starttime = 24 : i64} : memref<f32>
          %56 = arith.mulf %49, %49 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : f32
          %57 = memref.load %alloca_2[] {endtime = 26 : i64, ref_endtime = 26 : i64, ref_starttime = 25 : i64, starttime = 25 : i64} : memref<f32>
          %58 = arith.addf %57, %56 {endtime = 29 : i64, ref_endtime = 29 : i64, ref_starttime = 26 : i64, starttime = 26 : i64} : f32
          memref.store %58, %alloca_2[] {endtime = 30 : i64, ref_endtime = 30 : i64, ref_starttime = 29 : i64, starttime = 29 : i64} : memref<f32>
          %59 = arith.mulf %45, %47 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
          %60 = memref.load %alloca_3[] {endtime = 31 : i64, ref_endtime = 31 : i64, ref_starttime = 30 : i64, starttime = 30 : i64} : memref<f32>
          %61 = arith.addf %60, %59 {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 31 : i64, starttime = 31 : i64} : f32
          memref.store %61, %alloca_3[] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<f32>
          %62 = arith.mulf %45, %49 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : f32
          %63 = memref.load %alloca_4[] {endtime = 36 : i64, ref_endtime = 36 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : memref<f32>
          %64 = arith.addf %63, %62 {endtime = 39 : i64, ref_endtime = 39 : i64, ref_starttime = 36 : i64, starttime = 36 : i64} : f32
          memref.store %64, %alloca_4[] {endtime = 40 : i64, ref_endtime = 40 : i64, ref_starttime = 39 : i64, starttime = 39 : i64} : memref<f32>
          %65 = arith.mulf %47, %49 {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 15 : i64, starttime = 15 : i64} : f32
          %66 = memref.load %alloca_5[] {endtime = 41 : i64, ref_endtime = 41 : i64, ref_starttime = 40 : i64, starttime = 40 : i64} : memref<f32>
          %67 = arith.addf %66, %65 {endtime = 44 : i64, ref_endtime = 44 : i64, ref_starttime = 41 : i64, starttime = 41 : i64} : f32
          memref.store %67, %alloca_5[] {endtime = 45 : i64, ref_endtime = 45 : i64, ref_starttime = 44 : i64, starttime = 44 : i64} : memref<f32>
        } {cycle = 226 : i64, endtime = 226 : i64, ref_endtime = 226 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 5 : i64}
      } {cycle = 1131 : i64, endtime = 1131 : i64, ref_endtime = 1131 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 5 : i64}
      %4 = memref.load %alloca[] {endtime = 1132 : i64, ref_endtime = 1132 : i64, ref_starttime = 1131 : i64, starttime = 1131 : i64} : memref<f32>
      %5 = memref.load %alloca_1[] {endtime = 1133 : i64, ref_endtime = 1133 : i64, ref_starttime = 1132 : i64, starttime = 1132 : i64} : memref<f32>
      %6 = arith.addf %4, %5 {endtime = 1136 : i64, ref_endtime = 1136 : i64, ref_starttime = 1133 : i64, starttime = 1133 : i64} : f32
      %7 = memref.load %alloca_2[] {endtime = 1134 : i64, ref_endtime = 1134 : i64, ref_starttime = 1133 : i64, starttime = 1133 : i64} : memref<f32>
      %8 = arith.addf %6, %7 {endtime = 1139 : i64, ref_endtime = 1139 : i64, ref_starttime = 1136 : i64, starttime = 1136 : i64} : f32
      %9 = memref.load %alloca_3[] {endtime = 1135 : i64, ref_endtime = 1135 : i64, ref_starttime = 1134 : i64, starttime = 1134 : i64} : memref<f32>
      %10 = arith.cmpf olt, %9, %cst_0 {endtime = 1136 : i64, ref_endtime = 1136 : i64, ref_starttime = 1135 : i64, starttime = 1135 : i64} : f32
      %11 = scf.if %10 -> (f32) {
        %22 = arith.negf %9 {endtime = 1137 : i64, ref_endtime = 1137 : i64, ref_starttime = 1136 : i64, starttime = 1136 : i64} : f32
        scf.yield %22 : f32
      } else {
        scf.yield %9 : f32
      } {cycle = 1 : i64, endtime = 1137 : i64, ref_endtime = 1137 : i64, ref_starttime = 1136 : i64, starttime = 1136 : i64}
      %12 = memref.load %alloca_4[] {endtime = 1138 : i64, ref_endtime = 1138 : i64, ref_starttime = 1137 : i64, starttime = 1137 : i64} : memref<f32>
      %13 = arith.cmpf olt, %12, %cst_0 {endtime = 1139 : i64, ref_endtime = 1139 : i64, ref_starttime = 1138 : i64, starttime = 1138 : i64} : f32
      %14 = scf.if %13 -> (f32) {
        %22 = arith.negf %12 {endtime = 1140 : i64, ref_endtime = 1140 : i64, ref_starttime = 1139 : i64, starttime = 1139 : i64} : f32
        scf.yield %22 : f32
      } else {
        scf.yield %12 : f32
      } {cycle = 1 : i64, endtime = 1140 : i64, ref_endtime = 1140 : i64, ref_starttime = 1139 : i64, starttime = 1139 : i64}
      %15 = arith.addf %11, %14 {endtime = 1143 : i64, ref_endtime = 1143 : i64, ref_starttime = 1140 : i64, starttime = 1140 : i64} : f32
      %16 = memref.load %alloca_5[] {endtime = 1141 : i64, ref_endtime = 1141 : i64, ref_starttime = 1140 : i64, starttime = 1140 : i64} : memref<f32>
      %17 = arith.cmpf olt, %16, %cst_0 {endtime = 1142 : i64, ref_endtime = 1142 : i64, ref_starttime = 1141 : i64, starttime = 1141 : i64} : f32
      %18 = scf.if %17 -> (f32) {
        %22 = arith.negf %16 {endtime = 1143 : i64, ref_endtime = 1143 : i64, ref_starttime = 1142 : i64, starttime = 1142 : i64} : f32
        scf.yield %22 : f32
      } else {
        scf.yield %16 : f32
      } {cycle = 1 : i64, endtime = 1143 : i64, ref_endtime = 1143 : i64, ref_starttime = 1142 : i64, starttime = 1142 : i64}
      %19 = arith.addf %15, %18 {endtime = 1146 : i64, ref_endtime = 1146 : i64, ref_starttime = 1143 : i64, starttime = 1143 : i64} : f32
      %20 = arith.mulf %19, %cst {endtime = 1149 : i64, ref_endtime = 1149 : i64, ref_starttime = 1146 : i64, starttime = 1146 : i64} : f32
      %21 = arith.subf %8, %20 {endtime = 1152 : i64, ref_endtime = 1152 : i64, ref_starttime = 1149 : i64, starttime = 1149 : i64} : f32
      memref.store %21, %arg3[%arg1, %arg4] {endtime = 1153 : i64, ref_endtime = 1153 : i64, ref_starttime = 1152 : i64, starttime = 1152 : i64} : memref<16x16xf32>
    } {cycle = 18449 : i64, endtime = 18449 : i64, ref_endtime = 18449 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 16 : i64}
    return
  }
}

