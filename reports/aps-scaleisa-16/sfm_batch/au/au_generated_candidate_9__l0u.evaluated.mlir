module {
  func.func @au_generated_candidate_9__l0u(%arg0: memref<8x3xf32>) attributes {area_square_microns = 3958.0617600000005 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 0.000000e+00 : f64, cycle = 108 : i64, datapath_area_square_microns = 3958.0617600000005 : f64, evaluation_status = "ok", fully_unrolled = true, latency_cycles = 108 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-sfm_batch/target-resource.json", schedule = true} {
    %c7 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 7 : index
    %c6 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6 : index
    %c5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 5 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2.400000e-01 : f32
    %cst_0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4.000000e-02 : f32
    %cst_1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1.200000e-01 : f32
    %cst_2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8.000000e-02 : f32
    %cst_3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0.000000e+00 : f32
    %cst_4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -4.000000e-02 : f32
    %cst_5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -1.200000e-01 : f32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %0 = memref.load %arg0[%c0, %c0] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<8x3xf32>
    %1 = arith.addf %0, %cst_5 {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : f32
    memref.store %1, %arg0[%c0, %c0] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<8x3xf32>
    %2 = memref.load %arg0[%c0, %c1] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<8x3xf32>
    memref.store %2, %arg0[%c0, %c1] {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<8x3xf32>
    %3 = memref.load %arg0[%c0, %c2] {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 7 : i64, starttime = 7 : i64} : memref<8x3xf32>
    %4 = arith.addf %3, %cst_4 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : f32
    memref.store %4, %arg0[%c0, %c2] {endtime = 12 : i64, ref_endtime = 12 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : memref<8x3xf32>
    %5 = memref.load %arg0[%c1, %c0] {endtime = 13 : i64, ref_endtime = 13 : i64, ref_starttime = 12 : i64, starttime = 12 : i64} : memref<8x3xf32>
    %6 = arith.addf %5, %cst_3 {endtime = 16 : i64, ref_endtime = 16 : i64, ref_starttime = 13 : i64, starttime = 13 : i64} : f32
    memref.store %6, %arg0[%c1, %c0] {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 16 : i64, starttime = 16 : i64} : memref<8x3xf32>
    %7 = memref.load %arg0[%c1, %c1] {endtime = 18 : i64, ref_endtime = 18 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : memref<8x3xf32>
    %8 = arith.subf %7, %cst_2 {endtime = 21 : i64, ref_endtime = 21 : i64, ref_starttime = 18 : i64, starttime = 18 : i64} : f32
    memref.store %8, %arg0[%c1, %c1] {endtime = 22 : i64, ref_endtime = 22 : i64, ref_starttime = 21 : i64, starttime = 21 : i64} : memref<8x3xf32>
    %9 = memref.load %arg0[%c1, %c2] {endtime = 23 : i64, ref_endtime = 23 : i64, ref_starttime = 22 : i64, starttime = 22 : i64} : memref<8x3xf32>
    %10 = arith.addf %9, %cst_3 {endtime = 26 : i64, ref_endtime = 26 : i64, ref_starttime = 23 : i64, starttime = 23 : i64} : f32
    memref.store %10, %arg0[%c1, %c2] {endtime = 27 : i64, ref_endtime = 27 : i64, ref_starttime = 26 : i64, starttime = 26 : i64} : memref<8x3xf32>
    %11 = memref.load %arg0[%c2, %c0] {endtime = 28 : i64, ref_endtime = 28 : i64, ref_starttime = 27 : i64, starttime = 27 : i64} : memref<8x3xf32>
    %12 = arith.addf %11, %cst_1 {endtime = 31 : i64, ref_endtime = 31 : i64, ref_starttime = 28 : i64, starttime = 28 : i64} : f32
    memref.store %12, %arg0[%c2, %c0] {endtime = 32 : i64, ref_endtime = 32 : i64, ref_starttime = 31 : i64, starttime = 31 : i64} : memref<8x3xf32>
    %13 = memref.load %arg0[%c2, %c1] {endtime = 33 : i64, ref_endtime = 33 : i64, ref_starttime = 32 : i64, starttime = 32 : i64} : memref<8x3xf32>
    memref.store %13, %arg0[%c2, %c1] {endtime = 34 : i64, ref_endtime = 34 : i64, ref_starttime = 33 : i64, starttime = 33 : i64} : memref<8x3xf32>
    %14 = memref.load %arg0[%c2, %c2] {endtime = 35 : i64, ref_endtime = 35 : i64, ref_starttime = 34 : i64, starttime = 34 : i64} : memref<8x3xf32>
    %15 = arith.addf %14, %cst_0 {endtime = 38 : i64, ref_endtime = 38 : i64, ref_starttime = 35 : i64, starttime = 35 : i64} : f32
    memref.store %15, %arg0[%c2, %c2] {endtime = 39 : i64, ref_endtime = 39 : i64, ref_starttime = 38 : i64, starttime = 38 : i64} : memref<8x3xf32>
    %16 = memref.load %arg0[%c3, %c0] {endtime = 40 : i64, ref_endtime = 40 : i64, ref_starttime = 39 : i64, starttime = 39 : i64} : memref<8x3xf32>
    %17 = arith.addf %16, %cst {endtime = 43 : i64, ref_endtime = 43 : i64, ref_starttime = 40 : i64, starttime = 40 : i64} : f32
    memref.store %17, %arg0[%c3, %c0] {endtime = 44 : i64, ref_endtime = 44 : i64, ref_starttime = 43 : i64, starttime = 43 : i64} : memref<8x3xf32>
    %18 = memref.load %arg0[%c3, %c1] {endtime = 45 : i64, ref_endtime = 45 : i64, ref_starttime = 44 : i64, starttime = 44 : i64} : memref<8x3xf32>
    %19 = arith.subf %18, %cst_2 {endtime = 48 : i64, ref_endtime = 48 : i64, ref_starttime = 45 : i64, starttime = 45 : i64} : f32
    memref.store %19, %arg0[%c3, %c1] {endtime = 49 : i64, ref_endtime = 49 : i64, ref_starttime = 48 : i64, starttime = 48 : i64} : memref<8x3xf32>
    %20 = memref.load %arg0[%c3, %c2] {endtime = 50 : i64, ref_endtime = 50 : i64, ref_starttime = 49 : i64, starttime = 49 : i64} : memref<8x3xf32>
    %21 = arith.addf %20, %cst_4 {endtime = 53 : i64, ref_endtime = 53 : i64, ref_starttime = 50 : i64, starttime = 50 : i64} : f32
    memref.store %21, %arg0[%c3, %c2] {endtime = 54 : i64, ref_endtime = 54 : i64, ref_starttime = 53 : i64, starttime = 53 : i64} : memref<8x3xf32>
    %22 = memref.load %arg0[%c4, %c0] {endtime = 55 : i64, ref_endtime = 55 : i64, ref_starttime = 54 : i64, starttime = 54 : i64} : memref<8x3xf32>
    %23 = arith.addf %22, %cst_5 {endtime = 58 : i64, ref_endtime = 58 : i64, ref_starttime = 55 : i64, starttime = 55 : i64} : f32
    memref.store %23, %arg0[%c4, %c0] {endtime = 59 : i64, ref_endtime = 59 : i64, ref_starttime = 58 : i64, starttime = 58 : i64} : memref<8x3xf32>
    %24 = memref.load %arg0[%c4, %c1] {endtime = 60 : i64, ref_endtime = 60 : i64, ref_starttime = 59 : i64, starttime = 59 : i64} : memref<8x3xf32>
    memref.store %24, %arg0[%c4, %c1] {endtime = 61 : i64, ref_endtime = 61 : i64, ref_starttime = 60 : i64, starttime = 60 : i64} : memref<8x3xf32>
    %25 = memref.load %arg0[%c4, %c2] {endtime = 62 : i64, ref_endtime = 62 : i64, ref_starttime = 61 : i64, starttime = 61 : i64} : memref<8x3xf32>
    %26 = arith.addf %25, %cst_3 {endtime = 65 : i64, ref_endtime = 65 : i64, ref_starttime = 62 : i64, starttime = 62 : i64} : f32
    memref.store %26, %arg0[%c4, %c2] {endtime = 66 : i64, ref_endtime = 66 : i64, ref_starttime = 65 : i64, starttime = 65 : i64} : memref<8x3xf32>
    %27 = memref.load %arg0[%c5, %c0] {endtime = 67 : i64, ref_endtime = 67 : i64, ref_starttime = 66 : i64, starttime = 66 : i64} : memref<8x3xf32>
    %28 = arith.addf %27, %cst_3 {endtime = 70 : i64, ref_endtime = 70 : i64, ref_starttime = 67 : i64, starttime = 67 : i64} : f32
    memref.store %28, %arg0[%c5, %c0] {endtime = 71 : i64, ref_endtime = 71 : i64, ref_starttime = 70 : i64, starttime = 70 : i64} : memref<8x3xf32>
    %29 = memref.load %arg0[%c5, %c1] {endtime = 72 : i64, ref_endtime = 72 : i64, ref_starttime = 71 : i64, starttime = 71 : i64} : memref<8x3xf32>
    %30 = arith.subf %29, %cst_2 {endtime = 75 : i64, ref_endtime = 75 : i64, ref_starttime = 72 : i64, starttime = 72 : i64} : f32
    memref.store %30, %arg0[%c5, %c1] {endtime = 76 : i64, ref_endtime = 76 : i64, ref_starttime = 75 : i64, starttime = 75 : i64} : memref<8x3xf32>
    %31 = memref.load %arg0[%c5, %c2] {endtime = 77 : i64, ref_endtime = 77 : i64, ref_starttime = 76 : i64, starttime = 76 : i64} : memref<8x3xf32>
    %32 = arith.addf %31, %cst_0 {endtime = 80 : i64, ref_endtime = 80 : i64, ref_starttime = 77 : i64, starttime = 77 : i64} : f32
    memref.store %32, %arg0[%c5, %c2] {endtime = 81 : i64, ref_endtime = 81 : i64, ref_starttime = 80 : i64, starttime = 80 : i64} : memref<8x3xf32>
    %33 = memref.load %arg0[%c6, %c0] {endtime = 82 : i64, ref_endtime = 82 : i64, ref_starttime = 81 : i64, starttime = 81 : i64} : memref<8x3xf32>
    %34 = arith.addf %33, %cst_1 {endtime = 85 : i64, ref_endtime = 85 : i64, ref_starttime = 82 : i64, starttime = 82 : i64} : f32
    memref.store %34, %arg0[%c6, %c0] {endtime = 86 : i64, ref_endtime = 86 : i64, ref_starttime = 85 : i64, starttime = 85 : i64} : memref<8x3xf32>
    %35 = memref.load %arg0[%c6, %c1] {endtime = 87 : i64, ref_endtime = 87 : i64, ref_starttime = 86 : i64, starttime = 86 : i64} : memref<8x3xf32>
    memref.store %35, %arg0[%c6, %c1] {endtime = 88 : i64, ref_endtime = 88 : i64, ref_starttime = 87 : i64, starttime = 87 : i64} : memref<8x3xf32>
    %36 = memref.load %arg0[%c6, %c2] {endtime = 89 : i64, ref_endtime = 89 : i64, ref_starttime = 88 : i64, starttime = 88 : i64} : memref<8x3xf32>
    %37 = arith.addf %36, %cst_4 {endtime = 92 : i64, ref_endtime = 92 : i64, ref_starttime = 89 : i64, starttime = 89 : i64} : f32
    memref.store %37, %arg0[%c6, %c2] {endtime = 93 : i64, ref_endtime = 93 : i64, ref_starttime = 92 : i64, starttime = 92 : i64} : memref<8x3xf32>
    %38 = memref.load %arg0[%c7, %c0] {endtime = 94 : i64, ref_endtime = 94 : i64, ref_starttime = 93 : i64, starttime = 93 : i64} : memref<8x3xf32>
    %39 = arith.addf %38, %cst {endtime = 97 : i64, ref_endtime = 97 : i64, ref_starttime = 94 : i64, starttime = 94 : i64} : f32
    memref.store %39, %arg0[%c7, %c0] {endtime = 98 : i64, ref_endtime = 98 : i64, ref_starttime = 97 : i64, starttime = 97 : i64} : memref<8x3xf32>
    %40 = memref.load %arg0[%c7, %c1] {endtime = 99 : i64, ref_endtime = 99 : i64, ref_starttime = 98 : i64, starttime = 98 : i64} : memref<8x3xf32>
    %41 = arith.subf %40, %cst_2 {endtime = 102 : i64, ref_endtime = 102 : i64, ref_starttime = 99 : i64, starttime = 99 : i64} : f32
    memref.store %41, %arg0[%c7, %c1] {endtime = 103 : i64, ref_endtime = 103 : i64, ref_starttime = 102 : i64, starttime = 102 : i64} : memref<8x3xf32>
    %42 = memref.load %arg0[%c7, %c2] {endtime = 104 : i64, ref_endtime = 104 : i64, ref_starttime = 103 : i64, starttime = 103 : i64} : memref<8x3xf32>
    %43 = arith.addf %42, %cst_3 {endtime = 107 : i64, ref_endtime = 107 : i64, ref_starttime = 104 : i64, starttime = 104 : i64} : f32
    memref.store %43, %arg0[%c7, %c2] {endtime = 108 : i64, ref_endtime = 108 : i64, ref_starttime = 107 : i64, starttime = 107 : i64} : memref<8x3xf32>
    return
  }
}

