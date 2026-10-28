module {
  func.func @au_generated_candidate_9__direct(%arg0: memref<216xf32>, %arg1: memref<216xf32>, %arg2: memref<216xf32>) attributes {area_square_microns = 4990.8068999999969 : f64, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 493.97040000000004 : f64, cycle = 3093 : i64, datapath_area_square_microns = 4496.8364999999967 : f64, evaluation_status = "ok", fully_unrolled = false, latency_cycles = 3093 : i64, resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-aps-suite-16-poisson/target-resource.json", schedule = true} {
    %cst = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6.000000e+00 : f32
    %c1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : i32
    %c-1_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} -1 : i32
    %c6_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6 : i32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg3 = %c0 to %c4 step %c1 {
      %0 = arith.addi %arg3, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
      %1 = arith.index_cast %0 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
      %2 = arith.muli %1, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
      %3 = arith.addi %1, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
      %4 = arith.muli %3, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
      %5 = arith.addi %1, %c1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
      %6 = arith.muli %5, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
      scf.for %arg4 = %c0 to %c4 step %c1 {
        %7 = arith.addi %arg4, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        %8 = arith.index_cast %7 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %9 = arith.addi %arg5, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
          %10 = arith.index_cast %9 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index to i32
          %11 = arith.addi %2, %8 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %12 = arith.muli %11, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %13 = arith.addi %12, %10 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %14 = arith.addi %4, %8 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %15 = arith.muli %14, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %16 = arith.addi %15, %10 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %17 = arith.addi %6, %8 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %18 = arith.muli %17, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %19 = arith.addi %18, %10 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %20 = arith.addi %8, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %21 = arith.addi %2, %20 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %22 = arith.muli %21, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %23 = arith.addi %22, %10 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %24 = arith.addi %8, %c1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %25 = arith.addi %2, %24 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %26 = arith.muli %25, %c6_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %27 = arith.addi %26, %10 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %28 = arith.addi %10, %c-1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %29 = arith.addi %12, %28 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %30 = arith.addi %10, %c1_i32 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %31 = arith.addi %12, %30 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32
          %32 = arith.index_cast %16 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %33 = memref.load %arg0[%32] {endtime = 1 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<216xf32>
          %34 = arith.index_cast %19 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %35 = memref.load %arg0[%34] {endtime = 2 : i64, ref_endtime = 2 : i64, ref_starttime = 1 : i64, starttime = 1 : i64} : memref<216xf32>
          %36 = arith.addf %33, %35 {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : f32
          %37 = arith.index_cast %23 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %38 = memref.load %arg0[%37] {endtime = 3 : i64, ref_endtime = 3 : i64, ref_starttime = 2 : i64, starttime = 2 : i64} : memref<216xf32>
          %39 = arith.addf %36, %38 {endtime = 8 : i64, ref_endtime = 8 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : f32
          %40 = arith.index_cast %27 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %41 = memref.load %arg0[%40] {endtime = 4 : i64, ref_endtime = 4 : i64, ref_starttime = 3 : i64, starttime = 3 : i64} : memref<216xf32>
          %42 = arith.addf %39, %41 {endtime = 11 : i64, ref_endtime = 11 : i64, ref_starttime = 8 : i64, starttime = 8 : i64} : f32
          %43 = arith.index_cast %29 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %44 = memref.load %arg0[%43] {endtime = 5 : i64, ref_endtime = 5 : i64, ref_starttime = 4 : i64, starttime = 4 : i64} : memref<216xf32>
          %45 = arith.addf %42, %44 {endtime = 14 : i64, ref_endtime = 14 : i64, ref_starttime = 11 : i64, starttime = 11 : i64} : f32
          %46 = arith.index_cast %31 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %47 = memref.load %arg0[%46] {endtime = 6 : i64, ref_endtime = 6 : i64, ref_starttime = 5 : i64, starttime = 5 : i64} : memref<216xf32>
          %48 = arith.addf %45, %47 {endtime = 17 : i64, ref_endtime = 17 : i64, ref_starttime = 14 : i64, starttime = 14 : i64} : f32
          %49 = arith.index_cast %13 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : i32 to index
          %50 = memref.load %arg1[%49] {endtime = 7 : i64, ref_endtime = 7 : i64, ref_starttime = 6 : i64, starttime = 6 : i64} : memref<216xf32>
          %51 = arith.addf %50, %48 {endtime = 20 : i64, ref_endtime = 20 : i64, ref_starttime = 17 : i64, starttime = 17 : i64} : f32
          %52 = arith.divf %51, %cst {endtime = 47 : i64, ref_endtime = 47 : i64, ref_starttime = 20 : i64, starttime = 20 : i64} : f32
          memref.store %52, %arg2[%49] {endtime = 48 : i64, ref_endtime = 48 : i64, ref_starttime = 47 : i64, starttime = 47 : i64} : memref<216xf32>
        } {cycle = 193 : i64, endtime = 193 : i64, ref_endtime = 193 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 4 : i64}
      } {cycle = 773 : i64, endtime = 773 : i64, ref_endtime = 773 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 4 : i64}
    } {cycle = 3093 : i64, endtime = 3093 : i64, ref_endtime = 3093 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 4 : i64}
    return
  }
}

