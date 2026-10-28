module {
  aps.memitfc @busitfc {cache_hint = "cold", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 8 : i64, min_beats = 1 : i64, min_write_beats = 2 : i64, read_latency = 5 : i64, width = 8 : i64, write_completion = 2 : i64}
  aps.memitfc @cpuitfc {cache_hint = "warm", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 1 : i64, min_beats = 1 : i64, min_write_beats = 1 : i64, read_latency = 2 : i64, width = 4 : i64, write_completion = 2 : i64}
  tor.design @aps_isaxes {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
    %c8_i32 = arith.constant 8 : i32
    %c9_i32 = arith.constant 9 : i32
    %c10_i32 = arith.constant 10 : i32
    %c11_i32 = arith.constant 11 : i32
    %c12_i32 = arith.constant 12 : i32
    %c13_i32 = arith.constant 13 : i32
    %c14_i32 = arith.constant 14 : i32
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %c17_i32 = arith.constant 17 : i32
    %c18_i32 = arith.constant 18 : i32
    %c19_i32 = arith.constant 19 : i32
    %c20_i32 = arith.constant 20 : i32
    %c21_i32 = arith.constant 21 : i32
    %c22_i32 = arith.constant 22 : i32
    %c23_i32 = arith.constant 23 : i32
    %c24_i32 = arith.constant 24 : i32
    %c25_i32 = arith.constant 25 : i32
    %c26_i32 = arith.constant 26 : i32
    %c27_i32 = arith.constant 27 : i32
    %c28_i32 = arith.constant 28 : i32
    %c29_i32 = arith.constant 29 : i32
    %c30_i32 = arith.constant 30 : i32
    %c31_i32 = arith.constant 31 : i32
    tor.func @au_generated_candidate_9__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 1.000000e+01 : f32, funct7 = 0 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/scalars/bitcount/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_0", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = arith.andi %0, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %2 = arith.shrui %0, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %3 = arith.andi %2, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %4 = tor.addi %1 %3 on (0 to 1) {dump = "unknown_1", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %5 = arith.shrui %0, %c2_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %6 = arith.andi %5, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %7 = tor.addi %4 %6 on (0 to 1) {dump = "unknown_2", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %8 = arith.shrui %0, %c3_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %9 = arith.andi %8, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %10 = tor.addi %7 %9 on (0 to 1) {dump = "unknown_3", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %11 = arith.shrui %0, %c4_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %12 = arith.andi %11, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %13 = tor.addi %10 %12 on (0 to 1) {dump = "unknown_4", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %14 = arith.shrui %0, %c5_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %15 = arith.andi %14, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %16 = tor.addi %13 %15 on (0 to 1) {dump = "unknown_5", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %17 = arith.shrui %0, %c6_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %18 = arith.andi %17, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %19 = tor.addi %16 %18 on (0 to 1) {dump = "unknown_6", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %20 = arith.shrui %0, %c7_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %21 = arith.andi %20, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %22 = tor.addi %19 %21 on (0 to 1) {dump = "unknown_7", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %23 = arith.shrui %0, %c8_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %24 = arith.andi %23, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %25 = tor.addi %22 %24 on (0 to 1) {dump = "unknown_8", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %26 = arith.shrui %0, %c9_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %27 = arith.andi %26, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %28 = tor.addi %25 %27 on (0 to 1) {dump = "unknown_9", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %29 = arith.shrui %0, %c10_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %30 = arith.andi %29, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %31 = tor.addi %28 %30 on (0 to 1) {dump = "unknown_10", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %32 = arith.shrui %0, %c11_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %33 = arith.andi %32, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %34 = tor.addi %31 %33 on (0 to 1) {dump = "unknown_11", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %35 = arith.shrui %0, %c12_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %36 = arith.andi %35, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %37 = tor.addi %34 %36 on (0 to 1) {dump = "unknown_12", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %38 = arith.shrui %0, %c13_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %39 = arith.andi %38, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %40 = tor.addi %37 %39 on (0 to 1) {dump = "unknown_13", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %41 = arith.shrui %0, %c14_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %42 = arith.andi %41, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %43 = tor.addi %40 %42 on (0 to 1) {dump = "unknown_14", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %44 = arith.shrui %0, %c15_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %45 = arith.andi %44, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %46 = tor.addi %43 %45 on (0 to 1) {dump = "unknown_15", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %47 = arith.shrui %0, %c16_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %48 = arith.andi %47, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %49 = tor.addi %46 %48 on (0 to 1) {dump = "unknown_16", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %50 = arith.shrui %0, %c17_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %51 = arith.andi %50, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %52 = tor.addi %49 %51 on (0 to 1) {dump = "unknown_17", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %53 = arith.shrui %0, %c18_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %54 = arith.andi %53, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %55 = tor.addi %52 %54 on (0 to 1) {dump = "unknown_18", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %56 = arith.shrui %0, %c19_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %57 = arith.andi %56, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %58 = tor.addi %55 %57 on (0 to 1) {dump = "unknown_19", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %59 = arith.shrui %0, %c20_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %60 = arith.andi %59, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %61 = tor.addi %58 %60 on (0 to 1) {dump = "unknown_20", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %62 = arith.shrui %0, %c21_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %63 = arith.andi %62, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %64 = tor.addi %61 %63 on (0 to 1) {dump = "unknown_21", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %65 = arith.shrui %0, %c22_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %66 = arith.andi %65, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %67 = tor.addi %64 %66 on (0 to 1) {dump = "unknown_22", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %68 = arith.shrui %0, %c23_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %69 = arith.andi %68, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %70 = tor.addi %67 %69 on (0 to 1) {dump = "unknown_23", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %71 = arith.shrui %0, %c24_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %72 = arith.andi %71, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %73 = tor.addi %70 %72 on (0 to 1) {dump = "unknown_24", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %74 = arith.shrui %0, %c25_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %75 = arith.andi %74, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %76 = tor.addi %73 %75 on (0 to 1) {dump = "unknown_25", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %77 = arith.shrui %0, %c26_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %78 = arith.andi %77, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %79 = tor.addi %76 %78 on (0 to 1) {dump = "unknown_26", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %80 = arith.shrui %0, %c27_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %81 = arith.andi %80, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %82 = tor.addi %79 %81 on (0 to 1) {dump = "unknown_27", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %83 = arith.shrui %0, %c28_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %84 = arith.andi %83, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %85 = tor.addi %82 %84 on (0 to 1) {dump = "unknown_28", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %86 = arith.shrui %0, %c29_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %87 = arith.andi %86, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %88 = tor.addi %85 %87 on (0 to 1) {dump = "unknown_29", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %89 = arith.shrui %0, %c30_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %90 = arith.andi %89, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %91 = tor.addi %88 %90 on (0 to 1) {dump = "unknown_30", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      %92 = arith.shrui %0, %c31_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %93 = arith.andi %92, %c1_i32 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %94 = tor.addi %91 %93 on (0 to 1) {dump = "unknown_31", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.write_irf %arg2, %94 {dump = "unknown_32", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5, i32
      tor.return
    }
  } {HoistConstCondIfOp = 1 : i32, schedule = true}
}

