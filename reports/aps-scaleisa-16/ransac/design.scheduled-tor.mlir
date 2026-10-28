module {
  aps.memitfc @busitfc {cache_hint = "cold", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 8 : i64, min_beats = 1 : i64, min_write_beats = 2 : i64, read_latency = 5 : i64, width = 8 : i64, write_completion = 2 : i64}
  aps.memitfc @cpuitfc {cache_hint = "warm", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 1 : i64, min_beats = 1 : i64, min_write_beats = 1 : i64, read_latency = 2 : i64, width = 4 : i64, write_completion = 2 : i64}
  aps.memorymap {
    aps.mem_entry "au_generated_candidate_0__direct_buffer" : banks([@au_generated_candidate_0__direct_buffer_0_0, @au_generated_candidate_0__direct_buffer_1_0]), base(0), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_0__direct_buffer_0" : banks([@au_generated_candidate_0__direct_buffer_0]), base(512), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_0__direct_buffer_1" : banks([@au_generated_candidate_0__direct_buffer_1]), base(516), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_1__l0u_buffer" : banks([@au_generated_candidate_1__l0u_buffer_0, @au_generated_candidate_1__l0u_buffer_1]), base(520), size(384), count(2), cyclic(1)
    aps.mem_finish
  }
  tor.design @aps_isaxes {
    %c31_i6 = arith.constant 31 : i6
    %c1_i6 = arith.constant 1 : i6
    %c0_i6 = arith.constant 0 : i6
    %cst = arith.constant 6.000000e-01 : f32
    %cst_0 = arith.constant 4.000000e-01 : f32
    %cst_1 = arith.constant 3.000000e-01 : f32
    %cst_2 = arith.constant 0.603125036 : f32
    %cst_3 = arith.constant 0.306451619 : f32
    %cst_4 = arith.constant 0.606250048 : f32
    %cst_5 = arith.constant 0.312903225 : f32
    %cst_6 = arith.constant 6.093750e-01 : f32
    %cst_7 = arith.constant 0.319354862 : f32
    %cst_8 = arith.constant 6.125000e-01 : f32
    %cst_9 = arith.constant 0.325806469 : f32
    %cst_10 = arith.constant 6.156250e-01 : f32
    %cst_11 = arith.constant 0.332258075 : f32
    %cst_12 = arith.constant 0.618750035 : f32
    %cst_13 = arith.constant 0.338709682 : f32
    %cst_14 = arith.constant 0.621875048 : f32
    %cst_15 = arith.constant 0.345161289 : f32
    %cst_16 = arith.constant 6.250000e-01 : f32
    %cst_17 = arith.constant 0.351612926 : f32
    %cst_18 = arith.constant 6.281250e-01 : f32
    %cst_19 = arith.constant 0.358064532 : f32
    %cst_20 = arith.constant 6.312500e-01 : f32
    %cst_21 = arith.constant 0.364516139 : f32
    %cst_22 = arith.constant 0.634375035 : f32
    %cst_23 = arith.constant 0.370967746 : f32
    %cst_24 = arith.constant 0.637500048 : f32
    %cst_25 = arith.constant 0.377419353 : f32
    %cst_26 = arith.constant 6.406250e-01 : f32
    %cst_27 = arith.constant 3.838710e-01 : f32
    %cst_28 = arith.constant 6.437500e-01 : f32
    %cst_29 = arith.constant 0.390322596 : f32
    %cst_30 = arith.constant 6.468750e-01 : f32
    %cst_31 = arith.constant 0.396774203 : f32
    %cst_32 = arith.constant 0.650000035 : f32
    %cst_33 = arith.constant 0.403225809 : f32
    %cst_34 = arith.constant 0.653125048 : f32
    %cst_35 = arith.constant 0.409677446 : f32
    %cst_36 = arith.constant 6.562500e-01 : f32
    %cst_37 = arith.constant 0.416129053 : f32
    %cst_38 = arith.constant 6.593750e-01 : f32
    %cst_39 = arith.constant 0.422580659 : f32
    %cst_40 = arith.constant 6.625000e-01 : f32
    %cst_41 = arith.constant 0.429032266 : f32
    %cst_42 = arith.constant 0.665625035 : f32
    %cst_43 = arith.constant 0.435483873 : f32
    %cst_44 = arith.constant 0.668750048 : f32
    %cst_45 = arith.constant 0.44193548 : f32
    %cst_46 = arith.constant 6.718750e-01 : f32
    %cst_47 = arith.constant 0.448387086 : f32
    %cst_48 = arith.constant 6.750000e-01 : f32
    %cst_49 = arith.constant 0.454838723 : f32
    %cst_50 = arith.constant 6.781250e-01 : f32
    %cst_51 = arith.constant 0.46129033 : f32
    %cst_52 = arith.constant 0.681250035 : f32
    %cst_53 = arith.constant 0.467741966 : f32
    %cst_54 = arith.constant 0.684375048 : f32
    %cst_55 = arith.constant 0.474193573 : f32
    %cst_56 = arith.constant 6.875000e-01 : f32
    %cst_57 = arith.constant 0.48064518 : f32
    %cst_58 = arith.constant 6.906250e-01 : f32
    %cst_59 = arith.constant 0.487096786 : f32
    %cst_60 = arith.constant 6.937500e-01 : f32
    %cst_61 = arith.constant 0.493548393 : f32
    %cst_62 = arith.constant 0.696875035 : f32
    %c2_i32 = arith.constant 2 : i32
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
    %c33_i32 = arith.constant 33 : i32
    %c34_i32 = arith.constant 34 : i32
    %c35_i32 = arith.constant 35 : i32
    %c36_i32 = arith.constant 36 : i32
    %c37_i32 = arith.constant 37 : i32
    %c38_i32 = arith.constant 38 : i32
    %c39_i32 = arith.constant 39 : i32
    %c40_i32 = arith.constant 40 : i32
    %c41_i32 = arith.constant 41 : i32
    %c42_i32 = arith.constant 42 : i32
    %c43_i32 = arith.constant 43 : i32
    %c44_i32 = arith.constant 44 : i32
    %c45_i32 = arith.constant 45 : i32
    %c46_i32 = arith.constant 46 : i32
    %c47_i32 = arith.constant 47 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_63 = arith.constant 8.000000e-02 : f32
    %cst_64 = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c192_i32 = arith.constant 192 : i32
    %c256_i32 = arith.constant 256 : i32
    %c320_i32 = arith.constant 320 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c80_i32 = arith.constant 80 : i32
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c-2_i32 = arith.constant -2 : i32
    memref.global @au_generated_candidate_1__l0u_buffer_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_1__l0u_buffer_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_0__direct_buffer_1 : memref<1xf32> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_buffer_1"}
    memref.global @au_generated_candidate_0__direct_buffer_0 : memref<1xi32> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_buffer_0"}
    memref.global @au_generated_candidate_0__direct_buffer_0_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_0__direct_buffer_1_0 : memref<48xf32> = uninitialized
    memref.global "private" @au_generated_candidate_0__direct_csr0 : memref<i32> = uninitialized {csr_address = 2049 : i32}
    memref.global "private" @au_generated_candidate_0__direct_csr1 : memref<i32> = uninitialized {csr_address = 2050 : i32}
    memref.global "private" @au_generated_candidate_0__direct_csr2 : memref<i32> = uninitialized {csr_address = 2051 : i32}
    memref.global "private" @au_generated_candidate_0__direct_csr3 : memref<i32> = uninitialized {csr_address = 2052 : i32}
    memref.global "private" @au_generated_candidate_0__direct_csr4 : memref<i32> = uninitialized {csr_address = 2053 : i32}
    tor.func @au_generated_candidate_0__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 0 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/ransac/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_1", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_2", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_0__direct_buffer_0_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_0__direct_buffer_1_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 13 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 1.350000e+02 : f64, starttime = 1 : i32, tl_channel = 0 : i32, transaction_group = 0 : i64, transaction_id = 0 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_3", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 0, 0>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 21 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 21 : i32, ref_starttime = 2 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 1.350000e+02 : f64, starttime = 2 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 1 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %5 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_4", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 1, 0, 1>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 31 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 31 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 1.350000e+02 : f64, starttime = 16 : i32, tl_channel = 0 : i32, transaction_group = 0 : i64, transaction_id = 2 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %6 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_5", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%2, %3)[%c48_i32], %c16_i32 {after = array<i64: 2, 1, 2>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 39 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 39 : i32, ref_starttime = 24 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 1.350000e+02 : f64, starttime = 24 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 3 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %7 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_6", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%2, %3)[%c64_i32], %c16_i32 {after = array<i64: 3, 2, 3>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 47 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 47 : i32, ref_starttime = 32 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 1.350000e+02 : f64, starttime = 32 : i32, tl_channel = 0 : i32, transaction_group = 0 : i64, transaction_id = 4 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %8 = tor.addi %0 %c320_i32 on (0 to 1) {dump = "unknown_7", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %8, (%2, %3)[%c80_i32], %c16_i32 {after = array<i64: 4, 3, 4>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 55 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 55 : i32, ref_starttime = 40 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 1.350000e+02 : f64, starttime = 40 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 5 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %9 = arith.bitcast %1 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32 to f32
      %10 = aps.read_csr @au_generated_candidate_0__direct_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %11 = arith.bitcast %10 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32 to f32
      %12 = aps.read_csr @au_generated_candidate_0__direct_csr1 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %13 = arith.bitcast %12 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32 to f32
      %14 = aps.read_csr @au_generated_candidate_0__direct_csr2 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %15 = arith.bitcast %14 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32 to f32
      %16 = aps.read_csr @au_generated_candidate_0__direct_csr3 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %17 = aps.load_by @cpuitfc, %16 {after = array<i64>, after_gap = array<i64>, endtime = 2 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 2 : i32, ref_starttime = 0 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 1.350000e+02 : f64, starttime = 0 : i32, transaction_id = 6 : i64, transfer_bytes = 4 : i64} : i32 -> i32
      aps.globalstore %17, @au_generated_candidate_0__direct_buffer_0 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
      %18 = aps.read_csr @au_generated_candidate_0__direct_csr4 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %19 = aps.load_by @cpuitfc, %18 {after = array<i64>, after_gap = array<i64>, endtime = 5 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 5 : i32, ref_starttime = 3 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 1.350000e+02 : f64, starttime = 3 : i32, transaction_id = 7 : i64, transfer_bytes = 4 : i64} : i32 -> f32
      aps.globalstore %19, @au_generated_candidate_0__direct_buffer_1 {endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, starttime = 5 : i32} : f32
      tor.for %arg3 = (%c0_i6 : i6) to (%c31_i6 : i6) step (%c1_i6 : i6)
      on (57 to 93){
        %22 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %23 = tor.muli %22 %c3_i32 on (57 to 58) {dump = "unknown_8", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %24 = tor.cmpi slt %23 %c0_i32 on (57 to 58) {dump = "unknown_9", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %25 = tor.subi %c-1_i32 %23 on (57 to 58) {dump = "unknown_10", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %26 = arith.select %24, %25, %23 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %27 = tor.cmpi slt %26 %c0_i32 on (57 to 58) {dump = "unknown_11", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %28 = arith.extui %27 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %29 = tor.addi %26 %28 on (57 to 58) {dump = "unknown_12", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %30 = arith.shrsi %29, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %31 = tor.subi %c-1_i32 %30 on (57 to 58) {dump = "unknown_13", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %32 = arith.select %24, %31, %30 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %33 = aps.read_smem %2[%32] {aps.mem_id = 5 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : memref<48xf32>, i32 -> f32
        %34 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %35 = tor.muli %34 %c3_i32 on (57 to 58) {dump = "unknown_14", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %36 = tor.cmpi slt %35 %c0_i32 on (57 to 58) {dump = "unknown_15", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %37 = tor.subi %c-1_i32 %35 on (57 to 58) {dump = "unknown_16", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %38 = arith.select %36, %37, %35 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %39 = tor.cmpi slt %38 %c0_i32 on (57 to 58) {dump = "unknown_17", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %40 = arith.extui %39 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %41 = tor.addi %38 %40 on (57 to 58) {dump = "unknown_18", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %42 = arith.shrsi %41, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %43 = tor.subi %c-1_i32 %42 on (57 to 58) {dump = "unknown_19", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %44 = arith.select %36, %43, %42 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %45 = aps.read_smem %3[%44] {aps.mem_id = 4 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : memref<48xf32>, i32 -> f32
        %46 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %47 = arith.andi %46, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %48 = tor.cmpi eq %47 %c0_i32 on (57 to 58) {dump = "unknown_20", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %49 = arith.select %48, %33, %45 {endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, starttime = 58 : i32} : f32
        %50 = tor.subf %49 %9 on (58 to 61) {dump = "unknown_21", fastmath = #arith.fastmath<none>, ref_endtime = 61 : i32, ref_starttime = 58 : i32} : f32
        %51 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %52 = tor.muli %51 %c3_i32 on (57 to 58) {dump = "unknown_22", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %53 = tor.addi %52 %c1_i32 on (57 to 58) {dump = "unknown_23", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %54 = tor.cmpi slt %53 %c0_i32 on (57 to 58) {dump = "unknown_24", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %55 = tor.subi %c-2_i32 %52 on (57 to 58) {dump = "unknown_25", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %56 = arith.select %54, %55, %53 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %57 = tor.cmpi slt %56 %c0_i32 on (57 to 58) {dump = "unknown_26", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %58 = arith.extui %57 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %59 = tor.addi %56 %58 on (57 to 58) {dump = "unknown_27", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %60 = arith.shrsi %59, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %61 = tor.subi %c-1_i32 %60 on (57 to 58) {dump = "unknown_28", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %62 = arith.select %54, %61, %60 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %63 = aps.read_smem %2[%62] {aps.mem_id = 3 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : memref<48xf32>, i32 -> f32
        %64 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %65 = tor.muli %64 %c3_i32 on (57 to 58) {dump = "unknown_29", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %66 = tor.addi %65 %c1_i32 on (57 to 58) {dump = "unknown_30", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %67 = tor.cmpi slt %66 %c0_i32 on (57 to 58) {dump = "unknown_31", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %68 = tor.subi %c-2_i32 %65 on (57 to 58) {dump = "unknown_32", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %69 = arith.select %67, %68, %66 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %70 = tor.cmpi slt %69 %c0_i32 on (57 to 58) {dump = "unknown_33", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %71 = arith.extui %70 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %72 = tor.addi %69 %71 on (57 to 58) {dump = "unknown_34", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %73 = arith.shrsi %72, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %74 = tor.subi %c-1_i32 %73 on (57 to 58) {dump = "unknown_35", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %75 = arith.select %67, %74, %73 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %76 = aps.read_smem %3[%75] {aps.mem_id = 2 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : memref<48xf32>, i32 -> f32
        %77 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %78 = tor.addi %77 %c1_i32 on (57 to 58) {dump = "unknown_36", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %79 = tor.cmpi slt %78 %c0_i32 on (57 to 58) {dump = "unknown_37", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %80 = tor.subi %c-2_i32 %77 on (57 to 58) {dump = "unknown_38", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %81 = arith.select %79, %80, %78 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %82 = tor.cmpi slt %81 %c0_i32 on (57 to 58) {dump = "unknown_39", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %83 = arith.extui %82 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %84 = tor.addi %81 %83 on (57 to 58) {dump = "unknown_40", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %85 = arith.shrsi %84, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %86 = tor.subi %c-1_i32 %85 on (57 to 58) {dump = "unknown_41", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %87 = arith.select %79, %86, %85 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %88 = tor.muli %87 %c-2_i32 on (57 to 58) {dump = "unknown_42", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %89 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %90 = tor.addi %89 %88 on (57 to 58) {dump = "unknown_43", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %91 = tor.addi %90 %c1_i32 on (57 to 58) {dump = "unknown_44", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %92 = tor.cmpi eq %91 %c0_i32 on (57 to 58) {dump = "unknown_45", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %93 = arith.select %92, %63, %76 {endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, starttime = 59 : i32} : f32
        %94 = tor.subf %93 %11 on (59 to 62) {dump = "unknown_46", fastmath = #arith.fastmath<none>, ref_endtime = 62 : i32, ref_starttime = 59 : i32} : f32
        %95 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %96 = tor.muli %95 %c3_i32 on (57 to 58) {dump = "unknown_47", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %97 = tor.cmpi slt %96 %c0_i32 on (57 to 58) {dump = "unknown_48", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %98 = tor.subi %c-1_i32 %96 on (57 to 58) {dump = "unknown_49", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %99 = arith.select %97, %98, %96 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %100 = tor.cmpi slt %99 %c0_i32 on (57 to 58) {dump = "unknown_50", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %101 = arith.extui %100 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %102 = tor.addi %99 %101 on (57 to 58) {dump = "unknown_51", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %103 = arith.shrsi %102, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %104 = tor.subi %c-1_i32 %103 on (57 to 58) {dump = "unknown_52", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %105 = arith.select %97, %104, %103 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %106 = tor.addi %105 %c1_i32 on (57 to 58) {dump = "unknown_53", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %107 = aps.read_smem %2[%106] {aps.mem_id = 1 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : memref<48xf32>, i32 -> f32
        %108 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %109 = tor.muli %108 %c3_i32 on (57 to 58) {dump = "unknown_54", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %110 = tor.cmpi slt %109 %c0_i32 on (57 to 58) {dump = "unknown_55", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %111 = tor.subi %c-1_i32 %109 on (57 to 58) {dump = "unknown_56", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %112 = arith.select %110, %111, %109 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %113 = tor.cmpi slt %112 %c0_i32 on (57 to 58) {dump = "unknown_57", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %114 = arith.extui %113 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %115 = tor.addi %112 %114 on (57 to 58) {dump = "unknown_58", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %116 = arith.shrsi %115, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %117 = tor.subi %c-1_i32 %116 on (57 to 58) {dump = "unknown_59", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %118 = arith.select %110, %117, %116 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %119 = tor.addi %118 %c1_i32 on (57 to 58) {dump = "unknown_60", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %120 = aps.read_smem %3[%119] {aps.mem_id = 0 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : memref<48xf32>, i32 -> f32
        %121 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %122 = arith.andi %121, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %123 = tor.cmpi eq %122 %c0_i32 on (57 to 58) {dump = "unknown_61", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %124 = arith.select %123, %107, %120 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : f32
        %125 = tor.subf %124 %13 on (60 to 63) {dump = "unknown_62", fastmath = #arith.fastmath<none>, ref_endtime = 63 : i32, ref_starttime = 60 : i32} : f32
        %126 = tor.cmpf olt %50 %cst_64 on (61 to 62) {dump = "unknown_63", fastmath = #arith.fastmath<none>, ref_endtime = 62 : i32, ref_starttime = 61 : i32} : f32 -> i1
        %127 = tor.if %126 on (64 to 65) -> (f32) then {
          %140 = arith.negf %50 {endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, starttime = 64 : i32} : f32
          tor.yield %140 : f32
        } else {
          tor.yield %50 : f32
        } {dump = "unknown_64", ref_endtime = 65 : i32, ref_starttime = 64 : i32}
        %128 = tor.cmpf olt %94 %cst_64 on (66 to 67) {dump = "unknown_65", fastmath = #arith.fastmath<none>, ref_endtime = 67 : i32, ref_starttime = 66 : i32} : f32 -> i1
        %129 = tor.if %128 on (68 to 69) -> (f32) then {
          %140 = arith.negf %94 {endtime = 69 : i32, ref_endtime = 69 : i32, ref_starttime = 68 : i32, starttime = 68 : i32} : f32
          tor.yield %140 : f32
        } else {
          tor.yield %94 : f32
        } {dump = "unknown_66", ref_endtime = 69 : i32, ref_starttime = 68 : i32}
        %130 = tor.addf %127 %129 on (70 to 73) {dump = "unknown_67", fastmath = #arith.fastmath<none>, ref_endtime = 73 : i32, ref_starttime = 70 : i32} : f32
        %131 = tor.cmpf olt %125 %cst_64 on (70 to 71) {dump = "unknown_68", fastmath = #arith.fastmath<none>, ref_endtime = 71 : i32, ref_starttime = 70 : i32} : f32 -> i1
        %132 = tor.if %131 on (74 to 75) -> (f32) then {
          %140 = arith.negf %125 {endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, starttime = 74 : i32} : f32
          tor.yield %140 : f32
        } else {
          tor.yield %125 : f32
        } {dump = "unknown_69", ref_endtime = 75 : i32, ref_starttime = 74 : i32}
        %133 = tor.addf %130 %132 on (76 to 79) {dump = "unknown_70", fastmath = #arith.fastmath<none>, ref_endtime = 79 : i32, ref_starttime = 76 : i32} : f32
        %134 = tor.subf %133 %15 on (79 to 82) {dump = "unknown_71", fastmath = #arith.fastmath<none>, ref_endtime = 82 : i32, ref_starttime = 79 : i32} : f32
        %135 = tor.cmpf olt %134 %cst_64 on (82 to 83) {dump = "unknown_72", fastmath = #arith.fastmath<none>, ref_endtime = 83 : i32, ref_starttime = 82 : i32} : f32 -> i1
        %136 = tor.if %135 on (84 to 85) -> (f32) then {
          %140 = arith.negf %134 {endtime = 85 : i32, ref_endtime = 85 : i32, ref_starttime = 84 : i32, starttime = 84 : i32} : f32
          tor.yield %140 : f32
        } else {
          tor.yield %134 : f32
        } {dump = "unknown_73", ref_endtime = 85 : i32, ref_starttime = 84 : i32}
        %137 = tor.cmpf olt %136 %cst_63 on (86 to 87) {dump = "unknown_74", fastmath = #arith.fastmath<none>, ref_endtime = 87 : i32, ref_starttime = 86 : i32} : f32 -> i1
        tor.if %137 on (88 to 89) then {
          %140 = aps.globalload @au_generated_candidate_0__direct_buffer_0 {endtime = 89 : i32, ref_endtime = 89 : i32, ref_starttime = 88 : i32, starttime = 88 : i32} : i32
          %141 = tor.addi %140 %c1_i32 on (88 to 89) {dump = "unknown_76", ref_endtime = 89 : i32, ref_starttime = 88 : i32} : (i32, i32) -> i32
          aps.globalstore %141, @au_generated_candidate_0__direct_buffer_0 {endtime = 89 : i32, ref_endtime = 89 : i32, ref_starttime = 88 : i32, starttime = 88 : i32} : i32
        } {dump = "unknown_75", ref_endtime = 89 : i32, ref_starttime = 88 : i32}
        %138 = aps.globalload @au_generated_candidate_0__direct_buffer_1 {endtime = 90 : i32, ref_endtime = 90 : i32, ref_starttime = 89 : i32, starttime = 89 : i32} : f32
        %139 = tor.addf %138 %136 on (89 to 92) {dump = "unknown_77", fastmath = #arith.fastmath<none>, ref_endtime = 92 : i32, ref_starttime = 89 : i32} : f32
        aps.globalstore %139, @au_generated_candidate_0__direct_buffer_1 {endtime = 93 : i32, ref_endtime = 93 : i32, ref_starttime = 92 : i32, starttime = 92 : i32} : f32
      } {ref_endtime = 93 : i32, ref_starttime = 57 : i32}
      %20 = aps.globalload @au_generated_candidate_0__direct_buffer_0 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
      aps.store_by @cpuitfc, %20, %16 {after = array<i64>, after_gap = array<i64>, endtime = 59 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 59 : i32, ref_starttime = 57 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 3.700000e+01 : f64, starttime = 57 : i32, transaction_id = 8 : i64, transfer_bytes = 4 : i64} : i32, i32
      %21 = aps.globalload @au_generated_candidate_0__direct_buffer_1 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : f32
      aps.store_by @cpuitfc, %21, %18 {after = array<i64>, after_gap = array<i64>, endtime = 61 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 61 : i32, ref_starttime = 59 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 3.700000e+01 : f64, starttime = 59 : i32, transaction_id = 9 : i64, transfer_bytes = 4 : i64} : f32, i32
      tor.return
    }
    tor.func @au_generated_candidate_1__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 1 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/ransac/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_78", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = memref.get_global @au_generated_candidate_1__l0u_buffer_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %2 = memref.get_global @au_generated_candidate_1__l0u_buffer_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%1, %2)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 10 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %3 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_79", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %3, (%1, %2)[%c16_i32], %c16_i32 {after = array<i64: 10, 10>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 20 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 20 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 11 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %4 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_80", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%1, %2)[%c32_i32], %c16_i32 {after = array<i64: 11, 10, 11>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 30 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 30 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 12 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %5 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_81", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%1, %2)[%c48_i32], %c16_i32 {after = array<i64: 12, 11, 12>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 38 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 38 : i32, ref_starttime = 23 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 23 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 13 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %6 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_82", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%1, %2)[%c64_i32], %c16_i32 {after = array<i64: 13, 12, 13>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 46 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 46 : i32, ref_starttime = 31 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 31 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 14 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %7 = tor.addi %0 %c320_i32 on (0 to 1) {dump = "unknown_83", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%1, %2)[%c80_i32], %c16_i32 {after = array<i64: 14, 13, 14>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 54 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 54 : i32, ref_starttime = 39 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 39 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 15 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      aps.write_smem %cst, %1[%c0_i32] {aps.mem_id = 95 : i64, endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, slot = 0 : i32, starttime = 54 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c0_i32] {aps.mem_id = 94 : i64, endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, slot = 0 : i32, starttime = 54 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_1, %1[%c1_i32] {aps.mem_id = 93 : i64, endtime = 56 : i32, ref_endtime = 56 : i32, ref_starttime = 55 : i32, slot = 0 : i32, starttime = 55 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_2, %2[%c1_i32] {aps.mem_id = 92 : i64, endtime = 56 : i32, ref_endtime = 56 : i32, ref_starttime = 55 : i32, slot = 0 : i32, starttime = 55 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c2_i32] {aps.mem_id = 91 : i64, endtime = 57 : i32, ref_endtime = 57 : i32, ref_starttime = 56 : i32, slot = 0 : i32, starttime = 56 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_3, %2[%c2_i32] {aps.mem_id = 90 : i64, endtime = 57 : i32, ref_endtime = 57 : i32, ref_starttime = 56 : i32, slot = 0 : i32, starttime = 56 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_4, %1[%c3_i32] {aps.mem_id = 89 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c3_i32] {aps.mem_id = 88 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_5, %1[%c4_i32] {aps.mem_id = 87 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_6, %2[%c4_i32] {aps.mem_id = 86 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c5_i32] {aps.mem_id = 85 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_7, %2[%c5_i32] {aps.mem_id = 84 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_8, %1[%c6_i32] {aps.mem_id = 83 : i64, endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, slot = 0 : i32, starttime = 60 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c6_i32] {aps.mem_id = 82 : i64, endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, slot = 0 : i32, starttime = 60 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_9, %1[%c7_i32] {aps.mem_id = 81 : i64, endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, slot = 0 : i32, starttime = 61 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_10, %2[%c7_i32] {aps.mem_id = 80 : i64, endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, slot = 0 : i32, starttime = 61 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c8_i32] {aps.mem_id = 79 : i64, endtime = 63 : i32, ref_endtime = 63 : i32, ref_starttime = 62 : i32, slot = 0 : i32, starttime = 62 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_11, %2[%c8_i32] {aps.mem_id = 78 : i64, endtime = 63 : i32, ref_endtime = 63 : i32, ref_starttime = 62 : i32, slot = 0 : i32, starttime = 62 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_12, %1[%c9_i32] {aps.mem_id = 77 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c9_i32] {aps.mem_id = 76 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_13, %1[%c10_i32] {aps.mem_id = 75 : i64, endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, slot = 0 : i32, starttime = 64 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_14, %2[%c10_i32] {aps.mem_id = 74 : i64, endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, slot = 0 : i32, starttime = 64 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c11_i32] {aps.mem_id = 73 : i64, endtime = 66 : i32, ref_endtime = 66 : i32, ref_starttime = 65 : i32, slot = 0 : i32, starttime = 65 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_15, %2[%c11_i32] {aps.mem_id = 72 : i64, endtime = 66 : i32, ref_endtime = 66 : i32, ref_starttime = 65 : i32, slot = 0 : i32, starttime = 65 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_16, %1[%c12_i32] {aps.mem_id = 71 : i64, endtime = 67 : i32, ref_endtime = 67 : i32, ref_starttime = 66 : i32, slot = 0 : i32, starttime = 66 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c12_i32] {aps.mem_id = 70 : i64, endtime = 67 : i32, ref_endtime = 67 : i32, ref_starttime = 66 : i32, slot = 0 : i32, starttime = 66 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_17, %1[%c13_i32] {aps.mem_id = 69 : i64, endtime = 68 : i32, ref_endtime = 68 : i32, ref_starttime = 67 : i32, slot = 0 : i32, starttime = 67 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_18, %2[%c13_i32] {aps.mem_id = 68 : i64, endtime = 68 : i32, ref_endtime = 68 : i32, ref_starttime = 67 : i32, slot = 0 : i32, starttime = 67 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c14_i32] {aps.mem_id = 67 : i64, endtime = 69 : i32, ref_endtime = 69 : i32, ref_starttime = 68 : i32, slot = 0 : i32, starttime = 68 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_19, %2[%c14_i32] {aps.mem_id = 66 : i64, endtime = 69 : i32, ref_endtime = 69 : i32, ref_starttime = 68 : i32, slot = 0 : i32, starttime = 68 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_20, %1[%c15_i32] {aps.mem_id = 65 : i64, endtime = 70 : i32, ref_endtime = 70 : i32, ref_starttime = 69 : i32, slot = 0 : i32, starttime = 69 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c15_i32] {aps.mem_id = 64 : i64, endtime = 70 : i32, ref_endtime = 70 : i32, ref_starttime = 69 : i32, slot = 0 : i32, starttime = 69 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_21, %1[%c16_i32] {aps.mem_id = 63 : i64, endtime = 71 : i32, ref_endtime = 71 : i32, ref_starttime = 70 : i32, slot = 0 : i32, starttime = 70 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_22, %2[%c16_i32] {aps.mem_id = 62 : i64, endtime = 71 : i32, ref_endtime = 71 : i32, ref_starttime = 70 : i32, slot = 0 : i32, starttime = 70 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c17_i32] {aps.mem_id = 61 : i64, endtime = 72 : i32, ref_endtime = 72 : i32, ref_starttime = 71 : i32, slot = 0 : i32, starttime = 71 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_23, %2[%c17_i32] {aps.mem_id = 60 : i64, endtime = 72 : i32, ref_endtime = 72 : i32, ref_starttime = 71 : i32, slot = 0 : i32, starttime = 71 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_24, %1[%c18_i32] {aps.mem_id = 59 : i64, endtime = 73 : i32, ref_endtime = 73 : i32, ref_starttime = 72 : i32, slot = 0 : i32, starttime = 72 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c18_i32] {aps.mem_id = 58 : i64, endtime = 73 : i32, ref_endtime = 73 : i32, ref_starttime = 72 : i32, slot = 0 : i32, starttime = 72 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_25, %1[%c19_i32] {aps.mem_id = 57 : i64, endtime = 74 : i32, ref_endtime = 74 : i32, ref_starttime = 73 : i32, slot = 0 : i32, starttime = 73 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_26, %2[%c19_i32] {aps.mem_id = 56 : i64, endtime = 74 : i32, ref_endtime = 74 : i32, ref_starttime = 73 : i32, slot = 0 : i32, starttime = 73 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c20_i32] {aps.mem_id = 55 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_27, %2[%c20_i32] {aps.mem_id = 54 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_28, %1[%c21_i32] {aps.mem_id = 53 : i64, endtime = 76 : i32, ref_endtime = 76 : i32, ref_starttime = 75 : i32, slot = 0 : i32, starttime = 75 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c21_i32] {aps.mem_id = 52 : i64, endtime = 76 : i32, ref_endtime = 76 : i32, ref_starttime = 75 : i32, slot = 0 : i32, starttime = 75 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_29, %1[%c22_i32] {aps.mem_id = 51 : i64, endtime = 77 : i32, ref_endtime = 77 : i32, ref_starttime = 76 : i32, slot = 0 : i32, starttime = 76 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_30, %2[%c22_i32] {aps.mem_id = 50 : i64, endtime = 77 : i32, ref_endtime = 77 : i32, ref_starttime = 76 : i32, slot = 0 : i32, starttime = 76 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c23_i32] {aps.mem_id = 49 : i64, endtime = 78 : i32, ref_endtime = 78 : i32, ref_starttime = 77 : i32, slot = 0 : i32, starttime = 77 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_31, %2[%c23_i32] {aps.mem_id = 48 : i64, endtime = 78 : i32, ref_endtime = 78 : i32, ref_starttime = 77 : i32, slot = 0 : i32, starttime = 77 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_32, %1[%c24_i32] {aps.mem_id = 47 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c24_i32] {aps.mem_id = 46 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_33, %1[%c25_i32] {aps.mem_id = 45 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_34, %2[%c25_i32] {aps.mem_id = 44 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c26_i32] {aps.mem_id = 43 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_35, %2[%c26_i32] {aps.mem_id = 42 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_36, %1[%c27_i32] {aps.mem_id = 41 : i64, endtime = 82 : i32, ref_endtime = 82 : i32, ref_starttime = 81 : i32, slot = 0 : i32, starttime = 81 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c27_i32] {aps.mem_id = 40 : i64, endtime = 82 : i32, ref_endtime = 82 : i32, ref_starttime = 81 : i32, slot = 0 : i32, starttime = 81 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_37, %1[%c28_i32] {aps.mem_id = 39 : i64, endtime = 83 : i32, ref_endtime = 83 : i32, ref_starttime = 82 : i32, slot = 0 : i32, starttime = 82 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_38, %2[%c28_i32] {aps.mem_id = 38 : i64, endtime = 83 : i32, ref_endtime = 83 : i32, ref_starttime = 82 : i32, slot = 0 : i32, starttime = 82 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c29_i32] {aps.mem_id = 37 : i64, endtime = 84 : i32, ref_endtime = 84 : i32, ref_starttime = 83 : i32, slot = 0 : i32, starttime = 83 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_39, %2[%c29_i32] {aps.mem_id = 36 : i64, endtime = 84 : i32, ref_endtime = 84 : i32, ref_starttime = 83 : i32, slot = 0 : i32, starttime = 83 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_40, %1[%c30_i32] {aps.mem_id = 35 : i64, endtime = 85 : i32, ref_endtime = 85 : i32, ref_starttime = 84 : i32, slot = 0 : i32, starttime = 84 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c30_i32] {aps.mem_id = 34 : i64, endtime = 85 : i32, ref_endtime = 85 : i32, ref_starttime = 84 : i32, slot = 0 : i32, starttime = 84 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_41, %1[%c31_i32] {aps.mem_id = 33 : i64, endtime = 86 : i32, ref_endtime = 86 : i32, ref_starttime = 85 : i32, slot = 0 : i32, starttime = 85 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_42, %2[%c31_i32] {aps.mem_id = 32 : i64, endtime = 86 : i32, ref_endtime = 86 : i32, ref_starttime = 85 : i32, slot = 0 : i32, starttime = 85 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c32_i32] {aps.mem_id = 31 : i64, endtime = 87 : i32, ref_endtime = 87 : i32, ref_starttime = 86 : i32, slot = 0 : i32, starttime = 86 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_43, %2[%c32_i32] {aps.mem_id = 30 : i64, endtime = 87 : i32, ref_endtime = 87 : i32, ref_starttime = 86 : i32, slot = 0 : i32, starttime = 86 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_44, %1[%c33_i32] {aps.mem_id = 29 : i64, endtime = 88 : i32, ref_endtime = 88 : i32, ref_starttime = 87 : i32, slot = 0 : i32, starttime = 87 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c33_i32] {aps.mem_id = 28 : i64, endtime = 88 : i32, ref_endtime = 88 : i32, ref_starttime = 87 : i32, slot = 0 : i32, starttime = 87 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_45, %1[%c34_i32] {aps.mem_id = 27 : i64, endtime = 89 : i32, ref_endtime = 89 : i32, ref_starttime = 88 : i32, slot = 0 : i32, starttime = 88 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_46, %2[%c34_i32] {aps.mem_id = 26 : i64, endtime = 89 : i32, ref_endtime = 89 : i32, ref_starttime = 88 : i32, slot = 0 : i32, starttime = 88 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c35_i32] {aps.mem_id = 25 : i64, endtime = 90 : i32, ref_endtime = 90 : i32, ref_starttime = 89 : i32, slot = 0 : i32, starttime = 89 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_47, %2[%c35_i32] {aps.mem_id = 24 : i64, endtime = 90 : i32, ref_endtime = 90 : i32, ref_starttime = 89 : i32, slot = 0 : i32, starttime = 89 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_48, %1[%c36_i32] {aps.mem_id = 23 : i64, endtime = 91 : i32, ref_endtime = 91 : i32, ref_starttime = 90 : i32, slot = 0 : i32, starttime = 90 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c36_i32] {aps.mem_id = 22 : i64, endtime = 91 : i32, ref_endtime = 91 : i32, ref_starttime = 90 : i32, slot = 0 : i32, starttime = 90 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_49, %1[%c37_i32] {aps.mem_id = 21 : i64, endtime = 92 : i32, ref_endtime = 92 : i32, ref_starttime = 91 : i32, slot = 0 : i32, starttime = 91 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_50, %2[%c37_i32] {aps.mem_id = 20 : i64, endtime = 92 : i32, ref_endtime = 92 : i32, ref_starttime = 91 : i32, slot = 0 : i32, starttime = 91 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c38_i32] {aps.mem_id = 19 : i64, endtime = 93 : i32, ref_endtime = 93 : i32, ref_starttime = 92 : i32, slot = 0 : i32, starttime = 92 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_51, %2[%c38_i32] {aps.mem_id = 18 : i64, endtime = 93 : i32, ref_endtime = 93 : i32, ref_starttime = 92 : i32, slot = 0 : i32, starttime = 92 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_52, %1[%c39_i32] {aps.mem_id = 17 : i64, endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, slot = 0 : i32, starttime = 93 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c39_i32] {aps.mem_id = 16 : i64, endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, slot = 0 : i32, starttime = 93 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_53, %1[%c40_i32] {aps.mem_id = 15 : i64, endtime = 95 : i32, ref_endtime = 95 : i32, ref_starttime = 94 : i32, slot = 0 : i32, starttime = 94 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_54, %2[%c40_i32] {aps.mem_id = 14 : i64, endtime = 95 : i32, ref_endtime = 95 : i32, ref_starttime = 94 : i32, slot = 0 : i32, starttime = 94 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c41_i32] {aps.mem_id = 13 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_55, %2[%c41_i32] {aps.mem_id = 12 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_56, %1[%c42_i32] {aps.mem_id = 11 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c42_i32] {aps.mem_id = 10 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_57, %1[%c43_i32] {aps.mem_id = 9 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_58, %2[%c43_i32] {aps.mem_id = 8 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c44_i32] {aps.mem_id = 7 : i64, endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, slot = 0 : i32, starttime = 98 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_59, %2[%c44_i32] {aps.mem_id = 6 : i64, endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, slot = 0 : i32, starttime = 98 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_60, %1[%c45_i32] {aps.mem_id = 5 : i64, endtime = 100 : i32, ref_endtime = 100 : i32, ref_starttime = 99 : i32, slot = 0 : i32, starttime = 99 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %2[%c45_i32] {aps.mem_id = 4 : i64, endtime = 100 : i32, ref_endtime = 100 : i32, ref_starttime = 99 : i32, slot = 0 : i32, starttime = 99 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_61, %1[%c46_i32] {aps.mem_id = 3 : i64, endtime = 101 : i32, ref_endtime = 101 : i32, ref_starttime = 100 : i32, slot = 0 : i32, starttime = 100 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_62, %2[%c46_i32] {aps.mem_id = 2 : i64, endtime = 101 : i32, ref_endtime = 101 : i32, ref_starttime = 100 : i32, slot = 0 : i32, starttime = 100 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %1[%c47_i32] {aps.mem_id = 1 : i64, endtime = 102 : i32, ref_endtime = 102 : i32, ref_starttime = 101 : i32, slot = 0 : i32, starttime = 101 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_1, %2[%c47_i32] {aps.mem_id = 0 : i64, endtime = 102 : i32, ref_endtime = 102 : i32, ref_starttime = 101 : i32, slot = 0 : i32, starttime = 101 : i32} : f32, memref<48xf32>, i32
      aps.copy_by @busitfc, (%1, %2)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 111 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 111 : i32, ref_starttime = 102 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 102 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 16 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %8 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_84", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c16_i32], %8, %c16_i32 {after = array<i64: 16, 16>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 121 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 121 : i32, ref_starttime = 103 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 103 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 17 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %9 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_85", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c32_i32], %9, %c16_i32 {after = array<i64: 17, 16, 17>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 136 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 136 : i32, ref_starttime = 117 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 117 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 18 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %10 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_86", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c48_i32], %10, %c16_i32 {after = array<i64: 18, 17, 18>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 146 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 20 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 146 : i32, ref_starttime = 127 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 127 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 19 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %11 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_87", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c64_i32], %11, %c16_i32 {after = array<i64: 19, 18, 19>, after_gap = array<i64: 10, 20, 10>, after_segment = 3 : i64, endtime = 156 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 49 : i64, memory_issue_cycle = 30 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 156 : i32, ref_starttime = 137 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 137 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 20 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %12 = tor.addi %0 %c320_i32 on (0 to 1) {dump = "unknown_88", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c80_i32], %12, %c16_i32 {after = array<i64: 20, 19, 20>, after_gap = array<i64: 10, 20, 10>, after_segment = 4 : i64, endtime = 166 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 59 : i64, memory_issue_cycle = 40 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 166 : i32, ref_starttime = 147 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 147 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 21 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      tor.return
    }
  } {HoistConstCondIfOp = 1 : i32, schedule = true}
}

