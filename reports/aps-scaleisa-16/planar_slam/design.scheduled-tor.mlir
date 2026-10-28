module {
  aps.memitfc @busitfc {cache_hint = "cold", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 8 : i64, min_beats = 1 : i64, min_write_beats = 2 : i64, read_latency = 5 : i64, width = 8 : i64, write_completion = 2 : i64}
  aps.memitfc @cpuitfc {cache_hint = "warm", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 1 : i64, min_beats = 1 : i64, min_write_beats = 1 : i64, read_latency = 2 : i64, width = 4 : i64, write_completion = 2 : i64}
  aps.memorymap {
    aps.mem_entry "au_generated_candidate_12__direct_buffer" : banks([@au_generated_candidate_12__direct_buffer_0_2, @au_generated_candidate_12__direct_buffer_1_2]), base(0), size(192), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_12__direct_buffer_0" : banks([@au_generated_candidate_12__direct_buffer_0_0, @au_generated_candidate_12__direct_buffer_0_1]), base(256), size(192), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_12__direct_buffer_1" : banks([@au_generated_candidate_12__direct_buffer_1_0, @au_generated_candidate_12__direct_buffer_1_1]), base(512), size(192), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_21__direct_buffer" : banks([@au_generated_candidate_21__direct_buffer_0, @au_generated_candidate_21__direct_buffer_1]), base(768), size(192), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_21__direct_buffer_2" : banks([@au_generated_candidate_21__direct_buffer_2]), base(1024), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_34__l0u_buffer" : banks([@au_generated_candidate_34__l0u_buffer_0, @au_generated_candidate_34__l0u_buffer_1]), base(1028), size(192), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_41__direct_buffer" : banks([@au_generated_candidate_41__direct_buffer_0, @au_generated_candidate_41__direct_buffer_1]), base(1280), size(160), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_41__direct_buffer_3" : banks([@au_generated_candidate_41__direct_buffer_3_0, @au_generated_candidate_41__direct_buffer_3_1]), base(1536), size(160), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_41__direct_buffer_4" : banks([@au_generated_candidate_41__direct_buffer_4_0, @au_generated_candidate_41__direct_buffer_4_1]), base(1792), size(160), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_51__direct_buffer" : banks([@au_generated_candidate_51__direct_buffer_0, @au_generated_candidate_51__direct_buffer_1]), base(2048), size(160), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_51__direct_buffer_5" : banks([@au_generated_candidate_51__direct_buffer_5]), base(2304), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_56__l0u_buffer" : banks([@au_generated_candidate_56__l0u_buffer_0, @au_generated_candidate_56__l0u_buffer_1]), base(2308), size(160), count(2), cyclic(1)
    aps.mem_finish
  }
  tor.design @aps_isaxes {
    %c-13_i5 = arith.constant -13 : i5
    %c15_i5 = arith.constant 15 : i5
    %c1_i5 = arith.constant 1 : i5
    %c0_i5 = arith.constant 0 : i5
    %c-2_i4 = arith.constant -2 : i4
    %c1_i4 = arith.constant 1 : i4
    %c0_i4 = arith.constant 0 : i4
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.980000e+00 : f32
    %cst_1 = arith.constant 2.36000013 : f32
    %cst_2 = arith.constant 2.740000e+00 : f32
    %cst_3 = arith.constant 3.120000e+00 : f32
    %cst_4 = arith.constant 1.470000e+00 : f32
    %cst_5 = arith.constant 1.740000e+00 : f32
    %cst_6 = arith.constant 2.01000023 : f32
    %cst_7 = arith.constant 1.810000e+02 : f32
    %cst_8 = arith.constant 1.510000e+02 : f32
    %cst_9 = arith.constant 2.500000e-01 : f32
    %cst_10 = arith.constant 0.000000e+00 : f32
    %cst_11 = arith.constant -3.000000e-02 : f32
    %cst_12 = arith.constant 8.000000e-01 : f32
    %cst_13 = arith.constant -1.500000e-02 : f32
    %cst_14 = arith.constant 1.600000e+00 : f32
    %cst_15 = arith.constant 2.250000e+00 : f32
    %cst_16 = arith.constant 1.500000e-02 : f32
    %cst_17 = arith.constant 3.050000e+00 : f32
    %cst_18 = arith.constant 3.000000e-02 : f32
    %cst_19 = arith.constant 3.850000e+00 : f32
    %cst_20 = arith.constant 4.500000e+00 : f32
    %cst_21 = arith.constant 5.300000e+00 : f32
    %cst_22 = arith.constant 6.100000e+00 : f32
    %cst_23 = arith.constant 6.750000e+00 : f32
    %cst_24 = arith.constant 7.550000e+00 : f32
    %cst_25 = arith.constant 8.350000e+00 : f32
    %cst_26 = arith.constant 9.000000e+00 : f32
    %cst_27 = arith.constant 9.800000e+00 : f32
    %cst_28 = arith.constant 1.060000e+01 : f32
    %cst_29 = arith.constant 1.125000e+01 : f32
    %c4_i32 = arith.constant 4 : i32
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
    %cst_30 = arith.constant 1.970000e+02 : f32
    %cst_31 = arith.constant 1.310000e+02 : f32
    %cst_32 = arith.constant 9.700000e+01 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_33 = arith.constant 1.800000e-01 : f32
    %cst_34 = arith.constant 2.200000e-01 : f32
    %c1_i32 = arith.constant {dump = "unknown_0"} 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %c3_i32 = arith.constant 3 : i32
    %c-4_i32 = arith.constant -4 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-2_i32 = arith.constant -2 : i32
    %c5_i32 = arith.constant 5 : i32
    %c-6_i32 = arith.constant -6 : i32
    memref.global @au_generated_candidate_56__l0u_buffer_0 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_56__l0u_buffer_1 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_51__direct_buffer_5 : memref<1xi32> {scaleisa.scratchpad, var_name = "au_generated_candidate_51__direct_buffer_5"}
    memref.global @au_generated_candidate_51__direct_buffer_0 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_51__direct_buffer_1 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_41__direct_buffer_4_0 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_41__direct_buffer_4_1 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_41__direct_buffer_3_0 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_41__direct_buffer_3_1 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_41__direct_buffer_0 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_41__direct_buffer_1 : memref<20xf32> = uninitialized
    memref.global @au_generated_candidate_34__l0u_buffer_0 : memref<24xf32> = uninitialized
    memref.global @au_generated_candidate_34__l0u_buffer_1 : memref<24xf32> = uninitialized
    memref.global @au_generated_candidate_21__direct_buffer_2 : memref<1xi32> {scaleisa.scratchpad, var_name = "au_generated_candidate_21__direct_buffer_2"}
    memref.global @au_generated_candidate_21__direct_buffer_0 : memref<24xf32> = uninitialized
    memref.global @au_generated_candidate_21__direct_buffer_1 : memref<24xf32> = uninitialized
    memref.global @au_generated_candidate_12__direct_buffer_1_0 : memref<24xf32> = uninitialized
    memref.global @au_generated_candidate_12__direct_buffer_1_1 : memref<24xf32> = uninitialized
    memref.global @au_generated_candidate_12__direct_buffer_0_0 : memref<24xf32> = uninitialized
    memref.global @au_generated_candidate_12__direct_buffer_0_1 : memref<24xf32> = uninitialized
    memref.global @au_generated_candidate_12__direct_buffer_0_2 : memref<24xf32> = uninitialized
    memref.global @au_generated_candidate_12__direct_buffer_1_2 : memref<24xf32> = uninitialized
    memref.global "private" @au_generated_candidate_12__direct_csr0 : memref<i32> = uninitialized {csr_address = 2049 : i32}
    memref.global "private" @au_generated_candidate_41__direct_csr0 : memref<i32> = uninitialized {csr_address = 2050 : i32}
    tor.func @au_generated_candidate_12__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 0 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/planar_slam/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_1", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_2", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_12__direct_buffer_0_2 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_12__direct_buffer_1_2 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 13 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 9 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 1.480000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 0 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_3", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 0, 0>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 35 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 9 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 35 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 1.480000e+02 : f64, starttime = 16 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 1 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %5 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_4", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 1, 0, 1>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 46 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 9 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 46 : i32, ref_starttime = 31 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 1.480000e+02 : f64, starttime = 31 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 2 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %6 = memref.get_global @au_generated_candidate_12__direct_buffer_0_0 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %7 = memref.get_global @au_generated_candidate_12__direct_buffer_0_1 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%6, %7)[%c0_i32], %c16_i32 {after = array<i64: 2, 1, 2>, after_gap = array<i64: 8, 20, 8>, endtime = 60 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 9 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 60 : i32, ref_starttime = 45 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 1.480000e+02 : f64, starttime = 45 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 3 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %8 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_5", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %8, (%6, %7)[%c16_i32], %c16_i32 {after = array<i64: 3, 2, 3>, after_gap = array<i64: 8, 16, 8>, after_segment = 0 : i64, endtime = 68 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 9 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 68 : i32, ref_starttime = 53 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 1.480000e+02 : f64, starttime = 53 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 4 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %9 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_6", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %9, (%6, %7)[%c32_i32], %c16_i32 {after = array<i64: 4, 3, 4>, after_gap = array<i64: 8, 16, 8>, after_segment = 1 : i64, endtime = 76 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 9 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 76 : i32, ref_starttime = 61 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 1.480000e+02 : f64, starttime = 61 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 5 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %10 = aps.read_csr @au_generated_candidate_12__direct_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %11 = memref.get_global @au_generated_candidate_12__direct_buffer_1_0 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %12 = memref.get_global @au_generated_candidate_12__direct_buffer_1_1 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %10, (%11, %12)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 1.480000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 6 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %13 = tor.addi %10 %c64_i32 on (0 to 1) {dump = "unknown_7", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %13, (%11, %12)[%c16_i32], %c16_i32 {after = array<i64: 6, 6>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 34 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 34 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 1.480000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 7 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %14 = tor.addi %10 %c128_i32 on (0 to 1) {dump = "unknown_8", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %14, (%11, %12)[%c32_i32], %c16_i32 {after = array<i64: 7, 6, 7>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 45 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 45 : i32, ref_starttime = 30 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 1.480000e+02 : f64, starttime = 30 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 8 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      tor.for %arg3 = (%c0_i4 : i4) to (%c-2_i4 : i4) step (%c1_i4 : i4)
      on (78 to 124){
        %17 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %18 = tor.muli %17 %c3_i32 on (78 to 79) {dump = "unknown_9", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %19 = tor.addi %18 %c3_i32 on (78 to 79) {dump = "unknown_10", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %20 = tor.cmpi slt %19 %c0_i32 on (78 to 79) {dump = "unknown_11", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %21 = tor.subi %c-4_i32 %18 on (78 to 79) {dump = "unknown_12", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %22 = arith.select %20, %21, %19 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %23 = tor.cmpi slt %22 %c0_i32 on (78 to 79) {dump = "unknown_13", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %24 = arith.extui %23 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %25 = tor.addi %22 %24 on (78 to 79) {dump = "unknown_14", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %26 = arith.shrsi %25, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %27 = tor.subi %c-1_i32 %26 on (78 to 79) {dump = "unknown_15", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %28 = arith.select %20, %27, %26 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %29 = aps.read_smem %2[%28] {aps.mem_id = 23 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : memref<24xf32>, i32 -> f32
        %30 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %31 = tor.muli %30 %c3_i32 on (78 to 79) {dump = "unknown_16", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %32 = tor.addi %31 %c3_i32 on (78 to 79) {dump = "unknown_17", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %33 = tor.cmpi slt %32 %c0_i32 on (78 to 79) {dump = "unknown_18", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %34 = tor.subi %c-4_i32 %31 on (78 to 79) {dump = "unknown_19", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %35 = arith.select %33, %34, %32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %36 = tor.cmpi slt %35 %c0_i32 on (78 to 79) {dump = "unknown_20", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %37 = arith.extui %36 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %38 = tor.addi %35 %37 on (78 to 79) {dump = "unknown_21", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %39 = arith.shrsi %38, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %40 = tor.subi %c-1_i32 %39 on (78 to 79) {dump = "unknown_22", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %41 = arith.select %33, %40, %39 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %42 = aps.read_smem %3[%41] {aps.mem_id = 22 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : memref<24xf32>, i32 -> f32
        %43 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %44 = tor.addi %43 %c1_i32 on (78 to 79) {dump = "unknown_23", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %45 = tor.cmpi slt %44 %c0_i32 on (78 to 79) {dump = "unknown_24", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %46 = tor.subi %c-2_i32 %43 on (78 to 79) {dump = "unknown_25", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %47 = arith.select %45, %46, %44 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %48 = tor.cmpi slt %47 %c0_i32 on (78 to 79) {dump = "unknown_26", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %49 = arith.extui %48 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %50 = tor.addi %47 %49 on (78 to 79) {dump = "unknown_27", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %51 = arith.shrsi %50, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %52 = tor.subi %c-1_i32 %51 on (78 to 79) {dump = "unknown_28", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %53 = arith.select %45, %52, %51 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %54 = tor.muli %53 %c-2_i32 on (78 to 79) {dump = "unknown_29", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %55 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %56 = tor.addi %55 %54 on (78 to 79) {dump = "unknown_30", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %57 = tor.addi %56 %c1_i32 on (78 to 79) {dump = "unknown_31", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %58 = tor.cmpi eq %57 %c0_i32 on (78 to 79) {dump = "unknown_32", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %59 = arith.select %58, %29, %42 {endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, starttime = 80 : i32} : f32
        %60 = tor.mulf %59 %cst_34 on (80 to 83) {dump = "unknown_33", fastmath = #arith.fastmath<none>, ref_endtime = 83 : i32, ref_starttime = 80 : i32} : f32
        %61 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %62 = tor.muli %61 %c3_i32 on (78 to 79) {dump = "unknown_34", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %63 = tor.addi %62 %c3_i32 on (78 to 79) {dump = "unknown_35", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %64 = tor.cmpi slt %63 %c0_i32 on (78 to 79) {dump = "unknown_36", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %65 = tor.subi %c-4_i32 %62 on (78 to 79) {dump = "unknown_37", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %66 = arith.select %64, %65, %63 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %67 = tor.cmpi slt %66 %c0_i32 on (78 to 79) {dump = "unknown_38", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %68 = arith.extui %67 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %69 = tor.addi %66 %68 on (78 to 79) {dump = "unknown_39", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %70 = arith.shrsi %69, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %71 = tor.subi %c-1_i32 %70 on (78 to 79) {dump = "unknown_40", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %72 = arith.select %64, %71, %70 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %73 = aps.read_smem %6[%72] {aps.mem_id = 21 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : memref<24xf32>, i32 -> f32
        %74 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %75 = tor.muli %74 %c3_i32 on (78 to 79) {dump = "unknown_41", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %76 = tor.addi %75 %c3_i32 on (78 to 79) {dump = "unknown_42", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %77 = tor.cmpi slt %76 %c0_i32 on (78 to 79) {dump = "unknown_43", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %78 = tor.subi %c-4_i32 %75 on (78 to 79) {dump = "unknown_44", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %79 = arith.select %77, %78, %76 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %80 = tor.cmpi slt %79 %c0_i32 on (78 to 79) {dump = "unknown_45", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %81 = arith.extui %80 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %82 = tor.addi %79 %81 on (78 to 79) {dump = "unknown_46", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %83 = arith.shrsi %82, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %84 = tor.subi %c-1_i32 %83 on (78 to 79) {dump = "unknown_47", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %85 = arith.select %77, %84, %83 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %86 = aps.read_smem %7[%85] {aps.mem_id = 20 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : memref<24xf32>, i32 -> f32
        %87 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %88 = tor.addi %87 %c1_i32 on (78 to 79) {dump = "unknown_48", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %89 = tor.cmpi slt %88 %c0_i32 on (78 to 79) {dump = "unknown_49", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %90 = tor.subi %c-2_i32 %87 on (78 to 79) {dump = "unknown_50", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %91 = arith.select %89, %90, %88 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %92 = tor.cmpi slt %91 %c0_i32 on (78 to 79) {dump = "unknown_51", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %93 = arith.extui %92 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %94 = tor.addi %91 %93 on (78 to 79) {dump = "unknown_52", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %95 = arith.shrsi %94, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %96 = tor.subi %c-1_i32 %95 on (78 to 79) {dump = "unknown_53", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %97 = arith.select %89, %96, %95 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %98 = tor.muli %97 %c-2_i32 on (78 to 79) {dump = "unknown_54", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %99 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %100 = tor.addi %99 %98 on (78 to 79) {dump = "unknown_55", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %101 = tor.addi %100 %c1_i32 on (78 to 79) {dump = "unknown_56", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %102 = tor.cmpi eq %101 %c0_i32 on (78 to 79) {dump = "unknown_57", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %103 = arith.select %102, %73, %86 {endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, starttime = 80 : i32} : f32
        %104 = tor.divf %60 %103 on (83 to 110) {dump = "unknown_58", fastmath = #arith.fastmath<none>, ref_endtime = 110 : i32, ref_starttime = 83 : i32} : f32
        %105 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %106 = tor.muli %105 %c3_i32 on (78 to 79) {dump = "unknown_59", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %107 = tor.addi %106 %c3_i32 on (78 to 79) {dump = "unknown_60", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %108 = tor.cmpi slt %107 %c0_i32 on (78 to 79) {dump = "unknown_61", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %109 = tor.subi %c-4_i32 %106 on (78 to 79) {dump = "unknown_62", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %110 = arith.select %108, %109, %107 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %111 = tor.cmpi slt %110 %c0_i32 on (78 to 79) {dump = "unknown_63", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %112 = arith.extui %111 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %113 = tor.addi %110 %112 on (78 to 79) {dump = "unknown_64", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %114 = arith.shrsi %113, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %115 = tor.subi %c-1_i32 %114 on (78 to 79) {dump = "unknown_65", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %116 = arith.select %108, %115, %114 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %117 = aps.read_smem %11[%116] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 17 : i64}], aps.mem_id = 19 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : memref<24xf32>, i32 -> f32
        %118 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %119 = tor.muli %118 %c3_i32 on (78 to 79) {dump = "unknown_66", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %120 = tor.addi %119 %c3_i32 on (78 to 79) {dump = "unknown_67", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %121 = tor.cmpi slt %120 %c0_i32 on (78 to 79) {dump = "unknown_68", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %122 = tor.subi %c-4_i32 %119 on (78 to 79) {dump = "unknown_69", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %123 = arith.select %121, %122, %120 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %124 = tor.cmpi slt %123 %c0_i32 on (78 to 79) {dump = "unknown_70", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %125 = arith.extui %124 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %126 = tor.addi %123 %125 on (78 to 79) {dump = "unknown_71", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %127 = arith.shrsi %126, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %128 = tor.subi %c-1_i32 %127 on (78 to 79) {dump = "unknown_72", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %129 = arith.select %121, %128, %127 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %130 = aps.read_smem %12[%129] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 16 : i64}], aps.mem_id = 18 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : memref<24xf32>, i32 -> f32
        %131 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %132 = tor.addi %131 %c1_i32 on (78 to 79) {dump = "unknown_73", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %133 = tor.cmpi slt %132 %c0_i32 on (78 to 79) {dump = "unknown_74", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %134 = tor.subi %c-2_i32 %131 on (78 to 79) {dump = "unknown_75", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %135 = arith.select %133, %134, %132 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %136 = tor.cmpi slt %135 %c0_i32 on (78 to 79) {dump = "unknown_76", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %137 = arith.extui %136 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %138 = tor.addi %135 %137 on (78 to 79) {dump = "unknown_77", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %139 = arith.shrsi %138, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %140 = tor.subi %c-1_i32 %139 on (78 to 79) {dump = "unknown_78", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %141 = arith.select %133, %140, %139 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %142 = tor.muli %141 %c-2_i32 on (78 to 79) {dump = "unknown_79", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %143 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %144 = tor.addi %143 %142 on (78 to 79) {dump = "unknown_80", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %145 = tor.addi %144 %c1_i32 on (78 to 79) {dump = "unknown_81", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %146 = tor.cmpi eq %145 %c0_i32 on (78 to 79) {dump = "unknown_82", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %147 = arith.select %146, %117, %130 {endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, starttime = 79 : i32} : f32
        %148 = tor.subf %147 %104 on (110 to 113) {dump = "unknown_83", fastmath = #arith.fastmath<none>, ref_endtime = 113 : i32, ref_starttime = 110 : i32} : f32
        %149 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %150 = tor.addi %149 %c1_i32 on (78 to 79) {dump = "unknown_84", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %151 = tor.cmpi slt %150 %c0_i32 on (78 to 79) {dump = "unknown_85", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %152 = tor.subi %c-2_i32 %149 on (78 to 79) {dump = "unknown_86", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %153 = arith.select %151, %152, %150 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %154 = tor.cmpi slt %153 %c0_i32 on (78 to 79) {dump = "unknown_87", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %155 = arith.extui %154 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %156 = tor.addi %153 %155 on (78 to 79) {dump = "unknown_88", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %157 = arith.shrsi %156, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %158 = tor.subi %c-1_i32 %157 on (78 to 79) {dump = "unknown_89", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %159 = arith.select %151, %158, %157 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %160 = tor.muli %159 %c-2_i32 on (78 to 79) {dump = "unknown_90", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %161 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %162 = tor.addi %161 %160 on (78 to 79) {dump = "unknown_91", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %163 = tor.addi %162 %c1_i32 on (78 to 79) {dump = "unknown_92", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %164 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %165 = tor.muli %164 %c3_i32 on (78 to 79) {dump = "unknown_93", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %166 = tor.addi %165 %c3_i32 on (78 to 79) {dump = "unknown_94", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %167 = tor.cmpi slt %166 %c0_i32 on (78 to 79) {dump = "unknown_95", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %168 = tor.subi %c-4_i32 %165 on (78 to 79) {dump = "unknown_96", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %169 = arith.select %167, %168, %166 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %170 = tor.cmpi slt %169 %c0_i32 on (78 to 79) {dump = "unknown_97", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %171 = arith.extui %170 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %172 = tor.addi %169 %171 on (78 to 79) {dump = "unknown_98", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %173 = arith.shrsi %172, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %174 = tor.subi %c-1_i32 %173 on (78 to 79) {dump = "unknown_99", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %175 = arith.select %167, %174, %173 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %176 = tor.cmpi eq %163 %c0_i32 on (78 to 79) {dump = "unknown_100", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        aps.write_smem_if %176, %148, %11[%175] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 3 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 11 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 19 : i64}], aps.mem_id = 17 : i64, endtime = 114 : i32, ref_endtime = 114 : i32, ref_starttime = 113 : i32, slot = 0 : i32, starttime = 113 : i32} : f32, memref<24xf32>, i32
        %177 = tor.cmpi eq %163 %c1_i32 on (78 to 79) {dump = "unknown_101", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        aps.write_smem_if %177, %148, %12[%175] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 2 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 10 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 18 : i64}], aps.mem_id = 16 : i64, endtime = 114 : i32, ref_endtime = 114 : i32, ref_starttime = 113 : i32, slot = 0 : i32, starttime = 113 : i32} : f32, memref<24xf32>, i32
        %178 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %179 = tor.muli %178 %c3_i32 on (78 to 79) {dump = "unknown_102", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %180 = tor.cmpi slt %179 %c0_i32 on (78 to 79) {dump = "unknown_103", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %181 = tor.subi %c-1_i32 %179 on (78 to 79) {dump = "unknown_104", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %182 = arith.select %180, %181, %179 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %183 = tor.cmpi slt %182 %c0_i32 on (78 to 79) {dump = "unknown_105", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %184 = arith.extui %183 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %185 = tor.addi %182 %184 on (78 to 79) {dump = "unknown_106", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %186 = arith.shrsi %185, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %187 = tor.subi %c-1_i32 %186 on (78 to 79) {dump = "unknown_107", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %188 = arith.select %180, %187, %186 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %189 = tor.addi %188 %c2_i32 on (78 to 79) {dump = "unknown_108", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %190 = aps.read_smem %2[%189] {aps.mem_id = 15 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : memref<24xf32>, i32 -> f32
        %191 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %192 = tor.muli %191 %c3_i32 on (78 to 79) {dump = "unknown_109", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %193 = tor.cmpi slt %192 %c0_i32 on (78 to 79) {dump = "unknown_110", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %194 = tor.subi %c-1_i32 %192 on (78 to 79) {dump = "unknown_111", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %195 = arith.select %193, %194, %192 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %196 = tor.cmpi slt %195 %c0_i32 on (78 to 79) {dump = "unknown_112", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %197 = arith.extui %196 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %198 = tor.addi %195 %197 on (78 to 79) {dump = "unknown_113", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %199 = arith.shrsi %198, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %200 = tor.subi %c-1_i32 %199 on (78 to 79) {dump = "unknown_114", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %201 = arith.select %193, %200, %199 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %202 = tor.addi %201 %c2_i32 on (78 to 79) {dump = "unknown_115", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %203 = aps.read_smem %3[%202] {aps.mem_id = 14 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : memref<24xf32>, i32 -> f32
        %204 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %205 = arith.andi %204, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %206 = tor.cmpi eq %205 %c0_i32 on (78 to 79) {dump = "unknown_116", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %207 = arith.select %206, %190, %203 {endtime = 82 : i32, ref_endtime = 82 : i32, ref_starttime = 81 : i32, starttime = 81 : i32} : f32
        %208 = tor.mulf %207 %cst_34 on (81 to 84) {dump = "unknown_117", fastmath = #arith.fastmath<none>, ref_endtime = 84 : i32, ref_starttime = 81 : i32} : f32
        %209 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %210 = tor.muli %209 %c3_i32 on (78 to 79) {dump = "unknown_118", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %211 = tor.cmpi slt %210 %c0_i32 on (78 to 79) {dump = "unknown_119", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %212 = tor.subi %c-1_i32 %210 on (78 to 79) {dump = "unknown_120", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %213 = arith.select %211, %212, %210 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %214 = tor.cmpi slt %213 %c0_i32 on (78 to 79) {dump = "unknown_121", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %215 = arith.extui %214 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %216 = tor.addi %213 %215 on (78 to 79) {dump = "unknown_122", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %217 = arith.shrsi %216, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %218 = tor.subi %c-1_i32 %217 on (78 to 79) {dump = "unknown_123", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %219 = arith.select %211, %218, %217 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %220 = tor.addi %219 %c2_i32 on (78 to 79) {dump = "unknown_124", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %221 = aps.read_smem %6[%220] {aps.mem_id = 13 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : memref<24xf32>, i32 -> f32
        %222 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %223 = tor.muli %222 %c3_i32 on (78 to 79) {dump = "unknown_125", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %224 = tor.cmpi slt %223 %c0_i32 on (78 to 79) {dump = "unknown_126", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %225 = tor.subi %c-1_i32 %223 on (78 to 79) {dump = "unknown_127", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %226 = arith.select %224, %225, %223 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %227 = tor.cmpi slt %226 %c0_i32 on (78 to 79) {dump = "unknown_128", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %228 = arith.extui %227 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %229 = tor.addi %226 %228 on (78 to 79) {dump = "unknown_129", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %230 = arith.shrsi %229, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %231 = tor.subi %c-1_i32 %230 on (78 to 79) {dump = "unknown_130", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %232 = arith.select %224, %231, %230 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %233 = tor.addi %232 %c2_i32 on (78 to 79) {dump = "unknown_131", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %234 = aps.read_smem %7[%233] {aps.mem_id = 12 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : memref<24xf32>, i32 -> f32
        %235 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %236 = arith.andi %235, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %237 = tor.cmpi eq %236 %c0_i32 on (78 to 79) {dump = "unknown_132", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %238 = arith.select %237, %221, %234 {endtime = 82 : i32, ref_endtime = 82 : i32, ref_starttime = 81 : i32, starttime = 81 : i32} : f32
        %239 = tor.divf %208 %238 on (84 to 111) {dump = "unknown_133", fastmath = #arith.fastmath<none>, ref_endtime = 111 : i32, ref_starttime = 84 : i32} : f32
        %240 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %241 = tor.muli %240 %c3_i32 on (78 to 79) {dump = "unknown_134", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %242 = tor.cmpi slt %241 %c0_i32 on (78 to 79) {dump = "unknown_135", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %243 = tor.subi %c-1_i32 %241 on (78 to 79) {dump = "unknown_136", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %244 = arith.select %242, %243, %241 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %245 = tor.cmpi slt %244 %c0_i32 on (78 to 79) {dump = "unknown_137", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %246 = arith.extui %245 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %247 = tor.addi %244 %246 on (78 to 79) {dump = "unknown_138", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %248 = arith.shrsi %247, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %249 = tor.subi %c-1_i32 %248 on (78 to 79) {dump = "unknown_139", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %250 = arith.select %242, %249, %248 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %251 = tor.addi %250 %c2_i32 on (78 to 79) {dump = "unknown_140", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %252 = aps.read_smem %11[%251] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 9 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 17 : i64}], aps.mem_id = 11 : i64, endtime = 115 : i32, ref_endtime = 115 : i32, ref_starttime = 114 : i32, slot = 0 : i32, starttime = 114 : i32} : memref<24xf32>, i32 -> f32
        %253 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %254 = tor.muli %253 %c3_i32 on (78 to 79) {dump = "unknown_141", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %255 = tor.cmpi slt %254 %c0_i32 on (78 to 79) {dump = "unknown_142", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %256 = tor.subi %c-1_i32 %254 on (78 to 79) {dump = "unknown_143", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %257 = arith.select %255, %256, %254 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %258 = tor.cmpi slt %257 %c0_i32 on (78 to 79) {dump = "unknown_144", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %259 = arith.extui %258 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %260 = tor.addi %257 %259 on (78 to 79) {dump = "unknown_145", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %261 = arith.shrsi %260, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %262 = tor.subi %c-1_i32 %261 on (78 to 79) {dump = "unknown_146", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %263 = arith.select %255, %262, %261 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %264 = tor.addi %263 %c2_i32 on (78 to 79) {dump = "unknown_147", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %265 = aps.read_smem %12[%264] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 8 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 16 : i64}], aps.mem_id = 10 : i64, endtime = 115 : i32, ref_endtime = 115 : i32, ref_starttime = 114 : i32, slot = 0 : i32, starttime = 114 : i32} : memref<24xf32>, i32 -> f32
        %266 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %267 = arith.andi %266, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %268 = tor.cmpi eq %267 %c0_i32 on (78 to 79) {dump = "unknown_148", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %269 = arith.select %268, %252, %265 {endtime = 116 : i32, ref_endtime = 116 : i32, ref_starttime = 115 : i32, starttime = 115 : i32} : f32
        %270 = tor.subf %269 %239 on (115 to 118) {dump = "unknown_149", fastmath = #arith.fastmath<none>, ref_endtime = 118 : i32, ref_starttime = 115 : i32} : f32
        %271 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %272 = arith.andi %271, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %273 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %274 = tor.muli %273 %c3_i32 on (78 to 79) {dump = "unknown_150", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %275 = tor.cmpi slt %274 %c0_i32 on (78 to 79) {dump = "unknown_151", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %276 = tor.subi %c-1_i32 %274 on (78 to 79) {dump = "unknown_152", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %277 = arith.select %275, %276, %274 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %278 = tor.cmpi slt %277 %c0_i32 on (78 to 79) {dump = "unknown_153", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %279 = arith.extui %278 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %280 = tor.addi %277 %279 on (78 to 79) {dump = "unknown_154", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %281 = arith.shrsi %280, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %282 = tor.subi %c-1_i32 %281 on (78 to 79) {dump = "unknown_155", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %283 = arith.select %275, %282, %281 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %284 = tor.addi %283 %c2_i32 on (78 to 79) {dump = "unknown_156", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %285 = tor.cmpi eq %272 %c0_i32 on (78 to 79) {dump = "unknown_157", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        aps.write_smem_if %285, %270, %11[%284] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 3 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 11 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 19 : i64}], aps.mem_id = 9 : i64, endtime = 119 : i32, ref_endtime = 119 : i32, ref_starttime = 118 : i32, slot = 0 : i32, starttime = 118 : i32} : f32, memref<24xf32>, i32
        %286 = tor.cmpi eq %272 %c1_i32 on (78 to 79) {dump = "unknown_158", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        aps.write_smem_if %286, %270, %12[%284] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 2 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 10 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 18 : i64}], aps.mem_id = 8 : i64, endtime = 119 : i32, ref_endtime = 119 : i32, ref_starttime = 118 : i32, slot = 0 : i32, starttime = 118 : i32} : f32, memref<24xf32>, i32
        %287 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %288 = tor.muli %287 %c3_i32 on (78 to 79) {dump = "unknown_159", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %289 = tor.addi %288 %c5_i32 on (78 to 79) {dump = "unknown_160", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %290 = tor.cmpi slt %289 %c0_i32 on (78 to 79) {dump = "unknown_161", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %291 = tor.subi %c-6_i32 %288 on (78 to 79) {dump = "unknown_162", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %292 = arith.select %290, %291, %289 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %293 = tor.cmpi slt %292 %c0_i32 on (78 to 79) {dump = "unknown_163", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %294 = arith.extui %293 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %295 = tor.addi %292 %294 on (78 to 79) {dump = "unknown_164", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %296 = arith.shrsi %295, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %297 = tor.subi %c-1_i32 %296 on (78 to 79) {dump = "unknown_165", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %298 = arith.select %290, %297, %296 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %299 = aps.read_smem %2[%298] {aps.mem_id = 7 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : memref<24xf32>, i32 -> f32
        %300 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %301 = tor.muli %300 %c3_i32 on (78 to 79) {dump = "unknown_166", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %302 = tor.addi %301 %c5_i32 on (78 to 79) {dump = "unknown_167", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %303 = tor.cmpi slt %302 %c0_i32 on (78 to 79) {dump = "unknown_168", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %304 = tor.subi %c-6_i32 %301 on (78 to 79) {dump = "unknown_169", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %305 = arith.select %303, %304, %302 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %306 = tor.cmpi slt %305 %c0_i32 on (78 to 79) {dump = "unknown_170", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %307 = arith.extui %306 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %308 = tor.addi %305 %307 on (78 to 79) {dump = "unknown_171", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %309 = arith.shrsi %308, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %310 = tor.subi %c-1_i32 %309 on (78 to 79) {dump = "unknown_172", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %311 = arith.select %303, %310, %309 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %312 = aps.read_smem %3[%311] {aps.mem_id = 6 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : memref<24xf32>, i32 -> f32
        %313 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %314 = tor.addi %313 %c1_i32 on (78 to 79) {dump = "unknown_173", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %315 = tor.cmpi slt %314 %c0_i32 on (78 to 79) {dump = "unknown_174", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %316 = tor.subi %c-2_i32 %313 on (78 to 79) {dump = "unknown_175", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %317 = arith.select %315, %316, %314 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %318 = tor.cmpi slt %317 %c0_i32 on (78 to 79) {dump = "unknown_176", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %319 = arith.extui %318 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %320 = tor.addi %317 %319 on (78 to 79) {dump = "unknown_177", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %321 = arith.shrsi %320, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %322 = tor.subi %c-1_i32 %321 on (78 to 79) {dump = "unknown_178", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %323 = arith.select %315, %322, %321 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %324 = tor.muli %323 %c-2_i32 on (78 to 79) {dump = "unknown_179", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %325 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %326 = tor.addi %325 %324 on (78 to 79) {dump = "unknown_180", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %327 = tor.addi %326 %c1_i32 on (78 to 79) {dump = "unknown_181", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %328 = tor.cmpi eq %327 %c0_i32 on (78 to 79) {dump = "unknown_182", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %329 = arith.select %328, %299, %312 {endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, starttime = 79 : i32} : f32
        %330 = tor.mulf %329 %cst_33 on (79 to 82) {dump = "unknown_183", fastmath = #arith.fastmath<none>, ref_endtime = 82 : i32, ref_starttime = 79 : i32} : f32
        %331 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %332 = tor.muli %331 %c3_i32 on (78 to 79) {dump = "unknown_184", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %333 = tor.addi %332 %c5_i32 on (78 to 79) {dump = "unknown_185", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %334 = tor.cmpi slt %333 %c0_i32 on (78 to 79) {dump = "unknown_186", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %335 = tor.subi %c-6_i32 %332 on (78 to 79) {dump = "unknown_187", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %336 = arith.select %334, %335, %333 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %337 = tor.cmpi slt %336 %c0_i32 on (78 to 79) {dump = "unknown_188", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %338 = arith.extui %337 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %339 = tor.addi %336 %338 on (78 to 79) {dump = "unknown_189", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %340 = arith.shrsi %339, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %341 = tor.subi %c-1_i32 %340 on (78 to 79) {dump = "unknown_190", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %342 = arith.select %334, %341, %340 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %343 = aps.read_smem %6[%342] {aps.mem_id = 5 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : memref<24xf32>, i32 -> f32
        %344 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %345 = tor.muli %344 %c3_i32 on (78 to 79) {dump = "unknown_191", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %346 = tor.addi %345 %c5_i32 on (78 to 79) {dump = "unknown_192", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %347 = tor.cmpi slt %346 %c0_i32 on (78 to 79) {dump = "unknown_193", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %348 = tor.subi %c-6_i32 %345 on (78 to 79) {dump = "unknown_194", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %349 = arith.select %347, %348, %346 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %350 = tor.cmpi slt %349 %c0_i32 on (78 to 79) {dump = "unknown_195", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %351 = arith.extui %350 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %352 = tor.addi %349 %351 on (78 to 79) {dump = "unknown_196", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %353 = arith.shrsi %352, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %354 = tor.subi %c-1_i32 %353 on (78 to 79) {dump = "unknown_197", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %355 = arith.select %347, %354, %353 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %356 = aps.read_smem %7[%355] {aps.mem_id = 4 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : memref<24xf32>, i32 -> f32
        %357 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %358 = tor.addi %357 %c1_i32 on (78 to 79) {dump = "unknown_198", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %359 = tor.cmpi slt %358 %c0_i32 on (78 to 79) {dump = "unknown_199", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %360 = tor.subi %c-2_i32 %357 on (78 to 79) {dump = "unknown_200", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %361 = arith.select %359, %360, %358 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %362 = tor.cmpi slt %361 %c0_i32 on (78 to 79) {dump = "unknown_201", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %363 = arith.extui %362 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %364 = tor.addi %361 %363 on (78 to 79) {dump = "unknown_202", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %365 = arith.shrsi %364, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %366 = tor.subi %c-1_i32 %365 on (78 to 79) {dump = "unknown_203", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %367 = arith.select %359, %366, %365 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %368 = tor.muli %367 %c-2_i32 on (78 to 79) {dump = "unknown_204", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %369 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %370 = tor.addi %369 %368 on (78 to 79) {dump = "unknown_205", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %371 = tor.addi %370 %c1_i32 on (78 to 79) {dump = "unknown_206", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %372 = tor.cmpi eq %371 %c0_i32 on (78 to 79) {dump = "unknown_207", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %373 = arith.select %372, %343, %356 {endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, starttime = 79 : i32} : f32
        %374 = tor.divf %330 %373 on (82 to 109) {dump = "unknown_208", fastmath = #arith.fastmath<none>, ref_endtime = 109 : i32, ref_starttime = 82 : i32} : f32
        %375 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %376 = tor.muli %375 %c3_i32 on (78 to 79) {dump = "unknown_209", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %377 = tor.addi %376 %c5_i32 on (78 to 79) {dump = "unknown_210", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %378 = tor.cmpi slt %377 %c0_i32 on (78 to 79) {dump = "unknown_211", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %379 = tor.subi %c-6_i32 %376 on (78 to 79) {dump = "unknown_212", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %380 = arith.select %378, %379, %377 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %381 = tor.cmpi slt %380 %c0_i32 on (78 to 79) {dump = "unknown_213", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %382 = arith.extui %381 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %383 = tor.addi %380 %382 on (78 to 79) {dump = "unknown_214", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %384 = arith.shrsi %383, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %385 = tor.subi %c-1_i32 %384 on (78 to 79) {dump = "unknown_215", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %386 = arith.select %378, %385, %384 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %387 = aps.read_smem %11[%386] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 9 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 17 : i64}], aps.mem_id = 3 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : memref<24xf32>, i32 -> f32
        %388 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %389 = tor.muli %388 %c3_i32 on (78 to 79) {dump = "unknown_216", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %390 = tor.addi %389 %c5_i32 on (78 to 79) {dump = "unknown_217", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %391 = tor.cmpi slt %390 %c0_i32 on (78 to 79) {dump = "unknown_218", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %392 = tor.subi %c-6_i32 %389 on (78 to 79) {dump = "unknown_219", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %393 = arith.select %391, %392, %390 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %394 = tor.cmpi slt %393 %c0_i32 on (78 to 79) {dump = "unknown_220", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %395 = arith.extui %394 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %396 = tor.addi %393 %395 on (78 to 79) {dump = "unknown_221", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %397 = arith.shrsi %396, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %398 = tor.subi %c-1_i32 %397 on (78 to 79) {dump = "unknown_222", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %399 = arith.select %391, %398, %397 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %400 = aps.read_smem %12[%399] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 8 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 16 : i64}], aps.mem_id = 2 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : memref<24xf32>, i32 -> f32
        %401 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %402 = tor.addi %401 %c1_i32 on (78 to 79) {dump = "unknown_223", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %403 = tor.cmpi slt %402 %c0_i32 on (78 to 79) {dump = "unknown_224", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %404 = tor.subi %c-2_i32 %401 on (78 to 79) {dump = "unknown_225", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %405 = arith.select %403, %404, %402 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %406 = tor.cmpi slt %405 %c0_i32 on (78 to 79) {dump = "unknown_226", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %407 = arith.extui %406 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %408 = tor.addi %405 %407 on (78 to 79) {dump = "unknown_227", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %409 = arith.shrsi %408, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %410 = tor.subi %c-1_i32 %409 on (78 to 79) {dump = "unknown_228", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %411 = arith.select %403, %410, %409 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %412 = tor.muli %411 %c-2_i32 on (78 to 79) {dump = "unknown_229", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %413 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %414 = tor.addi %413 %412 on (78 to 79) {dump = "unknown_230", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %415 = tor.addi %414 %c1_i32 on (78 to 79) {dump = "unknown_231", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %416 = tor.cmpi eq %415 %c0_i32 on (78 to 79) {dump = "unknown_232", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %417 = arith.select %416, %387, %400 {endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, starttime = 120 : i32} : f32
        %418 = tor.subf %417 %374 on (120 to 123) {dump = "unknown_233", fastmath = #arith.fastmath<none>, ref_endtime = 123 : i32, ref_starttime = 120 : i32} : f32
        %419 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %420 = tor.addi %419 %c1_i32 on (78 to 79) {dump = "unknown_234", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %421 = tor.cmpi slt %420 %c0_i32 on (78 to 79) {dump = "unknown_235", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %422 = tor.subi %c-2_i32 %419 on (78 to 79) {dump = "unknown_236", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %423 = arith.select %421, %422, %420 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %424 = tor.cmpi slt %423 %c0_i32 on (78 to 79) {dump = "unknown_237", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %425 = arith.extui %424 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %426 = tor.addi %423 %425 on (78 to 79) {dump = "unknown_238", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %427 = arith.shrsi %426, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %428 = tor.subi %c-1_i32 %427 on (78 to 79) {dump = "unknown_239", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %429 = arith.select %421, %428, %427 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %430 = tor.muli %429 %c-2_i32 on (78 to 79) {dump = "unknown_240", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %431 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %432 = tor.addi %431 %430 on (78 to 79) {dump = "unknown_241", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %433 = tor.addi %432 %c1_i32 on (78 to 79) {dump = "unknown_242", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %434 = arith.extui %arg3 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i4 to i32
        %435 = tor.muli %434 %c3_i32 on (78 to 79) {dump = "unknown_243", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %436 = tor.addi %435 %c5_i32 on (78 to 79) {dump = "unknown_244", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %437 = tor.cmpi slt %436 %c0_i32 on (78 to 79) {dump = "unknown_245", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %438 = tor.subi %c-6_i32 %435 on (78 to 79) {dump = "unknown_246", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %439 = arith.select %437, %438, %436 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %440 = tor.cmpi slt %439 %c0_i32 on (78 to 79) {dump = "unknown_247", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        %441 = arith.extui %440 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i1 to i32
        %442 = tor.addi %439 %441 on (78 to 79) {dump = "unknown_248", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %443 = arith.shrsi %442, %c1_i32 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %444 = tor.subi %c-1_i32 %443 on (78 to 79) {dump = "unknown_249", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
        %445 = arith.select %437, %444, %443 {endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, starttime = 78 : i32} : i32
        %446 = tor.cmpi eq %433 %c0_i32 on (78 to 79) {dump = "unknown_250", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        aps.write_smem_if %446, %418, %11[%445] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 3 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 11 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 19 : i64}], aps.mem_id = 1 : i64, endtime = 124 : i32, ref_endtime = 124 : i32, ref_starttime = 123 : i32, slot = 0 : i32, starttime = 123 : i32} : f32, memref<24xf32>, i32
        %447 = tor.cmpi eq %433 %c1_i32 on (78 to 79) {dump = "unknown_251", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i1
        aps.write_smem_if %447, %418, %12[%445] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 2 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 10 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 18 : i64}], aps.mem_id = 0 : i64, endtime = 124 : i32, ref_endtime = 124 : i32, ref_starttime = 123 : i32, slot = 0 : i32, starttime = 123 : i32} : f32, memref<24xf32>, i32
      } {ref_endtime = 124 : i32, ref_starttime = 78 : i32}
      aps.copy_by @busitfc, (%11, %12)[%c0_i32], %10, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 133 : i32, estimated_latency = 29 : i64, interface_cost = 53 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 133 : i32, ref_starttime = 124 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 5.300000e+01 : f64, starttime = 124 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 9 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<24xf32>, memref<24xf32>), i32, i32, i32
      %15 = tor.addi %10 %c64_i32 on (78 to 79) {dump = "unknown_252", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%11, %12)[%c16_i32], %15, %c16_i32 {after = array<i64: 9, 9>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 143 : i32, estimated_latency = 29 : i64, interface_cost = 53 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 143 : i32, ref_starttime = 125 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 5.300000e+01 : f64, starttime = 125 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 10 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<24xf32>, memref<24xf32>), i32, i32, i32
      %16 = tor.addi %10 %c128_i32 on (78 to 79) {dump = "unknown_253", ref_endtime = 79 : i32, ref_starttime = 78 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%11, %12)[%c32_i32], %16, %c16_i32 {after = array<i64: 10, 9, 10>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 158 : i32, estimated_latency = 29 : i64, interface_cost = 53 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 158 : i32, ref_starttime = 139 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 5.300000e+01 : f64, starttime = 139 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 11 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<24xf32>, memref<24xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_21__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 1 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/planar_slam/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_254", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_255", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_21__direct_buffer_0 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_21__direct_buffer_1 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64: 12>, after_gap = array<i64: 1>, endtime = 13 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 13 : i64, memory_issue_cycle = 1 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 7.000000e+01 : f64, starttime = 1 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 13 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_256", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 13, 13>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 21 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 21 : i64, memory_issue_cycle = 2 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 21 : i32, ref_starttime = 2 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 7.000000e+01 : f64, starttime = 2 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 14 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %5 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_257", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 14, 13, 14>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 31 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 14 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 31 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 7.000000e+01 : f64, starttime = 16 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 15 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %6 = aps.load_by @cpuitfc, %1 {after = array<i64>, after_gap = array<i64>, endtime = 2 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, ref_endtime = 2 : i32, ref_starttime = 0 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 7.000000e+01 : f64, starttime = 0 : i32, transaction_id = 12 : i64, transfer_bytes = 4 : i64} : i32 -> i32
      aps.globalstore %6, @au_generated_candidate_21__direct_buffer_2 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
      tor.for %arg3 = (%c0_i5 : i5) to (%c15_i5 : i5) step (%c1_i5 : i5)
      on (33 to 41){
        %8 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %9 = tor.muli %8 %c3_i32 on (33 to 34) {dump = "unknown_258", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %10 = tor.cmpi slt %9 %c0_i32 on (33 to 34) {dump = "unknown_259", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %11 = tor.subi %c-1_i32 %9 on (33 to 34) {dump = "unknown_260", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %12 = arith.select %10, %11, %9 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %13 = tor.cmpi slt %12 %c0_i32 on (33 to 34) {dump = "unknown_261", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %14 = arith.extui %13 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i1 to i32
        %15 = tor.addi %12 %14 on (33 to 34) {dump = "unknown_262", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %16 = arith.shrsi %15, %c1_i32 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %17 = tor.subi %c-1_i32 %16 on (33 to 34) {dump = "unknown_263", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %18 = arith.select %10, %17, %16 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %19 = aps.read_smem %2[%18] {aps.mem_id = 5 : i64, endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, slot = 0 : i32, starttime = 33 : i32} : memref<24xf32>, i32 -> f32
        %20 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %21 = tor.muli %20 %c3_i32 on (33 to 34) {dump = "unknown_264", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %22 = tor.cmpi slt %21 %c0_i32 on (33 to 34) {dump = "unknown_265", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %23 = tor.subi %c-1_i32 %21 on (33 to 34) {dump = "unknown_266", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %24 = arith.select %22, %23, %21 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %25 = tor.cmpi slt %24 %c0_i32 on (33 to 34) {dump = "unknown_267", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %26 = arith.extui %25 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i1 to i32
        %27 = tor.addi %24 %26 on (33 to 34) {dump = "unknown_268", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %28 = arith.shrsi %27, %c1_i32 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %29 = tor.subi %c-1_i32 %28 on (33 to 34) {dump = "unknown_269", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %30 = arith.select %22, %29, %28 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %31 = aps.read_smem %3[%30] {aps.mem_id = 4 : i64, endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, slot = 0 : i32, starttime = 33 : i32} : memref<24xf32>, i32 -> f32
        %32 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %33 = arith.andi %32, %c1_i32 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %34 = tor.cmpi eq %33 %c0_i32 on (33 to 34) {dump = "unknown_270", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %35 = arith.select %34, %19, %31 {endtime = 35 : i32, ref_endtime = 35 : i32, ref_starttime = 34 : i32, starttime = 34 : i32} : f32
        %36 = tor.mulf %35 %cst_32 on (34 to 37) {dump = "unknown_271", fastmath = #arith.fastmath<none>, ref_endtime = 37 : i32, ref_starttime = 34 : i32} : f32
        %37 = arith.fptosi %36 {endtime = 38 : i32, ref_endtime = 38 : i32, ref_starttime = 37 : i32, starttime = 37 : i32} : f32 to i32
        %38 = aps.globalload @au_generated_candidate_21__direct_buffer_2 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %39 = tor.addi %38 %37 on (38 to 39) {dump = "unknown_272", ref_endtime = 39 : i32, ref_starttime = 38 : i32} : (i32, i32) -> i32
        %40 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %41 = tor.muli %40 %c3_i32 on (33 to 34) {dump = "unknown_273", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %42 = tor.addi %41 %c1_i32 on (33 to 34) {dump = "unknown_274", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %43 = tor.cmpi slt %42 %c0_i32 on (33 to 34) {dump = "unknown_275", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %44 = tor.subi %c-2_i32 %41 on (33 to 34) {dump = "unknown_276", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %45 = arith.select %43, %44, %42 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %46 = tor.cmpi slt %45 %c0_i32 on (33 to 34) {dump = "unknown_277", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %47 = arith.extui %46 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i1 to i32
        %48 = tor.addi %45 %47 on (33 to 34) {dump = "unknown_278", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %49 = arith.shrsi %48, %c1_i32 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %50 = tor.subi %c-1_i32 %49 on (33 to 34) {dump = "unknown_279", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %51 = arith.select %43, %50, %49 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %52 = aps.read_smem %2[%51] {aps.mem_id = 3 : i64, endtime = 35 : i32, ref_endtime = 35 : i32, ref_starttime = 34 : i32, slot = 0 : i32, starttime = 34 : i32} : memref<24xf32>, i32 -> f32
        %53 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %54 = tor.muli %53 %c3_i32 on (33 to 34) {dump = "unknown_280", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %55 = tor.addi %54 %c1_i32 on (33 to 34) {dump = "unknown_281", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %56 = tor.cmpi slt %55 %c0_i32 on (33 to 34) {dump = "unknown_282", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %57 = tor.subi %c-2_i32 %54 on (33 to 34) {dump = "unknown_283", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %58 = arith.select %56, %57, %55 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %59 = tor.cmpi slt %58 %c0_i32 on (33 to 34) {dump = "unknown_284", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %60 = arith.extui %59 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i1 to i32
        %61 = tor.addi %58 %60 on (33 to 34) {dump = "unknown_285", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %62 = arith.shrsi %61, %c1_i32 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %63 = tor.subi %c-1_i32 %62 on (33 to 34) {dump = "unknown_286", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %64 = arith.select %56, %63, %62 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %65 = aps.read_smem %3[%64] {aps.mem_id = 2 : i64, endtime = 35 : i32, ref_endtime = 35 : i32, ref_starttime = 34 : i32, slot = 0 : i32, starttime = 34 : i32} : memref<24xf32>, i32 -> f32
        %66 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %67 = tor.addi %66 %c1_i32 on (33 to 34) {dump = "unknown_287", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %68 = tor.cmpi slt %67 %c0_i32 on (33 to 34) {dump = "unknown_288", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %69 = tor.subi %c-2_i32 %66 on (33 to 34) {dump = "unknown_289", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %70 = arith.select %68, %69, %67 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %71 = tor.cmpi slt %70 %c0_i32 on (33 to 34) {dump = "unknown_290", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %72 = arith.extui %71 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i1 to i32
        %73 = tor.addi %70 %72 on (33 to 34) {dump = "unknown_291", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %74 = arith.shrsi %73, %c1_i32 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %75 = tor.subi %c-1_i32 %74 on (33 to 34) {dump = "unknown_292", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %76 = arith.select %68, %75, %74 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %77 = tor.muli %76 %c-2_i32 on (33 to 34) {dump = "unknown_293", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %78 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %79 = tor.addi %78 %77 on (33 to 34) {dump = "unknown_294", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %80 = tor.addi %79 %c1_i32 on (33 to 34) {dump = "unknown_295", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %81 = tor.cmpi eq %80 %c0_i32 on (33 to 34) {dump = "unknown_296", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %82 = arith.select %81, %52, %65 {endtime = 36 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, starttime = 35 : i32} : f32
        %83 = tor.mulf %82 %cst_31 on (35 to 38) {dump = "unknown_297", fastmath = #arith.fastmath<none>, ref_endtime = 38 : i32, ref_starttime = 35 : i32} : f32
        %84 = arith.fptosi %83 {endtime = 39 : i32, ref_endtime = 39 : i32, ref_starttime = 38 : i32, starttime = 38 : i32} : f32 to i32
        %85 = tor.addi %39 %84 on (39 to 40) {dump = "unknown_298", ref_endtime = 40 : i32, ref_starttime = 39 : i32} : (i32, i32) -> i32
        %86 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %87 = tor.muli %86 %c3_i32 on (33 to 34) {dump = "unknown_299", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %88 = tor.cmpi slt %87 %c0_i32 on (33 to 34) {dump = "unknown_300", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %89 = tor.subi %c-1_i32 %87 on (33 to 34) {dump = "unknown_301", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %90 = arith.select %88, %89, %87 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %91 = tor.cmpi slt %90 %c0_i32 on (33 to 34) {dump = "unknown_302", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %92 = arith.extui %91 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i1 to i32
        %93 = tor.addi %90 %92 on (33 to 34) {dump = "unknown_303", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %94 = arith.shrsi %93, %c1_i32 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %95 = tor.subi %c-1_i32 %94 on (33 to 34) {dump = "unknown_304", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %96 = arith.select %88, %95, %94 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %97 = tor.addi %96 %c1_i32 on (33 to 34) {dump = "unknown_305", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %98 = aps.read_smem %2[%97] {aps.mem_id = 1 : i64, endtime = 36 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 35 : i32} : memref<24xf32>, i32 -> f32
        %99 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %100 = tor.muli %99 %c3_i32 on (33 to 34) {dump = "unknown_306", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %101 = tor.cmpi slt %100 %c0_i32 on (33 to 34) {dump = "unknown_307", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %102 = tor.subi %c-1_i32 %100 on (33 to 34) {dump = "unknown_308", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %103 = arith.select %101, %102, %100 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %104 = tor.cmpi slt %103 %c0_i32 on (33 to 34) {dump = "unknown_309", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %105 = arith.extui %104 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i1 to i32
        %106 = tor.addi %103 %105 on (33 to 34) {dump = "unknown_310", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %107 = arith.shrsi %106, %c1_i32 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %108 = tor.subi %c-1_i32 %107 on (33 to 34) {dump = "unknown_311", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %109 = arith.select %101, %108, %107 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %110 = tor.addi %109 %c1_i32 on (33 to 34) {dump = "unknown_312", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i32
        %111 = aps.read_smem %3[%110] {aps.mem_id = 0 : i64, endtime = 36 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 35 : i32} : memref<24xf32>, i32 -> f32
        %112 = arith.extui %arg3 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i5 to i32
        %113 = arith.andi %112, %c1_i32 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
        %114 = tor.cmpi eq %113 %c0_i32 on (33 to 34) {dump = "unknown_313", ref_endtime = 34 : i32, ref_starttime = 33 : i32} : (i32, i32) -> i1
        %115 = arith.select %114, %98, %111 {endtime = 37 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, starttime = 36 : i32} : f32
        %116 = tor.mulf %115 %cst_30 on (36 to 39) {dump = "unknown_314", fastmath = #arith.fastmath<none>, ref_endtime = 39 : i32, ref_starttime = 36 : i32} : f32
        %117 = arith.fptosi %116 {endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, starttime = 39 : i32} : f32 to i32
        %118 = tor.addi %85 %117 on (40 to 41) {dump = "unknown_315", ref_endtime = 41 : i32, ref_starttime = 40 : i32} : (i32, i32) -> i32
        aps.globalstore %118, @au_generated_candidate_21__direct_buffer_2 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i32
      } {ref_endtime = 41 : i32, ref_starttime = 33 : i32}
      %7 = aps.globalload @au_generated_candidate_21__direct_buffer_2 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : i32
      aps.store_by @cpuitfc, %7, %1 {after = array<i64>, after_gap = array<i64>, endtime = 35 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 35 : i32, ref_starttime = 33 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 1.800000e+01 : f64, starttime = 33 : i32, transaction_id = 16 : i64, transfer_bytes = 4 : i64} : i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_34__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 2 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/planar_slam/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_316", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = memref.get_global @au_generated_candidate_34__l0u_buffer_0 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %2 = memref.get_global @au_generated_candidate_34__l0u_buffer_1 : memref<24xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%1, %2)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 5.200000e+01 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 17 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %3 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_317", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %3, (%1, %2)[%c16_i32], %c16_i32 {after = array<i64: 17, 17>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 20 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 20 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 5.200000e+01 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 5 : i64, transaction_id = 18 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      %4 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_318", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%1, %2)[%c32_i32], %c16_i32 {after = array<i64: 18, 17, 18>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 30 : i32, estimated_latency = 28 : i64, interface_cost = 52 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 30 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 5.200000e+01 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 19 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<24xf32>, memref<24xf32>), i32, i32
      aps.write_smem %cst_10, %1[%c0_i32] {aps.mem_id = 47 : i64, endtime = 31 : i32, ref_endtime = 31 : i32, ref_starttime = 30 : i32, slot = 0 : i32, starttime = 30 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %2[%c0_i32] {aps.mem_id = 46 : i64, endtime = 31 : i32, ref_endtime = 31 : i32, ref_starttime = 30 : i32, slot = 0 : i32, starttime = 30 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_11, %1[%c1_i32] {aps.mem_id = 45 : i64, endtime = 32 : i32, ref_endtime = 32 : i32, ref_starttime = 31 : i32, slot = 0 : i32, starttime = 31 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_12, %2[%c1_i32] {aps.mem_id = 44 : i64, endtime = 32 : i32, ref_endtime = 32 : i32, ref_starttime = 31 : i32, slot = 0 : i32, starttime = 31 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_33, %1[%c2_i32] {aps.mem_id = 43 : i64, endtime = 33 : i32, ref_endtime = 33 : i32, ref_starttime = 32 : i32, slot = 0 : i32, starttime = 32 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_13, %2[%c2_i32] {aps.mem_id = 42 : i64, endtime = 33 : i32, ref_endtime = 33 : i32, ref_starttime = 32 : i32, slot = 0 : i32, starttime = 32 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_14, %1[%c3_i32] {aps.mem_id = 41 : i64, endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, slot = 0 : i32, starttime = 33 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %2[%c3_i32] {aps.mem_id = 40 : i64, endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, slot = 0 : i32, starttime = 33 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %1[%c4_i32] {aps.mem_id = 39 : i64, endtime = 35 : i32, ref_endtime = 35 : i32, ref_starttime = 34 : i32, slot = 0 : i32, starttime = 34 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_15, %2[%c4_i32] {aps.mem_id = 38 : i64, endtime = 35 : i32, ref_endtime = 35 : i32, ref_starttime = 34 : i32, slot = 0 : i32, starttime = 34 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_33, %1[%c5_i32] {aps.mem_id = 37 : i64, endtime = 36 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 35 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_16, %2[%c5_i32] {aps.mem_id = 36 : i64, endtime = 36 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 35 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_17, %1[%c6_i32] {aps.mem_id = 35 : i64, endtime = 37 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, slot = 0 : i32, starttime = 36 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %2[%c6_i32] {aps.mem_id = 34 : i64, endtime = 37 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, slot = 0 : i32, starttime = 36 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_18, %1[%c7_i32] {aps.mem_id = 33 : i64, endtime = 38 : i32, ref_endtime = 38 : i32, ref_starttime = 37 : i32, slot = 0 : i32, starttime = 37 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_19, %2[%c7_i32] {aps.mem_id = 32 : i64, endtime = 38 : i32, ref_endtime = 38 : i32, ref_starttime = 37 : i32, slot = 0 : i32, starttime = 37 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_33, %1[%c8_i32] {aps.mem_id = 31 : i64, endtime = 39 : i32, ref_endtime = 39 : i32, ref_starttime = 38 : i32, slot = 0 : i32, starttime = 38 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_11, %2[%c8_i32] {aps.mem_id = 30 : i64, endtime = 39 : i32, ref_endtime = 39 : i32, ref_starttime = 38 : i32, slot = 0 : i32, starttime = 38 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_20, %1[%c9_i32] {aps.mem_id = 29 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %2[%c9_i32] {aps.mem_id = 28 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_13, %1[%c10_i32] {aps.mem_id = 27 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_21, %2[%c10_i32] {aps.mem_id = 26 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_33, %1[%c11_i32] {aps.mem_id = 25 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %2[%c11_i32] {aps.mem_id = 24 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_22, %1[%c12_i32] {aps.mem_id = 23 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %2[%c12_i32] {aps.mem_id = 22 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_16, %1[%c13_i32] {aps.mem_id = 21 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_23, %2[%c13_i32] {aps.mem_id = 20 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_33, %1[%c14_i32] {aps.mem_id = 19 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_18, %2[%c14_i32] {aps.mem_id = 18 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_24, %1[%c15_i32] {aps.mem_id = 17 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %2[%c15_i32] {aps.mem_id = 16 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_11, %1[%c16_i32] {aps.mem_id = 15 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_25, %2[%c16_i32] {aps.mem_id = 14 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_33, %1[%c17_i32] {aps.mem_id = 13 : i64, endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, slot = 0 : i32, starttime = 47 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_13, %2[%c17_i32] {aps.mem_id = 12 : i64, endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, slot = 0 : i32, starttime = 47 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_26, %1[%c18_i32] {aps.mem_id = 11 : i64, endtime = 49 : i32, ref_endtime = 49 : i32, ref_starttime = 48 : i32, slot = 0 : i32, starttime = 48 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %2[%c18_i32] {aps.mem_id = 10 : i64, endtime = 49 : i32, ref_endtime = 49 : i32, ref_starttime = 48 : i32, slot = 0 : i32, starttime = 48 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %1[%c19_i32] {aps.mem_id = 9 : i64, endtime = 50 : i32, ref_endtime = 50 : i32, ref_starttime = 49 : i32, slot = 0 : i32, starttime = 49 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_27, %2[%c19_i32] {aps.mem_id = 8 : i64, endtime = 50 : i32, ref_endtime = 50 : i32, ref_starttime = 49 : i32, slot = 0 : i32, starttime = 49 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_33, %1[%c20_i32] {aps.mem_id = 7 : i64, endtime = 51 : i32, ref_endtime = 51 : i32, ref_starttime = 50 : i32, slot = 0 : i32, starttime = 50 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_16, %2[%c20_i32] {aps.mem_id = 6 : i64, endtime = 51 : i32, ref_endtime = 51 : i32, ref_starttime = 50 : i32, slot = 0 : i32, starttime = 50 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_28, %1[%c21_i32] {aps.mem_id = 5 : i64, endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, slot = 0 : i32, starttime = 51 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_10, %2[%c21_i32] {aps.mem_id = 4 : i64, endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, slot = 0 : i32, starttime = 51 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_18, %1[%c22_i32] {aps.mem_id = 3 : i64, endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, slot = 0 : i32, starttime = 52 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_29, %2[%c22_i32] {aps.mem_id = 2 : i64, endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, slot = 0 : i32, starttime = 52 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_33, %1[%c23_i32] {aps.mem_id = 1 : i64, endtime = 54 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, slot = 0 : i32, starttime = 53 : i32} : f32, memref<24xf32>, i32
      aps.write_smem %cst_11, %2[%c23_i32] {aps.mem_id = 0 : i64, endtime = 54 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, slot = 0 : i32, starttime = 53 : i32} : f32, memref<24xf32>, i32
      aps.copy_by @busitfc, (%1, %2)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 63 : i32, estimated_latency = 29 : i64, interface_cost = 53 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 63 : i32, ref_starttime = 54 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 5.300000e+01 : f64, starttime = 54 : i32, tl_channel = 0 : i32, transaction_group = 6 : i64, transaction_id = 20 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<24xf32>, memref<24xf32>), i32, i32, i32
      %5 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_319", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c16_i32], %5, %c16_i32 {after = array<i64: 20, 20>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 73 : i32, estimated_latency = 29 : i64, interface_cost = 53 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 73 : i32, ref_starttime = 55 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 5.300000e+01 : f64, starttime = 55 : i32, tl_channel = 1 : i32, transaction_group = 6 : i64, transaction_id = 21 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<24xf32>, memref<24xf32>), i32, i32, i32
      %6 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_320", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c32_i32], %6, %c16_i32 {after = array<i64: 21, 20, 21>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 88 : i32, estimated_latency = 29 : i64, interface_cost = 53 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 29 : i64, memory_search_hits = 0 : i64, memory_search_states = 5 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 88 : i32, ref_starttime = 69 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 5.300000e+01 : f64, starttime = 69 : i32, tl_channel = 0 : i32, transaction_group = 6 : i64, transaction_id = 22 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<24xf32>, memref<24xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_41__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 3 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/planar_slam/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_321", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_322", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_41__direct_buffer_0 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_41__direct_buffer_1 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64: 23, 23>, after_gap = array<i64: 1, 1>, after_segment = 2 : i64, endtime = 31 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 16 : i64, memory_issue_cycle = 1 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 44 : i64, memory_search_hits = 5 : i64, memory_search_states = 29 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 31 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 1.360000e+02 : f64, starttime = 16 : i32, tl_channel = 1 : i32, transaction_group = 7 : i64, transaction_id = 24 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_323", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 24, 23, 24>, after_gap = array<i64: 8, 9, 8>, after_segment = 0 : i64, endtime = 45 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 24 : i64, memory_issue_cycle = 9 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 44 : i64, memory_search_hits = 5 : i64, memory_search_states = 29 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 45 : i32, ref_starttime = 30 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 1.360000e+02 : f64, starttime = 30 : i32, tl_channel = 0 : i32, transaction_group = 7 : i64, transaction_id = 25 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %5 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_324", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c8_i32 {after = array<i64>, after_gap = array<i64>, endtime = 9 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 8 : i64, memory_issue_cycle = 0 : i64, memory_latency = 8 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 44 : i64, memory_search_hits = 5 : i64, memory_search_states = 29 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 9 : i32, ref_starttime = 1 : i32, required_alignment = 32 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 1.360000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 7 : i64, transaction_id = 23 : i64, transfer_bytes = 32 : i64, transfer_offset = 128 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %6 = memref.get_global @au_generated_candidate_41__direct_buffer_3_0 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %7 = memref.get_global @au_generated_candidate_41__direct_buffer_3_1 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%6, %7)[%c0_i32], %c16_i32 {after = array<i64: 25, 24, 25>, after_gap = array<i64: 8, 16, 8>, endtime = 60 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 32 : i64, memory_issue_cycle = 17 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 44 : i64, memory_search_hits = 5 : i64, memory_search_states = 29 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 60 : i32, ref_starttime = 45 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 1.360000e+02 : f64, starttime = 45 : i32, tl_channel = 0 : i32, transaction_group = 8 : i64, transaction_id = 26 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %8 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_325", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %8, (%6, %7)[%c16_i32], %c16_i32 {after = array<i64: 26, 25, 26>, after_gap = array<i64: 8, 16, 8>, after_segment = 0 : i64, endtime = 68 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 40 : i64, memory_issue_cycle = 25 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 44 : i64, memory_search_hits = 5 : i64, memory_search_states = 29 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 68 : i32, ref_starttime = 53 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 1.360000e+02 : f64, starttime = 53 : i32, tl_channel = 1 : i32, transaction_group = 8 : i64, transaction_id = 27 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %9 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_326", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %9, (%6, %7)[%c32_i32], %c8_i32 {after = array<i64: 27, 26, 27>, after_gap = array<i64: 8, 16, 8>, after_segment = 1 : i64, endtime = 72 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 33 : i64, memory_latency = 11 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 44 : i64, memory_search_hits = 5 : i64, memory_search_states = 29 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 72 : i32, ref_starttime = 61 : i32, required_alignment = 32 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 1.360000e+02 : f64, starttime = 61 : i32, tl_channel = 0 : i32, transaction_group = 8 : i64, transaction_id = 28 : i64, transfer_bytes = 32 : i64, transfer_offset = 128 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %10 = aps.read_csr @au_generated_candidate_41__direct_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %11 = memref.get_global @au_generated_candidate_41__direct_buffer_4_0 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %12 = memref.get_global @au_generated_candidate_41__direct_buffer_4_1 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %10, (%11, %12)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 24 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 1.360000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 9 : i64, transaction_id = 29 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %13 = tor.addi %10 %c64_i32 on (0 to 1) {dump = "unknown_327", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %13, (%11, %12)[%c16_i32], %c16_i32 {after = array<i64: 29, 29>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 34 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 24 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 34 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 1.360000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 9 : i64, transaction_id = 30 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %14 = tor.addi %10 %c128_i32 on (0 to 1) {dump = "unknown_328", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %14, (%11, %12)[%c32_i32], %c8_i32 {after = array<i64: 30, 29, 30>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 42 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 24 : i64, memory_issue_cycle = 13 : i64, memory_latency = 11 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 24 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 42 : i32, ref_starttime = 31 : i32, required_alignment = 32 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 1.360000e+02 : f64, starttime = 31 : i32, tl_channel = 1 : i32, transaction_group = 9 : i64, transaction_id = 31 : i64, transfer_bytes = 32 : i64, transfer_offset = 128 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      tor.for %arg3 = (%c0_i5 : i5) to (%c-13_i5 : i5) step (%c1_i5 : i5)
      on (74 to 109){
        %17 = arith.extui %arg3 {endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, starttime = 74 : i32} : i5 to i32
        %18 = aps.read_smem %2[%17] {aps.mem_id = 7 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : memref<20xf32>, i32 -> f32
        %19 = tor.mulf %18 %cst_9 on (75 to 78) {dump = "unknown_329", fastmath = #arith.fastmath<none>, ref_endtime = 78 : i32, ref_starttime = 75 : i32} : f32
        %20 = arith.extui %arg3 {endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, starttime = 74 : i32} : i5 to i32
        %21 = aps.read_smem %6[%20] {aps.mem_id = 6 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : memref<20xf32>, i32 -> f32
        %22 = tor.divf %19 %21 on (78 to 105) {dump = "unknown_330", fastmath = #arith.fastmath<none>, ref_endtime = 105 : i32, ref_starttime = 78 : i32} : f32
        %23 = arith.extui %arg3 {endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, starttime = 74 : i32} : i5 to i32
        %24 = aps.read_smem %11[%23] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 4 : i64}], aps.mem_id = 5 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : memref<20xf32>, i32 -> f32
        %25 = tor.subf %24 %22 on (105 to 108) {dump = "unknown_331", fastmath = #arith.fastmath<none>, ref_endtime = 108 : i32, ref_starttime = 105 : i32} : f32
        %26 = arith.extui %arg3 {endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, starttime = 74 : i32} : i5 to i32
        aps.write_smem %25, %11[%26] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 4 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 5 : i64}], aps.mem_id = 4 : i64, endtime = 109 : i32, ref_endtime = 109 : i32, ref_starttime = 108 : i32, slot = 0 : i32, starttime = 108 : i32} : f32, memref<20xf32>, i32
        %27 = arith.extui %arg3 {endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, starttime = 74 : i32} : i5 to i32
        %28 = aps.read_smem %3[%27] {aps.mem_id = 3 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : memref<20xf32>, i32 -> f32
        %29 = tor.mulf %28 %cst_9 on (75 to 78) {dump = "unknown_332", fastmath = #arith.fastmath<none>, ref_endtime = 78 : i32, ref_starttime = 75 : i32} : f32
        %30 = arith.extui %arg3 {endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, starttime = 74 : i32} : i5 to i32
        %31 = aps.read_smem %7[%30] {aps.mem_id = 2 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : memref<20xf32>, i32 -> f32
        %32 = tor.divf %29 %31 on (78 to 105) {dump = "unknown_333", fastmath = #arith.fastmath<none>, ref_endtime = 105 : i32, ref_starttime = 78 : i32} : f32
        %33 = arith.extui %arg3 {endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, starttime = 74 : i32} : i5 to i32
        %34 = aps.read_smem %12[%33] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}], aps.mem_id = 1 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : memref<20xf32>, i32 -> f32
        %35 = tor.subf %34 %32 on (105 to 108) {dump = "unknown_334", fastmath = #arith.fastmath<none>, ref_endtime = 108 : i32, ref_starttime = 105 : i32} : f32
        %36 = arith.extui %arg3 {endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, starttime = 74 : i32} : i5 to i32
        aps.write_smem %35, %12[%36] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 1 : i64}], aps.mem_id = 0 : i64, endtime = 109 : i32, ref_endtime = 109 : i32, ref_starttime = 108 : i32, slot = 0 : i32, starttime = 108 : i32} : f32, memref<20xf32>, i32
      } {ref_endtime = 109 : i32, ref_starttime = 74 : i32}
      aps.copy_by @busitfc, (%11, %12)[%c0_i32], %10, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 118 : i32, estimated_latency = 25 : i64, interface_cost = 49 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 118 : i32, ref_starttime = 109 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 4.900000e+01 : f64, starttime = 109 : i32, tl_channel = 0 : i32, transaction_group = 10 : i64, transaction_id = 32 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<20xf32>, memref<20xf32>), i32, i32, i32
      %15 = tor.addi %10 %c64_i32 on (74 to 75) {dump = "unknown_335", ref_endtime = 75 : i32, ref_starttime = 74 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%11, %12)[%c16_i32], %15, %c16_i32 {after = array<i64: 32, 32>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 128 : i32, estimated_latency = 25 : i64, interface_cost = 49 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 128 : i32, ref_starttime = 110 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 4.900000e+01 : f64, starttime = 110 : i32, tl_channel = 1 : i32, transaction_group = 10 : i64, transaction_id = 33 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<20xf32>, memref<20xf32>), i32, i32, i32
      %16 = tor.addi %10 %c128_i32 on (74 to 75) {dump = "unknown_336", ref_endtime = 75 : i32, ref_starttime = 74 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%11, %12)[%c32_i32], %16, %c8_i32 {after = array<i64: 33, 32, 33>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 139 : i32, estimated_latency = 25 : i64, interface_cost = 49 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 25 : i64, memory_issue_cycle = 10 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 139 : i32, ref_starttime = 124 : i32, required_alignment = 32 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 4.900000e+01 : f64, starttime = 124 : i32, tl_channel = 0 : i32, transaction_group = 10 : i64, transaction_id = 34 : i64, transfer_bytes = 32 : i64, transfer_offset = 128 : i64} : (memref<20xf32>, memref<20xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_51__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 4 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/planar_slam/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg1 {dump = "unknown_337", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg0 {dump = "unknown_338", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_51__direct_buffer_0 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_51__direct_buffer_1 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64: 35>, after_gap = array<i64: 1>, endtime = 13 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 13 : i64, memory_issue_cycle = 1 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 12 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 6.600000e+01 : f64, starttime = 1 : i32, tl_channel = 0 : i32, transaction_group = 11 : i64, transaction_id = 36 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %4 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_339", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 36, 36>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 21 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 21 : i64, memory_issue_cycle = 2 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 12 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 21 : i32, ref_starttime = 2 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 6.600000e+01 : f64, starttime = 2 : i32, tl_channel = 1 : i32, transaction_group = 11 : i64, transaction_id = 37 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %5 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_340", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c8_i32 {after = array<i64: 37, 36, 37>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 27 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 25 : i64, memory_issue_cycle = 14 : i64, memory_latency = 11 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 12 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 27 : i32, ref_starttime = 16 : i32, required_alignment = 32 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 6.600000e+01 : f64, starttime = 16 : i32, tl_channel = 0 : i32, transaction_group = 11 : i64, transaction_id = 38 : i64, transfer_bytes = 32 : i64, transfer_offset = 128 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %6 = aps.load_by @cpuitfc, %0 {after = array<i64>, after_gap = array<i64>, endtime = 2 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 12 : i64, ref_endtime = 2 : i32, ref_starttime = 0 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 6.600000e+01 : f64, starttime = 0 : i32, transaction_id = 35 : i64, transfer_bytes = 4 : i64} : i32 -> i32
      aps.globalstore %6, @au_generated_candidate_51__direct_buffer_5 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
      tor.for %arg3 = (%c0_i5 : i5) to (%c-13_i5 : i5) step (%c1_i5 : i5)
      on (29 to 35){
        %8 = arith.extui %arg3 {endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, starttime = 29 : i32} : i5 to i32
        %9 = aps.read_smem %2[%8] {aps.mem_id = 1 : i64, endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, slot = 0 : i32, starttime = 29 : i32} : memref<20xf32>, i32 -> f32
        %10 = tor.mulf %9 %cst_8 on (30 to 33) {dump = "unknown_341", fastmath = #arith.fastmath<none>, ref_endtime = 33 : i32, ref_starttime = 30 : i32} : f32
        %11 = arith.fptosi %10 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : f32 to i32
        %12 = aps.globalload @au_generated_candidate_51__direct_buffer_5 {endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, starttime = 29 : i32} : i32
        %13 = tor.addi %12 %11 on (34 to 35) {dump = "unknown_342", ref_endtime = 35 : i32, ref_starttime = 34 : i32} : (i32, i32) -> i32
        %14 = arith.extui %arg3 {endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, starttime = 29 : i32} : i5 to i32
        %15 = aps.read_smem %3[%14] {aps.mem_id = 0 : i64, endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, slot = 0 : i32, starttime = 29 : i32} : memref<20xf32>, i32 -> f32
        %16 = tor.mulf %15 %cst_7 on (30 to 33) {dump = "unknown_343", fastmath = #arith.fastmath<none>, ref_endtime = 33 : i32, ref_starttime = 30 : i32} : f32
        %17 = arith.fptosi %16 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : f32 to i32
        %18 = tor.addi %13 %17 on (34 to 35) {dump = "unknown_344", ref_endtime = 35 : i32, ref_starttime = 34 : i32} : (i32, i32) -> i32
        aps.globalstore %18, @au_generated_candidate_51__direct_buffer_5 {endtime = 35 : i32, ref_endtime = 35 : i32, ref_starttime = 34 : i32, starttime = 34 : i32} : i32
      } {ref_endtime = 35 : i32, ref_starttime = 29 : i32}
      %7 = aps.globalload @au_generated_candidate_51__direct_buffer_5 {endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, starttime = 29 : i32} : i32
      aps.store_by @cpuitfc, %7, %0 {after = array<i64>, after_gap = array<i64>, endtime = 31 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 31 : i32, ref_starttime = 29 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 1.800000e+01 : f64, starttime = 29 : i32, transaction_id = 39 : i64, transfer_bytes = 4 : i64} : i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_56__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 5 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/planar_slam/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_345", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = memref.get_global @au_generated_candidate_56__l0u_buffer_0 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %2 = memref.get_global @au_generated_candidate_56__l0u_buffer_1 : memref<20xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%1, %2)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 24 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 4.800000e+01 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 12 : i64, transaction_id = 40 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %3 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_346", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %3, (%1, %2)[%c16_i32], %c16_i32 {after = array<i64: 40, 40>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 20 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 24 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 20 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 4.800000e+01 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 12 : i64, transaction_id = 41 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      %4 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_347", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%1, %2)[%c32_i32], %c8_i32 {after = array<i64: 41, 40, 41>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 26 : i32, estimated_latency = 24 : i64, interface_cost = 48 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 24 : i64, memory_issue_cycle = 13 : i64, memory_latency = 11 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 24 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 26 : i32, ref_starttime = 15 : i32, required_alignment = 32 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 4.800000e+01 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 12 : i64, transaction_id = 42 : i64, transfer_bytes = 32 : i64, transfer_offset = 128 : i64} : i32, (memref<20xf32>, memref<20xf32>), i32, i32
      aps.write_smem %cst_14, %1[%c0_i32] {aps.mem_id = 39 : i64, endtime = 27 : i32, ref_endtime = 27 : i32, ref_starttime = 26 : i32, slot = 0 : i32, starttime = 26 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst, %2[%c0_i32] {aps.mem_id = 38 : i64, endtime = 27 : i32, ref_endtime = 27 : i32, ref_starttime = 26 : i32, slot = 0 : i32, starttime = 26 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_0, %1[%c1_i32] {aps.mem_id = 37 : i64, endtime = 28 : i32, ref_endtime = 28 : i32, ref_starttime = 27 : i32, slot = 0 : i32, starttime = 27 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst, %2[%c1_i32] {aps.mem_id = 36 : i64, endtime = 28 : i32, ref_endtime = 28 : i32, ref_starttime = 27 : i32, slot = 0 : i32, starttime = 27 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_1, %1[%c2_i32] {aps.mem_id = 35 : i64, endtime = 29 : i32, ref_endtime = 29 : i32, ref_starttime = 28 : i32, slot = 0 : i32, starttime = 28 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst, %2[%c2_i32] {aps.mem_id = 34 : i64, endtime = 29 : i32, ref_endtime = 29 : i32, ref_starttime = 28 : i32, slot = 0 : i32, starttime = 28 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_2, %1[%c3_i32] {aps.mem_id = 33 : i64, endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, slot = 0 : i32, starttime = 29 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst, %2[%c3_i32] {aps.mem_id = 32 : i64, endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, slot = 0 : i32, starttime = 29 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_3, %1[%c4_i32] {aps.mem_id = 31 : i64, endtime = 31 : i32, ref_endtime = 31 : i32, ref_starttime = 30 : i32, slot = 0 : i32, starttime = 30 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst, %2[%c4_i32] {aps.mem_id = 30 : i64, endtime = 31 : i32, ref_endtime = 31 : i32, ref_starttime = 30 : i32, slot = 0 : i32, starttime = 30 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_14, %1[%c5_i32] {aps.mem_id = 29 : i64, endtime = 32 : i32, ref_endtime = 32 : i32, ref_starttime = 31 : i32, slot = 0 : i32, starttime = 31 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_4, %2[%c5_i32] {aps.mem_id = 28 : i64, endtime = 32 : i32, ref_endtime = 32 : i32, ref_starttime = 31 : i32, slot = 0 : i32, starttime = 31 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_0, %1[%c6_i32] {aps.mem_id = 27 : i64, endtime = 33 : i32, ref_endtime = 33 : i32, ref_starttime = 32 : i32, slot = 0 : i32, starttime = 32 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_4, %2[%c6_i32] {aps.mem_id = 26 : i64, endtime = 33 : i32, ref_endtime = 33 : i32, ref_starttime = 32 : i32, slot = 0 : i32, starttime = 32 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_1, %1[%c7_i32] {aps.mem_id = 25 : i64, endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, slot = 0 : i32, starttime = 33 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_4, %2[%c7_i32] {aps.mem_id = 24 : i64, endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, slot = 0 : i32, starttime = 33 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_2, %1[%c8_i32] {aps.mem_id = 23 : i64, endtime = 35 : i32, ref_endtime = 35 : i32, ref_starttime = 34 : i32, slot = 0 : i32, starttime = 34 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_4, %2[%c8_i32] {aps.mem_id = 22 : i64, endtime = 35 : i32, ref_endtime = 35 : i32, ref_starttime = 34 : i32, slot = 0 : i32, starttime = 34 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_3, %1[%c9_i32] {aps.mem_id = 21 : i64, endtime = 36 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 35 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_4, %2[%c9_i32] {aps.mem_id = 20 : i64, endtime = 36 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 35 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_14, %1[%c10_i32] {aps.mem_id = 19 : i64, endtime = 37 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, slot = 0 : i32, starttime = 36 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_5, %2[%c10_i32] {aps.mem_id = 18 : i64, endtime = 37 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, slot = 0 : i32, starttime = 36 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_0, %1[%c11_i32] {aps.mem_id = 17 : i64, endtime = 38 : i32, ref_endtime = 38 : i32, ref_starttime = 37 : i32, slot = 0 : i32, starttime = 37 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_5, %2[%c11_i32] {aps.mem_id = 16 : i64, endtime = 38 : i32, ref_endtime = 38 : i32, ref_starttime = 37 : i32, slot = 0 : i32, starttime = 37 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_1, %1[%c12_i32] {aps.mem_id = 15 : i64, endtime = 39 : i32, ref_endtime = 39 : i32, ref_starttime = 38 : i32, slot = 0 : i32, starttime = 38 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_5, %2[%c12_i32] {aps.mem_id = 14 : i64, endtime = 39 : i32, ref_endtime = 39 : i32, ref_starttime = 38 : i32, slot = 0 : i32, starttime = 38 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_2, %1[%c13_i32] {aps.mem_id = 13 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_5, %2[%c13_i32] {aps.mem_id = 12 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_3, %1[%c14_i32] {aps.mem_id = 11 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_5, %2[%c14_i32] {aps.mem_id = 10 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_14, %1[%c15_i32] {aps.mem_id = 9 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_6, %2[%c15_i32] {aps.mem_id = 8 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_0, %1[%c16_i32] {aps.mem_id = 7 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_6, %2[%c16_i32] {aps.mem_id = 6 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_1, %1[%c17_i32] {aps.mem_id = 5 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_6, %2[%c17_i32] {aps.mem_id = 4 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_2, %1[%c18_i32] {aps.mem_id = 3 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_6, %2[%c18_i32] {aps.mem_id = 2 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_3, %1[%c19_i32] {aps.mem_id = 1 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : f32, memref<20xf32>, i32
      aps.write_smem %cst_6, %2[%c19_i32] {aps.mem_id = 0 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : f32, memref<20xf32>, i32
      aps.copy_by @busitfc, (%1, %2)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 55 : i32, estimated_latency = 25 : i64, interface_cost = 49 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 55 : i32, ref_starttime = 46 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 3 : i64, solver_objective = 4.900000e+01 : f64, starttime = 46 : i32, tl_channel = 0 : i32, transaction_group = 13 : i64, transaction_id = 43 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<20xf32>, memref<20xf32>), i32, i32, i32
      %5 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_348", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c16_i32], %5, %c16_i32 {after = array<i64: 43, 43>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 65 : i32, estimated_latency = 25 : i64, interface_cost = 49 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 65 : i32, ref_starttime = 47 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 3 : i64, solver_objective = 4.900000e+01 : f64, starttime = 47 : i32, tl_channel = 1 : i32, transaction_group = 13 : i64, transaction_id = 44 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<20xf32>, memref<20xf32>), i32, i32, i32
      %6 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_349", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c32_i32], %6, %c8_i32 {after = array<i64: 44, 43, 44>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 76 : i32, estimated_latency = 25 : i64, interface_cost = 49 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 25 : i64, memory_issue_cycle = 10 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 25 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 76 : i32, ref_starttime = 61 : i32, required_alignment = 32 : i64, segment = 2 : i64, segment_count = 3 : i64, solver_objective = 4.900000e+01 : f64, starttime = 61 : i32, tl_channel = 0 : i32, transaction_group = 13 : i64, transaction_id = 45 : i64, transfer_bytes = 32 : i64, transfer_offset = 128 : i64} : (memref<20xf32>, memref<20xf32>), i32, i32, i32
      tor.return
    }
  } {HoistConstCondIfOp = 1 : i32, schedule = true}
}

