module {
  aps.memitfc @busitfc {cache_hint = "cold", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 8 : i64, min_beats = 1 : i64, min_write_beats = 2 : i64, read_latency = 5 : i64, width = 8 : i64, write_completion = 2 : i64}
  aps.memitfc @cpuitfc {cache_hint = "warm", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 1 : i64, min_beats = 1 : i64, min_write_beats = 1 : i64, read_latency = 2 : i64, width = 4 : i64, write_completion = 2 : i64}
  aps.memorymap {
    aps.mem_entry "au_generated_candidate_0__direct_buffer" : banks([@au_generated_candidate_0__direct_buffer_0, @au_generated_candidate_0__direct_buffer_1_2]), base(0), size(256), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_0__direct_buffer_1" : banks([@au_generated_candidate_0__direct_buffer_1_0, @au_generated_candidate_0__direct_buffer_1_1]), base(256), size(256), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_0__direct_buffer_2" : banks([@au_generated_candidate_0__direct_buffer_2_0, @au_generated_candidate_0__direct_buffer_2_1]), base(512), size(256), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_0__direct_local" : banks([@au_generated_candidate_0__direct_local]), base(768), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_0__direct_local_0" : banks([@au_generated_candidate_0__direct_local_0]), base(772), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_10__direct_buffer" : banks([@au_generated_candidate_10__direct_buffer_0, @au_generated_candidate_10__direct_buffer_1]), base(776), size(256), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_10__direct_buffer_3" : banks([@au_generated_candidate_10__direct_buffer_3_0, @au_generated_candidate_10__direct_buffer_3_1]), base(1280), size(256), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_21__l0u_l1u_buffer" : banks([@au_generated_candidate_21__l0u_l1u_buffer_0, @au_generated_candidate_21__l0u_l1u_buffer_1, @au_generated_candidate_21__l0u_l1u_buffer_2, @au_generated_candidate_21__l0u_l1u_buffer_3, @au_generated_candidate_21__l0u_l1u_buffer_4_8, @au_generated_candidate_21__l0u_l1u_buffer_5, @au_generated_candidate_21__l0u_l1u_buffer_6, @au_generated_candidate_21__l0u_l1u_buffer_7]), base(1536), size(256), count(8), cyclic(1)
    aps.mem_entry "au_generated_candidate_21__l0u_l1u_buffer_4" : banks([@au_generated_candidate_21__l0u_l1u_buffer_4_0, @au_generated_candidate_21__l0u_l1u_buffer_4_1, @au_generated_candidate_21__l0u_l1u_buffer_4_2, @au_generated_candidate_21__l0u_l1u_buffer_4_3, @au_generated_candidate_21__l0u_l1u_buffer_4_4, @au_generated_candidate_21__l0u_l1u_buffer_4_5, @au_generated_candidate_21__l0u_l1u_buffer_4_6, @au_generated_candidate_21__l0u_l1u_buffer_4_7]), base(1792), size(256), count(8), cyclic(1)
    aps.mem_entry "au_generated_candidate_35__l0u_l1u_l2u_buffer" : banks([@au_generated_candidate_35__l0u_l1u_l2u_buffer_0, @au_generated_candidate_35__l0u_l1u_l2u_buffer_1, @au_generated_candidate_35__l0u_l1u_l2u_buffer_2, @au_generated_candidate_35__l0u_l1u_l2u_buffer_3, @au_generated_candidate_35__l0u_l1u_l2u_buffer_4, @au_generated_candidate_35__l0u_l1u_l2u_buffer_5_0, @au_generated_candidate_35__l0u_l1u_l2u_buffer_6, @au_generated_candidate_35__l0u_l1u_l2u_buffer_7]), base(2048), size(256), count(8), cyclic(1)
    aps.mem_entry "au_generated_candidate_35__l0u_l1u_l2u_buffer_5" : banks([@au_generated_candidate_35__l0u_l1u_l2u_buffer_5]), base(2304), size(4), count(1), cyclic(0)
    aps.mem_finish
  }
  tor.design @aps_isaxes {
    %c-2_i2 = arith.constant -2 : i2
    %c1_i2 = arith.constant 1 : i2
    %c0_i2 = arith.constant 0 : i2
    %c7_i4 = arith.constant 7 : i4
    %c1_i4 = arith.constant 1 : i4
    %c0_i4 = arith.constant 0 : i4
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant 3.500000e-02 : f32
    %cst_0 = arith.constant 0.0249999985 : f32
    %cst_1 = arith.constant 1.500000e-02 : f32
    %cst_2 = arith.constant 5.000000e-03 : f32
    %c4_i32 = arith.constant 4 : i32
    %cst_3 = arith.constant 4.000000e-02 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_4 = arith.constant 3.000000e-02 : f32
    %c6_i32 = arith.constant 6 : i32
    %cst_5 = arith.constant 2.000000e-02 : f32
    %c8_i32 = arith.constant 8 : i32
    %c9_i32 = arith.constant 9 : i32
    %c10_i32 = arith.constant 10 : i32
    %c11_i32 = arith.constant 11 : i32
    %c12_i32 = arith.constant 12 : i32
    %c13_i32 = arith.constant 13 : i32
    %cst_6 = arith.constant 0.00999999977 : f32
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
    %c49_i32 = arith.constant 49 : i32
    %c50_i32 = arith.constant 50 : i32
    %c51_i32 = arith.constant 51 : i32
    %c52_i32 = arith.constant 52 : i32
    %c53_i32 = arith.constant 53 : i32
    %c54_i32 = arith.constant 54 : i32
    %c55_i32 = arith.constant 55 : i32
    %c56_i32 = arith.constant 56 : i32
    %c57_i32 = arith.constant 57 : i32
    %c58_i32 = arith.constant 58 : i32
    %c59_i32 = arith.constant 59 : i32
    %c60_i32 = arith.constant 60 : i32
    %c61_i32 = arith.constant 61 : i32
    %c62_i32 = arith.constant 62 : i32
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %cst_7 = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %c7_i32 = arith.constant 7 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_8 = arith.constant 1.000000e+30 : f32
    %c1_i32 = arith.constant {dump = "unknown_0"} 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %c192_i32 = arith.constant 192 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    memref.global @au_generated_candidate_35__l0u_l1u_l2u_buffer_5 : memref<1xi32> {scaleisa.scratchpad, var_name = "au_generated_candidate_35__l0u_l1u_l2u_buffer_5"}
    memref.global @au_generated_candidate_35__l0u_l1u_l2u_buffer_0 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_35__l0u_l1u_l2u_buffer_1 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_35__l0u_l1u_l2u_buffer_2 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_35__l0u_l1u_l2u_buffer_3 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_35__l0u_l1u_l2u_buffer_4 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_35__l0u_l1u_l2u_buffer_5_0 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_35__l0u_l1u_l2u_buffer_6 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_35__l0u_l1u_l2u_buffer_7 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_4_0 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_4_1 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_4_2 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_4_3 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_4_4 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_4_5 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_4_6 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_4_7 : memref<8xi32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_0 : memref<8xf32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_1 : memref<8xf32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_2 : memref<8xf32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_3 : memref<8xf32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_4_8 : memref<8xf32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_5 : memref<8xf32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_6 : memref<8xf32> = uninitialized
    memref.global @au_generated_candidate_21__l0u_l1u_buffer_7 : memref<8xf32> = uninitialized
    memref.global @au_generated_candidate_10__direct_buffer_3_0 : memref<32xf32> = uninitialized
    memref.global @au_generated_candidate_10__direct_buffer_3_1 : memref<32xf32> = uninitialized
    memref.global @au_generated_candidate_10__direct_buffer_0 : memref<32xf32> = uninitialized
    memref.global @au_generated_candidate_10__direct_buffer_1 : memref<32xf32> = uninitialized
    memref.global @au_generated_candidate_0__direct_buffer_2_0 : memref<32xf32> = uninitialized
    memref.global @au_generated_candidate_0__direct_buffer_2_1 : memref<32xf32> = uninitialized
    memref.global @au_generated_candidate_0__direct_buffer_1_0 : memref<32xf32> = uninitialized
    memref.global @au_generated_candidate_0__direct_buffer_1_1 : memref<32xf32> = uninitialized
    memref.global @au_generated_candidate_0__direct_buffer_0 : memref<32xi32> = uninitialized
    memref.global @au_generated_candidate_0__direct_buffer_1_2 : memref<32xi32> = uninitialized
    memref.global @au_generated_candidate_0__direct_local_0 : memref<f32> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_local_0"}
    memref.global @au_generated_candidate_0__direct_local : memref<i32> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_local"}
    memref.global "private" @au_generated_candidate_0__direct_csr0 : memref<i32> = uninitialized {csr_address = 2049 : i32}
    memref.global "private" @au_generated_candidate_0__direct_csr1 : memref<i32> = uninitialized {csr_address = 2050 : i32}
    memref.global "private" @au_generated_candidate_0__direct_csr2 : memref<i32> = uninitialized {csr_address = 2051 : i32}
    tor.func @au_generated_candidate_0__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 0 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/segmentation/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_1", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_2", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_0__direct_buffer_0 : memref<32xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_0__direct_buffer_1_2 : memref<32xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 27 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 27 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 0 : i64, transaction_id = 0 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<32xi32>, memref<32xi32>), i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_3", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 0, 0>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 50 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 50 : i32, ref_starttime = 31 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 31 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 1 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<32xi32>, memref<32xi32>), i32, i32
      %5 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_4", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 1, 0, 1>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 75 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 75 : i32, ref_starttime = 60 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 60 : i32, tl_channel = 0 : i32, transaction_group = 0 : i64, transaction_id = 2 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<32xi32>, memref<32xi32>), i32, i32
      %6 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_5", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%2, %3)[%c48_i32], %c16_i32 {after = array<i64: 2, 1, 2>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 91 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 91 : i32, ref_starttime = 76 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 76 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 3 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<32xi32>, memref<32xi32>), i32, i32
      %7 = aps.read_csr @au_generated_candidate_0__direct_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %8 = aps.read_csr @au_generated_candidate_0__direct_csr1 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %9 = memref.get_global @au_generated_candidate_0__direct_buffer_1_0 : memref<32xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %10 = memref.get_global @au_generated_candidate_0__direct_buffer_1_1 : memref<32xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %8, (%9, %10)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 13 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 4 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %11 = tor.addi %8 %c64_i32 on (0 to 1) {dump = "unknown_6", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %11, (%9, %10)[%c16_i32], %c16_i32 {after = array<i64: 4, 4>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 49 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 49 : i32, ref_starttime = 30 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 30 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 5 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %12 = tor.addi %8 %c128_i32 on (0 to 1) {dump = "unknown_7", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %12, (%9, %10)[%c32_i32], %c16_i32 {after = array<i64: 5, 4, 5>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 61 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 61 : i32, ref_starttime = 46 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 46 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 6 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %13 = tor.addi %8 %c192_i32 on (0 to 1) {dump = "unknown_8", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %13, (%9, %10)[%c48_i32], %c16_i32 {after = array<i64: 6, 5, 6>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 90 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 90 : i32, ref_starttime = 75 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 75 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 7 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %14 = aps.read_csr @au_generated_candidate_0__direct_csr2 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %15 = memref.get_global @au_generated_candidate_0__direct_buffer_2_0 : memref<32xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %16 = memref.get_global @au_generated_candidate_0__direct_buffer_2_1 : memref<32xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %14, (%15, %16)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 8 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %17 = tor.addi %14 %c64_i32 on (0 to 1) {dump = "unknown_9", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %17, (%15, %16)[%c16_i32], %c16_i32 {after = array<i64: 8, 8>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 35 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 35 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 16 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 9 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %18 = tor.addi %14 %c128_i32 on (0 to 1) {dump = "unknown_10", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %18, (%15, %16)[%c32_i32], %c16_i32 {after = array<i64: 9, 8, 9>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 60 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 60 : i32, ref_starttime = 45 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 45 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 10 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %19 = tor.addi %14 %c192_i32 on (0 to 1) {dump = "unknown_11", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %19, (%15, %16)[%c48_i32], %c16_i32 {after = array<i64: 10, 9, 10>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 76 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 76 : i32, ref_starttime = 61 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 4 : i64, solver_objective = 1.960000e+02 : f64, starttime = 61 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 11 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      tor.for %arg3 = (%c0_i4 : i4) to (%c7_i4 : i4) step (%c1_i4 : i4)
      on (93 to 121){
        %23 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i4 to i32
        %24 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i4 to i32
        %25 = tor.cmpi slt %24 %c0_i32 on (93 to 94) {dump = "unknown_12", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i1
        %26 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i4 to i32
        %27 = tor.subi %c-1_i32 %26 on (93 to 94) {dump = "unknown_13", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
        %28 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i4 to i32
        %29 = arith.select %25, %27, %28 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %30 = tor.cmpi slt %29 %c0_i32 on (93 to 94) {dump = "unknown_14", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i1
        %31 = arith.extui %30 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i1 to i32
        %32 = tor.addi %29 %31 on (93 to 94) {dump = "unknown_15", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
        %33 = arith.shrsi %32, %c1_i32 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %34 = tor.subi %c-1_i32 %33 on (93 to 94) {dump = "unknown_16", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
        %35 = arith.select %25, %34, %33 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %36 = arith.shli %1, %c2_i32 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %37 = tor.addi %35 %36 on (93 to 94) {dump = "unknown_17", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
        %38 = aps.read_smem %2[%37] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}], aps.mem_id = 13 : i64, endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, slot = 0 : i32, starttime = 93 : i32} : memref<32xi32>, i32 -> i32
        %39 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i4 to i32
        %40 = tor.cmpi slt %39 %c0_i32 on (93 to 94) {dump = "unknown_18", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i1
        %41 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i4 to i32
        %42 = tor.subi %c-1_i32 %41 on (93 to 94) {dump = "unknown_19", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
        %43 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i4 to i32
        %44 = arith.select %40, %42, %43 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %45 = tor.cmpi slt %44 %c0_i32 on (93 to 94) {dump = "unknown_20", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i1
        %46 = arith.extui %45 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i1 to i32
        %47 = tor.addi %44 %46 on (93 to 94) {dump = "unknown_21", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
        %48 = arith.shrsi %47, %c1_i32 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %49 = tor.subi %c-1_i32 %48 on (93 to 94) {dump = "unknown_22", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
        %50 = arith.select %40, %49, %48 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %51 = arith.shli %1, %c2_i32 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %52 = tor.addi %50 %51 on (93 to 94) {dump = "unknown_23", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
        %53 = aps.read_smem %3[%52] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}], aps.mem_id = 12 : i64, endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, slot = 0 : i32, starttime = 93 : i32} : memref<32xi32>, i32 -> i32
        %54 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i4 to i32
        %55 = arith.andi %54, %c1_i32 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %56 = tor.cmpi eq %55 %c0_i32 on (93 to 94) {dump = "unknown_24", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i1
        %57 = arith.select %56, %38, %53 {endtime = 95 : i32, ref_endtime = 95 : i32, ref_starttime = 94 : i32, starttime = 94 : i32} : i32
        aps.globalstore %57, @au_generated_candidate_0__direct_local {endtime = 95 : i32, ref_endtime = 95 : i32, ref_starttime = 94 : i32, starttime = 94 : i32} : i32
        aps.globalstore %cst_8, @au_generated_candidate_0__direct_local_0 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : f32
        %58 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i4 to i32
        tor.for %arg4 = (%c0_i2 : i2) to (%c-2_i2 : i2) step (%c1_i2 : i2)
        on (96 to 121){
          %78 = arith.extui %arg4 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i2 to i32
          %79 = tor.addi %78 %c-1_i32 on (96 to 97) {dump = "unknown_25", ref_endtime = 97 : i32, ref_starttime = 96 : i32} : (i32, i32) -> i32
          tor.for %arg5 = (%c0_i2 : i2) to (%c-2_i2 : i2) step (%c1_i2 : i2)
          on (98 to 121){
            %80 = arith.extui %arg5 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i2 to i32
            %81 = tor.addi %80 %c-1_i32 on (98 to 99) {dump = "unknown_26", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %82 = tor.addi %7 %79 on (98 to 99) {dump = "unknown_27", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %83 = tor.cmpi sge %82 %c0_i32 on (98 to 99) {dump = "unknown_28", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %84 = tor.cmpi sgt %82 %c7_i32 on (98 to 99) {dump = "unknown_29", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %85 = arith.andi %83, %84 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i1
            %86 = arith.xori %85, %true {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i1
            %87 = arith.andi %86, %83 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i1
            %88 = arith.select %85, %c7_i32, %c0_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %89 = arith.select %87, %82, %88 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %90 = tor.addi %23 %81 on (98 to 99) {dump = "unknown_30", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %91 = tor.cmpi sge %90 %c0_i32 on (98 to 99) {dump = "unknown_31", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %92 = tor.cmpi sgt %90 %c7_i32 on (98 to 99) {dump = "unknown_32", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %93 = arith.andi %91, %92 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i1
            %94 = arith.xori %93, %true {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i1
            %95 = arith.andi %94, %91 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i1
            %96 = arith.select %93, %c7_i32, %c0_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %97 = arith.select %95, %90, %96 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %98 = tor.cmpi slt %58 %c0_i32 on (98 to 99) {dump = "unknown_33", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %99 = tor.subi %c-1_i32 %58 on (98 to 99) {dump = "unknown_34", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %100 = arith.select %98, %99, %58 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %101 = tor.cmpi slt %100 %c0_i32 on (98 to 99) {dump = "unknown_35", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %102 = arith.extui %101 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i1 to i32
            %103 = tor.addi %100 %102 on (98 to 99) {dump = "unknown_36", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %104 = arith.shrsi %103, %c1_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %105 = tor.subi %c-1_i32 %104 on (98 to 99) {dump = "unknown_37", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %106 = arith.select %98, %105, %104 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %107 = arith.shli %1, %c2_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %108 = tor.addi %106 %107 on (98 to 99) {dump = "unknown_38", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %109 = aps.read_smem %9[%108] {aps.mem_id = 11 : i64, endtime = 100 : i32, ref_endtime = 100 : i32, ref_starttime = 99 : i32, slot = 0 : i32, starttime = 99 : i32} : memref<32xf32>, i32 -> f32
            %110 = tor.cmpi slt %58 %c0_i32 on (98 to 99) {dump = "unknown_39", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %111 = tor.subi %c-1_i32 %58 on (98 to 99) {dump = "unknown_40", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %112 = arith.select %110, %111, %58 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %113 = tor.cmpi slt %112 %c0_i32 on (98 to 99) {dump = "unknown_41", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %114 = arith.extui %113 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i1 to i32
            %115 = tor.addi %112 %114 on (98 to 99) {dump = "unknown_42", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %116 = arith.shrsi %115, %c1_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %117 = tor.subi %c-1_i32 %116 on (98 to 99) {dump = "unknown_43", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %118 = arith.select %110, %117, %116 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %119 = arith.shli %1, %c2_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %120 = tor.addi %118 %119 on (98 to 99) {dump = "unknown_44", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %121 = aps.read_smem %10[%120] {aps.mem_id = 10 : i64, endtime = 100 : i32, ref_endtime = 100 : i32, ref_starttime = 99 : i32, slot = 0 : i32, starttime = 99 : i32} : memref<32xf32>, i32 -> f32
            %122 = arith.andi %58, %c1_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %123 = tor.cmpi eq %122 %c0_i32 on (98 to 99) {dump = "unknown_45", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %124 = arith.select %123, %109, %121 {endtime = 101 : i32, ref_endtime = 101 : i32, ref_starttime = 100 : i32, starttime = 100 : i32} : f32
            %125 = arith.shli %89, %c3_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %126 = tor.addi %125 %97 on (98 to 99) {dump = "unknown_46", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i32
            %127 = arith.andi %126, %c1_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %128 = arith.shrui %126, %c1_i32 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : i32
            %129 = aps.read_smem %9[%128] {aps.mem_id = 9 : i64, endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, slot = 0 : i32, starttime = 98 : i32} : memref<32xf32>, i32 -> f32
            %130 = tor.cmpi eq %127 %c1_i32 on (98 to 99) {dump = "unknown_47", ref_endtime = 99 : i32, ref_starttime = 98 : i32} : (i32, i32) -> i1
            %131 = aps.read_smem %10[%128] {aps.mem_id = 8 : i64, endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, slot = 0 : i32, starttime = 98 : i32} : memref<32xf32>, i32 -> f32
            %132 = arith.select %130, %131, %129 {endtime = 100 : i32, ref_endtime = 100 : i32, ref_starttime = 99 : i32, starttime = 99 : i32} : f32
            %133 = tor.subf %124 %132 on (100 to 103) {dump = "unknown_48", fastmath = #arith.fastmath<none>, ref_endtime = 103 : i32, ref_starttime = 100 : i32} : f32
            %134 = tor.cmpf olt %133 %cst_7 on (103 to 104) {dump = "unknown_49", fastmath = #arith.fastmath<none>, ref_endtime = 104 : i32, ref_starttime = 103 : i32} : f32 -> i1
            %135 = tor.if %134 on (105 to 106) -> (f32) then {
              %146 = arith.negf %133 {endtime = 106 : i32, ref_endtime = 106 : i32, ref_starttime = 105 : i32, starttime = 105 : i32} : f32
              tor.yield %146 : f32
            } else {
              tor.yield %133 : f32
            } {dump = "unknown_50", ref_endtime = 106 : i32, ref_starttime = 105 : i32}
            %136 = arith.andi %126, %c1_i32 {endtime = 108 : i32, ref_endtime = 108 : i32, ref_starttime = 107 : i32, starttime = 107 : i32} : i32
            %137 = arith.shrui %126, %c1_i32 {endtime = 108 : i32, ref_endtime = 108 : i32, ref_starttime = 107 : i32, starttime = 107 : i32} : i32
            %138 = aps.read_smem %15[%137] {aps.mem_id = 7 : i64, endtime = 108 : i32, ref_endtime = 108 : i32, ref_starttime = 107 : i32, slot = 0 : i32, starttime = 107 : i32} : memref<32xf32>, i32 -> f32
            %139 = tor.cmpi eq %136 %c1_i32 on (107 to 108) {dump = "unknown_51", ref_endtime = 108 : i32, ref_starttime = 107 : i32} : (i32, i32) -> i1
            %140 = aps.read_smem %16[%137] {aps.mem_id = 6 : i64, endtime = 108 : i32, ref_endtime = 108 : i32, ref_starttime = 107 : i32, slot = 0 : i32, starttime = 107 : i32} : memref<32xf32>, i32 -> f32
            %141 = arith.select %139, %140, %138 {endtime = 109 : i32, ref_endtime = 109 : i32, ref_starttime = 108 : i32, starttime = 108 : i32} : f32
            %142 = tor.addf %135 %141 on (108 to 111) {dump = "unknown_52", fastmath = #arith.fastmath<none>, ref_endtime = 111 : i32, ref_starttime = 108 : i32} : f32
            %143 = aps.globalload @au_generated_candidate_0__direct_local_0 {endtime = 108 : i32, ref_endtime = 108 : i32, ref_starttime = 107 : i32, starttime = 107 : i32} : f32
            %144 = tor.cmpf olt %142 %143 on (111 to 112) {dump = "unknown_53", fastmath = #arith.fastmath<none>, ref_endtime = 112 : i32, ref_starttime = 111 : i32} : f32 -> i1
            %145 = tor.if %144 on (113 to 117) -> (i1) then {
              tor.yield %true : i1
            } else {
              %146 = tor.cmpf oeq %142 %143 on (113 to 114) {dump = "unknown_55", fastmath = #arith.fastmath<none>, ref_endtime = 114 : i32, ref_starttime = 113 : i32} : f32 -> i1
              %147 = tor.if %146 on (115 to 117) -> (i1) then {
                %148 = arith.andi %126, %c1_i32 {endtime = 116 : i32, ref_endtime = 116 : i32, ref_starttime = 115 : i32, starttime = 115 : i32} : i32
                %149 = arith.shrui %126, %c1_i32 {endtime = 116 : i32, ref_endtime = 116 : i32, ref_starttime = 115 : i32, starttime = 115 : i32} : i32
                %150 = aps.read_smem %2[%149] {aps.mem_id = 5 : i64, endtime = 116 : i32, ref_endtime = 116 : i32, ref_starttime = 115 : i32, slot = 0 : i32, starttime = 115 : i32} : memref<32xi32>, i32 -> i32
                %151 = tor.cmpi eq %148 %c1_i32 on (115 to 116) {dump = "unknown_57", ref_endtime = 116 : i32, ref_starttime = 115 : i32} : (i32, i32) -> i1
                %152 = aps.read_smem %3[%149] {aps.mem_id = 4 : i64, endtime = 116 : i32, ref_endtime = 116 : i32, ref_starttime = 115 : i32, slot = 0 : i32, starttime = 115 : i32} : memref<32xi32>, i32 -> i32
                %153 = arith.select %151, %152, %150 {endtime = 117 : i32, ref_endtime = 117 : i32, ref_starttime = 116 : i32, starttime = 116 : i32} : i32
                %154 = aps.globalload @au_generated_candidate_0__direct_local {endtime = 116 : i32, ref_endtime = 116 : i32, ref_starttime = 115 : i32, starttime = 115 : i32} : i32
                %155 = tor.cmpi slt %153 %154 on (116 to 117) {dump = "unknown_58", ref_endtime = 117 : i32, ref_starttime = 116 : i32} : (i32, i32) -> i1
                tor.yield %155 : i1
              } else {
                tor.yield %false : i1
              } {dump = "unknown_56", ref_endtime = 117 : i32, ref_starttime = 115 : i32}
              tor.yield %147 : i1
            } {dump = "unknown_54", ref_endtime = 117 : i32, ref_starttime = 113 : i32}
            tor.if %145 on (119 to 121) then {
              aps.globalstore %142, @au_generated_candidate_0__direct_local_0 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : f32
              %146 = arith.andi %126, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
              %147 = arith.shrui %126, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
              %148 = aps.read_smem %2[%147] {aps.mem_id = 3 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : memref<32xi32>, i32 -> i32
              %149 = tor.cmpi eq %146 %c1_i32 on (119 to 120) {dump = "unknown_60", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
              %150 = aps.read_smem %3[%147] {aps.mem_id = 2 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : memref<32xi32>, i32 -> i32
              %151 = arith.select %149, %150, %148 {endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, starttime = 120 : i32} : i32
              aps.globalstore %151, @au_generated_candidate_0__direct_local {endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, starttime = 120 : i32} : i32
            } {dump = "unknown_59", ref_endtime = 121 : i32, ref_starttime = 119 : i32}
          } {ref_endtime = 121 : i32, ref_starttime = 98 : i32}
        } {ref_endtime = 121 : i32, ref_starttime = 96 : i32}
        %59 = aps.globalload @au_generated_candidate_0__direct_local {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i32
        %60 = arith.extui %arg3 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i4 to i32
        %61 = arith.andi %60, %c1_i32 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i32
        %62 = arith.extui %arg3 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i4 to i32
        %63 = tor.cmpi slt %62 %c0_i32 on (96 to 97) {dump = "unknown_61", ref_endtime = 97 : i32, ref_starttime = 96 : i32} : (i32, i32) -> i1
        %64 = arith.extui %arg3 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i4 to i32
        %65 = tor.subi %c-1_i32 %64 on (96 to 97) {dump = "unknown_62", ref_endtime = 97 : i32, ref_starttime = 96 : i32} : (i32, i32) -> i32
        %66 = arith.extui %arg3 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i4 to i32
        %67 = arith.select %63, %65, %66 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i32
        %68 = tor.cmpi slt %67 %c0_i32 on (96 to 97) {dump = "unknown_63", ref_endtime = 97 : i32, ref_starttime = 96 : i32} : (i32, i32) -> i1
        %69 = arith.extui %68 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i1 to i32
        %70 = tor.addi %67 %69 on (96 to 97) {dump = "unknown_64", ref_endtime = 97 : i32, ref_starttime = 96 : i32} : (i32, i32) -> i32
        %71 = arith.shrsi %70, %c1_i32 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i32
        %72 = tor.subi %c-1_i32 %71 on (96 to 97) {dump = "unknown_65", ref_endtime = 97 : i32, ref_starttime = 96 : i32} : (i32, i32) -> i32
        %73 = arith.select %63, %72, %71 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i32
        %74 = arith.shli %1, %c2_i32 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : i32
        %75 = tor.addi %73 %74 on (96 to 97) {dump = "unknown_66", ref_endtime = 97 : i32, ref_starttime = 96 : i32} : (i32, i32) -> i32
        %76 = tor.cmpi eq %61 %c0_i32 on (96 to 97) {dump = "unknown_67", ref_endtime = 97 : i32, ref_starttime = 96 : i32} : (i32, i32) -> i1
        aps.write_smem_if %76, %59, %2[%75] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 13 : i64}], aps.mem_id = 1 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : i32, memref<32xi32>, i32
        %77 = tor.cmpi eq %61 %c1_i32 on (96 to 97) {dump = "unknown_68", ref_endtime = 97 : i32, ref_starttime = 96 : i32} : (i32, i32) -> i1
        aps.write_smem_if %77, %59, %3[%75] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 12 : i64}], aps.mem_id = 0 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : i32, memref<32xi32>, i32
      } {ref_endtime = 121 : i32, ref_starttime = 93 : i32}
      aps.copy_by @busitfc, (%2, %3)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 129 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 39 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 129 : i32, ref_starttime = 120 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 4 : i64, solver_objective = 7.100000e+01 : f64, starttime = 120 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 12 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<32xi32>, memref<32xi32>), i32, i32, i32
      %20 = tor.addi %0 %c64_i32 on (93 to 94) {dump = "unknown_69", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3)[%c16_i32], %20, %c16_i32 {after = array<i64: 12, 12>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 139 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 39 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 139 : i32, ref_starttime = 121 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 4 : i64, solver_objective = 7.100000e+01 : f64, starttime = 121 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 13 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<32xi32>, memref<32xi32>), i32, i32, i32
      %21 = tor.addi %0 %c128_i32 on (93 to 94) {dump = "unknown_70", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3)[%c32_i32], %21, %c16_i32 {after = array<i64: 13, 12, 13>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 154 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 39 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 154 : i32, ref_starttime = 135 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 4 : i64, solver_objective = 7.100000e+01 : f64, starttime = 135 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 14 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<32xi32>, memref<32xi32>), i32, i32, i32
      %22 = tor.addi %0 %c192_i32 on (93 to 94) {dump = "unknown_71", ref_endtime = 94 : i32, ref_starttime = 93 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3)[%c48_i32], %22, %c16_i32 {after = array<i64: 14, 13, 14>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 164 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 20 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 39 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 164 : i32, ref_starttime = 145 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 4 : i64, solver_objective = 7.100000e+01 : f64, starttime = 145 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 15 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<32xi32>, memref<32xi32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_10__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 1 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/segmentation/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_72", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_73", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_10__direct_buffer_0 : memref<32xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_10__direct_buffer_1 : memref<32xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 4 : i64, solver_objective = 6.800000e+01 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 16 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_74", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 16, 16>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 20 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 20 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 4 : i64, solver_objective = 6.800000e+01 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 17 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %5 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_75", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 17, 16, 17>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 30 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 30 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 4 : i64, solver_objective = 6.800000e+01 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 18 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %6 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_76", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%2, %3)[%c48_i32], %c16_i32 {after = array<i64: 18, 17, 18>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 38 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 38 : i32, ref_starttime = 23 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 4 : i64, solver_objective = 6.800000e+01 : f64, starttime = 23 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 19 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<32xf32>, memref<32xf32>), i32, i32
      %7 = memref.get_global @au_generated_candidate_10__direct_buffer_3_0 : memref<32xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %8 = memref.get_global @au_generated_candidate_10__direct_buffer_3_1 : memref<32xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      tor.for %arg3 = (%c0_i4 : i4) to (%c7_i4 : i4) step (%c1_i4 : i4)
      on (40 to 64){
        %12 = arith.extui %arg3 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i4 to i32
        %13 = arith.extui %arg3 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i4 to i32
        tor.for %arg4 = (%c0_i4 : i4) to (%c7_i4 : i4) step (%c1_i4 : i4)
        on (42 to 64){
          %14 = arith.extui %arg4 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i4 to i32
          %15 = tor.addi %14 %c-1_i32 on (42 to 43) {dump = "unknown_77", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i32
          %16 = tor.cmpi sge %15 %c0_i32 on (42 to 43) {dump = "unknown_78", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %17 = tor.cmpi sgt %15 %c7_i32 on (42 to 43) {dump = "unknown_79", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %18 = arith.andi %16, %17 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %19 = arith.xori %18, %true {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %20 = arith.andi %19, %16 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %21 = arith.select %18, %c7_i32, %c0_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %22 = arith.select %20, %15, %21 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %23 = tor.addi %14 %c1_i32 on (42 to 43) {dump = "unknown_80", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i32
          %24 = tor.cmpi sge %23 %c0_i32 on (42 to 43) {dump = "unknown_81", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %25 = tor.cmpi sgt %23 %c7_i32 on (42 to 43) {dump = "unknown_82", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %26 = arith.andi %24, %25 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %27 = arith.xori %26, %true {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %28 = arith.andi %27, %24 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %29 = arith.select %26, %c7_i32, %c0_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %30 = arith.select %28, %23, %29 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %31 = tor.addi %12 %c-1_i32 on (42 to 43) {dump = "unknown_83", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i32
          %32 = tor.cmpi sge %31 %c0_i32 on (42 to 43) {dump = "unknown_84", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %33 = tor.cmpi sgt %31 %c7_i32 on (42 to 43) {dump = "unknown_85", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %34 = arith.andi %32, %33 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %35 = arith.xori %34, %true {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %36 = arith.andi %35, %32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %37 = arith.select %34, %c7_i32, %c0_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %38 = arith.select %36, %31, %37 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %39 = tor.addi %12 %c1_i32 on (42 to 43) {dump = "unknown_86", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i32
          %40 = tor.cmpi sge %39 %c0_i32 on (42 to 43) {dump = "unknown_87", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %41 = tor.cmpi sgt %39 %c7_i32 on (42 to 43) {dump = "unknown_88", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %42 = arith.andi %40, %41 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %43 = tor.subi %true %42 on (42 to 43) {dump = "unknown_89", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i1, i1) -> i1
          %44 = arith.andi %43, %40 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i1
          %45 = arith.select %42, %c7_i32, %c0_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %46 = arith.select %44, %39, %45 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %47 = arith.shli %12, %c3_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %48 = tor.addi %47 %30 on (42 to 43) {dump = "unknown_90", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i32
          %49 = arith.andi %48, %c1_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %50 = arith.shrui %48, %c1_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %51 = aps.read_smem %2[%50] {aps.mem_id = 9 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<32xf32>, i32 -> f32
          %52 = tor.cmpi eq %49 %c1_i32 on (42 to 43) {dump = "unknown_91", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %53 = aps.read_smem %3[%50] {aps.mem_id = 8 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<32xf32>, i32 -> f32
          %54 = arith.select %52, %53, %51 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : f32
          %55 = tor.addi %47 %22 on (42 to 43) {dump = "unknown_92", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i32
          %56 = arith.andi %55, %c1_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %57 = arith.shrui %55, %c1_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
          %58 = aps.read_smem %2[%57] {aps.mem_id = 7 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<32xf32>, i32 -> f32
          %59 = tor.cmpi eq %56 %c1_i32 on (42 to 43) {dump = "unknown_93", ref_endtime = 43 : i32, ref_starttime = 42 : i32} : (i32, i32) -> i1
          %60 = aps.read_smem %3[%57] {aps.mem_id = 6 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<32xf32>, i32 -> f32
          %61 = arith.select %59, %60, %58 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : f32
          %62 = tor.subf %54 %61 on (44 to 47) {dump = "unknown_94", fastmath = #arith.fastmath<none>, ref_endtime = 47 : i32, ref_starttime = 44 : i32} : f32
          %63 = tor.cmpf olt %62 %cst_7 on (47 to 48) {dump = "unknown_95", fastmath = #arith.fastmath<none>, ref_endtime = 48 : i32, ref_starttime = 47 : i32} : f32 -> i1
          %64 = tor.if %63 on (49 to 50) -> (f32) then {
            %103 = arith.negf %62 {endtime = 50 : i32, ref_endtime = 50 : i32, ref_starttime = 49 : i32, starttime = 49 : i32} : f32
            tor.yield %103 : f32
          } else {
            tor.yield %62 : f32
          } {dump = "unknown_96", ref_endtime = 50 : i32, ref_starttime = 49 : i32}
          %65 = arith.shli %46, %c3_i32 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : i32
          %66 = tor.addi %65 %14 on (51 to 52) {dump = "unknown_97", ref_endtime = 52 : i32, ref_starttime = 51 : i32} : (i32, i32) -> i32
          %67 = arith.andi %66, %c1_i32 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : i32
          %68 = arith.shrui %66, %c1_i32 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : i32
          %69 = aps.read_smem %2[%68] {aps.mem_id = 5 : i64, endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, slot = 0 : i32, starttime = 52 : i32} : memref<32xf32>, i32 -> f32
          %70 = tor.cmpi eq %67 %c1_i32 on (51 to 52) {dump = "unknown_98", ref_endtime = 52 : i32, ref_starttime = 51 : i32} : (i32, i32) -> i1
          %71 = aps.read_smem %3[%68] {aps.mem_id = 4 : i64, endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, slot = 0 : i32, starttime = 52 : i32} : memref<32xf32>, i32 -> f32
          %72 = arith.select %70, %71, %69 {endtime = 54 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, starttime = 53 : i32} : f32
          %73 = arith.shli %38, %c3_i32 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : i32
          %74 = tor.addi %73 %14 on (51 to 52) {dump = "unknown_99", ref_endtime = 52 : i32, ref_starttime = 51 : i32} : (i32, i32) -> i32
          %75 = arith.andi %74, %c1_i32 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : i32
          %76 = arith.shrui %74, %c1_i32 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : i32
          %77 = aps.read_smem %2[%76] {aps.mem_id = 3 : i64, endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, slot = 0 : i32, starttime = 51 : i32} : memref<32xf32>, i32 -> f32
          %78 = tor.cmpi eq %75 %c1_i32 on (51 to 52) {dump = "unknown_100", ref_endtime = 52 : i32, ref_starttime = 51 : i32} : (i32, i32) -> i1
          %79 = aps.read_smem %3[%76] {aps.mem_id = 2 : i64, endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, slot = 0 : i32, starttime = 51 : i32} : memref<32xf32>, i32 -> f32
          %80 = arith.select %78, %79, %77 {endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, starttime = 52 : i32} : f32
          %81 = tor.subf %72 %80 on (53 to 56) {dump = "unknown_101", fastmath = #arith.fastmath<none>, ref_endtime = 56 : i32, ref_starttime = 53 : i32} : f32
          %82 = tor.cmpf olt %81 %cst_7 on (56 to 57) {dump = "unknown_102", fastmath = #arith.fastmath<none>, ref_endtime = 57 : i32, ref_starttime = 56 : i32} : f32 -> i1
          %83 = tor.if %82 on (58 to 59) -> (f32) then {
            %103 = arith.negf %81 {endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, starttime = 58 : i32} : f32
            tor.yield %103 : f32
          } else {
            tor.yield %81 : f32
          } {dump = "unknown_103", ref_endtime = 59 : i32, ref_starttime = 58 : i32}
          %84 = tor.addf %64 %83 on (60 to 63) {dump = "unknown_104", fastmath = #arith.fastmath<none>, ref_endtime = 63 : i32, ref_starttime = 60 : i32} : f32
          %85 = arith.extui %arg4 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i4 to i32
          %86 = arith.andi %85, %c1_i32 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i32
          %87 = arith.shli %13, %c2_i32 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i32
          %88 = arith.extui %arg4 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i4 to i32
          %89 = tor.cmpi slt %88 %c0_i32 on (60 to 61) {dump = "unknown_105", ref_endtime = 61 : i32, ref_starttime = 60 : i32} : (i32, i32) -> i1
          %90 = arith.extui %arg4 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i4 to i32
          %91 = tor.subi %c-1_i32 %90 on (60 to 61) {dump = "unknown_106", ref_endtime = 61 : i32, ref_starttime = 60 : i32} : (i32, i32) -> i32
          %92 = arith.extui %arg4 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i4 to i32
          %93 = arith.select %89, %91, %92 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i32
          %94 = tor.cmpi slt %93 %c0_i32 on (60 to 61) {dump = "unknown_107", ref_endtime = 61 : i32, ref_starttime = 60 : i32} : (i32, i32) -> i1
          %95 = arith.extui %94 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i1 to i32
          %96 = tor.addi %93 %95 on (60 to 61) {dump = "unknown_108", ref_endtime = 61 : i32, ref_starttime = 60 : i32} : (i32, i32) -> i32
          %97 = arith.shrsi %96, %c1_i32 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i32
          %98 = tor.subi %c-1_i32 %97 on (60 to 61) {dump = "unknown_109", ref_endtime = 61 : i32, ref_starttime = 60 : i32} : (i32, i32) -> i32
          %99 = arith.select %89, %98, %97 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : i32
          %100 = tor.addi %87 %99 on (60 to 61) {dump = "unknown_110", ref_endtime = 61 : i32, ref_starttime = 60 : i32} : (i32, i32) -> i32
          %101 = tor.cmpi eq %86 %c0_i32 on (60 to 61) {dump = "unknown_111", ref_endtime = 61 : i32, ref_starttime = 60 : i32} : (i32, i32) -> i1
          aps.write_smem_if %101, %84, %7[%100] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 1 : i64}], aps.mem_id = 1 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<32xf32>, i32
          %102 = tor.cmpi eq %86 %c1_i32 on (60 to 61) {dump = "unknown_112", ref_endtime = 61 : i32, ref_starttime = 60 : i32} : (i32, i32) -> i1
          aps.write_smem_if %102, %84, %8[%100] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 0 : i64}], aps.mem_id = 0 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<32xf32>, i32
        } {ref_endtime = 64 : i32, ref_starttime = 42 : i32}
      } {ref_endtime = 64 : i32, ref_starttime = 40 : i32}
      aps.copy_by @busitfc, (%7, %8)[%c0_i32], %1, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 73 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 39 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 73 : i32, ref_starttime = 64 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 4 : i64, solver_objective = 7.100000e+01 : f64, starttime = 64 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 20 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<32xf32>, memref<32xf32>), i32, i32, i32
      %9 = tor.addi %1 %c64_i32 on (40 to 41) {dump = "unknown_113", ref_endtime = 41 : i32, ref_starttime = 40 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%7, %8)[%c16_i32], %9, %c16_i32 {after = array<i64: 20, 20>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 83 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 39 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 83 : i32, ref_starttime = 65 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 4 : i64, solver_objective = 7.100000e+01 : f64, starttime = 65 : i32, tl_channel = 1 : i32, transaction_group = 5 : i64, transaction_id = 21 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<32xf32>, memref<32xf32>), i32, i32, i32
      %10 = tor.addi %1 %c128_i32 on (40 to 41) {dump = "unknown_114", ref_endtime = 41 : i32, ref_starttime = 40 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%7, %8)[%c32_i32], %10, %c16_i32 {after = array<i64: 21, 20, 21>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 98 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 39 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 98 : i32, ref_starttime = 79 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 4 : i64, solver_objective = 7.100000e+01 : f64, starttime = 79 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 22 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<32xf32>, memref<32xf32>), i32, i32, i32
      %11 = tor.addi %1 %c192_i32 on (40 to 41) {dump = "unknown_115", ref_endtime = 41 : i32, ref_starttime = 40 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%7, %8)[%c48_i32], %11, %c16_i32 {after = array<i64: 22, 21, 22>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 108 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 20 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 39 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 108 : i32, ref_starttime = 89 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 4 : i64, solver_objective = 7.100000e+01 : f64, starttime = 89 : i32, tl_channel = 1 : i32, transaction_group = 5 : i64, transaction_id = 23 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<32xf32>, memref<32xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_21__l0u_l1u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 2 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/segmentation/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_116", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_117", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_0 : memref<8xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_1 : memref<8xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %4 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_2 : memref<8xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %5 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_3 : memref<8xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %6 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_4_8 : memref<8xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %7 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_5 : memref<8xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %8 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_6 : memref<8xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %9 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_7 : memref<8xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %10 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_4_0 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %11 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_4_1 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %12 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_4_2 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %13 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_4_3 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %14 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_4_4 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %15 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_4_5 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %16 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_4_6 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %17 = memref.get_global @au_generated_candidate_21__l0u_l1u_buffer_4_7 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.write_smem %cst_7, %2[%c0_i32] {aps.mem_id = 127 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c0_i32, %10[%c0_i32] {aps.mem_id = 126 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst, %3[%c0_i32] {aps.mem_id = 125 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c1_i32, %11[%c0_i32] {aps.mem_id = 124 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_0, %4[%c0_i32] {aps.mem_id = 123 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c2_i32, %12[%c0_i32] {aps.mem_id = 122 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_1, %5[%c0_i32] {aps.mem_id = 121 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c3_i32, %13[%c0_i32] {aps.mem_id = 120 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_2, %6[%c0_i32] {aps.mem_id = 119 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c4_i32, %14[%c0_i32] {aps.mem_id = 118 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_3, %7[%c0_i32] {aps.mem_id = 117 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c5_i32, %15[%c0_i32] {aps.mem_id = 116 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_4, %8[%c0_i32] {aps.mem_id = 115 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c6_i32, %16[%c0_i32] {aps.mem_id = 114 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_5, %9[%c0_i32] {aps.mem_id = 113 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c7_i32, %17[%c0_i32] {aps.mem_id = 112 : i64, endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, slot = 0 : i32, starttime = 0 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_0, %2[%c1_i32] {aps.mem_id = 111 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c8_i32, %10[%c1_i32] {aps.mem_id = 110 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_1, %3[%c1_i32] {aps.mem_id = 109 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c9_i32, %11[%c1_i32] {aps.mem_id = 108 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_2, %4[%c1_i32] {aps.mem_id = 107 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c10_i32, %12[%c1_i32] {aps.mem_id = 106 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_3, %5[%c1_i32] {aps.mem_id = 105 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c11_i32, %13[%c1_i32] {aps.mem_id = 104 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_4, %6[%c1_i32] {aps.mem_id = 103 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c12_i32, %14[%c1_i32] {aps.mem_id = 102 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_5, %7[%c1_i32] {aps.mem_id = 101 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c13_i32, %15[%c1_i32] {aps.mem_id = 100 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_6, %8[%c1_i32] {aps.mem_id = 99 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c14_i32, %16[%c1_i32] {aps.mem_id = 98 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_7, %9[%c1_i32] {aps.mem_id = 97 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c15_i32, %17[%c1_i32] {aps.mem_id = 96 : i64, endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, slot = 0 : i32, starttime = 1 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_2, %2[%c2_i32] {aps.mem_id = 95 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c16_i32, %10[%c2_i32] {aps.mem_id = 94 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_3, %3[%c2_i32] {aps.mem_id = 93 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c17_i32, %11[%c2_i32] {aps.mem_id = 92 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_4, %4[%c2_i32] {aps.mem_id = 91 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c18_i32, %12[%c2_i32] {aps.mem_id = 90 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_5, %5[%c2_i32] {aps.mem_id = 89 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c19_i32, %13[%c2_i32] {aps.mem_id = 88 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_6, %6[%c2_i32] {aps.mem_id = 87 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c20_i32, %14[%c2_i32] {aps.mem_id = 86 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_7, %7[%c2_i32] {aps.mem_id = 85 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c21_i32, %15[%c2_i32] {aps.mem_id = 84 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst, %8[%c2_i32] {aps.mem_id = 83 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c22_i32, %16[%c2_i32] {aps.mem_id = 82 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_0, %9[%c2_i32] {aps.mem_id = 81 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c23_i32, %17[%c2_i32] {aps.mem_id = 80 : i64, endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, slot = 0 : i32, starttime = 2 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_4, %2[%c3_i32] {aps.mem_id = 79 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c24_i32, %10[%c3_i32] {aps.mem_id = 78 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_5, %3[%c3_i32] {aps.mem_id = 77 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c25_i32, %11[%c3_i32] {aps.mem_id = 76 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_6, %4[%c3_i32] {aps.mem_id = 75 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c26_i32, %12[%c3_i32] {aps.mem_id = 74 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_7, %5[%c3_i32] {aps.mem_id = 73 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c27_i32, %13[%c3_i32] {aps.mem_id = 72 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst, %6[%c3_i32] {aps.mem_id = 71 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c28_i32, %14[%c3_i32] {aps.mem_id = 70 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_0, %7[%c3_i32] {aps.mem_id = 69 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c29_i32, %15[%c3_i32] {aps.mem_id = 68 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_1, %8[%c3_i32] {aps.mem_id = 67 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c30_i32, %16[%c3_i32] {aps.mem_id = 66 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_2, %9[%c3_i32] {aps.mem_id = 65 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c31_i32, %17[%c3_i32] {aps.mem_id = 64 : i64, endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, slot = 0 : i32, starttime = 3 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_6, %2[%c4_i32] {aps.mem_id = 63 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c32_i32, %10[%c4_i32] {aps.mem_id = 62 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_7, %3[%c4_i32] {aps.mem_id = 61 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c33_i32, %11[%c4_i32] {aps.mem_id = 60 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst, %4[%c4_i32] {aps.mem_id = 59 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c34_i32, %12[%c4_i32] {aps.mem_id = 58 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_0, %5[%c4_i32] {aps.mem_id = 57 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c35_i32, %13[%c4_i32] {aps.mem_id = 56 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_1, %6[%c4_i32] {aps.mem_id = 55 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c36_i32, %14[%c4_i32] {aps.mem_id = 54 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_2, %7[%c4_i32] {aps.mem_id = 53 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c37_i32, %15[%c4_i32] {aps.mem_id = 52 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_3, %8[%c4_i32] {aps.mem_id = 51 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c38_i32, %16[%c4_i32] {aps.mem_id = 50 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_4, %9[%c4_i32] {aps.mem_id = 49 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c39_i32, %17[%c4_i32] {aps.mem_id = 48 : i64, endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, slot = 0 : i32, starttime = 4 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst, %2[%c5_i32] {aps.mem_id = 47 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c40_i32, %10[%c5_i32] {aps.mem_id = 46 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_0, %3[%c5_i32] {aps.mem_id = 45 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c41_i32, %11[%c5_i32] {aps.mem_id = 44 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_1, %4[%c5_i32] {aps.mem_id = 43 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c42_i32, %12[%c5_i32] {aps.mem_id = 42 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_2, %5[%c5_i32] {aps.mem_id = 41 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c43_i32, %13[%c5_i32] {aps.mem_id = 40 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_3, %6[%c5_i32] {aps.mem_id = 39 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c44_i32, %14[%c5_i32] {aps.mem_id = 38 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_4, %7[%c5_i32] {aps.mem_id = 37 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c45_i32, %15[%c5_i32] {aps.mem_id = 36 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_5, %8[%c5_i32] {aps.mem_id = 35 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c46_i32, %16[%c5_i32] {aps.mem_id = 34 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_6, %9[%c5_i32] {aps.mem_id = 33 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c47_i32, %17[%c5_i32] {aps.mem_id = 32 : i64, endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, slot = 0 : i32, starttime = 5 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_1, %2[%c6_i32] {aps.mem_id = 31 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c48_i32, %10[%c6_i32] {aps.mem_id = 30 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_2, %3[%c6_i32] {aps.mem_id = 29 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c49_i32, %11[%c6_i32] {aps.mem_id = 28 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_3, %4[%c6_i32] {aps.mem_id = 27 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c50_i32, %12[%c6_i32] {aps.mem_id = 26 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_4, %5[%c6_i32] {aps.mem_id = 25 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c51_i32, %13[%c6_i32] {aps.mem_id = 24 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_5, %6[%c6_i32] {aps.mem_id = 23 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c52_i32, %14[%c6_i32] {aps.mem_id = 22 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_6, %7[%c6_i32] {aps.mem_id = 21 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c53_i32, %15[%c6_i32] {aps.mem_id = 20 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_7, %8[%c6_i32] {aps.mem_id = 19 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c54_i32, %16[%c6_i32] {aps.mem_id = 18 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst, %9[%c6_i32] {aps.mem_id = 17 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c55_i32, %17[%c6_i32] {aps.mem_id = 16 : i64, endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, slot = 0 : i32, starttime = 6 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_3, %2[%c7_i32] {aps.mem_id = 15 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c56_i32, %10[%c7_i32] {aps.mem_id = 14 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_4, %3[%c7_i32] {aps.mem_id = 13 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c57_i32, %11[%c7_i32] {aps.mem_id = 12 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_5, %4[%c7_i32] {aps.mem_id = 11 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c58_i32, %12[%c7_i32] {aps.mem_id = 10 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_6, %5[%c7_i32] {aps.mem_id = 9 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c59_i32, %13[%c7_i32] {aps.mem_id = 8 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_7, %6[%c7_i32] {aps.mem_id = 7 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c60_i32, %14[%c7_i32] {aps.mem_id = 6 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst, %7[%c7_i32] {aps.mem_id = 5 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c61_i32, %15[%c7_i32] {aps.mem_id = 4 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_0, %8[%c7_i32] {aps.mem_id = 3 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c62_i32, %16[%c7_i32] {aps.mem_id = 2 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<8xi32>, i32
      aps.write_smem %cst_1, %9[%c7_i32] {aps.mem_id = 1 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : f32, memref<8xf32>, i32
      aps.write_smem %c63_i32, %17[%c7_i32] {aps.mem_id = 0 : i64, endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<8xi32>, i32
      aps.copy_by @busitfc, (%2, %3, %4, %5, %6, %7, %8, %9)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 17 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 79 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 17 : i32, ref_starttime = 8 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 4 : i64, solver_objective = 1.430000e+02 : f64, starttime = 8 : i32, tl_channel = 0 : i32, transaction_group = 6 : i64, transaction_id = 24 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>), i32, i32, i32
      %18 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_118", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3, %4, %5, %6, %7, %8, %9)[%c16_i32], %18, %c16_i32 {after = array<i64: 24, 24>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 27 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 79 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 27 : i32, ref_starttime = 9 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 4 : i64, solver_objective = 1.430000e+02 : f64, starttime = 9 : i32, tl_channel = 1 : i32, transaction_group = 6 : i64, transaction_id = 25 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>), i32, i32, i32
      %19 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_119", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3, %4, %5, %6, %7, %8, %9)[%c32_i32], %19, %c16_i32 {after = array<i64: 25, 24, 25>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 42 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 79 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 42 : i32, ref_starttime = 23 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 4 : i64, solver_objective = 1.430000e+02 : f64, starttime = 23 : i32, tl_channel = 0 : i32, transaction_group = 6 : i64, transaction_id = 26 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>), i32, i32, i32
      %20 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_120", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3, %4, %5, %6, %7, %8, %9)[%c48_i32], %20, %c16_i32 {after = array<i64: 26, 25, 26>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 52 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 20 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 79 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 52 : i32, ref_starttime = 33 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 4 : i64, solver_objective = 1.430000e+02 : f64, starttime = 33 : i32, tl_channel = 1 : i32, transaction_group = 6 : i64, transaction_id = 27 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>), i32, i32, i32
      aps.copy_by @busitfc, (%10, %11, %12, %13, %14, %15, %16, %17)[%c0_i32], %1, %c16_i32 {after = array<i64: 27, 27, 26, 27>, after_gap = array<i64: 20, 20, 20, 20>, endtime = 62 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 49 : i64, memory_issue_cycle = 40 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 79 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 62 : i32, ref_starttime = 53 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 4 : i64, solver_objective = 1.430000e+02 : f64, starttime = 53 : i32, tl_channel = 0 : i32, transaction_group = 7 : i64, transaction_id = 28 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>), i32, i32, i32
      %21 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_121", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%10, %11, %12, %13, %14, %15, %16, %17)[%c16_i32], %21, %c16_i32 {after = array<i64: 28, 27, 28>, after_gap = array<i64: 1, 20, 1>, after_segment = 0 : i64, endtime = 72 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 59 : i64, memory_issue_cycle = 41 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 79 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 72 : i32, ref_starttime = 54 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 4 : i64, solver_objective = 1.430000e+02 : f64, starttime = 54 : i32, tl_channel = 1 : i32, transaction_group = 7 : i64, transaction_id = 29 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>), i32, i32, i32
      %22 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_122", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%10, %11, %12, %13, %14, %15, %16, %17)[%c32_i32], %22, %c16_i32 {after = array<i64: 29, 28, 29>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 87 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 69 : i64, memory_issue_cycle = 50 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 79 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 87 : i32, ref_starttime = 68 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 4 : i64, solver_objective = 1.430000e+02 : f64, starttime = 68 : i32, tl_channel = 0 : i32, transaction_group = 7 : i64, transaction_id = 30 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>), i32, i32, i32
      %23 = tor.addi %1 %c192_i32 on (0 to 1) {dump = "unknown_123", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%10, %11, %12, %13, %14, %15, %16, %17)[%c48_i32], %23, %c16_i32 {after = array<i64: 30, 29, 30>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 97 : i32, estimated_latency = 39 : i64, interface_cost = 71 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 79 : i64, memory_issue_cycle = 60 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 79 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 97 : i32, ref_starttime = 78 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 4 : i64, solver_objective = 1.430000e+02 : f64, starttime = 78 : i32, tl_channel = 1 : i32, transaction_group = 7 : i64, transaction_id = 31 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_35__l0u_l1u_l2u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 3 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/segmentation/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg1 {dump = "unknown_124", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg0 {dump = "unknown_125", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_35__l0u_l1u_l2u_buffer_0 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_35__l0u_l1u_l2u_buffer_1 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %4 = memref.get_global @au_generated_candidate_35__l0u_l1u_l2u_buffer_2 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %5 = memref.get_global @au_generated_candidate_35__l0u_l1u_l2u_buffer_3 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %6 = memref.get_global @au_generated_candidate_35__l0u_l1u_l2u_buffer_4 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %7 = memref.get_global @au_generated_candidate_35__l0u_l1u_l2u_buffer_5_0 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %8 = memref.get_global @au_generated_candidate_35__l0u_l1u_l2u_buffer_6 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %9 = memref.get_global @au_generated_candidate_35__l0u_l1u_l2u_buffer_7 : memref<8xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%2, %3, %4, %5, %6, %7, %8, %9)[%c0_i32], %c16_i32 {after = array<i64: 32>, after_gap = array<i64: 1>, endtime = 13 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 13 : i64, memory_issue_cycle = 1 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 37 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 4 : i64, solver_objective = 8.600000e+01 : f64, starttime = 1 : i32, tl_channel = 0 : i32, transaction_group = 8 : i64, transaction_id = 33 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>), i32, i32
      %10 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_126", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %10, (%2, %3, %4, %5, %6, %7, %8, %9)[%c16_i32], %c16_i32 {after = array<i64: 33, 33>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 21 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 21 : i64, memory_issue_cycle = 2 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 37 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 21 : i32, ref_starttime = 2 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 4 : i64, solver_objective = 8.600000e+01 : f64, starttime = 2 : i32, tl_channel = 1 : i32, transaction_group = 8 : i64, transaction_id = 34 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>), i32, i32
      %11 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_127", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %11, (%2, %3, %4, %5, %6, %7, %8, %9)[%c32_i32], %c16_i32 {after = array<i64: 34, 33, 34>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 31 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 14 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 37 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 31 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 4 : i64, solver_objective = 8.600000e+01 : f64, starttime = 16 : i32, tl_channel = 0 : i32, transaction_group = 8 : i64, transaction_id = 35 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>), i32, i32
      %12 = tor.addi %1 %c192_i32 on (0 to 1) {dump = "unknown_128", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %12, (%2, %3, %4, %5, %6, %7, %8, %9)[%c48_i32], %c16_i32 {after = array<i64: 35, 34, 35>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 39 : i32, estimated_latency = 36 : i64, interface_cost = 68 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 37 : i64, memory_issue_cycle = 22 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 37 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [8 : i32], ref_endtime = 39 : i32, ref_starttime = 24 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 4 : i64, solver_objective = 8.600000e+01 : f64, starttime = 24 : i32, tl_channel = 1 : i32, transaction_group = 8 : i64, transaction_id = 36 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>, memref<8xi32>), i32, i32
      %13 = aps.load_by @cpuitfc, %0 {after = array<i64>, after_gap = array<i64>, endtime = 2 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 37 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, ref_endtime = 2 : i32, ref_starttime = 0 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 8.600000e+01 : f64, starttime = 0 : i32, transaction_id = 32 : i64, transfer_bytes = 4 : i64} : i32 -> i32
      %14 = aps.read_smem %2[%c0_i32] {aps.mem_id = 63 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<8xi32>, i32 -> i32
      %15 = arith.andi %14, %c255_i32 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : i32
      %16 = tor.addi %13 %15 on (47 to 48) {dump = "unknown_129", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %17 = aps.read_smem %3[%c0_i32] {aps.mem_id = 62 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<8xi32>, i32 -> i32
      %18 = arith.andi %17, %c255_i32 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : i32
      %19 = tor.addi %16 %18 on (47 to 48) {dump = "unknown_130", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %20 = aps.read_smem %4[%c0_i32] {aps.mem_id = 61 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<8xi32>, i32 -> i32
      %21 = arith.andi %20, %c255_i32 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : i32
      %22 = tor.addi %19 %21 on (47 to 48) {dump = "unknown_131", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %23 = aps.read_smem %5[%c0_i32] {aps.mem_id = 60 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<8xi32>, i32 -> i32
      %24 = arith.andi %23, %c255_i32 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : i32
      %25 = tor.addi %22 %24 on (47 to 48) {dump = "unknown_132", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %26 = aps.read_smem %6[%c0_i32] {aps.mem_id = 59 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<8xi32>, i32 -> i32
      %27 = arith.andi %26, %c255_i32 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : i32
      %28 = tor.addi %25 %27 on (47 to 48) {dump = "unknown_133", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %29 = aps.read_smem %7[%c0_i32] {aps.mem_id = 58 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<8xi32>, i32 -> i32
      %30 = arith.andi %29, %c255_i32 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : i32
      %31 = tor.addi %28 %30 on (47 to 48) {dump = "unknown_134", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %32 = aps.read_smem %8[%c0_i32] {aps.mem_id = 57 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<8xi32>, i32 -> i32
      %33 = arith.andi %32, %c255_i32 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : i32
      %34 = tor.addi %31 %33 on (47 to 48) {dump = "unknown_135", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %35 = aps.read_smem %9[%c0_i32] {aps.mem_id = 56 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<8xi32>, i32 -> i32
      %36 = arith.andi %35, %c255_i32 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : i32
      %37 = tor.addi %34 %36 on (47 to 48) {dump = "unknown_136", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %38 = aps.read_smem %2[%c1_i32] {aps.mem_id = 55 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<8xi32>, i32 -> i32
      %39 = arith.andi %38, %c255_i32 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : i32
      %40 = tor.addi %37 %39 on (47 to 48) {dump = "unknown_137", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %41 = aps.read_smem %3[%c1_i32] {aps.mem_id = 54 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<8xi32>, i32 -> i32
      %42 = arith.andi %41, %c255_i32 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : i32
      %43 = tor.addi %40 %42 on (47 to 48) {dump = "unknown_138", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %44 = aps.read_smem %4[%c1_i32] {aps.mem_id = 53 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<8xi32>, i32 -> i32
      %45 = arith.andi %44, %c255_i32 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : i32
      %46 = tor.addi %43 %45 on (47 to 48) {dump = "unknown_139", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %47 = aps.read_smem %5[%c1_i32] {aps.mem_id = 52 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<8xi32>, i32 -> i32
      %48 = arith.andi %47, %c255_i32 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : i32
      %49 = tor.addi %46 %48 on (47 to 48) {dump = "unknown_140", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %50 = aps.read_smem %6[%c1_i32] {aps.mem_id = 51 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<8xi32>, i32 -> i32
      %51 = arith.andi %50, %c255_i32 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : i32
      %52 = tor.addi %49 %51 on (47 to 48) {dump = "unknown_141", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %53 = aps.read_smem %7[%c1_i32] {aps.mem_id = 50 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<8xi32>, i32 -> i32
      %54 = arith.andi %53, %c255_i32 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : i32
      %55 = tor.addi %52 %54 on (47 to 48) {dump = "unknown_142", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %56 = aps.read_smem %8[%c1_i32] {aps.mem_id = 49 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<8xi32>, i32 -> i32
      %57 = arith.andi %56, %c255_i32 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : i32
      %58 = tor.addi %55 %57 on (47 to 48) {dump = "unknown_143", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %59 = aps.read_smem %9[%c1_i32] {aps.mem_id = 48 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<8xi32>, i32 -> i32
      %60 = arith.andi %59, %c255_i32 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : i32
      %61 = tor.addi %58 %60 on (47 to 48) {dump = "unknown_144", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %62 = aps.read_smem %2[%c2_i32] {aps.mem_id = 47 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : memref<8xi32>, i32 -> i32
      %63 = arith.andi %62, %c255_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
      %64 = tor.addi %61 %63 on (47 to 48) {dump = "unknown_145", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %65 = aps.read_smem %3[%c2_i32] {aps.mem_id = 46 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : memref<8xi32>, i32 -> i32
      %66 = arith.andi %65, %c255_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
      %67 = tor.addi %64 %66 on (47 to 48) {dump = "unknown_146", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %68 = aps.read_smem %4[%c2_i32] {aps.mem_id = 45 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : memref<8xi32>, i32 -> i32
      %69 = arith.andi %68, %c255_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
      %70 = tor.addi %67 %69 on (47 to 48) {dump = "unknown_147", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %71 = aps.read_smem %5[%c2_i32] {aps.mem_id = 44 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : memref<8xi32>, i32 -> i32
      %72 = arith.andi %71, %c255_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
      %73 = tor.addi %70 %72 on (47 to 48) {dump = "unknown_148", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %74 = aps.read_smem %6[%c2_i32] {aps.mem_id = 43 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : memref<8xi32>, i32 -> i32
      %75 = arith.andi %74, %c255_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
      %76 = tor.addi %73 %75 on (47 to 48) {dump = "unknown_149", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %77 = aps.read_smem %7[%c2_i32] {aps.mem_id = 42 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : memref<8xi32>, i32 -> i32
      %78 = arith.andi %77, %c255_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
      %79 = tor.addi %76 %78 on (47 to 48) {dump = "unknown_150", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %80 = aps.read_smem %8[%c2_i32] {aps.mem_id = 41 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : memref<8xi32>, i32 -> i32
      %81 = arith.andi %80, %c255_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
      %82 = tor.addi %79 %81 on (47 to 48) {dump = "unknown_151", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %83 = aps.read_smem %9[%c2_i32] {aps.mem_id = 40 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : memref<8xi32>, i32 -> i32
      %84 = arith.andi %83, %c255_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
      %85 = tor.addi %82 %84 on (47 to 48) {dump = "unknown_152", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %86 = aps.read_smem %2[%c3_i32] {aps.mem_id = 39 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<8xi32>, i32 -> i32
      %87 = arith.andi %86, %c255_i32 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : i32
      %88 = tor.addi %85 %87 on (47 to 48) {dump = "unknown_153", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %89 = aps.read_smem %3[%c3_i32] {aps.mem_id = 38 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<8xi32>, i32 -> i32
      %90 = arith.andi %89, %c255_i32 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : i32
      %91 = tor.addi %88 %90 on (47 to 48) {dump = "unknown_154", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %92 = aps.read_smem %4[%c3_i32] {aps.mem_id = 37 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<8xi32>, i32 -> i32
      %93 = arith.andi %92, %c255_i32 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : i32
      %94 = tor.addi %91 %93 on (47 to 48) {dump = "unknown_155", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %95 = aps.read_smem %5[%c3_i32] {aps.mem_id = 36 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<8xi32>, i32 -> i32
      %96 = arith.andi %95, %c255_i32 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : i32
      %97 = tor.addi %94 %96 on (47 to 48) {dump = "unknown_156", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %98 = aps.read_smem %6[%c3_i32] {aps.mem_id = 35 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<8xi32>, i32 -> i32
      %99 = arith.andi %98, %c255_i32 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : i32
      %100 = tor.addi %97 %99 on (47 to 48) {dump = "unknown_157", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %101 = aps.read_smem %7[%c3_i32] {aps.mem_id = 34 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<8xi32>, i32 -> i32
      %102 = arith.andi %101, %c255_i32 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : i32
      %103 = tor.addi %100 %102 on (47 to 48) {dump = "unknown_158", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %104 = aps.read_smem %8[%c3_i32] {aps.mem_id = 33 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<8xi32>, i32 -> i32
      %105 = arith.andi %104, %c255_i32 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : i32
      %106 = tor.addi %103 %105 on (47 to 48) {dump = "unknown_159", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %107 = aps.read_smem %9[%c3_i32] {aps.mem_id = 32 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<8xi32>, i32 -> i32
      %108 = arith.andi %107, %c255_i32 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : i32
      %109 = tor.addi %106 %108 on (47 to 48) {dump = "unknown_160", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %110 = aps.read_smem %2[%c4_i32] {aps.mem_id = 31 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<8xi32>, i32 -> i32
      %111 = arith.andi %110, %c255_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
      %112 = tor.addi %109 %111 on (47 to 48) {dump = "unknown_161", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %113 = aps.read_smem %3[%c4_i32] {aps.mem_id = 30 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<8xi32>, i32 -> i32
      %114 = arith.andi %113, %c255_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
      %115 = tor.addi %112 %114 on (47 to 48) {dump = "unknown_162", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %116 = aps.read_smem %4[%c4_i32] {aps.mem_id = 29 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<8xi32>, i32 -> i32
      %117 = arith.andi %116, %c255_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
      %118 = tor.addi %115 %117 on (47 to 48) {dump = "unknown_163", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %119 = aps.read_smem %5[%c4_i32] {aps.mem_id = 28 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<8xi32>, i32 -> i32
      %120 = arith.andi %119, %c255_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
      %121 = tor.addi %118 %120 on (47 to 48) {dump = "unknown_164", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %122 = aps.read_smem %6[%c4_i32] {aps.mem_id = 27 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<8xi32>, i32 -> i32
      %123 = arith.andi %122, %c255_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
      %124 = tor.addi %121 %123 on (47 to 48) {dump = "unknown_165", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %125 = aps.read_smem %7[%c4_i32] {aps.mem_id = 26 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<8xi32>, i32 -> i32
      %126 = arith.andi %125, %c255_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
      %127 = tor.addi %124 %126 on (47 to 48) {dump = "unknown_166", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %128 = aps.read_smem %8[%c4_i32] {aps.mem_id = 25 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<8xi32>, i32 -> i32
      %129 = arith.andi %128, %c255_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
      %130 = tor.addi %127 %129 on (47 to 48) {dump = "unknown_167", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %131 = aps.read_smem %9[%c4_i32] {aps.mem_id = 24 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<8xi32>, i32 -> i32
      %132 = arith.andi %131, %c255_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
      %133 = tor.addi %130 %132 on (47 to 48) {dump = "unknown_168", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %134 = aps.read_smem %2[%c5_i32] {aps.mem_id = 23 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : memref<8xi32>, i32 -> i32
      %135 = arith.andi %134, %c255_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
      %136 = tor.addi %133 %135 on (47 to 48) {dump = "unknown_169", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %137 = aps.read_smem %3[%c5_i32] {aps.mem_id = 22 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : memref<8xi32>, i32 -> i32
      %138 = arith.andi %137, %c255_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
      %139 = tor.addi %136 %138 on (47 to 48) {dump = "unknown_170", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %140 = aps.read_smem %4[%c5_i32] {aps.mem_id = 21 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : memref<8xi32>, i32 -> i32
      %141 = arith.andi %140, %c255_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
      %142 = tor.addi %139 %141 on (47 to 48) {dump = "unknown_171", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %143 = aps.read_smem %5[%c5_i32] {aps.mem_id = 20 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : memref<8xi32>, i32 -> i32
      %144 = arith.andi %143, %c255_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
      %145 = tor.addi %142 %144 on (47 to 48) {dump = "unknown_172", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %146 = aps.read_smem %6[%c5_i32] {aps.mem_id = 19 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : memref<8xi32>, i32 -> i32
      %147 = arith.andi %146, %c255_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
      %148 = tor.addi %145 %147 on (47 to 48) {dump = "unknown_173", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %149 = aps.read_smem %7[%c5_i32] {aps.mem_id = 18 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : memref<8xi32>, i32 -> i32
      %150 = arith.andi %149, %c255_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
      %151 = tor.addi %148 %150 on (47 to 48) {dump = "unknown_174", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %152 = aps.read_smem %8[%c5_i32] {aps.mem_id = 17 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : memref<8xi32>, i32 -> i32
      %153 = arith.andi %152, %c255_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
      %154 = tor.addi %151 %153 on (47 to 48) {dump = "unknown_175", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %155 = aps.read_smem %9[%c5_i32] {aps.mem_id = 16 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : memref<8xi32>, i32 -> i32
      %156 = arith.andi %155, %c255_i32 {endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, starttime = 42 : i32} : i32
      %157 = tor.addi %154 %156 on (47 to 48) {dump = "unknown_176", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %158 = aps.read_smem %2[%c6_i32] {aps.mem_id = 15 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : memref<8xi32>, i32 -> i32
      %159 = arith.andi %158, %c255_i32 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i32
      %160 = tor.addi %157 %159 on (47 to 48) {dump = "unknown_177", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %161 = aps.read_smem %3[%c6_i32] {aps.mem_id = 14 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : memref<8xi32>, i32 -> i32
      %162 = arith.andi %161, %c255_i32 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i32
      %163 = tor.addi %160 %162 on (47 to 48) {dump = "unknown_178", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %164 = aps.read_smem %4[%c6_i32] {aps.mem_id = 13 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : memref<8xi32>, i32 -> i32
      %165 = arith.andi %164, %c255_i32 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i32
      %166 = tor.addi %163 %165 on (47 to 48) {dump = "unknown_179", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %167 = aps.read_smem %5[%c6_i32] {aps.mem_id = 12 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : memref<8xi32>, i32 -> i32
      %168 = arith.andi %167, %c255_i32 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i32
      %169 = tor.addi %166 %168 on (47 to 48) {dump = "unknown_180", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %170 = aps.read_smem %6[%c6_i32] {aps.mem_id = 11 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : memref<8xi32>, i32 -> i32
      %171 = arith.andi %170, %c255_i32 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i32
      %172 = tor.addi %169 %171 on (47 to 48) {dump = "unknown_181", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %173 = aps.read_smem %7[%c6_i32] {aps.mem_id = 10 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : memref<8xi32>, i32 -> i32
      %174 = arith.andi %173, %c255_i32 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i32
      %175 = tor.addi %172 %174 on (47 to 48) {dump = "unknown_182", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %176 = aps.read_smem %8[%c6_i32] {aps.mem_id = 9 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : memref<8xi32>, i32 -> i32
      %177 = arith.andi %176, %c255_i32 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i32
      %178 = tor.addi %175 %177 on (47 to 48) {dump = "unknown_183", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %179 = aps.read_smem %9[%c6_i32] {aps.mem_id = 8 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : memref<8xi32>, i32 -> i32
      %180 = arith.andi %179, %c255_i32 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i32
      %181 = tor.addi %178 %180 on (47 to 48) {dump = "unknown_184", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %182 = aps.read_smem %2[%c7_i32] {aps.mem_id = 7 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : memref<8xi32>, i32 -> i32
      %183 = arith.andi %182, %c255_i32 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i32
      %184 = tor.addi %181 %183 on (47 to 48) {dump = "unknown_185", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %185 = aps.read_smem %3[%c7_i32] {aps.mem_id = 6 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : memref<8xi32>, i32 -> i32
      %186 = arith.andi %185, %c255_i32 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i32
      %187 = tor.addi %184 %186 on (47 to 48) {dump = "unknown_186", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %188 = aps.read_smem %4[%c7_i32] {aps.mem_id = 5 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : memref<8xi32>, i32 -> i32
      %189 = arith.andi %188, %c255_i32 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i32
      %190 = tor.addi %187 %189 on (47 to 48) {dump = "unknown_187", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %191 = aps.read_smem %5[%c7_i32] {aps.mem_id = 4 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : memref<8xi32>, i32 -> i32
      %192 = arith.andi %191, %c255_i32 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i32
      %193 = tor.addi %190 %192 on (47 to 48) {dump = "unknown_188", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %194 = aps.read_smem %6[%c7_i32] {aps.mem_id = 3 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : memref<8xi32>, i32 -> i32
      %195 = arith.andi %194, %c255_i32 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i32
      %196 = tor.addi %193 %195 on (47 to 48) {dump = "unknown_189", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %197 = aps.read_smem %7[%c7_i32] {aps.mem_id = 2 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : memref<8xi32>, i32 -> i32
      %198 = arith.andi %197, %c255_i32 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i32
      %199 = tor.addi %196 %198 on (47 to 48) {dump = "unknown_190", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %200 = aps.read_smem %8[%c7_i32] {aps.mem_id = 1 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : memref<8xi32>, i32 -> i32
      %201 = arith.andi %200, %c255_i32 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i32
      %202 = tor.addi %199 %201 on (47 to 48) {dump = "unknown_191", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      %203 = aps.read_smem %9[%c7_i32] {aps.mem_id = 0 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : memref<8xi32>, i32 -> i32
      %204 = arith.andi %203, %c255_i32 {endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, starttime = 40 : i32} : i32
      %205 = tor.addi %202 %204 on (47 to 48) {dump = "unknown_192", ref_endtime = 48 : i32, ref_starttime = 47 : i32} : (i32, i32) -> i32
      aps.globalstore %205, @au_generated_candidate_35__l0u_l1u_l2u_buffer_5 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : i32
      aps.store_by @cpuitfc, %205, %0 {after = array<i64>, after_gap = array<i64>, endtime = 49 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 49 : i32, ref_starttime = 47 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 1.800000e+01 : f64, starttime = 47 : i32, transaction_id = 37 : i64, transfer_bytes = 4 : i64} : i32, i32
      tor.return
    }
  } {HoistConstCondIfOp = 1 : i32, schedule = true}
}

