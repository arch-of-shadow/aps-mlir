module {
  aps.memitfc @busitfc {cache_hint = "cold", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 8 : i64, min_beats = 1 : i64, min_write_beats = 2 : i64, read_latency = 5 : i64, width = 8 : i64, write_completion = 2 : i64}
  aps.memitfc @cpuitfc {cache_hint = "warm", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 1 : i64, min_beats = 1 : i64, min_write_beats = 1 : i64, read_latency = 2 : i64, width = 4 : i64, write_completion = 2 : i64}
  aps.memorymap {
    aps.mem_entry "au_generated_candidate_22__direct_buffer" : banks([@au_generated_candidate_22__direct_buffer_0_0, @au_generated_candidate_22__direct_buffer_1]), base(0), size(288), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_22__direct_buffer_0" : banks([@au_generated_candidate_22__direct_buffer_0]), base(512), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_38__l0u_buffer" : banks([@au_generated_candidate_38__l0u_buffer_0, @au_generated_candidate_38__l0u_buffer_1]), base(516), size(288), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_44__l0u_buffer" : banks([@au_generated_candidate_44__l0u_buffer_0, @au_generated_candidate_44__l0u_buffer_1]), base(1024), size(96), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_48__l0p_l1u_buffer" : banks([@au_generated_candidate_48__l0p_l1u_buffer_0, @au_generated_candidate_48__l0p_l1u_buffer_1_2]), base(1152), size(288), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_48__l0p_l1u_buffer_1" : banks([@au_generated_candidate_48__l0p_l1u_buffer_1_0, @au_generated_candidate_48__l0p_l1u_buffer_1_1]), base(1536), size(288), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_48__l0p_l1u_buffer_2" : banks([@au_generated_candidate_48__l0p_l1u_buffer_2_0, @au_generated_candidate_48__l0p_l1u_buffer_2_1]), base(2048), size(288), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_49__l1p_buffer" : banks([@au_generated_candidate_49__l1p_buffer_0, @au_generated_candidate_49__l1p_buffer_1]), base(2560), size(96), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_49__l1p_buffer_3" : banks([@au_generated_candidate_49__l1p_buffer_3_0, @au_generated_candidate_49__l1p_buffer_3_1]), base(2688), size(96), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_49__l1p_buffer_4" : banks([@au_generated_candidate_49__l1p_buffer_4_0, @au_generated_candidate_49__l1p_buffer_4_1]), base(2816), size(96), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_9__l0u_buffer" : banks([@au_generated_candidate_9__l0u_buffer_0, @au_generated_candidate_9__l0u_buffer_1]), base(2944), size(96), count(2), cyclic(1)
    aps.mem_finish
  }
  tor.design @aps_isaxes {
    %c-2_i2 = arith.constant -2 : i2
    %c1_i2 = arith.constant 1 : i2
    %c0_i2 = arith.constant 0 : i2
    %c7_i4 = arith.constant 7 : i4
    %c1_i4 = arith.constant 1 : i4
    %c0_i4 = arith.constant 0 : i4
    %c-9_i5 = arith.constant -9 : i5
    %c1_i5 = arith.constant 1 : i5
    %c0_i5 = arith.constant 0 : i5
    %cst = arith.constant -1.200000e-01 : f32
    %cst_0 = arith.constant -4.000000e-02 : f32
    %cst_1 = arith.constant 8.000000e-02 : f32
    %cst_2 = arith.constant 4.000000e-02 : f32
    %cst_3 = arith.constant 2.400000e-01 : f32
    %cst_4 = arith.constant 1.800000e-01 : f32
    %cst_5 = arith.constant 2.000000e-01 : f32
    %cst_6 = arith.constant -1.800000e+00 : f32
    %cst_7 = arith.constant -2.500000e-01 : f32
    %cst_8 = arith.constant 0.000000e+00 : f32
    %cst_9 = arith.constant -1.250000e+00 : f32
    %cst_10 = arith.constant 1.200000e-01 : f32
    %cst_11 = arith.constant -0.699999928 : f32
    %cst_12 = arith.constant 2.500000e-01 : f32
    %cst_13 = arith.constant -0.149999857 : f32
    %cst_14 = arith.constant 0.400000095 : f32
    %cst_15 = arith.constant 0.950000047 : f32
    %cst_16 = arith.constant 1.50000024 : f32
    %cst_17 = arith.constant 2.05000019 : f32
    %cst_18 = arith.constant -1.200000e+00 : f32
    %cst_19 = arith.constant -0.899999976 : f32
    %cst_20 = arith.constant 2.800000e+00 : f32
    %cst_21 = arith.constant -0.750000059 : f32
    %cst_22 = arith.constant 2.980000e+00 : f32
    %cst_23 = arith.constant -0.300000072 : f32
    %cst_24 = arith.constant 3.15999985 : f32
    %cst_25 = arith.constant 0.149999857 : f32
    %cst_26 = arith.constant 3.340000e+00 : f32
    %cst_27 = arith.constant 0.599999904 : f32
    %cst_28 = arith.constant 3.520000e+00 : f32
    %cst_29 = arith.constant 1.050000e+00 : f32
    %cst_30 = arith.constant -0.299999952 : f32
    %cst_31 = arith.constant 0.300000072 : f32
    %cst_32 = arith.constant 0.900000095 : f32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c5_i32 = arith.constant 5 : i32
    %c6_i32 = arith.constant 6 : i32
    %c7_i32 = arith.constant 7 : i32
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
    %c0_i32 = arith.constant 0 : i32
    %cst_33 = arith.constant 1.270000e+02 : f32
    %cst_34 = arith.constant 1.090000e+02 : f32
    %cst_35 = arith.constant 8.300000e+01 : f32
    %c1_i32 = arith.constant {dump = "unknown_0"} 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c192_i32 = arith.constant 192 : i32
    %c256_i32 = arith.constant 256 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c-2_i32 = arith.constant -2 : i32
    memref.global @au_generated_candidate_9__l0u_buffer_0 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_9__l0u_buffer_1 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_49__l1p_buffer_4_0 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_49__l1p_buffer_4_1 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_49__l1p_buffer_3_0 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_49__l1p_buffer_3_1 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_49__l1p_buffer_0 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_49__l1p_buffer_1 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_48__l0p_l1u_buffer_2_0 : memref<36xf32> = uninitialized
    memref.global @au_generated_candidate_48__l0p_l1u_buffer_2_1 : memref<36xf32> = uninitialized
    memref.global @au_generated_candidate_48__l0p_l1u_buffer_1_0 : memref<36xf32> = uninitialized
    memref.global @au_generated_candidate_48__l0p_l1u_buffer_1_1 : memref<36xf32> = uninitialized
    memref.global @au_generated_candidate_48__l0p_l1u_buffer_0 : memref<36xf32> = uninitialized
    memref.global @au_generated_candidate_48__l0p_l1u_buffer_1_2 : memref<36xf32> = uninitialized
    memref.global @au_generated_candidate_44__l0u_buffer_0 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_44__l0u_buffer_1 : memref<12xf32> = uninitialized
    memref.global @au_generated_candidate_38__l0u_buffer_0 : memref<36xf32> = uninitialized
    memref.global @au_generated_candidate_38__l0u_buffer_1 : memref<36xf32> = uninitialized
    memref.global @au_generated_candidate_22__direct_buffer_0 : memref<1xi32> {scaleisa.scratchpad, var_name = "au_generated_candidate_22__direct_buffer_0"}
    memref.global @au_generated_candidate_22__direct_buffer_0_0 : memref<36xf32> = uninitialized
    memref.global @au_generated_candidate_22__direct_buffer_1 : memref<36xf32> = uninitialized
    memref.global "private" @au_generated_candidate_48__l0p_l1u_csr0 : memref<i32> = uninitialized {csr_address = 2049 : i32}
    memref.global "private" @au_generated_candidate_49__l1p_csr0 : memref<i32> = uninitialized {csr_address = 2050 : i32}
    tor.func @au_generated_candidate_22__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 0 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/sfm_batch/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_1", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_2", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_22__direct_buffer_0_0 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_22__direct_buffer_1 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64: 0>, after_gap = array<i64: 1>, endtime = 13 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 13 : i64, memory_issue_cycle = 1 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 41 : i64, memory_search_hits = 1 : i64, memory_search_states = 22 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 5 : i64, solver_objective = 9.800000e+01 : f64, starttime = 1 : i32, tl_channel = 0 : i32, transaction_group = 0 : i64, transaction_id = 1 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_3", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 1, 1>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 21 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 21 : i64, memory_issue_cycle = 2 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 41 : i64, memory_search_hits = 1 : i64, memory_search_states = 22 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 21 : i32, ref_starttime = 2 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 5 : i64, solver_objective = 9.800000e+01 : f64, starttime = 2 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 2 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %5 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_4", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 2, 1, 2>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 31 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 14 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 41 : i64, memory_search_hits = 1 : i64, memory_search_states = 22 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 31 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 5 : i64, solver_objective = 9.800000e+01 : f64, starttime = 16 : i32, tl_channel = 0 : i32, transaction_group = 0 : i64, transaction_id = 3 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %6 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_5", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%2, %3)[%c48_i32], %c16_i32 {after = array<i64: 3, 2, 3>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 39 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 37 : i64, memory_issue_cycle = 22 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 41 : i64, memory_search_hits = 1 : i64, memory_search_states = 22 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 39 : i32, ref_starttime = 24 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 5 : i64, solver_objective = 9.800000e+01 : f64, starttime = 24 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 4 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %7 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_6", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%2, %3)[%c64_i32], %c8_i32 {after = array<i64: 4, 3, 4>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 43 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 41 : i64, memory_issue_cycle = 30 : i64, memory_latency = 11 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 41 : i64, memory_search_hits = 1 : i64, memory_search_states = 22 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 43 : i32, ref_starttime = 32 : i32, required_alignment = 32 : i64, segment = 4 : i64, segment_count = 5 : i64, solver_objective = 9.800000e+01 : f64, starttime = 32 : i32, tl_channel = 0 : i32, transaction_group = 0 : i64, transaction_id = 5 : i64, transfer_bytes = 32 : i64, transfer_offset = 256 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %8 = aps.load_by @cpuitfc, %1 {after = array<i64>, after_gap = array<i64>, endtime = 2 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 41 : i64, memory_search_hits = 1 : i64, memory_search_states = 22 : i64, ref_endtime = 2 : i32, ref_starttime = 0 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 9.800000e+01 : f64, starttime = 0 : i32, transaction_id = 0 : i64, transfer_bytes = 4 : i64} : i32 -> i32
      aps.globalstore %8, @au_generated_candidate_22__direct_buffer_0 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
      tor.for %arg3 = (%c0_i5 : i5) to (%c-9_i5 : i5) step (%c1_i5 : i5)
      on (45 to 53){
        %10 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %11 = tor.muli %10 %c3_i32 on (45 to 46) {dump = "unknown_7", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %12 = tor.cmpi slt %11 %c0_i32 on (45 to 46) {dump = "unknown_8", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %13 = tor.subi %c-1_i32 %11 on (45 to 46) {dump = "unknown_9", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %14 = arith.select %12, %13, %11 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %15 = tor.cmpi slt %14 %c0_i32 on (45 to 46) {dump = "unknown_10", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %16 = arith.extui %15 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
        %17 = tor.addi %14 %16 on (45 to 46) {dump = "unknown_11", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %18 = arith.shrsi %17, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %19 = tor.subi %c-1_i32 %18 on (45 to 46) {dump = "unknown_12", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %20 = arith.select %12, %19, %18 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %21 = aps.read_smem %2[%20] {aps.mem_id = 5 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<36xf32>, i32 -> f32
        %22 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %23 = tor.muli %22 %c3_i32 on (45 to 46) {dump = "unknown_13", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %24 = tor.cmpi slt %23 %c0_i32 on (45 to 46) {dump = "unknown_14", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %25 = tor.subi %c-1_i32 %23 on (45 to 46) {dump = "unknown_15", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %26 = arith.select %24, %25, %23 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %27 = tor.cmpi slt %26 %c0_i32 on (45 to 46) {dump = "unknown_16", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %28 = arith.extui %27 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
        %29 = tor.addi %26 %28 on (45 to 46) {dump = "unknown_17", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %30 = arith.shrsi %29, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %31 = tor.subi %c-1_i32 %30 on (45 to 46) {dump = "unknown_18", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %32 = arith.select %24, %31, %30 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %33 = aps.read_smem %3[%32] {aps.mem_id = 4 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<36xf32>, i32 -> f32
        %34 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %35 = arith.andi %34, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %36 = tor.cmpi eq %35 %c0_i32 on (45 to 46) {dump = "unknown_19", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %37 = arith.select %36, %21, %33 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : f32
        %38 = tor.mulf %37 %cst_35 on (46 to 49) {dump = "unknown_20", fastmath = #arith.fastmath<none>, ref_endtime = 49 : i32, ref_starttime = 46 : i32} : f32
        %39 = arith.fptosi %38 {endtime = 50 : i32, ref_endtime = 50 : i32, ref_starttime = 49 : i32, starttime = 49 : i32} : f32 to i32
        %40 = aps.globalload @au_generated_candidate_22__direct_buffer_0 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %41 = tor.addi %40 %39 on (50 to 51) {dump = "unknown_21", ref_endtime = 51 : i32, ref_starttime = 50 : i32} : (i32, i32) -> i32
        %42 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %43 = tor.muli %42 %c3_i32 on (45 to 46) {dump = "unknown_22", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %44 = tor.addi %43 %c1_i32 on (45 to 46) {dump = "unknown_23", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %45 = tor.cmpi slt %44 %c0_i32 on (45 to 46) {dump = "unknown_24", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %46 = tor.subi %c-2_i32 %43 on (45 to 46) {dump = "unknown_25", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %47 = arith.select %45, %46, %44 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %48 = tor.cmpi slt %47 %c0_i32 on (45 to 46) {dump = "unknown_26", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %49 = arith.extui %48 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
        %50 = tor.addi %47 %49 on (45 to 46) {dump = "unknown_27", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %51 = arith.shrsi %50, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %52 = tor.subi %c-1_i32 %51 on (45 to 46) {dump = "unknown_28", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %53 = arith.select %45, %52, %51 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %54 = aps.read_smem %2[%53] {aps.mem_id = 3 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<36xf32>, i32 -> f32
        %55 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %56 = tor.muli %55 %c3_i32 on (45 to 46) {dump = "unknown_29", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %57 = tor.addi %56 %c1_i32 on (45 to 46) {dump = "unknown_30", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %58 = tor.cmpi slt %57 %c0_i32 on (45 to 46) {dump = "unknown_31", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %59 = tor.subi %c-2_i32 %56 on (45 to 46) {dump = "unknown_32", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %60 = arith.select %58, %59, %57 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %61 = tor.cmpi slt %60 %c0_i32 on (45 to 46) {dump = "unknown_33", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %62 = arith.extui %61 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
        %63 = tor.addi %60 %62 on (45 to 46) {dump = "unknown_34", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %64 = arith.shrsi %63, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %65 = tor.subi %c-1_i32 %64 on (45 to 46) {dump = "unknown_35", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %66 = arith.select %58, %65, %64 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %67 = aps.read_smem %3[%66] {aps.mem_id = 2 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<36xf32>, i32 -> f32
        %68 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %69 = tor.addi %68 %c1_i32 on (45 to 46) {dump = "unknown_36", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %70 = tor.cmpi slt %69 %c0_i32 on (45 to 46) {dump = "unknown_37", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %71 = tor.subi %c-2_i32 %68 on (45 to 46) {dump = "unknown_38", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %72 = arith.select %70, %71, %69 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %73 = tor.cmpi slt %72 %c0_i32 on (45 to 46) {dump = "unknown_39", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %74 = arith.extui %73 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
        %75 = tor.addi %72 %74 on (45 to 46) {dump = "unknown_40", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %76 = arith.shrsi %75, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %77 = tor.subi %c-1_i32 %76 on (45 to 46) {dump = "unknown_41", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %78 = arith.select %70, %77, %76 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %79 = tor.muli %78 %c-2_i32 on (45 to 46) {dump = "unknown_42", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %80 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %81 = tor.addi %80 %79 on (45 to 46) {dump = "unknown_43", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %82 = tor.addi %81 %c1_i32 on (45 to 46) {dump = "unknown_44", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %83 = tor.cmpi eq %82 %c0_i32 on (45 to 46) {dump = "unknown_45", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %84 = arith.select %83, %54, %67 {endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, starttime = 47 : i32} : f32
        %85 = tor.mulf %84 %cst_34 on (47 to 50) {dump = "unknown_46", fastmath = #arith.fastmath<none>, ref_endtime = 50 : i32, ref_starttime = 47 : i32} : f32
        %86 = arith.fptosi %85 {endtime = 51 : i32, ref_endtime = 51 : i32, ref_starttime = 50 : i32, starttime = 50 : i32} : f32 to i32
        %87 = tor.addi %41 %86 on (51 to 52) {dump = "unknown_47", ref_endtime = 52 : i32, ref_starttime = 51 : i32} : (i32, i32) -> i32
        %88 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %89 = tor.muli %88 %c3_i32 on (45 to 46) {dump = "unknown_48", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %90 = tor.cmpi slt %89 %c0_i32 on (45 to 46) {dump = "unknown_49", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %91 = tor.subi %c-1_i32 %89 on (45 to 46) {dump = "unknown_50", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %92 = arith.select %90, %91, %89 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %93 = tor.cmpi slt %92 %c0_i32 on (45 to 46) {dump = "unknown_51", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %94 = arith.extui %93 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
        %95 = tor.addi %92 %94 on (45 to 46) {dump = "unknown_52", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %96 = arith.shrsi %95, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %97 = tor.subi %c-1_i32 %96 on (45 to 46) {dump = "unknown_53", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %98 = arith.select %90, %97, %96 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %99 = tor.addi %98 %c1_i32 on (45 to 46) {dump = "unknown_54", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %100 = aps.read_smem %2[%99] {aps.mem_id = 1 : i64, endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, slot = 0 : i32, starttime = 47 : i32} : memref<36xf32>, i32 -> f32
        %101 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %102 = tor.muli %101 %c3_i32 on (45 to 46) {dump = "unknown_55", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %103 = tor.cmpi slt %102 %c0_i32 on (45 to 46) {dump = "unknown_56", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %104 = tor.subi %c-1_i32 %102 on (45 to 46) {dump = "unknown_57", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %105 = arith.select %103, %104, %102 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %106 = tor.cmpi slt %105 %c0_i32 on (45 to 46) {dump = "unknown_58", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %107 = arith.extui %106 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
        %108 = tor.addi %105 %107 on (45 to 46) {dump = "unknown_59", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %109 = arith.shrsi %108, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %110 = tor.subi %c-1_i32 %109 on (45 to 46) {dump = "unknown_60", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %111 = arith.select %103, %110, %109 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %112 = tor.addi %111 %c1_i32 on (45 to 46) {dump = "unknown_61", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
        %113 = aps.read_smem %3[%112] {aps.mem_id = 0 : i64, endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, slot = 0 : i32, starttime = 47 : i32} : memref<36xf32>, i32 -> f32
        %114 = arith.extui %arg3 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i5 to i32
        %115 = arith.andi %114, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
        %116 = tor.cmpi eq %115 %c0_i32 on (45 to 46) {dump = "unknown_62", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
        %117 = arith.select %116, %100, %113 {endtime = 49 : i32, ref_endtime = 49 : i32, ref_starttime = 48 : i32, starttime = 48 : i32} : f32
        %118 = tor.mulf %117 %cst_33 on (48 to 51) {dump = "unknown_63", fastmath = #arith.fastmath<none>, ref_endtime = 51 : i32, ref_starttime = 48 : i32} : f32
        %119 = arith.fptosi %118 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : f32 to i32
        %120 = tor.addi %87 %119 on (52 to 53) {dump = "unknown_64", ref_endtime = 53 : i32, ref_starttime = 52 : i32} : (i32, i32) -> i32
        aps.globalstore %120, @au_generated_candidate_22__direct_buffer_0 {endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, starttime = 52 : i32} : i32
      } {ref_endtime = 53 : i32, ref_starttime = 45 : i32}
      %9 = aps.globalload @au_generated_candidate_22__direct_buffer_0 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
      aps.store_by @cpuitfc, %9, %1 {after = array<i64>, after_gap = array<i64>, endtime = 47 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 47 : i32, ref_starttime = 45 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 1.800000e+01 : f64, starttime = 45 : i32, transaction_id = 6 : i64, transfer_bytes = 4 : i64} : i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_38__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 1 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/sfm_batch/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_65", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = memref.get_global @au_generated_candidate_38__l0u_buffer_0 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %2 = memref.get_global @au_generated_candidate_38__l0u_buffer_1 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%1, %2)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 5 : i64, solver_objective = 8.000000e+01 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 7 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %3 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_66", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %3, (%1, %2)[%c16_i32], %c16_i32 {after = array<i64: 7, 7>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 20 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 20 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 5 : i64, solver_objective = 8.000000e+01 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 8 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %4 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_67", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%1, %2)[%c32_i32], %c16_i32 {after = array<i64: 8, 7, 8>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 30 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 30 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 5 : i64, solver_objective = 8.000000e+01 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 9 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %5 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_68", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%1, %2)[%c48_i32], %c16_i32 {after = array<i64: 9, 8, 9>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 38 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 38 : i32, ref_starttime = 23 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 5 : i64, solver_objective = 8.000000e+01 : f64, starttime = 23 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 10 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %6 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_69", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%1, %2)[%c64_i32], %c8_i32 {after = array<i64: 10, 9, 10>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 42 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 40 : i64, memory_issue_cycle = 29 : i64, memory_latency = 11 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 42 : i32, ref_starttime = 31 : i32, required_alignment = 32 : i64, segment = 4 : i64, segment_count = 5 : i64, solver_objective = 8.000000e+01 : f64, starttime = 31 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 11 : i64, transfer_bytes = 32 : i64, transfer_offset = 256 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      aps.write_smem %cst_18, %1[%c0_i32] {aps.mem_id = 71 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_19, %2[%c0_i32] {aps.mem_id = 70 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_20, %1[%c1_i32] {aps.mem_id = 69 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_21, %2[%c1_i32] {aps.mem_id = 68 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_19, %1[%c2_i32] {aps.mem_id = 67 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_22, %2[%c2_i32] {aps.mem_id = 66 : i64, endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, slot = 0 : i32, starttime = 44 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_23, %1[%c3_i32] {aps.mem_id = 65 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_19, %2[%c3_i32] {aps.mem_id = 64 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_24, %1[%c4_i32] {aps.mem_id = 63 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_25, %2[%c4_i32] {aps.mem_id = 62 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_19, %1[%c5_i32] {aps.mem_id = 61 : i64, endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, slot = 0 : i32, starttime = 47 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_26, %2[%c5_i32] {aps.mem_id = 60 : i64, endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, slot = 0 : i32, starttime = 47 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_27, %1[%c6_i32] {aps.mem_id = 59 : i64, endtime = 49 : i32, ref_endtime = 49 : i32, ref_starttime = 48 : i32, slot = 0 : i32, starttime = 48 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_19, %2[%c6_i32] {aps.mem_id = 58 : i64, endtime = 49 : i32, ref_endtime = 49 : i32, ref_starttime = 48 : i32, slot = 0 : i32, starttime = 48 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_28, %1[%c7_i32] {aps.mem_id = 57 : i64, endtime = 50 : i32, ref_endtime = 50 : i32, ref_starttime = 49 : i32, slot = 0 : i32, starttime = 49 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_29, %2[%c7_i32] {aps.mem_id = 56 : i64, endtime = 50 : i32, ref_endtime = 50 : i32, ref_starttime = 49 : i32, slot = 0 : i32, starttime = 49 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_19, %1[%c8_i32] {aps.mem_id = 55 : i64, endtime = 51 : i32, ref_endtime = 51 : i32, ref_starttime = 50 : i32, slot = 0 : i32, starttime = 50 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_20, %2[%c8_i32] {aps.mem_id = 54 : i64, endtime = 51 : i32, ref_endtime = 51 : i32, ref_starttime = 50 : i32, slot = 0 : i32, starttime = 50 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_18, %1[%c9_i32] {aps.mem_id = 53 : i64, endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, slot = 0 : i32, starttime = 51 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_30, %2[%c9_i32] {aps.mem_id = 52 : i64, endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, slot = 0 : i32, starttime = 51 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_22, %1[%c10_i32] {aps.mem_id = 51 : i64, endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, slot = 0 : i32, starttime = 52 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_21, %2[%c10_i32] {aps.mem_id = 50 : i64, endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, slot = 0 : i32, starttime = 52 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_30, %1[%c11_i32] {aps.mem_id = 49 : i64, endtime = 54 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, slot = 0 : i32, starttime = 53 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_24, %2[%c11_i32] {aps.mem_id = 48 : i64, endtime = 54 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, slot = 0 : i32, starttime = 53 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_23, %1[%c12_i32] {aps.mem_id = 47 : i64, endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, slot = 0 : i32, starttime = 54 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_30, %2[%c12_i32] {aps.mem_id = 46 : i64, endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, slot = 0 : i32, starttime = 54 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_26, %1[%c13_i32] {aps.mem_id = 45 : i64, endtime = 56 : i32, ref_endtime = 56 : i32, ref_starttime = 55 : i32, slot = 0 : i32, starttime = 55 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_25, %2[%c13_i32] {aps.mem_id = 44 : i64, endtime = 56 : i32, ref_endtime = 56 : i32, ref_starttime = 55 : i32, slot = 0 : i32, starttime = 55 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_30, %1[%c14_i32] {aps.mem_id = 43 : i64, endtime = 57 : i32, ref_endtime = 57 : i32, ref_starttime = 56 : i32, slot = 0 : i32, starttime = 56 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_28, %2[%c14_i32] {aps.mem_id = 42 : i64, endtime = 57 : i32, ref_endtime = 57 : i32, ref_starttime = 56 : i32, slot = 0 : i32, starttime = 56 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_27, %1[%c15_i32] {aps.mem_id = 41 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_30, %2[%c15_i32] {aps.mem_id = 40 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_20, %1[%c16_i32] {aps.mem_id = 39 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_29, %2[%c16_i32] {aps.mem_id = 38 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_30, %1[%c17_i32] {aps.mem_id = 37 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_22, %2[%c17_i32] {aps.mem_id = 36 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_18, %1[%c18_i32] {aps.mem_id = 35 : i64, endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, slot = 0 : i32, starttime = 60 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_31, %2[%c18_i32] {aps.mem_id = 34 : i64, endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, slot = 0 : i32, starttime = 60 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_24, %1[%c19_i32] {aps.mem_id = 33 : i64, endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, slot = 0 : i32, starttime = 61 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_21, %2[%c19_i32] {aps.mem_id = 32 : i64, endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, slot = 0 : i32, starttime = 61 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_31, %1[%c20_i32] {aps.mem_id = 31 : i64, endtime = 63 : i32, ref_endtime = 63 : i32, ref_starttime = 62 : i32, slot = 0 : i32, starttime = 62 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_26, %2[%c20_i32] {aps.mem_id = 30 : i64, endtime = 63 : i32, ref_endtime = 63 : i32, ref_starttime = 62 : i32, slot = 0 : i32, starttime = 62 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_23, %1[%c21_i32] {aps.mem_id = 29 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_31, %2[%c21_i32] {aps.mem_id = 28 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_28, %1[%c22_i32] {aps.mem_id = 27 : i64, endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, slot = 0 : i32, starttime = 64 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_25, %2[%c22_i32] {aps.mem_id = 26 : i64, endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, slot = 0 : i32, starttime = 64 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_31, %1[%c23_i32] {aps.mem_id = 25 : i64, endtime = 66 : i32, ref_endtime = 66 : i32, ref_starttime = 65 : i32, slot = 0 : i32, starttime = 65 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_20, %2[%c23_i32] {aps.mem_id = 24 : i64, endtime = 66 : i32, ref_endtime = 66 : i32, ref_starttime = 65 : i32, slot = 0 : i32, starttime = 65 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_27, %1[%c24_i32] {aps.mem_id = 23 : i64, endtime = 67 : i32, ref_endtime = 67 : i32, ref_starttime = 66 : i32, slot = 0 : i32, starttime = 66 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_31, %2[%c24_i32] {aps.mem_id = 22 : i64, endtime = 67 : i32, ref_endtime = 67 : i32, ref_starttime = 66 : i32, slot = 0 : i32, starttime = 66 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_22, %1[%c25_i32] {aps.mem_id = 21 : i64, endtime = 68 : i32, ref_endtime = 68 : i32, ref_starttime = 67 : i32, slot = 0 : i32, starttime = 67 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_29, %2[%c25_i32] {aps.mem_id = 20 : i64, endtime = 68 : i32, ref_endtime = 68 : i32, ref_starttime = 67 : i32, slot = 0 : i32, starttime = 67 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_31, %1[%c26_i32] {aps.mem_id = 19 : i64, endtime = 69 : i32, ref_endtime = 69 : i32, ref_starttime = 68 : i32, slot = 0 : i32, starttime = 68 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_24, %2[%c26_i32] {aps.mem_id = 18 : i64, endtime = 69 : i32, ref_endtime = 69 : i32, ref_starttime = 68 : i32, slot = 0 : i32, starttime = 68 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_18, %1[%c27_i32] {aps.mem_id = 17 : i64, endtime = 70 : i32, ref_endtime = 70 : i32, ref_starttime = 69 : i32, slot = 0 : i32, starttime = 69 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_32, %2[%c27_i32] {aps.mem_id = 16 : i64, endtime = 70 : i32, ref_endtime = 70 : i32, ref_starttime = 69 : i32, slot = 0 : i32, starttime = 69 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_26, %1[%c28_i32] {aps.mem_id = 15 : i64, endtime = 71 : i32, ref_endtime = 71 : i32, ref_starttime = 70 : i32, slot = 0 : i32, starttime = 70 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_21, %2[%c28_i32] {aps.mem_id = 14 : i64, endtime = 71 : i32, ref_endtime = 71 : i32, ref_starttime = 70 : i32, slot = 0 : i32, starttime = 70 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_32, %1[%c29_i32] {aps.mem_id = 13 : i64, endtime = 72 : i32, ref_endtime = 72 : i32, ref_starttime = 71 : i32, slot = 0 : i32, starttime = 71 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_28, %2[%c29_i32] {aps.mem_id = 12 : i64, endtime = 72 : i32, ref_endtime = 72 : i32, ref_starttime = 71 : i32, slot = 0 : i32, starttime = 71 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_23, %1[%c30_i32] {aps.mem_id = 11 : i64, endtime = 73 : i32, ref_endtime = 73 : i32, ref_starttime = 72 : i32, slot = 0 : i32, starttime = 72 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_32, %2[%c30_i32] {aps.mem_id = 10 : i64, endtime = 73 : i32, ref_endtime = 73 : i32, ref_starttime = 72 : i32, slot = 0 : i32, starttime = 72 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_20, %1[%c31_i32] {aps.mem_id = 9 : i64, endtime = 74 : i32, ref_endtime = 74 : i32, ref_starttime = 73 : i32, slot = 0 : i32, starttime = 73 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_25, %2[%c31_i32] {aps.mem_id = 8 : i64, endtime = 74 : i32, ref_endtime = 74 : i32, ref_starttime = 73 : i32, slot = 0 : i32, starttime = 73 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_32, %1[%c32_i32] {aps.mem_id = 7 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_22, %2[%c32_i32] {aps.mem_id = 6 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_27, %1[%c33_i32] {aps.mem_id = 5 : i64, endtime = 76 : i32, ref_endtime = 76 : i32, ref_starttime = 75 : i32, slot = 0 : i32, starttime = 75 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_32, %2[%c33_i32] {aps.mem_id = 4 : i64, endtime = 76 : i32, ref_endtime = 76 : i32, ref_starttime = 75 : i32, slot = 0 : i32, starttime = 75 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_24, %1[%c34_i32] {aps.mem_id = 3 : i64, endtime = 77 : i32, ref_endtime = 77 : i32, ref_starttime = 76 : i32, slot = 0 : i32, starttime = 76 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_29, %2[%c34_i32] {aps.mem_id = 2 : i64, endtime = 77 : i32, ref_endtime = 77 : i32, ref_starttime = 76 : i32, slot = 0 : i32, starttime = 76 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_32, %1[%c35_i32] {aps.mem_id = 1 : i64, endtime = 78 : i32, ref_endtime = 78 : i32, ref_starttime = 77 : i32, slot = 0 : i32, starttime = 77 : i32} : f32, memref<36xf32>, i32
      aps.write_smem %cst_26, %2[%c35_i32] {aps.mem_id = 0 : i64, endtime = 78 : i32, ref_endtime = 78 : i32, ref_starttime = 77 : i32, slot = 0 : i32, starttime = 77 : i32} : f32, memref<36xf32>, i32
      aps.copy_by @busitfc, (%1, %2)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 87 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 87 : i32, ref_starttime = 78 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 78 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 12 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      %7 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_70", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c16_i32], %7, %c16_i32 {after = array<i64: 12, 12>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 97 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 97 : i32, ref_starttime = 79 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 79 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 13 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      %8 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_71", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c32_i32], %8, %c16_i32 {after = array<i64: 13, 12, 13>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 112 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 112 : i32, ref_starttime = 93 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 93 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 14 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      %9 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_72", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c48_i32], %9, %c16_i32 {after = array<i64: 14, 13, 14>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 122 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 20 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 122 : i32, ref_starttime = 103 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 103 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 15 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      %10 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_73", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c64_i32], %10, %c8_i32 {after = array<i64: 15, 14, 15>, after_gap = array<i64: 10, 20, 10>, after_segment = 3 : i64, endtime = 128 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 45 : i64, memory_issue_cycle = 30 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 128 : i32, ref_starttime = 113 : i32, required_alignment = 32 : i64, segment = 4 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 113 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 16 : i64, transfer_bytes = 32 : i64, transfer_offset = 256 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_44__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 2 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/sfm_batch/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_74", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = memref.get_global @au_generated_candidate_44__l0u_buffer_0 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %2 = memref.get_global @au_generated_candidate_44__l0u_buffer_1 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%1, %2)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 16 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 3.200000e+01 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 17 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      %3 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_75", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %3, (%1, %2)[%c16_i32], %c8_i32 {after = array<i64: 17, 17>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 16 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 16 : i64, memory_issue_cycle = 1 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 16 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 16 : i32, ref_starttime = 1 : i32, required_alignment = 32 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 3.200000e+01 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 18 : i64, transfer_bytes = 32 : i64, transfer_offset = 64 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      aps.write_smem %cst_6, %1[%c0_i32] {aps.mem_id = 23 : i64, endtime = 17 : i32, ref_endtime = 17 : i32, ref_starttime = 16 : i32, slot = 0 : i32, starttime = 16 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_7, %2[%c0_i32] {aps.mem_id = 22 : i64, endtime = 17 : i32, ref_endtime = 17 : i32, ref_starttime = 16 : i32, slot = 0 : i32, starttime = 16 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_8, %1[%c1_i32] {aps.mem_id = 21 : i64, endtime = 18 : i32, ref_endtime = 18 : i32, ref_starttime = 17 : i32, slot = 0 : i32, starttime = 17 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_9, %2[%c1_i32] {aps.mem_id = 20 : i64, endtime = 18 : i32, ref_endtime = 18 : i32, ref_starttime = 17 : i32, slot = 0 : i32, starttime = 17 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_8, %1[%c2_i32] {aps.mem_id = 19 : i64, endtime = 19 : i32, ref_endtime = 19 : i32, ref_starttime = 18 : i32, slot = 0 : i32, starttime = 18 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_10, %2[%c2_i32] {aps.mem_id = 18 : i64, endtime = 19 : i32, ref_endtime = 19 : i32, ref_starttime = 18 : i32, slot = 0 : i32, starttime = 18 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_11, %1[%c3_i32] {aps.mem_id = 17 : i64, endtime = 20 : i32, ref_endtime = 20 : i32, ref_starttime = 19 : i32, slot = 0 : i32, starttime = 19 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_12, %2[%c3_i32] {aps.mem_id = 16 : i64, endtime = 20 : i32, ref_endtime = 20 : i32, ref_starttime = 19 : i32, slot = 0 : i32, starttime = 19 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_8, %1[%c4_i32] {aps.mem_id = 15 : i64, endtime = 21 : i32, ref_endtime = 21 : i32, ref_starttime = 20 : i32, slot = 0 : i32, starttime = 20 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_13, %2[%c4_i32] {aps.mem_id = 14 : i64, endtime = 21 : i32, ref_endtime = 21 : i32, ref_starttime = 20 : i32, slot = 0 : i32, starttime = 20 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_7, %1[%c5_i32] {aps.mem_id = 13 : i64, endtime = 22 : i32, ref_endtime = 22 : i32, ref_starttime = 21 : i32, slot = 0 : i32, starttime = 21 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_10, %2[%c5_i32] {aps.mem_id = 12 : i64, endtime = 22 : i32, ref_endtime = 22 : i32, ref_starttime = 21 : i32, slot = 0 : i32, starttime = 21 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_14, %1[%c6_i32] {aps.mem_id = 11 : i64, endtime = 23 : i32, ref_endtime = 23 : i32, ref_starttime = 22 : i32, slot = 0 : i32, starttime = 22 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_8, %2[%c6_i32] {aps.mem_id = 10 : i64, endtime = 23 : i32, ref_endtime = 23 : i32, ref_starttime = 22 : i32, slot = 0 : i32, starttime = 22 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_8, %1[%c7_i32] {aps.mem_id = 9 : i64, endtime = 24 : i32, ref_endtime = 24 : i32, ref_starttime = 23 : i32, slot = 0 : i32, starttime = 23 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_15, %2[%c7_i32] {aps.mem_id = 8 : i64, endtime = 24 : i32, ref_endtime = 24 : i32, ref_starttime = 23 : i32, slot = 0 : i32, starttime = 23 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_12, %1[%c8_i32] {aps.mem_id = 7 : i64, endtime = 25 : i32, ref_endtime = 25 : i32, ref_starttime = 24 : i32, slot = 0 : i32, starttime = 24 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_10, %2[%c8_i32] {aps.mem_id = 6 : i64, endtime = 25 : i32, ref_endtime = 25 : i32, ref_starttime = 24 : i32, slot = 0 : i32, starttime = 24 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_16, %1[%c9_i32] {aps.mem_id = 5 : i64, endtime = 26 : i32, ref_endtime = 26 : i32, ref_starttime = 25 : i32, slot = 0 : i32, starttime = 25 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_7, %2[%c9_i32] {aps.mem_id = 4 : i64, endtime = 26 : i32, ref_endtime = 26 : i32, ref_starttime = 25 : i32, slot = 0 : i32, starttime = 25 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_8, %1[%c10_i32] {aps.mem_id = 3 : i64, endtime = 27 : i32, ref_endtime = 27 : i32, ref_starttime = 26 : i32, slot = 0 : i32, starttime = 26 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_17, %2[%c10_i32] {aps.mem_id = 2 : i64, endtime = 27 : i32, ref_endtime = 27 : i32, ref_starttime = 26 : i32, slot = 0 : i32, starttime = 26 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_8, %1[%c11_i32] {aps.mem_id = 1 : i64, endtime = 28 : i32, ref_endtime = 28 : i32, ref_starttime = 27 : i32, slot = 0 : i32, starttime = 27 : i32} : f32, memref<12xf32>, i32
      aps.write_smem %cst_10, %2[%c11_i32] {aps.mem_id = 0 : i64, endtime = 28 : i32, ref_endtime = 28 : i32, ref_starttime = 27 : i32, slot = 0 : i32, starttime = 27 : i32} : f32, memref<12xf32>, i32
      aps.copy_by @busitfc, (%1, %2)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 37 : i32, estimated_latency = 15 : i64, interface_cost = 31 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 15 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 37 : i32, ref_starttime = 28 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 3.100000e+01 : f64, starttime = 28 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 19 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<12xf32>, memref<12xf32>), i32, i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_76", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c16_i32], %4, %c8_i32 {after = array<i64: 19, 19>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 43 : i32, estimated_latency = 15 : i64, interface_cost = 31 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 15 : i64, memory_issue_cycle = 1 : i64, memory_latency = 14 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 15 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 43 : i32, ref_starttime = 29 : i32, required_alignment = 32 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 3.100000e+01 : f64, starttime = 29 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 20 : i64, transfer_bytes = 32 : i64, transfer_offset = 64 : i64} : (memref<12xf32>, memref<12xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_48__l0p_l1u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 3 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/sfm_batch/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_77", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_78", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_48__l0p_l1u_buffer_0 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_48__l0p_l1u_buffer_1_2 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64: 21, 21>, after_gap = array<i64: 1, 1>, after_segment = 4 : i64, endtime = 31 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 16 : i64, memory_issue_cycle = 1 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 31 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 16 : i32, tl_channel = 1 : i32, transaction_group = 5 : i64, transaction_id = 22 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_79", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 22, 21, 22>, after_gap = array<i64: 8, 9, 8>, after_segment = 0 : i64, endtime = 45 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 24 : i64, memory_issue_cycle = 9 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 45 : i32, ref_starttime = 30 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 30 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 23 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %5 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_80", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 23, 22, 23>, after_gap = array<i64: 8, 16, 8>, after_segment = 1 : i64, endtime = 60 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 32 : i64, memory_issue_cycle = 17 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 60 : i32, ref_starttime = 45 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 45 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 24 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %6 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_81", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%2, %3)[%c48_i32], %c16_i32 {after = array<i64: 24, 23, 24>, after_gap = array<i64: 8, 16, 8>, after_segment = 2 : i64, endtime = 75 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 40 : i64, memory_issue_cycle = 25 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 75 : i32, ref_starttime = 60 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 60 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 25 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %7 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_82", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%2, %3)[%c64_i32], %c8_i32 {after = array<i64>, after_gap = array<i64>, endtime = 9 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 8 : i64, memory_issue_cycle = 0 : i64, memory_latency = 8 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 9 : i32, ref_starttime = 1 : i32, required_alignment = 32 : i64, segment = 4 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 5 : i64, transaction_id = 21 : i64, transfer_bytes = 32 : i64, transfer_offset = 256 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %8 = memref.get_global @au_generated_candidate_48__l0p_l1u_buffer_1_0 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %9 = memref.get_global @au_generated_candidate_48__l0p_l1u_buffer_1_1 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%8, %9)[%c0_i32], %c16_i32 {after = array<i64: 25, 24, 25>, after_gap = array<i64: 8, 16, 8>, endtime = 90 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 48 : i64, memory_issue_cycle = 33 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 90 : i32, ref_starttime = 75 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 75 : i32, tl_channel = 0 : i32, transaction_group = 6 : i64, transaction_id = 26 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %10 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_83", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %10, (%8, %9)[%c16_i32], %c16_i32 {after = array<i64: 26, 25, 26>, after_gap = array<i64: 8, 16, 8>, after_segment = 0 : i64, endtime = 98 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 56 : i64, memory_issue_cycle = 41 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 98 : i32, ref_starttime = 83 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 83 : i32, tl_channel = 1 : i32, transaction_group = 6 : i64, transaction_id = 27 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %11 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_84", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %11, (%8, %9)[%c32_i32], %c16_i32 {after = array<i64: 27, 26, 27>, after_gap = array<i64: 8, 16, 8>, after_segment = 1 : i64, endtime = 106 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 64 : i64, memory_issue_cycle = 49 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 106 : i32, ref_starttime = 91 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 91 : i32, tl_channel = 0 : i32, transaction_group = 6 : i64, transaction_id = 28 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %12 = tor.addi %1 %c192_i32 on (0 to 1) {dump = "unknown_85", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %12, (%8, %9)[%c48_i32], %c16_i32 {after = array<i64: 28, 27, 28>, after_gap = array<i64: 8, 16, 8>, after_segment = 2 : i64, endtime = 114 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 72 : i64, memory_issue_cycle = 57 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 114 : i32, ref_starttime = 99 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 99 : i32, tl_channel = 1 : i32, transaction_group = 6 : i64, transaction_id = 29 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %13 = tor.addi %1 %c256_i32 on (0 to 1) {dump = "unknown_86", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %13, (%8, %9)[%c64_i32], %c8_i32 {after = array<i64: 29, 28, 29>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 118 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 76 : i64, memory_issue_cycle = 65 : i64, memory_latency = 11 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 76 : i64, memory_search_hits = 11 : i64, memory_search_states = 50 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 118 : i32, ref_starttime = 107 : i32, required_alignment = 32 : i64, segment = 4 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 107 : i32, tl_channel = 0 : i32, transaction_group = 6 : i64, transaction_id = 30 : i64, transfer_bytes = 32 : i64, transfer_offset = 256 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %14 = aps.read_csr @au_generated_candidate_48__l0p_l1u_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %15 = memref.get_global @au_generated_candidate_48__l0p_l1u_buffer_2_0 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %16 = memref.get_global @au_generated_candidate_48__l0p_l1u_buffer_2_1 : memref<36xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %14, (%15, %16)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 7 : i64, transaction_id = 31 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %17 = tor.addi %14 %c64_i32 on (0 to 1) {dump = "unknown_87", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %17, (%15, %16)[%c16_i32], %c16_i32 {after = array<i64: 31, 31>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 34 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 34 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 7 : i64, transaction_id = 32 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %18 = tor.addi %14 %c128_i32 on (0 to 1) {dump = "unknown_88", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %18, (%15, %16)[%c32_i32], %c16_i32 {after = array<i64: 32, 31, 32>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 46 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 46 : i32, ref_starttime = 31 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 31 : i32, tl_channel = 1 : i32, transaction_group = 7 : i64, transaction_id = 33 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %19 = tor.addi %14 %c192_i32 on (0 to 1) {dump = "unknown_89", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %19, (%15, %16)[%c48_i32], %c16_i32 {after = array<i64: 33, 32, 33>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 61 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 61 : i32, ref_starttime = 46 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 46 : i32, tl_channel = 1 : i32, transaction_group = 7 : i64, transaction_id = 34 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      %20 = tor.addi %14 %c256_i32 on (0 to 1) {dump = "unknown_90", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %20, (%15, %16)[%c64_i32], %c8_i32 {after = array<i64: 34, 33, 34>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 72 : i32, estimated_latency = 40 : i64, interface_cost = 80 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 40 : i64, memory_issue_cycle = 29 : i64, memory_latency = 11 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 40 : i64, memory_search_hits = 1 : i64, memory_search_states = 20 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 72 : i32, ref_starttime = 61 : i32, required_alignment = 32 : i64, segment = 4 : i64, segment_count = 5 : i64, solver_objective = 2.320000e+02 : f64, starttime = 61 : i32, tl_channel = 1 : i32, transaction_group = 7 : i64, transaction_id = 35 : i64, transfer_bytes = 32 : i64, transfer_offset = 256 : i64} : i32, (memref<36xf32>, memref<36xf32>), i32, i32
      tor.for %arg3 = (%c0_i5 : i5) to (%c-9_i5 : i5) step (%c1_i5 : i5)
      on (1 to 59){
        %25 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %26 = tor.muli %25 %c3_i32 on (2 to 3) {dump = "unknown_91", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %27 = tor.cmpi slt %26 %c0_i32 on (3 to 4) {dump = "unknown_92", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %28 = tor.subi %c-1_i32 %26 on (3 to 4) {dump = "unknown_93", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %29 = arith.select %27, %28, %26 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %30 = tor.cmpi slt %29 %c0_i32 on (5 to 6) {dump = "unknown_94", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %31 = arith.extui %30 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %32 = tor.addi %29 %31 on (7 to 8) {dump = "unknown_95", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %33 = arith.shrsi %32, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %34 = tor.subi %c-1_i32 %33 on (9 to 10) {dump = "unknown_96", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %35 = arith.select %27, %34, %33 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %36 = aps.read_smem %2[%35] {aps.mem_id = 23 : i64, endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, slot = 0 : i32, starttime = 11 : i32} : memref<36xf32>, i32 -> f32
        %37 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %38 = tor.muli %37 %c3_i32 on (2 to 3) {dump = "unknown_97", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %39 = tor.cmpi slt %38 %c0_i32 on (3 to 4) {dump = "unknown_98", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %40 = tor.subi %c-1_i32 %38 on (3 to 4) {dump = "unknown_99", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %41 = arith.select %39, %40, %38 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %42 = tor.cmpi slt %41 %c0_i32 on (5 to 6) {dump = "unknown_100", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %43 = arith.extui %42 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %44 = tor.addi %41 %43 on (7 to 8) {dump = "unknown_101", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %45 = arith.shrsi %44, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %46 = tor.subi %c-1_i32 %45 on (9 to 10) {dump = "unknown_102", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %47 = arith.select %39, %46, %45 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %48 = aps.read_smem %3[%47] {aps.mem_id = 22 : i64, endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, slot = 0 : i32, starttime = 11 : i32} : memref<36xf32>, i32 -> f32
        %49 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %50 = arith.andi %49, %c1_i32 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
        %51 = tor.cmpi eq %50 %c0_i32 on (3 to 4) {dump = "unknown_103", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %52 = arith.select %51, %36, %48 {endtime = 13 : i32, ref_endtime = 13 : i32, ref_starttime = 12 : i32, starttime = 12 : i32} : f32
        %53 = tor.mulf %52 %cst_5 on (13 to 16) {dump = "unknown_104", fastmath = #arith.fastmath<none>, ref_endtime = 16 : i32, ref_starttime = 13 : i32} : f32
        %54 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %55 = tor.muli %54 %c3_i32 on (2 to 3) {dump = "unknown_105", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %56 = tor.cmpi slt %55 %c0_i32 on (3 to 4) {dump = "unknown_106", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %57 = tor.subi %c-1_i32 %55 on (3 to 4) {dump = "unknown_107", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %58 = arith.select %56, %57, %55 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %59 = tor.cmpi slt %58 %c0_i32 on (5 to 6) {dump = "unknown_108", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %60 = arith.extui %59 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %61 = tor.addi %58 %60 on (7 to 8) {dump = "unknown_109", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %62 = arith.shrsi %61, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %63 = tor.subi %c-1_i32 %62 on (9 to 10) {dump = "unknown_110", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %64 = arith.select %56, %63, %62 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %65 = aps.read_smem %8[%64] {aps.mem_id = 21 : i64, endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, slot = 0 : i32, starttime = 11 : i32} : memref<36xf32>, i32 -> f32
        %66 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %67 = tor.muli %66 %c3_i32 on (2 to 3) {dump = "unknown_111", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %68 = tor.cmpi slt %67 %c0_i32 on (3 to 4) {dump = "unknown_112", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %69 = tor.subi %c-1_i32 %67 on (3 to 4) {dump = "unknown_113", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %70 = arith.select %68, %69, %67 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %71 = tor.cmpi slt %70 %c0_i32 on (5 to 6) {dump = "unknown_114", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %72 = arith.extui %71 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %73 = tor.addi %70 %72 on (7 to 8) {dump = "unknown_115", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %74 = arith.shrsi %73, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %75 = tor.subi %c-1_i32 %74 on (9 to 10) {dump = "unknown_116", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %76 = arith.select %68, %75, %74 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %77 = aps.read_smem %9[%76] {aps.mem_id = 20 : i64, endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, slot = 0 : i32, starttime = 11 : i32} : memref<36xf32>, i32 -> f32
        %78 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %79 = arith.andi %78, %c1_i32 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
        %80 = tor.cmpi eq %79 %c0_i32 on (3 to 4) {dump = "unknown_117", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %81 = arith.select %80, %65, %77 {endtime = 13 : i32, ref_endtime = 13 : i32, ref_starttime = 12 : i32, starttime = 12 : i32} : f32
        %82 = tor.divf %53 %81 on (16 to 43) {dump = "unknown_118", fastmath = #arith.fastmath<none>, ref_endtime = 43 : i32, ref_starttime = 16 : i32} : f32
        %83 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %84 = tor.muli %83 %c3_i32 on (2 to 3) {dump = "unknown_119", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %85 = tor.cmpi slt %84 %c0_i32 on (3 to 4) {dump = "unknown_120", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %86 = tor.subi %c-1_i32 %84 on (3 to 4) {dump = "unknown_121", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %87 = arith.select %85, %86, %84 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %88 = tor.cmpi slt %87 %c0_i32 on (5 to 6) {dump = "unknown_122", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %89 = arith.extui %88 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %90 = tor.addi %87 %89 on (7 to 8) {dump = "unknown_123", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %91 = arith.shrsi %90, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %92 = tor.subi %c-1_i32 %91 on (9 to 10) {dump = "unknown_124", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %93 = arith.select %85, %92, %91 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %94 = aps.read_smem %15[%93] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 17 : i64}], aps.mem_id = 19 : i64, endtime = 33 : i32, ref_endtime = 33 : i32, ref_starttime = 32 : i32, slot = 0 : i32, starttime = 32 : i32} : memref<36xf32>, i32 -> f32
        %95 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %96 = tor.muli %95 %c3_i32 on (2 to 3) {dump = "unknown_125", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %97 = tor.cmpi slt %96 %c0_i32 on (3 to 4) {dump = "unknown_126", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %98 = tor.subi %c-1_i32 %96 on (3 to 4) {dump = "unknown_127", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %99 = arith.select %97, %98, %96 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %100 = tor.cmpi slt %99 %c0_i32 on (5 to 6) {dump = "unknown_128", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %101 = arith.extui %100 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %102 = tor.addi %99 %101 on (7 to 8) {dump = "unknown_129", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %103 = arith.shrsi %102, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %104 = tor.subi %c-1_i32 %103 on (9 to 10) {dump = "unknown_130", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %105 = arith.select %97, %104, %103 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %106 = aps.read_smem %16[%105] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 16 : i64}], aps.mem_id = 18 : i64, endtime = 33 : i32, ref_endtime = 33 : i32, ref_starttime = 32 : i32, slot = 0 : i32, starttime = 32 : i32} : memref<36xf32>, i32 -> f32
        %107 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %108 = arith.andi %107, %c1_i32 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
        %109 = tor.cmpi eq %108 %c0_i32 on (3 to 4) {dump = "unknown_131", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %110 = arith.select %109, %94, %106 {endtime = 34 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 33 : i32} : f32
        %111 = tor.subf %110 %82 on (43 to 46) {dump = "unknown_132", fastmath = #arith.fastmath<none>, ref_endtime = 46 : i32, ref_starttime = 43 : i32} : f32
        %112 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %113 = arith.andi %112, %c1_i32 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
        %114 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %115 = tor.muli %114 %c3_i32 on (2 to 3) {dump = "unknown_133", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %116 = tor.cmpi slt %115 %c0_i32 on (3 to 4) {dump = "unknown_134", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %117 = tor.subi %c-1_i32 %115 on (3 to 4) {dump = "unknown_135", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %118 = arith.select %116, %117, %115 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %119 = tor.cmpi slt %118 %c0_i32 on (5 to 6) {dump = "unknown_136", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %120 = arith.extui %119 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %121 = tor.addi %118 %120 on (7 to 8) {dump = "unknown_137", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %122 = arith.shrsi %121, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %123 = tor.subi %c-1_i32 %122 on (9 to 10) {dump = "unknown_138", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %124 = arith.select %116, %123, %122 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %125 = tor.cmpi eq %113 %c0_i32 on (3 to 4) {dump = "unknown_139", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        aps.write_smem_if %125, %111, %15[%124] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 3 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 11 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 19 : i64}], aps.mem_id = 17 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : f32, memref<36xf32>, i32
        %126 = tor.cmpi eq %113 %c1_i32 on (3 to 4) {dump = "unknown_140", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        aps.write_smem_if %126, %111, %16[%124] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 2 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 10 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 18 : i64}], aps.mem_id = 16 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : f32, memref<36xf32>, i32
        %127 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %128 = tor.muli %127 %c3_i32 on (2 to 3) {dump = "unknown_141", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %129 = tor.addi %128 %c1_i32 on (3 to 4) {dump = "unknown_142", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %130 = tor.cmpi slt %129 %c0_i32 on (4 to 5) {dump = "unknown_143", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
        %131 = tor.subi %c-2_i32 %128 on (3 to 4) {dump = "unknown_144", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %132 = arith.select %130, %131, %129 {endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, starttime = 5 : i32} : i32
        %133 = tor.cmpi slt %132 %c0_i32 on (6 to 7) {dump = "unknown_145", ref_endtime = 7 : i32, ref_starttime = 6 : i32} : (i32, i32) -> i1
        %134 = arith.extui %133 {endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, starttime = 7 : i32} : i1 to i32
        %135 = tor.addi %132 %134 on (8 to 9) {dump = "unknown_146", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i32
        %136 = arith.shrsi %135, %c1_i32 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i32
        %137 = tor.subi %c-1_i32 %136 on (10 to 11) {dump = "unknown_147", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
        %138 = arith.select %130, %137, %136 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
        %139 = aps.read_smem %2[%138] {aps.mem_id = 15 : i64, endtime = 13 : i32, ref_endtime = 13 : i32, ref_starttime = 12 : i32, slot = 0 : i32, starttime = 12 : i32} : memref<36xf32>, i32 -> f32
        %140 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %141 = tor.muli %140 %c3_i32 on (2 to 3) {dump = "unknown_148", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %142 = tor.addi %141 %c1_i32 on (3 to 4) {dump = "unknown_149", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %143 = tor.cmpi slt %142 %c0_i32 on (4 to 5) {dump = "unknown_150", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
        %144 = tor.subi %c-2_i32 %141 on (3 to 4) {dump = "unknown_151", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %145 = arith.select %143, %144, %142 {endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, starttime = 5 : i32} : i32
        %146 = tor.cmpi slt %145 %c0_i32 on (6 to 7) {dump = "unknown_152", ref_endtime = 7 : i32, ref_starttime = 6 : i32} : (i32, i32) -> i1
        %147 = arith.extui %146 {endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, starttime = 7 : i32} : i1 to i32
        %148 = tor.addi %145 %147 on (8 to 9) {dump = "unknown_153", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i32
        %149 = arith.shrsi %148, %c1_i32 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i32
        %150 = tor.subi %c-1_i32 %149 on (10 to 11) {dump = "unknown_154", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
        %151 = arith.select %143, %150, %149 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
        %152 = aps.read_smem %3[%151] {aps.mem_id = 14 : i64, endtime = 13 : i32, ref_endtime = 13 : i32, ref_starttime = 12 : i32, slot = 0 : i32, starttime = 12 : i32} : memref<36xf32>, i32 -> f32
        %153 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %154 = tor.addi %153 %c1_i32 on (2 to 3) {dump = "unknown_155", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %155 = tor.cmpi slt %154 %c0_i32 on (3 to 4) {dump = "unknown_156", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %156 = tor.subi %c-2_i32 %153 on (2 to 3) {dump = "unknown_157", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %157 = arith.select %155, %156, %154 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %158 = tor.cmpi slt %157 %c0_i32 on (5 to 6) {dump = "unknown_158", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %159 = arith.extui %158 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %160 = tor.addi %157 %159 on (7 to 8) {dump = "unknown_159", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %161 = arith.shrsi %160, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %162 = tor.subi %c-1_i32 %161 on (9 to 10) {dump = "unknown_160", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %163 = arith.select %155, %162, %161 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %164 = tor.muli %163 %c-2_i32 on (11 to 12) {dump = "unknown_161", ref_endtime = 12 : i32, ref_starttime = 11 : i32} : (i32, i32) -> i32
        %165 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %166 = tor.addi %165 %164 on (12 to 13) {dump = "unknown_162", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
        %167 = tor.addi %166 %c1_i32 on (13 to 14) {dump = "unknown_163", ref_endtime = 14 : i32, ref_starttime = 13 : i32} : (i32, i32) -> i32
        %168 = tor.cmpi eq %167 %c0_i32 on (14 to 15) {dump = "unknown_164", ref_endtime = 15 : i32, ref_starttime = 14 : i32} : (i32, i32) -> i1
        %169 = arith.select %168, %139, %152 {endtime = 16 : i32, ref_endtime = 16 : i32, ref_starttime = 15 : i32, starttime = 15 : i32} : f32
        %170 = tor.mulf %169 %cst_5 on (16 to 19) {dump = "unknown_165", fastmath = #arith.fastmath<none>, ref_endtime = 19 : i32, ref_starttime = 16 : i32} : f32
        %171 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %172 = tor.muli %171 %c3_i32 on (2 to 3) {dump = "unknown_166", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %173 = tor.addi %172 %c1_i32 on (3 to 4) {dump = "unknown_167", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %174 = tor.cmpi slt %173 %c0_i32 on (4 to 5) {dump = "unknown_168", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
        %175 = tor.subi %c-2_i32 %172 on (3 to 4) {dump = "unknown_169", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %176 = arith.select %174, %175, %173 {endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, starttime = 5 : i32} : i32
        %177 = tor.cmpi slt %176 %c0_i32 on (8 to 9) {dump = "unknown_170", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i1
        %178 = arith.extui %177 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i1 to i32
        %179 = tor.addi %176 %178 on (10 to 11) {dump = "unknown_171", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
        %180 = arith.shrsi %179, %c1_i32 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
        %181 = tor.subi %c-1_i32 %180 on (12 to 13) {dump = "unknown_172", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
        %182 = arith.select %174, %181, %180 {endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, starttime = 13 : i32} : i32
        %183 = aps.read_smem %8[%182] {aps.mem_id = 13 : i64, endtime = 15 : i32, ref_endtime = 15 : i32, ref_starttime = 14 : i32, slot = 0 : i32, starttime = 14 : i32} : memref<36xf32>, i32 -> f32
        %184 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %185 = tor.muli %184 %c3_i32 on (2 to 3) {dump = "unknown_173", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %186 = tor.addi %185 %c1_i32 on (3 to 4) {dump = "unknown_174", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %187 = tor.cmpi slt %186 %c0_i32 on (4 to 5) {dump = "unknown_175", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
        %188 = tor.subi %c-2_i32 %185 on (3 to 4) {dump = "unknown_176", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %189 = arith.select %187, %188, %186 {endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, starttime = 5 : i32} : i32
        %190 = tor.cmpi slt %189 %c0_i32 on (8 to 9) {dump = "unknown_177", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i1
        %191 = arith.extui %190 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i1 to i32
        %192 = tor.addi %189 %191 on (10 to 11) {dump = "unknown_178", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
        %193 = arith.shrsi %192, %c1_i32 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
        %194 = tor.subi %c-1_i32 %193 on (12 to 13) {dump = "unknown_179", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
        %195 = arith.select %187, %194, %193 {endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, starttime = 13 : i32} : i32
        %196 = aps.read_smem %9[%195] {aps.mem_id = 12 : i64, endtime = 15 : i32, ref_endtime = 15 : i32, ref_starttime = 14 : i32, slot = 0 : i32, starttime = 14 : i32} : memref<36xf32>, i32 -> f32
        %197 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %198 = tor.addi %197 %c1_i32 on (2 to 3) {dump = "unknown_180", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %199 = tor.cmpi slt %198 %c0_i32 on (3 to 4) {dump = "unknown_181", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %200 = tor.subi %c-2_i32 %197 on (2 to 3) {dump = "unknown_182", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %201 = arith.select %199, %200, %198 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %202 = tor.cmpi slt %201 %c0_i32 on (5 to 6) {dump = "unknown_183", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %203 = arith.extui %202 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %204 = tor.addi %201 %203 on (7 to 8) {dump = "unknown_184", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %205 = arith.shrsi %204, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %206 = tor.subi %c-1_i32 %205 on (9 to 10) {dump = "unknown_185", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %207 = arith.select %199, %206, %205 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %208 = tor.muli %207 %c-2_i32 on (11 to 12) {dump = "unknown_186", ref_endtime = 12 : i32, ref_starttime = 11 : i32} : (i32, i32) -> i32
        %209 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %210 = tor.addi %209 %208 on (12 to 13) {dump = "unknown_187", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
        %211 = tor.addi %210 %c1_i32 on (13 to 14) {dump = "unknown_188", ref_endtime = 14 : i32, ref_starttime = 13 : i32} : (i32, i32) -> i32
        %212 = tor.cmpi eq %211 %c0_i32 on (14 to 15) {dump = "unknown_189", ref_endtime = 15 : i32, ref_starttime = 14 : i32} : (i32, i32) -> i1
        %213 = arith.select %212, %183, %196 {endtime = 16 : i32, ref_endtime = 16 : i32, ref_starttime = 15 : i32, starttime = 15 : i32} : f32
        %214 = tor.divf %170 %213 on (19 to 46) {dump = "unknown_190", fastmath = #arith.fastmath<none>, ref_endtime = 46 : i32, ref_starttime = 19 : i32} : f32
        %215 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %216 = tor.muli %215 %c3_i32 on (2 to 3) {dump = "unknown_191", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %217 = tor.addi %216 %c1_i32 on (3 to 4) {dump = "unknown_192", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %218 = tor.cmpi slt %217 %c0_i32 on (4 to 5) {dump = "unknown_193", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
        %219 = tor.subi %c-2_i32 %216 on (3 to 4) {dump = "unknown_194", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %220 = arith.select %218, %219, %217 {endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, starttime = 5 : i32} : i32
        %221 = tor.cmpi slt %220 %c0_i32 on (6 to 7) {dump = "unknown_195", ref_endtime = 7 : i32, ref_starttime = 6 : i32} : (i32, i32) -> i1
        %222 = arith.extui %221 {endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, starttime = 7 : i32} : i1 to i32
        %223 = tor.addi %220 %222 on (8 to 9) {dump = "unknown_196", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i32
        %224 = arith.shrsi %223, %c1_i32 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i32
        %225 = tor.subi %c-1_i32 %224 on (10 to 11) {dump = "unknown_197", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
        %226 = arith.select %218, %225, %224 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
        %227 = aps.read_smem %15[%226] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 9 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 17 : i64}], aps.mem_id = 11 : i64, endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, slot = 0 : i32, starttime = 47 : i32} : memref<36xf32>, i32 -> f32
        %228 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %229 = tor.muli %228 %c3_i32 on (2 to 3) {dump = "unknown_198", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %230 = tor.addi %229 %c1_i32 on (3 to 4) {dump = "unknown_199", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %231 = tor.cmpi slt %230 %c0_i32 on (4 to 5) {dump = "unknown_200", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
        %232 = tor.subi %c-2_i32 %229 on (3 to 4) {dump = "unknown_201", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %233 = arith.select %231, %232, %230 {endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, starttime = 5 : i32} : i32
        %234 = tor.cmpi slt %233 %c0_i32 on (6 to 7) {dump = "unknown_202", ref_endtime = 7 : i32, ref_starttime = 6 : i32} : (i32, i32) -> i1
        %235 = arith.extui %234 {endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, starttime = 7 : i32} : i1 to i32
        %236 = tor.addi %233 %235 on (8 to 9) {dump = "unknown_203", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i32
        %237 = arith.shrsi %236, %c1_i32 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i32
        %238 = tor.subi %c-1_i32 %237 on (10 to 11) {dump = "unknown_204", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
        %239 = arith.select %231, %238, %237 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
        %240 = aps.read_smem %16[%239] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 8 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 16 : i64}], aps.mem_id = 10 : i64, endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, slot = 0 : i32, starttime = 47 : i32} : memref<36xf32>, i32 -> f32
        %241 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %242 = tor.addi %241 %c1_i32 on (2 to 3) {dump = "unknown_205", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %243 = tor.cmpi slt %242 %c0_i32 on (3 to 4) {dump = "unknown_206", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %244 = tor.subi %c-2_i32 %241 on (2 to 3) {dump = "unknown_207", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %245 = arith.select %243, %244, %242 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %246 = tor.cmpi slt %245 %c0_i32 on (5 to 6) {dump = "unknown_208", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %247 = arith.extui %246 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %248 = tor.addi %245 %247 on (7 to 8) {dump = "unknown_209", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %249 = arith.shrsi %248, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %250 = tor.subi %c-1_i32 %249 on (9 to 10) {dump = "unknown_210", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %251 = arith.select %243, %250, %249 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %252 = tor.muli %251 %c-2_i32 on (11 to 12) {dump = "unknown_211", ref_endtime = 12 : i32, ref_starttime = 11 : i32} : (i32, i32) -> i32
        %253 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %254 = tor.addi %253 %252 on (12 to 13) {dump = "unknown_212", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
        %255 = tor.addi %254 %c1_i32 on (13 to 14) {dump = "unknown_213", ref_endtime = 14 : i32, ref_starttime = 13 : i32} : (i32, i32) -> i32
        %256 = tor.cmpi eq %255 %c0_i32 on (14 to 15) {dump = "unknown_214", ref_endtime = 15 : i32, ref_starttime = 14 : i32} : (i32, i32) -> i1
        %257 = arith.select %256, %227, %240 {endtime = 49 : i32, ref_endtime = 49 : i32, ref_starttime = 48 : i32, starttime = 48 : i32} : f32
        %258 = tor.subf %257 %214 on (49 to 52) {dump = "unknown_215", fastmath = #arith.fastmath<none>, ref_endtime = 52 : i32, ref_starttime = 49 : i32} : f32
        %259 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %260 = tor.addi %259 %c1_i32 on (2 to 3) {dump = "unknown_216", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %261 = tor.cmpi slt %260 %c0_i32 on (3 to 4) {dump = "unknown_217", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %262 = tor.subi %c-2_i32 %259 on (2 to 3) {dump = "unknown_218", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %263 = arith.select %261, %262, %260 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %264 = tor.cmpi slt %263 %c0_i32 on (5 to 6) {dump = "unknown_219", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %265 = arith.extui %264 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %266 = tor.addi %263 %265 on (7 to 8) {dump = "unknown_220", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %267 = arith.shrsi %266, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %268 = tor.subi %c-1_i32 %267 on (9 to 10) {dump = "unknown_221", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %269 = arith.select %261, %268, %267 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %270 = tor.muli %269 %c-2_i32 on (11 to 12) {dump = "unknown_222", ref_endtime = 12 : i32, ref_starttime = 11 : i32} : (i32, i32) -> i32
        %271 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %272 = tor.addi %271 %270 on (12 to 13) {dump = "unknown_223", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
        %273 = tor.addi %272 %c1_i32 on (13 to 14) {dump = "unknown_224", ref_endtime = 14 : i32, ref_starttime = 13 : i32} : (i32, i32) -> i32
        %274 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %275 = tor.muli %274 %c3_i32 on (2 to 3) {dump = "unknown_225", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %276 = tor.addi %275 %c1_i32 on (3 to 4) {dump = "unknown_226", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %277 = tor.cmpi slt %276 %c0_i32 on (4 to 5) {dump = "unknown_227", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
        %278 = tor.subi %c-2_i32 %275 on (3 to 4) {dump = "unknown_228", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %279 = arith.select %277, %278, %276 {endtime = 6 : i32, ref_endtime = 6 : i32, ref_starttime = 5 : i32, starttime = 5 : i32} : i32
        %280 = tor.cmpi slt %279 %c0_i32 on (6 to 7) {dump = "unknown_229", ref_endtime = 7 : i32, ref_starttime = 6 : i32} : (i32, i32) -> i1
        %281 = arith.extui %280 {endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, starttime = 7 : i32} : i1 to i32
        %282 = tor.addi %279 %281 on (8 to 9) {dump = "unknown_230", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i32
        %283 = arith.shrsi %282, %c1_i32 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i32
        %284 = tor.subi %c-1_i32 %283 on (10 to 11) {dump = "unknown_231", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
        %285 = arith.select %277, %284, %283 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
        %286 = tor.cmpi eq %273 %c0_i32 on (14 to 15) {dump = "unknown_232", ref_endtime = 15 : i32, ref_starttime = 14 : i32} : (i32, i32) -> i1
        aps.write_smem_if %286, %258, %15[%285] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 3 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 11 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 19 : i64}], aps.mem_id = 9 : i64, endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, slot = 0 : i32, starttime = 52 : i32} : f32, memref<36xf32>, i32
        %287 = tor.cmpi eq %273 %c1_i32 on (14 to 15) {dump = "unknown_233", ref_endtime = 15 : i32, ref_starttime = 14 : i32} : (i32, i32) -> i1
        aps.write_smem_if %287, %258, %16[%285] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 2 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 10 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 18 : i64}], aps.mem_id = 8 : i64, endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, slot = 0 : i32, starttime = 52 : i32} : f32, memref<36xf32>, i32
        %288 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %289 = tor.muli %288 %c3_i32 on (2 to 3) {dump = "unknown_234", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %290 = tor.cmpi slt %289 %c0_i32 on (3 to 4) {dump = "unknown_235", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %291 = tor.subi %c-1_i32 %289 on (3 to 4) {dump = "unknown_236", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %292 = arith.select %290, %291, %289 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %293 = tor.cmpi slt %292 %c0_i32 on (6 to 7) {dump = "unknown_237", ref_endtime = 7 : i32, ref_starttime = 6 : i32} : (i32, i32) -> i1
        %294 = arith.extui %293 {endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, starttime = 7 : i32} : i1 to i32
        %295 = tor.addi %292 %294 on (8 to 9) {dump = "unknown_238", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i32
        %296 = arith.shrsi %295, %c1_i32 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i32
        %297 = tor.subi %c-1_i32 %296 on (10 to 11) {dump = "unknown_239", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
        %298 = arith.select %290, %297, %296 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
        %299 = tor.addi %298 %c1_i32 on (12 to 13) {dump = "unknown_240", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
        %300 = aps.read_smem %2[%299] {aps.mem_id = 7 : i64, endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, slot = 0 : i32, starttime = 13 : i32} : memref<36xf32>, i32 -> f32
        %301 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %302 = tor.muli %301 %c3_i32 on (2 to 3) {dump = "unknown_241", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %303 = tor.cmpi slt %302 %c0_i32 on (3 to 4) {dump = "unknown_242", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %304 = tor.subi %c-1_i32 %302 on (3 to 4) {dump = "unknown_243", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %305 = arith.select %303, %304, %302 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %306 = tor.cmpi slt %305 %c0_i32 on (6 to 7) {dump = "unknown_244", ref_endtime = 7 : i32, ref_starttime = 6 : i32} : (i32, i32) -> i1
        %307 = arith.extui %306 {endtime = 8 : i32, ref_endtime = 8 : i32, ref_starttime = 7 : i32, starttime = 7 : i32} : i1 to i32
        %308 = tor.addi %305 %307 on (8 to 9) {dump = "unknown_245", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i32
        %309 = arith.shrsi %308, %c1_i32 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i32
        %310 = tor.subi %c-1_i32 %309 on (10 to 11) {dump = "unknown_246", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
        %311 = arith.select %303, %310, %309 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
        %312 = tor.addi %311 %c1_i32 on (12 to 13) {dump = "unknown_247", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
        %313 = aps.read_smem %3[%312] {aps.mem_id = 6 : i64, endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, slot = 0 : i32, starttime = 13 : i32} : memref<36xf32>, i32 -> f32
        %314 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %315 = arith.andi %314, %c1_i32 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
        %316 = tor.cmpi eq %315 %c0_i32 on (3 to 4) {dump = "unknown_248", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %317 = arith.select %316, %300, %313 {endtime = 15 : i32, ref_endtime = 15 : i32, ref_starttime = 14 : i32, starttime = 14 : i32} : f32
        %318 = tor.mulf %317 %cst_5 on (15 to 18) {dump = "unknown_249", fastmath = #arith.fastmath<none>, ref_endtime = 18 : i32, ref_starttime = 15 : i32} : f32
        %319 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %320 = tor.muli %319 %c3_i32 on (2 to 3) {dump = "unknown_250", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %321 = tor.cmpi slt %320 %c0_i32 on (3 to 4) {dump = "unknown_251", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %322 = tor.subi %c-1_i32 %320 on (3 to 4) {dump = "unknown_252", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %323 = arith.select %321, %322, %320 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %324 = tor.cmpi slt %323 %c0_i32 on (5 to 6) {dump = "unknown_253", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %325 = arith.extui %324 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %326 = tor.addi %323 %325 on (7 to 8) {dump = "unknown_254", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %327 = arith.shrsi %326, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %328 = tor.subi %c-1_i32 %327 on (12 to 13) {dump = "unknown_255", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
        %329 = arith.select %321, %328, %327 {endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, starttime = 13 : i32} : i32
        %330 = tor.addi %329 %c1_i32 on (14 to 15) {dump = "unknown_256", ref_endtime = 15 : i32, ref_starttime = 14 : i32} : (i32, i32) -> i32
        %331 = aps.read_smem %8[%330] {aps.mem_id = 5 : i64, endtime = 16 : i32, ref_endtime = 16 : i32, ref_starttime = 15 : i32, slot = 0 : i32, starttime = 15 : i32} : memref<36xf32>, i32 -> f32
        %332 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %333 = tor.muli %332 %c3_i32 on (2 to 3) {dump = "unknown_257", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %334 = tor.cmpi slt %333 %c0_i32 on (3 to 4) {dump = "unknown_258", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %335 = tor.subi %c-1_i32 %333 on (3 to 4) {dump = "unknown_259", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %336 = arith.select %334, %335, %333 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %337 = tor.cmpi slt %336 %c0_i32 on (5 to 6) {dump = "unknown_260", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %338 = arith.extui %337 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %339 = tor.addi %336 %338 on (7 to 8) {dump = "unknown_261", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %340 = arith.shrsi %339, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %341 = tor.subi %c-1_i32 %340 on (9 to 10) {dump = "unknown_262", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %342 = arith.select %334, %341, %340 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %343 = tor.addi %342 %c1_i32 on (11 to 12) {dump = "unknown_263", ref_endtime = 12 : i32, ref_starttime = 11 : i32} : (i32, i32) -> i32
        %344 = aps.read_smem %9[%343] {aps.mem_id = 4 : i64, endtime = 16 : i32, ref_endtime = 16 : i32, ref_starttime = 15 : i32, slot = 0 : i32, starttime = 15 : i32} : memref<36xf32>, i32 -> f32
        %345 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %346 = arith.andi %345, %c1_i32 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
        %347 = tor.cmpi eq %346 %c0_i32 on (3 to 4) {dump = "unknown_264", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %348 = arith.select %347, %331, %344 {endtime = 17 : i32, ref_endtime = 17 : i32, ref_starttime = 16 : i32, starttime = 16 : i32} : f32
        %349 = tor.divf %318 %348 on (18 to 45) {dump = "unknown_265", fastmath = #arith.fastmath<none>, ref_endtime = 45 : i32, ref_starttime = 18 : i32} : f32
        %350 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %351 = tor.muli %350 %c3_i32 on (2 to 3) {dump = "unknown_266", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %352 = tor.cmpi slt %351 %c0_i32 on (3 to 4) {dump = "unknown_267", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %353 = tor.subi %c-1_i32 %351 on (3 to 4) {dump = "unknown_268", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %354 = arith.select %352, %353, %351 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %355 = tor.cmpi slt %354 %c0_i32 on (5 to 6) {dump = "unknown_269", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %356 = arith.extui %355 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %357 = tor.addi %354 %356 on (7 to 8) {dump = "unknown_270", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %358 = arith.shrsi %357, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %359 = tor.subi %c-1_i32 %358 on (9 to 10) {dump = "unknown_271", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %360 = arith.select %352, %359, %358 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %361 = tor.addi %360 %c1_i32 on (11 to 12) {dump = "unknown_272", ref_endtime = 12 : i32, ref_starttime = 11 : i32} : (i32, i32) -> i32
        %362 = aps.read_smem %15[%361] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 9 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 17 : i64}], aps.mem_id = 3 : i64, endtime = 54 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, slot = 0 : i32, starttime = 53 : i32} : memref<36xf32>, i32 -> f32
        %363 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %364 = tor.muli %363 %c3_i32 on (2 to 3) {dump = "unknown_273", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %365 = tor.cmpi slt %364 %c0_i32 on (3 to 4) {dump = "unknown_274", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %366 = tor.subi %c-1_i32 %364 on (3 to 4) {dump = "unknown_275", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %367 = arith.select %365, %366, %364 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %368 = tor.cmpi slt %367 %c0_i32 on (5 to 6) {dump = "unknown_276", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %369 = arith.extui %368 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %370 = tor.addi %367 %369 on (7 to 8) {dump = "unknown_277", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %371 = arith.shrsi %370, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %372 = tor.subi %c-1_i32 %371 on (9 to 10) {dump = "unknown_278", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %373 = arith.select %365, %372, %371 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %374 = tor.addi %373 %c1_i32 on (11 to 12) {dump = "unknown_279", ref_endtime = 12 : i32, ref_starttime = 11 : i32} : (i32, i32) -> i32
        %375 = aps.read_smem %16[%374] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 8 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 16 : i64}], aps.mem_id = 2 : i64, endtime = 54 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, slot = 0 : i32, starttime = 53 : i32} : memref<36xf32>, i32 -> f32
        %376 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %377 = arith.andi %376, %c1_i32 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
        %378 = tor.cmpi eq %377 %c0_i32 on (3 to 4) {dump = "unknown_280", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %379 = arith.select %378, %362, %375 {endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, starttime = 54 : i32} : f32
        %380 = tor.subf %379 %349 on (55 to 58) {dump = "unknown_281", fastmath = #arith.fastmath<none>, ref_endtime = 58 : i32, ref_starttime = 55 : i32} : f32
        %381 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %382 = arith.andi %381, %c1_i32 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
        %383 = arith.extui %arg3 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i5 to i32
        %384 = tor.muli %383 %c3_i32 on (2 to 3) {dump = "unknown_282", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
        %385 = tor.cmpi slt %384 %c0_i32 on (3 to 4) {dump = "unknown_283", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        %386 = tor.subi %c-1_i32 %384 on (3 to 4) {dump = "unknown_284", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
        %387 = arith.select %385, %386, %384 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
        %388 = tor.cmpi slt %387 %c0_i32 on (5 to 6) {dump = "unknown_285", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
        %389 = arith.extui %388 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
        %390 = tor.addi %387 %389 on (7 to 8) {dump = "unknown_286", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
        %391 = arith.shrsi %390, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
        %392 = tor.subi %c-1_i32 %391 on (9 to 10) {dump = "unknown_287", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
        %393 = arith.select %385, %392, %391 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
        %394 = tor.addi %393 %c1_i32 on (11 to 12) {dump = "unknown_288", ref_endtime = 12 : i32, ref_starttime = 11 : i32} : (i32, i32) -> i32
        %395 = tor.cmpi eq %382 %c0_i32 on (3 to 4) {dump = "unknown_289", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        aps.write_smem_if %395, %380, %15[%394] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 3 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 11 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 19 : i64}], aps.mem_id = 1 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<36xf32>, i32
        %396 = tor.cmpi eq %382 %c1_i32 on (3 to 4) {dump = "unknown_290", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
        aps.write_smem_if %396, %380, %16[%394] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 2 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 10 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 18 : i64}], aps.mem_id = 0 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<36xf32>, i32
      } {II = 27 : i32, aps.pipeline_achieved_ii = 27 : i64, aps.pipeline_requested, aps.pipeline_requested_ii = -1 : i64, pipeline = 1 : i32, ref_endtime = 59 : i32, ref_starttime = 1 : i32}
      aps.copy_by @busitfc, (%15, %16)[%c0_i32], %14, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 156 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 156 : i32, ref_starttime = 147 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 147 : i32, tl_channel = 0 : i32, transaction_group = 8 : i64, transaction_id = 36 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      %21 = tor.addi %14 %c64_i32 on (147 to 148) {dump = "unknown_291", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%15, %16)[%c16_i32], %21, %c16_i32 {after = array<i64: 36, 36>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 166 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 166 : i32, ref_starttime = 148 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 148 : i32, tl_channel = 1 : i32, transaction_group = 8 : i64, transaction_id = 37 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      %22 = tor.addi %14 %c128_i32 on (147 to 148) {dump = "unknown_292", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%15, %16)[%c32_i32], %22, %c16_i32 {after = array<i64: 37, 36, 37>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 181 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 181 : i32, ref_starttime = 162 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 162 : i32, tl_channel = 0 : i32, transaction_group = 8 : i64, transaction_id = 38 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      %23 = tor.addi %14 %c192_i32 on (147 to 148) {dump = "unknown_293", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%15, %16)[%c48_i32], %23, %c16_i32 {after = array<i64: 38, 37, 38>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 191 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 20 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 191 : i32, ref_starttime = 172 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 172 : i32, tl_channel = 1 : i32, transaction_group = 8 : i64, transaction_id = 39 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      %24 = tor.addi %14 %c256_i32 on (147 to 148) {dump = "unknown_294", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%15, %16)[%c64_i32], %24, %c8_i32 {after = array<i64: 39, 38, 39>, after_gap = array<i64: 10, 20, 10>, after_segment = 3 : i64, endtime = 197 : i32, estimated_latency = 45 : i64, interface_cost = 85 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 45 : i64, memory_issue_cycle = 30 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 45 : i64, memory_search_hits = 2 : i64, memory_search_states = 18 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 197 : i32, ref_starttime = 182 : i32, required_alignment = 32 : i64, segment = 4 : i64, segment_count = 5 : i64, solver_objective = 8.500000e+01 : f64, starttime = 182 : i32, tl_channel = 0 : i32, transaction_group = 8 : i64, transaction_id = 40 : i64, transfer_bytes = 32 : i64, transfer_offset = 256 : i64} : (memref<36xf32>, memref<36xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_49__l1p(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 4 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/sfm_batch/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg1 {dump = "unknown_295", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg0 {dump = "unknown_296", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_49__l1p_buffer_0 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_49__l1p_buffer_1 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64: 41, 41>, after_gap = array<i64: 1, 1>, after_segment = 1 : i64, endtime = 31 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 16 : i64, memory_issue_cycle = 1 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 16 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 31 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 8.800000e+01 : f64, starttime = 16 : i32, tl_channel = 1 : i32, transaction_group = 9 : i64, transaction_id = 42 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      %4 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_297", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c8_i32 {after = array<i64>, after_gap = array<i64>, endtime = 9 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 8 : i64, memory_issue_cycle = 0 : i64, memory_latency = 8 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 16 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 9 : i32, ref_starttime = 1 : i32, required_alignment = 32 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 8.800000e+01 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 9 : i64, transaction_id = 41 : i64, transfer_bytes = 32 : i64, transfer_offset = 64 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      %5 = memref.get_global @au_generated_candidate_49__l1p_buffer_3_0 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %6 = memref.get_global @au_generated_candidate_49__l1p_buffer_3_1 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%5, %6)[%c0_i32], %c16_i32 {after = array<i64: 42, 41, 42>, after_gap = array<i64: 8, 9, 8>, endtime = 45 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 24 : i64, memory_issue_cycle = 9 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 16 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 45 : i32, ref_starttime = 30 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 8.800000e+01 : f64, starttime = 30 : i32, tl_channel = 0 : i32, transaction_group = 10 : i64, transaction_id = 43 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      %7 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_298", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%5, %6)[%c16_i32], %c8_i32 {after = array<i64: 43, 42, 43>, after_gap = array<i64: 8, 16, 8>, after_segment = 0 : i64, endtime = 49 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 17 : i64, memory_latency = 11 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 28 : i64, memory_search_hits = 0 : i64, memory_search_states = 16 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 49 : i32, ref_starttime = 38 : i32, required_alignment = 32 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 8.800000e+01 : f64, starttime = 38 : i32, tl_channel = 1 : i32, transaction_group = 10 : i64, transaction_id = 44 : i64, transfer_bytes = 32 : i64, transfer_offset = 64 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      %8 = aps.read_csr @au_generated_candidate_49__l1p_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %9 = memref.get_global @au_generated_candidate_49__l1p_buffer_4_0 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %10 = memref.get_global @au_generated_candidate_49__l1p_buffer_4_1 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %8, (%9, %10)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 16 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 8.800000e+01 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 11 : i64, transaction_id = 45 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      %11 = tor.addi %8 %c64_i32 on (0 to 1) {dump = "unknown_299", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %11, (%9, %10)[%c16_i32], %c8_i32 {after = array<i64: 45, 45>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 30 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 16 : i64, memory_issue_cycle = 1 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 16 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 30 : i32, ref_starttime = 15 : i32, required_alignment = 32 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 8.800000e+01 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 11 : i64, transaction_id = 46 : i64, transfer_bytes = 32 : i64, transfer_offset = 64 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      tor.for %arg3 = (%c0_i4 : i4) to (%c7_i4 : i4) step (%c1_i4 : i4)
      on (1 to 52){
        %13 = arith.extui %arg3 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : i4 to i32
        tor.for %arg4 = (%c0_i2 : i2) to (%c-2_i2 : i2) step (%c1_i2 : i2)
        on (1 to 47){
          %14 = tor.muli %13 %c3_i32 on (1 to 2) {dump = "unknown_300", ref_endtime = 2 : i32, ref_starttime = 1 : i32} : (i32, i32) -> i32
          %15 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %16 = tor.addi %14 %15 on (2 to 3) {dump = "unknown_301", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
          %17 = tor.cmpi slt %16 %c0_i32 on (3 to 4) {dump = "unknown_302", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
          %18 = tor.subi %c-1_i32 %16 on (3 to 4) {dump = "unknown_303", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
          %19 = arith.select %17, %18, %16 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
          %20 = tor.cmpi slt %19 %c0_i32 on (5 to 6) {dump = "unknown_304", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
          %21 = arith.extui %20 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
          %22 = tor.addi %19 %21 on (7 to 8) {dump = "unknown_305", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
          %23 = arith.shrsi %22, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
          %24 = tor.subi %c-1_i32 %23 on (9 to 10) {dump = "unknown_306", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
          %25 = arith.select %17, %24, %23 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
          %26 = aps.read_smem %2[%25] {aps.mem_id = 7 : i64, endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, slot = 0 : i32, starttime = 11 : i32} : memref<12xf32>, i32 -> f32
          %27 = tor.muli %13 %c3_i32 on (1 to 2) {dump = "unknown_307", ref_endtime = 2 : i32, ref_starttime = 1 : i32} : (i32, i32) -> i32
          %28 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %29 = tor.addi %27 %28 on (2 to 3) {dump = "unknown_308", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
          %30 = tor.cmpi slt %29 %c0_i32 on (3 to 4) {dump = "unknown_309", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
          %31 = tor.subi %c-1_i32 %29 on (3 to 4) {dump = "unknown_310", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
          %32 = arith.select %30, %31, %29 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
          %33 = tor.cmpi slt %32 %c0_i32 on (5 to 6) {dump = "unknown_311", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
          %34 = arith.extui %33 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
          %35 = tor.addi %32 %34 on (7 to 8) {dump = "unknown_312", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
          %36 = arith.shrsi %35, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
          %37 = tor.subi %c-1_i32 %36 on (9 to 10) {dump = "unknown_313", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
          %38 = arith.select %30, %37, %36 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
          %39 = aps.read_smem %3[%38] {aps.mem_id = 6 : i64, endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, slot = 0 : i32, starttime = 11 : i32} : memref<12xf32>, i32 -> f32
          %40 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %41 = tor.addi %13 %40 on (9 to 10) {dump = "unknown_314", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
          %42 = arith.andi %41, %c1_i32 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
          %43 = tor.cmpi eq %42 %c0_i32 on (11 to 12) {dump = "unknown_315", ref_endtime = 12 : i32, ref_starttime = 11 : i32} : (i32, i32) -> i1
          %44 = arith.select %43, %26, %39 {endtime = 13 : i32, ref_endtime = 13 : i32, ref_starttime = 12 : i32, starttime = 12 : i32} : f32
          %45 = tor.mulf %44 %cst_4 on (13 to 16) {dump = "unknown_316", fastmath = #arith.fastmath<none>, ref_endtime = 16 : i32, ref_starttime = 13 : i32} : f32
          %46 = tor.muli %13 %c3_i32 on (1 to 2) {dump = "unknown_317", ref_endtime = 2 : i32, ref_starttime = 1 : i32} : (i32, i32) -> i32
          %47 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %48 = tor.addi %46 %47 on (2 to 3) {dump = "unknown_318", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
          %49 = tor.cmpi slt %48 %c0_i32 on (3 to 4) {dump = "unknown_319", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
          %50 = tor.subi %c-1_i32 %48 on (3 to 4) {dump = "unknown_320", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
          %51 = arith.select %49, %50, %48 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
          %52 = tor.cmpi slt %51 %c0_i32 on (8 to 9) {dump = "unknown_321", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i1
          %53 = arith.extui %52 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i1 to i32
          %54 = tor.addi %51 %53 on (10 to 11) {dump = "unknown_322", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
          %55 = arith.shrsi %54, %c1_i32 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
          %56 = tor.subi %c-1_i32 %55 on (12 to 13) {dump = "unknown_323", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
          %57 = arith.select %49, %56, %55 {endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, starttime = 13 : i32} : i32
          %58 = aps.read_smem %5[%57] {aps.mem_id = 5 : i64, endtime = 15 : i32, ref_endtime = 15 : i32, ref_starttime = 14 : i32, slot = 0 : i32, starttime = 14 : i32} : memref<12xf32>, i32 -> f32
          %59 = tor.muli %13 %c3_i32 on (1 to 2) {dump = "unknown_324", ref_endtime = 2 : i32, ref_starttime = 1 : i32} : (i32, i32) -> i32
          %60 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %61 = tor.addi %59 %60 on (2 to 3) {dump = "unknown_325", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
          %62 = tor.cmpi slt %61 %c0_i32 on (3 to 4) {dump = "unknown_326", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
          %63 = tor.subi %c-1_i32 %61 on (3 to 4) {dump = "unknown_327", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
          %64 = arith.select %62, %63, %61 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
          %65 = tor.cmpi slt %64 %c0_i32 on (8 to 9) {dump = "unknown_328", ref_endtime = 9 : i32, ref_starttime = 8 : i32} : (i32, i32) -> i1
          %66 = arith.extui %65 {endtime = 10 : i32, ref_endtime = 10 : i32, ref_starttime = 9 : i32, starttime = 9 : i32} : i1 to i32
          %67 = tor.addi %64 %66 on (10 to 11) {dump = "unknown_329", ref_endtime = 11 : i32, ref_starttime = 10 : i32} : (i32, i32) -> i32
          %68 = arith.shrsi %67, %c1_i32 {endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, starttime = 11 : i32} : i32
          %69 = tor.subi %c-1_i32 %68 on (12 to 13) {dump = "unknown_330", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
          %70 = arith.select %62, %69, %68 {endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, starttime = 13 : i32} : i32
          %71 = aps.read_smem %6[%70] {aps.mem_id = 4 : i64, endtime = 15 : i32, ref_endtime = 15 : i32, ref_starttime = 14 : i32, slot = 0 : i32, starttime = 14 : i32} : memref<12xf32>, i32 -> f32
          %72 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %73 = tor.addi %13 %72 on (12 to 13) {dump = "unknown_331", ref_endtime = 13 : i32, ref_starttime = 12 : i32} : (i32, i32) -> i32
          %74 = arith.andi %73, %c1_i32 {endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, starttime = 13 : i32} : i32
          %75 = tor.cmpi eq %74 %c0_i32 on (14 to 15) {dump = "unknown_332", ref_endtime = 15 : i32, ref_starttime = 14 : i32} : (i32, i32) -> i1
          %76 = arith.select %75, %58, %71 {endtime = 16 : i32, ref_endtime = 16 : i32, ref_starttime = 15 : i32, starttime = 15 : i32} : f32
          %77 = tor.divf %45 %76 on (16 to 43) {dump = "unknown_333", fastmath = #arith.fastmath<none>, ref_endtime = 43 : i32, ref_starttime = 16 : i32} : f32
          %78 = tor.muli %13 %c3_i32 on (1 to 2) {dump = "unknown_334", ref_endtime = 2 : i32, ref_starttime = 1 : i32} : (i32, i32) -> i32
          %79 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %80 = tor.addi %78 %79 on (2 to 3) {dump = "unknown_335", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
          %81 = tor.cmpi slt %80 %c0_i32 on (3 to 4) {dump = "unknown_336", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
          %82 = tor.subi %c-1_i32 %80 on (3 to 4) {dump = "unknown_337", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
          %83 = arith.select %81, %82, %80 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
          %84 = tor.cmpi slt %83 %c0_i32 on (5 to 6) {dump = "unknown_338", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
          %85 = arith.extui %84 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
          %86 = tor.addi %83 %85 on (7 to 8) {dump = "unknown_339", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
          %87 = arith.shrsi %86, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
          %88 = tor.subi %c-1_i32 %87 on (9 to 10) {dump = "unknown_340", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
          %89 = arith.select %81, %88, %87 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
          %90 = aps.read_smem %9[%89] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}], aps.mem_id = 3 : i64, endtime = 21 : i32, ref_endtime = 21 : i32, ref_starttime = 20 : i32, slot = 0 : i32, starttime = 20 : i32} : memref<12xf32>, i32 -> f32
          %91 = tor.muli %13 %c3_i32 on (1 to 2) {dump = "unknown_341", ref_endtime = 2 : i32, ref_starttime = 1 : i32} : (i32, i32) -> i32
          %92 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %93 = tor.addi %91 %92 on (2 to 3) {dump = "unknown_342", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
          %94 = tor.cmpi slt %93 %c0_i32 on (3 to 4) {dump = "unknown_343", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
          %95 = tor.subi %c-1_i32 %93 on (3 to 4) {dump = "unknown_344", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
          %96 = arith.select %94, %95, %93 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
          %97 = tor.cmpi slt %96 %c0_i32 on (5 to 6) {dump = "unknown_345", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
          %98 = arith.extui %97 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
          %99 = tor.addi %96 %98 on (7 to 8) {dump = "unknown_346", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
          %100 = arith.shrsi %99, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
          %101 = tor.subi %c-1_i32 %100 on (9 to 10) {dump = "unknown_347", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
          %102 = arith.select %94, %101, %100 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
          %103 = aps.read_smem %10[%102] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}], aps.mem_id = 2 : i64, endtime = 21 : i32, ref_endtime = 21 : i32, ref_starttime = 20 : i32, slot = 0 : i32, starttime = 20 : i32} : memref<12xf32>, i32 -> f32
          %104 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %105 = tor.addi %13 %104 on (2 to 3) {dump = "unknown_348", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
          %106 = arith.andi %105, %c1_i32 {endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, starttime = 3 : i32} : i32
          %107 = tor.cmpi eq %106 %c0_i32 on (4 to 5) {dump = "unknown_349", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
          %108 = arith.select %107, %90, %103 {endtime = 22 : i32, ref_endtime = 22 : i32, ref_starttime = 21 : i32, starttime = 21 : i32} : f32
          %109 = tor.subf %108 %77 on (43 to 46) {dump = "unknown_350", fastmath = #arith.fastmath<none>, ref_endtime = 46 : i32, ref_starttime = 43 : i32} : f32
          %110 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %111 = tor.addi %13 %110 on (2 to 3) {dump = "unknown_351", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
          %112 = arith.andi %111, %c1_i32 {endtime = 4 : i32, ref_endtime = 4 : i32, ref_starttime = 3 : i32, starttime = 3 : i32} : i32
          %113 = tor.muli %13 %c3_i32 on (1 to 2) {dump = "unknown_352", ref_endtime = 2 : i32, ref_starttime = 1 : i32} : (i32, i32) -> i32
          %114 = arith.extui %arg4 {endtime = 2 : i32, ref_endtime = 2 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i2 to i32
          %115 = tor.addi %113 %114 on (2 to 3) {dump = "unknown_353", ref_endtime = 3 : i32, ref_starttime = 2 : i32} : (i32, i32) -> i32
          %116 = tor.cmpi slt %115 %c0_i32 on (3 to 4) {dump = "unknown_354", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i1
          %117 = tor.subi %c-1_i32 %115 on (3 to 4) {dump = "unknown_355", ref_endtime = 4 : i32, ref_starttime = 3 : i32} : (i32, i32) -> i32
          %118 = arith.select %116, %117, %115 {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : i32
          %119 = tor.cmpi slt %118 %c0_i32 on (5 to 6) {dump = "unknown_356", ref_endtime = 6 : i32, ref_starttime = 5 : i32} : (i32, i32) -> i1
          %120 = arith.extui %119 {endtime = 7 : i32, ref_endtime = 7 : i32, ref_starttime = 6 : i32, starttime = 6 : i32} : i1 to i32
          %121 = tor.addi %118 %120 on (7 to 8) {dump = "unknown_357", ref_endtime = 8 : i32, ref_starttime = 7 : i32} : (i32, i32) -> i32
          %122 = arith.shrsi %121, %c1_i32 {endtime = 9 : i32, ref_endtime = 9 : i32, ref_starttime = 8 : i32, starttime = 8 : i32} : i32
          %123 = tor.subi %c-1_i32 %122 on (9 to 10) {dump = "unknown_358", ref_endtime = 10 : i32, ref_starttime = 9 : i32} : (i32, i32) -> i32
          %124 = arith.select %116, %123, %122 {endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, starttime = 10 : i32} : i32
          %125 = tor.cmpi eq %112 %c0_i32 on (4 to 5) {dump = "unknown_359", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
          aps.write_smem_if %125, %109, %9[%124] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 3 : i64}], aps.mem_id = 1 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : f32, memref<12xf32>, i32
          %126 = tor.cmpi eq %112 %c1_i32 on (4 to 5) {dump = "unknown_360", ref_endtime = 5 : i32, ref_starttime = 4 : i32} : (i32, i32) -> i1
          aps.write_smem_if %126, %109, %10[%124] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 2 : i64}], aps.mem_id = 0 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : f32, memref<12xf32>, i32
        } {II = 27 : i32, aps.pipeline_achieved_ii = 27 : i64, aps.pipeline_requested, aps.pipeline_requested_ii = -1 : i64, pipeline = 1 : i32, ref_endtime = 47 : i32, ref_starttime = 1 : i32}
      } {ref_endtime = 52 : i32, ref_starttime = 1 : i32}
      aps.copy_by @busitfc, (%9, %10)[%c0_i32], %8, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 60 : i32, estimated_latency = 15 : i64, interface_cost = 31 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 15 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 60 : i32, ref_starttime = 51 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 3.100000e+01 : f64, starttime = 51 : i32, tl_channel = 0 : i32, transaction_group = 12 : i64, transaction_id = 47 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<12xf32>, memref<12xf32>), i32, i32, i32
      %12 = tor.addi %8 %c64_i32 on (51 to 52) {dump = "unknown_361", ref_endtime = 52 : i32, ref_starttime = 51 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%9, %10)[%c16_i32], %12, %c8_i32 {after = array<i64: 47, 47>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 67 : i32, estimated_latency = 15 : i64, interface_cost = 31 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 15 : i64, memory_issue_cycle = 1 : i64, memory_latency = 14 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 15 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 67 : i32, ref_starttime = 53 : i32, required_alignment = 32 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 3.100000e+01 : f64, starttime = 53 : i32, tl_channel = 1 : i32, transaction_group = 12 : i64, transaction_id = 48 : i64, transfer_bytes = 32 : i64, transfer_offset = 64 : i64} : (memref<12xf32>, memref<12xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_9__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 5 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/sfm_batch/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_362", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = memref.get_global @au_generated_candidate_9__l0u_buffer_0 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %2 = memref.get_global @au_generated_candidate_9__l0u_buffer_1 : memref<12xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%1, %2)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 16 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 3.200000e+01 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 13 : i64, transaction_id = 49 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      %3 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_363", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %3, (%1, %2)[%c16_i32], %c8_i32 {after = array<i64: 49, 49>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 16 : i32, estimated_latency = 16 : i64, interface_cost = 32 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 16 : i64, memory_issue_cycle = 1 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 16 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 16 : i32, ref_starttime = 1 : i32, required_alignment = 32 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 3.200000e+01 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 13 : i64, transaction_id = 50 : i64, transfer_bytes = 32 : i64, transfer_offset = 64 : i64} : i32, (memref<12xf32>, memref<12xf32>), i32, i32
      %4 = aps.read_smem %1[%c0_i32] {aps.mem_id = 47 : i64, endtime = 17 : i32, ref_endtime = 17 : i32, ref_starttime = 16 : i32, slot = 0 : i32, starttime = 16 : i32} : memref<12xf32>, i32 -> f32
      %5 = tor.addf %4 %cst on (17 to 20) {dump = "unknown_364", fastmath = #arith.fastmath<none>, ref_endtime = 20 : i32, ref_starttime = 17 : i32} : f32
      aps.write_smem %5, %1[%c0_i32] {aps.mem_id = 46 : i64, endtime = 21 : i32, ref_endtime = 21 : i32, ref_starttime = 20 : i32, slot = 0 : i32, starttime = 20 : i32} : f32, memref<12xf32>, i32
      %6 = aps.read_smem %2[%c0_i32] {aps.mem_id = 45 : i64, endtime = 17 : i32, ref_endtime = 17 : i32, ref_starttime = 16 : i32, slot = 0 : i32, starttime = 16 : i32} : memref<12xf32>, i32 -> f32
      aps.write_smem %6, %2[%c0_i32] {aps.mem_id = 44 : i64, endtime = 18 : i32, ref_endtime = 18 : i32, ref_starttime = 17 : i32, slot = 0 : i32, starttime = 17 : i32} : f32, memref<12xf32>, i32
      %7 = aps.read_smem %1[%c1_i32] {aps.mem_id = 43 : i64, endtime = 22 : i32, ref_endtime = 22 : i32, ref_starttime = 21 : i32, slot = 0 : i32, starttime = 21 : i32} : memref<12xf32>, i32 -> f32
      %8 = tor.addf %7 %cst_0 on (22 to 25) {dump = "unknown_365", fastmath = #arith.fastmath<none>, ref_endtime = 25 : i32, ref_starttime = 22 : i32} : f32
      aps.write_smem %8, %1[%c1_i32] {aps.mem_id = 42 : i64, endtime = 26 : i32, ref_endtime = 26 : i32, ref_starttime = 25 : i32, slot = 0 : i32, starttime = 25 : i32} : f32, memref<12xf32>, i32
      %9 = aps.read_smem %2[%c1_i32] {aps.mem_id = 41 : i64, endtime = 19 : i32, ref_endtime = 19 : i32, ref_starttime = 18 : i32, slot = 0 : i32, starttime = 18 : i32} : memref<12xf32>, i32 -> f32
      %10 = tor.addf %9 %cst_8 on (19 to 22) {dump = "unknown_366", fastmath = #arith.fastmath<none>, ref_endtime = 22 : i32, ref_starttime = 19 : i32} : f32
      aps.write_smem %10, %2[%c1_i32] {aps.mem_id = 40 : i64, endtime = 23 : i32, ref_endtime = 23 : i32, ref_starttime = 22 : i32, slot = 0 : i32, starttime = 22 : i32} : f32, memref<12xf32>, i32
      %11 = aps.read_smem %1[%c2_i32] {aps.mem_id = 39 : i64, endtime = 27 : i32, ref_endtime = 27 : i32, ref_starttime = 26 : i32, slot = 0 : i32, starttime = 26 : i32} : memref<12xf32>, i32 -> f32
      %12 = tor.subf %11 %cst_1 on (27 to 30) {dump = "unknown_367", fastmath = #arith.fastmath<none>, ref_endtime = 30 : i32, ref_starttime = 27 : i32} : f32
      aps.write_smem %12, %1[%c2_i32] {aps.mem_id = 38 : i64, endtime = 31 : i32, ref_endtime = 31 : i32, ref_starttime = 30 : i32, slot = 0 : i32, starttime = 30 : i32} : f32, memref<12xf32>, i32
      %13 = aps.read_smem %2[%c2_i32] {aps.mem_id = 37 : i64, endtime = 24 : i32, ref_endtime = 24 : i32, ref_starttime = 23 : i32, slot = 0 : i32, starttime = 23 : i32} : memref<12xf32>, i32 -> f32
      %14 = tor.addf %13 %cst_8 on (24 to 27) {dump = "unknown_368", fastmath = #arith.fastmath<none>, ref_endtime = 27 : i32, ref_starttime = 24 : i32} : f32
      aps.write_smem %14, %2[%c2_i32] {aps.mem_id = 36 : i64, endtime = 28 : i32, ref_endtime = 28 : i32, ref_starttime = 27 : i32, slot = 0 : i32, starttime = 27 : i32} : f32, memref<12xf32>, i32
      %15 = aps.read_smem %1[%c3_i32] {aps.mem_id = 35 : i64, endtime = 32 : i32, ref_endtime = 32 : i32, ref_starttime = 31 : i32, slot = 0 : i32, starttime = 31 : i32} : memref<12xf32>, i32 -> f32
      %16 = tor.addf %15 %cst_10 on (32 to 35) {dump = "unknown_369", fastmath = #arith.fastmath<none>, ref_endtime = 35 : i32, ref_starttime = 32 : i32} : f32
      aps.write_smem %16, %1[%c3_i32] {aps.mem_id = 34 : i64, endtime = 36 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 35 : i32} : f32, memref<12xf32>, i32
      %17 = aps.read_smem %2[%c3_i32] {aps.mem_id = 33 : i64, endtime = 29 : i32, ref_endtime = 29 : i32, ref_starttime = 28 : i32, slot = 0 : i32, starttime = 28 : i32} : memref<12xf32>, i32 -> f32
      aps.write_smem %17, %2[%c3_i32] {aps.mem_id = 32 : i64, endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, slot = 0 : i32, starttime = 29 : i32} : f32, memref<12xf32>, i32
      %18 = aps.read_smem %1[%c4_i32] {aps.mem_id = 31 : i64, endtime = 37 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, slot = 0 : i32, starttime = 36 : i32} : memref<12xf32>, i32 -> f32
      %19 = tor.addf %18 %cst_2 on (37 to 40) {dump = "unknown_370", fastmath = #arith.fastmath<none>, ref_endtime = 40 : i32, ref_starttime = 37 : i32} : f32
      aps.write_smem %19, %1[%c4_i32] {aps.mem_id = 30 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : f32, memref<12xf32>, i32
      %20 = aps.read_smem %2[%c4_i32] {aps.mem_id = 29 : i64, endtime = 31 : i32, ref_endtime = 31 : i32, ref_starttime = 30 : i32, slot = 0 : i32, starttime = 30 : i32} : memref<12xf32>, i32 -> f32
      %21 = tor.addf %20 %cst_3 on (31 to 34) {dump = "unknown_371", fastmath = #arith.fastmath<none>, ref_endtime = 34 : i32, ref_starttime = 31 : i32} : f32
      aps.write_smem %21, %2[%c4_i32] {aps.mem_id = 28 : i64, endtime = 35 : i32, ref_endtime = 35 : i32, ref_starttime = 34 : i32, slot = 0 : i32, starttime = 34 : i32} : f32, memref<12xf32>, i32
      %22 = aps.read_smem %1[%c5_i32] {aps.mem_id = 27 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : memref<12xf32>, i32 -> f32
      %23 = tor.subf %22 %cst_1 on (42 to 45) {dump = "unknown_372", fastmath = #arith.fastmath<none>, ref_endtime = 45 : i32, ref_starttime = 42 : i32} : f32
      aps.write_smem %23, %1[%c5_i32] {aps.mem_id = 26 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : f32, memref<12xf32>, i32
      %24 = aps.read_smem %2[%c5_i32] {aps.mem_id = 25 : i64, endtime = 36 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 35 : i32} : memref<12xf32>, i32 -> f32
      %25 = tor.addf %24 %cst_0 on (36 to 39) {dump = "unknown_373", fastmath = #arith.fastmath<none>, ref_endtime = 39 : i32, ref_starttime = 36 : i32} : f32
      aps.write_smem %25, %2[%c5_i32] {aps.mem_id = 24 : i64, endtime = 40 : i32, ref_endtime = 40 : i32, ref_starttime = 39 : i32, slot = 0 : i32, starttime = 39 : i32} : f32, memref<12xf32>, i32
      %26 = aps.read_smem %1[%c6_i32] {aps.mem_id = 23 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : memref<12xf32>, i32 -> f32
      %27 = tor.addf %26 %cst on (47 to 50) {dump = "unknown_374", fastmath = #arith.fastmath<none>, ref_endtime = 50 : i32, ref_starttime = 47 : i32} : f32
      aps.write_smem %27, %1[%c6_i32] {aps.mem_id = 22 : i64, endtime = 51 : i32, ref_endtime = 51 : i32, ref_starttime = 50 : i32, slot = 0 : i32, starttime = 50 : i32} : f32, memref<12xf32>, i32
      %28 = aps.read_smem %2[%c6_i32] {aps.mem_id = 21 : i64, endtime = 41 : i32, ref_endtime = 41 : i32, ref_starttime = 40 : i32, slot = 0 : i32, starttime = 40 : i32} : memref<12xf32>, i32 -> f32
      aps.write_smem %28, %2[%c6_i32] {aps.mem_id = 20 : i64, endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, slot = 0 : i32, starttime = 41 : i32} : f32, memref<12xf32>, i32
      %29 = aps.read_smem %1[%c7_i32] {aps.mem_id = 19 : i64, endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, slot = 0 : i32, starttime = 51 : i32} : memref<12xf32>, i32 -> f32
      %30 = tor.addf %29 %cst_8 on (52 to 55) {dump = "unknown_375", fastmath = #arith.fastmath<none>, ref_endtime = 55 : i32, ref_starttime = 52 : i32} : f32
      aps.write_smem %30, %1[%c7_i32] {aps.mem_id = 18 : i64, endtime = 56 : i32, ref_endtime = 56 : i32, ref_starttime = 55 : i32, slot = 0 : i32, starttime = 55 : i32} : f32, memref<12xf32>, i32
      %31 = aps.read_smem %2[%c7_i32] {aps.mem_id = 17 : i64, endtime = 43 : i32, ref_endtime = 43 : i32, ref_starttime = 42 : i32, slot = 0 : i32, starttime = 42 : i32} : memref<12xf32>, i32 -> f32
      %32 = tor.addf %31 %cst_8 on (43 to 46) {dump = "unknown_376", fastmath = #arith.fastmath<none>, ref_endtime = 46 : i32, ref_starttime = 43 : i32} : f32
      aps.write_smem %32, %2[%c7_i32] {aps.mem_id = 16 : i64, endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, slot = 0 : i32, starttime = 46 : i32} : f32, memref<12xf32>, i32
      %33 = aps.read_smem %1[%c8_i32] {aps.mem_id = 15 : i64, endtime = 57 : i32, ref_endtime = 57 : i32, ref_starttime = 56 : i32, slot = 0 : i32, starttime = 56 : i32} : memref<12xf32>, i32 -> f32
      %34 = tor.subf %33 %cst_1 on (57 to 60) {dump = "unknown_377", fastmath = #arith.fastmath<none>, ref_endtime = 60 : i32, ref_starttime = 57 : i32} : f32
      aps.write_smem %34, %1[%c8_i32] {aps.mem_id = 14 : i64, endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, slot = 0 : i32, starttime = 60 : i32} : f32, memref<12xf32>, i32
      %35 = aps.read_smem %2[%c8_i32] {aps.mem_id = 13 : i64, endtime = 48 : i32, ref_endtime = 48 : i32, ref_starttime = 47 : i32, slot = 0 : i32, starttime = 47 : i32} : memref<12xf32>, i32 -> f32
      %36 = tor.addf %35 %cst_2 on (48 to 51) {dump = "unknown_378", fastmath = #arith.fastmath<none>, ref_endtime = 51 : i32, ref_starttime = 48 : i32} : f32
      aps.write_smem %36, %2[%c8_i32] {aps.mem_id = 12 : i64, endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, slot = 0 : i32, starttime = 51 : i32} : f32, memref<12xf32>, i32
      %37 = aps.read_smem %1[%c9_i32] {aps.mem_id = 11 : i64, endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, slot = 0 : i32, starttime = 61 : i32} : memref<12xf32>, i32 -> f32
      %38 = tor.addf %37 %cst_10 on (62 to 65) {dump = "unknown_379", fastmath = #arith.fastmath<none>, ref_endtime = 65 : i32, ref_starttime = 62 : i32} : f32
      aps.write_smem %38, %1[%c9_i32] {aps.mem_id = 10 : i64, endtime = 66 : i32, ref_endtime = 66 : i32, ref_starttime = 65 : i32, slot = 0 : i32, starttime = 65 : i32} : f32, memref<12xf32>, i32
      %39 = aps.read_smem %2[%c9_i32] {aps.mem_id = 9 : i64, endtime = 53 : i32, ref_endtime = 53 : i32, ref_starttime = 52 : i32, slot = 0 : i32, starttime = 52 : i32} : memref<12xf32>, i32 -> f32
      aps.write_smem %39, %2[%c9_i32] {aps.mem_id = 8 : i64, endtime = 54 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, slot = 0 : i32, starttime = 53 : i32} : f32, memref<12xf32>, i32
      %40 = aps.read_smem %1[%c10_i32] {aps.mem_id = 7 : i64, endtime = 67 : i32, ref_endtime = 67 : i32, ref_starttime = 66 : i32, slot = 0 : i32, starttime = 66 : i32} : memref<12xf32>, i32 -> f32
      %41 = tor.addf %40 %cst_0 on (67 to 70) {dump = "unknown_380", fastmath = #arith.fastmath<none>, ref_endtime = 70 : i32, ref_starttime = 67 : i32} : f32
      aps.write_smem %41, %1[%c10_i32] {aps.mem_id = 6 : i64, endtime = 71 : i32, ref_endtime = 71 : i32, ref_starttime = 70 : i32, slot = 0 : i32, starttime = 70 : i32} : f32, memref<12xf32>, i32
      %42 = aps.read_smem %2[%c10_i32] {aps.mem_id = 5 : i64, endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, slot = 0 : i32, starttime = 54 : i32} : memref<12xf32>, i32 -> f32
      %43 = tor.addf %42 %cst_3 on (55 to 58) {dump = "unknown_381", fastmath = #arith.fastmath<none>, ref_endtime = 58 : i32, ref_starttime = 55 : i32} : f32
      aps.write_smem %43, %2[%c10_i32] {aps.mem_id = 4 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<12xf32>, i32
      %44 = aps.read_smem %1[%c11_i32] {aps.mem_id = 3 : i64, endtime = 72 : i32, ref_endtime = 72 : i32, ref_starttime = 71 : i32, slot = 0 : i32, starttime = 71 : i32} : memref<12xf32>, i32 -> f32
      %45 = tor.subf %44 %cst_1 on (72 to 75) {dump = "unknown_382", fastmath = #arith.fastmath<none>, ref_endtime = 75 : i32, ref_starttime = 72 : i32} : f32
      aps.write_smem %45, %1[%c11_i32] {aps.mem_id = 2 : i64, endtime = 76 : i32, ref_endtime = 76 : i32, ref_starttime = 75 : i32, slot = 0 : i32, starttime = 75 : i32} : f32, memref<12xf32>, i32
      %46 = aps.read_smem %2[%c11_i32] {aps.mem_id = 1 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : memref<12xf32>, i32 -> f32
      %47 = tor.addf %46 %cst_8 on (60 to 63) {dump = "unknown_383", fastmath = #arith.fastmath<none>, ref_endtime = 63 : i32, ref_starttime = 60 : i32} : f32
      aps.write_smem %47, %2[%c11_i32] {aps.mem_id = 0 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<12xf32>, i32
      aps.copy_by @busitfc, (%1, %2)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 85 : i32, estimated_latency = 15 : i64, interface_cost = 31 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 15 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 85 : i32, ref_starttime = 76 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 3.100000e+01 : f64, starttime = 76 : i32, tl_channel = 0 : i32, transaction_group = 14 : i64, transaction_id = 51 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<12xf32>, memref<12xf32>), i32, i32, i32
      %48 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_384", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c16_i32], %48, %c8_i32 {after = array<i64: 51, 51>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 91 : i32, estimated_latency = 15 : i64, interface_cost = 31 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 15 : i64, memory_issue_cycle = 1 : i64, memory_latency = 14 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 15 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 91 : i32, ref_starttime = 77 : i32, required_alignment = 32 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 3.100000e+01 : f64, starttime = 77 : i32, tl_channel = 1 : i32, transaction_group = 14 : i64, transaction_id = 52 : i64, transfer_bytes = 32 : i64, transfer_offset = 64 : i64} : (memref<12xf32>, memref<12xf32>), i32, i32, i32
      tor.return
    }
  } {HoistConstCondIfOp = 1 : i32, schedule = true}
}

