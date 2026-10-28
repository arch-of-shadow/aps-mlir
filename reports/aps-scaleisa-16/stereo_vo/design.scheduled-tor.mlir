module {
  aps.memitfc @busitfc {cache_hint = "cold", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 8 : i64, min_beats = 1 : i64, min_write_beats = 2 : i64, read_latency = 5 : i64, width = 8 : i64, write_completion = 2 : i64}
  aps.memitfc @cpuitfc {cache_hint = "warm", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 1 : i64, min_beats = 1 : i64, min_write_beats = 1 : i64, read_latency = 2 : i64, width = 4 : i64, write_completion = 2 : i64}
  aps.memorymap {
    aps.mem_entry "au_generated_candidate_17__direct_buffer" : banks([@au_generated_candidate_17__direct_buffer_0_2, @au_generated_candidate_17__direct_buffer_1_2]), base(0), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_17__direct_buffer_0" : banks([@au_generated_candidate_17__direct_buffer_0_0, @au_generated_candidate_17__direct_buffer_0_1]), base(512), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_17__direct_buffer_1" : banks([@au_generated_candidate_17__direct_buffer_1_0, @au_generated_candidate_17__direct_buffer_1_1]), base(1024), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_26__direct_buffer" : banks([@au_generated_candidate_26__direct_buffer_0, @au_generated_candidate_26__direct_buffer_1]), base(1536), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_26__direct_buffer_2" : banks([@au_generated_candidate_26__direct_buffer_2]), base(2048), size(4), count(1), cyclic(0)
    aps.mem_finish
  }
  tor.design @aps_isaxes {
    %c31_i6 = arith.constant 31 : i6
    %c1_i6 = arith.constant 1 : i6
    %c0_i6 = arith.constant 0 : i6
    %cst = arith.constant 1.270000e+02 : f32
    %cst_0 = arith.constant 1.030000e+02 : f32
    %cst_1 = arith.constant 7.900000e+01 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 1.200000e-01 : f32
    %cst_3 = arith.constant 1.600000e-01 : f32
    %c1_i32 = arith.constant {dump = "unknown_0"} 1 : i32
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
    memref.global @au_generated_candidate_26__direct_buffer_2 : memref<1xi32> {scaleisa.scratchpad, var_name = "au_generated_candidate_26__direct_buffer_2"}
    memref.global @au_generated_candidate_26__direct_buffer_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_26__direct_buffer_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_17__direct_buffer_1_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_17__direct_buffer_1_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_17__direct_buffer_0_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_17__direct_buffer_0_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_17__direct_buffer_0_2 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_17__direct_buffer_1_2 : memref<48xf32> = uninitialized
    memref.global "private" @au_generated_candidate_17__direct_csr0 : memref<i32> = uninitialized {csr_address = 2049 : i32}
    tor.func @au_generated_candidate_17__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 0 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/stereo_vo/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg1 {dump = "unknown_1", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg0 {dump = "unknown_2", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_17__direct_buffer_0_2 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_17__direct_buffer_1_2 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 13 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 0 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %4 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_3", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 0, 0>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 35 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 35 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 16 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 1 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %5 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_4", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 1, 0, 1>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 46 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 46 : i32, ref_starttime = 31 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 31 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 2 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %6 = tor.addi %1 %c192_i32 on (0 to 1) {dump = "unknown_5", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%2, %3)[%c48_i32], %c16_i32 {after = array<i64: 2, 1, 2>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 61 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 61 : i32, ref_starttime = 46 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 46 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 3 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %7 = tor.addi %1 %c256_i32 on (0 to 1) {dump = "unknown_6", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%2, %3)[%c64_i32], %c16_i32 {after = array<i64: 3, 2, 3>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 76 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 76 : i32, ref_starttime = 61 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 61 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 4 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %8 = tor.addi %1 %c320_i32 on (0 to 1) {dump = "unknown_7", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %8, (%2, %3)[%c80_i32], %c16_i32 {after = array<i64: 4, 3, 4>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 91 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 91 : i32, ref_starttime = 76 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 76 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 5 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %9 = memref.get_global @au_generated_candidate_17__direct_buffer_0_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %10 = memref.get_global @au_generated_candidate_17__direct_buffer_0_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%9, %10)[%c0_i32], %c16_i32 {after = array<i64: 5, 4, 5>, after_gap = array<i64: 8, 16, 8>, endtime = 105 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 60 : i64, memory_issue_cycle = 45 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 105 : i32, ref_starttime = 90 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 90 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 6 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %11 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_8", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %11, (%9, %10)[%c16_i32], %c16_i32 {after = array<i64: 6, 5, 6>, after_gap = array<i64: 8, 16, 8>, after_segment = 0 : i64, endtime = 113 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 68 : i64, memory_issue_cycle = 53 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 113 : i32, ref_starttime = 98 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 98 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 7 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %12 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_9", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %12, (%9, %10)[%c32_i32], %c16_i32 {after = array<i64: 7, 6, 7>, after_gap = array<i64: 8, 16, 8>, after_segment = 1 : i64, endtime = 121 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 76 : i64, memory_issue_cycle = 61 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 121 : i32, ref_starttime = 106 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 106 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 8 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %13 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_10", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %13, (%9, %10)[%c48_i32], %c16_i32 {after = array<i64: 8, 7, 8>, after_gap = array<i64: 8, 16, 8>, after_segment = 2 : i64, endtime = 129 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 84 : i64, memory_issue_cycle = 69 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 129 : i32, ref_starttime = 114 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 114 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 9 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %14 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_11", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %14, (%9, %10)[%c64_i32], %c16_i32 {after = array<i64: 9, 8, 9>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 137 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 92 : i64, memory_issue_cycle = 77 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 137 : i32, ref_starttime = 122 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 122 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 10 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %15 = tor.addi %0 %c320_i32 on (0 to 1) {dump = "unknown_12", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %15, (%9, %10)[%c80_i32], %c16_i32 {after = array<i64: 10, 9, 10>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 145 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 100 : i64, memory_issue_cycle = 85 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 145 : i32, ref_starttime = 130 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 130 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 11 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %16 = aps.read_csr @au_generated_candidate_17__direct_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %17 = memref.get_global @au_generated_candidate_17__direct_buffer_1_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %18 = memref.get_global @au_generated_candidate_17__direct_buffer_1_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %16, (%17, %18)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 12 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %19 = tor.addi %16 %c64_i32 on (0 to 1) {dump = "unknown_13", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %19, (%17, %18)[%c16_i32], %c16_i32 {after = array<i64: 12, 12>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 34 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 34 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 13 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %20 = tor.addi %16 %c128_i32 on (0 to 1) {dump = "unknown_14", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %20, (%17, %18)[%c32_i32], %c16_i32 {after = array<i64: 13, 12, 13>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 45 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 45 : i32, ref_starttime = 30 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 30 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 14 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %21 = tor.addi %16 %c192_i32 on (0 to 1) {dump = "unknown_15", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %21, (%17, %18)[%c48_i32], %c16_i32 {after = array<i64: 14, 13, 14>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 60 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 60 : i32, ref_starttime = 45 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 45 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 15 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %22 = tor.addi %16 %c256_i32 on (0 to 1) {dump = "unknown_16", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %22, (%17, %18)[%c64_i32], %c16_i32 {after = array<i64: 15, 14, 15>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 75 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 75 : i32, ref_starttime = 60 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 60 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 16 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %23 = tor.addi %16 %c320_i32 on (0 to 1) {dump = "unknown_17", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %23, (%17, %18)[%c80_i32], %c16_i32 {after = array<i64: 16, 15, 16>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 90 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 90 : i32, ref_starttime = 75 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 2.920000e+02 : f64, starttime = 75 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 17 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      tor.for %arg3 = (%c0_i6 : i6) to (%c31_i6 : i6) step (%c1_i6 : i6)
      on (147 to 192){
        %29 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %30 = tor.muli %29 %c3_i32 on (147 to 148) {dump = "unknown_18", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %31 = tor.cmpi slt %30 %c0_i32 on (147 to 148) {dump = "unknown_19", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %32 = tor.subi %c-1_i32 %30 on (147 to 148) {dump = "unknown_20", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %33 = arith.select %31, %32, %30 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %34 = tor.cmpi slt %33 %c0_i32 on (147 to 148) {dump = "unknown_21", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %35 = arith.extui %34 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %36 = tor.addi %33 %35 on (147 to 148) {dump = "unknown_22", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %37 = arith.shrsi %36, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %38 = tor.subi %c-1_i32 %37 on (147 to 148) {dump = "unknown_23", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %39 = arith.select %31, %38, %37 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %40 = aps.read_smem %2[%39] {aps.mem_id = 23 : i64, endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, slot = 0 : i32, starttime = 147 : i32} : memref<48xf32>, i32 -> f32
        %41 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %42 = tor.muli %41 %c3_i32 on (147 to 148) {dump = "unknown_24", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %43 = tor.cmpi slt %42 %c0_i32 on (147 to 148) {dump = "unknown_25", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %44 = tor.subi %c-1_i32 %42 on (147 to 148) {dump = "unknown_26", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %45 = arith.select %43, %44, %42 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %46 = tor.cmpi slt %45 %c0_i32 on (147 to 148) {dump = "unknown_27", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %47 = arith.extui %46 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %48 = tor.addi %45 %47 on (147 to 148) {dump = "unknown_28", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %49 = arith.shrsi %48, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %50 = tor.subi %c-1_i32 %49 on (147 to 148) {dump = "unknown_29", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %51 = arith.select %43, %50, %49 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %52 = aps.read_smem %3[%51] {aps.mem_id = 22 : i64, endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, slot = 0 : i32, starttime = 147 : i32} : memref<48xf32>, i32 -> f32
        %53 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %54 = arith.andi %53, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %55 = tor.cmpi eq %54 %c0_i32 on (147 to 148) {dump = "unknown_30", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %56 = arith.select %55, %40, %52 {endtime = 149 : i32, ref_endtime = 149 : i32, ref_starttime = 148 : i32, starttime = 148 : i32} : f32
        %57 = tor.mulf %56 %cst_3 on (148 to 151) {dump = "unknown_31", fastmath = #arith.fastmath<none>, ref_endtime = 151 : i32, ref_starttime = 148 : i32} : f32
        %58 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %59 = tor.muli %58 %c3_i32 on (147 to 148) {dump = "unknown_32", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %60 = tor.cmpi slt %59 %c0_i32 on (147 to 148) {dump = "unknown_33", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %61 = tor.subi %c-1_i32 %59 on (147 to 148) {dump = "unknown_34", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %62 = arith.select %60, %61, %59 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %63 = tor.cmpi slt %62 %c0_i32 on (147 to 148) {dump = "unknown_35", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %64 = arith.extui %63 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %65 = tor.addi %62 %64 on (147 to 148) {dump = "unknown_36", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %66 = arith.shrsi %65, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %67 = tor.subi %c-1_i32 %66 on (147 to 148) {dump = "unknown_37", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %68 = arith.select %60, %67, %66 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %69 = aps.read_smem %9[%68] {aps.mem_id = 21 : i64, endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, slot = 0 : i32, starttime = 147 : i32} : memref<48xf32>, i32 -> f32
        %70 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %71 = tor.muli %70 %c3_i32 on (147 to 148) {dump = "unknown_38", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %72 = tor.cmpi slt %71 %c0_i32 on (147 to 148) {dump = "unknown_39", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %73 = tor.subi %c-1_i32 %71 on (147 to 148) {dump = "unknown_40", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %74 = arith.select %72, %73, %71 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %75 = tor.cmpi slt %74 %c0_i32 on (147 to 148) {dump = "unknown_41", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %76 = arith.extui %75 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %77 = tor.addi %74 %76 on (147 to 148) {dump = "unknown_42", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %78 = arith.shrsi %77, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %79 = tor.subi %c-1_i32 %78 on (147 to 148) {dump = "unknown_43", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %80 = arith.select %72, %79, %78 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %81 = aps.read_smem %10[%80] {aps.mem_id = 20 : i64, endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, slot = 0 : i32, starttime = 147 : i32} : memref<48xf32>, i32 -> f32
        %82 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %83 = arith.andi %82, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %84 = tor.cmpi eq %83 %c0_i32 on (147 to 148) {dump = "unknown_44", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %85 = arith.select %84, %69, %81 {endtime = 149 : i32, ref_endtime = 149 : i32, ref_starttime = 148 : i32, starttime = 148 : i32} : f32
        %86 = tor.divf %57 %85 on (151 to 178) {dump = "unknown_45", fastmath = #arith.fastmath<none>, ref_endtime = 178 : i32, ref_starttime = 151 : i32} : f32
        %87 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %88 = tor.muli %87 %c3_i32 on (147 to 148) {dump = "unknown_46", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %89 = tor.cmpi slt %88 %c0_i32 on (147 to 148) {dump = "unknown_47", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %90 = tor.subi %c-1_i32 %88 on (147 to 148) {dump = "unknown_48", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %91 = arith.select %89, %90, %88 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %92 = tor.cmpi slt %91 %c0_i32 on (147 to 148) {dump = "unknown_49", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %93 = arith.extui %92 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %94 = tor.addi %91 %93 on (147 to 148) {dump = "unknown_50", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %95 = arith.shrsi %94, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %96 = tor.subi %c-1_i32 %95 on (147 to 148) {dump = "unknown_51", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %97 = arith.select %89, %96, %95 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %98 = aps.read_smem %17[%97] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 17 : i64}], aps.mem_id = 19 : i64, endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, slot = 0 : i32, starttime = 147 : i32} : memref<48xf32>, i32 -> f32
        %99 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %100 = tor.muli %99 %c3_i32 on (147 to 148) {dump = "unknown_52", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %101 = tor.cmpi slt %100 %c0_i32 on (147 to 148) {dump = "unknown_53", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %102 = tor.subi %c-1_i32 %100 on (147 to 148) {dump = "unknown_54", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %103 = arith.select %101, %102, %100 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %104 = tor.cmpi slt %103 %c0_i32 on (147 to 148) {dump = "unknown_55", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %105 = arith.extui %104 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %106 = tor.addi %103 %105 on (147 to 148) {dump = "unknown_56", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %107 = arith.shrsi %106, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %108 = tor.subi %c-1_i32 %107 on (147 to 148) {dump = "unknown_57", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %109 = arith.select %101, %108, %107 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %110 = aps.read_smem %18[%109] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 16 : i64}], aps.mem_id = 18 : i64, endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, slot = 0 : i32, starttime = 147 : i32} : memref<48xf32>, i32 -> f32
        %111 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %112 = arith.andi %111, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %113 = tor.cmpi eq %112 %c0_i32 on (147 to 148) {dump = "unknown_58", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %114 = arith.select %113, %98, %110 {endtime = 149 : i32, ref_endtime = 149 : i32, ref_starttime = 148 : i32, starttime = 148 : i32} : f32
        %115 = tor.subf %114 %86 on (178 to 181) {dump = "unknown_59", fastmath = #arith.fastmath<none>, ref_endtime = 181 : i32, ref_starttime = 178 : i32} : f32
        %116 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %117 = arith.andi %116, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %118 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %119 = tor.muli %118 %c3_i32 on (147 to 148) {dump = "unknown_60", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %120 = tor.cmpi slt %119 %c0_i32 on (147 to 148) {dump = "unknown_61", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %121 = tor.subi %c-1_i32 %119 on (147 to 148) {dump = "unknown_62", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %122 = arith.select %120, %121, %119 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %123 = tor.cmpi slt %122 %c0_i32 on (147 to 148) {dump = "unknown_63", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %124 = arith.extui %123 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %125 = tor.addi %122 %124 on (147 to 148) {dump = "unknown_64", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %126 = arith.shrsi %125, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %127 = tor.subi %c-1_i32 %126 on (147 to 148) {dump = "unknown_65", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %128 = arith.select %120, %127, %126 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %129 = tor.cmpi eq %117 %c0_i32 on (147 to 148) {dump = "unknown_66", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        aps.write_smem_if %129, %115, %17[%128] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 3 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 11 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 19 : i64}], aps.mem_id = 17 : i64, endtime = 182 : i32, ref_endtime = 182 : i32, ref_starttime = 181 : i32, slot = 0 : i32, starttime = 181 : i32} : f32, memref<48xf32>, i32
        %130 = tor.cmpi eq %117 %c1_i32 on (147 to 148) {dump = "unknown_67", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        aps.write_smem_if %130, %115, %18[%128] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 2 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 10 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 18 : i64}], aps.mem_id = 16 : i64, endtime = 182 : i32, ref_endtime = 182 : i32, ref_starttime = 181 : i32, slot = 0 : i32, starttime = 181 : i32} : f32, memref<48xf32>, i32
        %131 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %132 = tor.muli %131 %c3_i32 on (147 to 148) {dump = "unknown_68", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %133 = tor.addi %132 %c1_i32 on (147 to 148) {dump = "unknown_69", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %134 = tor.cmpi slt %133 %c0_i32 on (147 to 148) {dump = "unknown_70", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %135 = tor.subi %c-2_i32 %132 on (147 to 148) {dump = "unknown_71", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %136 = arith.select %134, %135, %133 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %137 = tor.cmpi slt %136 %c0_i32 on (147 to 148) {dump = "unknown_72", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %138 = arith.extui %137 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %139 = tor.addi %136 %138 on (147 to 148) {dump = "unknown_73", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %140 = arith.shrsi %139, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %141 = tor.subi %c-1_i32 %140 on (147 to 148) {dump = "unknown_74", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %142 = arith.select %134, %141, %140 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %143 = aps.read_smem %2[%142] {aps.mem_id = 15 : i64, endtime = 149 : i32, ref_endtime = 149 : i32, ref_starttime = 148 : i32, slot = 0 : i32, starttime = 148 : i32} : memref<48xf32>, i32 -> f32
        %144 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %145 = tor.muli %144 %c3_i32 on (147 to 148) {dump = "unknown_75", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %146 = tor.addi %145 %c1_i32 on (147 to 148) {dump = "unknown_76", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %147 = tor.cmpi slt %146 %c0_i32 on (147 to 148) {dump = "unknown_77", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %148 = tor.subi %c-2_i32 %145 on (147 to 148) {dump = "unknown_78", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %149 = arith.select %147, %148, %146 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %150 = tor.cmpi slt %149 %c0_i32 on (147 to 148) {dump = "unknown_79", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %151 = arith.extui %150 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %152 = tor.addi %149 %151 on (147 to 148) {dump = "unknown_80", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %153 = arith.shrsi %152, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %154 = tor.subi %c-1_i32 %153 on (147 to 148) {dump = "unknown_81", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %155 = arith.select %147, %154, %153 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %156 = aps.read_smem %3[%155] {aps.mem_id = 14 : i64, endtime = 149 : i32, ref_endtime = 149 : i32, ref_starttime = 148 : i32, slot = 0 : i32, starttime = 148 : i32} : memref<48xf32>, i32 -> f32
        %157 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %158 = tor.addi %157 %c1_i32 on (147 to 148) {dump = "unknown_82", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %159 = tor.cmpi slt %158 %c0_i32 on (147 to 148) {dump = "unknown_83", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %160 = tor.subi %c-2_i32 %157 on (147 to 148) {dump = "unknown_84", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %161 = arith.select %159, %160, %158 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %162 = tor.cmpi slt %161 %c0_i32 on (147 to 148) {dump = "unknown_85", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %163 = arith.extui %162 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %164 = tor.addi %161 %163 on (147 to 148) {dump = "unknown_86", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %165 = arith.shrsi %164, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %166 = tor.subi %c-1_i32 %165 on (147 to 148) {dump = "unknown_87", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %167 = arith.select %159, %166, %165 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %168 = tor.muli %167 %c-2_i32 on (147 to 148) {dump = "unknown_88", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %169 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %170 = tor.addi %169 %168 on (147 to 148) {dump = "unknown_89", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %171 = tor.addi %170 %c1_i32 on (147 to 148) {dump = "unknown_90", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %172 = tor.cmpi eq %171 %c0_i32 on (147 to 148) {dump = "unknown_91", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %173 = arith.select %172, %143, %156 {endtime = 150 : i32, ref_endtime = 150 : i32, ref_starttime = 149 : i32, starttime = 149 : i32} : f32
        %174 = tor.mulf %173 %cst_3 on (149 to 152) {dump = "unknown_92", fastmath = #arith.fastmath<none>, ref_endtime = 152 : i32, ref_starttime = 149 : i32} : f32
        %175 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %176 = tor.muli %175 %c3_i32 on (147 to 148) {dump = "unknown_93", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %177 = tor.addi %176 %c1_i32 on (147 to 148) {dump = "unknown_94", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %178 = tor.cmpi slt %177 %c0_i32 on (147 to 148) {dump = "unknown_95", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %179 = tor.subi %c-2_i32 %176 on (147 to 148) {dump = "unknown_96", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %180 = arith.select %178, %179, %177 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %181 = tor.cmpi slt %180 %c0_i32 on (147 to 148) {dump = "unknown_97", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %182 = arith.extui %181 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %183 = tor.addi %180 %182 on (147 to 148) {dump = "unknown_98", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %184 = arith.shrsi %183, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %185 = tor.subi %c-1_i32 %184 on (147 to 148) {dump = "unknown_99", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %186 = arith.select %178, %185, %184 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %187 = aps.read_smem %9[%186] {aps.mem_id = 13 : i64, endtime = 149 : i32, ref_endtime = 149 : i32, ref_starttime = 148 : i32, slot = 0 : i32, starttime = 148 : i32} : memref<48xf32>, i32 -> f32
        %188 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %189 = tor.muli %188 %c3_i32 on (147 to 148) {dump = "unknown_100", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %190 = tor.addi %189 %c1_i32 on (147 to 148) {dump = "unknown_101", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %191 = tor.cmpi slt %190 %c0_i32 on (147 to 148) {dump = "unknown_102", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %192 = tor.subi %c-2_i32 %189 on (147 to 148) {dump = "unknown_103", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %193 = arith.select %191, %192, %190 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %194 = tor.cmpi slt %193 %c0_i32 on (147 to 148) {dump = "unknown_104", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %195 = arith.extui %194 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %196 = tor.addi %193 %195 on (147 to 148) {dump = "unknown_105", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %197 = arith.shrsi %196, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %198 = tor.subi %c-1_i32 %197 on (147 to 148) {dump = "unknown_106", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %199 = arith.select %191, %198, %197 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %200 = aps.read_smem %10[%199] {aps.mem_id = 12 : i64, endtime = 149 : i32, ref_endtime = 149 : i32, ref_starttime = 148 : i32, slot = 0 : i32, starttime = 148 : i32} : memref<48xf32>, i32 -> f32
        %201 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %202 = tor.addi %201 %c1_i32 on (147 to 148) {dump = "unknown_107", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %203 = tor.cmpi slt %202 %c0_i32 on (147 to 148) {dump = "unknown_108", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %204 = tor.subi %c-2_i32 %201 on (147 to 148) {dump = "unknown_109", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %205 = arith.select %203, %204, %202 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %206 = tor.cmpi slt %205 %c0_i32 on (147 to 148) {dump = "unknown_110", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %207 = arith.extui %206 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %208 = tor.addi %205 %207 on (147 to 148) {dump = "unknown_111", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %209 = arith.shrsi %208, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %210 = tor.subi %c-1_i32 %209 on (147 to 148) {dump = "unknown_112", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %211 = arith.select %203, %210, %209 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %212 = tor.muli %211 %c-2_i32 on (147 to 148) {dump = "unknown_113", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %213 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %214 = tor.addi %213 %212 on (147 to 148) {dump = "unknown_114", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %215 = tor.addi %214 %c1_i32 on (147 to 148) {dump = "unknown_115", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %216 = tor.cmpi eq %215 %c0_i32 on (147 to 148) {dump = "unknown_116", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %217 = arith.select %216, %187, %200 {endtime = 150 : i32, ref_endtime = 150 : i32, ref_starttime = 149 : i32, starttime = 149 : i32} : f32
        %218 = tor.divf %174 %217 on (152 to 179) {dump = "unknown_117", fastmath = #arith.fastmath<none>, ref_endtime = 179 : i32, ref_starttime = 152 : i32} : f32
        %219 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %220 = tor.muli %219 %c3_i32 on (147 to 148) {dump = "unknown_118", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %221 = tor.addi %220 %c1_i32 on (147 to 148) {dump = "unknown_119", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %222 = tor.cmpi slt %221 %c0_i32 on (147 to 148) {dump = "unknown_120", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %223 = tor.subi %c-2_i32 %220 on (147 to 148) {dump = "unknown_121", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %224 = arith.select %222, %223, %221 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %225 = tor.cmpi slt %224 %c0_i32 on (147 to 148) {dump = "unknown_122", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %226 = arith.extui %225 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %227 = tor.addi %224 %226 on (147 to 148) {dump = "unknown_123", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %228 = arith.shrsi %227, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %229 = tor.subi %c-1_i32 %228 on (147 to 148) {dump = "unknown_124", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %230 = arith.select %222, %229, %228 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %231 = aps.read_smem %17[%230] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 9 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 17 : i64}], aps.mem_id = 11 : i64, endtime = 183 : i32, ref_endtime = 183 : i32, ref_starttime = 182 : i32, slot = 0 : i32, starttime = 182 : i32} : memref<48xf32>, i32 -> f32
        %232 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %233 = tor.muli %232 %c3_i32 on (147 to 148) {dump = "unknown_125", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %234 = tor.addi %233 %c1_i32 on (147 to 148) {dump = "unknown_126", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %235 = tor.cmpi slt %234 %c0_i32 on (147 to 148) {dump = "unknown_127", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %236 = tor.subi %c-2_i32 %233 on (147 to 148) {dump = "unknown_128", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %237 = arith.select %235, %236, %234 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %238 = tor.cmpi slt %237 %c0_i32 on (147 to 148) {dump = "unknown_129", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %239 = arith.extui %238 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %240 = tor.addi %237 %239 on (147 to 148) {dump = "unknown_130", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %241 = arith.shrsi %240, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %242 = tor.subi %c-1_i32 %241 on (147 to 148) {dump = "unknown_131", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %243 = arith.select %235, %242, %241 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %244 = aps.read_smem %18[%243] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 0 : i32, source = 8 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 16 : i64}], aps.mem_id = 10 : i64, endtime = 183 : i32, ref_endtime = 183 : i32, ref_starttime = 182 : i32, slot = 0 : i32, starttime = 182 : i32} : memref<48xf32>, i32 -> f32
        %245 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %246 = tor.addi %245 %c1_i32 on (147 to 148) {dump = "unknown_132", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %247 = tor.cmpi slt %246 %c0_i32 on (147 to 148) {dump = "unknown_133", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %248 = tor.subi %c-2_i32 %245 on (147 to 148) {dump = "unknown_134", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %249 = arith.select %247, %248, %246 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %250 = tor.cmpi slt %249 %c0_i32 on (147 to 148) {dump = "unknown_135", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %251 = arith.extui %250 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %252 = tor.addi %249 %251 on (147 to 148) {dump = "unknown_136", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %253 = arith.shrsi %252, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %254 = tor.subi %c-1_i32 %253 on (147 to 148) {dump = "unknown_137", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %255 = arith.select %247, %254, %253 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %256 = tor.muli %255 %c-2_i32 on (147 to 148) {dump = "unknown_138", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %257 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %258 = tor.addi %257 %256 on (147 to 148) {dump = "unknown_139", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %259 = tor.addi %258 %c1_i32 on (147 to 148) {dump = "unknown_140", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %260 = tor.cmpi eq %259 %c0_i32 on (147 to 148) {dump = "unknown_141", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %261 = arith.select %260, %231, %244 {endtime = 184 : i32, ref_endtime = 184 : i32, ref_starttime = 183 : i32, starttime = 183 : i32} : f32
        %262 = tor.subf %261 %218 on (183 to 186) {dump = "unknown_142", fastmath = #arith.fastmath<none>, ref_endtime = 186 : i32, ref_starttime = 183 : i32} : f32
        %263 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %264 = tor.addi %263 %c1_i32 on (147 to 148) {dump = "unknown_143", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %265 = tor.cmpi slt %264 %c0_i32 on (147 to 148) {dump = "unknown_144", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %266 = tor.subi %c-2_i32 %263 on (147 to 148) {dump = "unknown_145", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %267 = arith.select %265, %266, %264 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %268 = tor.cmpi slt %267 %c0_i32 on (147 to 148) {dump = "unknown_146", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %269 = arith.extui %268 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %270 = tor.addi %267 %269 on (147 to 148) {dump = "unknown_147", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %271 = arith.shrsi %270, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %272 = tor.subi %c-1_i32 %271 on (147 to 148) {dump = "unknown_148", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %273 = arith.select %265, %272, %271 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %274 = tor.muli %273 %c-2_i32 on (147 to 148) {dump = "unknown_149", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %275 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %276 = tor.addi %275 %274 on (147 to 148) {dump = "unknown_150", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %277 = tor.addi %276 %c1_i32 on (147 to 148) {dump = "unknown_151", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %278 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %279 = tor.muli %278 %c3_i32 on (147 to 148) {dump = "unknown_152", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %280 = tor.addi %279 %c1_i32 on (147 to 148) {dump = "unknown_153", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %281 = tor.cmpi slt %280 %c0_i32 on (147 to 148) {dump = "unknown_154", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %282 = tor.subi %c-2_i32 %279 on (147 to 148) {dump = "unknown_155", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %283 = arith.select %281, %282, %280 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %284 = tor.cmpi slt %283 %c0_i32 on (147 to 148) {dump = "unknown_156", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %285 = arith.extui %284 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %286 = tor.addi %283 %285 on (147 to 148) {dump = "unknown_157", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %287 = arith.shrsi %286, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %288 = tor.subi %c-1_i32 %287 on (147 to 148) {dump = "unknown_158", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %289 = arith.select %281, %288, %287 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %290 = tor.cmpi eq %277 %c0_i32 on (147 to 148) {dump = "unknown_159", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        aps.write_smem_if %290, %262, %17[%289] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 3 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 11 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 19 : i64}], aps.mem_id = 9 : i64, endtime = 187 : i32, ref_endtime = 187 : i32, ref_starttime = 186 : i32, slot = 0 : i32, starttime = 186 : i32} : f32, memref<48xf32>, i32
        %291 = tor.cmpi eq %277 %c1_i32 on (147 to 148) {dump = "unknown_160", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        aps.write_smem_if %291, %262, %18[%289] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 1 : i64, kind = 2 : i32, source = 2 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 10 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 18 : i64}], aps.mem_id = 8 : i64, endtime = 187 : i32, ref_endtime = 187 : i32, ref_starttime = 186 : i32, slot = 0 : i32, starttime = 186 : i32} : f32, memref<48xf32>, i32
        %292 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %293 = tor.muli %292 %c3_i32 on (147 to 148) {dump = "unknown_161", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %294 = tor.cmpi slt %293 %c0_i32 on (147 to 148) {dump = "unknown_162", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %295 = tor.subi %c-1_i32 %293 on (147 to 148) {dump = "unknown_163", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %296 = arith.select %294, %295, %293 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %297 = tor.cmpi slt %296 %c0_i32 on (147 to 148) {dump = "unknown_164", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %298 = arith.extui %297 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %299 = tor.addi %296 %298 on (147 to 148) {dump = "unknown_165", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %300 = arith.shrsi %299, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %301 = tor.subi %c-1_i32 %300 on (147 to 148) {dump = "unknown_166", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %302 = arith.select %294, %301, %300 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %303 = tor.addi %302 %c1_i32 on (147 to 148) {dump = "unknown_167", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %304 = aps.read_smem %2[%303] {aps.mem_id = 7 : i64, endtime = 150 : i32, ref_endtime = 150 : i32, ref_starttime = 149 : i32, slot = 0 : i32, starttime = 149 : i32} : memref<48xf32>, i32 -> f32
        %305 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %306 = tor.muli %305 %c3_i32 on (147 to 148) {dump = "unknown_168", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %307 = tor.cmpi slt %306 %c0_i32 on (147 to 148) {dump = "unknown_169", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %308 = tor.subi %c-1_i32 %306 on (147 to 148) {dump = "unknown_170", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %309 = arith.select %307, %308, %306 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %310 = tor.cmpi slt %309 %c0_i32 on (147 to 148) {dump = "unknown_171", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %311 = arith.extui %310 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %312 = tor.addi %309 %311 on (147 to 148) {dump = "unknown_172", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %313 = arith.shrsi %312, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %314 = tor.subi %c-1_i32 %313 on (147 to 148) {dump = "unknown_173", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %315 = arith.select %307, %314, %313 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %316 = tor.addi %315 %c1_i32 on (147 to 148) {dump = "unknown_174", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %317 = aps.read_smem %3[%316] {aps.mem_id = 6 : i64, endtime = 150 : i32, ref_endtime = 150 : i32, ref_starttime = 149 : i32, slot = 0 : i32, starttime = 149 : i32} : memref<48xf32>, i32 -> f32
        %318 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %319 = arith.andi %318, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %320 = tor.cmpi eq %319 %c0_i32 on (147 to 148) {dump = "unknown_175", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %321 = arith.select %320, %304, %317 {endtime = 151 : i32, ref_endtime = 151 : i32, ref_starttime = 150 : i32, starttime = 150 : i32} : f32
        %322 = tor.mulf %321 %cst_2 on (150 to 153) {dump = "unknown_176", fastmath = #arith.fastmath<none>, ref_endtime = 153 : i32, ref_starttime = 150 : i32} : f32
        %323 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %324 = tor.muli %323 %c3_i32 on (147 to 148) {dump = "unknown_177", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %325 = tor.cmpi slt %324 %c0_i32 on (147 to 148) {dump = "unknown_178", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %326 = tor.subi %c-1_i32 %324 on (147 to 148) {dump = "unknown_179", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %327 = arith.select %325, %326, %324 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %328 = tor.cmpi slt %327 %c0_i32 on (147 to 148) {dump = "unknown_180", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %329 = arith.extui %328 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %330 = tor.addi %327 %329 on (147 to 148) {dump = "unknown_181", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %331 = arith.shrsi %330, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %332 = tor.subi %c-1_i32 %331 on (147 to 148) {dump = "unknown_182", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %333 = arith.select %325, %332, %331 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %334 = tor.addi %333 %c1_i32 on (147 to 148) {dump = "unknown_183", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %335 = aps.read_smem %9[%334] {aps.mem_id = 5 : i64, endtime = 150 : i32, ref_endtime = 150 : i32, ref_starttime = 149 : i32, slot = 0 : i32, starttime = 149 : i32} : memref<48xf32>, i32 -> f32
        %336 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %337 = tor.muli %336 %c3_i32 on (147 to 148) {dump = "unknown_184", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %338 = tor.cmpi slt %337 %c0_i32 on (147 to 148) {dump = "unknown_185", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %339 = tor.subi %c-1_i32 %337 on (147 to 148) {dump = "unknown_186", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %340 = arith.select %338, %339, %337 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %341 = tor.cmpi slt %340 %c0_i32 on (147 to 148) {dump = "unknown_187", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %342 = arith.extui %341 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %343 = tor.addi %340 %342 on (147 to 148) {dump = "unknown_188", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %344 = arith.shrsi %343, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %345 = tor.subi %c-1_i32 %344 on (147 to 148) {dump = "unknown_189", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %346 = arith.select %338, %345, %344 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %347 = tor.addi %346 %c1_i32 on (147 to 148) {dump = "unknown_190", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %348 = aps.read_smem %10[%347] {aps.mem_id = 4 : i64, endtime = 150 : i32, ref_endtime = 150 : i32, ref_starttime = 149 : i32, slot = 0 : i32, starttime = 149 : i32} : memref<48xf32>, i32 -> f32
        %349 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %350 = arith.andi %349, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %351 = tor.cmpi eq %350 %c0_i32 on (147 to 148) {dump = "unknown_191", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %352 = arith.select %351, %335, %348 {endtime = 151 : i32, ref_endtime = 151 : i32, ref_starttime = 150 : i32, starttime = 150 : i32} : f32
        %353 = tor.divf %322 %352 on (153 to 180) {dump = "unknown_192", fastmath = #arith.fastmath<none>, ref_endtime = 180 : i32, ref_starttime = 153 : i32} : f32
        %354 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %355 = tor.muli %354 %c3_i32 on (147 to 148) {dump = "unknown_193", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %356 = tor.cmpi slt %355 %c0_i32 on (147 to 148) {dump = "unknown_194", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %357 = tor.subi %c-1_i32 %355 on (147 to 148) {dump = "unknown_195", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %358 = arith.select %356, %357, %355 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %359 = tor.cmpi slt %358 %c0_i32 on (147 to 148) {dump = "unknown_196", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %360 = arith.extui %359 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %361 = tor.addi %358 %360 on (147 to 148) {dump = "unknown_197", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %362 = arith.shrsi %361, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %363 = tor.subi %c-1_i32 %362 on (147 to 148) {dump = "unknown_198", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %364 = arith.select %356, %363, %362 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %365 = tor.addi %364 %c1_i32 on (147 to 148) {dump = "unknown_199", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %366 = aps.read_smem %17[%365] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 1 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 9 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 17 : i64}], aps.mem_id = 3 : i64, endtime = 188 : i32, ref_endtime = 188 : i32, ref_starttime = 187 : i32, slot = 0 : i32, starttime = 187 : i32} : memref<48xf32>, i32 -> f32
        %367 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %368 = tor.muli %367 %c3_i32 on (147 to 148) {dump = "unknown_200", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %369 = tor.cmpi slt %368 %c0_i32 on (147 to 148) {dump = "unknown_201", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %370 = tor.subi %c-1_i32 %368 on (147 to 148) {dump = "unknown_202", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %371 = arith.select %369, %370, %368 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %372 = tor.cmpi slt %371 %c0_i32 on (147 to 148) {dump = "unknown_203", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %373 = arith.extui %372 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %374 = tor.addi %371 %373 on (147 to 148) {dump = "unknown_204", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %375 = arith.shrsi %374, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %376 = tor.subi %c-1_i32 %375 on (147 to 148) {dump = "unknown_205", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %377 = arith.select %369, %376, %375 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %378 = tor.addi %377 %c1_i32 on (147 to 148) {dump = "unknown_206", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %379 = aps.read_smem %18[%378] {aps.mem_deps = [{distance = 1 : i64, kind = 0 : i32, source = 0 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 8 : i64}, {distance = 0 : i64, kind = 0 : i32, source = 16 : i64}], aps.mem_id = 2 : i64, endtime = 188 : i32, ref_endtime = 188 : i32, ref_starttime = 187 : i32, slot = 0 : i32, starttime = 187 : i32} : memref<48xf32>, i32 -> f32
        %380 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %381 = arith.andi %380, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %382 = tor.cmpi eq %381 %c0_i32 on (147 to 148) {dump = "unknown_207", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %383 = arith.select %382, %366, %379 {endtime = 189 : i32, ref_endtime = 189 : i32, ref_starttime = 188 : i32, starttime = 188 : i32} : f32
        %384 = tor.subf %383 %353 on (188 to 191) {dump = "unknown_208", fastmath = #arith.fastmath<none>, ref_endtime = 191 : i32, ref_starttime = 188 : i32} : f32
        %385 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %386 = arith.andi %385, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %387 = arith.extui %arg3 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i6 to i32
        %388 = tor.muli %387 %c3_i32 on (147 to 148) {dump = "unknown_209", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %389 = tor.cmpi slt %388 %c0_i32 on (147 to 148) {dump = "unknown_210", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %390 = tor.subi %c-1_i32 %388 on (147 to 148) {dump = "unknown_211", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %391 = arith.select %389, %390, %388 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %392 = tor.cmpi slt %391 %c0_i32 on (147 to 148) {dump = "unknown_212", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        %393 = arith.extui %392 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i1 to i32
        %394 = tor.addi %391 %393 on (147 to 148) {dump = "unknown_213", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %395 = arith.shrsi %394, %c1_i32 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %396 = tor.subi %c-1_i32 %395 on (147 to 148) {dump = "unknown_214", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %397 = arith.select %389, %396, %395 {endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, starttime = 147 : i32} : i32
        %398 = tor.addi %397 %c1_i32 on (147 to 148) {dump = "unknown_215", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
        %399 = tor.cmpi eq %386 %c0_i32 on (147 to 148) {dump = "unknown_216", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        aps.write_smem_if %399, %384, %17[%398] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 1 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 3 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 11 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 19 : i64}], aps.mem_id = 1 : i64, endtime = 192 : i32, ref_endtime = 192 : i32, ref_starttime = 191 : i32, slot = 0 : i32, starttime = 191 : i32} : f32, memref<48xf32>, i32
        %400 = tor.cmpi eq %386 %c1_i32 on (147 to 148) {dump = "unknown_217", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i1
        aps.write_smem_if %400, %384, %18[%398] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 0 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 2 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 10 : i64}, {distance = 0 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 0 : i64, kind = 2 : i32, source = 18 : i64}], aps.mem_id = 0 : i64, endtime = 192 : i32, ref_endtime = 192 : i32, ref_starttime = 191 : i32, slot = 0 : i32, starttime = 191 : i32} : f32, memref<48xf32>, i32
      } {ref_endtime = 192 : i32, ref_starttime = 147 : i32}
      aps.copy_by @busitfc, (%17, %18)[%c0_i32], %16, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 201 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 201 : i32, ref_starttime = 192 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 192 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 18 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %24 = tor.addi %16 %c64_i32 on (147 to 148) {dump = "unknown_218", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%17, %18)[%c16_i32], %24, %c16_i32 {after = array<i64: 18, 18>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 211 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 211 : i32, ref_starttime = 193 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 193 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 19 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %25 = tor.addi %16 %c128_i32 on (147 to 148) {dump = "unknown_219", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%17, %18)[%c32_i32], %25, %c16_i32 {after = array<i64: 19, 18, 19>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 226 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 226 : i32, ref_starttime = 207 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 207 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 20 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %26 = tor.addi %16 %c192_i32 on (147 to 148) {dump = "unknown_220", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%17, %18)[%c48_i32], %26, %c16_i32 {after = array<i64: 20, 19, 20>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 236 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 20 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 236 : i32, ref_starttime = 217 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 217 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 21 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %27 = tor.addi %16 %c256_i32 on (147 to 148) {dump = "unknown_221", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%17, %18)[%c64_i32], %27, %c16_i32 {after = array<i64: 21, 20, 21>, after_gap = array<i64: 10, 20, 10>, after_segment = 3 : i64, endtime = 246 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 49 : i64, memory_issue_cycle = 30 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 246 : i32, ref_starttime = 227 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 227 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 22 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %28 = tor.addi %16 %c320_i32 on (147 to 148) {dump = "unknown_222", ref_endtime = 148 : i32, ref_starttime = 147 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%17, %18)[%c80_i32], %28, %c16_i32 {after = array<i64: 22, 21, 22>, after_gap = array<i64: 10, 20, 10>, after_segment = 4 : i64, endtime = 256 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 59 : i64, memory_issue_cycle = 40 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 256 : i32, ref_starttime = 237 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 237 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 23 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_26__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 1 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-suite-16/stereo_vo/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg1 {dump = "unknown_223", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg0 {dump = "unknown_224", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_26__direct_buffer_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_26__direct_buffer_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64: 24>, after_gap = array<i64: 1>, endtime = 13 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 13 : i64, memory_issue_cycle = 1 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 53 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 1.180000e+02 : f64, starttime = 1 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 25 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %4 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_225", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 25, 25>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 21 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 21 : i64, memory_issue_cycle = 2 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 53 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 21 : i32, ref_starttime = 2 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 1.180000e+02 : f64, starttime = 2 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 26 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %5 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_226", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 26, 25, 26>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 31 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 14 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 53 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 31 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 1.180000e+02 : f64, starttime = 16 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 27 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %6 = tor.addi %1 %c192_i32 on (0 to 1) {dump = "unknown_227", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%2, %3)[%c48_i32], %c16_i32 {after = array<i64: 27, 26, 27>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 39 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 37 : i64, memory_issue_cycle = 22 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 53 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 39 : i32, ref_starttime = 24 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 1.180000e+02 : f64, starttime = 24 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 28 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %7 = tor.addi %1 %c256_i32 on (0 to 1) {dump = "unknown_228", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%2, %3)[%c64_i32], %c16_i32 {after = array<i64: 28, 27, 28>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 47 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 45 : i64, memory_issue_cycle = 30 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 53 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 47 : i32, ref_starttime = 32 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 1.180000e+02 : f64, starttime = 32 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 29 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %8 = tor.addi %1 %c320_i32 on (0 to 1) {dump = "unknown_229", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %8, (%2, %3)[%c80_i32], %c16_i32 {after = array<i64: 29, 28, 29>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 55 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 53 : i64, memory_issue_cycle = 38 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 53 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 55 : i32, ref_starttime = 40 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 1.180000e+02 : f64, starttime = 40 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 30 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %9 = aps.load_by @cpuitfc, %0 {after = array<i64>, after_gap = array<i64>, endtime = 2 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 53 : i64, memory_search_hits = 0 : i64, memory_search_states = 10 : i64, ref_endtime = 2 : i32, ref_starttime = 0 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 1.180000e+02 : f64, starttime = 0 : i32, transaction_id = 24 : i64, transfer_bytes = 4 : i64} : i32 -> i32
      aps.globalstore %9, @au_generated_candidate_26__direct_buffer_2 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
      tor.for %arg3 = (%c0_i6 : i6) to (%c31_i6 : i6) step (%c1_i6 : i6)
      on (57 to 65){
        %11 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %12 = tor.muli %11 %c3_i32 on (57 to 58) {dump = "unknown_230", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %13 = tor.cmpi slt %12 %c0_i32 on (57 to 58) {dump = "unknown_231", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %14 = tor.subi %c-1_i32 %12 on (57 to 58) {dump = "unknown_232", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %15 = arith.select %13, %14, %12 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %16 = tor.cmpi slt %15 %c0_i32 on (57 to 58) {dump = "unknown_233", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %17 = arith.extui %16 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %18 = tor.addi %15 %17 on (57 to 58) {dump = "unknown_234", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %19 = arith.shrsi %18, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %20 = tor.subi %c-1_i32 %19 on (57 to 58) {dump = "unknown_235", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %21 = arith.select %13, %20, %19 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %22 = aps.read_smem %2[%21] {aps.mem_id = 5 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : memref<48xf32>, i32 -> f32
        %23 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %24 = tor.muli %23 %c3_i32 on (57 to 58) {dump = "unknown_236", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %25 = tor.cmpi slt %24 %c0_i32 on (57 to 58) {dump = "unknown_237", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %26 = tor.subi %c-1_i32 %24 on (57 to 58) {dump = "unknown_238", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %27 = arith.select %25, %26, %24 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %28 = tor.cmpi slt %27 %c0_i32 on (57 to 58) {dump = "unknown_239", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %29 = arith.extui %28 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %30 = tor.addi %27 %29 on (57 to 58) {dump = "unknown_240", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %31 = arith.shrsi %30, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %32 = tor.subi %c-1_i32 %31 on (57 to 58) {dump = "unknown_241", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %33 = arith.select %25, %32, %31 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %34 = aps.read_smem %3[%33] {aps.mem_id = 4 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : memref<48xf32>, i32 -> f32
        %35 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %36 = arith.andi %35, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %37 = tor.cmpi eq %36 %c0_i32 on (57 to 58) {dump = "unknown_242", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %38 = arith.select %37, %22, %34 {endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, starttime = 58 : i32} : f32
        %39 = tor.mulf %38 %cst_1 on (58 to 61) {dump = "unknown_243", fastmath = #arith.fastmath<none>, ref_endtime = 61 : i32, ref_starttime = 58 : i32} : f32
        %40 = arith.fptosi %39 {endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, starttime = 61 : i32} : f32 to i32
        %41 = aps.globalload @au_generated_candidate_26__direct_buffer_2 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %42 = tor.addi %41 %40 on (62 to 63) {dump = "unknown_244", ref_endtime = 63 : i32, ref_starttime = 62 : i32} : (i32, i32) -> i32
        %43 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %44 = tor.muli %43 %c3_i32 on (57 to 58) {dump = "unknown_245", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %45 = tor.addi %44 %c1_i32 on (57 to 58) {dump = "unknown_246", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %46 = tor.cmpi slt %45 %c0_i32 on (57 to 58) {dump = "unknown_247", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %47 = tor.subi %c-2_i32 %44 on (57 to 58) {dump = "unknown_248", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %48 = arith.select %46, %47, %45 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %49 = tor.cmpi slt %48 %c0_i32 on (57 to 58) {dump = "unknown_249", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %50 = arith.extui %49 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %51 = tor.addi %48 %50 on (57 to 58) {dump = "unknown_250", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %52 = arith.shrsi %51, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %53 = tor.subi %c-1_i32 %52 on (57 to 58) {dump = "unknown_251", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %54 = arith.select %46, %53, %52 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %55 = aps.read_smem %2[%54] {aps.mem_id = 3 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : memref<48xf32>, i32 -> f32
        %56 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %57 = tor.muli %56 %c3_i32 on (57 to 58) {dump = "unknown_252", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %58 = tor.addi %57 %c1_i32 on (57 to 58) {dump = "unknown_253", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %59 = tor.cmpi slt %58 %c0_i32 on (57 to 58) {dump = "unknown_254", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %60 = tor.subi %c-2_i32 %57 on (57 to 58) {dump = "unknown_255", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %61 = arith.select %59, %60, %58 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %62 = tor.cmpi slt %61 %c0_i32 on (57 to 58) {dump = "unknown_256", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %63 = arith.extui %62 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %64 = tor.addi %61 %63 on (57 to 58) {dump = "unknown_257", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %65 = arith.shrsi %64, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %66 = tor.subi %c-1_i32 %65 on (57 to 58) {dump = "unknown_258", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %67 = arith.select %59, %66, %65 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %68 = aps.read_smem %3[%67] {aps.mem_id = 2 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : memref<48xf32>, i32 -> f32
        %69 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %70 = tor.addi %69 %c1_i32 on (57 to 58) {dump = "unknown_259", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %71 = tor.cmpi slt %70 %c0_i32 on (57 to 58) {dump = "unknown_260", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %72 = tor.subi %c-2_i32 %69 on (57 to 58) {dump = "unknown_261", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %73 = arith.select %71, %72, %70 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %74 = tor.cmpi slt %73 %c0_i32 on (57 to 58) {dump = "unknown_262", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %75 = arith.extui %74 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %76 = tor.addi %73 %75 on (57 to 58) {dump = "unknown_263", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %77 = arith.shrsi %76, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %78 = tor.subi %c-1_i32 %77 on (57 to 58) {dump = "unknown_264", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %79 = arith.select %71, %78, %77 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %80 = tor.muli %79 %c-2_i32 on (57 to 58) {dump = "unknown_265", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %81 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %82 = tor.addi %81 %80 on (57 to 58) {dump = "unknown_266", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %83 = tor.addi %82 %c1_i32 on (57 to 58) {dump = "unknown_267", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %84 = tor.cmpi eq %83 %c0_i32 on (57 to 58) {dump = "unknown_268", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %85 = arith.select %84, %55, %68 {endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, starttime = 59 : i32} : f32
        %86 = tor.mulf %85 %cst_0 on (59 to 62) {dump = "unknown_269", fastmath = #arith.fastmath<none>, ref_endtime = 62 : i32, ref_starttime = 59 : i32} : f32
        %87 = arith.fptosi %86 {endtime = 63 : i32, ref_endtime = 63 : i32, ref_starttime = 62 : i32, starttime = 62 : i32} : f32 to i32
        %88 = tor.addi %42 %87 on (63 to 64) {dump = "unknown_270", ref_endtime = 64 : i32, ref_starttime = 63 : i32} : (i32, i32) -> i32
        %89 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %90 = tor.muli %89 %c3_i32 on (57 to 58) {dump = "unknown_271", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %91 = tor.cmpi slt %90 %c0_i32 on (57 to 58) {dump = "unknown_272", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %92 = tor.subi %c-1_i32 %90 on (57 to 58) {dump = "unknown_273", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %93 = arith.select %91, %92, %90 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %94 = tor.cmpi slt %93 %c0_i32 on (57 to 58) {dump = "unknown_274", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %95 = arith.extui %94 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %96 = tor.addi %93 %95 on (57 to 58) {dump = "unknown_275", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %97 = arith.shrsi %96, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %98 = tor.subi %c-1_i32 %97 on (57 to 58) {dump = "unknown_276", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %99 = arith.select %91, %98, %97 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %100 = tor.addi %99 %c1_i32 on (57 to 58) {dump = "unknown_277", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %101 = aps.read_smem %2[%100] {aps.mem_id = 1 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : memref<48xf32>, i32 -> f32
        %102 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %103 = tor.muli %102 %c3_i32 on (57 to 58) {dump = "unknown_278", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %104 = tor.cmpi slt %103 %c0_i32 on (57 to 58) {dump = "unknown_279", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %105 = tor.subi %c-1_i32 %103 on (57 to 58) {dump = "unknown_280", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %106 = arith.select %104, %105, %103 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %107 = tor.cmpi slt %106 %c0_i32 on (57 to 58) {dump = "unknown_281", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %108 = arith.extui %107 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i1 to i32
        %109 = tor.addi %106 %108 on (57 to 58) {dump = "unknown_282", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %110 = arith.shrsi %109, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %111 = tor.subi %c-1_i32 %110 on (57 to 58) {dump = "unknown_283", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %112 = arith.select %104, %111, %110 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %113 = tor.addi %112 %c1_i32 on (57 to 58) {dump = "unknown_284", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i32
        %114 = aps.read_smem %3[%113] {aps.mem_id = 0 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : memref<48xf32>, i32 -> f32
        %115 = arith.extui %arg3 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i6 to i32
        %116 = arith.andi %115, %c1_i32 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
        %117 = tor.cmpi eq %116 %c0_i32 on (57 to 58) {dump = "unknown_285", ref_endtime = 58 : i32, ref_starttime = 57 : i32} : (i32, i32) -> i1
        %118 = arith.select %117, %101, %114 {endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, starttime = 60 : i32} : f32
        %119 = tor.mulf %118 %cst on (60 to 63) {dump = "unknown_286", fastmath = #arith.fastmath<none>, ref_endtime = 63 : i32, ref_starttime = 60 : i32} : f32
        %120 = arith.fptosi %119 {endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, starttime = 63 : i32} : f32 to i32
        %121 = tor.addi %88 %120 on (64 to 65) {dump = "unknown_287", ref_endtime = 65 : i32, ref_starttime = 64 : i32} : (i32, i32) -> i32
        aps.globalstore %121, @au_generated_candidate_26__direct_buffer_2 {endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, starttime = 64 : i32} : i32
      } {ref_endtime = 65 : i32, ref_starttime = 57 : i32}
      %10 = aps.globalload @au_generated_candidate_26__direct_buffer_2 {endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, starttime = 57 : i32} : i32
      aps.store_by @cpuitfc, %10, %0 {after = array<i64>, after_gap = array<i64>, endtime = 59 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 59 : i32, ref_starttime = 57 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 1.800000e+01 : f64, starttime = 57 : i32, transaction_id = 31 : i64, transfer_bytes = 4 : i64} : i32, i32
      tor.return
    }
  } {HoistConstCondIfOp = 1 : i32, schedule = true}
}

