module {
  aps.memitfc @busitfc {cache_hint = "cold", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 8 : i64, min_beats = 1 : i64, min_write_beats = 2 : i64, read_latency = 5 : i64, width = 8 : i64, write_completion = 2 : i64}
  aps.memitfc @cpuitfc {cache_hint = "warm", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 1 : i64, min_beats = 1 : i64, min_write_beats = 1 : i64, read_latency = 2 : i64, width = 4 : i64, write_completion = 2 : i64}
  aps.memorymap {
    aps.mem_entry "au_generated_candidate_11__direct_buffer" : banks([@au_generated_candidate_11__direct_buffer_0_2, @au_generated_candidate_11__direct_buffer_1_0]), base(0), size(128), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_11__direct_buffer_0" : banks([@au_generated_candidate_11__direct_buffer_0_0, @au_generated_candidate_11__direct_buffer_0_1]), base(128), size(128), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_11__direct_buffer_1" : banks([@au_generated_candidate_11__direct_buffer_1]), base(256), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_24__l0u_buffer" : banks([@au_generated_candidate_24__l0u_buffer_0, @au_generated_candidate_24__l0u_buffer_1]), base(260), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_25__l0u_buffer" : banks([@au_generated_candidate_25__l0u_buffer_0, @au_generated_candidate_25__l0u_buffer_1]), base(1024), size(48), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_2__direct_buffer" : banks([@au_generated_candidate_2__direct_buffer_0, @au_generated_candidate_2__direct_buffer_1]), base(1088), size(128), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_2__direct_buffer_2" : banks([@au_generated_candidate_2__direct_buffer_2_0, @au_generated_candidate_2__direct_buffer_2_1]), base(1280), size(128), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_2__direct_buffer_3" : banks([@au_generated_candidate_2__direct_buffer_3_0, @au_generated_candidate_2__direct_buffer_3_1]), base(1408), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_2__direct_buffer_4" : banks([@au_generated_candidate_2__direct_buffer_4_0, @au_generated_candidate_2__direct_buffer_4_1]), base(2048), size(48), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_2__direct_buffer_5" : banks([@au_generated_candidate_2__direct_buffer_5_0, @au_generated_candidate_2__direct_buffer_5_1]), base(2112), size(32), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_2__direct_buffer_6" : banks([@au_generated_candidate_2__direct_buffer_6_0, @au_generated_candidate_2__direct_buffer_6_1]), base(2144), size(32), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_2__direct_local" : banks([@au_generated_candidate_2__direct_local]), base(2176), size(4), count(1), cyclic(0)
    aps.mem_finish
  }
  tor.design @aps_isaxes {
    %c31_i6 = arith.constant 31 : i6
    %c1_i6 = arith.constant 1 : i6
    %c0_i6 = arith.constant 0 : i6
    %c7_i4 = arith.constant 7 : i4
    %c1_i4 = arith.constant 1 : i4
    %c0_i4 = arith.constant 0 : i4
    %c3_i3 = arith.constant 3 : i3
    %c1_i3 = arith.constant 1 : i3
    %c0_i3 = arith.constant 0 : i3
    %cst = arith.constant 0.278999984 : f32
    %cst_0 = arith.constant 0.459000021 : f32
    %cst_1 = arith.constant 1.296000e+00 : f32
    %cst_2 = arith.constant 1.70499992 : f32
    %cst_3 = arith.constant 2.13300014 : f32
    %cst_4 = arith.constant 0.567000031 : f32
    %cst_5 = arith.constant 0.0929999947 : f32
    %cst_6 = arith.constant 1.450000e-01 : f32
    %cst_7 = arith.constant 0.18900001 : f32
    %cst_8 = arith.constant 4.340000e-01 : f32
    %cst_9 = arith.constant 6.380000e-01 : f32
    %cst_10 = arith.constant 7.020000e-01 : f32
    %cst_11 = arith.constant 0.774999976 : f32
    %cst_12 = arith.constant 1.13099992 : f32
    %cst_13 = arith.constant 1.215000e+00 : f32
    %cst_14 = arith.constant 1.11599994 : f32
    %cst_15 = arith.constant 1.624000e+00 : f32
    %cst_16 = arith.constant 1.728000e+00 : f32
    %cst_17 = arith.constant 1.457000e+00 : f32
    %cst_18 = arith.constant 2.11699986 : f32
    %cst_19 = arith.constant 2.24100018 : f32
    %cst_20 = arith.constant 1.798000e+00 : f32
    %cst_21 = arith.constant 2.610000e+00 : f32
    %cst_22 = arith.constant 3.510000e-01 : f32
    %cst_23 = arith.constant 2.139000e+00 : f32
    %cst_24 = arith.constant 2.900000e-01 : f32
    %cst_25 = arith.constant 8.640000e-01 : f32
    %cst_26 = arith.constant 2.480000e+00 : f32
    %cst_27 = arith.constant 0.782999992 : f32
    %cst_28 = arith.constant 1.37700009 : f32
    %cst_29 = arith.constant 2.82099986 : f32
    %cst_30 = arith.constant 1.276000e+00 : f32
    %cst_31 = arith.constant 1.8900001 : f32
    %cst_32 = arith.constant 3.100000e-02 : f32
    %cst_33 = arith.constant 1.76899993 : f32
    %cst_34 = arith.constant 0.000000e+00 : f32
    %cst_35 = arith.constant 0.371999979 : f32
    %cst_36 = arith.constant 2.26199985 : f32
    %cst_37 = arith.constant 5.130000e-01 : f32
    %cst_38 = arith.constant 0.712999999 : f32
    %cst_39 = arith.constant 2.75499988 : f32
    %cst_40 = arith.constant 1.026000e+00 : f32
    %cst_41 = arith.constant 1.054000e+00 : f32
    %cst_42 = arith.constant 4.350000e-01 : f32
    %cst_43 = arith.constant 1.539000e+00 : f32
    %cst_44 = arith.constant 1.395000e+00 : f32
    %cst_45 = arith.constant 0.927999973 : f32
    %cst_46 = arith.constant 2.052000e+00 : f32
    %cst_47 = arith.constant 1.736000e+00 : f32
    %cst_48 = arith.constant 1.421000e+00 : f32
    %cst_49 = arith.constant 1.620000e-01 : f32
    %cst_50 = arith.constant 2.077000e+00 : f32
    %cst_51 = arith.constant 1.91399992 : f32
    %cst_52 = arith.constant 6.750000e-01 : f32
    %cst_53 = arith.constant 2.418000e+00 : f32
    %cst_54 = arith.constant 2.40699983 : f32
    %cst_55 = arith.constant 1.18800008 : f32
    %cst_56 = arith.constant 2.759000e+00 : f32
    %cst_57 = arith.constant 8.700000e-02 : f32
    %cst_58 = arith.constant 1.70100009 : f32
    %cst_59 = arith.constant 3.100000e+00 : f32
    %cst_60 = arith.constant 5.800000e-01 : f32
    %cst_61 = arith.constant 2.214000e+00 : f32
    %cst_62 = arith.constant 3.100000e-01 : f32
    %cst_63 = arith.constant 1.073000e+00 : f32
    %cst_64 = arith.constant 3.240000e-01 : f32
    %cst_65 = arith.constant 0.650999963 : f32
    %cst_66 = arith.constant 1.566000e+00 : f32
    %cst_67 = arith.constant 8.370000e-01 : f32
    %cst_68 = arith.constant 0.991999983 : f32
    %cst_69 = arith.constant 2.059000e+00 : f32
    %cst_70 = arith.constant 1.350000e+00 : f32
    %cst_71 = arith.constant 1.333000e+00 : f32
    %cst_72 = arith.constant 2.552000e+00 : f32
    %cst_73 = arith.constant 1.863000e+00 : f32
    %cst_74 = arith.constant 1.674000e+00 : f32
    %cst_75 = arith.constant 2.320000e-01 : f32
    %cst_76 = arith.constant 2.37600017 : f32
    %cst_77 = arith.constant 2.01499987 : f32
    %cst_78 = arith.constant 0.724999964 : f32
    %cst_79 = arith.constant 4.860000e-01 : f32
    %cst_80 = arith.constant 2.356000e+00 : f32
    %cst_81 = arith.constant 1.21799994 : f32
    %cst_82 = arith.constant 9.990000e-01 : f32
    %cst_83 = arith.constant 2.697000e+00 : f32
    %cst_84 = arith.constant 1.711000e+00 : f32
    %cst_85 = arith.constant 1.51200008 : f32
    %cst_86 = arith.constant 3.03799987 : f32
    %cst_87 = arith.constant 2.204000e+00 : f32
    %cst_88 = arith.constant 2.025000e+00 : f32
    %cst_89 = arith.constant 2.480000e-01 : f32
    %cst_90 = arith.constant 1.350000e-01 : f32
    %cst_91 = arith.constant 5.890000e-01 : f32
    %cst_92 = arith.constant 0.376999974 : f32
    %cst_93 = arith.constant 6.480000e-01 : f32
    %cst_94 = arith.constant 9.300000e-01 : f32
    %cst_95 = arith.constant 8.700000e-01 : f32
    %cst_96 = arith.constant 1.161000e+00 : f32
    %cst_97 = arith.constant 1.271000e+00 : f32
    %cst_98 = arith.constant 1.36299992 : f32
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
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %false = arith.constant false
    %cst_99 = arith.constant 1.000000e+30 : f32
    %c128_i32 = arith.constant 128 : i32
    %c192_i32 = arith.constant 192 : i32
    %c256_i32 = arith.constant 256 : i32
    %c320_i32 = arith.constant 320 : i32
    %c4_i32 = arith.constant 4 : i32
    %c48_i32 = arith.constant 48 : i32
    %c80_i32 = arith.constant 80 : i32
    %c3_i32 = arith.constant 3 : i32
    %c-2_i32 = arith.constant -2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_100 = arith.constant 2.560000e+02 : f32
    %c-1_i32 = arith.constant {dump = "unknown_1"} -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2_i32 = arith.constant {dump = "unknown_2"} 2 : i32
    %c1_i32 = arith.constant 1 : i32
    memref.global @au_generated_candidate_25__l0u_buffer_0 : memref<6xf32> = uninitialized
    memref.global @au_generated_candidate_25__l0u_buffer_1 : memref<6xf32> = uninitialized
    memref.global @au_generated_candidate_24__l0u_buffer_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_24__l0u_buffer_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_6_0 : memref<4xi32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_6_1 : memref<4xi32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_5_0 : memref<4xf32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_5_1 : memref<4xf32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_4_0 : memref<6xf32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_4_1 : memref<6xf32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_3_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_3_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_2_0 : memref<16xi32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_2_1 : memref<16xi32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_0 : memref<16xf32> = uninitialized
    memref.global @au_generated_candidate_2__direct_buffer_1 : memref<16xf32> = uninitialized
    memref.global @au_generated_candidate_2__direct_local : memref<i32> {scaleisa.scratchpad, var_name = "au_generated_candidate_2__direct_local"}
    memref.global @au_generated_candidate_11__direct_buffer_1 : memref<1xi32> {scaleisa.scratchpad, var_name = "au_generated_candidate_11__direct_buffer_1"}
    memref.global @au_generated_candidate_11__direct_buffer_0_0 : memref<16xf32> = uninitialized
    memref.global @au_generated_candidate_11__direct_buffer_0_1 : memref<16xf32> = uninitialized
    memref.global @au_generated_candidate_11__direct_buffer_0_2 : memref<16xi32> = uninitialized
    memref.global @au_generated_candidate_11__direct_buffer_1_0 : memref<16xi32> = uninitialized
    memref.global "private" @au_generated_candidate_11__direct_csr0 : memref<i32> = uninitialized {csr_address = 2049 : i32}
    memref.global "private" @au_generated_candidate_2__direct_csr0 : memref<i32> = uninitialized {csr_address = 2050 : i32}
    memref.global "private" @au_generated_candidate_2__direct_csr1 : memref<i32> = uninitialized {csr_address = 2051 : i32}
    memref.global "private" @au_generated_candidate_2__direct_csr2 : memref<i32> = uninitialized {csr_address = 2052 : i32}
    memref.global "private" @au_generated_candidate_2__direct_csr3 : memref<i32> = uninitialized {csr_address = 2053 : i32}
    tor.func @au_generated_candidate_11__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 0 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-knn-32/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg1 {dump = "unknown_4", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg0 {dump = "unknown_5", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_11__direct_buffer_0_2 : memref<16xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_11__direct_buffer_1_0 : memref<16xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 13 : i32, estimated_latency = 20 : i64, interface_cost = 36 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 8.600000e+01 : f64, starttime = 1 : i32, tl_channel = 0 : i32, transaction_group = 0 : i64, transaction_id = 0 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<16xi32>, memref<16xi32>), i32, i32
      %4 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_6", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 0, 0>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 21 : i32, estimated_latency = 20 : i64, interface_cost = 36 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 21 : i32, ref_starttime = 2 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 8.600000e+01 : f64, starttime = 2 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 1 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<16xi32>, memref<16xi32>), i32, i32
      %5 = memref.get_global @au_generated_candidate_11__direct_buffer_0_0 : memref<16xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %6 = memref.get_global @au_generated_candidate_11__direct_buffer_0_1 : memref<16xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%5, %6)[%c0_i32], %c16_i32 {after = array<i64: 1, 0, 1>, after_gap = array<i64: 12, 13, 12>, endtime = 31 : i32, estimated_latency = 20 : i64, interface_cost = 36 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 31 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 8.600000e+01 : f64, starttime = 16 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 2 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<16xf32>, memref<16xf32>), i32, i32
      %7 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_7", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%5, %6)[%c16_i32], %c16_i32 {after = array<i64: 2, 1, 2>, after_gap = array<i64: 8, 20, 8>, after_segment = 0 : i64, endtime = 39 : i32, estimated_latency = 20 : i64, interface_cost = 36 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 39 : i32, ref_starttime = 24 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 8.600000e+01 : f64, starttime = 24 : i32, tl_channel = 1 : i32, transaction_group = 1 : i64, transaction_id = 3 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<16xf32>, memref<16xf32>), i32, i32
      %8 = aps.read_csr @au_generated_candidate_11__direct_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %9 = aps.load_by @cpuitfc, %8 {after = array<i64>, after_gap = array<i64>, direction = 0 : i32, endtime = 2 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 2 : i32, ref_starttime = 0 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 8.600000e+01 : f64, starttime = 0 : i32, transaction_id = 4 : i64, transfer_bytes = 4 : i64} : i32 -> i32
      aps.globalstore %9, @au_generated_candidate_11__direct_buffer_1 {endtime = 3 : i32, ref_endtime = 3 : i32, ref_starttime = 2 : i32, starttime = 2 : i32} : i32
      tor.for %arg3 = (%c0_i3 : i3) to (%c3_i3 : i3) step (%c1_i3 : i3)
      on (41 to 52){
        %11 = arith.extui %arg3 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i3 to i32
        tor.for %arg4 = (%c0_i4 : i4) to (%c7_i4 : i4) step (%c1_i4 : i4)
        on (43 to 52){
          %12 = arith.shli %11, %c2_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
          %13 = arith.extui %arg4 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i4 to i32
          %14 = tor.cmpi slt %13 %c0_i32 on (43 to 44) {dump = "unknown_8", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i1
          %15 = arith.extui %arg4 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i4 to i32
          %16 = tor.subi %c-1_i32 %15 on (43 to 44) {dump = "unknown_9", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i32
          %17 = arith.extui %arg4 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i4 to i32
          %18 = arith.select %14, %16, %17 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
          %19 = tor.cmpi slt %18 %c0_i32 on (43 to 44) {dump = "unknown_10", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i1
          %20 = arith.extui %19 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i1 to i32
          %21 = tor.addi %18 %20 on (43 to 44) {dump = "unknown_11", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i32
          %22 = arith.shrsi %21, %c1_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
          %23 = tor.subi %c-1_i32 %22 on (43 to 44) {dump = "unknown_12", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i32
          %24 = arith.select %14, %23, %22 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
          %25 = tor.addi %12 %24 on (43 to 44) {dump = "unknown_13", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i32
          %26 = aps.read_smem %2[%25] {aps.mem_id = 5 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<16xi32>, i32 -> i32
          %27 = arith.shli %11, %c2_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
          %28 = arith.extui %arg4 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i4 to i32
          %29 = tor.cmpi slt %28 %c0_i32 on (43 to 44) {dump = "unknown_14", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i1
          %30 = arith.extui %arg4 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i4 to i32
          %31 = tor.subi %c-1_i32 %30 on (43 to 44) {dump = "unknown_15", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i32
          %32 = arith.extui %arg4 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i4 to i32
          %33 = arith.select %29, %31, %32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
          %34 = tor.cmpi slt %33 %c0_i32 on (43 to 44) {dump = "unknown_16", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i1
          %35 = arith.extui %34 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i1 to i32
          %36 = tor.addi %33 %35 on (43 to 44) {dump = "unknown_17", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i32
          %37 = arith.shrsi %36, %c1_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
          %38 = tor.subi %c-1_i32 %37 on (43 to 44) {dump = "unknown_18", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i32
          %39 = arith.select %29, %38, %37 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
          %40 = tor.addi %27 %39 on (43 to 44) {dump = "unknown_19", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i32
          %41 = aps.read_smem %3[%40] {aps.mem_id = 4 : i64, endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, slot = 0 : i32, starttime = 43 : i32} : memref<16xi32>, i32 -> i32
          %42 = arith.extui %arg4 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i4 to i32
          %43 = arith.andi %42, %c1_i32 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i32
          %44 = tor.cmpi eq %43 %c0_i32 on (43 to 44) {dump = "unknown_20", ref_endtime = 44 : i32, ref_starttime = 43 : i32} : (i32, i32) -> i1
          %45 = arith.select %44, %26, %41 {endtime = 45 : i32, ref_endtime = 45 : i32, ref_starttime = 44 : i32, starttime = 44 : i32} : i32
          %46 = tor.cmpi sge %45 %c0_i32 on (44 to 45) {dump = "unknown_21", ref_endtime = 45 : i32, ref_starttime = 44 : i32} : (i32, i32) -> i1
          %47 = arith.extui %arg4 {endtime = 44 : i32, ref_endtime = 44 : i32, ref_starttime = 43 : i32, starttime = 43 : i32} : i4 to i32
          %48 = tor.if %46 on (45 to 51) -> (i32) then {
            %51 = arith.shli %11, %c2_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %52 = tor.cmpi slt %47 %c0_i32 on (45 to 46) {dump = "unknown_23", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %53 = tor.subi %c-1_i32 %47 on (45 to 46) {dump = "unknown_24", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %54 = arith.select %52, %53, %47 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %55 = tor.cmpi slt %54 %c0_i32 on (45 to 46) {dump = "unknown_25", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %56 = arith.extui %55 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
            %57 = tor.addi %54 %56 on (45 to 46) {dump = "unknown_26", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %58 = arith.shrsi %57, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %59 = tor.subi %c-1_i32 %58 on (45 to 46) {dump = "unknown_27", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %60 = arith.select %52, %59, %58 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %61 = tor.addi %51 %60 on (45 to 46) {dump = "unknown_28", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %62 = aps.read_smem %2[%61] {aps.mem_id = 3 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<16xi32>, i32 -> i32
            %63 = arith.shli %11, %c2_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %64 = tor.cmpi slt %47 %c0_i32 on (45 to 46) {dump = "unknown_29", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %65 = tor.subi %c-1_i32 %47 on (45 to 46) {dump = "unknown_30", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %66 = arith.select %64, %65, %47 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %67 = tor.cmpi slt %66 %c0_i32 on (45 to 46) {dump = "unknown_31", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %68 = arith.extui %67 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
            %69 = tor.addi %66 %68 on (45 to 46) {dump = "unknown_32", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %70 = arith.shrsi %69, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %71 = tor.subi %c-1_i32 %70 on (45 to 46) {dump = "unknown_33", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %72 = arith.select %64, %71, %70 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %73 = tor.addi %63 %72 on (45 to 46) {dump = "unknown_34", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %74 = aps.read_smem %3[%73] {aps.mem_id = 2 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<16xi32>, i32 -> i32
            %75 = arith.andi %47, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %76 = tor.cmpi eq %75 %c0_i32 on (45 to 46) {dump = "unknown_35", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %77 = arith.select %76, %62, %74 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : i32
            %78 = arith.shli %11, %c2_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %79 = tor.cmpi slt %47 %c0_i32 on (45 to 46) {dump = "unknown_36", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %80 = tor.subi %c-1_i32 %47 on (45 to 46) {dump = "unknown_37", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %81 = arith.select %79, %80, %47 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %82 = tor.cmpi slt %81 %c0_i32 on (45 to 46) {dump = "unknown_38", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %83 = arith.extui %82 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
            %84 = tor.addi %81 %83 on (45 to 46) {dump = "unknown_39", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %85 = arith.shrsi %84, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %86 = tor.subi %c-1_i32 %85 on (45 to 46) {dump = "unknown_40", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %87 = arith.select %79, %86, %85 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %88 = tor.addi %78 %87 on (45 to 46) {dump = "unknown_41", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %89 = aps.read_smem %5[%88] {aps.mem_id = 1 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<16xf32>, i32 -> f32
            %90 = arith.shli %11, %c2_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %91 = tor.cmpi slt %47 %c0_i32 on (45 to 46) {dump = "unknown_42", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %92 = tor.subi %c-1_i32 %47 on (45 to 46) {dump = "unknown_43", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %93 = arith.select %91, %92, %47 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %94 = tor.cmpi slt %93 %c0_i32 on (45 to 46) {dump = "unknown_44", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %95 = arith.extui %94 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i1 to i32
            %96 = tor.addi %93 %95 on (45 to 46) {dump = "unknown_45", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %97 = arith.shrsi %96, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %98 = tor.subi %c-1_i32 %97 on (45 to 46) {dump = "unknown_46", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %99 = arith.select %91, %98, %97 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %100 = tor.addi %90 %99 on (45 to 46) {dump = "unknown_47", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i32
            %101 = aps.read_smem %6[%100] {aps.mem_id = 0 : i64, endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, slot = 0 : i32, starttime = 45 : i32} : memref<16xf32>, i32 -> f32
            %102 = arith.andi %47, %c1_i32 {endtime = 46 : i32, ref_endtime = 46 : i32, ref_starttime = 45 : i32, starttime = 45 : i32} : i32
            %103 = tor.cmpi eq %102 %c0_i32 on (45 to 46) {dump = "unknown_48", ref_endtime = 46 : i32, ref_starttime = 45 : i32} : (i32, i32) -> i1
            %104 = arith.select %103, %89, %101 {endtime = 47 : i32, ref_endtime = 47 : i32, ref_starttime = 46 : i32, starttime = 46 : i32} : f32
            %105 = tor.mulf %104 %cst_100 on (46 to 49) {dump = "unknown_49", fastmath = #arith.fastmath<none>, ref_endtime = 49 : i32, ref_starttime = 46 : i32} : f32
            %106 = arith.fptosi %105 {endtime = 50 : i32, ref_endtime = 50 : i32, ref_starttime = 49 : i32, starttime = 49 : i32} : f32 to i32
            %107 = tor.addi %77 %106 on (50 to 51) {dump = "unknown_50", ref_endtime = 51 : i32, ref_starttime = 50 : i32} : (i32, i32) -> i32
            tor.yield %107 : i32
          } else {
            tor.yield %c0_i32 : i32
          } {dump = "unknown_22", ref_endtime = 51 : i32, ref_starttime = 45 : i32}
          %49 = aps.globalload @au_generated_candidate_11__direct_buffer_1 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : i32
          %50 = tor.addi %49 %48 on (51 to 52) {dump = "unknown_51", ref_endtime = 52 : i32, ref_starttime = 51 : i32} : (i32, i32) -> i32
          aps.globalstore %50, @au_generated_candidate_11__direct_buffer_1 {endtime = 52 : i32, ref_endtime = 52 : i32, ref_starttime = 51 : i32, starttime = 51 : i32} : i32
        } {ref_endtime = 52 : i32, ref_starttime = 43 : i32}
      } {ref_endtime = 52 : i32, ref_starttime = 41 : i32}
      %10 = aps.globalload @au_generated_candidate_11__direct_buffer_1 {endtime = 42 : i32, ref_endtime = 42 : i32, ref_starttime = 41 : i32, starttime = 41 : i32} : i32
      aps.store_by @cpuitfc, %10, %8 {after = array<i64>, after_gap = array<i64>, direction = 1 : i32, endtime = 43 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 43 : i32, ref_starttime = 41 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 1.800000e+01 : f64, starttime = 41 : i32, transaction_id = 5 : i64, transfer_bytes = 4 : i64} : i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_2__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 1 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-knn-32/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_52", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_53", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_2__direct_buffer_0 : memref<16xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_2__direct_buffer_1 : memref<16xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 42 : i32, estimated_latency = 20 : i64, interface_cost = 36 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 42 : i32, ref_starttime = 30 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 2.025000e+02 : f64, starttime = 30 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 6 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<16xf32>, memref<16xf32>), i32, i32
      %4 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_54", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 6, 6>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 65 : i32, estimated_latency = 20 : i64, interface_cost = 36 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 65 : i32, ref_starttime = 46 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 2.025000e+02 : f64, starttime = 46 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 7 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<16xf32>, memref<16xf32>), i32, i32
      %5 = memref.get_global @au_generated_candidate_2__direct_buffer_2_0 : memref<16xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %6 = memref.get_global @au_generated_candidate_2__direct_buffer_2_1 : memref<16xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%5, %6)[%c0_i32], %c16_i32 {after = array<i64: 7, 6, 7>, after_gap = array<i64: 12, 13, 12>, endtime = 76 : i32, estimated_latency = 20 : i64, interface_cost = 36 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 76 : i32, ref_starttime = 61 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 2.025000e+02 : f64, starttime = 61 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 8 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<16xi32>, memref<16xi32>), i32, i32
      %7 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_55", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%5, %6)[%c16_i32], %c16_i32 {after = array<i64: 8, 7, 8>, after_gap = array<i64: 8, 20, 8>, after_segment = 0 : i64, endtime = 91 : i32, estimated_latency = 20 : i64, interface_cost = 36 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 36 : i64, memory_search_hits = 0 : i64, memory_search_states = 7 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 91 : i32, ref_starttime = 76 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 2.025000e+02 : f64, starttime = 76 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 9 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<16xi32>, memref<16xi32>), i32, i32
      %8 = aps.read_csr @au_generated_candidate_2__direct_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %9 = memref.get_global @au_generated_candidate_2__direct_buffer_3_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %10 = memref.get_global @au_generated_candidate_2__direct_buffer_3_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %8, (%9, %10)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 28 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 28 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 2.025000e+02 : f64, starttime = 16 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 10 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %11 = tor.addi %8 %c64_i32 on (0 to 1) {dump = "unknown_56", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %11, (%9, %10)[%c16_i32], %c16_i32 {after = array<i64: 10, 10>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 64 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 64 : i32, ref_starttime = 45 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 2.025000e+02 : f64, starttime = 45 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 11 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %12 = tor.addi %8 %c128_i32 on (0 to 1) {dump = "unknown_57", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %12, (%9, %10)[%c32_i32], %c16_i32 {after = array<i64: 11, 10, 11>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 75 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 75 : i32, ref_starttime = 60 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 2.025000e+02 : f64, starttime = 60 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 12 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %13 = tor.addi %8 %c192_i32 on (0 to 1) {dump = "unknown_58", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %13, (%9, %10)[%c48_i32], %c16_i32 {after = array<i64: 12, 11, 12>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 90 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 90 : i32, ref_starttime = 75 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 2.025000e+02 : f64, starttime = 75 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 13 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %14 = tor.addi %8 %c256_i32 on (0 to 1) {dump = "unknown_59", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %14, (%9, %10)[%c64_i32], %c16_i32 {after = array<i64: 13, 12, 13>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 105 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 105 : i32, ref_starttime = 90 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 2.025000e+02 : f64, starttime = 90 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 14 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %15 = tor.addi %8 %c320_i32 on (0 to 1) {dump = "unknown_60", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %15, (%9, %10)[%c80_i32], %c16_i32 {after = array<i64: 14, 13, 14>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 113 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 113 : i32, ref_starttime = 98 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 2.025000e+02 : f64, starttime = 98 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 15 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %16 = aps.read_csr @au_generated_candidate_2__direct_csr1 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %17 = memref.get_global @au_generated_candidate_2__direct_buffer_4_0 : memref<6xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %18 = memref.get_global @au_generated_candidate_2__direct_buffer_4_1 : memref<6xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %16, (%17, %18)[%c0_i32], %c8_i32 {after = array<i64>, after_gap = array<i64>, endtime = 23 : i32, estimated_latency = 10 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 8 : i64, memory_issue_cycle = 0 : i64, memory_latency = 8 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 10 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 23 : i32, ref_starttime = 15 : i32, required_alignment = 32 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 2.025000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 16 : i64, transfer_bytes = 32 : i64, transfer_offset = 0 : i64} : i32, (memref<6xf32>, memref<6xf32>), i32, i32
      %19 = tor.addi %16 %c32_i32 on (0 to 1) {dump = "unknown_61", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %19, (%17, %18)[%c8_i32], %c4_i32 {after = array<i64: 16, 16>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 40 : i32, estimated_latency = 10 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 10 : i64, memory_issue_cycle = 1 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 10 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 40 : i32, ref_starttime = 31 : i32, required_alignment = 16 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 2.025000e+02 : f64, starttime = 31 : i32, tl_channel = 1 : i32, transaction_group = 5 : i64, transaction_id = 17 : i64, transfer_bytes = 16 : i64, transfer_offset = 32 : i64} : i32, (memref<6xf32>, memref<6xf32>), i32, i32
      %20 = aps.read_csr @au_generated_candidate_2__direct_csr2 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %21 = memref.get_global @au_generated_candidate_2__direct_buffer_5_0 : memref<4xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %22 = memref.get_global @au_generated_candidate_2__direct_buffer_5_1 : memref<4xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %20, (%21, %22)[%c0_i32], %c8_i32 {after = array<i64>, after_gap = array<i64>, endtime = 9 : i32, estimated_latency = 8 : i64, interface_cost = 16 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 8 : i64, memory_issue_cycle = 0 : i64, memory_latency = 8 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 8 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 9 : i32, ref_starttime = 1 : i32, required_alignment = 32 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 2.025000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_id = 18 : i64, transfer_bytes = 32 : i64, transfer_offset = 0 : i64} : i32, (memref<4xf32>, memref<4xf32>), i32, i32
      %23 = aps.read_csr @au_generated_candidate_2__direct_csr3 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %24 = memref.get_global @au_generated_candidate_2__direct_buffer_6_0 : memref<4xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %25 = memref.get_global @au_generated_candidate_2__direct_buffer_6_1 : memref<4xi32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %23, (%24, %25)[%c0_i32], %c8_i32 {after = array<i64>, after_gap = array<i64>, endtime = 8 : i32, estimated_latency = 8 : i64, interface_cost = 16 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 8 : i64, memory_issue_cycle = 0 : i64, memory_latency = 8 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 8 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 8 : i32, ref_starttime = 0 : i32, required_alignment = 32 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 2.025000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_id = 19 : i64, transfer_bytes = 32 : i64, transfer_offset = 0 : i64} : i32, (memref<4xi32>, memref<4xi32>), i32, i32
      tor.for %arg3 = (%c0_i3 : i3) to (%c3_i3 : i3) step (%c1_i3 : i3)
      on (115 to 142){
        %28 = arith.extui %arg3 {endtime = 116 : i32, ref_endtime = 116 : i32, ref_starttime = 115 : i32, starttime = 115 : i32} : i3 to i32
        tor.for %arg4 = (%c0_i4 : i4) to (%c7_i4 : i4) step (%c1_i4 : i4)
        on (117 to 118){
          %30 = arith.extui %arg4 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i4 to i32
          %31 = arith.andi %30, %c1_i32 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %32 = arith.shli %28, %c2_i32 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %33 = arith.extui %arg4 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i4 to i32
          %34 = tor.cmpi slt %33 %c0_i32 on (117 to 118) {dump = "unknown_62", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i1
          %35 = arith.extui %arg4 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i4 to i32
          %36 = tor.subi %c-1_i32 %35 on (117 to 118) {dump = "unknown_63", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i32
          %37 = arith.extui %arg4 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i4 to i32
          %38 = arith.select %34, %36, %37 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %39 = tor.cmpi slt %38 %c0_i32 on (117 to 118) {dump = "unknown_64", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i1
          %40 = arith.extui %39 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i1 to i32
          %41 = tor.addi %38 %40 on (117 to 118) {dump = "unknown_65", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i32
          %42 = arith.shrsi %41, %c1_i32 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %43 = tor.subi %c-1_i32 %42 on (117 to 118) {dump = "unknown_66", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i32
          %44 = arith.select %34, %43, %42 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %45 = tor.addi %32 %44 on (117 to 118) {dump = "unknown_67", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i32
          %46 = tor.cmpi eq %31 %c0_i32 on (117 to 118) {dump = "unknown_68", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i1
          aps.write_smem_if %46, %cst_99, %2[%45] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 37 : i64}], aps.mem_id = 37 : i64, endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, slot = 0 : i32, starttime = 117 : i32} : f32, memref<16xf32>, i32
          %47 = tor.cmpi eq %31 %c1_i32 on (117 to 118) {dump = "unknown_69", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i1
          aps.write_smem_if %47, %cst_99, %3[%45] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 36 : i64}], aps.mem_id = 36 : i64, endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, slot = 0 : i32, starttime = 117 : i32} : f32, memref<16xf32>, i32
          %48 = arith.extui %arg4 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i4 to i32
          %49 = arith.andi %48, %c1_i32 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %50 = arith.shli %28, %c2_i32 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %51 = arith.extui %arg4 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i4 to i32
          %52 = tor.cmpi slt %51 %c0_i32 on (117 to 118) {dump = "unknown_70", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i1
          %53 = arith.extui %arg4 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i4 to i32
          %54 = tor.subi %c-1_i32 %53 on (117 to 118) {dump = "unknown_71", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i32
          %55 = arith.extui %arg4 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i4 to i32
          %56 = arith.select %52, %54, %55 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %57 = tor.cmpi slt %56 %c0_i32 on (117 to 118) {dump = "unknown_72", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i1
          %58 = arith.extui %57 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i1 to i32
          %59 = tor.addi %56 %58 on (117 to 118) {dump = "unknown_73", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i32
          %60 = arith.shrsi %59, %c1_i32 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %61 = tor.subi %c-1_i32 %60 on (117 to 118) {dump = "unknown_74", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i32
          %62 = arith.select %52, %61, %60 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i32
          %63 = tor.addi %50 %62 on (117 to 118) {dump = "unknown_75", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i32
          %64 = tor.cmpi eq %49 %c0_i32 on (117 to 118) {dump = "unknown_76", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i1
          aps.write_smem_if %64, %c-1_i32, %5[%63] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 35 : i64}], aps.mem_id = 35 : i64, endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, slot = 0 : i32, starttime = 117 : i32} : i32, memref<16xi32>, i32
          %65 = tor.cmpi eq %49 %c1_i32 on (117 to 118) {dump = "unknown_77", ref_endtime = 118 : i32, ref_starttime = 117 : i32} : (i32, i32) -> i1
          aps.write_smem_if %65, %c-1_i32, %6[%63] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 34 : i64}], aps.mem_id = 34 : i64, endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, slot = 0 : i32, starttime = 117 : i32} : i32, memref<16xi32>, i32
        } {ref_endtime = 118 : i32, ref_starttime = 117 : i32}
        %29 = arith.extui %arg3 {endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, starttime = 117 : i32} : i3 to i32
        tor.for %arg4 = (%c0_i6 : i6) to (%c31_i6 : i6) step (%c1_i6 : i6)
        on (119 to 142){
          %30 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %31 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %32 = tor.muli %31 %c3_i32 on (119 to 120) {dump = "unknown_78", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %33 = tor.cmpi slt %32 %c0_i32 on (119 to 120) {dump = "unknown_79", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %34 = tor.subi %c-1_i32 %32 on (119 to 120) {dump = "unknown_80", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %35 = arith.select %33, %34, %32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %36 = tor.cmpi slt %35 %c0_i32 on (119 to 120) {dump = "unknown_81", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %37 = arith.extui %36 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %38 = tor.addi %35 %37 on (119 to 120) {dump = "unknown_82", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %39 = arith.shrsi %38, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %40 = tor.subi %c-1_i32 %39 on (119 to 120) {dump = "unknown_83", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %41 = arith.select %33, %40, %39 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %42 = aps.read_smem %9[%41] {aps.mem_id = 33 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : memref<48xf32>, i32 -> f32
          %43 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %44 = tor.muli %43 %c3_i32 on (119 to 120) {dump = "unknown_84", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %45 = tor.cmpi slt %44 %c0_i32 on (119 to 120) {dump = "unknown_85", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %46 = tor.subi %c-1_i32 %44 on (119 to 120) {dump = "unknown_86", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %47 = arith.select %45, %46, %44 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %48 = tor.cmpi slt %47 %c0_i32 on (119 to 120) {dump = "unknown_87", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %49 = arith.extui %48 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %50 = tor.addi %47 %49 on (119 to 120) {dump = "unknown_88", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %51 = arith.shrsi %50, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %52 = tor.subi %c-1_i32 %51 on (119 to 120) {dump = "unknown_89", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %53 = arith.select %45, %52, %51 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %54 = aps.read_smem %10[%53] {aps.mem_id = 32 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : memref<48xf32>, i32 -> f32
          %55 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %56 = arith.andi %55, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %57 = tor.cmpi eq %56 %c0_i32 on (119 to 120) {dump = "unknown_90", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %58 = arith.select %57, %42, %54 {endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, starttime = 120 : i32} : f32
          %59 = tor.muli %29 %c3_i32 on (119 to 120) {dump = "unknown_91", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %60 = tor.cmpi slt %59 %c0_i32 on (119 to 120) {dump = "unknown_92", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %61 = tor.subi %c-1_i32 %59 on (119 to 120) {dump = "unknown_93", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %62 = arith.select %60, %61, %59 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %63 = tor.cmpi slt %62 %c0_i32 on (119 to 120) {dump = "unknown_94", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %64 = arith.extui %63 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %65 = tor.addi %62 %64 on (119 to 120) {dump = "unknown_95", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %66 = arith.shrsi %65, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %67 = tor.subi %c-1_i32 %66 on (119 to 120) {dump = "unknown_96", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %68 = arith.select %60, %67, %66 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %69 = aps.read_smem %17[%68] {aps.mem_id = 31 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : memref<6xf32>, i32 -> f32
          %70 = tor.muli %29 %c3_i32 on (119 to 120) {dump = "unknown_97", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %71 = tor.cmpi slt %70 %c0_i32 on (119 to 120) {dump = "unknown_98", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %72 = tor.subi %c-1_i32 %70 on (119 to 120) {dump = "unknown_99", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %73 = arith.select %71, %72, %70 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %74 = tor.cmpi slt %73 %c0_i32 on (119 to 120) {dump = "unknown_100", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %75 = arith.extui %74 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %76 = tor.addi %73 %75 on (119 to 120) {dump = "unknown_101", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %77 = arith.shrsi %76, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %78 = tor.subi %c-1_i32 %77 on (119 to 120) {dump = "unknown_102", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %79 = arith.select %71, %78, %77 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %80 = aps.read_smem %18[%79] {aps.mem_id = 30 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : memref<6xf32>, i32 -> f32
          %81 = arith.andi %29, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %82 = tor.cmpi eq %81 %c0_i32 on (119 to 120) {dump = "unknown_103", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %83 = arith.select %82, %69, %80 {endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, starttime = 120 : i32} : f32
          %84 = tor.subf %58 %83 on (120 to 123) {dump = "unknown_104", fastmath = #arith.fastmath<none>, ref_endtime = 123 : i32, ref_starttime = 120 : i32} : f32
          %85 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %86 = tor.muli %85 %c3_i32 on (119 to 120) {dump = "unknown_105", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %87 = tor.addi %86 %c1_i32 on (119 to 120) {dump = "unknown_106", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %88 = tor.cmpi slt %87 %c0_i32 on (119 to 120) {dump = "unknown_107", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %89 = tor.subi %c-2_i32 %86 on (119 to 120) {dump = "unknown_108", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %90 = arith.select %88, %89, %87 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %91 = tor.cmpi slt %90 %c0_i32 on (119 to 120) {dump = "unknown_109", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %92 = arith.extui %91 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %93 = tor.addi %90 %92 on (119 to 120) {dump = "unknown_110", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %94 = arith.shrsi %93, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %95 = tor.subi %c-1_i32 %94 on (119 to 120) {dump = "unknown_111", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %96 = arith.select %88, %95, %94 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %97 = aps.read_smem %9[%96] {aps.mem_id = 29 : i64, endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, slot = 0 : i32, starttime = 120 : i32} : memref<48xf32>, i32 -> f32
          %98 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %99 = tor.muli %98 %c3_i32 on (119 to 120) {dump = "unknown_112", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %100 = tor.addi %99 %c1_i32 on (119 to 120) {dump = "unknown_113", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %101 = tor.cmpi slt %100 %c0_i32 on (119 to 120) {dump = "unknown_114", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %102 = tor.subi %c-2_i32 %99 on (119 to 120) {dump = "unknown_115", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %103 = arith.select %101, %102, %100 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %104 = tor.cmpi slt %103 %c0_i32 on (119 to 120) {dump = "unknown_116", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %105 = arith.extui %104 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %106 = tor.addi %103 %105 on (119 to 120) {dump = "unknown_117", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %107 = arith.shrsi %106, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %108 = tor.subi %c-1_i32 %107 on (119 to 120) {dump = "unknown_118", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %109 = arith.select %101, %108, %107 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %110 = aps.read_smem %10[%109] {aps.mem_id = 28 : i64, endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, slot = 0 : i32, starttime = 120 : i32} : memref<48xf32>, i32 -> f32
          %111 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %112 = tor.addi %111 %c1_i32 on (119 to 120) {dump = "unknown_119", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %113 = tor.cmpi slt %112 %c0_i32 on (119 to 120) {dump = "unknown_120", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %114 = tor.subi %c-2_i32 %111 on (119 to 120) {dump = "unknown_121", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %115 = arith.select %113, %114, %112 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %116 = tor.cmpi slt %115 %c0_i32 on (119 to 120) {dump = "unknown_122", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %117 = arith.extui %116 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %118 = tor.addi %115 %117 on (119 to 120) {dump = "unknown_123", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %119 = arith.shrsi %118, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %120 = tor.subi %c-1_i32 %119 on (119 to 120) {dump = "unknown_124", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %121 = arith.select %113, %120, %119 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %122 = tor.muli %121 %c-2_i32 on (119 to 120) {dump = "unknown_125", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %123 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %124 = tor.addi %123 %122 on (119 to 120) {dump = "unknown_126", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %125 = tor.addi %124 %c1_i32 on (119 to 120) {dump = "unknown_127", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %126 = tor.cmpi eq %125 %c0_i32 on (119 to 120) {dump = "unknown_128", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %127 = arith.select %126, %97, %110 {endtime = 122 : i32, ref_endtime = 122 : i32, ref_starttime = 121 : i32, starttime = 121 : i32} : f32
          %128 = tor.muli %29 %c3_i32 on (119 to 120) {dump = "unknown_129", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %129 = tor.addi %128 %c1_i32 on (119 to 120) {dump = "unknown_130", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %130 = tor.cmpi slt %129 %c0_i32 on (119 to 120) {dump = "unknown_131", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %131 = tor.subi %c-2_i32 %128 on (119 to 120) {dump = "unknown_132", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %132 = arith.select %130, %131, %129 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %133 = tor.cmpi slt %132 %c0_i32 on (119 to 120) {dump = "unknown_133", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %134 = arith.extui %133 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %135 = tor.addi %132 %134 on (119 to 120) {dump = "unknown_134", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %136 = arith.shrsi %135, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %137 = tor.subi %c-1_i32 %136 on (119 to 120) {dump = "unknown_135", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %138 = arith.select %130, %137, %136 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %139 = aps.read_smem %17[%138] {aps.mem_id = 27 : i64, endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, slot = 0 : i32, starttime = 120 : i32} : memref<6xf32>, i32 -> f32
          %140 = tor.muli %29 %c3_i32 on (119 to 120) {dump = "unknown_136", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %141 = tor.addi %140 %c1_i32 on (119 to 120) {dump = "unknown_137", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %142 = tor.cmpi slt %141 %c0_i32 on (119 to 120) {dump = "unknown_138", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %143 = tor.subi %c-2_i32 %140 on (119 to 120) {dump = "unknown_139", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %144 = arith.select %142, %143, %141 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %145 = tor.cmpi slt %144 %c0_i32 on (119 to 120) {dump = "unknown_140", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %146 = arith.extui %145 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %147 = tor.addi %144 %146 on (119 to 120) {dump = "unknown_141", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %148 = arith.shrsi %147, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %149 = tor.subi %c-1_i32 %148 on (119 to 120) {dump = "unknown_142", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %150 = arith.select %142, %149, %148 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %151 = aps.read_smem %18[%150] {aps.mem_id = 26 : i64, endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, slot = 0 : i32, starttime = 120 : i32} : memref<6xf32>, i32 -> f32
          %152 = tor.addi %29 %c1_i32 on (119 to 120) {dump = "unknown_143", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %153 = tor.cmpi slt %152 %c0_i32 on (119 to 120) {dump = "unknown_144", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %154 = tor.subi %c-2_i32 %29 on (119 to 120) {dump = "unknown_145", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %155 = arith.select %153, %154, %152 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %156 = tor.cmpi slt %155 %c0_i32 on (119 to 120) {dump = "unknown_146", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %157 = arith.extui %156 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %158 = tor.addi %155 %157 on (119 to 120) {dump = "unknown_147", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %159 = arith.shrsi %158, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %160 = tor.subi %c-1_i32 %159 on (119 to 120) {dump = "unknown_148", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %161 = arith.select %153, %160, %159 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %162 = tor.muli %161 %c-2_i32 on (119 to 120) {dump = "unknown_149", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %163 = tor.addi %29 %162 on (119 to 120) {dump = "unknown_150", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %164 = tor.addi %163 %c1_i32 on (119 to 120) {dump = "unknown_151", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %165 = tor.cmpi eq %164 %c0_i32 on (119 to 120) {dump = "unknown_152", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %166 = arith.select %165, %139, %151 {endtime = 122 : i32, ref_endtime = 122 : i32, ref_starttime = 121 : i32, starttime = 121 : i32} : f32
          %167 = tor.subf %127 %166 on (121 to 124) {dump = "unknown_153", fastmath = #arith.fastmath<none>, ref_endtime = 124 : i32, ref_starttime = 121 : i32} : f32
          %168 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %169 = tor.muli %168 %c3_i32 on (119 to 120) {dump = "unknown_154", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %170 = tor.cmpi slt %169 %c0_i32 on (119 to 120) {dump = "unknown_155", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %171 = tor.subi %c-1_i32 %169 on (119 to 120) {dump = "unknown_156", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %172 = arith.select %170, %171, %169 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %173 = tor.cmpi slt %172 %c0_i32 on (119 to 120) {dump = "unknown_157", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %174 = arith.extui %173 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %175 = tor.addi %172 %174 on (119 to 120) {dump = "unknown_158", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %176 = arith.shrsi %175, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %177 = tor.subi %c-1_i32 %176 on (119 to 120) {dump = "unknown_159", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %178 = arith.select %170, %177, %176 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %179 = tor.addi %178 %c1_i32 on (119 to 120) {dump = "unknown_160", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %180 = aps.read_smem %9[%179] {aps.mem_id = 25 : i64, endtime = 122 : i32, ref_endtime = 122 : i32, ref_starttime = 121 : i32, slot = 0 : i32, starttime = 121 : i32} : memref<48xf32>, i32 -> f32
          %181 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %182 = tor.muli %181 %c3_i32 on (119 to 120) {dump = "unknown_161", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %183 = tor.cmpi slt %182 %c0_i32 on (119 to 120) {dump = "unknown_162", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %184 = tor.subi %c-1_i32 %182 on (119 to 120) {dump = "unknown_163", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %185 = arith.select %183, %184, %182 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %186 = tor.cmpi slt %185 %c0_i32 on (119 to 120) {dump = "unknown_164", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %187 = arith.extui %186 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %188 = tor.addi %185 %187 on (119 to 120) {dump = "unknown_165", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %189 = arith.shrsi %188, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %190 = tor.subi %c-1_i32 %189 on (119 to 120) {dump = "unknown_166", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %191 = arith.select %183, %190, %189 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %192 = tor.addi %191 %c1_i32 on (119 to 120) {dump = "unknown_167", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %193 = aps.read_smem %10[%192] {aps.mem_id = 24 : i64, endtime = 122 : i32, ref_endtime = 122 : i32, ref_starttime = 121 : i32, slot = 0 : i32, starttime = 121 : i32} : memref<48xf32>, i32 -> f32
          %194 = arith.extui %arg4 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i6 to i32
          %195 = arith.andi %194, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %196 = tor.cmpi eq %195 %c0_i32 on (119 to 120) {dump = "unknown_168", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %197 = arith.select %196, %180, %193 {endtime = 123 : i32, ref_endtime = 123 : i32, ref_starttime = 122 : i32, starttime = 122 : i32} : f32
          %198 = tor.muli %29 %c3_i32 on (119 to 120) {dump = "unknown_169", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %199 = tor.cmpi slt %198 %c0_i32 on (119 to 120) {dump = "unknown_170", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %200 = tor.subi %c-1_i32 %198 on (119 to 120) {dump = "unknown_171", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %201 = arith.select %199, %200, %198 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %202 = tor.cmpi slt %201 %c0_i32 on (119 to 120) {dump = "unknown_172", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %203 = arith.extui %202 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %204 = tor.addi %201 %203 on (119 to 120) {dump = "unknown_173", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %205 = arith.shrsi %204, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %206 = tor.subi %c-1_i32 %205 on (119 to 120) {dump = "unknown_174", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %207 = arith.select %199, %206, %205 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %208 = tor.addi %207 %c1_i32 on (119 to 120) {dump = "unknown_175", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %209 = aps.read_smem %17[%208] {aps.mem_id = 23 : i64, endtime = 122 : i32, ref_endtime = 122 : i32, ref_starttime = 121 : i32, slot = 0 : i32, starttime = 121 : i32} : memref<6xf32>, i32 -> f32
          %210 = tor.muli %29 %c3_i32 on (119 to 120) {dump = "unknown_176", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %211 = tor.cmpi slt %210 %c0_i32 on (119 to 120) {dump = "unknown_177", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %212 = tor.subi %c-1_i32 %210 on (119 to 120) {dump = "unknown_178", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %213 = arith.select %211, %212, %210 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %214 = tor.cmpi slt %213 %c0_i32 on (119 to 120) {dump = "unknown_179", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %215 = arith.extui %214 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i1 to i32
          %216 = tor.addi %213 %215 on (119 to 120) {dump = "unknown_180", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %217 = arith.shrsi %216, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %218 = tor.subi %c-1_i32 %217 on (119 to 120) {dump = "unknown_181", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %219 = arith.select %211, %218, %217 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %220 = tor.addi %219 %c1_i32 on (119 to 120) {dump = "unknown_182", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i32
          %221 = aps.read_smem %18[%220] {aps.mem_id = 22 : i64, endtime = 122 : i32, ref_endtime = 122 : i32, ref_starttime = 121 : i32, slot = 0 : i32, starttime = 121 : i32} : memref<6xf32>, i32 -> f32
          %222 = arith.andi %29, %c1_i32 {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          %223 = tor.cmpi eq %222 %c0_i32 on (119 to 120) {dump = "unknown_183", ref_endtime = 120 : i32, ref_starttime = 119 : i32} : (i32, i32) -> i1
          %224 = arith.select %223, %209, %221 {endtime = 123 : i32, ref_endtime = 123 : i32, ref_starttime = 122 : i32, starttime = 122 : i32} : f32
          %225 = tor.subf %197 %224 on (122 to 125) {dump = "unknown_184", fastmath = #arith.fastmath<none>, ref_endtime = 125 : i32, ref_starttime = 122 : i32} : f32
          %226 = tor.mulf %84 %84 on (123 to 126) {dump = "unknown_185", fastmath = #arith.fastmath<none>, ref_endtime = 126 : i32, ref_starttime = 123 : i32} : f32
          %227 = tor.mulf %167 %167 on (124 to 127) {dump = "unknown_186", fastmath = #arith.fastmath<none>, ref_endtime = 127 : i32, ref_starttime = 124 : i32} : f32
          %228 = tor.addf %226 %227 on (127 to 130) {dump = "unknown_187", fastmath = #arith.fastmath<none>, ref_endtime = 130 : i32, ref_starttime = 127 : i32} : f32
          %229 = tor.mulf %225 %225 on (125 to 128) {dump = "unknown_188", fastmath = #arith.fastmath<none>, ref_endtime = 128 : i32, ref_starttime = 125 : i32} : f32
          %230 = tor.addf %228 %229 on (130 to 133) {dump = "unknown_189", fastmath = #arith.fastmath<none>, ref_endtime = 133 : i32, ref_starttime = 130 : i32} : f32
          aps.globalstore %c0_i32, @au_generated_candidate_2__direct_local {endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, starttime = 119 : i32} : i32
          tor.for %arg5 = (%c0_i4 : i4) to (%c7_i4 : i4) step (%c1_i4 : i4)
          on (135 to 142){
            %231 = arith.extui %arg5 {endtime = 136 : i32, ref_endtime = 136 : i32, ref_starttime = 135 : i32, starttime = 135 : i32} : i4 to i32
            %232 = aps.globalload @au_generated_candidate_2__direct_local {endtime = 136 : i32, ref_endtime = 136 : i32, ref_starttime = 135 : i32, starttime = 135 : i32} : i32
            %233 = tor.cmpi eq %232 %c0_i32 on (135 to 136) {dump = "unknown_190", ref_endtime = 136 : i32, ref_starttime = 135 : i32} : (i32, i32) -> i1
            %234 = arith.extui %arg5 {endtime = 136 : i32, ref_endtime = 136 : i32, ref_starttime = 135 : i32, starttime = 135 : i32} : i4 to i32
            %235 = tor.if %233 on (136 to 138) -> (i1) then {
              %237 = arith.shli %29, %c2_i32 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i32
              %238 = tor.cmpi slt %234 %c0_i32 on (136 to 137) {dump = "unknown_192", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i1
              %239 = tor.subi %c-1_i32 %234 on (136 to 137) {dump = "unknown_193", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i32
              %240 = arith.select %238, %239, %234 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i32
              %241 = tor.cmpi slt %240 %c0_i32 on (136 to 137) {dump = "unknown_194", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i1
              %242 = arith.extui %241 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i1 to i32
              %243 = tor.addi %240 %242 on (136 to 137) {dump = "unknown_195", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i32
              %244 = arith.shrsi %243, %c1_i32 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i32
              %245 = tor.subi %c-1_i32 %244 on (136 to 137) {dump = "unknown_196", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i32
              %246 = arith.select %238, %245, %244 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i32
              %247 = tor.addi %237 %246 on (136 to 137) {dump = "unknown_197", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i32
              %248 = aps.read_smem %2[%247] {aps.mem_id = 21 : i64, endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, slot = 0 : i32, starttime = 136 : i32} : memref<16xf32>, i32 -> f32
              %249 = arith.shli %29, %c2_i32 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i32
              %250 = tor.cmpi slt %234 %c0_i32 on (136 to 137) {dump = "unknown_198", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i1
              %251 = tor.subi %c-1_i32 %234 on (136 to 137) {dump = "unknown_199", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i32
              %252 = arith.select %250, %251, %234 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i32
              %253 = tor.cmpi slt %252 %c0_i32 on (136 to 137) {dump = "unknown_200", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i1
              %254 = arith.extui %253 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i1 to i32
              %255 = tor.addi %252 %254 on (136 to 137) {dump = "unknown_201", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i32
              %256 = arith.shrsi %255, %c1_i32 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i32
              %257 = tor.subi %c-1_i32 %256 on (136 to 137) {dump = "unknown_202", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i32
              %258 = arith.select %250, %257, %256 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i32
              %259 = tor.addi %249 %258 on (136 to 137) {dump = "unknown_203", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i32
              %260 = aps.read_smem %3[%259] {aps.mem_id = 20 : i64, endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, slot = 0 : i32, starttime = 136 : i32} : memref<16xf32>, i32 -> f32
              %261 = arith.andi %234, %c1_i32 {endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, starttime = 136 : i32} : i32
              %262 = tor.cmpi eq %261 %c0_i32 on (136 to 137) {dump = "unknown_204", ref_endtime = 137 : i32, ref_starttime = 136 : i32} : (i32, i32) -> i1
              %263 = arith.select %262, %248, %260 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : f32
              %264 = tor.cmpf olt %230 %263 on (137 to 138) {dump = "unknown_205", fastmath = #arith.fastmath<none>, ref_endtime = 138 : i32, ref_starttime = 137 : i32} : f32 -> i1
              tor.yield %264 : i1
            } else {
              tor.yield %false : i1
            } {dump = "unknown_191", ref_endtime = 138 : i32, ref_starttime = 136 : i32}
            %236 = arith.extui %arg5 {endtime = 140 : i32, ref_endtime = 140 : i32, ref_starttime = 139 : i32, starttime = 139 : i32} : i4 to i32
            tor.if %235 on (140 to 142) then {
              %237 = arith.andi %236, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %238 = tor.cmpi slt %236 %c0_i32 on (140 to 141) {dump = "unknown_207", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %239 = tor.subi %c-1_i32 %236 on (140 to 141) {dump = "unknown_208", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %240 = arith.select %238, %239, %236 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %241 = tor.cmpi slt %240 %c0_i32 on (140 to 141) {dump = "unknown_209", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %242 = arith.extui %241 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i1 to i32
              %243 = tor.addi %240 %242 on (140 to 141) {dump = "unknown_210", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %244 = arith.shrsi %243, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %245 = tor.subi %c-1_i32 %244 on (140 to 141) {dump = "unknown_211", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %246 = arith.select %238, %245, %244 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %247 = tor.cmpi eq %237 %c0_i32 on (140 to 141) {dump = "unknown_212", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              aps.write_smem_if %247, %230, %21[%246] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 11 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 17 : i64}], aps.mem_id = 11 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : f32, memref<4xf32>, i32
              %248 = tor.cmpi eq %237 %c1_i32 on (140 to 141) {dump = "unknown_213", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              aps.write_smem_if %248, %230, %22[%246] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 10 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 16 : i64}], aps.mem_id = 10 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : f32, memref<4xf32>, i32
              %249 = arith.andi %236, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %250 = tor.cmpi slt %236 %c0_i32 on (140 to 141) {dump = "unknown_214", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %251 = tor.subi %c-1_i32 %236 on (140 to 141) {dump = "unknown_215", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %252 = arith.select %250, %251, %236 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %253 = tor.cmpi slt %252 %c0_i32 on (140 to 141) {dump = "unknown_216", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %254 = arith.extui %253 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i1 to i32
              %255 = tor.addi %252 %254 on (140 to 141) {dump = "unknown_217", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %256 = arith.shrsi %255, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %257 = tor.subi %c-1_i32 %256 on (140 to 141) {dump = "unknown_218", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %258 = arith.select %250, %257, %256 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %259 = tor.cmpi eq %249 %c0_i32 on (140 to 141) {dump = "unknown_219", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              aps.write_smem_if %259, %30, %24[%258] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 9 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 13 : i64}], aps.mem_id = 9 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : i32, memref<4xi32>, i32
              %260 = tor.cmpi eq %249 %c1_i32 on (140 to 141) {dump = "unknown_220", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              aps.write_smem_if %260, %30, %25[%258] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 8 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 12 : i64}], aps.mem_id = 8 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : i32, memref<4xi32>, i32
              aps.globalstore %c1_i32, @au_generated_candidate_2__direct_local {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
            } else {
              %237 = tor.subi %231 %232 on (140 to 141) {dump = "unknown_221", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %238 = arith.shli %29, %c3_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %239 = tor.addi %238 %237 on (140 to 141) {dump = "unknown_222", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %240 = arith.andi %239, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %241 = arith.shrui %239, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %242 = aps.read_smem %2[%241] {aps.mem_id = 19 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : memref<16xf32>, i32 -> f32
              %243 = tor.cmpi eq %240 %c1_i32 on (140 to 141) {dump = "unknown_223", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %244 = aps.read_smem %3[%241] {aps.mem_id = 18 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : memref<16xf32>, i32 -> f32
              %245 = arith.select %243, %244, %242 {endtime = 142 : i32, ref_endtime = 142 : i32, ref_starttime = 141 : i32, starttime = 141 : i32} : f32
              %246 = arith.andi %236, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %247 = tor.cmpi slt %236 %c0_i32 on (140 to 141) {dump = "unknown_224", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %248 = tor.subi %c-1_i32 %236 on (140 to 141) {dump = "unknown_225", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %249 = arith.select %247, %248, %236 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %250 = tor.cmpi slt %249 %c0_i32 on (140 to 141) {dump = "unknown_226", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %251 = arith.extui %250 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i1 to i32
              %252 = tor.addi %249 %251 on (140 to 141) {dump = "unknown_227", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %253 = arith.shrsi %252, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %254 = tor.subi %c-1_i32 %253 on (140 to 141) {dump = "unknown_228", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %255 = arith.select %247, %254, %253 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %256 = tor.cmpi eq %246 %c0_i32 on (140 to 141) {dump = "unknown_229", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              aps.write_smem_if %256, %245, %21[%255] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 17 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 11 : i64}], aps.mem_id = 17 : i64, endtime = 142 : i32, ref_endtime = 142 : i32, ref_starttime = 141 : i32, slot = 0 : i32, starttime = 141 : i32} : f32, memref<4xf32>, i32
              %257 = tor.cmpi eq %246 %c1_i32 on (140 to 141) {dump = "unknown_230", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              aps.write_smem_if %257, %245, %22[%255] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 16 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 10 : i64}], aps.mem_id = 16 : i64, endtime = 142 : i32, ref_endtime = 142 : i32, ref_starttime = 141 : i32, slot = 0 : i32, starttime = 141 : i32} : f32, memref<4xf32>, i32
              %258 = arith.andi %239, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %259 = arith.shrui %239, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %260 = aps.read_smem %5[%259] {aps.mem_id = 15 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : memref<16xi32>, i32 -> i32
              %261 = tor.cmpi eq %258 %c1_i32 on (140 to 141) {dump = "unknown_231", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %262 = aps.read_smem %6[%259] {aps.mem_id = 14 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : memref<16xi32>, i32 -> i32
              %263 = arith.select %261, %262, %260 {endtime = 142 : i32, ref_endtime = 142 : i32, ref_starttime = 141 : i32, starttime = 141 : i32} : i32
              %264 = arith.andi %236, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %265 = tor.cmpi slt %236 %c0_i32 on (140 to 141) {dump = "unknown_232", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %266 = tor.subi %c-1_i32 %236 on (140 to 141) {dump = "unknown_233", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %267 = arith.select %265, %266, %236 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %268 = tor.cmpi slt %267 %c0_i32 on (140 to 141) {dump = "unknown_234", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              %269 = arith.extui %268 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i1 to i32
              %270 = tor.addi %267 %269 on (140 to 141) {dump = "unknown_235", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %271 = arith.shrsi %270, %c1_i32 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %272 = tor.subi %c-1_i32 %271 on (140 to 141) {dump = "unknown_236", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i32
              %273 = arith.select %265, %272, %271 {endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, starttime = 140 : i32} : i32
              %274 = tor.cmpi eq %264 %c0_i32 on (140 to 141) {dump = "unknown_237", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              aps.write_smem_if %274, %263, %24[%273] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 13 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 9 : i64}], aps.mem_id = 13 : i64, endtime = 142 : i32, ref_endtime = 142 : i32, ref_starttime = 141 : i32, slot = 0 : i32, starttime = 141 : i32} : i32, memref<4xi32>, i32
              %275 = tor.cmpi eq %264 %c1_i32 on (140 to 141) {dump = "unknown_238", ref_endtime = 141 : i32, ref_starttime = 140 : i32} : (i32, i32) -> i1
              aps.write_smem_if %275, %263, %25[%273] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 12 : i64}, {distance = 1 : i64, kind = 1 : i32, source = 8 : i64}], aps.mem_id = 12 : i64, endtime = 142 : i32, ref_endtime = 142 : i32, ref_starttime = 141 : i32, slot = 0 : i32, starttime = 141 : i32} : i32, memref<4xi32>, i32
            } {dump = "unknown_206", ref_endtime = 142 : i32, ref_starttime = 140 : i32}
          } {ref_endtime = 142 : i32, ref_starttime = 135 : i32}
          tor.for %arg5 = (%c0_i4 : i4) to (%c7_i4 : i4) step (%c1_i4 : i4)
          on (137 to 139){
            %231 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %232 = tor.cmpi slt %231 %c0_i32 on (137 to 138) {dump = "unknown_239", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %233 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %234 = tor.subi %c-1_i32 %233 on (137 to 138) {dump = "unknown_240", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %235 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %236 = arith.select %232, %234, %235 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %237 = tor.cmpi slt %236 %c0_i32 on (137 to 138) {dump = "unknown_241", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %238 = arith.extui %237 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i1 to i32
            %239 = tor.addi %236 %238 on (137 to 138) {dump = "unknown_242", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %240 = arith.shrsi %239, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %241 = tor.subi %c-1_i32 %240 on (137 to 138) {dump = "unknown_243", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %242 = arith.select %232, %241, %240 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %243 = aps.read_smem %21[%242] {aps.mem_id = 7 : i64, endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, slot = 0 : i32, starttime = 137 : i32} : memref<4xf32>, i32 -> f32
            %244 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %245 = tor.cmpi slt %244 %c0_i32 on (137 to 138) {dump = "unknown_244", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %246 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %247 = tor.subi %c-1_i32 %246 on (137 to 138) {dump = "unknown_245", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %248 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %249 = arith.select %245, %247, %248 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %250 = tor.cmpi slt %249 %c0_i32 on (137 to 138) {dump = "unknown_246", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %251 = arith.extui %250 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i1 to i32
            %252 = tor.addi %249 %251 on (137 to 138) {dump = "unknown_247", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %253 = arith.shrsi %252, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %254 = tor.subi %c-1_i32 %253 on (137 to 138) {dump = "unknown_248", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %255 = arith.select %245, %254, %253 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %256 = aps.read_smem %22[%255] {aps.mem_id = 6 : i64, endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, slot = 0 : i32, starttime = 137 : i32} : memref<4xf32>, i32 -> f32
            %257 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %258 = arith.andi %257, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %259 = tor.cmpi eq %258 %c0_i32 on (137 to 138) {dump = "unknown_249", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %260 = arith.select %259, %243, %256 {endtime = 139 : i32, ref_endtime = 139 : i32, ref_starttime = 138 : i32, starttime = 138 : i32} : f32
            %261 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %262 = arith.andi %261, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %263 = arith.shli %29, %c2_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %264 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %265 = tor.cmpi slt %264 %c0_i32 on (137 to 138) {dump = "unknown_250", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %266 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %267 = tor.subi %c-1_i32 %266 on (137 to 138) {dump = "unknown_251", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %268 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %269 = arith.select %265, %267, %268 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %270 = tor.cmpi slt %269 %c0_i32 on (137 to 138) {dump = "unknown_252", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %271 = arith.extui %270 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i1 to i32
            %272 = tor.addi %269 %271 on (137 to 138) {dump = "unknown_253", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %273 = arith.shrsi %272, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %274 = tor.subi %c-1_i32 %273 on (137 to 138) {dump = "unknown_254", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %275 = arith.select %265, %274, %273 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %276 = tor.addi %263 %275 on (137 to 138) {dump = "unknown_255", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %277 = tor.cmpi eq %262 %c0_i32 on (137 to 138) {dump = "unknown_256", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            aps.write_smem_if %277, %260, %2[%276] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 5 : i64}], aps.mem_id = 5 : i64, endtime = 139 : i32, ref_endtime = 139 : i32, ref_starttime = 138 : i32, slot = 0 : i32, starttime = 138 : i32} : f32, memref<16xf32>, i32
            %278 = tor.cmpi eq %262 %c1_i32 on (137 to 138) {dump = "unknown_257", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            aps.write_smem_if %278, %260, %3[%276] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 4 : i64}], aps.mem_id = 4 : i64, endtime = 139 : i32, ref_endtime = 139 : i32, ref_starttime = 138 : i32, slot = 0 : i32, starttime = 138 : i32} : f32, memref<16xf32>, i32
            %279 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %280 = tor.cmpi slt %279 %c0_i32 on (137 to 138) {dump = "unknown_258", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %281 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %282 = tor.subi %c-1_i32 %281 on (137 to 138) {dump = "unknown_259", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %283 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %284 = arith.select %280, %282, %283 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %285 = tor.cmpi slt %284 %c0_i32 on (137 to 138) {dump = "unknown_260", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %286 = arith.extui %285 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i1 to i32
            %287 = tor.addi %284 %286 on (137 to 138) {dump = "unknown_261", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %288 = arith.shrsi %287, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %289 = tor.subi %c-1_i32 %288 on (137 to 138) {dump = "unknown_262", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %290 = arith.select %280, %289, %288 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %291 = aps.read_smem %24[%290] {aps.mem_id = 3 : i64, endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, slot = 0 : i32, starttime = 137 : i32} : memref<4xi32>, i32 -> i32
            %292 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %293 = tor.cmpi slt %292 %c0_i32 on (137 to 138) {dump = "unknown_263", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %294 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %295 = tor.subi %c-1_i32 %294 on (137 to 138) {dump = "unknown_264", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %296 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %297 = arith.select %293, %295, %296 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %298 = tor.cmpi slt %297 %c0_i32 on (137 to 138) {dump = "unknown_265", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %299 = arith.extui %298 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i1 to i32
            %300 = tor.addi %297 %299 on (137 to 138) {dump = "unknown_266", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %301 = arith.shrsi %300, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %302 = tor.subi %c-1_i32 %301 on (137 to 138) {dump = "unknown_267", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %303 = arith.select %293, %302, %301 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %304 = aps.read_smem %25[%303] {aps.mem_id = 2 : i64, endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, slot = 0 : i32, starttime = 137 : i32} : memref<4xi32>, i32 -> i32
            %305 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %306 = arith.andi %305, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %307 = tor.cmpi eq %306 %c0_i32 on (137 to 138) {dump = "unknown_268", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %308 = arith.select %307, %291, %304 {endtime = 139 : i32, ref_endtime = 139 : i32, ref_starttime = 138 : i32, starttime = 138 : i32} : i32
            %309 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %310 = arith.andi %309, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %311 = arith.shli %29, %c2_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %312 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %313 = tor.cmpi slt %312 %c0_i32 on (137 to 138) {dump = "unknown_269", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %314 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %315 = tor.subi %c-1_i32 %314 on (137 to 138) {dump = "unknown_270", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %316 = arith.extui %arg5 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i4 to i32
            %317 = arith.select %313, %315, %316 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %318 = tor.cmpi slt %317 %c0_i32 on (137 to 138) {dump = "unknown_271", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            %319 = arith.extui %318 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i1 to i32
            %320 = tor.addi %317 %319 on (137 to 138) {dump = "unknown_272", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %321 = arith.shrsi %320, %c1_i32 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %322 = tor.subi %c-1_i32 %321 on (137 to 138) {dump = "unknown_273", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %323 = arith.select %313, %322, %321 {endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, starttime = 137 : i32} : i32
            %324 = tor.addi %311 %323 on (137 to 138) {dump = "unknown_274", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i32
            %325 = tor.cmpi eq %310 %c0_i32 on (137 to 138) {dump = "unknown_275", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            aps.write_smem_if %325, %308, %5[%324] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 1 : i64}], aps.mem_id = 1 : i64, endtime = 139 : i32, ref_endtime = 139 : i32, ref_starttime = 138 : i32, slot = 0 : i32, starttime = 138 : i32} : i32, memref<16xi32>, i32
            %326 = tor.cmpi eq %310 %c1_i32 on (137 to 138) {dump = "unknown_276", ref_endtime = 138 : i32, ref_starttime = 137 : i32} : (i32, i32) -> i1
            aps.write_smem_if %326, %308, %6[%324] {aps.mem_deps = [{distance = 1 : i64, kind = 1 : i32, source = 0 : i64}], aps.mem_id = 0 : i64, endtime = 139 : i32, ref_endtime = 139 : i32, ref_starttime = 138 : i32, slot = 0 : i32, starttime = 138 : i32} : i32, memref<16xi32>, i32
          } {ref_endtime = 139 : i32, ref_starttime = 137 : i32}
        } {ref_endtime = 142 : i32, ref_starttime = 119 : i32}
      } {ref_endtime = 142 : i32, ref_starttime = 115 : i32}
      aps.copy_by @busitfc, (%2, %3)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 150 : i32, estimated_latency = 19 : i64, interface_cost = 35 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 51 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 150 : i32, ref_starttime = 141 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 9.900000e+01 : f64, starttime = 141 : i32, tl_channel = 0 : i32, transaction_group = 8 : i64, transaction_id = 20 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<16xf32>, memref<16xf32>), i32, i32, i32
      %26 = tor.addi %0 %c64_i32 on (115 to 116) {dump = "unknown_277", ref_endtime = 116 : i32, ref_starttime = 115 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3)[%c16_i32], %26, %c16_i32 {after = array<i64: 20, 20>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 160 : i32, estimated_latency = 19 : i64, interface_cost = 35 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 51 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 160 : i32, ref_starttime = 142 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 9.900000e+01 : f64, starttime = 142 : i32, tl_channel = 1 : i32, transaction_group = 8 : i64, transaction_id = 21 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<16xf32>, memref<16xf32>), i32, i32, i32
      aps.copy_by @busitfc, (%5, %6)[%c0_i32], %1, %c16_i32 {after = array<i64: 21, 21, 20, 21>, after_gap = array<i64: 19, 19, 10, 19>, endtime = 170 : i32, estimated_latency = 19 : i64, interface_cost = 35 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 20 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 51 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 170 : i32, ref_starttime = 161 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 9.900000e+01 : f64, starttime = 161 : i32, tl_channel = 0 : i32, transaction_group = 9 : i64, transaction_id = 22 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<16xi32>, memref<16xi32>), i32, i32, i32
      %27 = tor.addi %1 %c64_i32 on (115 to 116) {dump = "unknown_278", ref_endtime = 116 : i32, ref_starttime = 115 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%5, %6)[%c16_i32], %27, %c16_i32 {after = array<i64: 22, 21, 22>, after_gap = array<i64: 1, 19, 1>, after_segment = 0 : i64, endtime = 180 : i32, estimated_latency = 19 : i64, interface_cost = 35 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 21 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 51 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 180 : i32, ref_starttime = 162 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 9.900000e+01 : f64, starttime = 162 : i32, tl_channel = 1 : i32, transaction_group = 9 : i64, transaction_id = 23 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<16xi32>, memref<16xi32>), i32, i32, i32
      aps.copy_by @busitfc, (%21, %22)[%c0_i32], %20, %c8_i32 {after = array<i64: 21, 23, 23, 22, 23>, after_gap = array<i64: 19, 19, 19, 10, 19>, endtime = 186 : i32, estimated_latency = 5 : i64, interface_cost = 13 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 45 : i64, memory_issue_cycle = 40 : i64, memory_latency = 5 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 51 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 186 : i32, ref_starttime = 181 : i32, required_alignment = 32 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 9.900000e+01 : f64, starttime = 181 : i32, tl_channel = 0 : i32, transaction_id = 24 : i64, transfer_bytes = 32 : i64, transfer_offset = 0 : i64} : (memref<4xf32>, memref<4xf32>), i32, i32, i32
      aps.copy_by @busitfc, (%24, %25)[%c0_i32], %23, %c8_i32 {after = array<i64: 21, 23, 24, 24, 23, 24>, after_gap = array<i64: 19, 19, 6, 6, 19, 6>, endtime = 192 : i32, estimated_latency = 5 : i64, interface_cost = 13 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 51 : i64, memory_issue_cycle = 46 : i64, memory_latency = 5 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 51 : i64, memory_search_hits = 0 : i64, memory_search_states = 11 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 192 : i32, ref_starttime = 187 : i32, required_alignment = 32 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 9.900000e+01 : f64, starttime = 187 : i32, tl_channel = 1 : i32, transaction_id = 25 : i64, transfer_bytes = 32 : i64, transfer_offset = 0 : i64} : (memref<4xi32>, memref<4xi32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_24__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 2 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-knn-32/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_279", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = memref.get_global @au_generated_candidate_24__l0u_buffer_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %2 = memref.get_global @au_generated_candidate_24__l0u_buffer_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%1, %2)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 12 : i64, transaction_id = 26 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %3 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_280", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %3, (%1, %2)[%c16_i32], %c16_i32 {after = array<i64: 26, 26>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 20 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 20 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 12 : i64, transaction_id = 27 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %4 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_281", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%1, %2)[%c32_i32], %c16_i32 {after = array<i64: 27, 26, 27>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 30 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 30 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 12 : i64, transaction_id = 28 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %5 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_282", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%1, %2)[%c48_i32], %c16_i32 {after = array<i64: 28, 27, 28>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 38 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 38 : i32, ref_starttime = 23 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 23 : i32, tl_channel = 1 : i32, transaction_group = 12 : i64, transaction_id = 29 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %6 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_283", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%1, %2)[%c64_i32], %c16_i32 {after = array<i64: 29, 28, 29>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 46 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 46 : i32, ref_starttime = 31 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 31 : i32, tl_channel = 0 : i32, transaction_group = 12 : i64, transaction_id = 30 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %7 = tor.addi %0 %c320_i32 on (0 to 1) {dump = "unknown_284", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%1, %2)[%c80_i32], %c16_i32 {after = array<i64: 30, 29, 30>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 54 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 54 : i32, ref_starttime = 39 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 1.000000e+02 : f64, starttime = 39 : i32, tl_channel = 1 : i32, transaction_group = 12 : i64, transaction_id = 31 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      aps.write_smem %cst_5, %1[%c0_i32] {aps.mem_id = 95 : i64, endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, slot = 0 : i32, starttime = 54 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_6, %2[%c0_i32] {aps.mem_id = 94 : i64, endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, slot = 0 : i32, starttime = 54 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_7, %1[%c1_i32] {aps.mem_id = 93 : i64, endtime = 56 : i32, ref_endtime = 56 : i32, ref_starttime = 55 : i32, slot = 0 : i32, starttime = 55 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_8, %2[%c1_i32] {aps.mem_id = 92 : i64, endtime = 56 : i32, ref_endtime = 56 : i32, ref_starttime = 55 : i32, slot = 0 : i32, starttime = 55 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_9, %1[%c2_i32] {aps.mem_id = 91 : i64, endtime = 57 : i32, ref_endtime = 57 : i32, ref_starttime = 56 : i32, slot = 0 : i32, starttime = 56 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_10, %2[%c2_i32] {aps.mem_id = 90 : i64, endtime = 57 : i32, ref_endtime = 57 : i32, ref_starttime = 56 : i32, slot = 0 : i32, starttime = 56 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_11, %1[%c3_i32] {aps.mem_id = 89 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_12, %2[%c3_i32] {aps.mem_id = 88 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_13, %1[%c4_i32] {aps.mem_id = 87 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_14, %2[%c4_i32] {aps.mem_id = 86 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_15, %1[%c5_i32] {aps.mem_id = 85 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_16, %2[%c5_i32] {aps.mem_id = 84 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_17, %1[%c6_i32] {aps.mem_id = 83 : i64, endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, slot = 0 : i32, starttime = 60 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_18, %2[%c6_i32] {aps.mem_id = 82 : i64, endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, slot = 0 : i32, starttime = 60 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_19, %1[%c7_i32] {aps.mem_id = 81 : i64, endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, slot = 0 : i32, starttime = 61 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_20, %2[%c7_i32] {aps.mem_id = 80 : i64, endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, slot = 0 : i32, starttime = 61 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_21, %1[%c8_i32] {aps.mem_id = 79 : i64, endtime = 63 : i32, ref_endtime = 63 : i32, ref_starttime = 62 : i32, slot = 0 : i32, starttime = 62 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_22, %2[%c8_i32] {aps.mem_id = 78 : i64, endtime = 63 : i32, ref_endtime = 63 : i32, ref_starttime = 62 : i32, slot = 0 : i32, starttime = 62 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_23, %1[%c9_i32] {aps.mem_id = 77 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_24, %2[%c9_i32] {aps.mem_id = 76 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_25, %1[%c10_i32] {aps.mem_id = 75 : i64, endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, slot = 0 : i32, starttime = 64 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_26, %2[%c10_i32] {aps.mem_id = 74 : i64, endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, slot = 0 : i32, starttime = 64 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_27, %1[%c11_i32] {aps.mem_id = 73 : i64, endtime = 66 : i32, ref_endtime = 66 : i32, ref_starttime = 65 : i32, slot = 0 : i32, starttime = 65 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_28, %2[%c11_i32] {aps.mem_id = 72 : i64, endtime = 66 : i32, ref_endtime = 66 : i32, ref_starttime = 65 : i32, slot = 0 : i32, starttime = 65 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_29, %1[%c12_i32] {aps.mem_id = 71 : i64, endtime = 67 : i32, ref_endtime = 67 : i32, ref_starttime = 66 : i32, slot = 0 : i32, starttime = 66 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_30, %2[%c12_i32] {aps.mem_id = 70 : i64, endtime = 67 : i32, ref_endtime = 67 : i32, ref_starttime = 66 : i32, slot = 0 : i32, starttime = 66 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_31, %1[%c13_i32] {aps.mem_id = 69 : i64, endtime = 68 : i32, ref_endtime = 68 : i32, ref_starttime = 67 : i32, slot = 0 : i32, starttime = 67 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_32, %2[%c13_i32] {aps.mem_id = 68 : i64, endtime = 68 : i32, ref_endtime = 68 : i32, ref_starttime = 67 : i32, slot = 0 : i32, starttime = 67 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_33, %1[%c14_i32] {aps.mem_id = 67 : i64, endtime = 69 : i32, ref_endtime = 69 : i32, ref_starttime = 68 : i32, slot = 0 : i32, starttime = 68 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_34, %2[%c14_i32] {aps.mem_id = 66 : i64, endtime = 69 : i32, ref_endtime = 69 : i32, ref_starttime = 68 : i32, slot = 0 : i32, starttime = 68 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_35, %1[%c15_i32] {aps.mem_id = 65 : i64, endtime = 70 : i32, ref_endtime = 70 : i32, ref_starttime = 69 : i32, slot = 0 : i32, starttime = 69 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_36, %2[%c15_i32] {aps.mem_id = 64 : i64, endtime = 70 : i32, ref_endtime = 70 : i32, ref_starttime = 69 : i32, slot = 0 : i32, starttime = 69 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_37, %1[%c16_i32] {aps.mem_id = 63 : i64, endtime = 71 : i32, ref_endtime = 71 : i32, ref_starttime = 70 : i32, slot = 0 : i32, starttime = 70 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_38, %2[%c16_i32] {aps.mem_id = 62 : i64, endtime = 71 : i32, ref_endtime = 71 : i32, ref_starttime = 70 : i32, slot = 0 : i32, starttime = 70 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_39, %1[%c17_i32] {aps.mem_id = 61 : i64, endtime = 72 : i32, ref_endtime = 72 : i32, ref_starttime = 71 : i32, slot = 0 : i32, starttime = 71 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_40, %2[%c17_i32] {aps.mem_id = 60 : i64, endtime = 72 : i32, ref_endtime = 72 : i32, ref_starttime = 71 : i32, slot = 0 : i32, starttime = 71 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_41, %1[%c18_i32] {aps.mem_id = 59 : i64, endtime = 73 : i32, ref_endtime = 73 : i32, ref_starttime = 72 : i32, slot = 0 : i32, starttime = 72 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_42, %2[%c18_i32] {aps.mem_id = 58 : i64, endtime = 73 : i32, ref_endtime = 73 : i32, ref_starttime = 72 : i32, slot = 0 : i32, starttime = 72 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_43, %1[%c19_i32] {aps.mem_id = 57 : i64, endtime = 74 : i32, ref_endtime = 74 : i32, ref_starttime = 73 : i32, slot = 0 : i32, starttime = 73 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_44, %2[%c19_i32] {aps.mem_id = 56 : i64, endtime = 74 : i32, ref_endtime = 74 : i32, ref_starttime = 73 : i32, slot = 0 : i32, starttime = 73 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_45, %1[%c20_i32] {aps.mem_id = 55 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_46, %2[%c20_i32] {aps.mem_id = 54 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_47, %1[%c21_i32] {aps.mem_id = 53 : i64, endtime = 76 : i32, ref_endtime = 76 : i32, ref_starttime = 75 : i32, slot = 0 : i32, starttime = 75 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_48, %2[%c21_i32] {aps.mem_id = 52 : i64, endtime = 76 : i32, ref_endtime = 76 : i32, ref_starttime = 75 : i32, slot = 0 : i32, starttime = 75 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_49, %1[%c22_i32] {aps.mem_id = 51 : i64, endtime = 77 : i32, ref_endtime = 77 : i32, ref_starttime = 76 : i32, slot = 0 : i32, starttime = 76 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_50, %2[%c22_i32] {aps.mem_id = 50 : i64, endtime = 77 : i32, ref_endtime = 77 : i32, ref_starttime = 76 : i32, slot = 0 : i32, starttime = 76 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_51, %1[%c23_i32] {aps.mem_id = 49 : i64, endtime = 78 : i32, ref_endtime = 78 : i32, ref_starttime = 77 : i32, slot = 0 : i32, starttime = 77 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_52, %2[%c23_i32] {aps.mem_id = 48 : i64, endtime = 78 : i32, ref_endtime = 78 : i32, ref_starttime = 77 : i32, slot = 0 : i32, starttime = 77 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_53, %1[%c24_i32] {aps.mem_id = 47 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_54, %2[%c24_i32] {aps.mem_id = 46 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_55, %1[%c25_i32] {aps.mem_id = 45 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_56, %2[%c25_i32] {aps.mem_id = 44 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_57, %1[%c26_i32] {aps.mem_id = 43 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_58, %2[%c26_i32] {aps.mem_id = 42 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_59, %1[%c27_i32] {aps.mem_id = 41 : i64, endtime = 82 : i32, ref_endtime = 82 : i32, ref_starttime = 81 : i32, slot = 0 : i32, starttime = 81 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_60, %2[%c27_i32] {aps.mem_id = 40 : i64, endtime = 82 : i32, ref_endtime = 82 : i32, ref_starttime = 81 : i32, slot = 0 : i32, starttime = 81 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_61, %1[%c28_i32] {aps.mem_id = 39 : i64, endtime = 83 : i32, ref_endtime = 83 : i32, ref_starttime = 82 : i32, slot = 0 : i32, starttime = 82 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_62, %2[%c28_i32] {aps.mem_id = 38 : i64, endtime = 83 : i32, ref_endtime = 83 : i32, ref_starttime = 82 : i32, slot = 0 : i32, starttime = 82 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_63, %1[%c29_i32] {aps.mem_id = 37 : i64, endtime = 84 : i32, ref_endtime = 84 : i32, ref_starttime = 83 : i32, slot = 0 : i32, starttime = 83 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_64, %2[%c29_i32] {aps.mem_id = 36 : i64, endtime = 84 : i32, ref_endtime = 84 : i32, ref_starttime = 83 : i32, slot = 0 : i32, starttime = 83 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_65, %1[%c30_i32] {aps.mem_id = 35 : i64, endtime = 85 : i32, ref_endtime = 85 : i32, ref_starttime = 84 : i32, slot = 0 : i32, starttime = 84 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_66, %2[%c30_i32] {aps.mem_id = 34 : i64, endtime = 85 : i32, ref_endtime = 85 : i32, ref_starttime = 84 : i32, slot = 0 : i32, starttime = 84 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_67, %1[%c31_i32] {aps.mem_id = 33 : i64, endtime = 86 : i32, ref_endtime = 86 : i32, ref_starttime = 85 : i32, slot = 0 : i32, starttime = 85 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_68, %2[%c31_i32] {aps.mem_id = 32 : i64, endtime = 86 : i32, ref_endtime = 86 : i32, ref_starttime = 85 : i32, slot = 0 : i32, starttime = 85 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_69, %1[%c32_i32] {aps.mem_id = 31 : i64, endtime = 87 : i32, ref_endtime = 87 : i32, ref_starttime = 86 : i32, slot = 0 : i32, starttime = 86 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_70, %2[%c32_i32] {aps.mem_id = 30 : i64, endtime = 87 : i32, ref_endtime = 87 : i32, ref_starttime = 86 : i32, slot = 0 : i32, starttime = 86 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_71, %1[%c33_i32] {aps.mem_id = 29 : i64, endtime = 88 : i32, ref_endtime = 88 : i32, ref_starttime = 87 : i32, slot = 0 : i32, starttime = 87 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_72, %2[%c33_i32] {aps.mem_id = 28 : i64, endtime = 88 : i32, ref_endtime = 88 : i32, ref_starttime = 87 : i32, slot = 0 : i32, starttime = 87 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_73, %1[%c34_i32] {aps.mem_id = 27 : i64, endtime = 89 : i32, ref_endtime = 89 : i32, ref_starttime = 88 : i32, slot = 0 : i32, starttime = 88 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_74, %2[%c34_i32] {aps.mem_id = 26 : i64, endtime = 89 : i32, ref_endtime = 89 : i32, ref_starttime = 88 : i32, slot = 0 : i32, starttime = 88 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_75, %1[%c35_i32] {aps.mem_id = 25 : i64, endtime = 90 : i32, ref_endtime = 90 : i32, ref_starttime = 89 : i32, slot = 0 : i32, starttime = 89 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_76, %2[%c35_i32] {aps.mem_id = 24 : i64, endtime = 90 : i32, ref_endtime = 90 : i32, ref_starttime = 89 : i32, slot = 0 : i32, starttime = 89 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_77, %1[%c36_i32] {aps.mem_id = 23 : i64, endtime = 91 : i32, ref_endtime = 91 : i32, ref_starttime = 90 : i32, slot = 0 : i32, starttime = 90 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_78, %2[%c36_i32] {aps.mem_id = 22 : i64, endtime = 91 : i32, ref_endtime = 91 : i32, ref_starttime = 90 : i32, slot = 0 : i32, starttime = 90 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_79, %1[%c37_i32] {aps.mem_id = 21 : i64, endtime = 92 : i32, ref_endtime = 92 : i32, ref_starttime = 91 : i32, slot = 0 : i32, starttime = 91 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_80, %2[%c37_i32] {aps.mem_id = 20 : i64, endtime = 92 : i32, ref_endtime = 92 : i32, ref_starttime = 91 : i32, slot = 0 : i32, starttime = 91 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_81, %1[%c38_i32] {aps.mem_id = 19 : i64, endtime = 93 : i32, ref_endtime = 93 : i32, ref_starttime = 92 : i32, slot = 0 : i32, starttime = 92 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_82, %2[%c38_i32] {aps.mem_id = 18 : i64, endtime = 93 : i32, ref_endtime = 93 : i32, ref_starttime = 92 : i32, slot = 0 : i32, starttime = 92 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_83, %1[%c39_i32] {aps.mem_id = 17 : i64, endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, slot = 0 : i32, starttime = 93 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_84, %2[%c39_i32] {aps.mem_id = 16 : i64, endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, slot = 0 : i32, starttime = 93 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_85, %1[%c40_i32] {aps.mem_id = 15 : i64, endtime = 95 : i32, ref_endtime = 95 : i32, ref_starttime = 94 : i32, slot = 0 : i32, starttime = 94 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_86, %2[%c40_i32] {aps.mem_id = 14 : i64, endtime = 95 : i32, ref_endtime = 95 : i32, ref_starttime = 94 : i32, slot = 0 : i32, starttime = 94 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_87, %1[%c41_i32] {aps.mem_id = 13 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_88, %2[%c41_i32] {aps.mem_id = 12 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_89, %1[%c42_i32] {aps.mem_id = 11 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_83, %2[%c42_i32] {aps.mem_id = 10 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_90, %1[%c43_i32] {aps.mem_id = 9 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_91, %2[%c43_i32] {aps.mem_id = 8 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_92, %1[%c44_i32] {aps.mem_id = 7 : i64, endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, slot = 0 : i32, starttime = 98 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_93, %2[%c44_i32] {aps.mem_id = 6 : i64, endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, slot = 0 : i32, starttime = 98 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_94, %1[%c45_i32] {aps.mem_id = 5 : i64, endtime = 100 : i32, ref_endtime = 100 : i32, ref_starttime = 99 : i32, slot = 0 : i32, starttime = 99 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_95, %2[%c45_i32] {aps.mem_id = 4 : i64, endtime = 100 : i32, ref_endtime = 100 : i32, ref_starttime = 99 : i32, slot = 0 : i32, starttime = 99 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_96, %1[%c46_i32] {aps.mem_id = 3 : i64, endtime = 101 : i32, ref_endtime = 101 : i32, ref_starttime = 100 : i32, slot = 0 : i32, starttime = 100 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_97, %2[%c46_i32] {aps.mem_id = 2 : i64, endtime = 101 : i32, ref_endtime = 101 : i32, ref_starttime = 100 : i32, slot = 0 : i32, starttime = 100 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_98, %1[%c47_i32] {aps.mem_id = 1 : i64, endtime = 102 : i32, ref_endtime = 102 : i32, ref_starttime = 101 : i32, slot = 0 : i32, starttime = 101 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_74, %2[%c47_i32] {aps.mem_id = 0 : i64, endtime = 102 : i32, ref_endtime = 102 : i32, ref_starttime = 101 : i32, slot = 0 : i32, starttime = 101 : i32} : f32, memref<48xf32>, i32
      aps.copy_by @busitfc, (%1, %2)[%c0_i32], %0, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 111 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 111 : i32, ref_starttime = 102 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 102 : i32, tl_channel = 0 : i32, transaction_group = 13 : i64, transaction_id = 32 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %8 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_285", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c16_i32], %8, %c16_i32 {after = array<i64: 32, 32>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 121 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 121 : i32, ref_starttime = 103 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 103 : i32, tl_channel = 1 : i32, transaction_group = 13 : i64, transaction_id = 33 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %9 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_286", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c32_i32], %9, %c16_i32 {after = array<i64: 33, 32, 33>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 136 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 136 : i32, ref_starttime = 117 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 117 : i32, tl_channel = 0 : i32, transaction_group = 13 : i64, transaction_id = 34 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %10 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_287", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c48_i32], %10, %c16_i32 {after = array<i64: 34, 33, 34>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 146 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 20 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 146 : i32, ref_starttime = 127 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 127 : i32, tl_channel = 1 : i32, transaction_group = 13 : i64, transaction_id = 35 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %11 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_288", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c64_i32], %11, %c16_i32 {after = array<i64: 35, 34, 35>, after_gap = array<i64: 10, 20, 10>, after_segment = 3 : i64, endtime = 156 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 49 : i64, memory_issue_cycle = 30 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 156 : i32, ref_starttime = 137 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 137 : i32, tl_channel = 0 : i32, transaction_group = 13 : i64, transaction_id = 36 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %12 = tor.addi %0 %c320_i32 on (0 to 1) {dump = "unknown_289", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c80_i32], %12, %c16_i32 {after = array<i64: 36, 35, 36>, after_gap = array<i64: 10, 20, 10>, after_segment = 4 : i64, endtime = 166 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 59 : i64, memory_issue_cycle = 40 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 59 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 166 : i32, ref_starttime = 147 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 1.070000e+02 : f64, starttime = 147 : i32, tl_channel = 1 : i32, transaction_group = 13 : i64, transaction_id = 37 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      tor.return
    }
    tor.func @au_generated_candidate_25__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 3 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-knn-32/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_290", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = memref.get_global @au_generated_candidate_25__l0u_buffer_0 : memref<6xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %2 = memref.get_global @au_generated_candidate_25__l0u_buffer_1 : memref<6xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%1, %2)[%c0_i32], %c8_i32 {after = array<i64>, after_gap = array<i64>, endtime = 8 : i32, estimated_latency = 10 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 8 : i64, memory_issue_cycle = 0 : i64, memory_latency = 8 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 10 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 8 : i32, ref_starttime = 0 : i32, required_alignment = 32 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 1.850000e+01 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 14 : i64, transaction_id = 38 : i64, transfer_bytes = 32 : i64, transfer_offset = 0 : i64} : i32, (memref<6xf32>, memref<6xf32>), i32, i32
      %3 = tor.addi %0 %c32_i32 on (0 to 1) {dump = "unknown_291", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %3, (%1, %2)[%c8_i32], %c4_i32 {after = array<i64: 38, 38>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 10 : i32, estimated_latency = 10 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 10 : i64, memory_issue_cycle = 1 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 10 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 10 : i32, ref_starttime = 1 : i32, required_alignment = 16 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 1.850000e+01 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 14 : i64, transaction_id = 39 : i64, transfer_bytes = 16 : i64, transfer_offset = 32 : i64} : i32, (memref<6xf32>, memref<6xf32>), i32, i32
      aps.write_smem %cst, %1[%c0_i32] {aps.mem_id = 11 : i64, endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, slot = 0 : i32, starttime = 10 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_92, %2[%c0_i32] {aps.mem_id = 10 : i64, endtime = 11 : i32, ref_endtime = 11 : i32, ref_starttime = 10 : i32, slot = 0 : i32, starttime = 10 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_0, %1[%c1_i32] {aps.mem_id = 9 : i64, endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, slot = 0 : i32, starttime = 11 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_68, %2[%c1_i32] {aps.mem_id = 8 : i64, endtime = 12 : i32, ref_endtime = 12 : i32, ref_starttime = 11 : i32, slot = 0 : i32, starttime = 11 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_81, %1[%c2_i32] {aps.mem_id = 7 : i64, endtime = 13 : i32, ref_endtime = 13 : i32, ref_starttime = 12 : i32, slot = 0 : i32, starttime = 12 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_1, %2[%c2_i32] {aps.mem_id = 6 : i64, endtime = 13 : i32, ref_endtime = 13 : i32, ref_starttime = 12 : i32, slot = 0 : i32, starttime = 12 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_2, %1[%c3_i32] {aps.mem_id = 5 : i64, endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, slot = 0 : i32, starttime = 13 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_69, %2[%c3_i32] {aps.mem_id = 4 : i64, endtime = 14 : i32, ref_endtime = 14 : i32, ref_starttime = 13 : i32, slot = 0 : i32, starttime = 13 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_3, %1[%c4_i32] {aps.mem_id = 3 : i64, endtime = 15 : i32, ref_endtime = 15 : i32, ref_starttime = 14 : i32, slot = 0 : i32, starttime = 14 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_53, %2[%c4_i32] {aps.mem_id = 2 : i64, endtime = 15 : i32, ref_endtime = 15 : i32, ref_starttime = 14 : i32, slot = 0 : i32, starttime = 14 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_57, %1[%c5_i32] {aps.mem_id = 1 : i64, endtime = 16 : i32, ref_endtime = 16 : i32, ref_starttime = 15 : i32, slot = 0 : i32, starttime = 15 : i32} : f32, memref<6xf32>, i32
      aps.write_smem %cst_4, %2[%c5_i32] {aps.mem_id = 0 : i64, endtime = 16 : i32, ref_endtime = 16 : i32, ref_starttime = 15 : i32, slot = 0 : i32, starttime = 15 : i32} : f32, memref<6xf32>, i32
      aps.copy_by @busitfc, (%1, %2)[%c0_i32], %0, %c8_i32 {after = array<i64>, after_gap = array<i64>, endtime = 21 : i32, estimated_latency = 9 : i64, interface_cost = 17 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 5 : i64, memory_issue_cycle = 0 : i64, memory_latency = 5 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 9 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 21 : i32, ref_starttime = 16 : i32, required_alignment = 32 : i64, segment = 0 : i64, segment_count = 2 : i64, solver_objective = 1.700000e+01 : f64, starttime = 16 : i32, tl_channel = 0 : i32, transaction_group = 15 : i64, transaction_id = 40 : i64, transfer_bytes = 32 : i64, transfer_offset = 0 : i64} : (memref<6xf32>, memref<6xf32>), i32, i32, i32
      %4 = tor.addi %0 %c32_i32 on (0 to 1) {dump = "unknown_292", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%1, %2)[%c8_i32], %4, %c4_i32 {after = array<i64: 40, 40>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 25 : i32, estimated_latency = 9 : i64, interface_cost = 17 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 1 : i64, memory_latency = 8 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 9 : i64, memory_search_hits = 0 : i64, memory_search_states = 6 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 25 : i32, ref_starttime = 17 : i32, required_alignment = 16 : i64, segment = 1 : i64, segment_count = 2 : i64, solver_objective = 1.700000e+01 : f64, starttime = 17 : i32, tl_channel = 1 : i32, transaction_group = 15 : i64, transaction_id = 41 : i64, transfer_bytes = 16 : i64, transfer_offset = 32 : i64} : (memref<6xf32>, memref<6xf32>), i32, i32, i32
      tor.return
    }
  } {HoistConstCondIfOp = 1 : i32, schedule = true}
}

