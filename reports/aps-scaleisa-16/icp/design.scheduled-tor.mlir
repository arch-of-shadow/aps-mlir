module {
  aps.memitfc @busitfc {cache_hint = "cold", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 8 : i64, min_beats = 1 : i64, min_write_beats = 2 : i64, read_latency = 5 : i64, width = 8 : i64, write_completion = 2 : i64}
  aps.memitfc @cpuitfc {cache_hint = "warm", cache_line = 64 : i64, inflight = 2 : i64, max_beats = 1 : i64, min_beats = 1 : i64, min_write_beats = 1 : i64, read_latency = 2 : i64, width = 4 : i64, write_completion = 2 : i64}
  aps.memorymap {
    aps.mem_entry "au_generated_candidate_1__direct_buffer" : banks([@au_generated_candidate_1__direct_buffer]), base(0), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_1__direct_buffer_1" : banks([@au_generated_candidate_1__direct_buffer_1]), base(4), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_1__direct_buffer_2" : banks([@au_generated_candidate_1__direct_buffer_2]), base(8), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_1__direct_buffer_3" : banks([@au_generated_candidate_1__direct_buffer_3_0, @au_generated_candidate_1__direct_buffer_3_1]), base(12), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_1__direct_buffer_4" : banks([@au_generated_candidate_1__direct_buffer_4_0, @au_generated_candidate_1__direct_buffer_4_1]), base(512), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_1__direct_buffer_5" : banks([@au_generated_candidate_1__direct_buffer_5]), base(1024), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_1__direct_buffer_6" : banks([@au_generated_candidate_1__direct_buffer_6]), base(1028), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_1__direct_buffer_7" : banks([@au_generated_candidate_1__direct_buffer_7]), base(1032), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_1__direct_buffer_8" : banks([@au_generated_candidate_1__direct_buffer_8]), base(1036), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_1__direct_local" : banks([@au_generated_candidate_1__direct_local]), base(1040), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_1__direct_local_0" : banks([@au_generated_candidate_1__direct_local_0]), base(1044), size(4), count(1), cyclic(0)
    aps.mem_entry "au_generated_candidate_8__l0u_buffer" : banks([@au_generated_candidate_8__l0u_buffer_0, @au_generated_candidate_8__l0u_buffer_1]), base(1048), size(384), count(2), cyclic(1)
    aps.mem_entry "au_generated_candidate_8__l0u_buffer_9" : banks([@au_generated_candidate_8__l0u_buffer_9_0, @au_generated_candidate_8__l0u_buffer_9_1]), base(1536), size(384), count(2), cyclic(1)
    aps.mem_finish
  }
  tor.design @aps_isaxes {
    %c31_i6 = arith.constant 31 : i6
    %c1_i6 = arith.constant 1 : i6
    %c0_i6 = arith.constant 0 : i6
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e-02 : f32
    %cst_1 = arith.constant -4.000000e-02 : f32
    %cst_2 = arith.constant 6.000000e-02 : f32
    %cst_3 = arith.constant 2.100000e-01 : f32
    %cst_4 = arith.constant 2.750000e-01 : f32
    %cst_5 = arith.constant 2.600000e-01 : f32
    %cst_6 = arith.constant 2.900000e-01 : f32
    %cst_7 = arith.constant 0.235000014 : f32
    %cst_8 = arith.constant 3.200000e-01 : f32
    %cst_9 = arith.constant 4.200000e-01 : f32
    %cst_10 = arith.constant 5.500000e-01 : f32
    %cst_11 = arith.constant 5.200000e-01 : f32
    %cst_12 = arith.constant 5.000000e-01 : f32
    %cst_13 = arith.constant 5.100000e-01 : f32
    %cst_14 = arith.constant 5.800000e-01 : f32
    %cst_15 = arith.constant 6.300000e-01 : f32
    %cst_16 = arith.constant 0.824999988 : f32
    %cst_17 = arith.constant 1.600000e-01 : f32
    %cst_18 = arith.constant 0.709999978 : f32
    %cst_19 = arith.constant 0.784999966 : f32
    %cst_20 = arith.constant 2.200000e-01 : f32
    %cst_21 = arith.constant 0.839999973 : f32
    %cst_22 = arith.constant 1.750000e-01 : f32
    %cst_23 = arith.constant 0.919999957 : f32
    %cst_24 = arith.constant 0.13499999 : f32
    %cst_25 = arith.constant 4.800000e-01 : f32
    %cst_26 = arith.constant 1.050000e+00 : f32
    %cst_27 = arith.constant 0.450000018 : f32
    %cst_28 = arith.constant 1.130000e+00 : f32
    %cst_29 = arith.constant 0.410000026 : f32
    %cst_30 = arith.constant 1.200000e-01 : f32
    %cst_31 = arith.constant 3.000000e-02 : f32
    %cst_32 = arith.constant 7.250000e-01 : f32
    %cst_33 = arith.constant 1.100000e-01 : f32
    %cst_34 = arith.constant 6.850000e-01 : f32
    %cst_35 = arith.constant 3.800000e-01 : f32
    %cst_36 = arith.constant 2.400000e-01 : f32
    %cst_37 = arith.constant 7.500000e-02 : f32
    %cst_38 = arith.constant 0.0350000039 : f32
    %cst_39 = arith.constant 6.400000e-01 : f32
    %cst_40 = arith.constant 4.500000e-01 : f32
    %cst_41 = arith.constant 3.500000e-01 : f32
    %cst_42 = arith.constant 5.300000e-01 : f32
    %cst_43 = arith.constant 3.100000e-01 : f32
    %cst_44 = arith.constant 2.800000e-01 : f32
    %cst_45 = arith.constant 0.659999967 : f32
    %cst_46 = arith.constant 6.250000e-01 : f32
    %cst_47 = arith.constant 0.739999949 : f32
    %cst_48 = arith.constant 5.850000e-01 : f32
    %cst_49 = arith.constant 0.539999962 : f32
    %cst_50 = arith.constant 8.700000e-01 : f32
    %cst_51 = arith.constant 0.900000035 : f32
    %cst_52 = arith.constant 0.949999988 : f32
    %cst_53 = arith.constant 8.600000e-01 : f32
    %cst_54 = arith.constant 0.179999992 : f32
    %cst_55 = arith.constant 1.07999992 : f32
    %cst_56 = arith.constant 2.500000e-01 : f32
    %cst_57 = arith.constant 1.160000e+00 : f32
    %cst_58 = arith.constant 0.210000008 : f32
    %cst_59 = arith.constant 4.400000e-01 : f32
    %cst_60 = arith.constant 0.525000036 : f32
    %cst_61 = arith.constant 2.000000e-02 : f32
    %cst_62 = arith.constant 1.400000e-01 : f32
    %cst_63 = arith.constant 0.485000044 : f32
    %cst_64 = arith.constant 0.269999981 : f32
    %cst_65 = arith.constant 8.000000e-01 : f32
    %cst_66 = arith.constant 0.349999964 : f32
    %cst_67 = arith.constant 7.600000e-01 : f32
    %cst_68 = arith.constant 3.400000e-01 : f32
    %cst_69 = arith.constant 1.500000e-01 : f32
    %cst_70 = arith.constant 5.600000e-01 : f32
    %cst_71 = arith.constant 0.110000007 : f32
    %cst_72 = arith.constant 0.599999964 : f32
    %cst_73 = arith.constant 0.689999997 : f32
    %cst_74 = arith.constant 4.250000e-01 : f32
    %cst_75 = arith.constant 0.76999998 : f32
    %cst_76 = arith.constant 0.38500002 : f32
    %cst_77 = arith.constant 0.899999976 : f32
    %cst_78 = arith.constant 0.699999988 : f32
    %cst_79 = arith.constant 0.979999959 : f32
    %cst_80 = arith.constant 1.110000e+00 : f32
    %cst_81 = arith.constant 5.000000e-02 : f32
    %cst_82 = arith.constant 1.190000e+00 : f32
    %cst_83 = arith.constant 0.0100000016 : f32
    %cst_84 = arith.constant 0.0899999961 : f32
    %cst_85 = arith.constant 0.325000018 : f32
    %cst_86 = arith.constant 0.169999987 : f32
    %cst_87 = arith.constant 0.285000026 : f32
    %cst_88 = arith.constant 4.000000e-01 : f32
    %cst_89 = arith.constant 0.299999982 : f32
    %cst_90 = arith.constant 6.000000e-01 : f32
    %cst_91 = arith.constant 8.750000e-01 : f32
    %cst_92 = arith.constant 5.900000e-01 : f32
    %cst_93 = arith.constant 0.834999978 : f32
    %cst_94 = arith.constant 0.719999969 : f32
    %cst_95 = arith.constant 0.225000009 : f32
    %cst_96 = arith.constant 0.799999952 : f32
    %cst_97 = arith.constant 1.850000e-01 : f32
    %cst_98 = arith.constant 9.300000e-01 : f32
    %cst_99 = arith.constant 1.010000e+00 : f32
    %cst_100 = arith.constant 4.600000e-01 : f32
    %cst_101 = arith.constant 2.000000e-01 : f32
    %cst_102 = arith.constant 1.140000e+00 : f32
    %cst_103 = arith.constant 0.775000035 : f32
    %cst_104 = arith.constant 0.399999976 : f32
    %cst_105 = arith.constant 1.220000e+00 : f32
    %cst_106 = arith.constant 7.350000e-01 : f32
    %cst_107 = arith.constant 0.459999979 : f32
    %cst_108 = arith.constant 1.250000e-01 : f32
    %cst_109 = arith.constant 4.000000e-02 : f32
    %cst_110 = arith.constant 0.199999988 : f32
    %cst_111 = arith.constant 8.500000e-02 : f32
    %cst_112 = arith.constant 0.099999994 : f32
    %cst_113 = arith.constant 0.329999983 : f32
    %cst_114 = arith.constant 0.409999967 : f32
    %cst_115 = arith.constant 3.600000e-01 : f32
    %cst_116 = arith.constant 0.359999985 : f32
    %cst_117 = arith.constant 6.750000e-01 : f32
    %cst_118 = arith.constant 0.619999945 : f32
    %cst_119 = arith.constant 6.350000e-01 : f32
    %cst_120 = arith.constant 6.200000e-01 : f32
    %cst_121 = arith.constant 7.500000e-01 : f32
    %cst_122 = arith.constant 2.500000e-02 : f32
    %cst_123 = arith.constant 0.829999983 : f32
    %cst_124 = arith.constant -0.0149999987 : f32
    %cst_125 = arith.constant 0.959999978 : f32
    %cst_126 = arith.constant 3.000000e-01 : f32
    %cst_127 = arith.constant 1.040000e+00 : f32
    %cst_128 = arith.constant 0.26000002 : f32
    %cst_129 = arith.constant 1.170000e+00 : f32
    %cst_130 = arith.constant 5.750000e-01 : f32
    %cst_131 = arith.constant 1.250000e+00 : f32
    %cst_132 = arith.constant 0.534999967 : f32
    %cst_133 = arith.constant 0.149999991 : f32
    %cst_134 = arith.constant 8.500000e-01 : f32
    %cst_135 = arith.constant 0.229999989 : f32
    %cst_136 = arith.constant 8.100000e-01 : f32
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
    %c2_i32 = arith.constant 2 : i32
    %cst_137 = arith.constant 1.000000e+30 : f32
    %c3_i32 = arith.constant 3 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c192_i32 = arith.constant 192 : i32
    %c256_i32 = arith.constant 256 : i32
    %c320_i32 = arith.constant 320 : i32
    %c32_i32 = arith.constant 32 : i32
    %c48_i32 = arith.constant 48 : i32
    %c80_i32 = arith.constant 80 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c-2_i32 = arith.constant -2 : i32
    memref.global @au_generated_candidate_8__l0u_buffer_9_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_8__l0u_buffer_9_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_8__l0u_buffer_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_8__l0u_buffer_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_1__direct_buffer_8 : memref<1xf32> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_8"}
    memref.global @au_generated_candidate_1__direct_buffer_7 : memref<1xf32> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_7"}
    memref.global @au_generated_candidate_1__direct_buffer_6 : memref<1xf32> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_6"}
    memref.global @au_generated_candidate_1__direct_buffer_5 : memref<1xf32> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_5"}
    memref.global @au_generated_candidate_1__direct_buffer_4_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_1__direct_buffer_4_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_1__direct_buffer_3_0 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_1__direct_buffer_3_1 : memref<48xf32> = uninitialized
    memref.global @au_generated_candidate_1__direct_buffer_2 : memref<1xf32> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_2"}
    memref.global @au_generated_candidate_1__direct_buffer_1 : memref<1xf32> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_1"}
    memref.global @au_generated_candidate_1__direct_buffer : memref<1xf32> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer"}
    memref.global @au_generated_candidate_1__direct_local_0 : memref<i32> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_local_0"}
    memref.global @au_generated_candidate_1__direct_local : memref<f32> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_local"}
    memref.global "private" @au_generated_candidate_1__direct_csr0 : memref<i32> = uninitialized {csr_address = 2049 : i32}
    memref.global "private" @au_generated_candidate_1__direct_csr1 : memref<i32> = uninitialized {csr_address = 2050 : i32}
    memref.global "private" @au_generated_candidate_1__direct_csr2 : memref<i32> = uninitialized {csr_address = 2051 : i32}
    memref.global "private" @au_generated_candidate_1__direct_csr3 : memref<i32> = uninitialized {csr_address = 2052 : i32}
    memref.global "private" @au_generated_candidate_1__direct_csr4 : memref<i32> = uninitialized {csr_address = 2053 : i32}
    memref.global "private" @au_generated_candidate_1__direct_csr5 : memref<i32> = uninitialized {csr_address = 2054 : i32}
    memref.global "private" @au_generated_candidate_1__direct_csr6 : memref<i32> = uninitialized {csr_address = 2055 : i32}
    tor.func @au_generated_candidate_1__direct(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 0 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-icp-32/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg0 {dump = "unknown_1", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg1 {dump = "unknown_2", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = aps.load_by @cpuitfc, %0 {after = array<i64>, after_gap = array<i64>, direction = 0 : i32, endtime = 4 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 4 : i32, ref_starttime = 2 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 3.160000e+02 : f64, starttime = 2 : i32, transaction_id = 0 : i64, transfer_bytes = 4 : i64} : i32 -> f32
      aps.globalstore %2, @au_generated_candidate_1__direct_buffer {endtime = 5 : i32, ref_endtime = 5 : i32, ref_starttime = 4 : i32, starttime = 4 : i32} : f32
      %3 = aps.load_by @cpuitfc, %1 {after = array<i64>, after_gap = array<i64>, direction = 0 : i32, endtime = 19 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 19 : i32, ref_starttime = 17 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 3.160000e+02 : f64, starttime = 17 : i32, transaction_id = 1 : i64, transfer_bytes = 4 : i64} : i32 -> f32
      aps.globalstore %3, @au_generated_candidate_1__direct_buffer_1 {endtime = 20 : i32, ref_endtime = 20 : i32, ref_starttime = 19 : i32, starttime = 19 : i32} : f32
      %4 = aps.read_csr @au_generated_candidate_1__direct_csr0 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %5 = aps.load_by @cpuitfc, %4 {after = array<i64>, after_gap = array<i64>, direction = 0 : i32, endtime = 21 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 21 : i32, ref_starttime = 19 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 3.160000e+02 : f64, starttime = 19 : i32, transaction_id = 2 : i64, transfer_bytes = 4 : i64} : i32 -> f32
      aps.globalstore %5, @au_generated_candidate_1__direct_buffer_2 {endtime = 22 : i32, ref_endtime = 22 : i32, ref_starttime = 21 : i32, starttime = 21 : i32} : f32
      %6 = aps.read_csr @au_generated_candidate_1__direct_csr1 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %7 = memref.get_global @au_generated_candidate_1__direct_buffer_3_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %8 = memref.get_global @au_generated_candidate_1__direct_buffer_3_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %6, (%7, %8)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 13 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 13 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 3 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %9 = tor.addi %6 %c64_i32 on (0 to 1) {dump = "unknown_3", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %9, (%7, %8)[%c16_i32], %c16_i32 {after = array<i64: 3, 3>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 35 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 35 : i32, ref_starttime = 16 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 16 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 4 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %10 = tor.addi %6 %c128_i32 on (0 to 1) {dump = "unknown_4", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %10, (%7, %8)[%c32_i32], %c16_i32 {after = array<i64: 4, 3, 4>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 46 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 46 : i32, ref_starttime = 31 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 31 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 5 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %11 = tor.addi %6 %c192_i32 on (0 to 1) {dump = "unknown_5", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %11, (%7, %8)[%c48_i32], %c16_i32 {after = array<i64: 5, 4, 5>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 61 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 61 : i32, ref_starttime = 46 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 46 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 6 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %12 = tor.addi %6 %c256_i32 on (0 to 1) {dump = "unknown_6", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %12, (%7, %8)[%c64_i32], %c16_i32 {after = array<i64: 6, 5, 6>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 76 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 76 : i32, ref_starttime = 61 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 61 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 7 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %13 = tor.addi %6 %c320_i32 on (0 to 1) {dump = "unknown_7", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %13, (%7, %8)[%c80_i32], %c16_i32 {after = array<i64: 7, 6, 7>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 91 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 91 : i32, ref_starttime = 76 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 76 : i32, tl_channel = 1 : i32, transaction_group = 0 : i64, transaction_id = 8 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %14 = aps.read_csr @au_generated_candidate_1__direct_csr2 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %15 = memref.get_global @au_generated_candidate_1__direct_buffer_4_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %16 = memref.get_global @au_generated_candidate_1__direct_buffer_4_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %14, (%15, %16)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 9 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %17 = tor.addi %14 %c64_i32 on (0 to 1) {dump = "unknown_8", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %17, (%15, %16)[%c16_i32], %c16_i32 {after = array<i64: 9, 9>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 34 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 34 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 10 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %18 = tor.addi %14 %c128_i32 on (0 to 1) {dump = "unknown_9", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %18, (%15, %16)[%c32_i32], %c16_i32 {after = array<i64: 10, 9, 10>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 45 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 45 : i32, ref_starttime = 30 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 30 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 11 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %19 = tor.addi %14 %c192_i32 on (0 to 1) {dump = "unknown_10", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %19, (%15, %16)[%c48_i32], %c16_i32 {after = array<i64: 11, 10, 11>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 60 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 60 : i32, ref_starttime = 45 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 45 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 12 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %20 = tor.addi %14 %c256_i32 on (0 to 1) {dump = "unknown_11", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %20, (%15, %16)[%c64_i32], %c16_i32 {after = array<i64: 12, 11, 12>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 75 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 75 : i32, ref_starttime = 60 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 60 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 13 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %21 = tor.addi %14 %c320_i32 on (0 to 1) {dump = "unknown_12", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %21, (%15, %16)[%c80_i32], %c16_i32 {after = array<i64: 13, 12, 13>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 90 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 52 : i64, memory_search_hits = 0 : i64, memory_search_states = 8 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 90 : i32, ref_starttime = 75 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 3.160000e+02 : f64, starttime = 75 : i32, tl_channel = 0 : i32, transaction_group = 1 : i64, transaction_id = 14 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %22 = aps.read_csr @au_generated_candidate_1__direct_csr3 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %23 = aps.load_by @cpuitfc, %22 {after = array<i64>, after_gap = array<i64>, direction = 0 : i32, endtime = 23 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 23 : i32, ref_starttime = 21 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 3.160000e+02 : f64, starttime = 21 : i32, transaction_id = 15 : i64, transfer_bytes = 4 : i64} : i32 -> f32
      aps.globalstore %23, @au_generated_candidate_1__direct_buffer_5 {endtime = 24 : i32, ref_endtime = 24 : i32, ref_starttime = 23 : i32, starttime = 23 : i32} : f32
      %24 = aps.read_csr @au_generated_candidate_1__direct_csr4 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %25 = aps.load_by @cpuitfc, %24 {after = array<i64>, after_gap = array<i64>, direction = 0 : i32, endtime = 25 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 25 : i32, ref_starttime = 23 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 3.160000e+02 : f64, starttime = 23 : i32, transaction_id = 16 : i64, transfer_bytes = 4 : i64} : i32 -> f32
      aps.globalstore %25, @au_generated_candidate_1__direct_buffer_6 {endtime = 26 : i32, ref_endtime = 26 : i32, ref_starttime = 25 : i32, starttime = 25 : i32} : f32
      %26 = aps.read_csr @au_generated_candidate_1__direct_csr5 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %27 = aps.load_by @cpuitfc, %26 {after = array<i64>, after_gap = array<i64>, direction = 0 : i32, endtime = 27 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 27 : i32, ref_starttime = 25 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 3.160000e+02 : f64, starttime = 25 : i32, transaction_id = 17 : i64, transfer_bytes = 4 : i64} : i32 -> f32
      aps.globalstore %27, @au_generated_candidate_1__direct_buffer_7 {endtime = 28 : i32, ref_endtime = 28 : i32, ref_starttime = 27 : i32, starttime = 27 : i32} : f32
      %28 = aps.read_csr @au_generated_candidate_1__direct_csr6 {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i32
      %29 = aps.load_by @cpuitfc, %28 {after = array<i64>, after_gap = array<i64>, direction = 0 : i32, endtime = 29 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 29 : i32, ref_starttime = 27 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 3.160000e+02 : f64, starttime = 27 : i32, transaction_id = 18 : i64, transfer_bytes = 4 : i64} : i32 -> f32
      aps.globalstore %29, @au_generated_candidate_1__direct_buffer_8 {endtime = 30 : i32, ref_endtime = 30 : i32, ref_starttime = 29 : i32, starttime = 29 : i32} : f32
      tor.for %arg3 = (%c0_i6 : i6) to (%c31_i6 : i6) step (%c1_i6 : i6)
      on (93 to 115){
        aps.globalstore %cst_137, @au_generated_candidate_1__direct_local {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : f32
        aps.globalstore %c0_i32, @au_generated_candidate_1__direct_local_0 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i32
        %34 = arith.extui %arg3 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : i6 to i32
        tor.for %arg4 = (%c0_i6 : i6) to (%c31_i6 : i6) step (%c1_i6 : i6)
        on (95 to 115){
          %176 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %177 = aps.globalload @au_generated_candidate_1__direct_buffer {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
          %178 = aps.globalload @au_generated_candidate_1__direct_buffer_1 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
          %179 = aps.globalload @au_generated_candidate_1__direct_buffer_2 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
          %180 = tor.muli %34 %c3_i32 on (95 to 96) {dump = "unknown_13", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %181 = tor.cmpi slt %180 %c0_i32 on (95 to 96) {dump = "unknown_14", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %182 = tor.subi %c-1_i32 %180 on (95 to 96) {dump = "unknown_15", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %183 = arith.select %181, %182, %180 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %184 = tor.cmpi slt %183 %c0_i32 on (95 to 96) {dump = "unknown_16", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %185 = arith.extui %184 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %186 = tor.addi %183 %185 on (95 to 96) {dump = "unknown_17", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %187 = arith.shrsi %186, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %188 = tor.subi %c-1_i32 %187 on (95 to 96) {dump = "unknown_18", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %189 = arith.select %181, %188, %187 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %190 = aps.read_smem %7[%189] {aps.mem_id = 23 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : memref<48xf32>, i32 -> f32
          %191 = tor.muli %34 %c3_i32 on (95 to 96) {dump = "unknown_19", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %192 = tor.cmpi slt %191 %c0_i32 on (95 to 96) {dump = "unknown_20", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %193 = tor.subi %c-1_i32 %191 on (95 to 96) {dump = "unknown_21", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %194 = arith.select %192, %193, %191 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %195 = tor.cmpi slt %194 %c0_i32 on (95 to 96) {dump = "unknown_22", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %196 = arith.extui %195 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %197 = tor.addi %194 %196 on (95 to 96) {dump = "unknown_23", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %198 = arith.shrsi %197, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %199 = tor.subi %c-1_i32 %198 on (95 to 96) {dump = "unknown_24", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %200 = arith.select %192, %199, %198 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %201 = aps.read_smem %8[%200] {aps.mem_id = 22 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : memref<48xf32>, i32 -> f32
          %202 = arith.andi %34, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %203 = tor.cmpi eq %202 %c0_i32 on (95 to 96) {dump = "unknown_25", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %204 = arith.select %203, %190, %201 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : f32
          %205 = tor.addf %204 %177 on (96 to 99) {dump = "unknown_26", fastmath = #arith.fastmath<none>, ref_endtime = 99 : i32, ref_starttime = 96 : i32} : f32
          %206 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %207 = tor.muli %206 %c3_i32 on (95 to 96) {dump = "unknown_27", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %208 = tor.cmpi slt %207 %c0_i32 on (95 to 96) {dump = "unknown_28", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %209 = tor.subi %c-1_i32 %207 on (95 to 96) {dump = "unknown_29", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %210 = arith.select %208, %209, %207 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %211 = tor.cmpi slt %210 %c0_i32 on (95 to 96) {dump = "unknown_30", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %212 = arith.extui %211 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %213 = tor.addi %210 %212 on (95 to 96) {dump = "unknown_31", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %214 = arith.shrsi %213, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %215 = tor.subi %c-1_i32 %214 on (95 to 96) {dump = "unknown_32", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %216 = arith.select %208, %215, %214 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %217 = aps.read_smem %15[%216] {aps.mem_id = 21 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : memref<48xf32>, i32 -> f32
          %218 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %219 = tor.muli %218 %c3_i32 on (95 to 96) {dump = "unknown_33", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %220 = tor.cmpi slt %219 %c0_i32 on (95 to 96) {dump = "unknown_34", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %221 = tor.subi %c-1_i32 %219 on (95 to 96) {dump = "unknown_35", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %222 = arith.select %220, %221, %219 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %223 = tor.cmpi slt %222 %c0_i32 on (95 to 96) {dump = "unknown_36", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %224 = arith.extui %223 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %225 = tor.addi %222 %224 on (95 to 96) {dump = "unknown_37", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %226 = arith.shrsi %225, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %227 = tor.subi %c-1_i32 %226 on (95 to 96) {dump = "unknown_38", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %228 = arith.select %220, %227, %226 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %229 = aps.read_smem %16[%228] {aps.mem_id = 20 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : memref<48xf32>, i32 -> f32
          %230 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %231 = arith.andi %230, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %232 = tor.cmpi eq %231 %c0_i32 on (95 to 96) {dump = "unknown_39", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %233 = arith.select %232, %217, %229 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : f32
          %234 = tor.subf %205 %233 on (99 to 102) {dump = "unknown_40", fastmath = #arith.fastmath<none>, ref_endtime = 102 : i32, ref_starttime = 99 : i32} : f32
          %235 = tor.muli %34 %c3_i32 on (95 to 96) {dump = "unknown_41", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %236 = tor.addi %235 %c1_i32 on (95 to 96) {dump = "unknown_42", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %237 = tor.cmpi slt %236 %c0_i32 on (95 to 96) {dump = "unknown_43", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %238 = tor.subi %c-2_i32 %235 on (95 to 96) {dump = "unknown_44", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %239 = arith.select %237, %238, %236 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %240 = tor.cmpi slt %239 %c0_i32 on (95 to 96) {dump = "unknown_45", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %241 = arith.extui %240 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %242 = tor.addi %239 %241 on (95 to 96) {dump = "unknown_46", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %243 = arith.shrsi %242, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %244 = tor.subi %c-1_i32 %243 on (95 to 96) {dump = "unknown_47", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %245 = arith.select %237, %244, %243 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %246 = aps.read_smem %7[%245] {aps.mem_id = 19 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : memref<48xf32>, i32 -> f32
          %247 = tor.muli %34 %c3_i32 on (95 to 96) {dump = "unknown_48", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %248 = tor.addi %247 %c1_i32 on (95 to 96) {dump = "unknown_49", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %249 = tor.cmpi slt %248 %c0_i32 on (95 to 96) {dump = "unknown_50", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %250 = tor.subi %c-2_i32 %247 on (95 to 96) {dump = "unknown_51", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %251 = arith.select %249, %250, %248 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %252 = tor.cmpi slt %251 %c0_i32 on (95 to 96) {dump = "unknown_52", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %253 = arith.extui %252 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %254 = tor.addi %251 %253 on (95 to 96) {dump = "unknown_53", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %255 = arith.shrsi %254, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %256 = tor.subi %c-1_i32 %255 on (95 to 96) {dump = "unknown_54", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %257 = arith.select %249, %256, %255 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %258 = aps.read_smem %8[%257] {aps.mem_id = 18 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : memref<48xf32>, i32 -> f32
          %259 = tor.addi %34 %c1_i32 on (95 to 96) {dump = "unknown_55", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %260 = tor.cmpi slt %259 %c0_i32 on (95 to 96) {dump = "unknown_56", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %261 = tor.subi %c-2_i32 %34 on (95 to 96) {dump = "unknown_57", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %262 = arith.select %260, %261, %259 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %263 = tor.cmpi slt %262 %c0_i32 on (95 to 96) {dump = "unknown_58", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %264 = arith.extui %263 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %265 = tor.addi %262 %264 on (95 to 96) {dump = "unknown_59", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %266 = arith.shrsi %265, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %267 = tor.subi %c-1_i32 %266 on (95 to 96) {dump = "unknown_60", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %268 = arith.select %260, %267, %266 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %269 = tor.muli %268 %c-2_i32 on (95 to 96) {dump = "unknown_61", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %270 = tor.addi %34 %269 on (95 to 96) {dump = "unknown_62", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %271 = tor.addi %270 %c1_i32 on (95 to 96) {dump = "unknown_63", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %272 = tor.cmpi eq %271 %c0_i32 on (95 to 96) {dump = "unknown_64", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %273 = arith.select %272, %246, %258 {endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, starttime = 97 : i32} : f32
          %274 = tor.addf %273 %178 on (97 to 100) {dump = "unknown_65", fastmath = #arith.fastmath<none>, ref_endtime = 100 : i32, ref_starttime = 97 : i32} : f32
          %275 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %276 = tor.muli %275 %c3_i32 on (95 to 96) {dump = "unknown_66", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %277 = tor.addi %276 %c1_i32 on (95 to 96) {dump = "unknown_67", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %278 = tor.cmpi slt %277 %c0_i32 on (95 to 96) {dump = "unknown_68", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %279 = tor.subi %c-2_i32 %276 on (95 to 96) {dump = "unknown_69", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %280 = arith.select %278, %279, %277 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %281 = tor.cmpi slt %280 %c0_i32 on (95 to 96) {dump = "unknown_70", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %282 = arith.extui %281 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %283 = tor.addi %280 %282 on (95 to 96) {dump = "unknown_71", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %284 = arith.shrsi %283, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %285 = tor.subi %c-1_i32 %284 on (95 to 96) {dump = "unknown_72", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %286 = arith.select %278, %285, %284 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %287 = aps.read_smem %15[%286] {aps.mem_id = 17 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : memref<48xf32>, i32 -> f32
          %288 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %289 = tor.muli %288 %c3_i32 on (95 to 96) {dump = "unknown_73", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %290 = tor.addi %289 %c1_i32 on (95 to 96) {dump = "unknown_74", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %291 = tor.cmpi slt %290 %c0_i32 on (95 to 96) {dump = "unknown_75", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %292 = tor.subi %c-2_i32 %289 on (95 to 96) {dump = "unknown_76", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %293 = arith.select %291, %292, %290 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %294 = tor.cmpi slt %293 %c0_i32 on (95 to 96) {dump = "unknown_77", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %295 = arith.extui %294 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %296 = tor.addi %293 %295 on (95 to 96) {dump = "unknown_78", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %297 = arith.shrsi %296, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %298 = tor.subi %c-1_i32 %297 on (95 to 96) {dump = "unknown_79", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %299 = arith.select %291, %298, %297 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %300 = aps.read_smem %16[%299] {aps.mem_id = 16 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : memref<48xf32>, i32 -> f32
          %301 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %302 = tor.addi %301 %c1_i32 on (95 to 96) {dump = "unknown_80", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %303 = tor.cmpi slt %302 %c0_i32 on (95 to 96) {dump = "unknown_81", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %304 = tor.subi %c-2_i32 %301 on (95 to 96) {dump = "unknown_82", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %305 = arith.select %303, %304, %302 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %306 = tor.cmpi slt %305 %c0_i32 on (95 to 96) {dump = "unknown_83", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %307 = arith.extui %306 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %308 = tor.addi %305 %307 on (95 to 96) {dump = "unknown_84", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %309 = arith.shrsi %308, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %310 = tor.subi %c-1_i32 %309 on (95 to 96) {dump = "unknown_85", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %311 = arith.select %303, %310, %309 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %312 = tor.muli %311 %c-2_i32 on (95 to 96) {dump = "unknown_86", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %313 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %314 = tor.addi %313 %312 on (95 to 96) {dump = "unknown_87", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %315 = tor.addi %314 %c1_i32 on (95 to 96) {dump = "unknown_88", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %316 = tor.cmpi eq %315 %c0_i32 on (95 to 96) {dump = "unknown_89", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %317 = arith.select %316, %287, %300 {endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, starttime = 97 : i32} : f32
          %318 = tor.subf %274 %317 on (100 to 103) {dump = "unknown_90", fastmath = #arith.fastmath<none>, ref_endtime = 103 : i32, ref_starttime = 100 : i32} : f32
          %319 = tor.muli %34 %c3_i32 on (95 to 96) {dump = "unknown_91", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %320 = tor.cmpi slt %319 %c0_i32 on (95 to 96) {dump = "unknown_92", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %321 = tor.subi %c-1_i32 %319 on (95 to 96) {dump = "unknown_93", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %322 = arith.select %320, %321, %319 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %323 = tor.cmpi slt %322 %c0_i32 on (95 to 96) {dump = "unknown_94", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %324 = arith.extui %323 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %325 = tor.addi %322 %324 on (95 to 96) {dump = "unknown_95", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %326 = arith.shrsi %325, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %327 = tor.subi %c-1_i32 %326 on (95 to 96) {dump = "unknown_96", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %328 = arith.select %320, %327, %326 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %329 = tor.addi %328 %c1_i32 on (95 to 96) {dump = "unknown_97", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %330 = aps.read_smem %7[%329] {aps.mem_id = 15 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : memref<48xf32>, i32 -> f32
          %331 = tor.muli %34 %c3_i32 on (95 to 96) {dump = "unknown_98", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %332 = tor.cmpi slt %331 %c0_i32 on (95 to 96) {dump = "unknown_99", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %333 = tor.subi %c-1_i32 %331 on (95 to 96) {dump = "unknown_100", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %334 = arith.select %332, %333, %331 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %335 = tor.cmpi slt %334 %c0_i32 on (95 to 96) {dump = "unknown_101", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %336 = arith.extui %335 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %337 = tor.addi %334 %336 on (95 to 96) {dump = "unknown_102", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %338 = arith.shrsi %337, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %339 = tor.subi %c-1_i32 %338 on (95 to 96) {dump = "unknown_103", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %340 = arith.select %332, %339, %338 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %341 = tor.addi %340 %c1_i32 on (95 to 96) {dump = "unknown_104", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %342 = aps.read_smem %8[%341] {aps.mem_id = 14 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : memref<48xf32>, i32 -> f32
          %343 = arith.andi %34, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %344 = tor.cmpi eq %343 %c0_i32 on (95 to 96) {dump = "unknown_105", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %345 = arith.select %344, %330, %342 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : f32
          %346 = tor.addf %345 %179 on (98 to 101) {dump = "unknown_106", fastmath = #arith.fastmath<none>, ref_endtime = 101 : i32, ref_starttime = 98 : i32} : f32
          %347 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %348 = tor.muli %347 %c3_i32 on (95 to 96) {dump = "unknown_107", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %349 = tor.cmpi slt %348 %c0_i32 on (95 to 96) {dump = "unknown_108", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %350 = tor.subi %c-1_i32 %348 on (95 to 96) {dump = "unknown_109", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %351 = arith.select %349, %350, %348 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %352 = tor.cmpi slt %351 %c0_i32 on (95 to 96) {dump = "unknown_110", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %353 = arith.extui %352 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %354 = tor.addi %351 %353 on (95 to 96) {dump = "unknown_111", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %355 = arith.shrsi %354, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %356 = tor.subi %c-1_i32 %355 on (95 to 96) {dump = "unknown_112", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %357 = arith.select %349, %356, %355 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %358 = tor.addi %357 %c1_i32 on (95 to 96) {dump = "unknown_113", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %359 = aps.read_smem %15[%358] {aps.mem_id = 13 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : memref<48xf32>, i32 -> f32
          %360 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %361 = tor.muli %360 %c3_i32 on (95 to 96) {dump = "unknown_114", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %362 = tor.cmpi slt %361 %c0_i32 on (95 to 96) {dump = "unknown_115", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %363 = tor.subi %c-1_i32 %361 on (95 to 96) {dump = "unknown_116", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %364 = arith.select %362, %363, %361 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %365 = tor.cmpi slt %364 %c0_i32 on (95 to 96) {dump = "unknown_117", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %366 = arith.extui %365 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
          %367 = tor.addi %364 %366 on (95 to 96) {dump = "unknown_118", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %368 = arith.shrsi %367, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %369 = tor.subi %c-1_i32 %368 on (95 to 96) {dump = "unknown_119", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %370 = arith.select %362, %369, %368 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %371 = tor.addi %370 %c1_i32 on (95 to 96) {dump = "unknown_120", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
          %372 = aps.read_smem %16[%371] {aps.mem_id = 12 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : memref<48xf32>, i32 -> f32
          %373 = arith.extui %arg4 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
          %374 = arith.andi %373, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
          %375 = tor.cmpi eq %374 %c0_i32 on (95 to 96) {dump = "unknown_121", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
          %376 = arith.select %375, %359, %372 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : f32
          %377 = tor.subf %346 %376 on (101 to 104) {dump = "unknown_122", fastmath = #arith.fastmath<none>, ref_endtime = 104 : i32, ref_starttime = 101 : i32} : f32
          %378 = tor.mulf %234 %234 on (102 to 105) {dump = "unknown_123", fastmath = #arith.fastmath<none>, ref_endtime = 105 : i32, ref_starttime = 102 : i32} : f32
          %379 = tor.mulf %318 %318 on (103 to 106) {dump = "unknown_124", fastmath = #arith.fastmath<none>, ref_endtime = 106 : i32, ref_starttime = 103 : i32} : f32
          %380 = tor.addf %378 %379 on (106 to 109) {dump = "unknown_125", fastmath = #arith.fastmath<none>, ref_endtime = 109 : i32, ref_starttime = 106 : i32} : f32
          %381 = tor.mulf %377 %377 on (104 to 107) {dump = "unknown_126", fastmath = #arith.fastmath<none>, ref_endtime = 107 : i32, ref_starttime = 104 : i32} : f32
          %382 = tor.addf %380 %381 on (109 to 112) {dump = "unknown_127", fastmath = #arith.fastmath<none>, ref_endtime = 112 : i32, ref_starttime = 109 : i32} : f32
          %383 = aps.globalload @au_generated_candidate_1__direct_local {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
          %384 = tor.cmpf olt %382 %383 on (112 to 113) {dump = "unknown_128", fastmath = #arith.fastmath<none>, ref_endtime = 113 : i32, ref_starttime = 112 : i32} : f32 -> i1
          tor.if %384 on (114 to 115) then {
            aps.globalstore %382, @au_generated_candidate_1__direct_local {endtime = 115 : i32, ref_endtime = 115 : i32, ref_starttime = 114 : i32, starttime = 114 : i32} : f32
            aps.globalstore %176, @au_generated_candidate_1__direct_local_0 {endtime = 115 : i32, ref_endtime = 115 : i32, ref_starttime = 114 : i32, starttime = 114 : i32} : i32
          } {dump = "unknown_129", ref_endtime = 115 : i32, ref_starttime = 114 : i32}
        } {ref_endtime = 115 : i32, ref_starttime = 95 : i32}
        %35 = aps.globalload @au_generated_candidate_1__direct_local_0 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %36 = tor.muli %35 %c3_i32 on (95 to 96) {dump = "unknown_130", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %37 = arith.andi %36, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %38 = arith.shrui %36, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %39 = aps.read_smem %15[%38] {aps.mem_id = 11 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : memref<48xf32>, i32 -> f32
        %40 = tor.cmpi eq %37 %c1_i32 on (95 to 96) {dump = "unknown_131", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %41 = aps.read_smem %16[%38] {aps.mem_id = 10 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : memref<48xf32>, i32 -> f32
        %42 = arith.select %40, %41, %39 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : f32
        %43 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %44 = tor.muli %43 %c3_i32 on (95 to 96) {dump = "unknown_132", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %45 = tor.cmpi slt %44 %c0_i32 on (95 to 96) {dump = "unknown_133", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %46 = tor.subi %c-1_i32 %44 on (95 to 96) {dump = "unknown_134", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %47 = arith.select %45, %46, %44 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %48 = tor.cmpi slt %47 %c0_i32 on (95 to 96) {dump = "unknown_135", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %49 = arith.extui %48 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
        %50 = tor.addi %47 %49 on (95 to 96) {dump = "unknown_136", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %51 = arith.shrsi %50, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %52 = tor.subi %c-1_i32 %51 on (95 to 96) {dump = "unknown_137", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %53 = arith.select %45, %52, %51 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %54 = aps.read_smem %7[%53] {aps.mem_id = 9 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : memref<48xf32>, i32 -> f32
        %55 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %56 = tor.muli %55 %c3_i32 on (95 to 96) {dump = "unknown_138", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %57 = tor.cmpi slt %56 %c0_i32 on (95 to 96) {dump = "unknown_139", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %58 = tor.subi %c-1_i32 %56 on (95 to 96) {dump = "unknown_140", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %59 = arith.select %57, %58, %56 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %60 = tor.cmpi slt %59 %c0_i32 on (95 to 96) {dump = "unknown_141", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %61 = arith.extui %60 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
        %62 = tor.addi %59 %61 on (95 to 96) {dump = "unknown_142", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %63 = arith.shrsi %62, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %64 = tor.subi %c-1_i32 %63 on (95 to 96) {dump = "unknown_143", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %65 = arith.select %57, %64, %63 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %66 = aps.read_smem %8[%65] {aps.mem_id = 8 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : memref<48xf32>, i32 -> f32
        %67 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %68 = arith.andi %67, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %69 = tor.cmpi eq %68 %c0_i32 on (95 to 96) {dump = "unknown_144", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %70 = arith.select %69, %54, %66 {endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, starttime = 96 : i32} : f32
        %71 = aps.globalload @au_generated_candidate_1__direct_buffer {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
        %72 = tor.addf %70 %71 on (96 to 99) {dump = "unknown_145", fastmath = #arith.fastmath<none>, ref_endtime = 99 : i32, ref_starttime = 96 : i32} : f32
        %73 = tor.subf %42 %72 on (99 to 102) {dump = "unknown_146", fastmath = #arith.fastmath<none>, ref_endtime = 102 : i32, ref_starttime = 99 : i32} : f32
        %74 = aps.globalload @au_generated_candidate_1__direct_buffer_5 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
        %75 = tor.addf %74 %73 on (102 to 105) {dump = "unknown_147", fastmath = #arith.fastmath<none>, ref_endtime = 105 : i32, ref_starttime = 102 : i32} : f32
        aps.globalstore %75, @au_generated_candidate_1__direct_buffer_5 {endtime = 106 : i32, ref_endtime = 106 : i32, ref_starttime = 105 : i32, starttime = 105 : i32} : f32
        %76 = tor.addi %36 %c1_i32 on (95 to 96) {dump = "unknown_148", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %77 = arith.andi %76, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %78 = arith.shrui %76, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %79 = aps.read_smem %15[%78] {aps.mem_id = 7 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : memref<48xf32>, i32 -> f32
        %80 = tor.cmpi eq %77 %c1_i32 on (95 to 96) {dump = "unknown_149", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %81 = aps.read_smem %16[%78] {aps.mem_id = 6 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : memref<48xf32>, i32 -> f32
        %82 = arith.select %80, %81, %79 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : f32
        %83 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %84 = tor.muli %83 %c3_i32 on (95 to 96) {dump = "unknown_150", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %85 = tor.addi %84 %c1_i32 on (95 to 96) {dump = "unknown_151", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %86 = tor.cmpi slt %85 %c0_i32 on (95 to 96) {dump = "unknown_152", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %87 = tor.subi %c-2_i32 %84 on (95 to 96) {dump = "unknown_153", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %88 = arith.select %86, %87, %85 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %89 = tor.cmpi slt %88 %c0_i32 on (95 to 96) {dump = "unknown_154", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %90 = arith.extui %89 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
        %91 = tor.addi %88 %90 on (95 to 96) {dump = "unknown_155", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %92 = arith.shrsi %91, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %93 = tor.subi %c-1_i32 %92 on (95 to 96) {dump = "unknown_156", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %94 = arith.select %86, %93, %92 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %95 = aps.read_smem %7[%94] {aps.mem_id = 5 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : memref<48xf32>, i32 -> f32
        %96 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %97 = tor.muli %96 %c3_i32 on (95 to 96) {dump = "unknown_157", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %98 = tor.addi %97 %c1_i32 on (95 to 96) {dump = "unknown_158", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %99 = tor.cmpi slt %98 %c0_i32 on (95 to 96) {dump = "unknown_159", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %100 = tor.subi %c-2_i32 %97 on (95 to 96) {dump = "unknown_160", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %101 = arith.select %99, %100, %98 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %102 = tor.cmpi slt %101 %c0_i32 on (95 to 96) {dump = "unknown_161", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %103 = arith.extui %102 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
        %104 = tor.addi %101 %103 on (95 to 96) {dump = "unknown_162", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %105 = arith.shrsi %104, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %106 = tor.subi %c-1_i32 %105 on (95 to 96) {dump = "unknown_163", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %107 = arith.select %99, %106, %105 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %108 = aps.read_smem %8[%107] {aps.mem_id = 4 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : memref<48xf32>, i32 -> f32
        %109 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %110 = tor.addi %109 %c1_i32 on (95 to 96) {dump = "unknown_164", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %111 = tor.cmpi slt %110 %c0_i32 on (95 to 96) {dump = "unknown_165", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %112 = tor.subi %c-2_i32 %109 on (95 to 96) {dump = "unknown_166", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %113 = arith.select %111, %112, %110 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %114 = tor.cmpi slt %113 %c0_i32 on (95 to 96) {dump = "unknown_167", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %115 = arith.extui %114 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
        %116 = tor.addi %113 %115 on (95 to 96) {dump = "unknown_168", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %117 = arith.shrsi %116, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %118 = tor.subi %c-1_i32 %117 on (95 to 96) {dump = "unknown_169", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %119 = arith.select %111, %118, %117 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %120 = tor.muli %119 %c-2_i32 on (95 to 96) {dump = "unknown_170", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %121 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %122 = tor.addi %121 %120 on (95 to 96) {dump = "unknown_171", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %123 = tor.addi %122 %c1_i32 on (95 to 96) {dump = "unknown_172", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %124 = tor.cmpi eq %123 %c0_i32 on (95 to 96) {dump = "unknown_173", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %125 = arith.select %124, %95, %108 {endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, starttime = 97 : i32} : f32
        %126 = aps.globalload @au_generated_candidate_1__direct_buffer_1 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
        %127 = tor.addf %125 %126 on (97 to 100) {dump = "unknown_174", fastmath = #arith.fastmath<none>, ref_endtime = 100 : i32, ref_starttime = 97 : i32} : f32
        %128 = tor.subf %82 %127 on (100 to 103) {dump = "unknown_175", fastmath = #arith.fastmath<none>, ref_endtime = 103 : i32, ref_starttime = 100 : i32} : f32
        %129 = aps.globalload @au_generated_candidate_1__direct_buffer_6 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
        %130 = tor.addf %129 %128 on (103 to 106) {dump = "unknown_176", fastmath = #arith.fastmath<none>, ref_endtime = 106 : i32, ref_starttime = 103 : i32} : f32
        aps.globalstore %130, @au_generated_candidate_1__direct_buffer_6 {endtime = 107 : i32, ref_endtime = 107 : i32, ref_starttime = 106 : i32, starttime = 106 : i32} : f32
        %131 = tor.addi %36 %c2_i32 on (95 to 96) {dump = "unknown_177", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %132 = arith.andi %131, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %133 = arith.shrui %131, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %134 = aps.read_smem %15[%133] {aps.mem_id = 3 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : memref<48xf32>, i32 -> f32
        %135 = tor.cmpi eq %132 %c1_i32 on (95 to 96) {dump = "unknown_178", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %136 = aps.read_smem %16[%133] {aps.mem_id = 2 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : memref<48xf32>, i32 -> f32
        %137 = arith.select %135, %136, %134 {endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, starttime = 97 : i32} : f32
        %138 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %139 = tor.muli %138 %c3_i32 on (95 to 96) {dump = "unknown_179", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %140 = tor.cmpi slt %139 %c0_i32 on (95 to 96) {dump = "unknown_180", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %141 = tor.subi %c-1_i32 %139 on (95 to 96) {dump = "unknown_181", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %142 = arith.select %140, %141, %139 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %143 = tor.cmpi slt %142 %c0_i32 on (95 to 96) {dump = "unknown_182", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %144 = arith.extui %143 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
        %145 = tor.addi %142 %144 on (95 to 96) {dump = "unknown_183", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %146 = arith.shrsi %145, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %147 = tor.subi %c-1_i32 %146 on (95 to 96) {dump = "unknown_184", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %148 = arith.select %140, %147, %146 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %149 = tor.addi %148 %c1_i32 on (95 to 96) {dump = "unknown_185", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %150 = aps.read_smem %7[%149] {aps.mem_id = 1 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : memref<48xf32>, i32 -> f32
        %151 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %152 = tor.muli %151 %c3_i32 on (95 to 96) {dump = "unknown_186", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %153 = tor.cmpi slt %152 %c0_i32 on (95 to 96) {dump = "unknown_187", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %154 = tor.subi %c-1_i32 %152 on (95 to 96) {dump = "unknown_188", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %155 = arith.select %153, %154, %152 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %156 = tor.cmpi slt %155 %c0_i32 on (95 to 96) {dump = "unknown_189", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %157 = arith.extui %156 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i1 to i32
        %158 = tor.addi %155 %157 on (95 to 96) {dump = "unknown_190", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %159 = arith.shrsi %158, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %160 = tor.subi %c-1_i32 %159 on (95 to 96) {dump = "unknown_191", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %161 = arith.select %153, %160, %159 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %162 = tor.addi %161 %c1_i32 on (95 to 96) {dump = "unknown_192", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i32
        %163 = aps.read_smem %8[%162] {aps.mem_id = 0 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : memref<48xf32>, i32 -> f32
        %164 = arith.extui %arg3 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i6 to i32
        %165 = arith.andi %164, %c1_i32 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : i32
        %166 = tor.cmpi eq %165 %c0_i32 on (95 to 96) {dump = "unknown_193", ref_endtime = 96 : i32, ref_starttime = 95 : i32} : (i32, i32) -> i1
        %167 = arith.select %166, %150, %163 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : f32
        %168 = aps.globalload @au_generated_candidate_1__direct_buffer_2 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
        %169 = tor.addf %167 %168 on (98 to 101) {dump = "unknown_194", fastmath = #arith.fastmath<none>, ref_endtime = 101 : i32, ref_starttime = 98 : i32} : f32
        %170 = tor.subf %137 %169 on (101 to 104) {dump = "unknown_195", fastmath = #arith.fastmath<none>, ref_endtime = 104 : i32, ref_starttime = 101 : i32} : f32
        %171 = aps.globalload @au_generated_candidate_1__direct_buffer_7 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
        %172 = tor.addf %171 %170 on (104 to 107) {dump = "unknown_196", fastmath = #arith.fastmath<none>, ref_endtime = 107 : i32, ref_starttime = 104 : i32} : f32
        aps.globalstore %172, @au_generated_candidate_1__direct_buffer_7 {endtime = 108 : i32, ref_endtime = 108 : i32, ref_starttime = 107 : i32, starttime = 107 : i32} : f32
        %173 = aps.globalload @au_generated_candidate_1__direct_local {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
        %174 = aps.globalload @au_generated_candidate_1__direct_buffer_8 {endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, starttime = 95 : i32} : f32
        %175 = tor.addf %174 %173 on (95 to 98) {dump = "unknown_197", fastmath = #arith.fastmath<none>, ref_endtime = 98 : i32, ref_starttime = 95 : i32} : f32
        aps.globalstore %175, @au_generated_candidate_1__direct_buffer_8 {endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, starttime = 98 : i32} : f32
      } {ref_endtime = 115 : i32, ref_starttime = 93 : i32}
      %30 = aps.globalload @au_generated_candidate_1__direct_buffer_5 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : f32
      aps.store_by @cpuitfc, %30, %22 {after = array<i64>, after_gap = array<i64>, direction = 1 : i32, endtime = 95 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 95 : i32, ref_starttime = 93 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 7.500000e+01 : f64, starttime = 93 : i32, transaction_id = 19 : i64, transfer_bytes = 4 : i64} : f32, i32
      %31 = aps.globalload @au_generated_candidate_1__direct_buffer_6 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : f32
      aps.store_by @cpuitfc, %31, %24 {after = array<i64>, after_gap = array<i64>, direction = 1 : i32, endtime = 97 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 97 : i32, ref_starttime = 95 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 7.500000e+01 : f64, starttime = 95 : i32, transaction_id = 20 : i64, transfer_bytes = 4 : i64} : f32, i32
      %32 = aps.globalload @au_generated_candidate_1__direct_buffer_7 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : f32
      aps.store_by @cpuitfc, %32, %26 {after = array<i64>, after_gap = array<i64>, direction = 1 : i32, endtime = 99 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 99 : i32, ref_starttime = 97 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 7.500000e+01 : f64, starttime = 97 : i32, transaction_id = 21 : i64, transfer_bytes = 4 : i64} : f32, i32
      %33 = aps.globalload @au_generated_candidate_1__direct_buffer_8 {endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, starttime = 93 : i32} : f32
      aps.store_by @cpuitfc, %33, %28 {after = array<i64>, after_gap = array<i64>, direction = 1 : i32, endtime = 101 : i32, estimated_latency = 2 : i64, interface_cost = 18 : i64, interface_solver = "active-set-dp", memory_completion_cycle = 2 : i64, memory_issue_cycle = 0 : i64, memory_latency = 2 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 2 : i64, memory_search_hits = 0 : i64, memory_search_states = 3 : i64, ref_endtime = 101 : i32, ref_starttime = 99 : i32, required_alignment = 4 : i64, segment = 0 : i64, segment_count = 1 : i64, solver_objective = 7.500000e+01 : f64, starttime = 99 : i32, transaction_id = 22 : i64, transfer_bytes = 4 : i64} : f32, i32
      tor.return
    }
    tor.func @au_generated_candidate_8__l0u(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 8.000000e+00 : f32, funct7 = 1 : i32, opcode = 43 : i32, resource = "/newdisk/aps-mlir/build/aps-icp-32/resource.json", scheduled = true} {
      %0 = aps.read_irf %arg1 {dump = "unknown_198", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.read_irf %arg0 {dump = "unknown_199", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @au_generated_candidate_8__l0u_buffer_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @au_generated_candidate_8__l0u_buffer_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %1, (%2, %3)[%c0_i32], %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 12 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 12 : i64, memory_issue_cycle = 0 : i64, memory_latency = 12 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 12 : i32, ref_starttime = 0 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 0 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 23 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %4 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_200", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %4, (%2, %3)[%c16_i32], %c16_i32 {after = array<i64: 23, 23>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 20 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 20 : i64, memory_issue_cycle = 1 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 20 : i32, ref_starttime = 1 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 1 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 24 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %5 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_201", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %5, (%2, %3)[%c32_i32], %c16_i32 {after = array<i64: 24, 23, 24>, after_gap = array<i64: 12, 13, 12>, after_segment = 1 : i64, endtime = 30 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 28 : i64, memory_issue_cycle = 13 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 30 : i32, ref_starttime = 15 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 15 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 25 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %6 = tor.addi %1 %c192_i32 on (0 to 1) {dump = "unknown_202", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %6, (%2, %3)[%c48_i32], %c16_i32 {after = array<i64: 25, 24, 25>, after_gap = array<i64: 8, 20, 8>, after_segment = 2 : i64, endtime = 38 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 36 : i64, memory_issue_cycle = 21 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 38 : i32, ref_starttime = 23 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 23 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 26 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %7 = tor.addi %1 %c256_i32 on (0 to 1) {dump = "unknown_203", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %7, (%2, %3)[%c64_i32], %c16_i32 {after = array<i64: 26, 25, 26>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 46 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 44 : i64, memory_issue_cycle = 29 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 46 : i32, ref_starttime = 31 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 31 : i32, tl_channel = 0 : i32, transaction_group = 2 : i64, transaction_id = 27 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %8 = tor.addi %1 %c320_i32 on (0 to 1) {dump = "unknown_204", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %8, (%2, %3)[%c80_i32], %c16_i32 {after = array<i64: 27, 26, 27>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 54 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 52 : i64, memory_issue_cycle = 37 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 54 : i32, ref_starttime = 39 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 39 : i32, tl_channel = 1 : i32, transaction_group = 2 : i64, transaction_id = 28 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %9 = memref.get_global @au_generated_candidate_8__l0u_buffer_9_0 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %10 = memref.get_global @au_generated_candidate_8__l0u_buffer_9_1 : memref<48xf32> {endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      aps.copy_by @busitfc, %0, (%9, %10)[%c0_i32], %c16_i32 {after = array<i64: 28, 27, 28>, after_gap = array<i64: 8, 16, 8>, endtime = 62 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 60 : i64, memory_issue_cycle = 45 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 62 : i32, ref_starttime = 47 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 47 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 29 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %11 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_205", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %11, (%9, %10)[%c16_i32], %c16_i32 {after = array<i64: 29, 28, 29>, after_gap = array<i64: 8, 16, 8>, after_segment = 0 : i64, endtime = 70 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 68 : i64, memory_issue_cycle = 53 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 70 : i32, ref_starttime = 55 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 55 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 30 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %12 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_206", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %12, (%9, %10)[%c32_i32], %c16_i32 {after = array<i64: 30, 29, 30>, after_gap = array<i64: 8, 16, 8>, after_segment = 1 : i64, endtime = 78 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 76 : i64, memory_issue_cycle = 61 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 78 : i32, ref_starttime = 63 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 63 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 31 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %13 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_207", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %13, (%9, %10)[%c48_i32], %c16_i32 {after = array<i64: 31, 30, 31>, after_gap = array<i64: 8, 16, 8>, after_segment = 2 : i64, endtime = 86 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 84 : i64, memory_issue_cycle = 69 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 86 : i32, ref_starttime = 71 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 71 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 32 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %14 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_208", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %14, (%9, %10)[%c64_i32], %c16_i32 {after = array<i64: 32, 31, 32>, after_gap = array<i64: 8, 16, 8>, after_segment = 3 : i64, endtime = 94 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 92 : i64, memory_issue_cycle = 77 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 94 : i32, ref_starttime = 79 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 79 : i32, tl_channel = 0 : i32, transaction_group = 3 : i64, transaction_id = 33 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      %15 = tor.addi %0 %c320_i32 on (0 to 1) {dump = "unknown_209", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, %15, (%9, %10)[%c80_i32], %c16_i32 {after = array<i64: 33, 32, 33>, after_gap = array<i64: 8, 16, 8>, after_segment = 4 : i64, endtime = 102 : i32, estimated_latency = 52 : i64, interface_cost = 100 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 100 : i64, memory_issue_cycle = 85 : i64, memory_latency = 15 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 100 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 102 : i32, ref_starttime = 87 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 1.960000e+02 : f64, starttime = 87 : i32, tl_channel = 1 : i32, transaction_group = 3 : i64, transaction_id = 34 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : i32, (memref<48xf32>, memref<48xf32>), i32, i32
      aps.write_smem %cst, %2[%c0_i32] {aps.mem_id = 191 : i64, endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, slot = 0 : i32, starttime = 54 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst, %3[%c0_i32] {aps.mem_id = 190 : i64, endtime = 55 : i32, ref_endtime = 55 : i32, ref_starttime = 54 : i32, slot = 0 : i32, starttime = 54 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst, %2[%c1_i32] {aps.mem_id = 189 : i64, endtime = 56 : i32, ref_endtime = 56 : i32, ref_starttime = 55 : i32, slot = 0 : i32, starttime = 55 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %9[%c0_i32] {aps.mem_id = 188 : i64, endtime = 103 : i32, ref_endtime = 103 : i32, ref_starttime = 102 : i32, slot = 0 : i32, starttime = 102 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_1, %10[%c0_i32] {aps.mem_id = 187 : i64, endtime = 103 : i32, ref_endtime = 103 : i32, ref_starttime = 102 : i32, slot = 0 : i32, starttime = 102 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_2, %9[%c1_i32] {aps.mem_id = 186 : i64, endtime = 104 : i32, ref_endtime = 104 : i32, ref_starttime = 103 : i32, slot = 0 : i32, starttime = 103 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_3, %3[%c1_i32] {aps.mem_id = 185 : i64, endtime = 56 : i32, ref_endtime = 56 : i32, ref_starttime = 55 : i32, slot = 0 : i32, starttime = 55 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_4, %2[%c2_i32] {aps.mem_id = 184 : i64, endtime = 57 : i32, ref_endtime = 57 : i32, ref_starttime = 56 : i32, slot = 0 : i32, starttime = 56 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_5, %3[%c2_i32] {aps.mem_id = 183 : i64, endtime = 57 : i32, ref_endtime = 57 : i32, ref_starttime = 56 : i32, slot = 0 : i32, starttime = 56 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_6, %10[%c1_i32] {aps.mem_id = 182 : i64, endtime = 104 : i32, ref_endtime = 104 : i32, ref_starttime = 103 : i32, slot = 0 : i32, starttime = 103 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_7, %9[%c2_i32] {aps.mem_id = 181 : i64, endtime = 105 : i32, ref_endtime = 105 : i32, ref_starttime = 104 : i32, slot = 0 : i32, starttime = 104 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_8, %10[%c2_i32] {aps.mem_id = 180 : i64, endtime = 105 : i32, ref_endtime = 105 : i32, ref_starttime = 104 : i32, slot = 0 : i32, starttime = 104 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_9, %2[%c3_i32] {aps.mem_id = 179 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_10, %3[%c3_i32] {aps.mem_id = 178 : i64, endtime = 58 : i32, ref_endtime = 58 : i32, ref_starttime = 57 : i32, slot = 0 : i32, starttime = 57 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_11, %2[%c4_i32] {aps.mem_id = 177 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_12, %9[%c3_i32] {aps.mem_id = 176 : i64, endtime = 106 : i32, ref_endtime = 106 : i32, ref_starttime = 105 : i32, slot = 0 : i32, starttime = 105 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_13, %10[%c3_i32] {aps.mem_id = 175 : i64, endtime = 106 : i32, ref_endtime = 106 : i32, ref_starttime = 105 : i32, slot = 0 : i32, starttime = 105 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_14, %9[%c4_i32] {aps.mem_id = 174 : i64, endtime = 107 : i32, ref_endtime = 107 : i32, ref_starttime = 106 : i32, slot = 0 : i32, starttime = 106 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_15, %3[%c4_i32] {aps.mem_id = 173 : i64, endtime = 59 : i32, ref_endtime = 59 : i32, ref_starttime = 58 : i32, slot = 0 : i32, starttime = 58 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_16, %2[%c5_i32] {aps.mem_id = 172 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_17, %3[%c5_i32] {aps.mem_id = 171 : i64, endtime = 60 : i32, ref_endtime = 60 : i32, ref_starttime = 59 : i32, slot = 0 : i32, starttime = 59 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_18, %10[%c4_i32] {aps.mem_id = 170 : i64, endtime = 107 : i32, ref_endtime = 107 : i32, ref_starttime = 106 : i32, slot = 0 : i32, starttime = 106 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_19, %9[%c5_i32] {aps.mem_id = 169 : i64, endtime = 108 : i32, ref_endtime = 108 : i32, ref_starttime = 107 : i32, slot = 0 : i32, starttime = 107 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_20, %10[%c5_i32] {aps.mem_id = 168 : i64, endtime = 108 : i32, ref_endtime = 108 : i32, ref_starttime = 107 : i32, slot = 0 : i32, starttime = 107 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_21, %2[%c6_i32] {aps.mem_id = 167 : i64, endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, slot = 0 : i32, starttime = 60 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_22, %3[%c6_i32] {aps.mem_id = 166 : i64, endtime = 61 : i32, ref_endtime = 61 : i32, ref_starttime = 60 : i32, slot = 0 : i32, starttime = 60 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_9, %2[%c7_i32] {aps.mem_id = 165 : i64, endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, slot = 0 : i32, starttime = 61 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_23, %9[%c6_i32] {aps.mem_id = 164 : i64, endtime = 109 : i32, ref_endtime = 109 : i32, ref_starttime = 108 : i32, slot = 0 : i32, starttime = 108 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_24, %10[%c6_i32] {aps.mem_id = 163 : i64, endtime = 109 : i32, ref_endtime = 109 : i32, ref_starttime = 108 : i32, slot = 0 : i32, starttime = 108 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_25, %9[%c7_i32] {aps.mem_id = 162 : i64, endtime = 110 : i32, ref_endtime = 110 : i32, ref_starttime = 109 : i32, slot = 0 : i32, starttime = 109 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_26, %3[%c7_i32] {aps.mem_id = 161 : i64, endtime = 62 : i32, ref_endtime = 62 : i32, ref_starttime = 61 : i32, slot = 0 : i32, starttime = 61 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_27, %2[%c8_i32] {aps.mem_id = 160 : i64, endtime = 63 : i32, ref_endtime = 63 : i32, ref_starttime = 62 : i32, slot = 0 : i32, starttime = 62 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_2, %3[%c8_i32] {aps.mem_id = 159 : i64, endtime = 63 : i32, ref_endtime = 63 : i32, ref_starttime = 62 : i32, slot = 0 : i32, starttime = 62 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_28, %10[%c7_i32] {aps.mem_id = 158 : i64, endtime = 110 : i32, ref_endtime = 110 : i32, ref_starttime = 109 : i32, slot = 0 : i32, starttime = 109 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_29, %9[%c8_i32] {aps.mem_id = 157 : i64, endtime = 111 : i32, ref_endtime = 111 : i32, ref_starttime = 110 : i32, slot = 0 : i32, starttime = 110 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_30, %10[%c8_i32] {aps.mem_id = 156 : i64, endtime = 111 : i32, ref_endtime = 111 : i32, ref_starttime = 110 : i32, slot = 0 : i32, starttime = 110 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_31, %2[%c9_i32] {aps.mem_id = 155 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_32, %3[%c9_i32] {aps.mem_id = 154 : i64, endtime = 64 : i32, ref_endtime = 64 : i32, ref_starttime = 63 : i32, slot = 0 : i32, starttime = 63 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_8, %2[%c10_i32] {aps.mem_id = 153 : i64, endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, slot = 0 : i32, starttime = 64 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_33, %9[%c9_i32] {aps.mem_id = 152 : i64, endtime = 112 : i32, ref_endtime = 112 : i32, ref_starttime = 111 : i32, slot = 0 : i32, starttime = 111 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_34, %10[%c9_i32] {aps.mem_id = 151 : i64, endtime = 112 : i32, ref_endtime = 112 : i32, ref_starttime = 111 : i32, slot = 0 : i32, starttime = 111 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_35, %9[%c10_i32] {aps.mem_id = 150 : i64, endtime = 113 : i32, ref_endtime = 113 : i32, ref_starttime = 112 : i32, slot = 0 : i32, starttime = 112 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_36, %3[%c10_i32] {aps.mem_id = 149 : i64, endtime = 65 : i32, ref_endtime = 65 : i32, ref_starttime = 64 : i32, slot = 0 : i32, starttime = 64 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_37, %2[%c11_i32] {aps.mem_id = 148 : i64, endtime = 66 : i32, ref_endtime = 66 : i32, ref_starttime = 65 : i32, slot = 0 : i32, starttime = 65 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_14, %3[%c11_i32] {aps.mem_id = 147 : i64, endtime = 66 : i32, ref_endtime = 66 : i32, ref_starttime = 65 : i32, slot = 0 : i32, starttime = 65 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_8, %10[%c10_i32] {aps.mem_id = 146 : i64, endtime = 113 : i32, ref_endtime = 113 : i32, ref_starttime = 112 : i32, slot = 0 : i32, starttime = 112 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_38, %9[%c11_i32] {aps.mem_id = 145 : i64, endtime = 114 : i32, ref_endtime = 114 : i32, ref_starttime = 113 : i32, slot = 0 : i32, starttime = 113 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_39, %10[%c11_i32] {aps.mem_id = 144 : i64, endtime = 114 : i32, ref_endtime = 114 : i32, ref_starttime = 113 : i32, slot = 0 : i32, starttime = 113 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_40, %2[%c12_i32] {aps.mem_id = 143 : i64, endtime = 67 : i32, ref_endtime = 67 : i32, ref_starttime = 66 : i32, slot = 0 : i32, starttime = 66 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_41, %3[%c12_i32] {aps.mem_id = 142 : i64, endtime = 67 : i32, ref_endtime = 67 : i32, ref_starttime = 66 : i32, slot = 0 : i32, starttime = 66 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_20, %2[%c13_i32] {aps.mem_id = 141 : i64, endtime = 68 : i32, ref_endtime = 68 : i32, ref_starttime = 67 : i32, slot = 0 : i32, starttime = 67 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_42, %9[%c12_i32] {aps.mem_id = 140 : i64, endtime = 115 : i32, ref_endtime = 115 : i32, ref_starttime = 114 : i32, slot = 0 : i32, starttime = 114 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_43, %10[%c12_i32] {aps.mem_id = 139 : i64, endtime = 115 : i32, ref_endtime = 115 : i32, ref_starttime = 114 : i32, slot = 0 : i32, starttime = 114 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_44, %9[%c13_i32] {aps.mem_id = 138 : i64, endtime = 116 : i32, ref_endtime = 116 : i32, ref_starttime = 115 : i32, slot = 0 : i32, starttime = 115 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_45, %3[%c13_i32] {aps.mem_id = 137 : i64, endtime = 68 : i32, ref_endtime = 68 : i32, ref_starttime = 67 : i32, slot = 0 : i32, starttime = 67 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_46, %2[%c14_i32] {aps.mem_id = 136 : i64, endtime = 69 : i32, ref_endtime = 69 : i32, ref_starttime = 68 : i32, slot = 0 : i32, starttime = 68 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_25, %3[%c14_i32] {aps.mem_id = 135 : i64, endtime = 69 : i32, ref_endtime = 69 : i32, ref_starttime = 68 : i32, slot = 0 : i32, starttime = 68 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_47, %10[%c13_i32] {aps.mem_id = 134 : i64, endtime = 116 : i32, ref_endtime = 116 : i32, ref_starttime = 115 : i32, slot = 0 : i32, starttime = 115 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_48, %9[%c14_i32] {aps.mem_id = 133 : i64, endtime = 117 : i32, ref_endtime = 117 : i32, ref_starttime = 116 : i32, slot = 0 : i32, starttime = 116 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_49, %10[%c14_i32] {aps.mem_id = 132 : i64, endtime = 117 : i32, ref_endtime = 117 : i32, ref_starttime = 116 : i32, slot = 0 : i32, starttime = 116 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_50, %2[%c15_i32] {aps.mem_id = 131 : i64, endtime = 70 : i32, ref_endtime = 70 : i32, ref_starttime = 69 : i32, slot = 0 : i32, starttime = 69 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_51, %3[%c15_i32] {aps.mem_id = 130 : i64, endtime = 70 : i32, ref_endtime = 70 : i32, ref_starttime = 69 : i32, slot = 0 : i32, starttime = 69 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_30, %2[%c16_i32] {aps.mem_id = 129 : i64, endtime = 71 : i32, ref_endtime = 71 : i32, ref_starttime = 70 : i32, slot = 0 : i32, starttime = 70 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_52, %9[%c15_i32] {aps.mem_id = 128 : i64, endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, slot = 0 : i32, starttime = 117 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_53, %10[%c15_i32] {aps.mem_id = 127 : i64, endtime = 118 : i32, ref_endtime = 118 : i32, ref_starttime = 117 : i32, slot = 0 : i32, starttime = 117 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_54, %9[%c16_i32] {aps.mem_id = 126 : i64, endtime = 119 : i32, ref_endtime = 119 : i32, ref_starttime = 118 : i32, slot = 0 : i32, starttime = 118 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_55, %3[%c16_i32] {aps.mem_id = 125 : i64, endtime = 71 : i32, ref_endtime = 71 : i32, ref_starttime = 70 : i32, slot = 0 : i32, starttime = 70 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_56, %2[%c17_i32] {aps.mem_id = 124 : i64, endtime = 72 : i32, ref_endtime = 72 : i32, ref_starttime = 71 : i32, slot = 0 : i32, starttime = 71 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_35, %3[%c17_i32] {aps.mem_id = 123 : i64, endtime = 72 : i32, ref_endtime = 72 : i32, ref_starttime = 71 : i32, slot = 0 : i32, starttime = 71 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_57, %10[%c16_i32] {aps.mem_id = 122 : i64, endtime = 119 : i32, ref_endtime = 119 : i32, ref_starttime = 118 : i32, slot = 0 : i32, starttime = 118 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_58, %9[%c17_i32] {aps.mem_id = 121 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_59, %10[%c17_i32] {aps.mem_id = 120 : i64, endtime = 120 : i32, ref_endtime = 120 : i32, ref_starttime = 119 : i32, slot = 0 : i32, starttime = 119 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_2, %2[%c18_i32] {aps.mem_id = 119 : i64, endtime = 73 : i32, ref_endtime = 73 : i32, ref_starttime = 72 : i32, slot = 0 : i32, starttime = 72 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_60, %3[%c18_i32] {aps.mem_id = 118 : i64, endtime = 73 : i32, ref_endtime = 73 : i32, ref_starttime = 72 : i32, slot = 0 : i32, starttime = 72 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_61, %2[%c19_i32] {aps.mem_id = 117 : i64, endtime = 74 : i32, ref_endtime = 74 : i32, ref_starttime = 73 : i32, slot = 0 : i32, starttime = 73 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_62, %9[%c18_i32] {aps.mem_id = 116 : i64, endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, slot = 0 : i32, starttime = 120 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_63, %10[%c18_i32] {aps.mem_id = 115 : i64, endtime = 121 : i32, ref_endtime = 121 : i32, ref_starttime = 120 : i32, slot = 0 : i32, starttime = 120 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %9[%c19_i32] {aps.mem_id = 114 : i64, endtime = 122 : i32, ref_endtime = 122 : i32, ref_starttime = 121 : i32, slot = 0 : i32, starttime = 121 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_64, %3[%c19_i32] {aps.mem_id = 113 : i64, endtime = 74 : i32, ref_endtime = 74 : i32, ref_starttime = 73 : i32, slot = 0 : i32, starttime = 73 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_65, %2[%c20_i32] {aps.mem_id = 112 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_44, %3[%c20_i32] {aps.mem_id = 111 : i64, endtime = 75 : i32, ref_endtime = 75 : i32, ref_starttime = 74 : i32, slot = 0 : i32, starttime = 74 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_66, %10[%c19_i32] {aps.mem_id = 110 : i64, endtime = 122 : i32, ref_endtime = 122 : i32, ref_starttime = 121 : i32, slot = 0 : i32, starttime = 121 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_67, %9[%c20_i32] {aps.mem_id = 109 : i64, endtime = 123 : i32, ref_endtime = 123 : i32, ref_starttime = 122 : i32, slot = 0 : i32, starttime = 122 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_68, %10[%c20_i32] {aps.mem_id = 108 : i64, endtime = 123 : i32, ref_endtime = 123 : i32, ref_starttime = 122 : i32, slot = 0 : i32, starttime = 122 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_25, %2[%c21_i32] {aps.mem_id = 107 : i64, endtime = 76 : i32, ref_endtime = 76 : i32, ref_starttime = 75 : i32, slot = 0 : i32, starttime = 75 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_69, %3[%c21_i32] {aps.mem_id = 106 : i64, endtime = 76 : i32, ref_endtime = 76 : i32, ref_starttime = 75 : i32, slot = 0 : i32, starttime = 75 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_49, %2[%c22_i32] {aps.mem_id = 105 : i64, endtime = 77 : i32, ref_endtime = 77 : i32, ref_starttime = 76 : i32, slot = 0 : i32, starttime = 76 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_70, %9[%c21_i32] {aps.mem_id = 104 : i64, endtime = 124 : i32, ref_endtime = 124 : i32, ref_starttime = 123 : i32, slot = 0 : i32, starttime = 123 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_71, %10[%c21_i32] {aps.mem_id = 103 : i64, endtime = 124 : i32, ref_endtime = 124 : i32, ref_starttime = 123 : i32, slot = 0 : i32, starttime = 123 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_72, %9[%c22_i32] {aps.mem_id = 102 : i64, endtime = 125 : i32, ref_endtime = 125 : i32, ref_starttime = 124 : i32, slot = 0 : i32, starttime = 124 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_73, %3[%c22_i32] {aps.mem_id = 101 : i64, endtime = 77 : i32, ref_endtime = 77 : i32, ref_starttime = 76 : i32, slot = 0 : i32, starttime = 76 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_74, %2[%c23_i32] {aps.mem_id = 100 : i64, endtime = 78 : i32, ref_endtime = 78 : i32, ref_starttime = 77 : i32, slot = 0 : i32, starttime = 77 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_54, %3[%c23_i32] {aps.mem_id = 99 : i64, endtime = 78 : i32, ref_endtime = 78 : i32, ref_starttime = 77 : i32, slot = 0 : i32, starttime = 77 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_75, %10[%c22_i32] {aps.mem_id = 98 : i64, endtime = 125 : i32, ref_endtime = 125 : i32, ref_starttime = 124 : i32, slot = 0 : i32, starttime = 124 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_76, %9[%c23_i32] {aps.mem_id = 97 : i64, endtime = 126 : i32, ref_endtime = 126 : i32, ref_starttime = 125 : i32, slot = 0 : i32, starttime = 125 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_36, %10[%c23_i32] {aps.mem_id = 96 : i64, endtime = 126 : i32, ref_endtime = 126 : i32, ref_starttime = 125 : i32, slot = 0 : i32, starttime = 125 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_77, %2[%c24_i32] {aps.mem_id = 95 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_78, %3[%c24_i32] {aps.mem_id = 94 : i64, endtime = 79 : i32, ref_endtime = 79 : i32, ref_starttime = 78 : i32, slot = 0 : i32, starttime = 78 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_59, %2[%c25_i32] {aps.mem_id = 93 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_79, %9[%c24_i32] {aps.mem_id = 92 : i64, endtime = 127 : i32, ref_endtime = 127 : i32, ref_starttime = 126 : i32, slot = 0 : i32, starttime = 126 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_45, %10[%c24_i32] {aps.mem_id = 91 : i64, endtime = 127 : i32, ref_endtime = 127 : i32, ref_starttime = 126 : i32, slot = 0 : i32, starttime = 126 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_12, %9[%c25_i32] {aps.mem_id = 90 : i64, endtime = 128 : i32, ref_endtime = 128 : i32, ref_starttime = 127 : i32, slot = 0 : i32, starttime = 127 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_80, %3[%c25_i32] {aps.mem_id = 89 : i64, endtime = 80 : i32, ref_endtime = 80 : i32, ref_starttime = 79 : i32, slot = 0 : i32, starttime = 79 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_81, %2[%c26_i32] {aps.mem_id = 88 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_0, %3[%c26_i32] {aps.mem_id = 87 : i64, endtime = 81 : i32, ref_endtime = 81 : i32, ref_starttime = 80 : i32, slot = 0 : i32, starttime = 80 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_82, %10[%c25_i32] {aps.mem_id = 86 : i64, endtime = 128 : i32, ref_endtime = 128 : i32, ref_starttime = 127 : i32, slot = 0 : i32, starttime = 127 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_83, %9[%c26_i32] {aps.mem_id = 85 : i64, endtime = 129 : i32, ref_endtime = 129 : i32, ref_starttime = 128 : i32, slot = 0 : i32, starttime = 128 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_62, %10[%c26_i32] {aps.mem_id = 84 : i64, endtime = 129 : i32, ref_endtime = 129 : i32, ref_starttime = 128 : i32, slot = 0 : i32, starttime = 128 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_84, %2[%c27_i32] {aps.mem_id = 83 : i64, endtime = 82 : i32, ref_endtime = 82 : i32, ref_starttime = 81 : i32, slot = 0 : i32, starttime = 81 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_85, %3[%c27_i32] {aps.mem_id = 82 : i64, endtime = 82 : i32, ref_endtime = 82 : i32, ref_starttime = 81 : i32, slot = 0 : i32, starttime = 81 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_68, %2[%c28_i32] {aps.mem_id = 81 : i64, endtime = 83 : i32, ref_endtime = 83 : i32, ref_starttime = 82 : i32, slot = 0 : i32, starttime = 82 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_86, %9[%c27_i32] {aps.mem_id = 80 : i64, endtime = 130 : i32, ref_endtime = 130 : i32, ref_starttime = 129 : i32, slot = 0 : i32, starttime = 129 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_87, %10[%c27_i32] {aps.mem_id = 79 : i64, endtime = 130 : i32, ref_endtime = 130 : i32, ref_starttime = 129 : i32, slot = 0 : i32, starttime = 129 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_88, %9[%c28_i32] {aps.mem_id = 78 : i64, endtime = 131 : i32, ref_endtime = 131 : i32, ref_starttime = 130 : i32, slot = 0 : i32, starttime = 130 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_89, %3[%c28_i32] {aps.mem_id = 77 : i64, endtime = 83 : i32, ref_endtime = 83 : i32, ref_starttime = 82 : i32, slot = 0 : i32, starttime = 82 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_90, %2[%c29_i32] {aps.mem_id = 76 : i64, endtime = 84 : i32, ref_endtime = 84 : i32, ref_starttime = 83 : i32, slot = 0 : i32, starttime = 83 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_72, %3[%c29_i32] {aps.mem_id = 75 : i64, endtime = 84 : i32, ref_endtime = 84 : i32, ref_starttime = 83 : i32, slot = 0 : i32, starttime = 83 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_35, %10[%c28_i32] {aps.mem_id = 74 : i64, endtime = 131 : i32, ref_endtime = 131 : i32, ref_starttime = 130 : i32, slot = 0 : i32, starttime = 130 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_70, %9[%c29_i32] {aps.mem_id = 73 : i64, endtime = 132 : i32, ref_endtime = 132 : i32, ref_starttime = 131 : i32, slot = 0 : i32, starttime = 131 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_45, %10[%c29_i32] {aps.mem_id = 72 : i64, endtime = 132 : i32, ref_endtime = 132 : i32, ref_starttime = 131 : i32, slot = 0 : i32, starttime = 131 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_13, %2[%c30_i32] {aps.mem_id = 71 : i64, endtime = 85 : i32, ref_endtime = 85 : i32, ref_starttime = 84 : i32, slot = 0 : i32, starttime = 84 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_91, %3[%c30_i32] {aps.mem_id = 70 : i64, endtime = 85 : i32, ref_endtime = 85 : i32, ref_starttime = 84 : i32, slot = 0 : i32, starttime = 84 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_36, %2[%c31_i32] {aps.mem_id = 69 : i64, endtime = 86 : i32, ref_endtime = 86 : i32, ref_starttime = 85 : i32, slot = 0 : i32, starttime = 85 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_92, %9[%c30_i32] {aps.mem_id = 68 : i64, endtime = 133 : i32, ref_endtime = 133 : i32, ref_starttime = 132 : i32, slot = 0 : i32, starttime = 132 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_93, %10[%c30_i32] {aps.mem_id = 67 : i64, endtime = 133 : i32, ref_endtime = 133 : i32, ref_starttime = 132 : i32, slot = 0 : i32, starttime = 132 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_89, %9[%c31_i32] {aps.mem_id = 66 : i64, endtime = 134 : i32, ref_endtime = 134 : i32, ref_starttime = 133 : i32, slot = 0 : i32, starttime = 133 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_94, %3[%c31_i32] {aps.mem_id = 65 : i64, endtime = 86 : i32, ref_endtime = 86 : i32, ref_starttime = 85 : i32, slot = 0 : i32, starttime = 85 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_95, %2[%c32_i32] {aps.mem_id = 64 : i64, endtime = 87 : i32, ref_endtime = 87 : i32, ref_starttime = 86 : i32, slot = 0 : i32, starttime = 86 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_12, %3[%c32_i32] {aps.mem_id = 63 : i64, endtime = 87 : i32, ref_endtime = 87 : i32, ref_starttime = 86 : i32, slot = 0 : i32, starttime = 86 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_96, %10[%c31_i32] {aps.mem_id = 62 : i64, endtime = 134 : i32, ref_endtime = 134 : i32, ref_starttime = 133 : i32, slot = 0 : i32, starttime = 133 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_97, %9[%c32_i32] {aps.mem_id = 61 : i64, endtime = 135 : i32, ref_endtime = 135 : i32, ref_starttime = 134 : i32, slot = 0 : i32, starttime = 134 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_70, %10[%c32_i32] {aps.mem_id = 60 : i64, endtime = 135 : i32, ref_endtime = 135 : i32, ref_starttime = 134 : i32, slot = 0 : i32, starttime = 134 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_98, %2[%c33_i32] {aps.mem_id = 59 : i64, endtime = 88 : i32, ref_endtime = 88 : i32, ref_starttime = 87 : i32, slot = 0 : i32, starttime = 87 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_12, %3[%c33_i32] {aps.mem_id = 58 : i64, endtime = 88 : i32, ref_endtime = 88 : i32, ref_starttime = 87 : i32, slot = 0 : i32, starttime = 87 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_62, %2[%c34_i32] {aps.mem_id = 57 : i64, endtime = 89 : i32, ref_endtime = 89 : i32, ref_starttime = 88 : i32, slot = 0 : i32, starttime = 88 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_99, %9[%c33_i32] {aps.mem_id = 56 : i64, endtime = 136 : i32, ref_endtime = 136 : i32, ref_starttime = 135 : i32, slot = 0 : i32, starttime = 135 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_100, %10[%c33_i32] {aps.mem_id = 55 : i64, endtime = 136 : i32, ref_endtime = 136 : i32, ref_starttime = 135 : i32, slot = 0 : i32, starttime = 135 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_101, %9[%c34_i32] {aps.mem_id = 54 : i64, endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, slot = 0 : i32, starttime = 136 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_102, %3[%c34_i32] {aps.mem_id = 53 : i64, endtime = 89 : i32, ref_endtime = 89 : i32, ref_starttime = 88 : i32, slot = 0 : i32, starttime = 88 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_103, %2[%c35_i32] {aps.mem_id = 52 : i64, endtime = 90 : i32, ref_endtime = 90 : i32, ref_starttime = 89 : i32, slot = 0 : i32, starttime = 89 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_104, %3[%c35_i32] {aps.mem_id = 51 : i64, endtime = 90 : i32, ref_endtime = 90 : i32, ref_starttime = 89 : i32, slot = 0 : i32, starttime = 89 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_105, %10[%c34_i32] {aps.mem_id = 50 : i64, endtime = 137 : i32, ref_endtime = 137 : i32, ref_starttime = 136 : i32, slot = 0 : i32, starttime = 136 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_106, %9[%c35_i32] {aps.mem_id = 49 : i64, endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, slot = 0 : i32, starttime = 137 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_107, %10[%c35_i32] {aps.mem_id = 48 : i64, endtime = 138 : i32, ref_endtime = 138 : i32, ref_starttime = 137 : i32, slot = 0 : i32, starttime = 137 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_30, %2[%c36_i32] {aps.mem_id = 47 : i64, endtime = 91 : i32, ref_endtime = 91 : i32, ref_starttime = 90 : i32, slot = 0 : i32, starttime = 90 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_108, %3[%c36_i32] {aps.mem_id = 46 : i64, endtime = 91 : i32, ref_endtime = 91 : i32, ref_starttime = 90 : i32, slot = 0 : i32, starttime = 90 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_109, %2[%c37_i32] {aps.mem_id = 45 : i64, endtime = 92 : i32, ref_endtime = 92 : i32, ref_starttime = 91 : i32, slot = 0 : i32, starttime = 91 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_110, %9[%c36_i32] {aps.mem_id = 44 : i64, endtime = 139 : i32, ref_endtime = 139 : i32, ref_starttime = 138 : i32, slot = 0 : i32, starttime = 138 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_111, %10[%c36_i32] {aps.mem_id = 43 : i64, endtime = 139 : i32, ref_endtime = 139 : i32, ref_starttime = 138 : i32, slot = 0 : i32, starttime = 138 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_112, %9[%c37_i32] {aps.mem_id = 42 : i64, endtime = 140 : i32, ref_endtime = 140 : i32, ref_starttime = 139 : i32, slot = 0 : i32, starttime = 139 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_113, %3[%c37_i32] {aps.mem_id = 41 : i64, endtime = 92 : i32, ref_endtime = 92 : i32, ref_starttime = 91 : i32, slot = 0 : i32, starttime = 91 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_88, %2[%c38_i32] {aps.mem_id = 40 : i64, endtime = 93 : i32, ref_endtime = 93 : i32, ref_starttime = 92 : i32, slot = 0 : i32, starttime = 92 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_89, %3[%c38_i32] {aps.mem_id = 39 : i64, endtime = 93 : i32, ref_endtime = 93 : i32, ref_starttime = 92 : i32, slot = 0 : i32, starttime = 92 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_114, %10[%c37_i32] {aps.mem_id = 38 : i64, endtime = 140 : i32, ref_endtime = 140 : i32, ref_starttime = 139 : i32, slot = 0 : i32, starttime = 139 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_115, %9[%c38_i32] {aps.mem_id = 37 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_116, %10[%c38_i32] {aps.mem_id = 36 : i64, endtime = 141 : i32, ref_endtime = 141 : i32, ref_starttime = 140 : i32, slot = 0 : i32, starttime = 140 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_49, %2[%c39_i32] {aps.mem_id = 35 : i64, endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, slot = 0 : i32, starttime = 93 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_117, %3[%c39_i32] {aps.mem_id = 34 : i64, endtime = 94 : i32, ref_endtime = 94 : i32, ref_starttime = 93 : i32, slot = 0 : i32, starttime = 93 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_70, %2[%c40_i32] {aps.mem_id = 33 : i64, endtime = 95 : i32, ref_endtime = 95 : i32, ref_starttime = 94 : i32, slot = 0 : i32, starttime = 94 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_118, %9[%c39_i32] {aps.mem_id = 32 : i64, endtime = 142 : i32, ref_endtime = 142 : i32, ref_starttime = 141 : i32, slot = 0 : i32, starttime = 141 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_119, %10[%c39_i32] {aps.mem_id = 31 : i64, endtime = 142 : i32, ref_endtime = 142 : i32, ref_starttime = 141 : i32, slot = 0 : i32, starttime = 141 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_120, %9[%c40_i32] {aps.mem_id = 30 : i64, endtime = 143 : i32, ref_endtime = 143 : i32, ref_starttime = 142 : i32, slot = 0 : i32, starttime = 142 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_121, %3[%c40_i32] {aps.mem_id = 29 : i64, endtime = 95 : i32, ref_endtime = 95 : i32, ref_starttime = 94 : i32, slot = 0 : i32, starttime = 94 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_122, %2[%c41_i32] {aps.mem_id = 28 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_110, %3[%c41_i32] {aps.mem_id = 27 : i64, endtime = 96 : i32, ref_endtime = 96 : i32, ref_starttime = 95 : i32, slot = 0 : i32, starttime = 95 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_123, %10[%c40_i32] {aps.mem_id = 26 : i64, endtime = 143 : i32, ref_endtime = 143 : i32, ref_starttime = 142 : i32, slot = 0 : i32, starttime = 142 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_124, %9[%c41_i32] {aps.mem_id = 25 : i64, endtime = 144 : i32, ref_endtime = 144 : i32, ref_starttime = 143 : i32, slot = 0 : i32, starttime = 143 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_5, %10[%c41_i32] {aps.mem_id = 24 : i64, endtime = 144 : i32, ref_endtime = 144 : i32, ref_starttime = 143 : i32, slot = 0 : i32, starttime = 143 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_125, %2[%c42_i32] {aps.mem_id = 23 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_126, %3[%c42_i32] {aps.mem_id = 22 : i64, endtime = 97 : i32, ref_endtime = 97 : i32, ref_starttime = 96 : i32, slot = 0 : i32, starttime = 96 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_107, %2[%c43_i32] {aps.mem_id = 21 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_127, %9[%c42_i32] {aps.mem_id = 20 : i64, endtime = 145 : i32, ref_endtime = 145 : i32, ref_starttime = 144 : i32, slot = 0 : i32, starttime = 144 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_128, %10[%c42_i32] {aps.mem_id = 19 : i64, endtime = 145 : i32, ref_endtime = 145 : i32, ref_starttime = 144 : i32, slot = 0 : i32, starttime = 144 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_11, %9[%c43_i32] {aps.mem_id = 18 : i64, endtime = 146 : i32, ref_endtime = 146 : i32, ref_starttime = 145 : i32, slot = 0 : i32, starttime = 145 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_129, %3[%c43_i32] {aps.mem_id = 17 : i64, endtime = 98 : i32, ref_endtime = 98 : i32, ref_starttime = 97 : i32, slot = 0 : i32, starttime = 97 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_130, %2[%c44_i32] {aps.mem_id = 16 : i64, endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, slot = 0 : i32, starttime = 98 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_112, %3[%c44_i32] {aps.mem_id = 15 : i64, endtime = 99 : i32, ref_endtime = 99 : i32, ref_starttime = 98 : i32, slot = 0 : i32, starttime = 98 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_131, %10[%c43_i32] {aps.mem_id = 14 : i64, endtime = 146 : i32, ref_endtime = 146 : i32, ref_starttime = 145 : i32, slot = 0 : i32, starttime = 145 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_132, %9[%c44_i32] {aps.mem_id = 13 : i64, endtime = 147 : i32, ref_endtime = 147 : i32, ref_starttime = 146 : i32, slot = 0 : i32, starttime = 146 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_17, %10[%c44_i32] {aps.mem_id = 12 : i64, endtime = 147 : i32, ref_endtime = 147 : i32, ref_starttime = 146 : i32, slot = 0 : i32, starttime = 146 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_133, %2[%c45_i32] {aps.mem_id = 11 : i64, endtime = 100 : i32, ref_endtime = 100 : i32, ref_starttime = 99 : i32, slot = 0 : i32, starttime = 99 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_134, %3[%c45_i32] {aps.mem_id = 10 : i64, endtime = 100 : i32, ref_endtime = 100 : i32, ref_starttime = 99 : i32, slot = 0 : i32, starttime = 99 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_116, %2[%c46_i32] {aps.mem_id = 9 : i64, endtime = 101 : i32, ref_endtime = 101 : i32, ref_starttime = 100 : i32, slot = 0 : i32, starttime = 100 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_135, %9[%c45_i32] {aps.mem_id = 8 : i64, endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, slot = 0 : i32, starttime = 147 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_136, %10[%c45_i32] {aps.mem_id = 7 : i64, endtime = 148 : i32, ref_endtime = 148 : i32, ref_starttime = 147 : i32, slot = 0 : i32, starttime = 147 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_9, %9[%c46_i32] {aps.mem_id = 6 : i64, endtime = 149 : i32, ref_endtime = 149 : i32, ref_starttime = 148 : i32, slot = 0 : i32, starttime = 148 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_116, %3[%c46_i32] {aps.mem_id = 5 : i64, endtime = 101 : i32, ref_endtime = 101 : i32, ref_starttime = 100 : i32, slot = 0 : i32, starttime = 100 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_101, %2[%c47_i32] {aps.mem_id = 4 : i64, endtime = 102 : i32, ref_endtime = 102 : i32, ref_starttime = 101 : i32, slot = 0 : i32, starttime = 101 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst, %3[%c47_i32] {aps.mem_id = 3 : i64, endtime = 102 : i32, ref_endtime = 102 : i32, ref_starttime = 101 : i32, slot = 0 : i32, starttime = 101 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_59, %10[%c46_i32] {aps.mem_id = 2 : i64, endtime = 149 : i32, ref_endtime = 149 : i32, ref_starttime = 148 : i32, slot = 0 : i32, starttime = 148 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_17, %9[%c47_i32] {aps.mem_id = 1 : i64, endtime = 150 : i32, ref_endtime = 150 : i32, ref_starttime = 149 : i32, slot = 0 : i32, starttime = 149 : i32} : f32, memref<48xf32>, i32
      aps.write_smem %cst_2, %10[%c47_i32] {aps.mem_id = 0 : i64, endtime = 150 : i32, ref_endtime = 150 : i32, ref_starttime = 149 : i32, slot = 0 : i32, starttime = 149 : i32} : f32, memref<48xf32>, i32
      aps.copy_by @busitfc, (%2, %3)[%c0_i32], %1, %c16_i32 {after = array<i64>, after_gap = array<i64>, endtime = 111 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 9 : i64, memory_issue_cycle = 0 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 111 : i32, ref_starttime = 102 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 102 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 35 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %16 = tor.addi %1 %c64_i32 on (0 to 1) {dump = "unknown_210", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3)[%c16_i32], %16, %c16_i32 {after = array<i64: 35, 35>, after_gap = array<i64: 1, 1>, after_segment = 0 : i64, endtime = 121 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 19 : i64, memory_issue_cycle = 1 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 121 : i32, ref_starttime = 103 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 103 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 36 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %17 = tor.addi %1 %c128_i32 on (0 to 1) {dump = "unknown_211", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3)[%c32_i32], %17, %c16_i32 {after = array<i64: 36, 35, 36>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 136 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 29 : i64, memory_issue_cycle = 10 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 136 : i32, ref_starttime = 117 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 117 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 37 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %18 = tor.addi %1 %c192_i32 on (0 to 1) {dump = "unknown_212", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3)[%c48_i32], %18, %c16_i32 {after = array<i64: 37, 36, 37>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 146 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 39 : i64, memory_issue_cycle = 20 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 146 : i32, ref_starttime = 127 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 127 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 38 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %19 = tor.addi %1 %c256_i32 on (0 to 1) {dump = "unknown_213", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3)[%c64_i32], %19, %c16_i32 {after = array<i64: 38, 37, 38>, after_gap = array<i64: 10, 20, 10>, after_segment = 3 : i64, endtime = 156 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 49 : i64, memory_issue_cycle = 30 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 156 : i32, ref_starttime = 137 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 137 : i32, tl_channel = 0 : i32, transaction_group = 4 : i64, transaction_id = 39 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %20 = tor.addi %1 %c320_i32 on (0 to 1) {dump = "unknown_214", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%2, %3)[%c80_i32], %20, %c16_i32 {after = array<i64: 39, 38, 39>, after_gap = array<i64: 10, 20, 10>, after_segment = 4 : i64, endtime = 166 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 59 : i64, memory_issue_cycle = 40 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 166 : i32, ref_starttime = 147 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 147 : i32, tl_channel = 1 : i32, transaction_group = 4 : i64, transaction_id = 40 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      aps.copy_by @busitfc, (%9, %10)[%c0_i32], %0, %c16_i32 {after = array<i64: 40, 40, 39, 40>, after_gap = array<i64: 20, 20, 20, 20>, endtime = 176 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 0 : i64, memory_completion_cycle = 69 : i64, memory_issue_cycle = 60 : i64, memory_latency = 9 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 176 : i32, ref_starttime = 167 : i32, required_alignment = 64 : i64, segment = 0 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 167 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 41 : i64, transfer_bytes = 64 : i64, transfer_offset = 0 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %21 = tor.addi %0 %c64_i32 on (0 to 1) {dump = "unknown_215", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%9, %10)[%c16_i32], %21, %c16_i32 {after = array<i64: 41, 40, 41>, after_gap = array<i64: 1, 20, 1>, after_segment = 0 : i64, endtime = 186 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 1 : i64, memory_completion_cycle = 79 : i64, memory_issue_cycle = 61 : i64, memory_latency = 18 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 186 : i32, ref_starttime = 168 : i32, required_alignment = 64 : i64, segment = 1 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 168 : i32, tl_channel = 1 : i32, transaction_group = 5 : i64, transaction_id = 42 : i64, transfer_bytes = 64 : i64, transfer_offset = 64 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %22 = tor.addi %0 %c128_i32 on (0 to 1) {dump = "unknown_216", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%9, %10)[%c32_i32], %22, %c16_i32 {after = array<i64: 42, 41, 42>, after_gap = array<i64: 9, 10, 9>, after_segment = 1 : i64, endtime = 201 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 2 : i64, memory_completion_cycle = 89 : i64, memory_issue_cycle = 70 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 201 : i32, ref_starttime = 182 : i32, required_alignment = 64 : i64, segment = 2 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 182 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 43 : i64, transfer_bytes = 64 : i64, transfer_offset = 128 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %23 = tor.addi %0 %c192_i32 on (0 to 1) {dump = "unknown_217", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%9, %10)[%c48_i32], %23, %c16_i32 {after = array<i64: 43, 42, 43>, after_gap = array<i64: 10, 19, 10>, after_segment = 2 : i64, endtime = 211 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 3 : i64, memory_completion_cycle = 99 : i64, memory_issue_cycle = 80 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 211 : i32, ref_starttime = 192 : i32, required_alignment = 64 : i64, segment = 3 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 192 : i32, tl_channel = 1 : i32, transaction_group = 5 : i64, transaction_id = 44 : i64, transfer_bytes = 64 : i64, transfer_offset = 192 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %24 = tor.addi %0 %c256_i32 on (0 to 1) {dump = "unknown_218", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%9, %10)[%c64_i32], %24, %c16_i32 {after = array<i64: 44, 43, 44>, after_gap = array<i64: 10, 20, 10>, after_segment = 3 : i64, endtime = 221 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 4 : i64, memory_completion_cycle = 109 : i64, memory_issue_cycle = 90 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 221 : i32, ref_starttime = 202 : i32, required_alignment = 64 : i64, segment = 4 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 202 : i32, tl_channel = 0 : i32, transaction_group = 5 : i64, transaction_id = 45 : i64, transfer_bytes = 64 : i64, transfer_offset = 256 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      %25 = tor.addi %0 %c320_i32 on (0 to 1) {dump = "unknown_219", ref_endtime = 1 : i32, ref_starttime = 0 : i32} : (i32, i32) -> i32
      aps.copy_by @busitfc, (%9, %10)[%c80_i32], %25, %c16_i32 {after = array<i64: 45, 44, 45>, after_gap = array<i64: 10, 20, 10>, after_segment = 4 : i64, endtime = 231 : i32, estimated_latency = 59 : i64, interface_cost = 107 : i64, interface_solver = "active-set-dp", issue_segment = 5 : i64, memory_completion_cycle = 119 : i64, memory_issue_cycle = 100 : i64, memory_latency = 19 : i64, memory_order_solver = "memoized-relative-window", memory_schedule_latency = 119 : i64, memory_search_hits = 0 : i64, memory_search_states = 15 : i64, partition_cyclic_array = [1 : i32], partition_dim_array = [0 : i32], partition_factor_array = [2 : i32], ref_endtime = 231 : i32, ref_starttime = 212 : i32, required_alignment = 64 : i64, segment = 5 : i64, segment_count = 6 : i64, solver_objective = 2.150000e+02 : f64, starttime = 212 : i32, tl_channel = 1 : i32, transaction_group = 5 : i64, transaction_id = 46 : i64, transfer_bytes = 64 : i64, transfer_offset = 320 : i64} : (memref<48xf32>, memref<48xf32>), i32, i32, i32
      tor.return
    }
  } {HoistConstCondIfOp = 1 : i32, schedule = true}
}

