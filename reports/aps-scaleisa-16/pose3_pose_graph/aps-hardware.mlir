#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (6)>
#map2 = affine_map<() -> (18)>
#map3 = affine_map<() -> (30)>
#map4 = affine_map<() -> (17)>
#map5 = affine_map<() -> (7)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_5__l4p_buffer_8", type = memref<180xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l4p_buffer_8"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l4p_buffer_7", type = memref<180xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l4p_buffer_7"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l4p_buffer_6", type = memref<108xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l4p_buffer_6"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l4p_buffer_5", type = memref<30xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l4p_buffer_5"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l4p_buffer_4", type = memref<30xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l4p_buffer_4"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l4p_buffer_3", type = memref<108xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l4p_buffer_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l4p_buffer", type = memref<108xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l4p_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_43__l0u_l1u_buffer_2", type = memref<108xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [2], scaleisa.scratchpad, var_name = "au_generated_candidate_43__l0u_l1u_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_43__l0u_l1u_buffer", type = memref<108xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [2], scaleisa.scratchpad, var_name = "au_generated_candidate_43__l0u_l1u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_40__l0u_l1u_buffer_1", type = memref<180xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [2], scaleisa.scratchpad, var_name = "au_generated_candidate_40__l0u_l1u_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_40__l0u_l1u_buffer", type = memref<180xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [2], scaleisa.scratchpad, var_name = "au_generated_candidate_40__l0u_l1u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_32__direct_buffer_0", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_32__direct_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_32__direct_buffer", type = memref<108xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_32__direct_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_32__direct"}> ({
  ^bb0(%arg21: i5, %arg22: i5, %arg23: i5):
    %327 = "aps.read_irf"(%arg21) : (i5) -> i32
    %328 = "aps.read_irf"(%arg22) : (i5) -> i32
    %329 = "memref.get_global"() <{name = @au_generated_candidate_32__direct_buffer}> : () -> memref<108xf32>
    %330 = "arith.constant"() <{value = 108 : i32}> : () -> i32
    %331 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%327, %329, %331, %330) <{direction = 0 : i32}> : (i32, memref<108xf32>, i32, i32) -> ()
    %332 = "memref.get_global"() <{name = @au_generated_candidate_32__direct_buffer_0}> : () -> memref<1xi32>
    %333 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %334 = "arith.addi"(%328, %333) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %335 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %336 = "aps.load"(%334) : (i32) -> i32
    "aps.write_smem"(%336, %332, %335) : (i32, memref<1xi32>, i32) -> ()
    %337 = "arith.constant"() <{value = 8.300000e+01 : f32}> : () -> f32
    %338 = "arith.constant"() <{value = 1.900000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg24: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg25: index):
        %343 = "arith.index_cast"(%arg25) : (index) -> i32
        %344 = "arith.index_cast"(%arg24) : (index) -> i32
        %345 = "arith.index_cast"(%arg25) : (index) -> i32
        %346 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %347 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %348 = "arith.constant"() <{value = 6 : i32}> : () -> i32
        %349 = "arith.muli"(%344, %348) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %350 = "arith.addi"(%347, %349) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %351 = "arith.addi"(%350, %345) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %352 = "arith.addi"(%346, %351) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %353 = "aps.read_smem"(%329, %352) : (memref<108xf32>, i32) -> f32
        %354 = "arith.sitofp"(%343) : (i32) -> f32
        %355 = "arith.mulf"(%354, %338) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %356 = "arith.addf"(%355, %337) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %357 = "arith.mulf"(%353, %356) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %358 = "arith.fptosi"(%357) : (f32) -> i32
        %359 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %360 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %361 = "arith.addi"(%359, %360) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %362 = "aps.read_smem"(%332, %361) : (memref<1xi32>, i32) -> i32
        %363 = "arith.addi"(%358, %362) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %364 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %365 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %366 = "arith.addi"(%364, %365) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%363, %332, %366) : (i32, memref<1xi32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %339 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %340 = "arith.addi"(%328, %339) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %341 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %342 = "aps.read_smem"(%332, %341) : (memref<1xi32>, i32) -> i32
    "aps.store"(%342, %340) : (i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<18x6xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<18x6xf32>, memref<i32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_40__l0u_l1u"}> ({
  ^bb0(%arg16: i5, %arg17: i5, %arg18: i5):
    %300 = "aps.read_irf"(%arg16) : (i5) -> i32
    %301 = "aps.read_irf"(%arg17) : (i5) -> i32
    %302 = "memref.get_global"() <{name = @au_generated_candidate_40__l0u_l1u_buffer}> : () -> memref<180xf32>
    %303 = "arith.constant"() <{value = 180 : i32}> : () -> i32
    %304 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%300, %302, %304, %303) <{direction = 0 : i32}> : (i32, memref<180xf32>, i32, i32) -> ()
    %305 = "memref.get_global"() <{name = @au_generated_candidate_40__l0u_l1u_buffer_1}> : () -> memref<180xf32>
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg19: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg20: index):
        %308 = "arith.index_cast"(%arg19) : (index) -> i32
        %309 = "arith.index_cast"(%arg20) : (index) -> i32
        %310 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %311 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %312 = "arith.constant"() <{value = 6 : i32}> : () -> i32
        %313 = "arith.muli"(%308, %312) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %314 = "arith.addi"(%311, %313) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %315 = "arith.addi"(%314, %309) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %316 = "arith.addi"(%310, %315) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %317 = "aps.read_smem"(%302, %316) : (memref<180xf32>, i32) -> f32
        %318 = "arith.index_cast"(%arg19) : (index) -> i32
        %319 = "arith.index_cast"(%arg20) : (index) -> i32
        %320 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %321 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %322 = "arith.constant"() <{value = 6 : i32}> : () -> i32
        %323 = "arith.muli"(%318, %322) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %324 = "arith.addi"(%321, %323) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %325 = "arith.addi"(%324, %319) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %326 = "arith.addi"(%320, %325) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%317, %305, %326) : (f32, memref<180xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %306 = "arith.constant"() <{value = 180 : i32}> : () -> i32
    %307 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%301, %305, %307, %306) <{direction = 1 : i32}> : (i32, memref<180xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<30x6xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<30x6xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<30x6xf32>, memref<30x6xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_43__l0u_l1u"}> ({
  ^bb0(%arg11: i5, %arg12: i5, %arg13: i5):
    %273 = "aps.read_irf"(%arg11) : (i5) -> i32
    %274 = "aps.read_irf"(%arg12) : (i5) -> i32
    %275 = "memref.get_global"() <{name = @au_generated_candidate_43__l0u_l1u_buffer}> : () -> memref<108xf32>
    %276 = "arith.constant"() <{value = 108 : i32}> : () -> i32
    %277 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%273, %275, %277, %276) <{direction = 0 : i32}> : (i32, memref<108xf32>, i32, i32) -> ()
    %278 = "memref.get_global"() <{name = @au_generated_candidate_43__l0u_l1u_buffer_2}> : () -> memref<108xf32>
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg14: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg15: index):
        %281 = "arith.index_cast"(%arg14) : (index) -> i32
        %282 = "arith.index_cast"(%arg15) : (index) -> i32
        %283 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %284 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %285 = "arith.constant"() <{value = 6 : i32}> : () -> i32
        %286 = "arith.muli"(%281, %285) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %287 = "arith.addi"(%284, %286) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %288 = "arith.addi"(%287, %282) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %289 = "arith.addi"(%283, %288) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %290 = "aps.read_smem"(%275, %289) : (memref<108xf32>, i32) -> f32
        %291 = "arith.index_cast"(%arg14) : (index) -> i32
        %292 = "arith.index_cast"(%arg15) : (index) -> i32
        %293 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %294 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %295 = "arith.constant"() <{value = 6 : i32}> : () -> i32
        %296 = "arith.muli"(%291, %295) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %297 = "arith.addi"(%294, %296) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %298 = "arith.addi"(%297, %292) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %299 = "arith.addi"(%293, %298) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%290, %278, %299) : (f32, memref<108xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %279 = "arith.constant"() <{value = 108 : i32}> : () -> i32
    %280 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%274, %278, %280, %279) <{direction = 1 : i32}> : (i32, memref<108xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 2 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<18x6xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<18x6xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<18x6xf32>, memref<18x6xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_5__l4p"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_5__l4p_buffer}> : () -> memref<108xf32>
    %3 = "arith.constant"() <{value = 108 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %2, %4, %3) <{direction = 0 : i32}> : (i32, memref<108xf32>, i32, i32) -> ()
    %5 = "memref.get_global"() <{name = @au_generated_candidate_5__l4p_buffer_3}> : () -> memref<108xf32>
    %6 = "arith.constant"() <{value = 108 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%1, %5, %7, %6) <{direction = 0 : i32}> : (i32, memref<108xf32>, i32, i32) -> ()
    %8 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l4p_csr0}> : () -> i32
    %9 = "memref.get_global"() <{name = @au_generated_candidate_5__l4p_buffer_4}> : () -> memref<30xi32>
    %10 = "arith.constant"() <{value = 30 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%8, %9, %11, %10) <{direction = 0 : i32}> : (i32, memref<30xi32>, i32, i32) -> ()
    %12 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l4p_csr1}> : () -> i32
    %13 = "memref.get_global"() <{name = @au_generated_candidate_5__l4p_buffer_5}> : () -> memref<30xi32>
    %14 = "arith.constant"() <{value = 30 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%12, %13, %15, %14) <{direction = 0 : i32}> : (i32, memref<30xi32>, i32, i32) -> ()
    %16 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l4p_csr2}> : () -> i32
    %17 = "memref.get_global"() <{name = @au_generated_candidate_5__l4p_buffer_6}> : () -> memref<108xf32>
    %18 = "arith.constant"() <{value = 108 : i32}> : () -> i32
    %19 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%16, %17, %19, %18) <{direction = 0 : i32}> : (i32, memref<108xf32>, i32, i32) -> ()
    %20 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l4p_csr3}> : () -> i32
    %21 = "memref.get_global"() <{name = @au_generated_candidate_5__l4p_buffer_7}> : () -> memref<180xf32>
    %22 = "arith.constant"() <{value = 180 : i32}> : () -> i32
    %23 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%20, %21, %23, %22) <{direction = 0 : i32}> : (i32, memref<180xf32>, i32, i32) -> ()
    %24 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l4p_csr4}> : () -> i32
    %25 = "memref.get_global"() <{name = @au_generated_candidate_5__l4p_buffer_8}> : () -> memref<180xf32>
    %26 = "arith.constant"() <{value = 180 : i32}> : () -> i32
    %27 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%24, %25, %27, %26) <{direction = 0 : i32}> : (i32, memref<180xf32>, i32, i32) -> ()
    %28 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %29 = "arith.constant"() <{value = 1 : index}> : () -> index
    %30 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %31 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    %32 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %33 = "arith.constant"() <{value = 1.000000e-03 : f32}> : () -> f32
    %34 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg3: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg9: index):
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg10: index):
          %255 = "arith.index_cast"(%arg10) : (index) -> i32
          %256 = "arith.index_cast"(%arg9) : (index) -> i32
          %257 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %258 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %259 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %260 = "arith.muli"(%255, %259) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %261 = "arith.addi"(%258, %260) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %262 = "arith.addi"(%261, %256) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %263 = "arith.addi"(%257, %262) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%34, %2, %263) : (f32, memref<108xf32>, i32) -> ()
          %264 = "arith.index_cast"(%arg10) : (index) -> i32
          %265 = "arith.index_cast"(%arg9) : (index) -> i32
          %266 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %267 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %268 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %269 = "arith.muli"(%264, %268) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %270 = "arith.addi"(%267, %269) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %271 = "arith.addi"(%270, %265) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %272 = "arith.addi"(%266, %271) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%33, %5, %272) : (f32, memref<108xf32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg7: index):
        %98 = "arith.index_cast"(%arg7) : (index) -> i32
        %99 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %100 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %101 = "arith.addi"(%100, %98) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %102 = "arith.addi"(%99, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %103 = "aps.read_smem"(%9, %102) : (memref<30xi32>, i32) -> i32
        %104 = "arith.index_cast"(%arg7) : (index) -> i32
        %105 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %106 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %107 = "arith.addi"(%106, %104) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %108 = "arith.addi"(%105, %107) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %109 = "aps.read_smem"(%13, %108) : (memref<30xi32>, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg8: index):
          %110 = "arith.index_cast"(%109) : (i32) -> index
          %111 = "arith.index_cast"(%110) : (index) -> i32
          %112 = "arith.index_cast"(%arg8) : (index) -> i32
          %113 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %114 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %115 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %116 = "arith.muli"(%111, %115) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %117 = "arith.addi"(%114, %116) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %118 = "arith.addi"(%117, %112) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %119 = "arith.addi"(%113, %118) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %120 = "aps.read_smem"(%17, %119) : (memref<108xf32>, i32) -> f32
          %121 = "arith.index_cast"(%103) : (i32) -> index
          %122 = "arith.index_cast"(%121) : (index) -> i32
          %123 = "arith.index_cast"(%arg8) : (index) -> i32
          %124 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %125 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %126 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %127 = "arith.muli"(%122, %126) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %128 = "arith.addi"(%125, %127) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %129 = "arith.addi"(%128, %123) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %130 = "arith.addi"(%124, %129) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %131 = "aps.read_smem"(%17, %130) : (memref<108xf32>, i32) -> f32
          %132 = "arith.subf"(%120, %131) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %133 = "arith.index_cast"(%arg7) : (index) -> i32
          %134 = "arith.index_cast"(%arg8) : (index) -> i32
          %135 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %136 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %137 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %138 = "arith.muli"(%133, %137) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %139 = "arith.addi"(%136, %138) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %140 = "arith.addi"(%139, %134) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %141 = "arith.addi"(%135, %140) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %142 = "aps.read_smem"(%21, %141) : (memref<180xf32>, i32) -> f32
          %143 = "arith.subf"(%132, %142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %144 = "arith.index_cast"(%arg7) : (index) -> i32
          %145 = "arith.index_cast"(%arg8) : (index) -> i32
          %146 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %147 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %148 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %149 = "arith.muli"(%144, %148) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %150 = "arith.addi"(%147, %149) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %151 = "arith.addi"(%150, %145) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %152 = "arith.addi"(%146, %151) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %153 = "aps.read_smem"(%25, %152) : (memref<180xf32>, i32) -> f32
          %154 = "arith.mulf"(%153, %143) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %155 = "arith.index_cast"(%121) : (index) -> i32
          %156 = "arith.index_cast"(%arg8) : (index) -> i32
          %157 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %158 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %159 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %160 = "arith.muli"(%155, %159) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %161 = "arith.addi"(%158, %160) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %162 = "arith.addi"(%161, %156) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %163 = "arith.addi"(%157, %162) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %164 = "aps.read_smem"(%2, %163) : (memref<108xf32>, i32) -> f32
          %165 = "arith.addf"(%164, %154) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %166 = "arith.index_cast"(%121) : (index) -> i32
          %167 = "arith.index_cast"(%arg8) : (index) -> i32
          %168 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %169 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %170 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %171 = "arith.muli"(%166, %170) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %172 = "arith.addi"(%169, %171) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %173 = "arith.addi"(%172, %167) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %174 = "arith.addi"(%168, %173) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%165, %2, %174) : (f32, memref<108xf32>, i32) -> ()
          %175 = "arith.index_cast"(%110) : (index) -> i32
          %176 = "arith.index_cast"(%arg8) : (index) -> i32
          %177 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %178 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %179 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %180 = "arith.muli"(%175, %179) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %181 = "arith.addi"(%178, %180) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %182 = "arith.addi"(%181, %176) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %183 = "arith.addi"(%177, %182) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %184 = "aps.read_smem"(%2, %183) : (memref<108xf32>, i32) -> f32
          %185 = "arith.subf"(%184, %154) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %186 = "arith.index_cast"(%110) : (index) -> i32
          %187 = "arith.index_cast"(%arg8) : (index) -> i32
          %188 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %189 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %190 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %191 = "arith.muli"(%186, %190) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %192 = "arith.addi"(%189, %191) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %193 = "arith.addi"(%192, %187) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %194 = "arith.addi"(%188, %193) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%185, %2, %194) : (f32, memref<108xf32>, i32) -> ()
          %195 = "arith.index_cast"(%arg7) : (index) -> i32
          %196 = "arith.index_cast"(%arg8) : (index) -> i32
          %197 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %198 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %199 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %200 = "arith.muli"(%195, %199) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %201 = "arith.addi"(%198, %200) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %202 = "arith.addi"(%201, %196) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %203 = "arith.addi"(%197, %202) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %204 = "aps.read_smem"(%25, %203) : (memref<180xf32>, i32) -> f32
          %205 = "arith.index_cast"(%121) : (index) -> i32
          %206 = "arith.index_cast"(%arg8) : (index) -> i32
          %207 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %208 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %209 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %210 = "arith.muli"(%205, %209) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %211 = "arith.addi"(%208, %210) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %212 = "arith.addi"(%211, %206) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %213 = "arith.addi"(%207, %212) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %214 = "aps.read_smem"(%5, %213) : (memref<108xf32>, i32) -> f32
          %215 = "arith.addf"(%214, %204) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %216 = "arith.index_cast"(%121) : (index) -> i32
          %217 = "arith.index_cast"(%arg8) : (index) -> i32
          %218 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %219 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %220 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %221 = "arith.muli"(%216, %220) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %222 = "arith.addi"(%219, %221) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %223 = "arith.addi"(%222, %217) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %224 = "arith.addi"(%218, %223) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%215, %5, %224) : (f32, memref<108xf32>, i32) -> ()
          %225 = "arith.index_cast"(%arg7) : (index) -> i32
          %226 = "arith.index_cast"(%arg8) : (index) -> i32
          %227 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %228 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %229 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %230 = "arith.muli"(%225, %229) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %231 = "arith.addi"(%228, %230) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %232 = "arith.addi"(%231, %226) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %233 = "arith.addi"(%227, %232) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %234 = "aps.read_smem"(%25, %233) : (memref<180xf32>, i32) -> f32
          %235 = "arith.index_cast"(%110) : (index) -> i32
          %236 = "arith.index_cast"(%arg8) : (index) -> i32
          %237 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %238 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %239 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %240 = "arith.muli"(%235, %239) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %241 = "arith.addi"(%238, %240) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %242 = "arith.addi"(%241, %236) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %243 = "arith.addi"(%237, %242) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %244 = "aps.read_smem"(%5, %243) : (memref<108xf32>, i32) -> f32
          %245 = "arith.addf"(%244, %234) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %246 = "arith.index_cast"(%110) : (index) -> i32
          %247 = "arith.index_cast"(%arg8) : (index) -> i32
          %248 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %249 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %250 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %251 = "arith.muli"(%246, %250) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %252 = "arith.addi"(%249, %251) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %253 = "arith.addi"(%252, %247) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %254 = "arith.addi"(%248, %253) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%245, %5, %254) : (f32, memref<108xf32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) {pipeline = 1 : i64} : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg6: index):
        %90 = "arith.index_cast"(%arg6) : (index) -> i32
        %91 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %92 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %93 = "arith.constant"() <{value = 6 : i32}> : () -> i32
        %94 = "arith.muli"(%28, %93) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %95 = "arith.addi"(%92, %94) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %96 = "arith.addi"(%95, %90) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %97 = "arith.addi"(%91, %96) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%34, %17, %97) : (f32, memref<108xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
      ^bb0(%arg4: index):
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg5: index):
          %41 = "arith.index_cast"(%arg5) : (index) -> i32
          %42 = "arith.cmpi"(%41, %32) <{predicate = 2 : i64}> : (i32, i32) -> i1
          %43 = "arith.select"(%42, %31, %30) : (i1, f32, f32) -> f32
          %44 = "arith.addi"(%arg4, %29) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %45 = "arith.index_cast"(%44) : (index) -> i32
          %46 = "arith.index_cast"(%arg5) : (index) -> i32
          %47 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %48 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %49 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %50 = "arith.muli"(%45, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %51 = "arith.addi"(%48, %50) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %52 = "arith.addi"(%51, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %53 = "arith.addi"(%47, %52) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %54 = "aps.read_smem"(%2, %53) : (memref<108xf32>, i32) -> f32
          %55 = "arith.mulf"(%43, %54) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %56 = "arith.addi"(%arg4, %29) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %57 = "arith.index_cast"(%56) : (index) -> i32
          %58 = "arith.index_cast"(%arg5) : (index) -> i32
          %59 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %60 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %61 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %62 = "arith.muli"(%57, %61) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %63 = "arith.addi"(%60, %62) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %64 = "arith.addi"(%63, %58) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %65 = "arith.addi"(%59, %64) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %66 = "aps.read_smem"(%5, %65) : (memref<108xf32>, i32) -> f32
          %67 = "arith.divf"(%55, %66) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %68 = "arith.addi"(%arg4, %29) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %69 = "arith.index_cast"(%68) : (index) -> i32
          %70 = "arith.index_cast"(%arg5) : (index) -> i32
          %71 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %72 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %73 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %74 = "arith.muli"(%69, %73) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %75 = "arith.addi"(%72, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %76 = "arith.addi"(%75, %70) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %77 = "arith.addi"(%71, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %78 = "aps.read_smem"(%17, %77) : (memref<108xf32>, i32) -> f32
          %79 = "arith.subf"(%78, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %80 = "arith.addi"(%arg4, %29) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %81 = "arith.index_cast"(%80) : (index) -> i32
          %82 = "arith.index_cast"(%arg5) : (index) -> i32
          %83 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %84 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %85 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %86 = "arith.muli"(%81, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %87 = "arith.addi"(%84, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %88 = "arith.addi"(%87, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %89 = "arith.addi"(%83, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%79, %17, %89) : (f32, memref<108xf32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %35 = "arith.constant"() <{value = 108 : i32}> : () -> i32
    %36 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %2, %36, %35) <{direction = 1 : i32}> : (i32, memref<108xf32>, i32, i32) -> ()
    %37 = "arith.constant"() <{value = 108 : i32}> : () -> i32
    %38 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%1, %5, %38, %37) <{direction = 1 : i32}> : (i32, memref<108xf32>, i32, i32) -> ()
    %39 = "arith.constant"() <{value = 108 : i32}> : () -> i32
    %40 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%16, %17, %40, %39) <{direction = 1 : i32}> : (i32, memref<108xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 3 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<18x6xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<18x6xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<30xi32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2050 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<30xi32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2051 : i32, arg_index = 4 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<18x6xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2052 : i32, arg_index = 5 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<30x6xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2053 : i32, arg_index = 6 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<30x6xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<18x6xf32>, memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<18x6xf32>, memref<30x6xf32>, memref<30x6xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l4p_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l4p_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l4p_csr2", sym_visibility = "private", type = memref<i32>}> {csr_address = 2051 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l4p_csr3", sym_visibility = "private", type = memref<i32>}> {csr_address = 2052 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l4p_csr4", sym_visibility = "private", type = memref<i32>}> {csr_address = 2053 : i32} : () -> ()
}) : () -> ()

