#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (24)>
#map2 = affine_map<() -> (23)>
#map3 = affine_map<() -> (3)>
#map4 = affine_map<() -> (40)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_5__l0p_l1u_buffer_10", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l0p_l1u_buffer_10"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l0p_l1u_buffer_9", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l0p_l1u_buffer_9"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l0p_l1u_buffer_8", type = memref<120xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l0p_l1u_buffer_8"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l0p_l1u_buffer_7", type = memref<120xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l0p_l1u_buffer_7"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l0p_l1u_buffer_6", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_5__l0p_l1u_buffer_6"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l0p_l1u_buffer_5", type = memref<40xi32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [2], scaleisa.scratchpad, var_name = "au_generated_candidate_5__l0p_l1u_buffer_5"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_5__l0p_l1u_buffer", type = memref<40xi32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [2], scaleisa.scratchpad, var_name = "au_generated_candidate_5__l0p_l1u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_38__l0u_l1u_buffer_4", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_38__l0u_l1u_buffer_4"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_38__l0u_l1u_buffer", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_38__l0u_l1u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_36__l0u_l1u_buffer_3", type = memref<120xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_36__l0u_l1u_buffer_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_36__l0u_l1u_buffer", type = memref<120xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_36__l0u_l1u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_31__l1p_buffer_2", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_31__l1p_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_31__l1p_buffer_1", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_31__l1p_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_31__l1p_buffer", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_31__l1p_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_21__direct_buffer_0", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_21__direct_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_21__direct_buffer", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_21__direct_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_21__direct"}> ({
  ^bb0(%arg20: i5, %arg21: i5, %arg22: i5):
    %305 = "aps.read_irf"(%arg20) : (i5) -> i32
    %306 = "aps.read_irf"(%arg21) : (i5) -> i32
    %307 = "memref.get_global"() <{name = @au_generated_candidate_21__direct_buffer}> : () -> memref<72xf32>
    %308 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %309 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%305, %307, %309, %308) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %310 = "memref.get_global"() <{name = @au_generated_candidate_21__direct_buffer_0}> : () -> memref<1xi32>
    %311 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %312 = "arith.addi"(%306, %311) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %313 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %314 = "aps.load"(%312) : (i32) -> i32
    "aps.write_smem"(%314, %310, %313) : (i32, memref<1xi32>, i32) -> ()
    %315 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %316 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %317 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %318 = "arith.constant"() <{value = 2.570000e+02 : f32}> : () -> f32
    %319 = "arith.constant"() <{value = 1.730000e+02 : f32}> : () -> f32
    %320 = "arith.constant"() <{value = 1.010000e+02 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg23: index):
      %325 = "arith.index_cast"(%arg23) : (index) -> i32
      %326 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %327 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %328 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %329 = "arith.muli"(%325, %328) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %330 = "arith.addi"(%327, %329) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %331 = "arith.addi"(%330, %317) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %332 = "arith.addi"(%326, %331) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %333 = "aps.read_smem"(%307, %332) : (memref<72xf32>, i32) -> f32
      %334 = "arith.mulf"(%333, %320) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %335 = "arith.fptosi"(%334) : (f32) -> i32
      %336 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %337 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %338 = "arith.addi"(%336, %337) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %339 = "aps.read_smem"(%310, %338) : (memref<1xi32>, i32) -> i32
      %340 = "arith.addi"(%339, %335) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %341 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %342 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %343 = "arith.addi"(%341, %342) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%340, %310, %343) : (i32, memref<1xi32>, i32) -> ()
      %344 = "arith.index_cast"(%arg23) : (index) -> i32
      %345 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %346 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %347 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %348 = "arith.muli"(%344, %347) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %349 = "arith.addi"(%346, %348) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %350 = "arith.addi"(%349, %316) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %351 = "arith.addi"(%345, %350) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %352 = "aps.read_smem"(%307, %351) : (memref<72xf32>, i32) -> f32
      %353 = "arith.mulf"(%352, %319) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %354 = "arith.fptosi"(%353) : (f32) -> i32
      %355 = "arith.addi"(%340, %354) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %356 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %357 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %358 = "arith.addi"(%356, %357) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%355, %310, %358) : (i32, memref<1xi32>, i32) -> ()
      %359 = "arith.index_cast"(%arg23) : (index) -> i32
      %360 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %361 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %362 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %363 = "arith.muli"(%359, %362) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %364 = "arith.addi"(%361, %363) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %365 = "arith.addi"(%364, %315) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %366 = "arith.addi"(%360, %365) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %367 = "aps.read_smem"(%307, %366) : (memref<72xf32>, i32) -> f32
      %368 = "arith.mulf"(%367, %318) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %369 = "arith.fptosi"(%368) : (f32) -> i32
      %370 = "arith.andi"(%355, %369) : (i32, i32) -> i32
      %371 = "arith.ori"(%355, %369) : (i32, i32) -> i32
      %372 = "arith.addi"(%370, %371) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %373 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %374 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %375 = "arith.addi"(%373, %374) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%372, %310, %375) : (i32, memref<1xi32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %321 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %322 = "arith.addi"(%306, %321) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %323 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %324 = "aps.read_smem"(%310, %323) : (memref<1xi32>, i32) -> i32
    "aps.store"(%324, %322) : (i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<24x3xf32>, memref<i32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_31__l1p"}> ({
  ^bb0(%arg15: i5, %arg16: i5, %arg17: i5):
    %243 = "aps.read_irf"(%arg15) : (i5) -> i32
    %244 = "aps.read_irf"(%arg16) : (i5) -> i32
    %245 = "memref.get_global"() <{name = @au_generated_candidate_31__l1p_buffer}> : () -> memref<72xf32>
    %246 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %247 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%243, %245, %247, %246) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %248 = "memref.get_global"() <{name = @au_generated_candidate_31__l1p_buffer_1}> : () -> memref<72xf32>
    %249 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %250 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%244, %248, %250, %249) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %251 = "aps.read_csr"() <{global_name = @au_generated_candidate_31__l1p_csr0}> : () -> i32
    %252 = "memref.get_global"() <{name = @au_generated_candidate_31__l1p_buffer_2}> : () -> memref<72xf32>
    %253 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %254 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%251, %252, %254, %253) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %255 = "arith.constant"() <{value = 1 : index}> : () -> index
    %256 = "arith.constant"() <{value = 2.800000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg18: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg19: index):
        %259 = "arith.addi"(%arg19, %255) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %260 = "arith.index_cast"(%259) : (index) -> i32
        %261 = "arith.index_cast"(%arg18) : (index) -> i32
        %262 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %263 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %264 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %265 = "arith.muli"(%260, %264) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %266 = "arith.addi"(%263, %265) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %267 = "arith.addi"(%266, %261) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %268 = "arith.addi"(%262, %267) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %269 = "aps.read_smem"(%245, %268) : (memref<72xf32>, i32) -> f32
        %270 = "arith.mulf"(%269, %256) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %271 = "arith.addi"(%arg19, %255) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %272 = "arith.index_cast"(%271) : (index) -> i32
        %273 = "arith.index_cast"(%arg18) : (index) -> i32
        %274 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %275 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %276 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %277 = "arith.muli"(%272, %276) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %278 = "arith.addi"(%275, %277) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %279 = "arith.addi"(%278, %273) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %280 = "arith.addi"(%274, %279) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %281 = "aps.read_smem"(%248, %280) : (memref<72xf32>, i32) -> f32
        %282 = "arith.divf"(%270, %281) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %283 = "arith.addi"(%arg19, %255) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %284 = "arith.index_cast"(%283) : (index) -> i32
        %285 = "arith.index_cast"(%arg18) : (index) -> i32
        %286 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %287 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %288 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %289 = "arith.muli"(%284, %288) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %290 = "arith.addi"(%287, %289) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %291 = "arith.addi"(%290, %285) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %292 = "arith.addi"(%286, %291) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %293 = "aps.read_smem"(%252, %292) : (memref<72xf32>, i32) -> f32
        %294 = "arith.subf"(%293, %282) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %295 = "arith.addi"(%arg19, %255) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %296 = "arith.index_cast"(%295) : (index) -> i32
        %297 = "arith.index_cast"(%arg18) : (index) -> i32
        %298 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %299 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %300 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %301 = "arith.muli"(%296, %300) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %302 = "arith.addi"(%299, %301) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %303 = "arith.addi"(%302, %297) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %304 = "arith.addi"(%298, %303) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%294, %252, %304) : (f32, memref<72xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {pipeline = 1 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %257 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %258 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%251, %252, %258, %257) <{direction = 1 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_36__l0u_l1u"}> ({
  ^bb0(%arg10: i5, %arg11: i5, %arg12: i5):
    %216 = "aps.read_irf"(%arg10) : (i5) -> i32
    %217 = "aps.read_irf"(%arg11) : (i5) -> i32
    %218 = "memref.get_global"() <{name = @au_generated_candidate_36__l0u_l1u_buffer}> : () -> memref<120xf32>
    %219 = "arith.constant"() <{value = 120 : i32}> : () -> i32
    %220 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%216, %218, %220, %219) <{direction = 0 : i32}> : (i32, memref<120xf32>, i32, i32) -> ()
    %221 = "memref.get_global"() <{name = @au_generated_candidate_36__l0u_l1u_buffer_3}> : () -> memref<120xf32>
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg13: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg14: index):
        %224 = "arith.index_cast"(%arg13) : (index) -> i32
        %225 = "arith.index_cast"(%arg14) : (index) -> i32
        %226 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %227 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %228 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %229 = "arith.muli"(%224, %228) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %230 = "arith.addi"(%227, %229) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %231 = "arith.addi"(%230, %225) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %232 = "arith.addi"(%226, %231) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %233 = "aps.read_smem"(%218, %232) : (memref<120xf32>, i32) -> f32
        %234 = "arith.index_cast"(%arg13) : (index) -> i32
        %235 = "arith.index_cast"(%arg14) : (index) -> i32
        %236 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %237 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %238 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %239 = "arith.muli"(%234, %238) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %240 = "arith.addi"(%237, %239) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %241 = "arith.addi"(%240, %235) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %242 = "arith.addi"(%236, %241) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%233, %221, %242) : (f32, memref<120xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %222 = "arith.constant"() <{value = 120 : i32}> : () -> i32
    %223 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%217, %221, %223, %222) <{direction = 1 : i32}> : (i32, memref<120xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 2 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<40x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<40x3xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<40x3xf32>, memref<40x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_38__l0u_l1u"}> ({
  ^bb0(%arg5: i5, %arg6: i5, %arg7: i5):
    %189 = "aps.read_irf"(%arg5) : (i5) -> i32
    %190 = "aps.read_irf"(%arg6) : (i5) -> i32
    %191 = "memref.get_global"() <{name = @au_generated_candidate_38__l0u_l1u_buffer}> : () -> memref<72xf32>
    %192 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %193 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%189, %191, %193, %192) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %194 = "memref.get_global"() <{name = @au_generated_candidate_38__l0u_l1u_buffer_4}> : () -> memref<72xf32>
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg8: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg9: index):
        %197 = "arith.index_cast"(%arg8) : (index) -> i32
        %198 = "arith.index_cast"(%arg9) : (index) -> i32
        %199 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %200 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %201 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %202 = "arith.muli"(%197, %201) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %203 = "arith.addi"(%200, %202) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %204 = "arith.addi"(%203, %198) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %205 = "arith.addi"(%199, %204) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %206 = "aps.read_smem"(%191, %205) : (memref<72xf32>, i32) -> f32
        %207 = "arith.index_cast"(%arg8) : (index) -> i32
        %208 = "arith.index_cast"(%arg9) : (index) -> i32
        %209 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %210 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %211 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %212 = "arith.muli"(%207, %211) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %213 = "arith.addi"(%210, %212) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %214 = "arith.addi"(%213, %208) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %215 = "arith.addi"(%209, %214) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%206, %194, %215) : (f32, memref<72xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %195 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %196 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%190, %194, %196, %195) <{direction = 1 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 3 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<24x3xf32>, memref<24x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_5__l0p_l1u"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_5__l0p_l1u_buffer}> : () -> memref<40xi32>
    %3 = "arith.constant"() <{value = 40 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %2, %4, %3) <{direction = 0 : i32}> : (i32, memref<40xi32>, i32, i32) -> ()
    %5 = "memref.get_global"() <{name = @au_generated_candidate_5__l0p_l1u_buffer_5}> : () -> memref<40xi32>
    %6 = "arith.constant"() <{value = 40 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%1, %5, %7, %6) <{direction = 0 : i32}> : (i32, memref<40xi32>, i32, i32) -> ()
    %8 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l0p_l1u_csr0}> : () -> i32
    %9 = "memref.get_global"() <{name = @au_generated_candidate_5__l0p_l1u_buffer_6}> : () -> memref<72xf32>
    %10 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%8, %9, %11, %10) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %12 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l0p_l1u_csr1}> : () -> i32
    %13 = "memref.get_global"() <{name = @au_generated_candidate_5__l0p_l1u_buffer_7}> : () -> memref<120xf32>
    %14 = "arith.constant"() <{value = 120 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%12, %13, %15, %14) <{direction = 0 : i32}> : (i32, memref<120xf32>, i32, i32) -> ()
    %16 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l0p_l1u_csr2}> : () -> i32
    %17 = "memref.get_global"() <{name = @au_generated_candidate_5__l0p_l1u_buffer_8}> : () -> memref<120xf32>
    %18 = "arith.constant"() <{value = 120 : i32}> : () -> i32
    %19 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%16, %17, %19, %18) <{direction = 0 : i32}> : (i32, memref<120xf32>, i32, i32) -> ()
    %20 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l0p_l1u_csr3}> : () -> i32
    %21 = "memref.get_global"() <{name = @au_generated_candidate_5__l0p_l1u_buffer_9}> : () -> memref<72xf32>
    %22 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %23 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%20, %21, %23, %22) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %24 = "aps.read_csr"() <{global_name = @au_generated_candidate_5__l0p_l1u_csr4}> : () -> i32
    %25 = "memref.get_global"() <{name = @au_generated_candidate_5__l0p_l1u_buffer_10}> : () -> memref<72xf32>
    %26 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %27 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%24, %25, %27, %26) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg3: index):
      %32 = "arith.index_cast"(%arg3) : (index) -> i32
      %33 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %34 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %35 = "arith.addi"(%34, %32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %36 = "arith.addi"(%33, %35) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %37 = "aps.read_smem"(%2, %36) : (memref<40xi32>, i32) -> i32
      %38 = "arith.index_cast"(%arg3) : (index) -> i32
      %39 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %40 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %41 = "arith.addi"(%40, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %42 = "arith.addi"(%39, %41) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %43 = "aps.read_smem"(%5, %42) : (memref<40xi32>, i32) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg4: index):
        %44 = "arith.index_cast"(%43) : (i32) -> index
        %45 = "arith.index_cast"(%44) : (index) -> i32
        %46 = "arith.index_cast"(%arg4) : (index) -> i32
        %47 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %48 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %49 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %50 = "arith.muli"(%45, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %51 = "arith.addi"(%48, %50) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %52 = "arith.addi"(%51, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %53 = "arith.addi"(%47, %52) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %54 = "aps.read_smem"(%9, %53) : (memref<72xf32>, i32) -> f32
        %55 = "arith.index_cast"(%37) : (i32) -> index
        %56 = "arith.index_cast"(%55) : (index) -> i32
        %57 = "arith.index_cast"(%arg4) : (index) -> i32
        %58 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %59 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %60 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %61 = "arith.muli"(%56, %60) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %62 = "arith.addi"(%59, %61) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %63 = "arith.addi"(%62, %57) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %64 = "arith.addi"(%58, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %65 = "aps.read_smem"(%9, %64) : (memref<72xf32>, i32) -> f32
        %66 = "arith.subf"(%54, %65) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %67 = "arith.index_cast"(%arg3) : (index) -> i32
        %68 = "arith.index_cast"(%arg4) : (index) -> i32
        %69 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %70 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %71 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %72 = "arith.muli"(%67, %71) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %73 = "arith.addi"(%70, %72) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %74 = "arith.addi"(%73, %68) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %75 = "arith.addi"(%69, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %76 = "aps.read_smem"(%13, %75) : (memref<120xf32>, i32) -> f32
        %77 = "arith.subf"(%66, %76) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %78 = "arith.index_cast"(%arg3) : (index) -> i32
        %79 = "arith.index_cast"(%arg4) : (index) -> i32
        %80 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %81 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %82 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %83 = "arith.muli"(%78, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %84 = "arith.addi"(%81, %83) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %85 = "arith.addi"(%84, %79) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %86 = "arith.addi"(%80, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %87 = "aps.read_smem"(%17, %86) : (memref<120xf32>, i32) -> f32
        %88 = "arith.mulf"(%87, %77) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %89 = "arith.index_cast"(%55) : (index) -> i32
        %90 = "arith.index_cast"(%arg4) : (index) -> i32
        %91 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %92 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %93 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %94 = "arith.muli"(%89, %93) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %95 = "arith.addi"(%92, %94) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %96 = "arith.addi"(%95, %90) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %97 = "arith.addi"(%91, %96) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %98 = "aps.read_smem"(%21, %97) : (memref<72xf32>, i32) -> f32
        %99 = "arith.addf"(%98, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %100 = "arith.index_cast"(%55) : (index) -> i32
        %101 = "arith.index_cast"(%arg4) : (index) -> i32
        %102 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %103 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %104 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %105 = "arith.muli"(%100, %104) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %106 = "arith.addi"(%103, %105) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %107 = "arith.addi"(%106, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %108 = "arith.addi"(%102, %107) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%99, %21, %108) : (f32, memref<72xf32>, i32) -> ()
        %109 = "arith.index_cast"(%44) : (index) -> i32
        %110 = "arith.index_cast"(%arg4) : (index) -> i32
        %111 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %112 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %113 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %114 = "arith.muli"(%109, %113) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %115 = "arith.addi"(%112, %114) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %116 = "arith.addi"(%115, %110) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %117 = "arith.addi"(%111, %116) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %118 = "aps.read_smem"(%21, %117) : (memref<72xf32>, i32) -> f32
        %119 = "arith.subf"(%118, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %120 = "arith.index_cast"(%44) : (index) -> i32
        %121 = "arith.index_cast"(%arg4) : (index) -> i32
        %122 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %123 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %124 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %125 = "arith.muli"(%120, %124) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %126 = "arith.addi"(%123, %125) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %127 = "arith.addi"(%126, %121) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %128 = "arith.addi"(%122, %127) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%119, %21, %128) : (f32, memref<72xf32>, i32) -> ()
        %129 = "arith.index_cast"(%arg3) : (index) -> i32
        %130 = "arith.index_cast"(%arg4) : (index) -> i32
        %131 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %132 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %133 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %134 = "arith.muli"(%129, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %135 = "arith.addi"(%132, %134) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %136 = "arith.addi"(%135, %130) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %137 = "arith.addi"(%131, %136) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %138 = "aps.read_smem"(%17, %137) : (memref<120xf32>, i32) -> f32
        %139 = "arith.index_cast"(%55) : (index) -> i32
        %140 = "arith.index_cast"(%arg4) : (index) -> i32
        %141 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %142 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %143 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %144 = "arith.muli"(%139, %143) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %145 = "arith.addi"(%142, %144) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %146 = "arith.addi"(%145, %140) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %147 = "arith.addi"(%141, %146) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %148 = "aps.read_smem"(%25, %147) : (memref<72xf32>, i32) -> f32
        %149 = "arith.addf"(%148, %138) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %150 = "arith.index_cast"(%55) : (index) -> i32
        %151 = "arith.index_cast"(%arg4) : (index) -> i32
        %152 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %153 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %154 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %155 = "arith.muli"(%150, %154) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %156 = "arith.addi"(%153, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %157 = "arith.addi"(%156, %151) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %158 = "arith.addi"(%152, %157) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%149, %25, %158) : (f32, memref<72xf32>, i32) -> ()
        %159 = "arith.index_cast"(%arg3) : (index) -> i32
        %160 = "arith.index_cast"(%arg4) : (index) -> i32
        %161 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %162 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %163 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %164 = "arith.muli"(%159, %163) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %165 = "arith.addi"(%162, %164) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %166 = "arith.addi"(%165, %160) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %167 = "arith.addi"(%161, %166) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %168 = "aps.read_smem"(%17, %167) : (memref<120xf32>, i32) -> f32
        %169 = "arith.index_cast"(%44) : (index) -> i32
        %170 = "arith.index_cast"(%arg4) : (index) -> i32
        %171 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %172 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %173 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %174 = "arith.muli"(%169, %173) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %175 = "arith.addi"(%172, %174) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %176 = "arith.addi"(%175, %170) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %177 = "arith.addi"(%171, %176) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %178 = "aps.read_smem"(%25, %177) : (memref<72xf32>, i32) -> f32
        %179 = "arith.addf"(%178, %168) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %180 = "arith.index_cast"(%44) : (index) -> i32
        %181 = "arith.index_cast"(%arg4) : (index) -> i32
        %182 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %183 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %184 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %185 = "arith.muli"(%180, %184) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %186 = "arith.addi"(%183, %185) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %187 = "arith.addi"(%186, %181) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %188 = "arith.addi"(%182, %187) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%179, %25, %188) : (f32, memref<72xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {pipeline = 1 : i64} : () -> ()
    %28 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %29 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%20, %21, %29, %28) <{direction = 1 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %30 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %31 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%24, %25, %31, %30) <{direction = 1 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 4 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<40xi32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<40xi32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2050 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2051 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<40x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2052 : i32, arg_index = 4 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<40x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2053 : i32, arg_index = 5 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2054 : i32, arg_index = 6 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<40xi32>, memref<40xi32>, memref<24x3xf32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_31__l1p_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l0p_l1u_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l0p_l1u_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2051 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l0p_l1u_csr2", sym_visibility = "private", type = memref<i32>}> {csr_address = 2052 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l0p_l1u_csr3", sym_visibility = "private", type = memref<i32>}> {csr_address = 2053 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_5__l0p_l1u_csr4", sym_visibility = "private", type = memref<i32>}> {csr_address = 2054 : i32} : () -> ()
}) : () -> ()

