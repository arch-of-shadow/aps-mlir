#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (15)>
#map2 = affine_map<() -> (16)>
#map3 = affine_map<() -> (20)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_56__l0u_buffer", type = memref<40xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [2], scaleisa.scratchpad, var_name = "au_generated_candidate_56__l0u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_51__direct_buffer_5", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_51__direct_buffer_5"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_51__direct_buffer", type = memref<40xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_51__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_41__direct_buffer_4", type = memref<40xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_41__direct_buffer_4"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_41__direct_buffer_3", type = memref<40xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_41__direct_buffer_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_41__direct_buffer", type = memref<40xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_41__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_34__l0u_buffer", type = memref<48xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_34__l0u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_21__direct_buffer_2", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_21__direct_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_21__direct_buffer", type = memref<48xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_21__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_12__direct_buffer_1", type = memref<48xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_12__direct_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_12__direct_buffer_0", type = memref<48xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_12__direct_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_12__direct_buffer", type = memref<48xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_12__direct_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_12__direct"}> ({
  ^bb0(%arg20: i5, %arg21: i5, %arg22: i5):
    %306 = "aps.read_irf"(%arg20) : (i5) -> i32
    %307 = "aps.read_irf"(%arg21) : (i5) -> i32
    %308 = "memref.get_global"() <{name = @au_generated_candidate_12__direct_buffer}> : () -> memref<48xf32>
    %309 = "arith.constant"() <{value = 48 : i32}> : () -> i32
    %310 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%306, %308, %310, %309) <{direction = 0 : i32}> : (i32, memref<48xf32>, i32, i32) -> ()
    %311 = "memref.get_global"() <{name = @au_generated_candidate_12__direct_buffer_0}> : () -> memref<48xf32>
    %312 = "arith.constant"() <{value = 48 : i32}> : () -> i32
    %313 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%307, %311, %313, %312) <{direction = 0 : i32}> : (i32, memref<48xf32>, i32, i32) -> ()
    %314 = "aps.read_csr"() <{global_name = @au_generated_candidate_12__direct_csr0}> : () -> i32
    %315 = "memref.get_global"() <{name = @au_generated_candidate_12__direct_buffer_1}> : () -> memref<48xf32>
    %316 = "arith.constant"() <{value = 48 : i32}> : () -> i32
    %317 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%314, %315, %317, %316) <{direction = 0 : i32}> : (i32, memref<48xf32>, i32, i32) -> ()
    %318 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %319 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %320 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %321 = "arith.constant"() <{value = 1 : index}> : () -> index
    %322 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %323 = "arith.constant"() <{value = 2.200000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg23: index):
      %326 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %327 = "arith.index_cast"(%326) : (index) -> i32
      %328 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %329 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %330 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %331 = "arith.muli"(%327, %330) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %332 = "arith.addi"(%329, %331) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %333 = "arith.addi"(%332, %320) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %334 = "arith.addi"(%328, %333) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %335 = "aps.read_smem"(%308, %334) : (memref<48xf32>, i32) -> f32
      %336 = "arith.mulf"(%335, %323) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %337 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %338 = "arith.index_cast"(%337) : (index) -> i32
      %339 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %340 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %341 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %342 = "arith.muli"(%338, %341) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %343 = "arith.addi"(%340, %342) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %344 = "arith.addi"(%343, %320) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %345 = "arith.addi"(%339, %344) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %346 = "aps.read_smem"(%311, %345) : (memref<48xf32>, i32) -> f32
      %347 = "arith.divf"(%336, %346) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %348 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %349 = "arith.index_cast"(%348) : (index) -> i32
      %350 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %351 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %352 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %353 = "arith.muli"(%349, %352) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %354 = "arith.addi"(%351, %353) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %355 = "arith.addi"(%354, %320) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %356 = "arith.addi"(%350, %355) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %357 = "aps.read_smem"(%315, %356) : (memref<48xf32>, i32) -> f32
      %358 = "arith.subf"(%357, %347) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %359 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %360 = "arith.index_cast"(%359) : (index) -> i32
      %361 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %362 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %363 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %364 = "arith.muli"(%360, %363) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %365 = "arith.addi"(%362, %364) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %366 = "arith.addi"(%365, %320) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %367 = "arith.addi"(%361, %366) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%358, %315, %367) : (f32, memref<48xf32>, i32) -> ()
      %368 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %369 = "arith.index_cast"(%368) : (index) -> i32
      %370 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %371 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %372 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %373 = "arith.muli"(%369, %372) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %374 = "arith.addi"(%371, %373) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %375 = "arith.addi"(%374, %319) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %376 = "arith.addi"(%370, %375) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %377 = "aps.read_smem"(%308, %376) : (memref<48xf32>, i32) -> f32
      %378 = "arith.mulf"(%377, %323) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %379 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %380 = "arith.index_cast"(%379) : (index) -> i32
      %381 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %382 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %383 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %384 = "arith.muli"(%380, %383) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %385 = "arith.addi"(%382, %384) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %386 = "arith.addi"(%385, %319) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %387 = "arith.addi"(%381, %386) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %388 = "aps.read_smem"(%311, %387) : (memref<48xf32>, i32) -> f32
      %389 = "arith.divf"(%378, %388) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %390 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %391 = "arith.index_cast"(%390) : (index) -> i32
      %392 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %393 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %394 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %395 = "arith.muli"(%391, %394) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %396 = "arith.addi"(%393, %395) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %397 = "arith.addi"(%396, %319) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %398 = "arith.addi"(%392, %397) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %399 = "aps.read_smem"(%315, %398) : (memref<48xf32>, i32) -> f32
      %400 = "arith.subf"(%399, %389) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %401 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %402 = "arith.index_cast"(%401) : (index) -> i32
      %403 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %404 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %405 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %406 = "arith.muli"(%402, %405) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %407 = "arith.addi"(%404, %406) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %408 = "arith.addi"(%407, %319) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %409 = "arith.addi"(%403, %408) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%400, %315, %409) : (f32, memref<48xf32>, i32) -> ()
      %410 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %411 = "arith.index_cast"(%410) : (index) -> i32
      %412 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %413 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %414 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %415 = "arith.muli"(%411, %414) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %416 = "arith.addi"(%413, %415) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %417 = "arith.addi"(%416, %318) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %418 = "arith.addi"(%412, %417) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %419 = "aps.read_smem"(%308, %418) : (memref<48xf32>, i32) -> f32
      %420 = "arith.mulf"(%419, %322) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %421 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %422 = "arith.index_cast"(%421) : (index) -> i32
      %423 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %424 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %425 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %426 = "arith.muli"(%422, %425) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %427 = "arith.addi"(%424, %426) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %428 = "arith.addi"(%427, %318) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %429 = "arith.addi"(%423, %428) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %430 = "aps.read_smem"(%311, %429) : (memref<48xf32>, i32) -> f32
      %431 = "arith.divf"(%420, %430) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %432 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %433 = "arith.index_cast"(%432) : (index) -> i32
      %434 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %435 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %436 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %437 = "arith.muli"(%433, %436) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %438 = "arith.addi"(%435, %437) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %439 = "arith.addi"(%438, %318) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %440 = "arith.addi"(%434, %439) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %441 = "aps.read_smem"(%315, %440) : (memref<48xf32>, i32) -> f32
      %442 = "arith.subf"(%441, %431) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %443 = "arith.addi"(%arg23, %321) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %444 = "arith.index_cast"(%443) : (index) -> i32
      %445 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %446 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %447 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %448 = "arith.muli"(%444, %447) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %449 = "arith.addi"(%446, %448) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %450 = "arith.addi"(%449, %318) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %451 = "arith.addi"(%445, %450) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%442, %315, %451) : (f32, memref<48xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %324 = "arith.constant"() <{value = 48 : i32}> : () -> i32
    %325 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%314, %315, %325, %324) <{direction = 1 : i32}> : (i32, memref<48xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<16x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<16x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<16x3xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<16x3xf32>, memref<16x3xf32>, memref<16x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_21__direct"}> ({
  ^bb0(%arg16: i5, %arg17: i5, %arg18: i5):
    %237 = "aps.read_irf"(%arg16) : (i5) -> i32
    %238 = "aps.read_irf"(%arg17) : (i5) -> i32
    %239 = "memref.get_global"() <{name = @au_generated_candidate_21__direct_buffer}> : () -> memref<48xf32>
    %240 = "arith.constant"() <{value = 48 : i32}> : () -> i32
    %241 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%237, %239, %241, %240) <{direction = 0 : i32}> : (i32, memref<48xf32>, i32, i32) -> ()
    %242 = "memref.get_global"() <{name = @au_generated_candidate_21__direct_buffer_2}> : () -> memref<1xi32>
    %243 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %244 = "arith.addi"(%238, %243) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %245 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %246 = "aps.load"(%244) : (i32) -> i32
    "aps.write_smem"(%246, %242, %245) : (i32, memref<1xi32>, i32) -> ()
    %247 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %248 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %249 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %250 = "arith.constant"() <{value = 1.970000e+02 : f32}> : () -> f32
    %251 = "arith.constant"() <{value = 1.310000e+02 : f32}> : () -> f32
    %252 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg19: index):
      %257 = "arith.index_cast"(%arg19) : (index) -> i32
      %258 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %259 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %260 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %261 = "arith.muli"(%257, %260) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %262 = "arith.addi"(%259, %261) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %263 = "arith.addi"(%262, %249) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %264 = "arith.addi"(%258, %263) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %265 = "aps.read_smem"(%239, %264) : (memref<48xf32>, i32) -> f32
      %266 = "arith.mulf"(%265, %252) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %267 = "arith.fptosi"(%266) : (f32) -> i32
      %268 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %269 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %270 = "arith.addi"(%268, %269) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %271 = "aps.read_smem"(%242, %270) : (memref<1xi32>, i32) -> i32
      %272 = "arith.addi"(%271, %267) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %273 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %274 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %275 = "arith.addi"(%273, %274) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%272, %242, %275) : (i32, memref<1xi32>, i32) -> ()
      %276 = "arith.index_cast"(%arg19) : (index) -> i32
      %277 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %278 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %279 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %280 = "arith.muli"(%276, %279) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %281 = "arith.addi"(%278, %280) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %282 = "arith.addi"(%281, %248) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %283 = "arith.addi"(%277, %282) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %284 = "aps.read_smem"(%239, %283) : (memref<48xf32>, i32) -> f32
      %285 = "arith.mulf"(%284, %251) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %286 = "arith.fptosi"(%285) : (f32) -> i32
      %287 = "arith.addi"(%272, %286) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %288 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %289 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %290 = "arith.addi"(%288, %289) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%287, %242, %290) : (i32, memref<1xi32>, i32) -> ()
      %291 = "arith.index_cast"(%arg19) : (index) -> i32
      %292 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %293 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %294 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %295 = "arith.muli"(%291, %294) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %296 = "arith.addi"(%293, %295) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %297 = "arith.addi"(%296, %247) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %298 = "arith.addi"(%292, %297) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %299 = "aps.read_smem"(%239, %298) : (memref<48xf32>, i32) -> f32
      %300 = "arith.mulf"(%299, %250) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %301 = "arith.fptosi"(%300) : (f32) -> i32
      %302 = "arith.addi"(%287, %301) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %303 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %304 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %305 = "arith.addi"(%303, %304) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%302, %242, %305) : (i32, memref<1xi32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %253 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %254 = "arith.addi"(%238, %253) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %255 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %256 = "aps.read_smem"(%242, %255) : (memref<1xi32>, i32) -> i32
    "aps.store"(%256, %254) : (i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<16x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<16x3xf32>, memref<i32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_34__l0u"}> ({
  ^bb0(%arg12: i5, %arg13: i5, %arg14: i5):
    %183 = "aps.read_irf"(%arg12) : (i5) -> i32
    %184 = "memref.get_global"() <{name = @au_generated_candidate_34__l0u_buffer}> : () -> memref<48xf32>
    %185 = "arith.constant"() <{value = 48 : i32}> : () -> i32
    %186 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%183, %184, %186, %185) <{direction = 0 : i32}> : (i32, memref<48xf32>, i32, i32) -> ()
    %187 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %188 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %189 = "arith.constant"() <{value = 1.500000e-02 : f32}> : () -> f32
    %190 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %191 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %192 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %193 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %194 = "arith.constant"() <{value = 5.000000e-02 : f32}> : () -> f32
    %195 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %196 = "arith.constant"() <{value = 7.500000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg15: index):
      %199 = "arith.index_cast"(%arg15) : (index) -> i32
      %200 = "arith.sitofp"(%199) : (i32) -> f32
      %201 = "arith.mulf"(%200, %196) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %202 = "arith.remsi"(%199, %195) : (i32, i32) -> i32
      %203 = "arith.sitofp"(%202) : (i32) -> f32
      %204 = "arith.mulf"(%203, %194) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %205 = "arith.addf"(%201, %204) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %206 = "arith.index_cast"(%arg15) : (index) -> i32
      %207 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %208 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %209 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %210 = "arith.muli"(%206, %209) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %211 = "arith.addi"(%208, %210) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %212 = "arith.addi"(%211, %188) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %213 = "arith.addi"(%207, %212) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%205, %184, %213) : (f32, memref<48xf32>, i32) -> ()
      %214 = "arith.remsi"(%199, %193) : (i32, i32) -> i32
      %215 = "arith.sitofp"(%214) : (i32) -> f32
      %216 = "arith.mulf"(%215, %192) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %217 = "arith.index_cast"(%arg15) : (index) -> i32
      %218 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %219 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %220 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %221 = "arith.muli"(%217, %220) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %222 = "arith.addi"(%219, %221) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %223 = "arith.addi"(%222, %187) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %224 = "arith.addi"(%218, %223) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%216, %184, %224) : (f32, memref<48xf32>, i32) -> ()
      %225 = "arith.remsi"(%199, %191) : (i32, i32) -> i32
      %226 = "arith.addi"(%225, %190) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %227 = "arith.sitofp"(%226) : (i32) -> f32
      %228 = "arith.mulf"(%227, %189) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %229 = "arith.index_cast"(%arg15) : (index) -> i32
      %230 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %231 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %232 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %233 = "arith.muli"(%229, %232) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %234 = "arith.addi"(%231, %233) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %235 = "arith.addi"(%234, %193) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %236 = "arith.addi"(%230, %235) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%228, %184, %236) : (f32, memref<48xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %197 = "arith.constant"() <{value = 48 : i32}> : () -> i32
    %198 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%183, %184, %198, %197) <{direction = 1 : i32}> : (i32, memref<48xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 2 : i32, funct7 = 2 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<16x3xf32>, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<16x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_41__direct"}> ({
  ^bb0(%arg8: i5, %arg9: i5, %arg10: i5):
    %90 = "aps.read_irf"(%arg8) : (i5) -> i32
    %91 = "aps.read_irf"(%arg9) : (i5) -> i32
    %92 = "memref.get_global"() <{name = @au_generated_candidate_41__direct_buffer}> : () -> memref<40xf32>
    %93 = "arith.constant"() <{value = 40 : i32}> : () -> i32
    %94 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%90, %92, %94, %93) <{direction = 0 : i32}> : (i32, memref<40xf32>, i32, i32) -> ()
    %95 = "memref.get_global"() <{name = @au_generated_candidate_41__direct_buffer_3}> : () -> memref<40xf32>
    %96 = "arith.constant"() <{value = 40 : i32}> : () -> i32
    %97 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%91, %95, %97, %96) <{direction = 0 : i32}> : (i32, memref<40xf32>, i32, i32) -> ()
    %98 = "aps.read_csr"() <{global_name = @au_generated_candidate_41__direct_csr0}> : () -> i32
    %99 = "memref.get_global"() <{name = @au_generated_candidate_41__direct_buffer_4}> : () -> memref<40xf32>
    %100 = "arith.constant"() <{value = 40 : i32}> : () -> i32
    %101 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%98, %99, %101, %100) <{direction = 0 : i32}> : (i32, memref<40xf32>, i32, i32) -> ()
    %102 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %103 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %104 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg11: index):
      %107 = "arith.index_cast"(%arg11) : (index) -> i32
      %108 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %109 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %110 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %111 = "arith.muli"(%107, %110) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %112 = "arith.addi"(%109, %111) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %113 = "arith.addi"(%112, %103) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %114 = "arith.addi"(%108, %113) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %115 = "aps.read_smem"(%92, %114) : (memref<40xf32>, i32) -> f32
      %116 = "arith.mulf"(%115, %104) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %117 = "arith.index_cast"(%arg11) : (index) -> i32
      %118 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %119 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %120 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %121 = "arith.muli"(%117, %120) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %122 = "arith.addi"(%119, %121) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %123 = "arith.addi"(%122, %103) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %124 = "arith.addi"(%118, %123) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %125 = "aps.read_smem"(%95, %124) : (memref<40xf32>, i32) -> f32
      %126 = "arith.divf"(%116, %125) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %127 = "arith.index_cast"(%arg11) : (index) -> i32
      %128 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %129 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %130 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %131 = "arith.muli"(%127, %130) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %132 = "arith.addi"(%129, %131) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %133 = "arith.addi"(%132, %103) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %134 = "arith.addi"(%128, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %135 = "aps.read_smem"(%99, %134) : (memref<40xf32>, i32) -> f32
      %136 = "arith.subf"(%135, %126) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %137 = "arith.index_cast"(%arg11) : (index) -> i32
      %138 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %139 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %140 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %141 = "arith.muli"(%137, %140) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %142 = "arith.addi"(%139, %141) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %143 = "arith.addi"(%142, %103) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %144 = "arith.addi"(%138, %143) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%136, %99, %144) : (f32, memref<40xf32>, i32) -> ()
      %145 = "arith.index_cast"(%arg11) : (index) -> i32
      %146 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %147 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %148 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %149 = "arith.muli"(%145, %148) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %150 = "arith.addi"(%147, %149) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %151 = "arith.addi"(%150, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %152 = "arith.addi"(%146, %151) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %153 = "aps.read_smem"(%92, %152) : (memref<40xf32>, i32) -> f32
      %154 = "arith.mulf"(%153, %104) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %155 = "arith.index_cast"(%arg11) : (index) -> i32
      %156 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %157 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %158 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %159 = "arith.muli"(%155, %158) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %160 = "arith.addi"(%157, %159) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %161 = "arith.addi"(%160, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %162 = "arith.addi"(%156, %161) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %163 = "aps.read_smem"(%95, %162) : (memref<40xf32>, i32) -> f32
      %164 = "arith.divf"(%154, %163) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %165 = "arith.index_cast"(%arg11) : (index) -> i32
      %166 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %167 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %168 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %169 = "arith.muli"(%165, %168) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %170 = "arith.addi"(%167, %169) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %171 = "arith.addi"(%170, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %172 = "arith.addi"(%166, %171) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %173 = "aps.read_smem"(%99, %172) : (memref<40xf32>, i32) -> f32
      %174 = "arith.subf"(%173, %164) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %175 = "arith.index_cast"(%arg11) : (index) -> i32
      %176 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %177 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %178 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %179 = "arith.muli"(%175, %178) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %180 = "arith.addi"(%177, %179) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %181 = "arith.addi"(%180, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %182 = "arith.addi"(%176, %181) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%174, %99, %182) : (f32, memref<40xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %105 = "arith.constant"() <{value = 40 : i32}> : () -> i32
    %106 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%98, %99, %106, %105) <{direction = 1 : i32}> : (i32, memref<40xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 3 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<20x2xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<20x2xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2050 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<20x2xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<20x2xf32>, memref<20x2xf32>, memref<20x2xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_51__direct"}> ({
  ^bb0(%arg4: i5, %arg5: i5, %arg6: i5):
    %38 = "aps.read_irf"(%arg4) : (i5) -> i32
    %39 = "aps.read_irf"(%arg5) : (i5) -> i32
    %40 = "memref.get_global"() <{name = @au_generated_candidate_51__direct_buffer}> : () -> memref<40xf32>
    %41 = "arith.constant"() <{value = 40 : i32}> : () -> i32
    %42 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%38, %40, %42, %41) <{direction = 0 : i32}> : (i32, memref<40xf32>, i32, i32) -> ()
    %43 = "memref.get_global"() <{name = @au_generated_candidate_51__direct_buffer_5}> : () -> memref<1xi32>
    %44 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %45 = "arith.addi"(%39, %44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %46 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %47 = "aps.load"(%45) : (i32) -> i32
    "aps.write_smem"(%47, %43, %46) : (i32, memref<1xi32>, i32) -> ()
    %48 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %49 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %50 = "arith.constant"() <{value = 1.810000e+02 : f32}> : () -> f32
    %51 = "arith.constant"() <{value = 1.510000e+02 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg7: index):
      %56 = "arith.index_cast"(%arg7) : (index) -> i32
      %57 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %58 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %59 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %60 = "arith.muli"(%56, %59) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %61 = "arith.addi"(%58, %60) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %62 = "arith.addi"(%61, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %63 = "arith.addi"(%57, %62) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %64 = "aps.read_smem"(%40, %63) : (memref<40xf32>, i32) -> f32
      %65 = "arith.mulf"(%64, %51) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %66 = "arith.fptosi"(%65) : (f32) -> i32
      %67 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %68 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %69 = "arith.addi"(%67, %68) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %70 = "aps.read_smem"(%43, %69) : (memref<1xi32>, i32) -> i32
      %71 = "arith.addi"(%70, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %72 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %73 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %74 = "arith.addi"(%72, %73) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%71, %43, %74) : (i32, memref<1xi32>, i32) -> ()
      %75 = "arith.index_cast"(%arg7) : (index) -> i32
      %76 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %77 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %78 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %79 = "arith.muli"(%75, %78) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %80 = "arith.addi"(%77, %79) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %81 = "arith.addi"(%80, %48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %82 = "arith.addi"(%76, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %83 = "aps.read_smem"(%40, %82) : (memref<40xf32>, i32) -> f32
      %84 = "arith.mulf"(%83, %50) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %85 = "arith.fptosi"(%84) : (f32) -> i32
      %86 = "arith.addi"(%71, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %87 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %88 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %89 = "arith.addi"(%87, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%86, %43, %89) : (i32, memref<1xi32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %52 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %53 = "arith.addi"(%39, %52) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %54 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %55 = "aps.read_smem"(%43, %54) : (memref<1xi32>, i32) -> i32
    "aps.store"(%55, %53) : (i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 4 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<20x2xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<20x2xf32>, memref<i32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_56__l0u"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "memref.get_global"() <{name = @au_generated_candidate_56__l0u_buffer}> : () -> memref<40xf32>
    %2 = "arith.constant"() <{value = 40 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %1, %3, %2) <{direction = 0 : i32}> : (i32, memref<40xf32>, i32, i32) -> ()
    %4 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 1.200000e+00 : f32}> : () -> f32
    %7 = "arith.constant"() <{value = 2.700000e-01 : f32}> : () -> f32
    %8 = "arith.constant"() <{value = 1.600000e+00 : f32}> : () -> f32
    %9 = "arith.constant"() <{value = 3.800000e-01 : f32}> : () -> f32
    %10 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg3: index):
      %13 = "arith.index_cast"(%arg3) : (index) -> i32
      %14 = "arith.remsi"(%13, %10) : (i32, i32) -> i32
      %15 = "arith.sitofp"(%14) : (i32) -> f32
      %16 = "arith.mulf"(%15, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %17 = "arith.addf"(%16, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %18 = "arith.index_cast"(%arg3) : (index) -> i32
      %19 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %20 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %21 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %22 = "arith.muli"(%18, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %23 = "arith.addi"(%20, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %24 = "arith.addi"(%23, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %25 = "arith.addi"(%19, %24) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%17, %1, %25) : (f32, memref<40xf32>, i32) -> ()
      %26 = "arith.divsi"(%13, %10) : (i32, i32) -> i32
      %27 = "arith.sitofp"(%26) : (i32) -> f32
      %28 = "arith.mulf"(%27, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %29 = "arith.addf"(%28, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %30 = "arith.index_cast"(%arg3) : (index) -> i32
      %31 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %32 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %33 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %34 = "arith.muli"(%30, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %35 = "arith.addi"(%32, %34) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %36 = "arith.addi"(%35, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %37 = "arith.addi"(%31, %36) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%29, %1, %37) : (f32, memref<40xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %11 = "arith.constant"() <{value = 40 : i32}> : () -> i32
    %12 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %1, %12, %11) <{direction = 1 : i32}> : (i32, memref<40xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 2 : i32, funct7 = 5 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<20x2xf32>, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<20x2xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_12__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_41__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
}) : () -> ()

