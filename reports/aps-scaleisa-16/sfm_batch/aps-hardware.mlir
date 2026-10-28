#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (24)>
#map2 = affine_map<() -> (8)>
#map3 = affine_map<() -> (3)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_9__l0u_buffer", type = memref<24xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_9__l0u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_49__l1p_buffer_4", type = memref<24xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_49__l1p_buffer_4"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_49__l1p_buffer_3", type = memref<24xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_49__l1p_buffer_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_49__l1p_buffer", type = memref<24xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_49__l1p_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_48__l0p_l1u_buffer_2", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_48__l0p_l1u_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_48__l0p_l1u_buffer_1", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_48__l0p_l1u_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_48__l0p_l1u_buffer", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_48__l0p_l1u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_44__l0u_buffer", type = memref<24xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_44__l0u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_38__l0u_buffer", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_38__l0u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_22__direct_buffer_0", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_22__direct_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_22__direct_buffer", type = memref<72xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_22__direct_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_22__direct"}> ({
  ^bb0(%arg22: i5, %arg23: i5, %arg24: i5):
    %299 = "aps.read_irf"(%arg22) : (i5) -> i32
    %300 = "aps.read_irf"(%arg23) : (i5) -> i32
    %301 = "memref.get_global"() <{name = @au_generated_candidate_22__direct_buffer}> : () -> memref<72xf32>
    %302 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %303 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%299, %301, %303, %302) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %304 = "memref.get_global"() <{name = @au_generated_candidate_22__direct_buffer_0}> : () -> memref<1xi32>
    %305 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %306 = "arith.addi"(%300, %305) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %307 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %308 = "aps.load"(%306) : (i32) -> i32
    "aps.write_smem"(%308, %304, %307) : (i32, memref<1xi32>, i32) -> ()
    %309 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %310 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %311 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %312 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
    %313 = "arith.constant"() <{value = 1.090000e+02 : f32}> : () -> f32
    %314 = "arith.constant"() <{value = 8.300000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg25: index):
      %319 = "arith.index_cast"(%arg25) : (index) -> i32
      %320 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %321 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %322 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %323 = "arith.muli"(%319, %322) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %324 = "arith.addi"(%321, %323) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %325 = "arith.addi"(%324, %311) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %326 = "arith.addi"(%320, %325) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %327 = "aps.read_smem"(%301, %326) : (memref<72xf32>, i32) -> f32
      %328 = "arith.mulf"(%327, %314) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %329 = "arith.fptosi"(%328) : (f32) -> i32
      %330 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %331 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %332 = "arith.addi"(%330, %331) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %333 = "aps.read_smem"(%304, %332) : (memref<1xi32>, i32) -> i32
      %334 = "arith.addi"(%333, %329) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %335 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %336 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %337 = "arith.addi"(%335, %336) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%334, %304, %337) : (i32, memref<1xi32>, i32) -> ()
      %338 = "arith.index_cast"(%arg25) : (index) -> i32
      %339 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %340 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %341 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %342 = "arith.muli"(%338, %341) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %343 = "arith.addi"(%340, %342) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %344 = "arith.addi"(%343, %310) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %345 = "arith.addi"(%339, %344) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %346 = "aps.read_smem"(%301, %345) : (memref<72xf32>, i32) -> f32
      %347 = "arith.mulf"(%346, %313) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %348 = "arith.fptosi"(%347) : (f32) -> i32
      %349 = "arith.addi"(%334, %348) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %350 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %351 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %352 = "arith.addi"(%350, %351) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%349, %304, %352) : (i32, memref<1xi32>, i32) -> ()
      %353 = "arith.index_cast"(%arg25) : (index) -> i32
      %354 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %355 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %356 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %357 = "arith.muli"(%353, %356) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %358 = "arith.addi"(%355, %357) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %359 = "arith.addi"(%358, %309) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %360 = "arith.addi"(%354, %359) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %361 = "aps.read_smem"(%301, %360) : (memref<72xf32>, i32) -> f32
      %362 = "arith.mulf"(%361, %312) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %363 = "arith.fptosi"(%362) : (f32) -> i32
      %364 = "arith.addi"(%349, %363) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %365 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %366 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %367 = "arith.addi"(%365, %366) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%364, %304, %367) : (i32, memref<1xi32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %315 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %316 = "arith.addi"(%300, %315) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %317 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %318 = "aps.read_smem"(%304, %317) : (memref<1xi32>, i32) -> i32
    "aps.store"(%318, %316) : (i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<24x3xf32>, memref<i32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_38__l0u"}> ({
  ^bb0(%arg18: i5, %arg19: i5, %arg20: i5):
    %245 = "aps.read_irf"(%arg18) : (i5) -> i32
    %246 = "memref.get_global"() <{name = @au_generated_candidate_38__l0u_buffer}> : () -> memref<72xf32>
    %247 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %248 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%245, %246, %248, %247) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %249 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %250 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %251 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %252 = "arith.constant"() <{value = 2.800000e+00 : f32}> : () -> f32
    %253 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %254 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %255 = "arith.constant"() <{value = -0.899999976 : f32}> : () -> f32
    %256 = "arith.constant"() <{value = 6.000000e-01 : f32}> : () -> f32
    %257 = "arith.constant"() <{value = -1.200000e+00 : f32}> : () -> f32
    %258 = "arith.constant"() <{value = 4.500000e-01 : f32}> : () -> f32
    %259 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg21: index):
      %262 = "arith.index_cast"(%arg21) : (index) -> i32
      %263 = "arith.remsi"(%262, %259) : (i32, i32) -> i32
      %264 = "arith.sitofp"(%263) : (i32) -> f32
      %265 = "arith.mulf"(%264, %258) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %266 = "arith.addf"(%265, %257) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %267 = "arith.index_cast"(%arg21) : (index) -> i32
      %268 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %269 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %270 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %271 = "arith.muli"(%267, %270) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %272 = "arith.addi"(%269, %271) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %273 = "arith.addi"(%272, %251) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %274 = "arith.addi"(%268, %273) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%266, %246, %274) : (f32, memref<72xf32>, i32) -> ()
      %275 = "arith.divsi"(%262, %259) : (i32, i32) -> i32
      %276 = "arith.sitofp"(%275) : (i32) -> f32
      %277 = "arith.mulf"(%276, %256) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %278 = "arith.addf"(%277, %255) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %279 = "arith.index_cast"(%arg21) : (index) -> i32
      %280 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %281 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %282 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %283 = "arith.muli"(%279, %282) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %284 = "arith.addi"(%281, %283) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %285 = "arith.addi"(%284, %250) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %286 = "arith.addi"(%280, %285) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%278, %246, %286) : (f32, memref<72xf32>, i32) -> ()
      %287 = "arith.remsi"(%262, %254) : (i32, i32) -> i32
      %288 = "arith.sitofp"(%287) : (i32) -> f32
      %289 = "arith.mulf"(%288, %253) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %290 = "arith.addf"(%289, %252) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %291 = "arith.index_cast"(%arg21) : (index) -> i32
      %292 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %293 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %294 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %295 = "arith.muli"(%291, %294) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %296 = "arith.addi"(%293, %295) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %297 = "arith.addi"(%296, %249) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %298 = "arith.addi"(%292, %297) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%290, %246, %298) : (f32, memref<72xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %260 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %261 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%245, %246, %261, %260) <{direction = 1 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 2 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<24x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_44__l0u"}> ({
  ^bb0(%arg14: i5, %arg15: i5, %arg16: i5):
    %195 = "aps.read_irf"(%arg14) : (i5) -> i32
    %196 = "memref.get_global"() <{name = @au_generated_candidate_44__l0u_buffer}> : () -> memref<24xf32>
    %197 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %198 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%195, %196, %198, %197) <{direction = 0 : i32}> : (i32, memref<24xf32>, i32, i32) -> ()
    %199 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %200 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %201 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %202 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %203 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
    %204 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %205 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %206 = "arith.constant"() <{value = -1.800000e+00 : f32}> : () -> f32
    %207 = "arith.constant"() <{value = 5.500000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg17: index):
      %210 = "arith.index_cast"(%arg17) : (index) -> i32
      %211 = "arith.sitofp"(%210) : (i32) -> f32
      %212 = "arith.mulf"(%211, %207) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %213 = "arith.addf"(%212, %206) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %214 = "arith.index_cast"(%arg17) : (index) -> i32
      %215 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %216 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %217 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %218 = "arith.muli"(%214, %217) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %219 = "arith.addi"(%216, %218) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %220 = "arith.addi"(%219, %200) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %221 = "arith.addi"(%215, %220) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%213, %196, %221) : (f32, memref<24xf32>, i32) -> ()
      %222 = "arith.remsi"(%210, %205) : (i32, i32) -> i32
      %223 = "arith.addi"(%222, %204) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %224 = "arith.sitofp"(%223) : (i32) -> f32
      %225 = "arith.mulf"(%224, %203) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %226 = "arith.index_cast"(%arg17) : (index) -> i32
      %227 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %228 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %229 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %230 = "arith.muli"(%226, %229) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %231 = "arith.addi"(%228, %230) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %232 = "arith.addi"(%231, %199) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %233 = "arith.addi"(%227, %232) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%225, %196, %233) : (f32, memref<24xf32>, i32) -> ()
      %234 = "arith.remsi"(%210, %202) : (i32, i32) -> i32
      %235 = "arith.sitofp"(%234) : (i32) -> f32
      %236 = "arith.mulf"(%235, %201) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %237 = "arith.index_cast"(%arg17) : (index) -> i32
      %238 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %239 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %240 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %241 = "arith.muli"(%237, %240) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %242 = "arith.addi"(%239, %241) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %243 = "arith.addi"(%242, %202) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %244 = "arith.addi"(%238, %243) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%236, %196, %244) : (f32, memref<24xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %208 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %209 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%195, %196, %209, %208) <{direction = 1 : i32}> : (i32, memref<24xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 2 : i32, funct7 = 2 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<8x3xf32>, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<8x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_48__l0p_l1u"}> ({
  ^bb0(%arg9: i5, %arg10: i5, %arg11: i5):
    %138 = "aps.read_irf"(%arg9) : (i5) -> i32
    %139 = "aps.read_irf"(%arg10) : (i5) -> i32
    %140 = "memref.get_global"() <{name = @au_generated_candidate_48__l0p_l1u_buffer}> : () -> memref<72xf32>
    %141 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %142 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%138, %140, %142, %141) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %143 = "memref.get_global"() <{name = @au_generated_candidate_48__l0p_l1u_buffer_1}> : () -> memref<72xf32>
    %144 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %145 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%139, %143, %145, %144) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %146 = "aps.read_csr"() <{global_name = @au_generated_candidate_48__l0p_l1u_csr0}> : () -> i32
    %147 = "memref.get_global"() <{name = @au_generated_candidate_48__l0p_l1u_buffer_2}> : () -> memref<72xf32>
    %148 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %149 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%146, %147, %149, %148) <{direction = 0 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    %150 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg12: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg13: index):
        %153 = "arith.index_cast"(%arg12) : (index) -> i32
        %154 = "arith.index_cast"(%arg13) : (index) -> i32
        %155 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %156 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %157 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %158 = "arith.muli"(%153, %157) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %159 = "arith.addi"(%156, %158) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %160 = "arith.addi"(%159, %154) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %161 = "arith.addi"(%155, %160) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %162 = "aps.read_smem"(%140, %161) : (memref<72xf32>, i32) -> f32
        %163 = "arith.mulf"(%162, %150) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %164 = "arith.index_cast"(%arg12) : (index) -> i32
        %165 = "arith.index_cast"(%arg13) : (index) -> i32
        %166 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %167 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %168 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %169 = "arith.muli"(%164, %168) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %170 = "arith.addi"(%167, %169) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %171 = "arith.addi"(%170, %165) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %172 = "arith.addi"(%166, %171) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %173 = "aps.read_smem"(%143, %172) : (memref<72xf32>, i32) -> f32
        %174 = "arith.divf"(%163, %173) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %175 = "arith.index_cast"(%arg12) : (index) -> i32
        %176 = "arith.index_cast"(%arg13) : (index) -> i32
        %177 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %178 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %179 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %180 = "arith.muli"(%175, %179) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %181 = "arith.addi"(%178, %180) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %182 = "arith.addi"(%181, %176) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %183 = "arith.addi"(%177, %182) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %184 = "aps.read_smem"(%147, %183) : (memref<72xf32>, i32) -> f32
        %185 = "arith.subf"(%184, %174) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %186 = "arith.index_cast"(%arg12) : (index) -> i32
        %187 = "arith.index_cast"(%arg13) : (index) -> i32
        %188 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %189 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %190 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %191 = "arith.muli"(%186, %190) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %192 = "arith.addi"(%189, %191) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %193 = "arith.addi"(%192, %187) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %194 = "arith.addi"(%188, %193) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%185, %147, %194) : (f32, memref<72xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {pipeline = 1 : i64} : () -> ()
    %151 = "arith.constant"() <{value = 72 : i32}> : () -> i32
    %152 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%146, %147, %152, %151) <{direction = 1 : i32}> : (i32, memref<72xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 3 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<24x3xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_49__l1p"}> ({
  ^bb0(%arg4: i5, %arg5: i5, %arg6: i5):
    %81 = "aps.read_irf"(%arg4) : (i5) -> i32
    %82 = "aps.read_irf"(%arg5) : (i5) -> i32
    %83 = "memref.get_global"() <{name = @au_generated_candidate_49__l1p_buffer}> : () -> memref<24xf32>
    %84 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %85 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%81, %83, %85, %84) <{direction = 0 : i32}> : (i32, memref<24xf32>, i32, i32) -> ()
    %86 = "memref.get_global"() <{name = @au_generated_candidate_49__l1p_buffer_3}> : () -> memref<24xf32>
    %87 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %88 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%82, %86, %88, %87) <{direction = 0 : i32}> : (i32, memref<24xf32>, i32, i32) -> ()
    %89 = "aps.read_csr"() <{global_name = @au_generated_candidate_49__l1p_csr0}> : () -> i32
    %90 = "memref.get_global"() <{name = @au_generated_candidate_49__l1p_buffer_4}> : () -> memref<24xf32>
    %91 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %92 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%89, %90, %92, %91) <{direction = 0 : i32}> : (i32, memref<24xf32>, i32, i32) -> ()
    %93 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg7: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg8: index):
        %96 = "arith.index_cast"(%arg7) : (index) -> i32
        %97 = "arith.index_cast"(%arg8) : (index) -> i32
        %98 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %99 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %100 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %101 = "arith.muli"(%96, %100) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %102 = "arith.addi"(%99, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %103 = "arith.addi"(%102, %97) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %104 = "arith.addi"(%98, %103) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %105 = "aps.read_smem"(%83, %104) : (memref<24xf32>, i32) -> f32
        %106 = "arith.mulf"(%105, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %107 = "arith.index_cast"(%arg7) : (index) -> i32
        %108 = "arith.index_cast"(%arg8) : (index) -> i32
        %109 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %110 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %111 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %112 = "arith.muli"(%107, %111) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %113 = "arith.addi"(%110, %112) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %114 = "arith.addi"(%113, %108) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %115 = "arith.addi"(%109, %114) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %116 = "aps.read_smem"(%86, %115) : (memref<24xf32>, i32) -> f32
        %117 = "arith.divf"(%106, %116) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %118 = "arith.index_cast"(%arg7) : (index) -> i32
        %119 = "arith.index_cast"(%arg8) : (index) -> i32
        %120 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %121 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %122 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %123 = "arith.muli"(%118, %122) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %124 = "arith.addi"(%121, %123) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %125 = "arith.addi"(%124, %119) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %126 = "arith.addi"(%120, %125) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %127 = "aps.read_smem"(%90, %126) : (memref<24xf32>, i32) -> f32
        %128 = "arith.subf"(%127, %117) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %129 = "arith.index_cast"(%arg7) : (index) -> i32
        %130 = "arith.index_cast"(%arg8) : (index) -> i32
        %131 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %132 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %133 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %134 = "arith.muli"(%129, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %135 = "arith.addi"(%132, %134) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %136 = "arith.addi"(%135, %130) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %137 = "arith.addi"(%131, %136) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%128, %90, %137) : (f32, memref<24xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {pipeline = 1 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %94 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %95 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%89, %90, %95, %94) <{direction = 1 : i32}> : (i32, memref<24xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 4 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<8x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<8x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2050 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<8x3xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<8x3xf32>, memref<8x3xf32>, memref<8x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_9__l0u"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "memref.get_global"() <{name = @au_generated_candidate_9__l0u_buffer}> : () -> memref<24xf32>
    %2 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %1, %3, %2) <{direction = 0 : i32}> : (i32, memref<24xf32>, i32, i32) -> ()
    %4 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
    %7 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %9 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %11 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %12 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg3: index):
      %15 = "arith.index_cast"(%arg3) : (index) -> i32
      %16 = "arith.remsi"(%15, %12) : (i32, i32) -> i32
      %17 = "arith.addi"(%16, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %18 = "arith.sitofp"(%17) : (i32) -> f32
      %19 = "arith.mulf"(%18, %10) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %20 = "arith.index_cast"(%arg3) : (index) -> i32
      %21 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %22 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %23 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %24 = "arith.muli"(%20, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %25 = "arith.addi"(%22, %24) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %26 = "arith.addi"(%25, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %27 = "arith.addi"(%21, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %28 = "aps.read_smem"(%1, %27) : (memref<24xf32>, i32) -> f32
      %29 = "arith.addf"(%28, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %30 = "arith.index_cast"(%arg3) : (index) -> i32
      %31 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %32 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %33 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %34 = "arith.muli"(%30, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %35 = "arith.addi"(%32, %34) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %36 = "arith.addi"(%35, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %37 = "arith.addi"(%31, %36) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%29, %1, %37) : (f32, memref<24xf32>, i32) -> ()
      %38 = "arith.remsi"(%15, %9) : (i32, i32) -> i32
      %39 = "arith.sitofp"(%38) : (i32) -> f32
      %40 = "arith.mulf"(%39, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %41 = "arith.index_cast"(%arg3) : (index) -> i32
      %42 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %43 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %44 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %45 = "arith.muli"(%41, %44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %46 = "arith.addi"(%43, %45) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %47 = "arith.addi"(%46, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %48 = "arith.addi"(%42, %47) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %49 = "aps.read_smem"(%1, %48) : (memref<24xf32>, i32) -> f32
      %50 = "arith.subf"(%49, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %51 = "arith.index_cast"(%arg3) : (index) -> i32
      %52 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %53 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %54 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %55 = "arith.muli"(%51, %54) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %56 = "arith.addi"(%53, %55) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %57 = "arith.addi"(%56, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %58 = "arith.addi"(%52, %57) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%50, %1, %58) : (f32, memref<24xf32>, i32) -> ()
      %59 = "arith.remsi"(%15, %7) : (i32, i32) -> i32
      %60 = "arith.addi"(%59, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %61 = "arith.sitofp"(%60) : (i32) -> f32
      %62 = "arith.mulf"(%61, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %63 = "arith.index_cast"(%arg3) : (index) -> i32
      %64 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %65 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %66 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %67 = "arith.muli"(%63, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %68 = "arith.addi"(%65, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %69 = "arith.addi"(%68, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %70 = "arith.addi"(%64, %69) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %71 = "aps.read_smem"(%1, %70) : (memref<24xf32>, i32) -> f32
      %72 = "arith.addf"(%71, %62) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %73 = "arith.index_cast"(%arg3) : (index) -> i32
      %74 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %75 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %76 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %77 = "arith.muli"(%73, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %78 = "arith.addi"(%75, %77) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %79 = "arith.addi"(%78, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %80 = "arith.addi"(%74, %79) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%72, %1, %80) : (f32, memref<24xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %13 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %1, %14, %13) <{direction = 1 : i32}> : (i32, memref<24xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 2 : i32, funct7 = 5 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<8x3xf32>, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<8x3xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_48__l0p_l1u_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_49__l1p_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
}) : () -> ()

