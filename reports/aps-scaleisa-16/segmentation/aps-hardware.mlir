#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (3)>
#map2 = affine_map<() -> (8)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_35__l0u_l1u_l2u_buffer_5", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_35__l0u_l1u_l2u_buffer_5"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_35__l0u_l1u_l2u_buffer", type = memref<64xi32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [8], scaleisa.scratchpad, var_name = "au_generated_candidate_35__l0u_l1u_l2u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_21__l0u_l1u_buffer_4", type = memref<64xi32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [8], scaleisa.scratchpad, var_name = "au_generated_candidate_21__l0u_l1u_buffer_4"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_21__l0u_l1u_buffer", type = memref<64xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [8], scaleisa.scratchpad, var_name = "au_generated_candidate_21__l0u_l1u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_buffer_3", type = memref<64xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_buffer_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_buffer", type = memref<64xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__direct_buffer_2", type = memref<64xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__direct_buffer_1", type = memref<64xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__direct_buffer", type = memref<64xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__direct_local_0", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_local_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__direct_local", type = memref<i32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_local"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_0__direct"}> ({
  ^bb0(%arg16: i5, %arg17: i5, %arg18: i5):
    %200 = "aps.read_irf"(%arg16) : (i5) -> i32
    %201 = "aps.read_irf"(%arg17) : (i5) -> i32
    %202 = "memref.get_global"() <{name = @au_generated_candidate_0__direct_buffer}> : () -> memref<64xi32>
    %203 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %204 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%200, %202, %204, %203) <{direction = 0 : i32}> : (i32, memref<64xi32>, i32, i32) -> ()
    %205 = "arith.index_castui"(%201) : (i32) -> index
    %206 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__direct_csr0}> : () -> i32
    %207 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__direct_csr1}> : () -> i32
    %208 = "memref.get_global"() <{name = @au_generated_candidate_0__direct_buffer_1}> : () -> memref<64xf32>
    %209 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %210 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%207, %208, %210, %209) <{direction = 0 : i32}> : (i32, memref<64xf32>, i32, i32) -> ()
    %211 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__direct_csr2}> : () -> i32
    %212 = "memref.get_global"() <{name = @au_generated_candidate_0__direct_buffer_2}> : () -> memref<64xf32>
    %213 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %214 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%211, %212, %214, %213) <{direction = 0 : i32}> : (i32, memref<64xf32>, i32, i32) -> ()
    %215 = "arith.constant"() <{value = false}> : () -> i1
    %216 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %217 = "arith.constant"() <{value = true}> : () -> i1
    %218 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %219 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %220 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %221 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg19: index):
      %224 = "arith.index_cast"(%arg19) : (index) -> i32
      %225 = "arith.index_cast"(%205) : (index) -> i32
      %226 = "arith.index_cast"(%arg19) : (index) -> i32
      %227 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %228 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %229 = "arith.constant"() <{value = 8 : i32}> : () -> i32
      %230 = "arith.muli"(%225, %229) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %231 = "arith.addi"(%228, %230) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %232 = "arith.addi"(%231, %226) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %233 = "arith.addi"(%227, %232) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %234 = "aps.read_smem"(%202, %233) : (memref<64xi32>, i32) -> i32
      %235 = "memref.get_global"() <{name = @au_generated_candidate_0__direct_local}> : () -> memref<i32>
      "aps.globalstore"(%234) <{global_name = @au_generated_candidate_0__direct_local}> : (i32) -> ()
      %236 = "memref.get_global"() <{name = @au_generated_candidate_0__direct_local_0}> : () -> memref<f32>
      "aps.globalstore"(%221) <{global_name = @au_generated_candidate_0__direct_local_0}> : (f32) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg20: index):
        %247 = "arith.index_cast"(%arg20) : (index) -> i32
        %248 = "arith.addi"(%247, %220) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg21: index):
          %249 = "arith.index_cast"(%arg21) : (index) -> i32
          %250 = "arith.addi"(%249, %220) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %251 = "arith.addi"(%206, %248) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %252 = "arith.cmpi"(%251, %219) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %253 = "arith.cmpi"(%251, %218) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %254 = "arith.andi"(%252, %253) : (i1, i1) -> i1
          %255 = "arith.xori"(%254, %217) : (i1, i1) -> i1
          %256 = "arith.andi"(%255, %252) : (i1, i1) -> i1
          %257 = "arith.select"(%254, %218, %219) : (i1, i32, i32) -> i32
          %258 = "arith.select"(%256, %251, %257) : (i1, i32, i32) -> i32
          %259 = "arith.addi"(%224, %250) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %260 = "arith.cmpi"(%259, %219) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %261 = "arith.cmpi"(%259, %218) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %262 = "arith.andi"(%260, %261) : (i1, i1) -> i1
          %263 = "arith.xori"(%262, %217) : (i1, i1) -> i1
          %264 = "arith.andi"(%263, %260) : (i1, i1) -> i1
          %265 = "arith.select"(%262, %218, %219) : (i1, i32, i32) -> i32
          %266 = "arith.select"(%264, %259, %265) : (i1, i32, i32) -> i32
          %267 = "arith.index_cast"(%205) : (index) -> i32
          %268 = "arith.index_cast"(%arg19) : (index) -> i32
          %269 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %270 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %271 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %272 = "arith.muli"(%267, %271) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %273 = "arith.addi"(%270, %272) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %274 = "arith.addi"(%273, %268) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %275 = "arith.addi"(%269, %274) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %276 = "aps.read_smem"(%208, %275) : (memref<64xf32>, i32) -> f32
          %277 = "arith.index_cast"(%258) : (i32) -> index
          %278 = "arith.index_cast"(%266) : (i32) -> index
          %279 = "arith.index_cast"(%277) : (index) -> i32
          %280 = "arith.index_cast"(%278) : (index) -> i32
          %281 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %282 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %283 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %284 = "arith.muli"(%279, %283) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %285 = "arith.addi"(%282, %284) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %286 = "arith.addi"(%285, %280) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %287 = "arith.addi"(%281, %286) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %288 = "aps.read_smem"(%208, %287) : (memref<64xf32>, i32) -> f32
          %289 = "arith.subf"(%276, %288) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %290 = "arith.cmpf"(%289, %216) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %291 = "scf.if"(%290) ({
            %330 = "arith.negf"(%289) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
            "scf.yield"(%330) : (f32) -> ()
          }, {
            "scf.yield"(%289) : (f32) -> ()
          }) : (i1) -> f32
          %292 = "arith.index_cast"(%277) : (index) -> i32
          %293 = "arith.index_cast"(%278) : (index) -> i32
          %294 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %295 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %296 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %297 = "arith.muli"(%292, %296) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %298 = "arith.addi"(%295, %297) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %299 = "arith.addi"(%298, %293) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %300 = "arith.addi"(%294, %299) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %301 = "aps.read_smem"(%212, %300) : (memref<64xf32>, i32) -> f32
          %302 = "arith.addf"(%291, %301) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %303 = "aps.globalload"() <{global_name = @au_generated_candidate_0__direct_local_0}> : () -> f32
          %304 = "arith.cmpf"(%302, %303) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %305 = "scf.if"(%304) ({
            "scf.yield"(%217) : (i1) -> ()
          }, {
            %316 = "arith.cmpf"(%302, %303) <{fastmath = #arith.fastmath<none>, predicate = 1 : i64}> : (f32, f32) -> i1
            %317 = "scf.if"(%316) ({
              %318 = "arith.index_cast"(%277) : (index) -> i32
              %319 = "arith.index_cast"(%278) : (index) -> i32
              %320 = "arith.constant"() <{value = 0 : i32}> : () -> i32
              %321 = "arith.constant"() <{value = 0 : i32}> : () -> i32
              %322 = "arith.constant"() <{value = 8 : i32}> : () -> i32
              %323 = "arith.muli"(%318, %322) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %324 = "arith.addi"(%321, %323) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %325 = "arith.addi"(%324, %319) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %326 = "arith.addi"(%320, %325) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %327 = "aps.read_smem"(%202, %326) : (memref<64xi32>, i32) -> i32
              %328 = "aps.globalload"() <{global_name = @au_generated_candidate_0__direct_local}> : () -> i32
              %329 = "arith.cmpi"(%327, %328) <{predicate = 2 : i64}> : (i32, i32) -> i1
              "scf.yield"(%329) : (i1) -> ()
            }, {
              "scf.yield"(%215) : (i1) -> ()
            }) : (i1) -> i1
            "scf.yield"(%317) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%305) ({
            "aps.globalstore"(%302) <{global_name = @au_generated_candidate_0__direct_local_0}> : (f32) -> ()
            %306 = "arith.index_cast"(%277) : (index) -> i32
            %307 = "arith.index_cast"(%278) : (index) -> i32
            %308 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %309 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %310 = "arith.constant"() <{value = 8 : i32}> : () -> i32
            %311 = "arith.muli"(%306, %310) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %312 = "arith.addi"(%309, %311) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %313 = "arith.addi"(%312, %307) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %314 = "arith.addi"(%308, %313) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %315 = "aps.read_smem"(%202, %314) : (memref<64xi32>, i32) -> i32
            "aps.globalstore"(%315) <{global_name = @au_generated_candidate_0__direct_local}> : (i32) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %237 = "aps.globalload"() <{global_name = @au_generated_candidate_0__direct_local}> : () -> i32
      %238 = "arith.index_cast"(%205) : (index) -> i32
      %239 = "arith.index_cast"(%arg19) : (index) -> i32
      %240 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %241 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %242 = "arith.constant"() <{value = 8 : i32}> : () -> i32
      %243 = "arith.muli"(%238, %242) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %244 = "arith.addi"(%241, %243) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %245 = "arith.addi"(%244, %239) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %246 = "arith.addi"(%240, %245) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%237, %202, %246) : (i32, memref<64xi32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %222 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %223 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%200, %202, %223, %222) <{direction = 1 : i32}> : (i32, memref<64xi32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<8x8xi32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr"}, {address = 2050 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<8x8xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2051 : i32, arg_index = 4 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<8x8xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<8x8xi32>, index, i32, memref<8x8xf32>, memref<8x8xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_10__direct"}> ({
  ^bb0(%arg11: i5, %arg12: i5, %arg13: i5):
    %90 = "aps.read_irf"(%arg11) : (i5) -> i32
    %91 = "aps.read_irf"(%arg12) : (i5) -> i32
    %92 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_buffer}> : () -> memref<64xf32>
    %93 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %94 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%90, %92, %94, %93) <{direction = 0 : i32}> : (i32, memref<64xf32>, i32, i32) -> ()
    %95 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_buffer_3}> : () -> memref<64xf32>
    %96 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %97 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %98 = "arith.constant"() <{value = true}> : () -> i1
    %99 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %100 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %101 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg14: index):
      %104 = "arith.index_cast"(%arg14) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg15: index):
        %105 = "arith.index_cast"(%arg15) : (index) -> i32
        %106 = "arith.addi"(%105, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %107 = "arith.cmpi"(%106, %100) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %108 = "arith.cmpi"(%106, %99) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %109 = "arith.andi"(%107, %108) : (i1, i1) -> i1
        %110 = "arith.xori"(%109, %98) : (i1, i1) -> i1
        %111 = "arith.andi"(%110, %107) : (i1, i1) -> i1
        %112 = "arith.select"(%109, %99, %100) : (i1, i32, i32) -> i32
        %113 = "arith.select"(%111, %106, %112) : (i1, i32, i32) -> i32
        %114 = "arith.addi"(%105, %97) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %115 = "arith.cmpi"(%114, %100) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %116 = "arith.cmpi"(%114, %99) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %117 = "arith.andi"(%115, %116) : (i1, i1) -> i1
        %118 = "arith.xori"(%117, %98) : (i1, i1) -> i1
        %119 = "arith.andi"(%118, %115) : (i1, i1) -> i1
        %120 = "arith.select"(%117, %99, %100) : (i1, i32, i32) -> i32
        %121 = "arith.select"(%119, %114, %120) : (i1, i32, i32) -> i32
        %122 = "arith.addi"(%104, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %123 = "arith.cmpi"(%122, %100) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %124 = "arith.cmpi"(%122, %99) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %125 = "arith.andi"(%123, %124) : (i1, i1) -> i1
        %126 = "arith.xori"(%125, %98) : (i1, i1) -> i1
        %127 = "arith.andi"(%126, %123) : (i1, i1) -> i1
        %128 = "arith.select"(%125, %99, %100) : (i1, i32, i32) -> i32
        %129 = "arith.select"(%127, %122, %128) : (i1, i32, i32) -> i32
        %130 = "arith.addi"(%104, %97) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %131 = "arith.cmpi"(%130, %100) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %132 = "arith.cmpi"(%130, %99) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %133 = "arith.andi"(%131, %132) : (i1, i1) -> i1
        %134 = "arith.subi"(%98, %133) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
        %135 = "arith.andi"(%134, %131) : (i1, i1) -> i1
        %136 = "arith.select"(%133, %99, %100) : (i1, i32, i32) -> i32
        %137 = "arith.select"(%135, %130, %136) : (i1, i32, i32) -> i32
        %138 = "arith.index_cast"(%121) : (i32) -> index
        %139 = "arith.index_cast"(%arg14) : (index) -> i32
        %140 = "arith.index_cast"(%138) : (index) -> i32
        %141 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %142 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %143 = "arith.constant"() <{value = 8 : i32}> : () -> i32
        %144 = "arith.muli"(%139, %143) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %145 = "arith.addi"(%142, %144) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %146 = "arith.addi"(%145, %140) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %147 = "arith.addi"(%141, %146) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %148 = "aps.read_smem"(%92, %147) : (memref<64xf32>, i32) -> f32
        %149 = "arith.index_cast"(%113) : (i32) -> index
        %150 = "arith.index_cast"(%arg14) : (index) -> i32
        %151 = "arith.index_cast"(%149) : (index) -> i32
        %152 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %153 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %154 = "arith.constant"() <{value = 8 : i32}> : () -> i32
        %155 = "arith.muli"(%150, %154) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %156 = "arith.addi"(%153, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %157 = "arith.addi"(%156, %151) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %158 = "arith.addi"(%152, %157) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %159 = "aps.read_smem"(%92, %158) : (memref<64xf32>, i32) -> f32
        %160 = "arith.subf"(%148, %159) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %161 = "arith.cmpf"(%160, %96) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %162 = "scf.if"(%161) ({
          %199 = "arith.negf"(%160) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%199) : (f32) -> ()
        }, {
          "scf.yield"(%160) : (f32) -> ()
        }) : (i1) -> f32
        %163 = "arith.index_cast"(%137) : (i32) -> index
        %164 = "arith.index_cast"(%163) : (index) -> i32
        %165 = "arith.index_cast"(%arg15) : (index) -> i32
        %166 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %167 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %168 = "arith.constant"() <{value = 8 : i32}> : () -> i32
        %169 = "arith.muli"(%164, %168) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %170 = "arith.addi"(%167, %169) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %171 = "arith.addi"(%170, %165) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %172 = "arith.addi"(%166, %171) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %173 = "aps.read_smem"(%92, %172) : (memref<64xf32>, i32) -> f32
        %174 = "arith.index_cast"(%129) : (i32) -> index
        %175 = "arith.index_cast"(%174) : (index) -> i32
        %176 = "arith.index_cast"(%arg15) : (index) -> i32
        %177 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %178 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %179 = "arith.constant"() <{value = 8 : i32}> : () -> i32
        %180 = "arith.muli"(%175, %179) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %181 = "arith.addi"(%178, %180) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %182 = "arith.addi"(%181, %176) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %183 = "arith.addi"(%177, %182) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %184 = "aps.read_smem"(%92, %183) : (memref<64xf32>, i32) -> f32
        %185 = "arith.subf"(%173, %184) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %186 = "arith.cmpf"(%185, %96) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %187 = "scf.if"(%186) ({
          %198 = "arith.negf"(%185) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%198) : (f32) -> ()
        }, {
          "scf.yield"(%185) : (f32) -> ()
        }) : (i1) -> f32
        %188 = "arith.addf"(%162, %187) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %189 = "arith.index_cast"(%arg14) : (index) -> i32
        %190 = "arith.index_cast"(%arg15) : (index) -> i32
        %191 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %192 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %193 = "arith.constant"() <{value = 8 : i32}> : () -> i32
        %194 = "arith.muli"(%189, %193) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %195 = "arith.addi"(%192, %194) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %196 = "arith.addi"(%195, %190) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %197 = "arith.addi"(%191, %196) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%188, %95, %197) : (f32, memref<64xf32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %102 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %103 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%91, %95, %103, %102) <{direction = 1 : i32}> : (i32, memref<64xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<8x8xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<8x8xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<8x8xf32>, memref<8x8xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_21__l0u_l1u"}> ({
  ^bb0(%arg6: i5, %arg7: i5, %arg8: i5):
    %34 = "aps.read_irf"(%arg6) : (i5) -> i32
    %35 = "aps.read_irf"(%arg7) : (i5) -> i32
    %36 = "memref.get_global"() <{name = @au_generated_candidate_21__l0u_l1u_buffer}> : () -> memref<64xf32>
    %37 = "memref.get_global"() <{name = @au_generated_candidate_21__l0u_l1u_buffer_4}> : () -> memref<64xi32>
    %38 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %39 = "arith.constant"() <{value = 5.000000e-03 : f32}> : () -> f32
    %40 = "arith.constant"() <{value = 9 : i32}> : () -> i32
    %41 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %42 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %43 = "arith.constant"() <{value = 1.000000e-01 : f32}> : () -> f32
    %44 = "arith.constant"() <{value = 30 : i32}> : () -> i32
    %45 = "arith.constant"() <{value = 56 : i32}> : () -> i32
    %46 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg9: index):
      %51 = "arith.index_cast"(%arg9) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg10: index):
        %52 = "arith.index_cast"(%arg10) : (index) -> i32
        %53 = "arith.cmpi"(%52, %46) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %54 = "arith.extui"(%53) : (i1) -> i32
        %55 = "arith.cmpi"(%52, %45) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %56 = "arith.extui"(%55) : (i1) -> i32
        %57 = "arith.addi"(%54, %56) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %58 = "arith.cmpi"(%51, %44) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %59 = "arith.extui"(%58) : (i1) -> i32
        %60 = "arith.addi"(%57, %59) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %61 = "arith.sitofp"(%60) : (i32) -> f32
        %62 = "arith.mulf"(%61, %43) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %63 = "arith.muli"(%52, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %64 = "arith.muli"(%51, %41) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %65 = "arith.addi"(%63, %64) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %66 = "arith.remsi"(%65, %40) : (i32, i32) -> i32
        %67 = "arith.sitofp"(%66) : (i32) -> f32
        %68 = "arith.mulf"(%67, %39) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %69 = "arith.addf"(%62, %68) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %70 = "arith.index_cast"(%arg9) : (index) -> i32
        %71 = "arith.index_cast"(%arg10) : (index) -> i32
        %72 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %73 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %74 = "arith.constant"() <{value = 8 : i32}> : () -> i32
        %75 = "arith.muli"(%70, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %76 = "arith.addi"(%73, %75) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %77 = "arith.addi"(%76, %71) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %78 = "arith.addi"(%72, %77) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%69, %36, %78) : (f32, memref<64xf32>, i32) -> ()
        %79 = "arith.muli"(%51, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %80 = "arith.addi"(%79, %52) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %81 = "arith.index_cast"(%arg9) : (index) -> i32
        %82 = "arith.index_cast"(%arg10) : (index) -> i32
        %83 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %84 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %85 = "arith.constant"() <{value = 8 : i32}> : () -> i32
        %86 = "arith.muli"(%81, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %87 = "arith.addi"(%84, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %88 = "arith.addi"(%87, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %89 = "arith.addi"(%83, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%80, %37, %89) : (i32, memref<64xi32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %47 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %48 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%34, %36, %48, %47) <{direction = 1 : i32}> : (i32, memref<64xf32>, i32, i32) -> ()
    %49 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %50 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%35, %37, %50, %49) <{direction = 1 : i32}> : (i32, memref<64xi32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 2 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<8x8xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<8x8xi32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<8x8xf32>, memref<8x8xi32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_35__l0u_l1u_l2u"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_35__l0u_l1u_l2u_buffer}> : () -> memref<64xi32>
    %3 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %2, %4, %3) <{direction = 0 : i32}> : (i32, memref<64xi32>, i32, i32) -> ()
    %5 = "memref.get_global"() <{name = @au_generated_candidate_35__l0u_l1u_l2u_buffer_5}> : () -> memref<1xi32>
    %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "arith.addi"(%1, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %8 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %9 = "aps.load"(%7) : (i32) -> i32
    "aps.write_smem"(%9, %5, %8) : (i32, memref<1xi32>, i32) -> ()
    %10 = "arith.constant"() <{value = 255 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg3: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 8 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg4: index):
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg5: index):
          %15 = "arith.index_cast"(%arg3) : (index) -> i32
          %16 = "arith.index_cast"(%arg5) : (index) -> i32
          %17 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %18 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %19 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %20 = "arith.muli"(%15, %19) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %21 = "arith.addi"(%18, %20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %22 = "arith.addi"(%21, %16) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %23 = "arith.addi"(%17, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %24 = "aps.read_smem"(%2, %23) : (memref<64xi32>, i32) -> i32
          %25 = "arith.andi"(%24, %10) : (i32, i32) -> i32
          %26 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %27 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %28 = "arith.addi"(%26, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %29 = "aps.read_smem"(%5, %28) : (memref<1xi32>, i32) -> i32
          %30 = "arith.addi"(%29, %25) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %31 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %32 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %33 = "arith.addi"(%31, %32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%30, %5, %33) : (i32, memref<1xi32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) {unroll = 0 : i64} : () -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %11 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %12 = "arith.addi"(%1, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %13 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %14 = "aps.read_smem"(%5, %13) : (memref<1xi32>, i32) -> i32
    "aps.store"(%14, %12) : (i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 3 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<8x8xi32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<8x8xi32>, memref<i32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__direct_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__direct_csr2", sym_visibility = "private", type = memref<i32>}> {csr_address = 2051 : i32} : () -> ()
}) : () -> ()

