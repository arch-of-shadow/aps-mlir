#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (8)>
#map2 = affine_map<() -> (4)>
#map3 = affine_map<() -> (32)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_25__l0u_buffer", type = memref<12xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_25__l0u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_24__l0u_buffer", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_24__l0u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_2__direct_buffer_6", type = memref<8xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_2__direct_buffer_6"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_2__direct_buffer_5", type = memref<8xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_2__direct_buffer_5"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_2__direct_buffer_4", type = memref<12xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_2__direct_buffer_4"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_2__direct_buffer_3", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_2__direct_buffer_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_2__direct_buffer_2", type = memref<32xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_2__direct_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_2__direct_buffer", type = memref<32xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_2__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_2__direct_local", type = memref<i32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_2__direct_local"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_11__direct_buffer_1", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_11__direct_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_11__direct_buffer_0", type = memref<32xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_11__direct_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_11__direct_buffer", type = memref<32xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_11__direct_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_11__direct"}> ({
  ^bb0(%arg17: i5, %arg18: i5, %arg19: i5):
    %317 = "aps.read_irf"(%arg17) : (i5) -> i32
    %318 = "aps.read_irf"(%arg18) : (i5) -> i32
    %319 = "memref.get_global"() <{name = @au_generated_candidate_11__direct_buffer}> : () -> memref<32xi32>
    %320 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %321 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    "aps.copy"(%317, %319, %320, %321) <{direction = 0 : i32}> : (i32, memref<32xi32>, i32, i32) -> ()
    %322 = "memref.get_global"() <{name = @au_generated_candidate_11__direct_buffer_0}> : () -> memref<32xf32>
    %323 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %324 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    "aps.copy"(%318, %322, %323, %324) <{direction = 0 : i32}> : (i32, memref<32xf32>, i32, i32) -> ()
    %325 = "aps.read_csr"() <{global_name = @au_generated_candidate_11__direct_csr0}> : () -> i32
    %326 = "memref.get_global"() <{name = @au_generated_candidate_11__direct_buffer_1}> : () -> memref<1xi32>
    %327 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %328 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%325, %326, %327, %328) <{direction = 0 : i32}> : (i32, memref<1xi32>, i32, i32) -> ()
    %329 = "arith.constant"() <{value = 2.560000e+02 : f32}> : () -> f32
    %330 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg20: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg21: index):
        %331 = "arith.index_cast"(%arg20) : (index) -> i32
        %332 = "arith.index_cast"(%arg21) : (index) -> i32
        %333 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %334 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %335 = "arith.constant"() <{value = 8 : i32}> : () -> i32
        %336 = "arith.muli"(%331, %335) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %337 = "arith.addi"(%334, %336) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %338 = "arith.addi"(%337, %332) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %339 = "arith.addi"(%333, %338) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %340 = "aps.read_smem"(%319, %339) : (memref<32xi32>, i32) -> i32
        %341 = "arith.cmpi"(%340, %330) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %342 = "scf.if"(%341) ({
          %351 = "arith.index_cast"(%arg20) : (index) -> i32
          %352 = "arith.index_cast"(%arg21) : (index) -> i32
          %353 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %354 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %355 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %356 = "arith.muli"(%351, %355) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %357 = "arith.addi"(%354, %356) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %358 = "arith.addi"(%357, %352) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %359 = "arith.addi"(%353, %358) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %360 = "aps.read_smem"(%319, %359) : (memref<32xi32>, i32) -> i32
          %361 = "arith.index_cast"(%arg20) : (index) -> i32
          %362 = "arith.index_cast"(%arg21) : (index) -> i32
          %363 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %364 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %365 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %366 = "arith.muli"(%361, %365) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %367 = "arith.addi"(%364, %366) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %368 = "arith.addi"(%367, %362) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %369 = "arith.addi"(%363, %368) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %370 = "aps.read_smem"(%322, %369) : (memref<32xf32>, i32) -> f32
          %371 = "arith.mulf"(%370, %329) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %372 = "arith.fptosi"(%371) : (f32) -> i32
          %373 = "arith.addi"(%360, %372) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "scf.yield"(%373) : (i32) -> ()
        }, {
          "scf.yield"(%330) : (i32) -> ()
        }) : (i1) -> i32
        %343 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %344 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %345 = "arith.addi"(%343, %344) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %346 = "aps.read_smem"(%326, %345) : (memref<1xi32>, i32) -> i32
        %347 = "arith.addi"(%346, %342) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %348 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %349 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %350 = "arith.addi"(%348, %349) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%347, %326, %350) : (i32, memref<1xi32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%325, %326, %327, %328) <{direction = 1 : i32}> : (i32, memref<1xi32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<4x8xi32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<4x8xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<4x8xi32>, memref<4x8xf32>, memref<i32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_2__direct"}> ({
  ^bb0(%arg8: i5, %arg9: i5, %arg10: i5):
    %118 = "aps.read_irf"(%arg8) : (i5) -> i32
    %119 = "aps.read_irf"(%arg9) : (i5) -> i32
    %120 = "memref.get_global"() <{name = @au_generated_candidate_2__direct_buffer}> : () -> memref<32xf32>
    %121 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %122 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    "aps.copy"(%118, %120, %121, %122) <{direction = 0 : i32}> : (i32, memref<32xf32>, i32, i32) -> ()
    %123 = "memref.get_global"() <{name = @au_generated_candidate_2__direct_buffer_2}> : () -> memref<32xi32>
    %124 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %125 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    "aps.copy"(%119, %123, %124, %125) <{direction = 0 : i32}> : (i32, memref<32xi32>, i32, i32) -> ()
    %126 = "aps.read_csr"() <{global_name = @au_generated_candidate_2__direct_csr0}> : () -> i32
    %127 = "memref.get_global"() <{name = @au_generated_candidate_2__direct_buffer_3}> : () -> memref<96xf32>
    %128 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %129 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    "aps.copy"(%126, %127, %128, %129) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %130 = "aps.read_csr"() <{global_name = @au_generated_candidate_2__direct_csr1}> : () -> i32
    %131 = "memref.get_global"() <{name = @au_generated_candidate_2__direct_buffer_4}> : () -> memref<12xf32>
    %132 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %133 = "arith.constant"() <{value = 12 : i32}> : () -> i32
    "aps.copy"(%130, %131, %132, %133) <{direction = 0 : i32}> : (i32, memref<12xf32>, i32, i32) -> ()
    %134 = "aps.read_csr"() <{global_name = @au_generated_candidate_2__direct_csr2}> : () -> i32
    %135 = "memref.get_global"() <{name = @au_generated_candidate_2__direct_buffer_5}> : () -> memref<8xf32>
    %136 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %137 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    "aps.copy"(%134, %135, %136, %137) <{direction = 0 : i32}> : (i32, memref<8xf32>, i32, i32) -> ()
    %138 = "aps.read_csr"() <{global_name = @au_generated_candidate_2__direct_csr3}> : () -> i32
    %139 = "memref.get_global"() <{name = @au_generated_candidate_2__direct_buffer_6}> : () -> memref<8xi32>
    %140 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %141 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    "aps.copy"(%138, %139, %140, %141) <{direction = 0 : i32}> : (i32, memref<8xi32>, i32, i32) -> ()
    %142 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %143 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %144 = "arith.constant"() <{value = false}> : () -> i1
    %145 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %146 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %147 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg11: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 8 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg15: index):
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg16: index):
          %299 = "arith.index_cast"(%arg11) : (index) -> i32
          %300 = "arith.index_cast"(%arg16) : (index) -> i32
          %301 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %302 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %303 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %304 = "arith.muli"(%299, %303) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %305 = "arith.addi"(%302, %304) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %306 = "arith.addi"(%305, %300) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %307 = "arith.addi"(%301, %306) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%147, %120, %307) : (f32, memref<32xf32>, i32) -> ()
          %308 = "arith.index_cast"(%arg11) : (index) -> i32
          %309 = "arith.index_cast"(%arg16) : (index) -> i32
          %310 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %311 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %312 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %313 = "arith.muli"(%308, %312) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %314 = "arith.addi"(%311, %313) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %315 = "arith.addi"(%314, %309) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %316 = "arith.addi"(%310, %315) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%146, %123, %316) : (i32, memref<32xi32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg12: index):
        %148 = "arith.index_cast"(%arg12) : (index) -> i32
        %149 = "arith.index_cast"(%arg12) : (index) -> i32
        %150 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %151 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %152 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %153 = "arith.muli"(%149, %152) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %154 = "arith.addi"(%151, %153) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %155 = "arith.addi"(%154, %145) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %156 = "arith.addi"(%150, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %157 = "aps.read_smem"(%127, %156) : (memref<96xf32>, i32) -> f32
        %158 = "arith.index_cast"(%arg11) : (index) -> i32
        %159 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %160 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %161 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %162 = "arith.muli"(%158, %161) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %163 = "arith.addi"(%160, %162) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %164 = "arith.addi"(%163, %145) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %165 = "arith.addi"(%159, %164) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %166 = "aps.read_smem"(%131, %165) : (memref<12xf32>, i32) -> f32
        %167 = "arith.subf"(%157, %166) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %168 = "arith.index_cast"(%arg12) : (index) -> i32
        %169 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %170 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %171 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %172 = "arith.muli"(%168, %171) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %173 = "arith.addi"(%170, %172) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %174 = "arith.addi"(%173, %143) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %175 = "arith.addi"(%169, %174) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %176 = "aps.read_smem"(%127, %175) : (memref<96xf32>, i32) -> f32
        %177 = "arith.index_cast"(%arg11) : (index) -> i32
        %178 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %179 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %180 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %181 = "arith.muli"(%177, %180) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %182 = "arith.addi"(%179, %181) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %183 = "arith.addi"(%182, %143) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %184 = "arith.addi"(%178, %183) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %185 = "aps.read_smem"(%131, %184) : (memref<12xf32>, i32) -> f32
        %186 = "arith.subf"(%176, %185) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %187 = "arith.index_cast"(%arg12) : (index) -> i32
        %188 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %189 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %190 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %191 = "arith.muli"(%187, %190) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %192 = "arith.addi"(%189, %191) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %193 = "arith.addi"(%192, %142) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %194 = "arith.addi"(%188, %193) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %195 = "aps.read_smem"(%127, %194) : (memref<96xf32>, i32) -> f32
        %196 = "arith.index_cast"(%arg11) : (index) -> i32
        %197 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %198 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %199 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %200 = "arith.muli"(%196, %199) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %201 = "arith.addi"(%198, %200) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %202 = "arith.addi"(%201, %142) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %203 = "arith.addi"(%197, %202) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %204 = "aps.read_smem"(%131, %203) : (memref<12xf32>, i32) -> f32
        %205 = "arith.subf"(%195, %204) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %206 = "arith.mulf"(%167, %167) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %207 = "arith.mulf"(%186, %186) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %208 = "arith.addf"(%206, %207) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %209 = "arith.mulf"(%205, %205) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %210 = "arith.addf"(%208, %209) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %211 = "memref.get_global"() <{name = @au_generated_candidate_2__direct_local}> : () -> memref<i32>
        "aps.globalstore"(%145) <{global_name = @au_generated_candidate_2__direct_local}> : (i32) -> ()
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg14: index):
          %242 = "arith.index_cast"(%arg14) : (index) -> i32
          %243 = "aps.globalload"() <{global_name = @au_generated_candidate_2__direct_local}> : () -> i32
          %244 = "arith.cmpi"(%243, %145) <{predicate = 0 : i64}> : (i32, i32) -> i1
          %245 = "scf.if"(%244) ({
            %288 = "arith.index_cast"(%arg11) : (index) -> i32
            %289 = "arith.index_cast"(%arg14) : (index) -> i32
            %290 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %291 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %292 = "arith.constant"() <{value = 8 : i32}> : () -> i32
            %293 = "arith.muli"(%288, %292) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %294 = "arith.addi"(%291, %293) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %295 = "arith.addi"(%294, %289) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %296 = "arith.addi"(%290, %295) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %297 = "aps.read_smem"(%120, %296) : (memref<32xf32>, i32) -> f32
            %298 = "arith.cmpf"(%210, %297) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
            "scf.yield"(%298) : (i1) -> ()
          }, {
            "scf.yield"(%144) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%245) ({
            %278 = "arith.index_cast"(%arg14) : (index) -> i32
            %279 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %280 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %281 = "arith.addi"(%280, %278) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %282 = "arith.addi"(%279, %281) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            "aps.write_smem"(%210, %135, %282) : (f32, memref<8xf32>, i32) -> ()
            %283 = "arith.index_cast"(%arg14) : (index) -> i32
            %284 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %285 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %286 = "arith.addi"(%285, %283) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %287 = "arith.addi"(%284, %286) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            "aps.write_smem"(%148, %139, %287) : (i32, memref<8xi32>, i32) -> ()
            "aps.globalstore"(%143) <{global_name = @au_generated_candidate_2__direct_local}> : (i32) -> ()
            "scf.yield"() : () -> ()
          }, {
            %246 = "arith.subi"(%242, %243) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %247 = "arith.index_cast"(%246) : (i32) -> index
            %248 = "arith.index_cast"(%arg11) : (index) -> i32
            %249 = "arith.index_cast"(%247) : (index) -> i32
            %250 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %251 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %252 = "arith.constant"() <{value = 8 : i32}> : () -> i32
            %253 = "arith.muli"(%248, %252) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %254 = "arith.addi"(%251, %253) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %255 = "arith.addi"(%254, %249) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %256 = "arith.addi"(%250, %255) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %257 = "aps.read_smem"(%120, %256) : (memref<32xf32>, i32) -> f32
            %258 = "arith.index_cast"(%arg14) : (index) -> i32
            %259 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %260 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %261 = "arith.addi"(%260, %258) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %262 = "arith.addi"(%259, %261) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            "aps.write_smem"(%257, %135, %262) : (f32, memref<8xf32>, i32) -> ()
            %263 = "arith.index_cast"(%arg11) : (index) -> i32
            %264 = "arith.index_cast"(%247) : (index) -> i32
            %265 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %266 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %267 = "arith.constant"() <{value = 8 : i32}> : () -> i32
            %268 = "arith.muli"(%263, %267) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %269 = "arith.addi"(%266, %268) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %270 = "arith.addi"(%269, %264) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %271 = "arith.addi"(%265, %270) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %272 = "aps.read_smem"(%123, %271) : (memref<32xi32>, i32) -> i32
            %273 = "arith.index_cast"(%arg14) : (index) -> i32
            %274 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %275 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %276 = "arith.addi"(%275, %273) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %277 = "arith.addi"(%274, %276) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            "aps.write_smem"(%272, %139, %277) : (i32, memref<8xi32>, i32) -> ()
            "scf.yield"() : () -> ()
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg13: index):
          %212 = "arith.index_cast"(%arg13) : (index) -> i32
          %213 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %214 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %215 = "arith.addi"(%214, %212) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %216 = "arith.addi"(%213, %215) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %217 = "aps.read_smem"(%135, %216) : (memref<8xf32>, i32) -> f32
          %218 = "arith.index_cast"(%arg11) : (index) -> i32
          %219 = "arith.index_cast"(%arg13) : (index) -> i32
          %220 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %221 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %222 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %223 = "arith.muli"(%218, %222) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %224 = "arith.addi"(%221, %223) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %225 = "arith.addi"(%224, %219) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %226 = "arith.addi"(%220, %225) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%217, %120, %226) : (f32, memref<32xf32>, i32) -> ()
          %227 = "arith.index_cast"(%arg13) : (index) -> i32
          %228 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %229 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %230 = "arith.addi"(%229, %227) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %231 = "arith.addi"(%228, %230) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %232 = "aps.read_smem"(%139, %231) : (memref<8xi32>, i32) -> i32
          %233 = "arith.index_cast"(%arg11) : (index) -> i32
          %234 = "arith.index_cast"(%arg13) : (index) -> i32
          %235 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %236 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %237 = "arith.constant"() <{value = 8 : i32}> : () -> i32
          %238 = "arith.muli"(%233, %237) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %239 = "arith.addi"(%236, %238) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %240 = "arith.addi"(%239, %234) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %241 = "arith.addi"(%235, %240) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%232, %123, %241) : (i32, memref<32xi32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%118, %120, %121, %122) <{direction = 1 : i32}> : (i32, memref<32xf32>, i32, i32) -> ()
    "aps.copy"(%119, %123, %124, %125) <{direction = 1 : i32}> : (i32, memref<32xi32>, i32, i32) -> ()
    "aps.copy"(%134, %135, %136, %137) <{direction = 1 : i32}> : (i32, memref<8xf32>, i32, i32) -> ()
    "aps.copy"(%138, %139, %140, %141) <{direction = 1 : i32}> : (i32, memref<8xi32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<4x8xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<4x8xi32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2050 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2051 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<4x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2052 : i32, arg_index = 4 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<8xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2053 : i32, arg_index = 5 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<8xi32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<4x8xf32>, memref<4x8xi32>, memref<32x3xf32>, memref<4x3xf32>, memref<8xf32>, memref<8xi32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_24__l0u"}> ({
  ^bb0(%arg4: i5, %arg5: i5, %arg6: i5):
    %59 = "aps.read_irf"(%arg4) : (i5) -> i32
    %60 = "memref.get_global"() <{name = @au_generated_candidate_24__l0u_buffer}> : () -> memref<96xf32>
    %61 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %62 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    "aps.copy"(%59, %60, %61, %62) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %63 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %64 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %65 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %66 = "arith.constant"() <{value = 2.700000e-02 : f32}> : () -> f32
    %67 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %68 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %69 = "arith.constant"() <{value = 19 : i32}> : () -> i32
    %70 = "arith.constant"() <{value = 2.900000e-02 : f32}> : () -> f32
    %71 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %72 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %73 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %74 = "arith.constant"() <{value = 3.100000e-02 : f32}> : () -> f32
    %75 = "arith.constant"() <{value = 101 : i32}> : () -> i32
    %76 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %77 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg7: index):
      %78 = "arith.index_cast"(%arg7) : (index) -> i32
      %79 = "arith.muli"(%78, %77) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %80 = "arith.addi"(%79, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %81 = "arith.remsi"(%80, %75) : (i32, i32) -> i32
      %82 = "arith.sitofp"(%81) : (i32) -> f32
      %83 = "arith.mulf"(%82, %74) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %84 = "arith.index_cast"(%arg7) : (index) -> i32
      %85 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %86 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %87 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %88 = "arith.muli"(%84, %87) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %89 = "arith.addi"(%86, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %90 = "arith.addi"(%89, %65) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %91 = "arith.addi"(%85, %90) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%83, %60, %91) : (f32, memref<96xf32>, i32) -> ()
      %92 = "arith.muli"(%78, %73) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %93 = "arith.addi"(%92, %72) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %94 = "arith.remsi"(%93, %71) : (i32, i32) -> i32
      %95 = "arith.sitofp"(%94) : (i32) -> f32
      %96 = "arith.mulf"(%95, %70) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %97 = "arith.index_cast"(%arg7) : (index) -> i32
      %98 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %99 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %100 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %101 = "arith.muli"(%97, %100) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %102 = "arith.addi"(%99, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %103 = "arith.addi"(%102, %64) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %104 = "arith.addi"(%98, %103) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%96, %60, %104) : (f32, memref<96xf32>, i32) -> ()
      %105 = "arith.muli"(%78, %69) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %106 = "arith.addi"(%105, %68) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %107 = "arith.remsi"(%106, %67) : (i32, i32) -> i32
      %108 = "arith.sitofp"(%107) : (i32) -> f32
      %109 = "arith.mulf"(%108, %66) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %110 = "arith.index_cast"(%arg7) : (index) -> i32
      %111 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %112 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %113 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %114 = "arith.muli"(%110, %113) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %115 = "arith.addi"(%112, %114) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %116 = "arith.addi"(%115, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %117 = "arith.addi"(%111, %116) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%109, %60, %117) : (f32, memref<96xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "aps.copy"(%59, %60, %61, %62) <{direction = 1 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 2 : i32, funct7 = 2 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<32x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_25__l0u"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "memref.get_global"() <{name = @au_generated_candidate_25__l0u_buffer}> : () -> memref<12xf32>
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 12 : i32}> : () -> i32
    "aps.copy"(%0, %1, %2, %3) <{direction = 0 : i32}> : (i32, memref<12xf32>, i32, i32) -> ()
    %4 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 2.700000e-02 : f32}> : () -> f32
    %8 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 2.900000e-02 : f32}> : () -> f32
    %12 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %13 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 29 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 3.100000e-02 : f32}> : () -> f32
    %16 = "arith.constant"() <{value = 101 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = 9 : i32}> : () -> i32
    %18 = "arith.constant"() <{value = 23 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg3: index):
      %19 = "arith.index_cast"(%arg3) : (index) -> i32
      %20 = "arith.muli"(%19, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %21 = "arith.addi"(%20, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %22 = "arith.remsi"(%21, %16) : (i32, i32) -> i32
      %23 = "arith.sitofp"(%22) : (i32) -> f32
      %24 = "arith.mulf"(%23, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %25 = "arith.index_cast"(%arg3) : (index) -> i32
      %26 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %27 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %28 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %29 = "arith.muli"(%25, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %30 = "arith.addi"(%27, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %31 = "arith.addi"(%30, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %32 = "arith.addi"(%26, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%24, %1, %32) : (f32, memref<12xf32>, i32) -> ()
      %33 = "arith.muli"(%19, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %34 = "arith.addi"(%33, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %35 = "arith.remsi"(%34, %12) : (i32, i32) -> i32
      %36 = "arith.sitofp"(%35) : (i32) -> f32
      %37 = "arith.mulf"(%36, %11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %38 = "arith.index_cast"(%arg3) : (index) -> i32
      %39 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %40 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %41 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %42 = "arith.muli"(%38, %41) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %43 = "arith.addi"(%40, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %44 = "arith.addi"(%43, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %45 = "arith.addi"(%39, %44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%37, %1, %45) : (f32, memref<12xf32>, i32) -> ()
      %46 = "arith.muli"(%19, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %47 = "arith.addi"(%46, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %48 = "arith.remsi"(%47, %8) : (i32, i32) -> i32
      %49 = "arith.sitofp"(%48) : (i32) -> f32
      %50 = "arith.mulf"(%49, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %51 = "arith.index_cast"(%arg3) : (index) -> i32
      %52 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %53 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %54 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %55 = "arith.muli"(%51, %54) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %56 = "arith.addi"(%53, %55) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %57 = "arith.addi"(%56, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %58 = "arith.addi"(%52, %57) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%50, %1, %58) : (f32, memref<12xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "aps.copy"(%0, %1, %2, %3) <{direction = 1 : i32}> : (i32, memref<12xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 2 : i32, funct7 = 3 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<4x3xf32>, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<4x3xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_11__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_2__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_2__direct_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2051 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_2__direct_csr2", sym_visibility = "private", type = memref<i32>}> {csr_address = 2052 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_2__direct_csr3", sym_visibility = "private", type = memref<i32>}> {csr_address = 2053 : i32} : () -> ()
}) : () -> ()

