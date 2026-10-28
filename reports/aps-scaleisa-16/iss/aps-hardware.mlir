#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (5)>
#map2 = affine_map<() -> (16)>
#map3 = affine_map<() -> (3)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_27__direct_buffer_8", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_27__direct_buffer_8"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_27__direct_buffer_7", type = memref<256xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_27__direct_buffer_7"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_27__direct_buffer", type = memref<256xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_27__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_24__direct_buffer_6", type = memref<256xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_24__direct_buffer_6"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_24__direct_buffer", type = memref<256xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_24__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_24__direct_local", type = memref<i32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_24__direct_local"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_buffer_5", type = memref<256xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_buffer_5"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_buffer", type = memref<768xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_local_4", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_local_4"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_local_3", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_local_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_local_2", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_local_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_local_1", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_local_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_local_0", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_local_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_10__direct_local", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_10__direct_local"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_10__direct"}> ({
  ^bb0(%arg11: i5, %arg12: i5, %arg13: i5):
    %126 = "aps.read_irf"(%arg11) : (i5) -> i32
    %127 = "aps.read_irf"(%arg12) : (i5) -> i32
    %128 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_buffer}> : () -> memref<768xf32>
    %129 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %130 = "arith.constant"() <{value = 768 : i32}> : () -> i32
    "aps.copy"(%126, %128, %129, %130) <{direction = 0 : i32}> : (i32, memref<768xf32>, i32, i32) -> ()
    %131 = "arith.index_castui"(%127) : (i32) -> index
    %132 = "aps.read_csr"() <{global_name = @au_generated_candidate_10__direct_csr0}> : () -> i32
    %133 = "aps.read_csr"() <{global_name = @au_generated_candidate_10__direct_csr1}> : () -> i32
    %134 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_buffer_5}> : () -> memref<256xf32>
    %135 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %136 = "arith.constant"() <{value = 256 : i32}> : () -> i32
    "aps.copy"(%133, %134, %135, %136) <{direction = 0 : i32}> : (i32, memref<256xf32>, i32, i32) -> ()
    %137 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %138 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %139 = "arith.constant"() <{value = 5.000000e-01 : f32}> : () -> f32
    %140 = "arith.constant"() <{value = 2 : index}> : () -> index
    %141 = "arith.constant"() <{value = 1 : index}> : () -> index
    %142 = "arith.constant"() <{value = 0 : index}> : () -> index
    %143 = "arith.constant"() <{value = true}> : () -> i1
    %144 = "arith.constant"() <{value = 15 : i32}> : () -> i32
    %145 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %146 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %147 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg14: index):
      %148 = "arith.index_cast"(%arg14) : (index) -> i32
      %149 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_local}> : () -> memref<f32>
      "aps.globalstore"(%147) <{global_name = @au_generated_candidate_10__direct_local}> : (f32) -> ()
      %150 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_local_0}> : () -> memref<f32>
      "aps.globalstore"(%147) <{global_name = @au_generated_candidate_10__direct_local_0}> : (f32) -> ()
      %151 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_local_1}> : () -> memref<f32>
      "aps.globalstore"(%147) <{global_name = @au_generated_candidate_10__direct_local_1}> : (f32) -> ()
      %152 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_local_2}> : () -> memref<f32>
      "aps.globalstore"(%147) <{global_name = @au_generated_candidate_10__direct_local_2}> : (f32) -> ()
      %153 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_local_3}> : () -> memref<f32>
      "aps.globalstore"(%147) <{global_name = @au_generated_candidate_10__direct_local_3}> : (f32) -> ()
      %154 = "memref.get_global"() <{name = @au_generated_candidate_10__direct_local_4}> : () -> memref<f32>
      "aps.globalstore"(%147) <{global_name = @au_generated_candidate_10__direct_local_4}> : (f32) -> ()
      %155 = "arith.index_cast"(%131) : (index) -> i32
      %156 = "arith.index_cast"(%arg14) : (index) -> i32
      %157 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %158 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %159 = "arith.constant"() <{value = 48 : i32}> : () -> i32
      %160 = "arith.muli"(%155, %159) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %161 = "arith.addi"(%158, %160) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %162 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %163 = "arith.muli"(%156, %162) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %164 = "arith.addi"(%161, %163) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %165 = "arith.addi"(%164, %145) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %166 = "arith.addi"(%157, %165) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %167 = "aps.read_smem"(%128, %166) : (memref<768xf32>, i32) -> f32
      %168 = "arith.index_cast"(%131) : (index) -> i32
      %169 = "arith.index_cast"(%arg14) : (index) -> i32
      %170 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %171 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %172 = "arith.constant"() <{value = 48 : i32}> : () -> i32
      %173 = "arith.muli"(%168, %172) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %174 = "arith.addi"(%171, %173) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %175 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %176 = "arith.muli"(%169, %175) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %177 = "arith.addi"(%174, %176) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %178 = "arith.addi"(%177, %138) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %179 = "arith.addi"(%170, %178) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %180 = "aps.read_smem"(%128, %179) : (memref<768xf32>, i32) -> f32
      %181 = "arith.index_cast"(%131) : (index) -> i32
      %182 = "arith.index_cast"(%arg14) : (index) -> i32
      %183 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %184 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %185 = "arith.constant"() <{value = 48 : i32}> : () -> i32
      %186 = "arith.muli"(%181, %185) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %187 = "arith.addi"(%184, %186) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %188 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %189 = "arith.muli"(%182, %188) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %190 = "arith.addi"(%187, %189) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %191 = "arith.addi"(%190, %137) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %192 = "arith.addi"(%183, %191) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %193 = "aps.read_smem"(%128, %192) : (memref<768xf32>, i32) -> f32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg15: index):
        %224 = "arith.index_cast"(%arg15) : (index) -> i32
        %225 = "arith.addi"(%224, %146) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg16: index):
          %226 = "arith.index_cast"(%arg16) : (index) -> i32
          %227 = "arith.addi"(%226, %146) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %228 = "arith.addi"(%132, %225) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %229 = "arith.cmpi"(%228, %145) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %230 = "arith.cmpi"(%228, %144) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %231 = "arith.andi"(%229, %230) : (i1, i1) -> i1
          %232 = "arith.xori"(%231, %143) : (i1, i1) -> i1
          %233 = "arith.andi"(%232, %229) : (i1, i1) -> i1
          %234 = "arith.select"(%231, %144, %145) : (i1, i32, i32) -> i32
          %235 = "arith.select"(%233, %228, %234) : (i1, i32, i32) -> i32
          %236 = "arith.addi"(%148, %227) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %237 = "arith.cmpi"(%236, %145) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %238 = "arith.cmpi"(%236, %144) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %239 = "arith.andi"(%237, %238) : (i1, i1) -> i1
          %240 = "arith.subi"(%143, %239) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
          %241 = "arith.andi"(%240, %237) : (i1, i1) -> i1
          %242 = "arith.select"(%239, %144, %145) : (i1, i32, i32) -> i32
          %243 = "arith.select"(%241, %236, %242) : (i1, i32, i32) -> i32
          %244 = "arith.index_cast"(%235) : (i32) -> index
          %245 = "arith.index_cast"(%243) : (i32) -> index
          %246 = "arith.index_cast"(%244) : (index) -> i32
          %247 = "arith.index_cast"(%245) : (index) -> i32
          %248 = "arith.index_cast"(%142) : (index) -> i32
          %249 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %250 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %251 = "arith.constant"() <{value = 48 : i32}> : () -> i32
          %252 = "arith.muli"(%246, %251) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %253 = "arith.addi"(%250, %252) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %254 = "arith.constant"() <{value = 3 : i32}> : () -> i32
          %255 = "arith.muli"(%247, %254) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %256 = "arith.addi"(%253, %255) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %257 = "arith.addi"(%256, %248) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %258 = "arith.addi"(%249, %257) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %259 = "aps.read_smem"(%128, %258) : (memref<768xf32>, i32) -> f32
          %260 = "arith.subf"(%259, %167) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %261 = "arith.index_cast"(%244) : (index) -> i32
          %262 = "arith.index_cast"(%245) : (index) -> i32
          %263 = "arith.index_cast"(%141) : (index) -> i32
          %264 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %265 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %266 = "arith.constant"() <{value = 48 : i32}> : () -> i32
          %267 = "arith.muli"(%261, %266) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %268 = "arith.addi"(%265, %267) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %269 = "arith.constant"() <{value = 3 : i32}> : () -> i32
          %270 = "arith.muli"(%262, %269) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %271 = "arith.addi"(%268, %270) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %272 = "arith.addi"(%271, %263) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %273 = "arith.addi"(%264, %272) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %274 = "aps.read_smem"(%128, %273) : (memref<768xf32>, i32) -> f32
          %275 = "arith.subf"(%274, %180) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %276 = "arith.index_cast"(%244) : (index) -> i32
          %277 = "arith.index_cast"(%245) : (index) -> i32
          %278 = "arith.index_cast"(%140) : (index) -> i32
          %279 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %280 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %281 = "arith.constant"() <{value = 48 : i32}> : () -> i32
          %282 = "arith.muli"(%276, %281) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %283 = "arith.addi"(%280, %282) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %284 = "arith.constant"() <{value = 3 : i32}> : () -> i32
          %285 = "arith.muli"(%277, %284) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %286 = "arith.addi"(%283, %285) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %287 = "arith.addi"(%286, %278) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %288 = "arith.addi"(%279, %287) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %289 = "aps.read_smem"(%128, %288) : (memref<768xf32>, i32) -> f32
          %290 = "arith.subf"(%289, %193) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %291 = "arith.mulf"(%260, %260) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %292 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local}> : () -> f32
          %293 = "arith.addf"(%292, %291) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "aps.globalstore"(%293) <{global_name = @au_generated_candidate_10__direct_local}> : (f32) -> ()
          %294 = "arith.mulf"(%275, %275) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %295 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_0}> : () -> f32
          %296 = "arith.addf"(%295, %294) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "aps.globalstore"(%296) <{global_name = @au_generated_candidate_10__direct_local_0}> : (f32) -> ()
          %297 = "arith.mulf"(%290, %290) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %298 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_1}> : () -> f32
          %299 = "arith.addf"(%298, %297) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "aps.globalstore"(%299) <{global_name = @au_generated_candidate_10__direct_local_1}> : (f32) -> ()
          %300 = "arith.mulf"(%260, %275) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %301 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_2}> : () -> f32
          %302 = "arith.addf"(%301, %300) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "aps.globalstore"(%302) <{global_name = @au_generated_candidate_10__direct_local_2}> : (f32) -> ()
          %303 = "arith.mulf"(%260, %290) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %304 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_3}> : () -> f32
          %305 = "arith.addf"(%304, %303) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "aps.globalstore"(%305) <{global_name = @au_generated_candidate_10__direct_local_3}> : (f32) -> ()
          %306 = "arith.mulf"(%275, %290) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %307 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_4}> : () -> f32
          %308 = "arith.addf"(%307, %306) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "aps.globalstore"(%308) <{global_name = @au_generated_candidate_10__direct_local_4}> : (f32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %194 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local}> : () -> f32
      %195 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_0}> : () -> f32
      %196 = "arith.addf"(%194, %195) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %197 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_1}> : () -> f32
      %198 = "arith.addf"(%196, %197) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %199 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_2}> : () -> f32
      %200 = "arith.cmpf"(%199, %147) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %201 = "scf.if"(%200) ({
        %223 = "arith.negf"(%199) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%223) : (f32) -> ()
      }, {
        "scf.yield"(%199) : (f32) -> ()
      }) : (i1) -> f32
      %202 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_3}> : () -> f32
      %203 = "arith.cmpf"(%202, %147) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %204 = "scf.if"(%203) ({
        %222 = "arith.negf"(%202) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%222) : (f32) -> ()
      }, {
        "scf.yield"(%202) : (f32) -> ()
      }) : (i1) -> f32
      %205 = "arith.addf"(%201, %204) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %206 = "aps.globalload"() <{global_name = @au_generated_candidate_10__direct_local_4}> : () -> f32
      %207 = "arith.cmpf"(%206, %147) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %208 = "scf.if"(%207) ({
        %221 = "arith.negf"(%206) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%221) : (f32) -> ()
      }, {
        "scf.yield"(%206) : (f32) -> ()
      }) : (i1) -> f32
      %209 = "arith.addf"(%205, %208) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %210 = "arith.mulf"(%209, %139) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %211 = "arith.subf"(%198, %210) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %212 = "arith.index_cast"(%131) : (index) -> i32
      %213 = "arith.index_cast"(%arg14) : (index) -> i32
      %214 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %215 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %216 = "arith.constant"() <{value = 16 : i32}> : () -> i32
      %217 = "arith.muli"(%212, %216) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %218 = "arith.addi"(%215, %217) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %219 = "arith.addi"(%218, %213) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %220 = "arith.addi"(%214, %219) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%211, %134, %220) : (f32, memref<256xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%133, %134, %135, %136) <{direction = 1 : i32}> : (i32, memref<256xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<16x16x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr"}, {address = 2050 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<16x16xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<16x16x3xf32>, index, i32, memref<16x16xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_24__direct"}> ({
  ^bb0(%arg5: i5, %arg6: i5, %arg7: i5):
    %50 = "aps.read_irf"(%arg5) : (i5) -> i32
    %51 = "aps.read_irf"(%arg6) : (i5) -> i32
    %52 = "memref.get_global"() <{name = @au_generated_candidate_24__direct_buffer}> : () -> memref<256xf32>
    %53 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %54 = "arith.constant"() <{value = 256 : i32}> : () -> i32
    "aps.copy"(%50, %52, %53, %54) <{direction = 0 : i32}> : (i32, memref<256xf32>, i32, i32) -> ()
    %55 = "arith.index_castui"(%51) : (i32) -> index
    %56 = "aps.read_csr"() <{global_name = @au_generated_candidate_24__direct_csr0}> : () -> i32
    %57 = "aps.read_csr"() <{global_name = @au_generated_candidate_24__direct_csr1}> : () -> i32
    %58 = "memref.get_global"() <{name = @au_generated_candidate_24__direct_buffer_6}> : () -> memref<256xi32>
    %59 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %60 = "arith.constant"() <{value = 256 : i32}> : () -> i32
    "aps.copy"(%57, %58, %59, %60) <{direction = 0 : i32}> : (i32, memref<256xi32>, i32, i32) -> ()
    %61 = "arith.constant"() <{value = true}> : () -> i1
    %62 = "arith.constant"() <{value = 15 : i32}> : () -> i32
    %63 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %64 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %65 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg8: index):
      %66 = "arith.index_cast"(%arg8) : (index) -> i32
      %67 = "arith.index_cast"(%55) : (index) -> i32
      %68 = "arith.index_cast"(%arg8) : (index) -> i32
      %69 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %70 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %71 = "arith.constant"() <{value = 16 : i32}> : () -> i32
      %72 = "arith.muli"(%67, %71) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %73 = "arith.addi"(%70, %72) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %74 = "arith.addi"(%73, %68) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %75 = "arith.addi"(%69, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %76 = "aps.read_smem"(%52, %75) : (memref<256xf32>, i32) -> f32
      %77 = "arith.cmpf"(%76, %65) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
      %78 = "arith.extui"(%77) : (i1) -> i32
      %79 = "memref.get_global"() <{name = @au_generated_candidate_24__direct_local}> : () -> memref<i32>
      "aps.globalstore"(%78) <{global_name = @au_generated_candidate_24__direct_local}> : (i32) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg9: index):
        %90 = "arith.index_cast"(%arg9) : (index) -> i32
        %91 = "arith.addi"(%90, %64) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg10: index):
          %92 = "arith.index_cast"(%arg10) : (index) -> i32
          %93 = "arith.addi"(%92, %64) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %94 = "arith.addi"(%56, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %95 = "arith.cmpi"(%94, %63) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %96 = "arith.cmpi"(%94, %62) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %97 = "arith.andi"(%95, %96) : (i1, i1) -> i1
          %98 = "arith.xori"(%97, %61) : (i1, i1) -> i1
          %99 = "arith.andi"(%98, %95) : (i1, i1) -> i1
          %100 = "arith.select"(%97, %62, %63) : (i1, i32, i32) -> i32
          %101 = "arith.select"(%99, %94, %100) : (i1, i32, i32) -> i32
          %102 = "arith.addi"(%66, %93) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %103 = "arith.cmpi"(%102, %63) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %104 = "arith.cmpi"(%102, %62) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %105 = "arith.andi"(%103, %104) : (i1, i1) -> i1
          %106 = "arith.xori"(%105, %61) : (i1, i1) -> i1
          %107 = "arith.andi"(%106, %103) : (i1, i1) -> i1
          %108 = "arith.select"(%105, %62, %63) : (i1, i32, i32) -> i32
          %109 = "arith.select"(%107, %102, %108) : (i1, i32, i32) -> i32
          %110 = "arith.cmpi"(%91, %63) <{predicate = 1 : i64}> : (i32, i32) -> i1
          %111 = "scf.if"(%110) ({
            "scf.yield"(%61) : (i1) -> ()
          }, {
            %125 = "arith.cmpi"(%93, %63) <{predicate = 1 : i64}> : (i32, i32) -> i1
            "scf.yield"(%125) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%111) ({
            %112 = "arith.index_cast"(%101) : (i32) -> index
            %113 = "arith.index_cast"(%109) : (i32) -> index
            %114 = "arith.index_cast"(%112) : (index) -> i32
            %115 = "arith.index_cast"(%113) : (index) -> i32
            %116 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %117 = "arith.constant"() <{value = 0 : i32}> : () -> i32
            %118 = "arith.constant"() <{value = 16 : i32}> : () -> i32
            %119 = "arith.muli"(%114, %118) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %120 = "arith.addi"(%117, %119) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %121 = "arith.addi"(%120, %115) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %122 = "arith.addi"(%116, %121) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %123 = "aps.read_smem"(%52, %122) : (memref<256xf32>, i32) -> f32
            %124 = "arith.cmpf"(%123, %76) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
            "scf.if"(%124) ({
              "aps.globalstore"(%63) <{global_name = @au_generated_candidate_24__direct_local}> : (i32) -> ()
              "scf.yield"() : () -> ()
            }, {
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %80 = "aps.globalload"() <{global_name = @au_generated_candidate_24__direct_local}> : () -> i32
      %81 = "arith.index_cast"(%55) : (index) -> i32
      %82 = "arith.index_cast"(%arg8) : (index) -> i32
      %83 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %84 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %85 = "arith.constant"() <{value = 16 : i32}> : () -> i32
      %86 = "arith.muli"(%81, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %87 = "arith.addi"(%84, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %88 = "arith.addi"(%87, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %89 = "arith.addi"(%83, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%80, %58, %89) : (i32, memref<256xi32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%57, %58, %59, %60) <{direction = 1 : i32}> : (i32, memref<256xi32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<16x16xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2"}, {address = 2051 : i32, arg_index = 2 : i32, kind = "csr"}, {address = 2052 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<16x16xi32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<16x16xf32>, index, i32, memref<16x16xi32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_27__direct"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_27__direct_buffer}> : () -> memref<256xi32>
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 256 : i32}> : () -> i32
    "aps.copy"(%0, %2, %3, %4) <{direction = 0 : i32}> : (i32, memref<256xi32>, i32, i32) -> ()
    %5 = "memref.get_global"() <{name = @au_generated_candidate_27__direct_buffer_7}> : () -> memref<256xf32>
    %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 256 : i32}> : () -> i32
    "aps.copy"(%1, %5, %6, %7) <{direction = 0 : i32}> : (i32, memref<256xf32>, i32, i32) -> ()
    %8 = "aps.read_csr"() <{global_name = @au_generated_candidate_27__direct_csr0}> : () -> i32
    %9 = "memref.get_global"() <{name = @au_generated_candidate_27__direct_buffer_8}> : () -> memref<1xi32>
    %10 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%8, %9, %10, %11) <{direction = 0 : i32}> : (i32, memref<1xi32>, i32, i32) -> ()
    %12 = "arith.constant"() <{value = 3.200000e+01 : f32}> : () -> f32
    %13 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg3: index):
      %14 = "arith.index_cast"(%arg3) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg4: index):
        %15 = "arith.index_cast"(%arg4) : (index) -> i32
        %16 = "arith.index_cast"(%arg3) : (index) -> i32
        %17 = "arith.index_cast"(%arg4) : (index) -> i32
        %18 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %19 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %20 = "arith.constant"() <{value = 16 : i32}> : () -> i32
        %21 = "arith.muli"(%16, %20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %22 = "arith.addi"(%19, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %23 = "arith.addi"(%22, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %24 = "arith.addi"(%18, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %25 = "aps.read_smem"(%2, %24) : (memref<256xi32>, i32) -> i32
        %26 = "arith.cmpi"(%25, %13) <{predicate = 1 : i64}> : (i32, i32) -> i1
        %27 = "scf.if"(%26) ({
          %36 = "arith.index_cast"(%arg3) : (index) -> i32
          %37 = "arith.index_cast"(%arg4) : (index) -> i32
          %38 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %39 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %40 = "arith.constant"() <{value = 16 : i32}> : () -> i32
          %41 = "arith.muli"(%36, %40) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %42 = "arith.addi"(%39, %41) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %43 = "arith.addi"(%42, %37) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %44 = "arith.addi"(%38, %43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %45 = "aps.read_smem"(%5, %44) : (memref<256xf32>, i32) -> f32
          %46 = "arith.mulf"(%45, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %47 = "arith.fptosi"(%46) : (f32) -> i32
          %48 = "arith.addi"(%47, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %49 = "arith.addi"(%48, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "scf.yield"(%49) : (i32) -> ()
        }, {
          "scf.yield"(%13) : (i32) -> ()
        }) : (i1) -> i32
        %28 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %29 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %30 = "arith.addi"(%28, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %31 = "aps.read_smem"(%9, %30) : (memref<1xi32>, i32) -> i32
        %32 = "arith.addi"(%31, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %33 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %34 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %35 = "arith.addi"(%33, %34) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%32, %9, %35) : (i32, memref<1xi32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%8, %9, %10, %11) <{direction = 1 : i32}> : (i32, memref<1xi32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 2 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<16x16xi32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<16x16xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2053 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<16x16xi32>, memref<16x16xf32>, memref<i32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_10__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_10__direct_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_24__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2051 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_24__direct_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2052 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_27__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2053 : i32} : () -> ()
}) : () -> ()

