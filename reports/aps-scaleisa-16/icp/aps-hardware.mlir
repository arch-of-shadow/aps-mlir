#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (32)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_8__l0u_buffer_9", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_8__l0u_buffer_9"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_8__l0u_buffer", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_8__l0u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_buffer_8", type = memref<1xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_8"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_buffer_7", type = memref<1xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_7"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_buffer_6", type = memref<1xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_6"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_buffer_5", type = memref<1xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_5"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_buffer_4", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_4"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_buffer_3", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_buffer_2", type = memref<1xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_buffer_1", type = memref<1xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_buffer", type = memref<1xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_local_0", type = memref<i32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_local_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_1__direct_local", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__direct_local"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_1__direct"}> ({
  ^bb0(%arg4: i5, %arg5: i5, %arg6: i5):
    %87 = "aps.read_irf"(%arg4) : (i5) -> i32
    %88 = "aps.read_irf"(%arg5) : (i5) -> i32
    %89 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_buffer}> : () -> memref<1xf32>
    %90 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %91 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%87, %89, %90, %91) <{direction = 0 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    %92 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_buffer_1}> : () -> memref<1xf32>
    %93 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %94 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%88, %92, %93, %94) <{direction = 0 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    %95 = "aps.read_csr"() <{global_name = @au_generated_candidate_1__direct_csr0}> : () -> i32
    %96 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_buffer_2}> : () -> memref<1xf32>
    %97 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %98 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%95, %96, %97, %98) <{direction = 0 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    %99 = "aps.read_csr"() <{global_name = @au_generated_candidate_1__direct_csr1}> : () -> i32
    %100 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_buffer_3}> : () -> memref<96xf32>
    %101 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %102 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    "aps.copy"(%99, %100, %101, %102) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %103 = "aps.read_csr"() <{global_name = @au_generated_candidate_1__direct_csr2}> : () -> i32
    %104 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_buffer_4}> : () -> memref<96xf32>
    %105 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %106 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    "aps.copy"(%103, %104, %105, %106) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %107 = "aps.read_csr"() <{global_name = @au_generated_candidate_1__direct_csr3}> : () -> i32
    %108 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_buffer_5}> : () -> memref<1xf32>
    %109 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %110 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%107, %108, %109, %110) <{direction = 0 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    %111 = "aps.read_csr"() <{global_name = @au_generated_candidate_1__direct_csr4}> : () -> i32
    %112 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_buffer_6}> : () -> memref<1xf32>
    %113 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %114 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%111, %112, %113, %114) <{direction = 0 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    %115 = "aps.read_csr"() <{global_name = @au_generated_candidate_1__direct_csr5}> : () -> i32
    %116 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_buffer_7}> : () -> memref<1xf32>
    %117 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %118 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%115, %116, %117, %118) <{direction = 0 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    %119 = "aps.read_csr"() <{global_name = @au_generated_candidate_1__direct_csr6}> : () -> i32
    %120 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_buffer_8}> : () -> memref<1xf32>
    %121 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %122 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%119, %120, %121, %122) <{direction = 0 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    %123 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %124 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %125 = "arith.constant"() <{value = 2 : index}> : () -> index
    %126 = "arith.constant"() <{value = 1 : index}> : () -> index
    %127 = "arith.constant"() <{value = 0 : index}> : () -> index
    %128 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %129 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg7: index):
      %130 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_local}> : () -> memref<f32>
      "aps.globalstore"(%129) <{global_name = @au_generated_candidate_1__direct_local}> : (f32) -> ()
      %131 = "memref.get_global"() <{name = @au_generated_candidate_1__direct_local_0}> : () -> memref<i32>
      "aps.globalstore"(%128) <{global_name = @au_generated_candidate_1__direct_local_0}> : (i32) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg8: index):
        %242 = "arith.index_cast"(%arg8) : (index) -> i32
        %243 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %244 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %245 = "arith.addi"(%243, %244) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %246 = "aps.read_smem"(%89, %245) : (memref<1xf32>, i32) -> f32
        %247 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %248 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %249 = "arith.addi"(%247, %248) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %250 = "aps.read_smem"(%92, %249) : (memref<1xf32>, i32) -> f32
        %251 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %252 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %253 = "arith.addi"(%251, %252) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %254 = "aps.read_smem"(%96, %253) : (memref<1xf32>, i32) -> f32
        %255 = "arith.index_cast"(%arg7) : (index) -> i32
        %256 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %257 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %258 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %259 = "arith.muli"(%255, %258) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %260 = "arith.addi"(%257, %259) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %261 = "arith.addi"(%260, %128) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %262 = "arith.addi"(%256, %261) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %263 = "aps.read_smem"(%100, %262) : (memref<96xf32>, i32) -> f32
        %264 = "arith.addf"(%263, %246) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %265 = "arith.index_cast"(%arg8) : (index) -> i32
        %266 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %267 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %268 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %269 = "arith.muli"(%265, %268) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %270 = "arith.addi"(%267, %269) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %271 = "arith.addi"(%270, %128) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %272 = "arith.addi"(%266, %271) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %273 = "aps.read_smem"(%104, %272) : (memref<96xf32>, i32) -> f32
        %274 = "arith.subf"(%264, %273) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %275 = "arith.index_cast"(%arg7) : (index) -> i32
        %276 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %277 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %278 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %279 = "arith.muli"(%275, %278) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %280 = "arith.addi"(%277, %279) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %281 = "arith.addi"(%280, %124) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %282 = "arith.addi"(%276, %281) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %283 = "aps.read_smem"(%100, %282) : (memref<96xf32>, i32) -> f32
        %284 = "arith.addf"(%283, %250) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %285 = "arith.index_cast"(%arg8) : (index) -> i32
        %286 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %287 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %288 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %289 = "arith.muli"(%285, %288) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %290 = "arith.addi"(%287, %289) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %291 = "arith.addi"(%290, %124) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %292 = "arith.addi"(%286, %291) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %293 = "aps.read_smem"(%104, %292) : (memref<96xf32>, i32) -> f32
        %294 = "arith.subf"(%284, %293) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %295 = "arith.index_cast"(%arg7) : (index) -> i32
        %296 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %297 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %298 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %299 = "arith.muli"(%295, %298) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %300 = "arith.addi"(%297, %299) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %301 = "arith.addi"(%300, %123) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %302 = "arith.addi"(%296, %301) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %303 = "aps.read_smem"(%100, %302) : (memref<96xf32>, i32) -> f32
        %304 = "arith.addf"(%303, %254) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %305 = "arith.index_cast"(%arg8) : (index) -> i32
        %306 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %307 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %308 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %309 = "arith.muli"(%305, %308) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %310 = "arith.addi"(%307, %309) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %311 = "arith.addi"(%310, %123) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %312 = "arith.addi"(%306, %311) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %313 = "aps.read_smem"(%104, %312) : (memref<96xf32>, i32) -> f32
        %314 = "arith.subf"(%304, %313) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %315 = "arith.mulf"(%274, %274) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %316 = "arith.mulf"(%294, %294) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %317 = "arith.addf"(%315, %316) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %318 = "arith.mulf"(%314, %314) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %319 = "arith.addf"(%317, %318) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %320 = "aps.globalload"() <{global_name = @au_generated_candidate_1__direct_local}> : () -> f32
        %321 = "arith.cmpf"(%319, %320) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        "scf.if"(%321) ({
          "aps.globalstore"(%319) <{global_name = @au_generated_candidate_1__direct_local}> : (f32) -> ()
          "aps.globalstore"(%242) <{global_name = @au_generated_candidate_1__direct_local_0}> : (i32) -> ()
          "scf.yield"() : () -> ()
        }, {
        }) : (i1) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %132 = "aps.globalload"() <{global_name = @au_generated_candidate_1__direct_local_0}> : () -> i32
      %133 = "arith.index_cast"(%132) : (i32) -> index
      %134 = "arith.index_cast"(%133) : (index) -> i32
      %135 = "arith.index_cast"(%127) : (index) -> i32
      %136 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %137 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %138 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %139 = "arith.muli"(%134, %138) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %140 = "arith.addi"(%137, %139) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %141 = "arith.addi"(%140, %135) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %142 = "arith.addi"(%136, %141) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %143 = "aps.read_smem"(%104, %142) : (memref<96xf32>, i32) -> f32
      %144 = "arith.index_cast"(%arg7) : (index) -> i32
      %145 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %146 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %147 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %148 = "arith.muli"(%144, %147) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %149 = "arith.addi"(%146, %148) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %150 = "arith.addi"(%149, %128) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %151 = "arith.addi"(%145, %150) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %152 = "aps.read_smem"(%100, %151) : (memref<96xf32>, i32) -> f32
      %153 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %154 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %155 = "arith.addi"(%153, %154) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %156 = "aps.read_smem"(%89, %155) : (memref<1xf32>, i32) -> f32
      %157 = "arith.addf"(%152, %156) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %158 = "arith.subf"(%143, %157) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %159 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %160 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %161 = "arith.addi"(%159, %160) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %162 = "aps.read_smem"(%108, %161) : (memref<1xf32>, i32) -> f32
      %163 = "arith.addf"(%162, %158) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %164 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %165 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %166 = "arith.addi"(%164, %165) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%163, %108, %166) : (f32, memref<1xf32>, i32) -> ()
      %167 = "arith.index_cast"(%133) : (index) -> i32
      %168 = "arith.index_cast"(%126) : (index) -> i32
      %169 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %170 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %171 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %172 = "arith.muli"(%167, %171) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %173 = "arith.addi"(%170, %172) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %174 = "arith.addi"(%173, %168) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %175 = "arith.addi"(%169, %174) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %176 = "aps.read_smem"(%104, %175) : (memref<96xf32>, i32) -> f32
      %177 = "arith.index_cast"(%arg7) : (index) -> i32
      %178 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %179 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %180 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %181 = "arith.muli"(%177, %180) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %182 = "arith.addi"(%179, %181) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %183 = "arith.addi"(%182, %124) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %184 = "arith.addi"(%178, %183) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %185 = "aps.read_smem"(%100, %184) : (memref<96xf32>, i32) -> f32
      %186 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %187 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %188 = "arith.addi"(%186, %187) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %189 = "aps.read_smem"(%92, %188) : (memref<1xf32>, i32) -> f32
      %190 = "arith.addf"(%185, %189) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %191 = "arith.subf"(%176, %190) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %192 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %193 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %194 = "arith.addi"(%192, %193) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %195 = "aps.read_smem"(%112, %194) : (memref<1xf32>, i32) -> f32
      %196 = "arith.addf"(%195, %191) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %197 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %198 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %199 = "arith.addi"(%197, %198) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%196, %112, %199) : (f32, memref<1xf32>, i32) -> ()
      %200 = "arith.index_cast"(%133) : (index) -> i32
      %201 = "arith.index_cast"(%125) : (index) -> i32
      %202 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %203 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %204 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %205 = "arith.muli"(%200, %204) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %206 = "arith.addi"(%203, %205) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %207 = "arith.addi"(%206, %201) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %208 = "arith.addi"(%202, %207) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %209 = "aps.read_smem"(%104, %208) : (memref<96xf32>, i32) -> f32
      %210 = "arith.index_cast"(%arg7) : (index) -> i32
      %211 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %212 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %213 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %214 = "arith.muli"(%210, %213) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %215 = "arith.addi"(%212, %214) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %216 = "arith.addi"(%215, %123) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %217 = "arith.addi"(%211, %216) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %218 = "aps.read_smem"(%100, %217) : (memref<96xf32>, i32) -> f32
      %219 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %220 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %221 = "arith.addi"(%219, %220) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %222 = "aps.read_smem"(%96, %221) : (memref<1xf32>, i32) -> f32
      %223 = "arith.addf"(%218, %222) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %224 = "arith.subf"(%209, %223) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %225 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %226 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %227 = "arith.addi"(%225, %226) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %228 = "aps.read_smem"(%116, %227) : (memref<1xf32>, i32) -> f32
      %229 = "arith.addf"(%228, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %230 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %231 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %232 = "arith.addi"(%230, %231) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%229, %116, %232) : (f32, memref<1xf32>, i32) -> ()
      %233 = "aps.globalload"() <{global_name = @au_generated_candidate_1__direct_local}> : () -> f32
      %234 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %235 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %236 = "arith.addi"(%234, %235) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %237 = "aps.read_smem"(%120, %236) : (memref<1xf32>, i32) -> f32
      %238 = "arith.addf"(%237, %233) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %239 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %240 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %241 = "arith.addi"(%239, %240) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%238, %120, %241) : (f32, memref<1xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%107, %108, %109, %110) <{direction = 1 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    "aps.copy"(%111, %112, %113, %114) <{direction = 1 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    "aps.copy"(%115, %116, %117, %118) <{direction = 1 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    "aps.copy"(%119, %120, %121, %122) <{direction = 1 : i32}> : (i32, memref<1xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<f32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<f32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<f32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2050 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2051 : i32, arg_index = 4 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2052 : i32, arg_index = 5 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<f32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2053 : i32, arg_index = 6 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<f32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2054 : i32, arg_index = 7 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<f32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2055 : i32, arg_index = 8 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<f32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<f32>, memref<f32>, memref<f32>, memref<32x3xf32>, memref<32x3xf32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_8__l0u"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_8__l0u_buffer}> : () -> memref<96xf32>
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    "aps.copy"(%0, %2, %3, %4) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %5 = "memref.get_global"() <{name = @au_generated_candidate_8__l0u_buffer_9}> : () -> memref<96xf32>
    %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    "aps.copy"(%1, %5, %6, %7) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %8 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 6.000000e-02 : f32}> : () -> f32
    %12 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
    %13 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %14 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    %15 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %16 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = 2.500000e-02 : f32}> : () -> f32
    %18 = "arith.constant"() <{value = 37 : i32}> : () -> i32
    %19 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %20 = "arith.constant"() <{value = 3.000000e-02 : f32}> : () -> f32
    %21 = "arith.constant"() <{value = 41 : i32}> : () -> i32
    %22 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg3: index):
      %23 = "arith.index_cast"(%arg3) : (index) -> i32
      %24 = "arith.muli"(%23, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %25 = "arith.remsi"(%24, %21) : (i32, i32) -> i32
      %26 = "arith.sitofp"(%25) : (i32) -> f32
      %27 = "arith.mulf"(%26, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %28 = "arith.muli"(%23, %19) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %29 = "arith.remsi"(%28, %18) : (i32, i32) -> i32
      %30 = "arith.sitofp"(%29) : (i32) -> f32
      %31 = "arith.mulf"(%30, %17) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %32 = "arith.muli"(%23, %16) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %33 = "arith.remsi"(%32, %15) : (i32, i32) -> i32
      %34 = "arith.sitofp"(%33) : (i32) -> f32
      %35 = "arith.mulf"(%34, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %36 = "arith.index_cast"(%arg3) : (index) -> i32
      %37 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %38 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %39 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %40 = "arith.muli"(%36, %39) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %41 = "arith.addi"(%38, %40) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %42 = "arith.addi"(%41, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %43 = "arith.addi"(%37, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%27, %2, %43) : (f32, memref<96xf32>, i32) -> ()
      %44 = "arith.index_cast"(%arg3) : (index) -> i32
      %45 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %46 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %47 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %48 = "arith.muli"(%44, %47) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %49 = "arith.addi"(%46, %48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %50 = "arith.addi"(%49, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %51 = "arith.addi"(%45, %50) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%31, %2, %51) : (f32, memref<96xf32>, i32) -> ()
      %52 = "arith.index_cast"(%arg3) : (index) -> i32
      %53 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %54 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %55 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %56 = "arith.muli"(%52, %55) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %57 = "arith.addi"(%54, %56) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %58 = "arith.addi"(%57, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %59 = "arith.addi"(%53, %58) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%35, %2, %59) : (f32, memref<96xf32>, i32) -> ()
      %60 = "arith.addf"(%27, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %61 = "arith.index_cast"(%arg3) : (index) -> i32
      %62 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %63 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %64 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %65 = "arith.muli"(%61, %64) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %66 = "arith.addi"(%63, %65) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %67 = "arith.addi"(%66, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %68 = "arith.addi"(%62, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%60, %5, %68) : (f32, memref<96xf32>, i32) -> ()
      %69 = "arith.subf"(%31, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %70 = "arith.index_cast"(%arg3) : (index) -> i32
      %71 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %72 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %73 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %74 = "arith.muli"(%70, %73) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %75 = "arith.addi"(%72, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %76 = "arith.addi"(%75, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %77 = "arith.addi"(%71, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%69, %5, %77) : (f32, memref<96xf32>, i32) -> ()
      %78 = "arith.addf"(%35, %11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %79 = "arith.index_cast"(%arg3) : (index) -> i32
      %80 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %81 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %82 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %83 = "arith.muli"(%79, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %84 = "arith.addi"(%81, %83) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %85 = "arith.addi"(%84, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %86 = "arith.addi"(%80, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%78, %5, %86) : (f32, memref<96xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "aps.copy"(%0, %2, %3, %4) <{direction = 1 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    "aps.copy"(%1, %5, %6, %7) <{direction = 1 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<32x3xf32>, memref<32x3xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_1__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_1__direct_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_1__direct_csr2", sym_visibility = "private", type = memref<i32>}> {csr_address = 2051 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_1__direct_csr3", sym_visibility = "private", type = memref<i32>}> {csr_address = 2052 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_1__direct_csr4", sym_visibility = "private", type = memref<i32>}> {csr_address = 2053 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_1__direct_csr5", sym_visibility = "private", type = memref<i32>}> {csr_address = 2054 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_1__direct_csr6", sym_visibility = "private", type = memref<i32>}> {csr_address = 2055 : i32} : () -> ()
}) : () -> ()

