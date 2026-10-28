"builtin.module"() ({
  "memref.global"() <{initial_value = dense<6.000000e+00> : tensor<1xf32>, sym_name = "_ZL10DEPTH_BIAS", sym_visibility = "private", type = memref<1xf32>}> : () -> ()
  "memref.global"() <{initial_value = dense<2.160000e+02> : tensor<1xf32>, sym_name = "_ZL2FY", sym_visibility = "private", type = memref<1xf32>}> : () -> ()
  "memref.global"() <{initial_value = dense<2.200000e+02> : tensor<1xf32>, sym_name = "_ZL2FX", sym_visibility = "private", type = memref<1xf32>}> : () -> ()
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %118 = "arith.constant"() <{value = 192 : index}> : () -> index
    %119 = "arith.constant"() <{value = 3 : index}> : () -> index
    %120 = "arith.constant"() <{value = 6 : index}> : () -> index
    %121 = "arith.constant"() <{value = 24 : index}> : () -> index
    %122 = "arith.constant"() <{value = 8 : index}> : () -> index
    %123 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
    %124 = "arith.constant"() <{value = 1.310000e+02 : f32}> : () -> f32
    %125 = "arith.constant"() <{value = 1.490000e+02 : f32}> : () -> f32
    %126 = "arith.constant"() <{value = 2 : index}> : () -> index
    %127 = "arith.constant"() <{value = 1.000000e-03 : f32}> : () -> f32
    %128 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %129 = "arith.constant"() <{value = 0 : index}> : () -> index
    %130 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %131 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %132 = "arith.constant"() <{value = 1 : index}> : () -> index
    %133 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %134 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %135 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %136 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %137 = "arith.constant"() <{value = 3.000000e-02 : f32}> : () -> f32
    %138 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    %139 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %140 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %141 = "arith.constant"() <{value = 1.000000e-01 : f32}> : () -> f32
    %142 = "arith.constant"() <{value = 7.000000e-02 : f32}> : () -> f32
    %143 = "arith.constant"() <{value = 5.000000e-02 : f32}> : () -> f32
    %144 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %145 = "arith.constant"() <{value = -3 : i32}> : () -> i32
    %146 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %147 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<192x2xf32>
    %148 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<192xi32>
    %149 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<192xi32>
    %150 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<24x3xf32>
    %151 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x3xf32>
    %152 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    "memref.store"(%131, %152, %129) : (f32, memref<1xf32>, index) -> ()
    %153 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    "memref.store"(%131, %153, %129) : (f32, memref<1xf32>, index) -> ()
    %154 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    "memref.store"(%131, %154, %129) : (f32, memref<1xf32>, index) -> ()
    %155 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%130, %155) : (i32, memref<i32>) -> ()
    "func.call"(%151) <{callee = @au_generated_candidate_44__l0u_site_10681}> : (memref<8x3xf32>) -> ()
    "func.call"(%150) <{callee = @au_generated_candidate_38__l0u_site_10702}> : (memref<24x3xf32>) -> ()
    "memref.store"(%130, %155) : (i32, memref<i32>) -> ()
    %156 = "memref.get_global"() <{name = @_ZL10DEPTH_BIAS}> : () -> memref<1xf32>
    %157 = "memref.get_global"() <{name = @_ZL2FX}> : () -> memref<1xf32>
    %158 = "memref.get_global"() <{name = @_ZL2FY}> : () -> memref<1xf32>
    "scf.for"(%129, %122, %132) ({
    ^bb0(%arg27: index):
      %292 = "arith.index_cast"(%arg27) : (index) -> i32
      "scf.for"(%129, %121, %132) ({
      ^bb0(%arg28: index):
        %293 = "arith.index_cast"(%arg28) : (index) -> i32
        %294 = "memref.load"(%155) : (memref<i32>) -> i32
        %295 = "arith.index_cast"(%294) : (i32) -> index
        "memref.store"(%292, %149, %295) : (i32, memref<192xi32>, index) -> ()
        "memref.store"(%293, %148, %295) : (i32, memref<192xi32>, index) -> ()
        %296 = "memref.load"(%150, %arg28, %129) : (memref<24x3xf32>, index, index) -> f32
        %297 = "memref.load"(%151, %arg27, %129) : (memref<8x3xf32>, index, index) -> f32
        %298 = "arith.subf"(%296, %297) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %299 = "memref.load"(%150, %arg28, %132) : (memref<24x3xf32>, index, index) -> f32
        %300 = "memref.load"(%151, %arg27, %132) : (memref<8x3xf32>, index, index) -> f32
        %301 = "arith.subf"(%299, %300) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %302 = "memref.load"(%150, %arg28, %126) : (memref<24x3xf32>, index, index) -> f32
        %303 = "memref.load"(%151, %arg27, %126) : (memref<8x3xf32>, index, index) -> f32
        %304 = "arith.subf"(%302, %303) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %305 = "memref.load"(%156, %129) : (memref<1xf32>, index) -> f32
        %306 = "arith.addf"(%304, %305) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %307 = "arith.divf"(%128, %306) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%307, %152, %129) : (f32, memref<1xf32>, index) -> ()
        %308 = "memref.load"(%157, %129) : (memref<1xf32>, index) -> f32
        %309 = "arith.mulf"(%308, %298) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %310 = "memref.load"(%152, %129) : (memref<1xf32>, index) -> f32
        %311 = "arith.mulf"(%309, %310) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%311, %154, %129) : (f32, memref<1xf32>, index) -> ()
        %312 = "memref.load"(%158, %129) : (memref<1xf32>, index) -> f32
        %313 = "arith.mulf"(%312, %301) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %314 = "memref.load"(%152, %129) : (memref<1xf32>, index) -> f32
        %315 = "arith.mulf"(%313, %314) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%315, %153, %129) : (f32, memref<1xf32>, index) -> ()
        %316 = "memref.load"(%154, %129) : (memref<1xf32>, index) -> f32
        %317 = "arith.remsi"(%294, %136) : (i32, i32) -> i32
        %318 = "arith.addi"(%317, %146) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %319 = "arith.sitofp"(%318) : (i32) -> f32
        %320 = "arith.mulf"(%319, %137) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %321 = "arith.addf"(%316, %320) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%321, %147, %295, %129) : (f32, memref<192x2xf32>, index, index) -> ()
        %322 = "memref.load"(%153, %129) : (memref<1xf32>, index) -> f32
        %323 = "arith.remsi"(%294, %139) : (i32, i32) -> i32
        %324 = "arith.addi"(%323, %145) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %325 = "arith.sitofp"(%324) : (i32) -> f32
        %326 = "arith.mulf"(%325, %138) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %327 = "arith.subf"(%322, %326) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%327, %147, %295, %132) : (f32, memref<192x2xf32>, index, index) -> ()
        %328 = "arith.addi"(%294, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%328, %155) : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.call"(%151) <{callee = @au_generated_candidate_9__l0u_site_10787}> : (memref<8x3xf32>) -> ()
    "scf.for"(%129, %121, %132) ({
    ^bb0(%arg26: index):
      %275 = "arith.index_cast"(%arg26) : (index) -> i32
      %276 = "arith.remsi"(%275, %134) : (i32, i32) -> i32
      %277 = "arith.sitofp"(%276) : (i32) -> f32
      %278 = "arith.mulf"(%277, %141) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %279 = "memref.load"(%150, %arg26, %129) : (memref<24x3xf32>, index, index) -> f32
      %280 = "arith.subf"(%279, %278) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%280, %150, %arg26, %129) : (f32, memref<24x3xf32>, index, index) -> ()
      %281 = "arith.remsi"(%275, %135) : (i32, i32) -> i32
      %282 = "arith.sitofp"(%281) : (i32) -> f32
      %283 = "arith.mulf"(%282, %142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %284 = "memref.load"(%150, %arg26, %132) : (memref<24x3xf32>, index, index) -> f32
      %285 = "arith.addf"(%284, %283) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%285, %150, %arg26, %132) : (f32, memref<24x3xf32>, index, index) -> ()
      %286 = "arith.remsi"(%275, %140) : (i32, i32) -> i32
      %287 = "arith.addi"(%286, %144) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %288 = "arith.sitofp"(%287) : (i32) -> f32
      %289 = "arith.mulf"(%288, %143) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %290 = "memref.load"(%150, %arg26, %126) : (memref<24x3xf32>, index, index) -> f32
      %291 = "arith.addf"(%290, %289) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%291, %150, %arg26, %126) : (f32, memref<24x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %159 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    "memref.store"(%131, %159, %129) : (f32, memref<1xf32>, index) -> ()
    %160 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    "memref.store"(%131, %160, %129) : (f32, memref<1xf32>, index) -> ()
    %161 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xf32>
    "memref.store"(%131, %161, %129) : (f32, memref<1xf32>, index) -> ()
    %162 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<24x3xf32>
    %163 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<24x3xf32>
    %164 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x3xf32>
    %165 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x3xf32>
    "scf.for"(%129, %120, %132) ({
    ^bb0(%arg20: index):
      "scf.for"(%129, %122, %132) ({
      ^bb0(%arg24: index):
        "scf.for"(%129, %119, %132) ({
        ^bb0(%arg25: index):
          "memref.store"(%131, %165, %arg24, %arg25) : (f32, memref<8x3xf32>, index, index) -> ()
          "memref.store"(%127, %164, %arg24, %arg25) : (f32, memref<8x3xf32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%129, %119, %132) ({
      ^bb0(%arg22: index):
        "scf.for"(%129, %121, %132) ({
        ^bb0(%arg23: index):
          "memref.store"(%131, %163, %arg23, %arg22) : (f32, memref<24x3xf32>, index, index) -> ()
          "memref.store"(%127, %162, %arg23, %arg22) : (f32, memref<24x3xf32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%129, %118, %132) ({
      ^bb0(%arg21: index):
        %187 = "memref.load"(%149, %arg21) : (memref<192xi32>, index) -> i32
        %188 = "memref.load"(%148, %arg21) : (memref<192xi32>, index) -> i32
        %189 = "arith.index_cast"(%187) : (i32) -> index
        %190 = "arith.index_cast"(%188) : (i32) -> index
        %191 = "memref.load"(%150, %190, %129) : (memref<24x3xf32>, index, index) -> f32
        %192 = "memref.load"(%151, %189, %129) : (memref<8x3xf32>, index, index) -> f32
        %193 = "arith.subf"(%191, %192) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %194 = "memref.load"(%150, %190, %132) : (memref<24x3xf32>, index, index) -> f32
        %195 = "memref.load"(%151, %189, %132) : (memref<8x3xf32>, index, index) -> f32
        %196 = "arith.subf"(%194, %195) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %197 = "memref.load"(%150, %190, %126) : (memref<24x3xf32>, index, index) -> f32
        %198 = "memref.load"(%151, %189, %126) : (memref<8x3xf32>, index, index) -> f32
        %199 = "arith.subf"(%197, %198) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %200 = "memref.load"(%156, %129) : (memref<1xf32>, index) -> f32
        %201 = "arith.addf"(%199, %200) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %202 = "arith.divf"(%128, %201) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%202, %159, %129) : (f32, memref<1xf32>, index) -> ()
        %203 = "memref.load"(%157, %129) : (memref<1xf32>, index) -> f32
        %204 = "arith.mulf"(%203, %193) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %205 = "memref.load"(%159, %129) : (memref<1xf32>, index) -> f32
        %206 = "arith.mulf"(%204, %205) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%206, %161, %129) : (f32, memref<1xf32>, index) -> ()
        %207 = "memref.load"(%158, %129) : (memref<1xf32>, index) -> f32
        %208 = "arith.mulf"(%207, %196) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %209 = "memref.load"(%159, %129) : (memref<1xf32>, index) -> f32
        %210 = "arith.mulf"(%208, %209) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%210, %160, %129) : (f32, memref<1xf32>, index) -> ()
        %211 = "memref.load"(%150, %190, %129) : (memref<24x3xf32>, index, index) -> f32
        %212 = "memref.load"(%151, %189, %129) : (memref<8x3xf32>, index, index) -> f32
        %213 = "arith.subf"(%211, %212) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %214 = "memref.load"(%150, %190, %132) : (memref<24x3xf32>, index, index) -> f32
        %215 = "memref.load"(%151, %189, %132) : (memref<8x3xf32>, index, index) -> f32
        %216 = "arith.subf"(%214, %215) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %217 = "memref.load"(%161, %129) : (memref<1xf32>, index) -> f32
        %218 = "memref.load"(%147, %arg21, %129) : (memref<192x2xf32>, index, index) -> f32
        %219 = "arith.subf"(%217, %218) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %220 = "memref.load"(%160, %129) : (memref<1xf32>, index) -> f32
        %221 = "memref.load"(%147, %arg21, %132) : (memref<192x2xf32>, index, index) -> f32
        %222 = "arith.subf"(%220, %221) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %223 = "memref.load"(%157, %129) : (memref<1xf32>, index) -> f32
        %224 = "memref.load"(%159, %129) : (memref<1xf32>, index) -> f32
        %225 = "arith.mulf"(%223, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %226 = "memref.load"(%158, %129) : (memref<1xf32>, index) -> f32
        %227 = "arith.mulf"(%226, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %228 = "arith.negf"(%223) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %229 = "arith.mulf"(%228, %213) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %230 = "arith.mulf"(%229, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %231 = "arith.mulf"(%230, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %232 = "arith.negf"(%226) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %233 = "arith.mulf"(%232, %216) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %234 = "arith.mulf"(%233, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %235 = "arith.mulf"(%234, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %236 = "arith.negf"(%219) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %237 = "arith.mulf"(%236, %225) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %238 = "memref.load"(%165, %189, %129) : (memref<8x3xf32>, index, index) -> f32
        %239 = "arith.addf"(%238, %237) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%239, %165, %189, %129) : (f32, memref<8x3xf32>, index, index) -> ()
        %240 = "arith.negf"(%222) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %241 = "arith.mulf"(%240, %227) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %242 = "memref.load"(%165, %189, %132) : (memref<8x3xf32>, index, index) -> f32
        %243 = "arith.addf"(%242, %241) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%243, %165, %189, %132) : (f32, memref<8x3xf32>, index, index) -> ()
        %244 = "arith.mulf"(%219, %231) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %245 = "arith.mulf"(%222, %235) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %246 = "arith.addf"(%244, %245) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %247 = "arith.negf"(%246) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %248 = "memref.load"(%165, %189, %126) : (memref<8x3xf32>, index, index) -> f32
        %249 = "arith.addf"(%248, %247) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%249, %165, %189, %126) : (f32, memref<8x3xf32>, index, index) -> ()
        %250 = "arith.mulf"(%225, %225) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %251 = "memref.load"(%164, %189, %129) : (memref<8x3xf32>, index, index) -> f32
        %252 = "arith.addf"(%251, %250) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%252, %164, %189, %129) : (f32, memref<8x3xf32>, index, index) -> ()
        %253 = "arith.mulf"(%227, %227) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %254 = "memref.load"(%164, %189, %132) : (memref<8x3xf32>, index, index) -> f32
        %255 = "arith.addf"(%254, %253) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%255, %164, %189, %132) : (f32, memref<8x3xf32>, index, index) -> ()
        %256 = "arith.mulf"(%231, %231) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %257 = "arith.mulf"(%235, %235) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %258 = "arith.addf"(%256, %257) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %259 = "memref.load"(%164, %189, %126) : (memref<8x3xf32>, index, index) -> f32
        %260 = "arith.addf"(%259, %258) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%260, %164, %189, %126) : (f32, memref<8x3xf32>, index, index) -> ()
        %261 = "arith.mulf"(%219, %225) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %262 = "memref.load"(%163, %190, %129) : (memref<24x3xf32>, index, index) -> f32
        %263 = "arith.addf"(%262, %261) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%263, %163, %190, %129) : (f32, memref<24x3xf32>, index, index) -> ()
        %264 = "arith.mulf"(%222, %227) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %265 = "memref.load"(%163, %190, %132) : (memref<24x3xf32>, index, index) -> f32
        %266 = "arith.addf"(%265, %264) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%266, %163, %190, %132) : (f32, memref<24x3xf32>, index, index) -> ()
        %267 = "memref.load"(%163, %190, %126) : (memref<24x3xf32>, index, index) -> f32
        %268 = "arith.addf"(%267, %246) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%268, %163, %190, %126) : (f32, memref<24x3xf32>, index, index) -> ()
        %269 = "memref.load"(%162, %190, %129) : (memref<24x3xf32>, index, index) -> f32
        %270 = "arith.addf"(%269, %250) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%270, %162, %190, %129) : (f32, memref<24x3xf32>, index, index) -> ()
        %271 = "memref.load"(%162, %190, %132) : (memref<24x3xf32>, index, index) -> f32
        %272 = "arith.addf"(%271, %253) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%272, %162, %190, %132) : (f32, memref<24x3xf32>, index, index) -> ()
        %273 = "memref.load"(%162, %190, %126) : (memref<24x3xf32>, index, index) -> f32
        %274 = "arith.addf"(%273, %258) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%274, %162, %190, %126) : (f32, memref<24x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "func.call"(%165, %164, %151) <{callee = @au_generated_candidate_49__l1p_site_10972}> : (memref<8x3xf32>, memref<8x3xf32>, memref<8x3xf32>) -> ()
      "func.call"(%163, %162, %150) <{callee = @au_generated_candidate_48__l0p_l1u_site_10989}> : (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %166 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%130, %166) : (i32, memref<i32>) -> ()
    "memref.store"(%130, %166) : (i32, memref<i32>) -> ()
    "scf.for"(%129, %122, %132) ({
    ^bb0(%arg19: index):
      %174 = "memref.load"(%151, %arg19, %129) : (memref<8x3xf32>, index, index) -> f32
      %175 = "arith.mulf"(%174, %123) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %176 = "arith.fptosi"(%175) : (f32) -> i32
      %177 = "memref.load"(%166) : (memref<i32>) -> i32
      %178 = "arith.addi"(%177, %176) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%178, %166) : (i32, memref<i32>) -> ()
      %179 = "memref.load"(%151, %arg19, %132) : (memref<8x3xf32>, index, index) -> f32
      %180 = "arith.mulf"(%179, %124) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %181 = "arith.fptosi"(%180) : (f32) -> i32
      %182 = "arith.addi"(%178, %181) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%182, %166) : (i32, memref<i32>) -> ()
      %183 = "memref.load"(%151, %arg19, %126) : (memref<8x3xf32>, index, index) -> f32
      %184 = "arith.mulf"(%183, %125) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %185 = "arith.fptosi"(%184) : (f32) -> i32
      %186 = "arith.addi"(%182, %185) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%186, %166) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.call"(%150, %166) <{callee = @au_generated_candidate_22__direct_site_11038}> : (memref<24x3xf32>, memref<i32>) -> ()
    %167 = "memref.load"(%166) : (memref<i32>) -> i32
    %168 = "arith.sitofp"(%167) : (i32) -> f32
    %169 = "arith.cmpi"(%167, %130) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %170 = "scf.if"(%169) ({
      %173 = "arith.negf"(%168) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "scf.yield"(%173) : (f32) -> ()
    }, {
      "scf.yield"(%168) : (f32) -> ()
    }) : (i1) -> f32
    %171 = "arith.cmpf"(%170, %131) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
    %172 = "arith.select"(%171, %167, %133) : (i1, i32, i32) -> i32
    "func.call"(%151, %150, %149, %148, %147) <{callee = @pcl_validate_outputs}> : (memref<8x3xf32>, memref<24x3xf32>, memref<192xi32>, memref<192xi32>, memref<192x2xf32>) -> ()
    "func.return"(%172) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<8x3xf32>) -> (), sym_name = "au_generated_candidate_44__l0u_site_10681"}> ({
  ^bb0(%arg17: memref<8x3xf32>):
    %96 = "arith.constant"() <{value = 2 : index}> : () -> index
    %97 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %98 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %99 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
    %100 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %101 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %102 = "arith.constant"() <{value = -1.800000e+00 : f32}> : () -> f32
    %103 = "arith.constant"() <{value = 5.500000e-01 : f32}> : () -> f32
    %104 = "arith.constant"() <{value = 0 : index}> : () -> index
    %105 = "arith.constant"() <{value = 8 : index}> : () -> index
    %106 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%104, %105, %106) ({
    ^bb0(%arg18: index):
      %107 = "arith.index_cast"(%arg18) : (index) -> i32
      %108 = "arith.sitofp"(%107) : (i32) -> f32
      %109 = "arith.mulf"(%108, %103) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %110 = "arith.addf"(%109, %102) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%110, %arg17, %arg18, %104) : (f32, memref<8x3xf32>, index, index) -> ()
      %111 = "arith.remsi"(%107, %101) : (i32, i32) -> i32
      %112 = "arith.addi"(%111, %100) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %113 = "arith.sitofp"(%112) : (i32) -> f32
      %114 = "arith.mulf"(%113, %99) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%114, %arg17, %arg18, %106) : (f32, memref<8x3xf32>, index, index) -> ()
      %115 = "arith.remsi"(%107, %98) : (i32, i32) -> i32
      %116 = "arith.sitofp"(%115) : (i32) -> f32
      %117 = "arith.mulf"(%116, %97) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%117, %arg17, %arg18, %96) : (f32, memref<8x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_38__l0u_site_10702"}> ({
  ^bb0(%arg15: memref<24x3xf32>):
    %71 = "arith.constant"() <{value = 2 : index}> : () -> index
    %72 = "arith.constant"() <{value = 2.800000e+00 : f32}> : () -> f32
    %73 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %74 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %75 = "arith.constant"() <{value = -0.899999976 : f32}> : () -> f32
    %76 = "arith.constant"() <{value = 6.000000e-01 : f32}> : () -> f32
    %77 = "arith.constant"() <{value = -1.200000e+00 : f32}> : () -> f32
    %78 = "arith.constant"() <{value = 4.500000e-01 : f32}> : () -> f32
    %79 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    %80 = "arith.constant"() <{value = 0 : index}> : () -> index
    %81 = "arith.constant"() <{value = 24 : index}> : () -> index
    %82 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%80, %81, %82) ({
    ^bb0(%arg16: index):
      %83 = "arith.index_cast"(%arg16) : (index) -> i32
      %84 = "arith.remsi"(%83, %79) : (i32, i32) -> i32
      %85 = "arith.sitofp"(%84) : (i32) -> f32
      %86 = "arith.mulf"(%85, %78) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %87 = "arith.addf"(%86, %77) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%87, %arg15, %arg16, %80) : (f32, memref<24x3xf32>, index, index) -> ()
      %88 = "arith.divsi"(%83, %79) : (i32, i32) -> i32
      %89 = "arith.sitofp"(%88) : (i32) -> f32
      %90 = "arith.mulf"(%89, %76) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %91 = "arith.addf"(%90, %75) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%91, %arg15, %arg16, %82) : (f32, memref<24x3xf32>, index, index) -> ()
      %92 = "arith.remsi"(%83, %74) : (i32, i32) -> i32
      %93 = "arith.sitofp"(%92) : (i32) -> f32
      %94 = "arith.mulf"(%93, %73) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %95 = "arith.addf"(%94, %72) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%95, %arg15, %arg16, %71) : (f32, memref<24x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<8x3xf32>) -> (), sym_name = "au_generated_candidate_9__l0u_site_10787"}> ({
  ^bb0(%arg13: memref<8x3xf32>):
    %42 = "arith.constant"() <{value = 2 : index}> : () -> index
    %43 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
    %44 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %45 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %46 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %47 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %48 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %49 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %50 = "arith.constant"() <{value = 0 : index}> : () -> index
    %51 = "arith.constant"() <{value = 8 : index}> : () -> index
    %52 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%50, %51, %52) ({
    ^bb0(%arg14: index):
      %53 = "arith.index_cast"(%arg14) : (index) -> i32
      %54 = "arith.remsi"(%53, %49) : (i32, i32) -> i32
      %55 = "arith.addi"(%54, %48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %56 = "arith.sitofp"(%55) : (i32) -> f32
      %57 = "arith.mulf"(%56, %47) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %58 = "memref.load"(%arg13, %arg14, %50) : (memref<8x3xf32>, index, index) -> f32
      %59 = "arith.addf"(%58, %57) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%59, %arg13, %arg14, %50) : (f32, memref<8x3xf32>, index, index) -> ()
      %60 = "arith.remsi"(%53, %46) : (i32, i32) -> i32
      %61 = "arith.sitofp"(%60) : (i32) -> f32
      %62 = "arith.mulf"(%61, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %63 = "memref.load"(%arg13, %arg14, %52) : (memref<8x3xf32>, index, index) -> f32
      %64 = "arith.subf"(%63, %62) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%64, %arg13, %arg14, %52) : (f32, memref<8x3xf32>, index, index) -> ()
      %65 = "arith.remsi"(%53, %44) : (i32, i32) -> i32
      %66 = "arith.addi"(%65, %48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %67 = "arith.sitofp"(%66) : (i32) -> f32
      %68 = "arith.mulf"(%67, %43) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %69 = "memref.load"(%arg13, %arg14, %42) : (memref<8x3xf32>, index, index) -> f32
      %70 = "arith.addf"(%69, %68) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%70, %arg13, %arg14, %42) : (f32, memref<8x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<8x3xf32>, memref<8x3xf32>, memref<8x3xf32>) -> (), sym_name = "au_generated_candidate_49__l1p_site_10972"}> ({
  ^bb0(%arg8: memref<8x3xf32>, %arg9: memref<8x3xf32>, %arg10: memref<8x3xf32>):
    %31 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %32 = "arith.constant"() <{value = 3 : index}> : () -> index
    %33 = "arith.constant"() <{value = 0 : index}> : () -> index
    %34 = "arith.constant"() <{value = 8 : index}> : () -> index
    %35 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%33, %34, %35) ({
    ^bb0(%arg11: index):
      "scf.for"(%33, %32, %35) ({
      ^bb0(%arg12: index):
        %36 = "memref.load"(%arg8, %arg11, %arg12) : (memref<8x3xf32>, index, index) -> f32
        %37 = "arith.mulf"(%36, %31) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %38 = "memref.load"(%arg9, %arg11, %arg12) : (memref<8x3xf32>, index, index) -> f32
        %39 = "arith.divf"(%37, %38) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %40 = "memref.load"(%arg10, %arg11, %arg12) : (memref<8x3xf32>, index, index) -> f32
        %41 = "arith.subf"(%40, %39) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%41, %arg10, %arg11, %arg12) : (f32, memref<8x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_48__l0p_l1u_site_10989"}> ({
  ^bb0(%arg3: memref<24x3xf32>, %arg4: memref<24x3xf32>, %arg5: memref<24x3xf32>):
    %20 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    %21 = "arith.constant"() <{value = 3 : index}> : () -> index
    %22 = "arith.constant"() <{value = 0 : index}> : () -> index
    %23 = "arith.constant"() <{value = 24 : index}> : () -> index
    %24 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%22, %23, %24) ({
    ^bb0(%arg6: index):
      "scf.for"(%22, %21, %24) ({
      ^bb0(%arg7: index):
        %25 = "memref.load"(%arg3, %arg6, %arg7) : (memref<24x3xf32>, index, index) -> f32
        %26 = "arith.mulf"(%25, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %27 = "memref.load"(%arg4, %arg6, %arg7) : (memref<24x3xf32>, index, index) -> f32
        %28 = "arith.divf"(%26, %27) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %29 = "memref.load"(%arg5, %arg6, %arg7) : (memref<24x3xf32>, index, index) -> f32
        %30 = "arith.subf"(%29, %28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%30, %arg5, %arg6, %arg7) : (f32, memref<24x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<24x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_22__direct_site_11038"}> ({
  ^bb0(%arg0: memref<24x3xf32>, %arg1: memref<i32>):
    %0 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 2 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1.090000e+02 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = 8.300000e+01 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 0 : index}> : () -> index
    %5 = "arith.constant"() <{value = 24 : index}> : () -> index
    %6 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%4, %5, %6) ({
    ^bb0(%arg2: index):
      %7 = "memref.load"(%arg0, %arg2, %4) : (memref<24x3xf32>, index, index) -> f32
      %8 = "arith.mulf"(%7, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %9 = "arith.fptosi"(%8) : (f32) -> i32
      %10 = "memref.load"(%arg1) : (memref<i32>) -> i32
      %11 = "arith.addi"(%10, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%11, %arg1) : (i32, memref<i32>) -> ()
      %12 = "memref.load"(%arg0, %arg2, %6) : (memref<24x3xf32>, index, index) -> f32
      %13 = "arith.mulf"(%12, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %14 = "arith.fptosi"(%13) : (f32) -> i32
      %15 = "arith.addi"(%11, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%15, %arg1) : (i32, memref<i32>) -> ()
      %16 = "memref.load"(%arg0, %arg2, %1) : (memref<24x3xf32>, index, index) -> f32
      %17 = "arith.mulf"(%16, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %18 = "arith.fptosi"(%17) : (f32) -> i32
      %19 = "arith.addi"(%15, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%19, %arg1) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<8x3xf32>, memref<24x3xf32>, memref<192xi32>, memref<192xi32>, memref<192x2xf32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
