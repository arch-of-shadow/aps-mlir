"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %118 = "arith.constant"() <{value = 7 : index}> : () -> index
    %119 = "arith.constant"() <{value = 64 : index}> : () -> index
    %120 = "arith.constant"() <{value = 20 : index}> : () -> index
    %121 = "arith.constant"() <{value = 16 : index}> : () -> index
    %122 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %123 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %124 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %125 = "arith.constant"() <{value = 1.000000e-03 : f32}> : () -> f32
    %126 = "arith.constant"() <{value = -0.899999976 : f32}> : () -> f32
    %127 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    %128 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %129 = "arith.constant"() <{value = 4.000000e-01 : f32}> : () -> f32
    %130 = "arith.constant"() <{value = 1.000000e-01 : f32}> : () -> f32
    %131 = "arith.constant"() <{value = 0 : index}> : () -> index
    %132 = "arith.constant"() <{value = 2 : index}> : () -> index
    %133 = "arith.constant"() <{value = 1 : index}> : () -> index
    %134 = "arith.constant"() <{value = 16 : i32}> : () -> i32
    %135 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %136 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %137 = "arith.constant"() <{value = 1.500000e-02 : f32}> : () -> f32
    %138 = "arith.constant"() <{value = 20 : i32}> : () -> i32
    %139 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %140 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    %141 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %142 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %143 = "arith.constant"() <{value = 7.000000e-02 : f32}> : () -> f32
    %144 = "arith.constant"() <{value = 0.00999999977 : f32}> : () -> f32
    %145 = "arith.constant"() <{value = 9.000000e-02 : f32}> : () -> f32
    %146 = "arith.constant"() <{value = 6.000000e-02 : f32}> : () -> f32
    %147 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %148 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64x2xf32>
    %149 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64xi32>
    %150 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<64xi32>
    %151 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<20x2xf32>
    %152 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x3xf32>
    "func.call"(%152) <{callee = @au_generated_candidate_34__l0u_site_3539}> : (memref<16x3xf32>) -> ()
    "func.call"(%151) <{callee = @au_generated_candidate_56__l0u_site_3555}> : (memref<20x2xf32>) -> ()
    "scf.for"(%131, %119, %133) ({
    ^bb0(%arg24: index):
      %252 = "arith.index_cast"(%arg24) : (index) -> i32
      %253 = "arith.remsi"(%252, %134) : (i32, i32) -> i32
      %254 = "arith.muli"(%252, %135) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %255 = "arith.divsi"(%252, %139) : (i32, i32) -> i32
      %256 = "arith.addi"(%254, %255) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %257 = "arith.remsi"(%256, %138) : (i32, i32) -> i32
      %258 = "arith.index_cast"(%257) : (i32) -> index
      %259 = "memref.load"(%151, %258, %131) : (memref<20x2xf32>, index, index) -> f32
      %260 = "arith.index_cast"(%253) : (i32) -> index
      %261 = "memref.load"(%152, %260, %131) : (memref<16x3xf32>, index, index) -> f32
      %262 = "arith.subf"(%259, %261) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %263 = "memref.load"(%151, %258, %133) : (memref<20x2xf32>, index, index) -> f32
      %264 = "memref.load"(%152, %260, %133) : (memref<16x3xf32>, index, index) -> f32
      %265 = "arith.subf"(%263, %264) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %266 = "memref.load"(%152, %260, %132) : (memref<16x3xf32>, index, index) -> f32
      "memref.store"(%253, %150, %arg24) : (i32, memref<64xi32>, index) -> ()
      "memref.store"(%257, %149, %arg24) : (i32, memref<64xi32>, index) -> ()
      %267 = "arith.mulf"(%266, %265) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %268 = "arith.addf"(%262, %267) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %269 = "arith.remsi"(%252, %135) : (i32, i32) -> i32
      %270 = "arith.sitofp"(%269) : (i32) -> f32
      %271 = "arith.mulf"(%270, %140) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %272 = "arith.addf"(%268, %271) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%272, %148, %arg24, %131) : (f32, memref<64x2xf32>, index, index) -> ()
      %273 = "arith.mulf"(%266, %262) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %274 = "arith.subf"(%265, %273) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %275 = "arith.remsi"(%252, %141) : (i32, i32) -> i32
      %276 = "arith.sitofp"(%275) : (i32) -> f32
      %277 = "arith.mulf"(%276, %137) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %278 = "arith.subf"(%274, %277) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%278, %148, %arg24, %133) : (f32, memref<64x2xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%131, %121, %133) ({
    ^bb0(%arg23: index):
      %235 = "arith.index_cast"(%arg23) : (index) -> i32
      %236 = "arith.remsi"(%235, %141) : (i32, i32) -> i32
      %237 = "arith.addi"(%236, %147) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %238 = "arith.sitofp"(%237) : (i32) -> f32
      %239 = "arith.mulf"(%238, %142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %240 = "memref.load"(%152, %arg23, %131) : (memref<16x3xf32>, index, index) -> f32
      %241 = "arith.addf"(%240, %239) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%241, %152, %arg23, %131) : (f32, memref<16x3xf32>, index, index) -> ()
      %242 = "arith.remsi"(%235, %135) : (i32, i32) -> i32
      %243 = "arith.sitofp"(%242) : (i32) -> f32
      %244 = "arith.mulf"(%243, %143) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %245 = "memref.load"(%152, %arg23, %133) : (memref<16x3xf32>, index, index) -> f32
      %246 = "arith.subf"(%245, %244) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%246, %152, %arg23, %133) : (f32, memref<16x3xf32>, index, index) -> ()
      %247 = "arith.remsi"(%235, %136) : (i32, i32) -> i32
      %248 = "arith.sitofp"(%247) : (i32) -> f32
      %249 = "arith.mulf"(%248, %144) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %250 = "memref.load"(%152, %arg23, %132) : (memref<16x3xf32>, index, index) -> f32
      %251 = "arith.addf"(%250, %249) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%251, %152, %arg23, %132) : (f32, memref<16x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%131, %120, %133) ({
    ^bb0(%arg22: index):
      %224 = "arith.index_cast"(%arg22) : (index) -> i32
      %225 = "arith.remsi"(%224, %135) : (i32, i32) -> i32
      %226 = "arith.sitofp"(%225) : (i32) -> f32
      %227 = "arith.mulf"(%226, %145) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %228 = "memref.load"(%151, %arg22, %131) : (memref<20x2xf32>, index, index) -> f32
      %229 = "arith.subf"(%228, %227) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%229, %151, %arg22, %131) : (f32, memref<20x2xf32>, index, index) -> ()
      %230 = "arith.remsi"(%224, %136) : (i32, i32) -> i32
      %231 = "arith.sitofp"(%230) : (i32) -> f32
      %232 = "arith.mulf"(%231, %146) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %233 = "memref.load"(%151, %arg22, %133) : (memref<20x2xf32>, index, index) -> f32
      %234 = "arith.addf"(%233, %232) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%234, %151, %arg22, %133) : (f32, memref<20x2xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %153 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<20x2xf32>
    %154 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<20x2xf32>
    %155 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x3xf32>
    %156 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x3xf32>
    "scf.for"(%131, %118, %133) ({
    ^bb0(%arg18: index):
      "scf.for"(%131, %121, %133) ({
      ^bb0(%arg21: index):
        "memref.store"(%124, %156, %arg21, %131) : (f32, memref<16x3xf32>, index, index) -> ()
        "memref.store"(%124, %156, %arg21, %133) : (f32, memref<16x3xf32>, index, index) -> ()
        "memref.store"(%124, %156, %arg21, %132) : (f32, memref<16x3xf32>, index, index) -> ()
        "memref.store"(%125, %155, %arg21, %131) : (f32, memref<16x3xf32>, index, index) -> ()
        "memref.store"(%125, %155, %arg21, %133) : (f32, memref<16x3xf32>, index, index) -> ()
        "memref.store"(%125, %155, %arg21, %132) : (f32, memref<16x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%131, %120, %133) ({
      ^bb0(%arg20: index):
        "memref.store"(%124, %154, %arg20, %131) : (f32, memref<20x2xf32>, index, index) -> ()
        "memref.store"(%124, %154, %arg20, %133) : (f32, memref<20x2xf32>, index, index) -> ()
        "memref.store"(%125, %153, %arg20, %131) : (f32, memref<20x2xf32>, index, index) -> ()
        "memref.store"(%125, %153, %arg20, %133) : (f32, memref<20x2xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%131, %119, %133) ({
      ^bb0(%arg19: index):
        %165 = "memref.load"(%150, %arg19) : (memref<64xi32>, index) -> i32
        %166 = "memref.load"(%149, %arg19) : (memref<64xi32>, index) -> i32
        %167 = "arith.index_cast"(%166) : (i32) -> index
        %168 = "memref.load"(%151, %167, %131) : (memref<20x2xf32>, index, index) -> f32
        %169 = "arith.index_cast"(%165) : (i32) -> index
        %170 = "memref.load"(%152, %169, %131) : (memref<16x3xf32>, index, index) -> f32
        %171 = "arith.subf"(%168, %170) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %172 = "memref.load"(%151, %167, %133) : (memref<20x2xf32>, index, index) -> f32
        %173 = "memref.load"(%152, %169, %133) : (memref<16x3xf32>, index, index) -> f32
        %174 = "arith.subf"(%172, %173) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %175 = "memref.load"(%152, %169, %132) : (memref<16x3xf32>, index, index) -> f32
        %176 = "arith.mulf"(%175, %174) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %177 = "arith.addf"(%171, %176) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %178 = "arith.mulf"(%175, %171) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %179 = "arith.subf"(%174, %178) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %180 = "memref.load"(%148, %arg19, %131) : (memref<64x2xf32>, index, index) -> f32
        %181 = "arith.subf"(%177, %180) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %182 = "memref.load"(%148, %arg19, %133) : (memref<64x2xf32>, index, index) -> f32
        %183 = "arith.subf"(%179, %182) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %184 = "arith.mulf"(%181, %126) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %185 = "arith.mulf"(%175, %127) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %186 = "arith.mulf"(%185, %183) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %187 = "arith.addf"(%184, %186) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %188 = "memref.load"(%156, %169, %131) : (memref<16x3xf32>, index, index) -> f32
        %189 = "arith.addf"(%188, %187) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%189, %156, %169, %131) : (f32, memref<16x3xf32>, index, index) -> ()
        %190 = "arith.mulf"(%183, %126) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %191 = "arith.mulf"(%185, %181) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %192 = "arith.subf"(%190, %191) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %193 = "memref.load"(%156, %169, %133) : (memref<16x3xf32>, index, index) -> f32
        %194 = "arith.addf"(%193, %192) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%194, %156, %169, %133) : (f32, memref<16x3xf32>, index, index) -> ()
        %195 = "arith.mulf"(%174, %181) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %196 = "arith.mulf"(%171, %183) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %197 = "arith.subf"(%195, %196) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %198 = "memref.load"(%156, %169, %132) : (memref<16x3xf32>, index, index) -> f32
        %199 = "arith.addf"(%198, %197) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%199, %156, %169, %132) : (f32, memref<16x3xf32>, index, index) -> ()
        %200 = "memref.load"(%155, %169, %131) : (memref<16x3xf32>, index, index) -> f32
        %201 = "arith.addf"(%200, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%201, %155, %169, %131) : (f32, memref<16x3xf32>, index, index) -> ()
        %202 = "memref.load"(%155, %169, %133) : (memref<16x3xf32>, index, index) -> f32
        %203 = "arith.addf"(%202, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%203, %155, %169, %133) : (f32, memref<16x3xf32>, index, index) -> ()
        %204 = "arith.mulf"(%171, %171) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %205 = "arith.mulf"(%174, %174) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %206 = "arith.addf"(%204, %205) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %207 = "arith.mulf"(%206, %129) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %208 = "arith.addf"(%207, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %209 = "memref.load"(%155, %169, %132) : (memref<16x3xf32>, index, index) -> f32
        %210 = "arith.addf"(%209, %208) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%210, %155, %169, %132) : (f32, memref<16x3xf32>, index, index) -> ()
        %211 = "arith.mulf"(%175, %130) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %212 = "arith.mulf"(%211, %183) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %213 = "arith.subf"(%181, %212) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %214 = "memref.load"(%154, %167, %131) : (memref<20x2xf32>, index, index) -> f32
        %215 = "arith.addf"(%214, %213) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%215, %154, %167, %131) : (f32, memref<20x2xf32>, index, index) -> ()
        %216 = "arith.mulf"(%211, %181) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %217 = "arith.addf"(%183, %216) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %218 = "memref.load"(%154, %167, %133) : (memref<20x2xf32>, index, index) -> f32
        %219 = "arith.addf"(%218, %217) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%219, %154, %167, %133) : (f32, memref<20x2xf32>, index, index) -> ()
        %220 = "memref.load"(%153, %167, %131) : (memref<20x2xf32>, index, index) -> f32
        %221 = "arith.addf"(%220, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%221, %153, %167, %131) : (f32, memref<20x2xf32>, index, index) -> ()
        %222 = "memref.load"(%153, %167, %133) : (memref<20x2xf32>, index, index) -> f32
        %223 = "arith.addf"(%222, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%223, %153, %167, %133) : (f32, memref<20x2xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "memref.store"(%124, %152, %131, %131) : (f32, memref<16x3xf32>, index, index) -> ()
      "memref.store"(%124, %152, %131, %133) : (f32, memref<16x3xf32>, index, index) -> ()
      "memref.store"(%124, %152, %131, %132) : (f32, memref<16x3xf32>, index, index) -> ()
      "func.call"(%156, %155, %152) <{callee = @au_generated_candidate_12__direct_site_3763}> : (memref<16x3xf32>, memref<16x3xf32>, memref<16x3xf32>) -> ()
      "func.call"(%154, %153, %151) <{callee = @au_generated_candidate_41__direct_site_3782}> : (memref<20x2xf32>, memref<20x2xf32>, memref<20x2xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %157 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%122, %157) : (i32, memref<i32>) -> ()
    "memref.store"(%122, %157) : (i32, memref<i32>) -> ()
    "func.call"(%152, %157) <{callee = @au_generated_candidate_21__direct_site_3810}> : (memref<16x3xf32>, memref<i32>) -> ()
    "func.call"(%151, %157) <{callee = @au_generated_candidate_51__direct_site_3826}> : (memref<20x2xf32>, memref<i32>) -> ()
    %158 = "memref.load"(%157) : (memref<i32>) -> i32
    %159 = "arith.sitofp"(%158) : (i32) -> f32
    %160 = "arith.cmpi"(%158, %122) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %161 = "scf.if"(%160) ({
      %164 = "arith.negf"(%159) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "scf.yield"(%164) : (f32) -> ()
    }, {
      "scf.yield"(%159) : (f32) -> ()
    }) : (i1) -> f32
    %162 = "arith.cmpf"(%161, %124) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
    %163 = "arith.select"(%162, %158, %123) : (i1, i32, i32) -> i32
    "func.call"(%152, %151, %150, %149, %148) <{callee = @pcl_validate_outputs}> : (memref<16x3xf32>, memref<20x2xf32>, memref<64xi32>, memref<64xi32>, memref<64x2xf32>) -> ()
    "func.return"(%163) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<16x3xf32>) -> (), sym_name = "au_generated_candidate_34__l0u_site_3539"}> ({
  ^bb0(%arg16: memref<16x3xf32>):
    %92 = "arith.constant"() <{value = 2 : index}> : () -> index
    %93 = "arith.constant"() <{value = 1.500000e-02 : f32}> : () -> f32
    %94 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %95 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %96 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %97 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %98 = "arith.constant"() <{value = 5.000000e-02 : f32}> : () -> f32
    %99 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %100 = "arith.constant"() <{value = 7.500000e-01 : f32}> : () -> f32
    %101 = "arith.constant"() <{value = 0 : index}> : () -> index
    %102 = "arith.constant"() <{value = 16 : index}> : () -> index
    %103 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%101, %102, %103) ({
    ^bb0(%arg17: index):
      %104 = "arith.index_cast"(%arg17) : (index) -> i32
      %105 = "arith.sitofp"(%104) : (i32) -> f32
      %106 = "arith.mulf"(%105, %100) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %107 = "arith.remsi"(%104, %99) : (i32, i32) -> i32
      %108 = "arith.sitofp"(%107) : (i32) -> f32
      %109 = "arith.mulf"(%108, %98) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %110 = "arith.addf"(%106, %109) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%110, %arg16, %arg17, %101) : (f32, memref<16x3xf32>, index, index) -> ()
      %111 = "arith.remsi"(%104, %97) : (i32, i32) -> i32
      %112 = "arith.sitofp"(%111) : (i32) -> f32
      %113 = "arith.mulf"(%112, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%113, %arg16, %arg17, %103) : (f32, memref<16x3xf32>, index, index) -> ()
      %114 = "arith.remsi"(%104, %95) : (i32, i32) -> i32
      %115 = "arith.addi"(%114, %94) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %116 = "arith.sitofp"(%115) : (i32) -> f32
      %117 = "arith.mulf"(%116, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%117, %arg16, %arg17, %92) : (f32, memref<16x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<20x2xf32>) -> (), sym_name = "au_generated_candidate_56__l0u_site_3555"}> ({
  ^bb0(%arg14: memref<20x2xf32>):
    %75 = "arith.constant"() <{value = 1.200000e+00 : f32}> : () -> f32
    %76 = "arith.constant"() <{value = 2.700000e-01 : f32}> : () -> f32
    %77 = "arith.constant"() <{value = 1.600000e+00 : f32}> : () -> f32
    %78 = "arith.constant"() <{value = 3.800000e-01 : f32}> : () -> f32
    %79 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %80 = "arith.constant"() <{value = 0 : index}> : () -> index
    %81 = "arith.constant"() <{value = 20 : index}> : () -> index
    %82 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%80, %81, %82) ({
    ^bb0(%arg15: index):
      %83 = "arith.index_cast"(%arg15) : (index) -> i32
      %84 = "arith.remsi"(%83, %79) : (i32, i32) -> i32
      %85 = "arith.sitofp"(%84) : (i32) -> f32
      %86 = "arith.mulf"(%85, %78) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %87 = "arith.addf"(%86, %77) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%87, %arg14, %arg15, %80) : (f32, memref<20x2xf32>, index, index) -> ()
      %88 = "arith.divsi"(%83, %79) : (i32, i32) -> i32
      %89 = "arith.sitofp"(%88) : (i32) -> f32
      %90 = "arith.mulf"(%89, %76) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %91 = "arith.addf"(%90, %75) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%91, %arg14, %arg15, %82) : (f32, memref<20x2xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<16x3xf32>, memref<16x3xf32>, memref<16x3xf32>) -> (), sym_name = "au_generated_candidate_12__direct_site_3763"}> ({
  ^bb0(%arg10: memref<16x3xf32>, %arg11: memref<16x3xf32>, %arg12: memref<16x3xf32>):
    %50 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %51 = "arith.constant"() <{value = 2 : index}> : () -> index
    %52 = "arith.constant"() <{value = 2.200000e-01 : f32}> : () -> f32
    %53 = "arith.constant"() <{value = 0 : index}> : () -> index
    %54 = "arith.constant"() <{value = 15 : index}> : () -> index
    %55 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%53, %54, %55) ({
    ^bb0(%arg13: index):
      %56 = "arith.addi"(%arg13, %55) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %57 = "memref.load"(%arg10, %56, %53) : (memref<16x3xf32>, index, index) -> f32
      %58 = "arith.mulf"(%57, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %59 = "memref.load"(%arg11, %56, %53) : (memref<16x3xf32>, index, index) -> f32
      %60 = "arith.divf"(%58, %59) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %61 = "memref.load"(%arg12, %56, %53) : (memref<16x3xf32>, index, index) -> f32
      %62 = "arith.subf"(%61, %60) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%62, %arg12, %56, %53) : (f32, memref<16x3xf32>, index, index) -> ()
      %63 = "memref.load"(%arg10, %56, %55) : (memref<16x3xf32>, index, index) -> f32
      %64 = "arith.mulf"(%63, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %65 = "memref.load"(%arg11, %56, %55) : (memref<16x3xf32>, index, index) -> f32
      %66 = "arith.divf"(%64, %65) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %67 = "memref.load"(%arg12, %56, %55) : (memref<16x3xf32>, index, index) -> f32
      %68 = "arith.subf"(%67, %66) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%68, %arg12, %56, %55) : (f32, memref<16x3xf32>, index, index) -> ()
      %69 = "memref.load"(%arg10, %56, %51) : (memref<16x3xf32>, index, index) -> f32
      %70 = "arith.mulf"(%69, %50) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %71 = "memref.load"(%arg11, %56, %51) : (memref<16x3xf32>, index, index) -> f32
      %72 = "arith.divf"(%70, %71) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %73 = "memref.load"(%arg12, %56, %51) : (memref<16x3xf32>, index, index) -> f32
      %74 = "arith.subf"(%73, %72) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%74, %arg12, %56, %51) : (f32, memref<16x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<20x2xf32>, memref<20x2xf32>, memref<20x2xf32>) -> (), sym_name = "au_generated_candidate_41__direct_site_3782"}> ({
  ^bb0(%arg6: memref<20x2xf32>, %arg7: memref<20x2xf32>, %arg8: memref<20x2xf32>):
    %34 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
    %35 = "arith.constant"() <{value = 0 : index}> : () -> index
    %36 = "arith.constant"() <{value = 20 : index}> : () -> index
    %37 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%35, %36, %37) ({
    ^bb0(%arg9: index):
      %38 = "memref.load"(%arg6, %arg9, %35) : (memref<20x2xf32>, index, index) -> f32
      %39 = "arith.mulf"(%38, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %40 = "memref.load"(%arg7, %arg9, %35) : (memref<20x2xf32>, index, index) -> f32
      %41 = "arith.divf"(%39, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %42 = "memref.load"(%arg8, %arg9, %35) : (memref<20x2xf32>, index, index) -> f32
      %43 = "arith.subf"(%42, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%43, %arg8, %arg9, %35) : (f32, memref<20x2xf32>, index, index) -> ()
      %44 = "memref.load"(%arg6, %arg9, %37) : (memref<20x2xf32>, index, index) -> f32
      %45 = "arith.mulf"(%44, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %46 = "memref.load"(%arg7, %arg9, %37) : (memref<20x2xf32>, index, index) -> f32
      %47 = "arith.divf"(%45, %46) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %48 = "memref.load"(%arg8, %arg9, %37) : (memref<20x2xf32>, index, index) -> f32
      %49 = "arith.subf"(%48, %47) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%49, %arg8, %arg9, %37) : (f32, memref<20x2xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<16x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_21__direct_site_3810"}> ({
  ^bb0(%arg3: memref<16x3xf32>, %arg4: memref<i32>):
    %14 = "arith.constant"() <{value = 1.970000e+02 : f32}> : () -> f32
    %15 = "arith.constant"() <{value = 2 : index}> : () -> index
    %16 = "arith.constant"() <{value = 1.310000e+02 : f32}> : () -> f32
    %17 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
    %18 = "arith.constant"() <{value = 0 : index}> : () -> index
    %19 = "arith.constant"() <{value = 16 : index}> : () -> index
    %20 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%18, %19, %20) ({
    ^bb0(%arg5: index):
      %21 = "memref.load"(%arg3, %arg5, %18) : (memref<16x3xf32>, index, index) -> f32
      %22 = "arith.mulf"(%21, %17) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %23 = "arith.fptosi"(%22) : (f32) -> i32
      %24 = "memref.load"(%arg4) : (memref<i32>) -> i32
      %25 = "arith.addi"(%24, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%25, %arg4) : (i32, memref<i32>) -> ()
      %26 = "memref.load"(%arg3, %arg5, %20) : (memref<16x3xf32>, index, index) -> f32
      %27 = "arith.mulf"(%26, %16) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %28 = "arith.fptosi"(%27) : (f32) -> i32
      %29 = "arith.addi"(%25, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%29, %arg4) : (i32, memref<i32>) -> ()
      %30 = "memref.load"(%arg3, %arg5, %15) : (memref<16x3xf32>, index, index) -> f32
      %31 = "arith.mulf"(%30, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %32 = "arith.fptosi"(%31) : (f32) -> i32
      %33 = "arith.addi"(%29, %32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%33, %arg4) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<20x2xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_51__direct_site_3826"}> ({
  ^bb0(%arg0: memref<20x2xf32>, %arg1: memref<i32>):
    %0 = "arith.constant"() <{value = 1.810000e+02 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 1.510000e+02 : f32}> : () -> f32
    %2 = "arith.constant"() <{value = 0 : index}> : () -> index
    %3 = "arith.constant"() <{value = 20 : index}> : () -> index
    %4 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%2, %3, %4) ({
    ^bb0(%arg2: index):
      %5 = "memref.load"(%arg0, %arg2, %2) : (memref<20x2xf32>, index, index) -> f32
      %6 = "arith.mulf"(%5, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %7 = "arith.fptosi"(%6) : (f32) -> i32
      %8 = "memref.load"(%arg1) : (memref<i32>) -> i32
      %9 = "arith.addi"(%8, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%9, %arg1) : (i32, memref<i32>) -> ()
      %10 = "memref.load"(%arg0, %arg2, %4) : (memref<20x2xf32>, index, index) -> f32
      %11 = "arith.mulf"(%10, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %12 = "arith.fptosi"(%11) : (f32) -> i32
      %13 = "arith.addi"(%9, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%13, %arg1) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<16x3xf32>, memref<20x2xf32>, memref<64xi32>, memref<64xi32>, memref<64x2xf32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
