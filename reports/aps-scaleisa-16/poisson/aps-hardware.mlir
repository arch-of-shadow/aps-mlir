#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (4)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_9__direct_buffer_3", type = memref<216xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_9__direct_buffer_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_9__direct_buffer_2", type = memref<216xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_9__direct_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_9__direct_buffer", type = memref<216xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_9__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_27__l0u_l1u_l2u_buffer_1", type = memref<216xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [4], scaleisa.scratchpad, var_name = "au_generated_candidate_27__l0u_l1u_l2u_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_27__l0u_l1u_l2u_buffer", type = memref<216xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [4], scaleisa.scratchpad, var_name = "au_generated_candidate_27__l0u_l1u_l2u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_11__direct_buffer_0", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_11__direct_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_11__direct_buffer", type = memref<216xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_11__direct_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_11__direct"}> ({
  ^bb0(%arg12: i5, %arg13: i5, %arg14: i5):
    %144 = "aps.read_irf"(%arg12) : (i5) -> i32
    %145 = "aps.read_irf"(%arg13) : (i5) -> i32
    %146 = "memref.get_global"() <{name = @au_generated_candidate_11__direct_buffer}> : () -> memref<216xf32>
    %147 = "arith.constant"() <{value = 216 : i32}> : () -> i32
    %148 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%144, %146, %148, %147) <{direction = 0 : i32}> : (i32, memref<216xf32>, i32, i32) -> ()
    %149 = "memref.get_global"() <{name = @au_generated_candidate_11__direct_buffer_0}> : () -> memref<1xi32>
    %150 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %151 = "arith.addi"(%145, %150) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %152 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %153 = "aps.load"(%151) : (i32) -> i32
    "aps.write_smem"(%153, %149, %152) : (i32, memref<1xi32>, i32) -> ()
    %154 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %155 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %156 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg15: index):
      %161 = "affine.apply"(%arg15) <{map = #map}> : (index) -> index
      %162 = "arith.index_cast"(%161) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg16: index):
        %163 = "affine.apply"(%arg16) <{map = #map}> : (index) -> index
        %164 = "arith.index_cast"(%163) : (index) -> i32
        "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg17: index):
          %165 = "affine.apply"(%arg17) <{map = #map}> : (index) -> index
          %166 = "arith.index_cast"(%165) : (index) -> i32
          %167 = "arith.muli"(%162, %156) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %168 = "arith.addi"(%167, %164) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %169 = "arith.muli"(%168, %156) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %170 = "arith.addi"(%169, %166) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %171 = "arith.addi"(%162, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %172 = "arith.muli"(%171, %156) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %173 = "arith.addi"(%172, %164) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %174 = "arith.muli"(%173, %156) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %175 = "arith.addi"(%174, %166) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %176 = "arith.addi"(%164, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %177 = "arith.addi"(%167, %176) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %178 = "arith.muli"(%177, %156) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %179 = "arith.addi"(%178, %166) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %180 = "arith.addi"(%166, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %181 = "arith.addi"(%169, %180) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %182 = "arith.index_cast"(%170) : (i32) -> index
          %183 = "arith.index_cast"(%182) : (index) -> i32
          %184 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %185 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %186 = "arith.addi"(%185, %183) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %187 = "arith.addi"(%184, %186) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %188 = "aps.read_smem"(%146, %187) : (memref<216xf32>, i32) -> f32
          %189 = "arith.index_cast"(%175) : (i32) -> index
          %190 = "arith.index_cast"(%189) : (index) -> i32
          %191 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %192 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %193 = "arith.addi"(%192, %190) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %194 = "arith.addi"(%191, %193) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %195 = "aps.read_smem"(%146, %194) : (memref<216xf32>, i32) -> f32
          %196 = "arith.mulf"(%188, %195) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %197 = "arith.cmpf"(%196, %154) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %198 = "arith.extui"(%197) : (i1) -> i32
          %199 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %200 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %201 = "arith.addi"(%199, %200) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %202 = "aps.read_smem"(%149, %201) : (memref<1xi32>, i32) -> i32
          %203 = "arith.addi"(%202, %198) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %204 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %205 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %206 = "arith.addi"(%204, %205) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%203, %149, %206) : (i32, memref<1xi32>, i32) -> ()
          %207 = "arith.index_cast"(%179) : (i32) -> index
          %208 = "arith.index_cast"(%207) : (index) -> i32
          %209 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %210 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %211 = "arith.addi"(%210, %208) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %212 = "arith.addi"(%209, %211) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %213 = "aps.read_smem"(%146, %212) : (memref<216xf32>, i32) -> f32
          %214 = "arith.mulf"(%188, %213) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %215 = "arith.cmpf"(%214, %154) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %216 = "arith.extui"(%215) : (i1) -> i32
          %217 = "arith.addi"(%203, %216) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %218 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %219 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %220 = "arith.addi"(%218, %219) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%217, %149, %220) : (i32, memref<1xi32>, i32) -> ()
          %221 = "arith.index_cast"(%181) : (i32) -> index
          %222 = "arith.index_cast"(%221) : (index) -> i32
          %223 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %224 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %225 = "arith.addi"(%224, %222) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %226 = "arith.addi"(%223, %225) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %227 = "aps.read_smem"(%146, %226) : (memref<216xf32>, i32) -> f32
          %228 = "arith.mulf"(%188, %227) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %229 = "arith.cmpf"(%228, %154) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %230 = "arith.extui"(%229) : (i1) -> i32
          %231 = "arith.addi"(%217, %230) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %232 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %233 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %234 = "arith.addi"(%232, %233) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%231, %149, %234) : (i32, memref<1xi32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %157 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %158 = "arith.addi"(%145, %157) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %159 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %160 = "aps.read_smem"(%149, %159) : (memref<1xi32>, i32) -> i32
    "aps.store"(%160, %158) : (i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<216xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<216xf32>, memref<i32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_27__l0u_l1u_l2u"}> ({
  ^bb0(%arg6: i5, %arg7: i5, %arg8: i5):
    %111 = "aps.read_irf"(%arg6) : (i5) -> i32
    %112 = "aps.read_irf"(%arg7) : (i5) -> i32
    %113 = "memref.get_global"() <{name = @au_generated_candidate_27__l0u_l1u_l2u_buffer}> : () -> memref<216xf32>
    %114 = "arith.constant"() <{value = 216 : i32}> : () -> i32
    %115 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%111, %113, %115, %114) <{direction = 0 : i32}> : (i32, memref<216xf32>, i32, i32) -> ()
    %116 = "memref.get_global"() <{name = @au_generated_candidate_27__l0u_l1u_l2u_buffer_1}> : () -> memref<216xf32>
    %117 = "arith.constant"() <{value = 216 : i32}> : () -> i32
    %118 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%112, %116, %118, %117) <{direction = 0 : i32}> : (i32, memref<216xf32>, i32, i32) -> ()
    %119 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg9: index):
      %122 = "affine.apply"(%arg9) <{map = #map}> : (index) -> index
      %123 = "arith.index_cast"(%122) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg10: index):
        %124 = "affine.apply"(%arg10) <{map = #map}> : (index) -> index
        %125 = "arith.index_cast"(%124) : (index) -> i32
        "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg11: index):
          %126 = "affine.apply"(%arg11) <{map = #map}> : (index) -> index
          %127 = "arith.index_cast"(%126) : (index) -> i32
          %128 = "arith.muli"(%123, %119) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %129 = "arith.addi"(%128, %125) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %130 = "arith.muli"(%129, %119) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %131 = "arith.addi"(%130, %127) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %132 = "arith.index_cast"(%131) : (i32) -> index
          %133 = "arith.index_cast"(%132) : (index) -> i32
          %134 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %135 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %136 = "arith.addi"(%135, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %137 = "arith.addi"(%134, %136) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %138 = "aps.read_smem"(%113, %137) : (memref<216xf32>, i32) -> f32
          %139 = "arith.index_cast"(%132) : (index) -> i32
          %140 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %141 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %142 = "arith.addi"(%141, %139) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %143 = "arith.addi"(%140, %142) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%138, %116, %143) : (f32, memref<216xf32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) {unroll = 0 : i64} : () -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %120 = "arith.constant"() <{value = 216 : i32}> : () -> i32
    %121 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%112, %116, %121, %120) <{direction = 1 : i32}> : (i32, memref<216xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<216xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<216xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<216xf32>, memref<216xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_9__direct"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_9__direct_buffer}> : () -> memref<216xf32>
    %3 = "arith.constant"() <{value = 216 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %2, %4, %3) <{direction = 0 : i32}> : (i32, memref<216xf32>, i32, i32) -> ()
    %5 = "memref.get_global"() <{name = @au_generated_candidate_9__direct_buffer_2}> : () -> memref<216xf32>
    %6 = "arith.constant"() <{value = 216 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%1, %5, %7, %6) <{direction = 0 : i32}> : (i32, memref<216xf32>, i32, i32) -> ()
    %8 = "aps.read_csr"() <{global_name = @au_generated_candidate_9__direct_csr0}> : () -> i32
    %9 = "memref.get_global"() <{name = @au_generated_candidate_9__direct_buffer_3}> : () -> memref<216xf32>
    %10 = "arith.constant"() <{value = 216 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%8, %9, %11, %10) <{direction = 0 : i32}> : (i32, memref<216xf32>, i32, i32) -> ()
    %12 = "arith.constant"() <{value = 6.000000e+00 : f32}> : () -> f32
    %13 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg3: index):
      %18 = "affine.apply"(%arg3) <{map = #map}> : (index) -> index
      %19 = "arith.index_cast"(%18) : (index) -> i32
      %20 = "arith.muli"(%19, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %21 = "arith.addi"(%19, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %22 = "arith.muli"(%21, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %23 = "arith.addi"(%19, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %24 = "arith.muli"(%23, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg4: index):
        %25 = "affine.apply"(%arg4) <{map = #map}> : (index) -> index
        %26 = "arith.index_cast"(%25) : (index) -> i32
        "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg5: index):
          %27 = "affine.apply"(%arg5) <{map = #map}> : (index) -> index
          %28 = "arith.index_cast"(%27) : (index) -> i32
          %29 = "arith.addi"(%20, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %30 = "arith.muli"(%29, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %31 = "arith.addi"(%30, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %32 = "arith.addi"(%22, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %33 = "arith.muli"(%32, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %34 = "arith.addi"(%33, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %35 = "arith.addi"(%24, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %36 = "arith.muli"(%35, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %37 = "arith.addi"(%36, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %38 = "arith.addi"(%26, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %39 = "arith.addi"(%20, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %40 = "arith.muli"(%39, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %41 = "arith.addi"(%40, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %42 = "arith.addi"(%26, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %43 = "arith.addi"(%20, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %44 = "arith.muli"(%43, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %45 = "arith.addi"(%44, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %46 = "arith.addi"(%28, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %47 = "arith.addi"(%30, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %48 = "arith.addi"(%28, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %49 = "arith.addi"(%30, %48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %50 = "arith.index_cast"(%34) : (i32) -> index
          %51 = "arith.index_cast"(%50) : (index) -> i32
          %52 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %53 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %54 = "arith.addi"(%53, %51) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %55 = "arith.addi"(%52, %54) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %56 = "aps.read_smem"(%2, %55) : (memref<216xf32>, i32) -> f32
          %57 = "arith.index_cast"(%37) : (i32) -> index
          %58 = "arith.index_cast"(%57) : (index) -> i32
          %59 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %60 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %61 = "arith.addi"(%60, %58) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %62 = "arith.addi"(%59, %61) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %63 = "aps.read_smem"(%2, %62) : (memref<216xf32>, i32) -> f32
          %64 = "arith.addf"(%56, %63) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %65 = "arith.index_cast"(%41) : (i32) -> index
          %66 = "arith.index_cast"(%65) : (index) -> i32
          %67 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %68 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %69 = "arith.addi"(%68, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %70 = "arith.addi"(%67, %69) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %71 = "aps.read_smem"(%2, %70) : (memref<216xf32>, i32) -> f32
          %72 = "arith.addf"(%64, %71) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %73 = "arith.index_cast"(%45) : (i32) -> index
          %74 = "arith.index_cast"(%73) : (index) -> i32
          %75 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %76 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %77 = "arith.addi"(%76, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %78 = "arith.addi"(%75, %77) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %79 = "aps.read_smem"(%2, %78) : (memref<216xf32>, i32) -> f32
          %80 = "arith.addf"(%72, %79) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %81 = "arith.index_cast"(%47) : (i32) -> index
          %82 = "arith.index_cast"(%81) : (index) -> i32
          %83 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %84 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %85 = "arith.addi"(%84, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %86 = "arith.addi"(%83, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %87 = "aps.read_smem"(%2, %86) : (memref<216xf32>, i32) -> f32
          %88 = "arith.addf"(%80, %87) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %89 = "arith.index_cast"(%49) : (i32) -> index
          %90 = "arith.index_cast"(%89) : (index) -> i32
          %91 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %92 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %93 = "arith.addi"(%92, %90) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %94 = "arith.addi"(%91, %93) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %95 = "aps.read_smem"(%2, %94) : (memref<216xf32>, i32) -> f32
          %96 = "arith.addf"(%88, %95) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %97 = "arith.index_cast"(%31) : (i32) -> index
          %98 = "arith.index_cast"(%97) : (index) -> i32
          %99 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %100 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %101 = "arith.addi"(%100, %98) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %102 = "arith.addi"(%99, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %103 = "aps.read_smem"(%5, %102) : (memref<216xf32>, i32) -> f32
          %104 = "arith.addf"(%103, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %105 = "arith.divf"(%104, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %106 = "arith.index_cast"(%97) : (index) -> i32
          %107 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %108 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %109 = "arith.addi"(%108, %106) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %110 = "arith.addi"(%107, %109) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "aps.write_smem"(%105, %9, %110) : (f32, memref<216xf32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %16 = "arith.constant"() <{value = 216 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%8, %9, %17, %16) <{direction = 1 : i32}> : (i32, memref<216xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 2 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<216xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<216xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<216xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<216xf32>, memref<216xf32>, memref<216xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_9__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
}) : () -> ()

