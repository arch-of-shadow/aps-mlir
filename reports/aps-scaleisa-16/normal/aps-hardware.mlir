#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (3)>
#map2 = affine_map<() -> (12)>
#map3 = affine_map<() -> (10)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_24__direct_buffer_4", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_24__direct_buffer_4"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_24__direct_buffer_3", type = memref<144xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_24__direct_buffer_3"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_24__direct_buffer", type = memref<432xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_24__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_12__direct_buffer_2", type = memref<432xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_12__direct_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_12__direct_buffer", type = memref<432xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_12__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_12__direct_local_1", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_12__direct_local_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_12__direct_local_0", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_12__direct_local_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_12__direct_local", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_12__direct_local"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_12__direct"}> ({
  ^bb0(%arg5: i5, %arg6: i5, %arg7: i5):
    %95 = "aps.read_irf"(%arg5) : (i5) -> i32
    %96 = "aps.read_irf"(%arg6) : (i5) -> i32
    %97 = "memref.get_global"() <{name = @au_generated_candidate_12__direct_buffer}> : () -> memref<432xf32>
    %98 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %99 = "arith.constant"() <{value = 432 : i32}> : () -> i32
    "aps.copy"(%96, %97, %98, %99) <{direction = 0 : i32}> : (i32, memref<432xf32>, i32, i32) -> ()
    %100 = "aps.read_csr"() <{global_name = @au_generated_candidate_12__direct_csr0}> : () -> i32
    %101 = "memref.get_global"() <{name = @au_generated_candidate_12__direct_buffer_2}> : () -> memref<432xf32>
    %102 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %103 = "arith.constant"() <{value = 432 : i32}> : () -> i32
    "aps.copy"(%100, %101, %102, %103) <{direction = 0 : i32}> : (i32, memref<432xf32>, i32, i32) -> ()
    %104 = "aps.read_csr"() <{global_name = @au_generated_candidate_12__direct_csr1}> : () -> i32
    %105 = "arith.index_castui"(%104) : (i32) -> index
    %106 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %107 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %108 = "arith.constant"() <{value = 9.000000e+00 : f32}> : () -> f32
    %109 = "arith.constant"() <{value = 2 : index}> : () -> index
    %110 = "arith.constant"() <{value = 1 : index}> : () -> index
    %111 = "arith.constant"() <{value = 0 : index}> : () -> index
    %112 = "arith.constant"() <{value = true}> : () -> i1
    %113 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %114 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %115 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %116 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg8: index):
      %117 = "arith.index_cast"(%arg8) : (index) -> i32
      %118 = "memref.get_global"() <{name = @au_generated_candidate_12__direct_local}> : () -> memref<f32>
      "aps.globalstore"(%116) <{global_name = @au_generated_candidate_12__direct_local}> : (f32) -> ()
      %119 = "memref.get_global"() <{name = @au_generated_candidate_12__direct_local_0}> : () -> memref<f32>
      "aps.globalstore"(%116) <{global_name = @au_generated_candidate_12__direct_local_0}> : (f32) -> ()
      %120 = "memref.get_global"() <{name = @au_generated_candidate_12__direct_local_1}> : () -> memref<f32>
      "aps.globalstore"(%116) <{global_name = @au_generated_candidate_12__direct_local_1}> : (f32) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg9: index):
        %163 = "arith.index_cast"(%arg9) : (index) -> i32
        %164 = "arith.addi"(%163, %115) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg10: index):
          %165 = "arith.index_cast"(%arg10) : (index) -> i32
          %166 = "arith.addi"(%165, %115) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %167 = "arith.addi"(%95, %164) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %168 = "arith.cmpi"(%167, %114) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %169 = "arith.cmpi"(%167, %113) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %170 = "arith.andi"(%168, %169) : (i1, i1) -> i1
          %171 = "arith.xori"(%170, %112) : (i1, i1) -> i1
          %172 = "arith.andi"(%171, %168) : (i1, i1) -> i1
          %173 = "arith.select"(%170, %113, %114) : (i1, i32, i32) -> i32
          %174 = "arith.select"(%172, %167, %173) : (i1, i32, i32) -> i32
          %175 = "arith.addi"(%117, %166) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %176 = "arith.cmpi"(%175, %114) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %177 = "arith.cmpi"(%175, %113) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %178 = "arith.andi"(%176, %177) : (i1, i1) -> i1
          %179 = "arith.xori"(%178, %112) : (i1, i1) -> i1
          %180 = "arith.andi"(%179, %176) : (i1, i1) -> i1
          %181 = "arith.select"(%178, %113, %114) : (i1, i32, i32) -> i32
          %182 = "arith.select"(%180, %175, %181) : (i1, i32, i32) -> i32
          %183 = "arith.index_cast"(%174) : (i32) -> index
          %184 = "arith.index_cast"(%182) : (i32) -> index
          %185 = "arith.index_cast"(%183) : (index) -> i32
          %186 = "arith.index_cast"(%184) : (index) -> i32
          %187 = "arith.index_cast"(%111) : (index) -> i32
          %188 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %189 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %190 = "arith.constant"() <{value = 36 : i32}> : () -> i32
          %191 = "arith.muli"(%185, %190) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %192 = "arith.addi"(%189, %191) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %193 = "arith.constant"() <{value = 3 : i32}> : () -> i32
          %194 = "arith.muli"(%186, %193) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %195 = "arith.addi"(%192, %194) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %196 = "arith.addi"(%195, %187) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %197 = "arith.addi"(%188, %196) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %198 = "aps.read_smem"(%97, %197) : (memref<432xf32>, i32) -> f32
          %199 = "aps.globalload"() <{global_name = @au_generated_candidate_12__direct_local}> : () -> f32
          %200 = "arith.addf"(%199, %198) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "aps.globalstore"(%200) <{global_name = @au_generated_candidate_12__direct_local}> : (f32) -> ()
          %201 = "arith.index_cast"(%183) : (index) -> i32
          %202 = "arith.index_cast"(%184) : (index) -> i32
          %203 = "arith.index_cast"(%110) : (index) -> i32
          %204 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %205 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %206 = "arith.constant"() <{value = 36 : i32}> : () -> i32
          %207 = "arith.muli"(%201, %206) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %208 = "arith.addi"(%205, %207) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %209 = "arith.constant"() <{value = 3 : i32}> : () -> i32
          %210 = "arith.muli"(%202, %209) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %211 = "arith.addi"(%208, %210) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %212 = "arith.addi"(%211, %203) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %213 = "arith.addi"(%204, %212) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %214 = "aps.read_smem"(%97, %213) : (memref<432xf32>, i32) -> f32
          %215 = "aps.globalload"() <{global_name = @au_generated_candidate_12__direct_local_0}> : () -> f32
          %216 = "arith.addf"(%215, %214) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "aps.globalstore"(%216) <{global_name = @au_generated_candidate_12__direct_local_0}> : (f32) -> ()
          %217 = "arith.index_cast"(%183) : (index) -> i32
          %218 = "arith.index_cast"(%184) : (index) -> i32
          %219 = "arith.index_cast"(%109) : (index) -> i32
          %220 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %221 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %222 = "arith.constant"() <{value = 36 : i32}> : () -> i32
          %223 = "arith.muli"(%217, %222) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %224 = "arith.addi"(%221, %223) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %225 = "arith.constant"() <{value = 3 : i32}> : () -> i32
          %226 = "arith.muli"(%218, %225) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %227 = "arith.addi"(%224, %226) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %228 = "arith.addi"(%227, %219) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %229 = "arith.addi"(%220, %228) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %230 = "aps.read_smem"(%97, %229) : (memref<432xf32>, i32) -> f32
          %231 = "aps.globalload"() <{global_name = @au_generated_candidate_12__direct_local_1}> : () -> f32
          %232 = "arith.addf"(%231, %230) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "aps.globalstore"(%232) <{global_name = @au_generated_candidate_12__direct_local_1}> : (f32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %121 = "aps.globalload"() <{global_name = @au_generated_candidate_12__direct_local}> : () -> f32
      %122 = "arith.divf"(%121, %108) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %123 = "arith.index_cast"(%105) : (index) -> i32
      %124 = "arith.index_cast"(%arg8) : (index) -> i32
      %125 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %126 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %127 = "arith.constant"() <{value = 36 : i32}> : () -> i32
      %128 = "arith.muli"(%123, %127) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %129 = "arith.addi"(%126, %128) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %130 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %131 = "arith.muli"(%124, %130) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %132 = "arith.addi"(%129, %131) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %133 = "arith.addi"(%132, %114) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %134 = "arith.addi"(%125, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%122, %101, %134) : (f32, memref<432xf32>, i32) -> ()
      %135 = "aps.globalload"() <{global_name = @au_generated_candidate_12__direct_local_0}> : () -> f32
      %136 = "arith.divf"(%135, %108) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %137 = "arith.index_cast"(%105) : (index) -> i32
      %138 = "arith.index_cast"(%arg8) : (index) -> i32
      %139 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %140 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %141 = "arith.constant"() <{value = 36 : i32}> : () -> i32
      %142 = "arith.muli"(%137, %141) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %143 = "arith.addi"(%140, %142) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %144 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %145 = "arith.muli"(%138, %144) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %146 = "arith.addi"(%143, %145) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %147 = "arith.addi"(%146, %107) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %148 = "arith.addi"(%139, %147) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%136, %101, %148) : (f32, memref<432xf32>, i32) -> ()
      %149 = "aps.globalload"() <{global_name = @au_generated_candidate_12__direct_local_1}> : () -> f32
      %150 = "arith.divf"(%149, %108) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %151 = "arith.index_cast"(%105) : (index) -> i32
      %152 = "arith.index_cast"(%arg8) : (index) -> i32
      %153 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %154 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %155 = "arith.constant"() <{value = 36 : i32}> : () -> i32
      %156 = "arith.muli"(%151, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %157 = "arith.addi"(%154, %156) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %158 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %159 = "arith.muli"(%152, %158) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %160 = "arith.addi"(%157, %159) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %161 = "arith.addi"(%160, %106) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %162 = "arith.addi"(%153, %161) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%150, %101, %162) : (f32, memref<432xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%100, %101, %102, %103) <{direction = 1 : i32}> : (i32, memref<432xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<12x12x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<12x12x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2050 : i32, arg_index = 3 : i32, kind = "csr"}], scaleisa.source_type = (i32, memref<12x12x3xf32>, memref<12x12x3xf32>, index) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_24__direct"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_24__direct_buffer}> : () -> memref<432xf32>
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 432 : i32}> : () -> i32
    "aps.copy"(%0, %2, %3, %4) <{direction = 0 : i32}> : (i32, memref<432xf32>, i32, i32) -> ()
    %5 = "memref.get_global"() <{name = @au_generated_candidate_24__direct_buffer_3}> : () -> memref<144xf32>
    %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 144 : i32}> : () -> i32
    "aps.copy"(%1, %5, %6, %7) <{direction = 0 : i32}> : (i32, memref<144xf32>, i32, i32) -> ()
    %8 = "aps.read_csr"() <{global_name = @au_generated_candidate_24__direct_csr0}> : () -> i32
    %9 = "memref.get_global"() <{name = @au_generated_candidate_24__direct_buffer_4}> : () -> memref<1xi32>
    %10 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%8, %9, %10, %11) <{direction = 0 : i32}> : (i32, memref<1xi32>, i32, i32) -> ()
    %12 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %13 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 1 : index}> : () -> index
    %16 = "arith.constant"() <{value = 6.400000e+01 : f32}> : () -> f32
    %17 = "arith.constant"() <{value = 1.100000e+01 : f32}> : () -> f32
    %18 = "arith.constant"() <{value = 3.100000e+01 : f32}> : () -> f32
    %19 = "arith.constant"() <{value = 2.300000e+01 : f32}> : () -> f32
    %20 = "arith.constant"() <{value = 1.700000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg3: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg4: index):
        %21 = "arith.addi"(%arg3, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %22 = "arith.addi"(%arg4, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %23 = "arith.index_cast"(%21) : (index) -> i32
        %24 = "arith.index_cast"(%22) : (index) -> i32
        %25 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %26 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %27 = "arith.constant"() <{value = 36 : i32}> : () -> i32
        %28 = "arith.muli"(%23, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %29 = "arith.addi"(%26, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %30 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %31 = "arith.muli"(%24, %30) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %32 = "arith.addi"(%29, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %33 = "arith.addi"(%32, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %34 = "arith.addi"(%25, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %35 = "aps.read_smem"(%2, %34) : (memref<432xf32>, i32) -> f32
        %36 = "arith.mulf"(%35, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %37 = "arith.addi"(%arg3, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %38 = "arith.addi"(%arg4, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %39 = "arith.index_cast"(%37) : (index) -> i32
        %40 = "arith.index_cast"(%38) : (index) -> i32
        %41 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %42 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %43 = "arith.constant"() <{value = 36 : i32}> : () -> i32
        %44 = "arith.muli"(%39, %43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %45 = "arith.addi"(%42, %44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %46 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %47 = "arith.muli"(%40, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %48 = "arith.addi"(%45, %47) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %49 = "arith.addi"(%48, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %50 = "arith.addi"(%41, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %51 = "aps.read_smem"(%2, %50) : (memref<432xf32>, i32) -> f32
        %52 = "arith.mulf"(%51, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %53 = "arith.addf"(%36, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %54 = "arith.addi"(%arg3, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %55 = "arith.addi"(%arg4, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %56 = "arith.index_cast"(%54) : (index) -> i32
        %57 = "arith.index_cast"(%55) : (index) -> i32
        %58 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %59 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %60 = "arith.constant"() <{value = 36 : i32}> : () -> i32
        %61 = "arith.muli"(%56, %60) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %62 = "arith.addi"(%59, %61) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %63 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %64 = "arith.muli"(%57, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %65 = "arith.addi"(%62, %64) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %66 = "arith.addi"(%65, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %67 = "arith.addi"(%58, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %68 = "aps.read_smem"(%2, %67) : (memref<432xf32>, i32) -> f32
        %69 = "arith.mulf"(%68, %18) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %70 = "arith.addf"(%53, %69) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %71 = "arith.addi"(%arg3, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %72 = "arith.addi"(%arg4, %15) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %73 = "arith.index_cast"(%71) : (index) -> i32
        %74 = "arith.index_cast"(%72) : (index) -> i32
        %75 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %76 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %77 = "arith.constant"() <{value = 12 : i32}> : () -> i32
        %78 = "arith.muli"(%73, %77) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %79 = "arith.addi"(%76, %78) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %80 = "arith.addi"(%79, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %81 = "arith.addi"(%75, %80) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %82 = "aps.read_smem"(%5, %81) : (memref<144xf32>, i32) -> f32
        %83 = "arith.mulf"(%82, %17) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %84 = "arith.addf"(%70, %83) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %85 = "arith.mulf"(%84, %16) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %86 = "arith.fptosi"(%85) : (f32) -> i32
        %87 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %88 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %89 = "arith.addi"(%87, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %90 = "aps.read_smem"(%9, %89) : (memref<1xi32>, i32) -> i32
        %91 = "arith.addi"(%90, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %92 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %93 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %94 = "arith.addi"(%92, %93) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%91, %9, %94) : (i32, memref<1xi32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%8, %9, %10, %11) <{direction = 1 : i32}> : (i32, memref<1xi32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<12x12x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<12x12xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2051 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<12x12x3xf32>, memref<12x12xf32>, memref<i32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_12__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_12__direct_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_24__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2051 : i32} : () -> ()
}) : () -> ()

