#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (32)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_26__direct_buffer_2", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_26__direct_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_26__direct_buffer", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_26__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_17__direct_buffer_1", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_17__direct_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_17__direct_buffer_0", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_17__direct_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_17__direct_buffer", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_17__direct_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_17__direct"}> ({
  ^bb0(%arg5: i5, %arg6: i5, %arg7: i5):
    %69 = "aps.read_irf"(%arg5) : (i5) -> i32
    %70 = "aps.read_irf"(%arg6) : (i5) -> i32
    %71 = "memref.get_global"() <{name = @au_generated_candidate_17__direct_buffer}> : () -> memref<96xf32>
    %72 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    %73 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%69, %71, %73, %72) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %74 = "memref.get_global"() <{name = @au_generated_candidate_17__direct_buffer_0}> : () -> memref<96xf32>
    %75 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    %76 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%70, %74, %76, %75) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %77 = "aps.read_csr"() <{global_name = @au_generated_candidate_17__direct_csr0}> : () -> i32
    %78 = "memref.get_global"() <{name = @au_generated_candidate_17__direct_buffer_1}> : () -> memref<96xf32>
    %79 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    %80 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%77, %78, %80, %79) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %81 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %82 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %83 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %84 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %85 = "arith.constant"() <{value = 1.600000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg8: index):
      %88 = "arith.index_cast"(%arg8) : (index) -> i32
      %89 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %90 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %91 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %92 = "arith.muli"(%88, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %93 = "arith.addi"(%90, %92) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %94 = "arith.addi"(%93, %83) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %95 = "arith.addi"(%89, %94) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %96 = "aps.read_smem"(%71, %95) : (memref<96xf32>, i32) -> f32
      %97 = "arith.mulf"(%96, %85) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %98 = "arith.index_cast"(%arg8) : (index) -> i32
      %99 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %100 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %101 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %102 = "arith.muli"(%98, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %103 = "arith.addi"(%100, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %104 = "arith.addi"(%103, %83) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %105 = "arith.addi"(%99, %104) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %106 = "aps.read_smem"(%74, %105) : (memref<96xf32>, i32) -> f32
      %107 = "arith.divf"(%97, %106) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %108 = "arith.index_cast"(%arg8) : (index) -> i32
      %109 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %110 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %111 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %112 = "arith.muli"(%108, %111) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %113 = "arith.addi"(%110, %112) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %114 = "arith.addi"(%113, %83) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %115 = "arith.addi"(%109, %114) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %116 = "aps.read_smem"(%78, %115) : (memref<96xf32>, i32) -> f32
      %117 = "arith.subf"(%116, %107) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %118 = "arith.index_cast"(%arg8) : (index) -> i32
      %119 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %120 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %121 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %122 = "arith.muli"(%118, %121) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %123 = "arith.addi"(%120, %122) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %124 = "arith.addi"(%123, %83) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %125 = "arith.addi"(%119, %124) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%117, %78, %125) : (f32, memref<96xf32>, i32) -> ()
      %126 = "arith.index_cast"(%arg8) : (index) -> i32
      %127 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %128 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %129 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %130 = "arith.muli"(%126, %129) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %131 = "arith.addi"(%128, %130) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %132 = "arith.addi"(%131, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %133 = "arith.addi"(%127, %132) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %134 = "aps.read_smem"(%71, %133) : (memref<96xf32>, i32) -> f32
      %135 = "arith.mulf"(%134, %85) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %136 = "arith.index_cast"(%arg8) : (index) -> i32
      %137 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %138 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %139 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %140 = "arith.muli"(%136, %139) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %141 = "arith.addi"(%138, %140) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %142 = "arith.addi"(%141, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %143 = "arith.addi"(%137, %142) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %144 = "aps.read_smem"(%74, %143) : (memref<96xf32>, i32) -> f32
      %145 = "arith.divf"(%135, %144) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %146 = "arith.index_cast"(%arg8) : (index) -> i32
      %147 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %148 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %149 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %150 = "arith.muli"(%146, %149) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %151 = "arith.addi"(%148, %150) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %152 = "arith.addi"(%151, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %153 = "arith.addi"(%147, %152) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %154 = "aps.read_smem"(%78, %153) : (memref<96xf32>, i32) -> f32
      %155 = "arith.subf"(%154, %145) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %156 = "arith.index_cast"(%arg8) : (index) -> i32
      %157 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %158 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %159 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %160 = "arith.muli"(%156, %159) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %161 = "arith.addi"(%158, %160) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %162 = "arith.addi"(%161, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %163 = "arith.addi"(%157, %162) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%155, %78, %163) : (f32, memref<96xf32>, i32) -> ()
      %164 = "arith.index_cast"(%arg8) : (index) -> i32
      %165 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %166 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %167 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %168 = "arith.muli"(%164, %167) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %169 = "arith.addi"(%166, %168) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %170 = "arith.addi"(%169, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %171 = "arith.addi"(%165, %170) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %172 = "aps.read_smem"(%71, %171) : (memref<96xf32>, i32) -> f32
      %173 = "arith.mulf"(%172, %84) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %174 = "arith.index_cast"(%arg8) : (index) -> i32
      %175 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %176 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %177 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %178 = "arith.muli"(%174, %177) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %179 = "arith.addi"(%176, %178) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %180 = "arith.addi"(%179, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %181 = "arith.addi"(%175, %180) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %182 = "aps.read_smem"(%74, %181) : (memref<96xf32>, i32) -> f32
      %183 = "arith.divf"(%173, %182) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %184 = "arith.index_cast"(%arg8) : (index) -> i32
      %185 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %186 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %187 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %188 = "arith.muli"(%184, %187) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %189 = "arith.addi"(%186, %188) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %190 = "arith.addi"(%189, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %191 = "arith.addi"(%185, %190) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %192 = "aps.read_smem"(%78, %191) : (memref<96xf32>, i32) -> f32
      %193 = "arith.subf"(%192, %183) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %194 = "arith.index_cast"(%arg8) : (index) -> i32
      %195 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %196 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %197 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %198 = "arith.muli"(%194, %197) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %199 = "arith.addi"(%196, %198) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %200 = "arith.addi"(%199, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %201 = "arith.addi"(%195, %200) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%193, %78, %201) : (f32, memref<96xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %86 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    %87 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%77, %78, %87, %86) <{direction = 1 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_26__direct"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_26__direct_buffer}> : () -> memref<96xf32>
    %3 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %2, %4, %3) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %5 = "memref.get_global"() <{name = @au_generated_candidate_26__direct_buffer_2}> : () -> memref<1xi32>
    %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "arith.addi"(%1, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %8 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %9 = "aps.load"(%7) : (i32) -> i32
    "aps.write_smem"(%9, %5, %8) : (i32, memref<1xi32>, i32) -> ()
    %10 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %12 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %13 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
    %14 = "arith.constant"() <{value = 1.030000e+02 : f32}> : () -> f32
    %15 = "arith.constant"() <{value = 7.900000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 32 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg3: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg4: index):
        %20 = "arith.index_cast"(%arg4) : (index) -> i32
        %21 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %22 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %23 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %24 = "arith.muli"(%20, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %25 = "arith.addi"(%22, %24) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %26 = "arith.addi"(%25, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %27 = "arith.addi"(%21, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %28 = "aps.read_smem"(%2, %27) : (memref<96xf32>, i32) -> f32
        %29 = "arith.mulf"(%28, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %30 = "arith.fptosi"(%29) : (f32) -> i32
        %31 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %32 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %33 = "arith.addi"(%31, %32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %34 = "aps.read_smem"(%5, %33) : (memref<1xi32>, i32) -> i32
        %35 = "arith.addi"(%34, %30) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %36 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %37 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %38 = "arith.addi"(%36, %37) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%35, %5, %38) : (i32, memref<1xi32>, i32) -> ()
        %39 = "arith.index_cast"(%arg4) : (index) -> i32
        %40 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %41 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %42 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %43 = "arith.muli"(%39, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %44 = "arith.addi"(%41, %43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %45 = "arith.addi"(%44, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %46 = "arith.addi"(%40, %45) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %47 = "aps.read_smem"(%2, %46) : (memref<96xf32>, i32) -> f32
        %48 = "arith.mulf"(%47, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %49 = "arith.fptosi"(%48) : (f32) -> i32
        %50 = "arith.addi"(%35, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %51 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %52 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %53 = "arith.addi"(%51, %52) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%50, %5, %53) : (i32, memref<1xi32>, i32) -> ()
        %54 = "arith.index_cast"(%arg4) : (index) -> i32
        %55 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %56 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %57 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %58 = "arith.muli"(%54, %57) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %59 = "arith.addi"(%56, %58) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %60 = "arith.addi"(%59, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %61 = "arith.addi"(%55, %60) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %62 = "aps.read_smem"(%2, %61) : (memref<96xf32>, i32) -> f32
        %63 = "arith.mulf"(%62, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %64 = "arith.fptosi"(%63) : (f32) -> i32
        %65 = "arith.addi"(%50, %64) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %66 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %67 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %68 = "arith.addi"(%66, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%65, %5, %68) : (i32, memref<1xi32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %16 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %17 = "arith.addi"(%1, %16) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %18 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %19 = "aps.read_smem"(%5, %18) : (memref<1xi32>, i32) -> i32
    "aps.store"(%19, %17) : (i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<32x3xf32>, memref<i32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_17__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
}) : () -> ()

