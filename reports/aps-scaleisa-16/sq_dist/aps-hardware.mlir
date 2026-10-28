"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_0__l0p_u0x2_buffer_2", type = memref<64xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [2], scaleisa.scratchpad, var_name = "au_generated_candidate_0__l0p_u0x2_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__l0p_u0x2_buffer_1", type = memref<3xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__l0p_u0x2_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__l0p_u0x2_buffer_0", type = memref<3xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__l0p_u0x2_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__l0p_u0x2_buffer", type = memref<384xf32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [2], scaleisa.scratchpad, var_name = "au_generated_candidate_0__l0p_u0x2_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_0__l0p_u0x2"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_0__l0p_u0x2_buffer}> : () -> memref<384xf32>
    %3 = "arith.constant"() <{value = 384 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %2, %4, %3) <{direction = 0 : i32}> : (i32, memref<384xf32>, i32, i32) -> ()
    %5 = "memref.get_global"() <{name = @au_generated_candidate_0__l0p_u0x2_buffer_0}> : () -> memref<3xf32>
    %6 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%1, %5, %7, %6) <{direction = 0 : i32}> : (i32, memref<3xf32>, i32, i32) -> ()
    %8 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %9 = "arith.addi"(%1, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %10 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %11 = "aps.load"(%9) : (i32) -> f32
    "aps.write_smem"(%11, %5, %10) : (f32, memref<3xf32>, i32) -> ()
    %12 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__l0p_u0x2_csr0}> : () -> i32
    %13 = "memref.get_global"() <{name = @au_generated_candidate_0__l0p_u0x2_buffer_1}> : () -> memref<3xf32>
    %14 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%12, %13, %15, %14) <{direction = 0 : i32}> : (i32, memref<3xf32>, i32, i32) -> ()
    %16 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %17 = "arith.addi"(%12, %16) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %18 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %19 = "aps.load"(%17) : (i32) -> f32
    "aps.write_smem"(%19, %13, %18) : (f32, memref<3xf32>, i32) -> ()
    %20 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__l0p_u0x2_csr1}> : () -> i32
    %21 = "memref.get_global"() <{name = @au_generated_candidate_0__l0p_u0x2_buffer_2}> : () -> memref<64xf32>
    %22 = "arith.constant"() <{value = 5 : index}> : () -> index
    %23 = "arith.constant"() <{value = 4 : index}> : () -> index
    %24 = "arith.constant"() <{value = 3 : index}> : () -> index
    %25 = "arith.constant"() <{value = 2 : index}> : () -> index
    %26 = "arith.constant"() <{value = 0 : index}> : () -> index
    %27 = "arith.constant"() <{value = 64 : index}> : () -> index
    %28 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%26, %27, %28) ({
    ^bb0(%arg3: index):
      %55 = "arith.index_cast"(%arg3) : (index) -> i32
      %56 = "arith.index_cast"(%26) : (index) -> i32
      %57 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %58 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %59 = "arith.constant"() <{value = 6 : i32}> : () -> i32
      %60 = "arith.muli"(%55, %59) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %61 = "arith.addi"(%58, %60) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %62 = "arith.addi"(%61, %56) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %63 = "arith.addi"(%57, %62) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %64 = "aps.read_smem"(%2, %63) : (memref<384xf32>, i32) -> f32
      %65 = "arith.index_cast"(%arg3) : (index) -> i32
      %66 = "arith.index_cast"(%28) : (index) -> i32
      %67 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %68 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %69 = "arith.constant"() <{value = 6 : i32}> : () -> i32
      %70 = "arith.muli"(%65, %69) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %71 = "arith.addi"(%68, %70) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %72 = "arith.addi"(%71, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %73 = "arith.addi"(%67, %72) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %74 = "aps.read_smem"(%2, %73) : (memref<384xf32>, i32) -> f32
      %75 = "arith.index_cast"(%arg3) : (index) -> i32
      %76 = "arith.index_cast"(%25) : (index) -> i32
      %77 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %78 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %79 = "arith.constant"() <{value = 6 : i32}> : () -> i32
      %80 = "arith.muli"(%75, %79) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %81 = "arith.addi"(%78, %80) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %82 = "arith.addi"(%81, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %83 = "arith.addi"(%77, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %84 = "aps.read_smem"(%2, %83) : (memref<384xf32>, i32) -> f32
      %85 = "arith.index_cast"(%arg3) : (index) -> i32
      %86 = "arith.index_cast"(%24) : (index) -> i32
      %87 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %88 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %89 = "arith.constant"() <{value = 6 : i32}> : () -> i32
      %90 = "arith.muli"(%85, %89) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %91 = "arith.addi"(%88, %90) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %92 = "arith.addi"(%91, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %93 = "arith.addi"(%87, %92) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %94 = "aps.read_smem"(%2, %93) : (memref<384xf32>, i32) -> f32
      %95 = "arith.index_cast"(%arg3) : (index) -> i32
      %96 = "arith.index_cast"(%23) : (index) -> i32
      %97 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %98 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %99 = "arith.constant"() <{value = 6 : i32}> : () -> i32
      %100 = "arith.muli"(%95, %99) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %101 = "arith.addi"(%98, %100) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %102 = "arith.addi"(%101, %96) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %103 = "arith.addi"(%97, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %104 = "aps.read_smem"(%2, %103) : (memref<384xf32>, i32) -> f32
      %105 = "arith.index_cast"(%arg3) : (index) -> i32
      %106 = "arith.index_cast"(%22) : (index) -> i32
      %107 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %108 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %109 = "arith.constant"() <{value = 6 : i32}> : () -> i32
      %110 = "arith.muli"(%105, %109) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %111 = "arith.addi"(%108, %110) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %112 = "arith.addi"(%111, %106) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %113 = "arith.addi"(%107, %112) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %114 = "aps.read_smem"(%2, %113) : (memref<384xf32>, i32) -> f32
      %115 = "arith.index_cast"(%26) : (index) -> i32
      %116 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %117 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %118 = "arith.addi"(%117, %115) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %119 = "arith.addi"(%116, %118) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%64, %5, %119) : (f32, memref<3xf32>, i32) -> ()
      %120 = "arith.index_cast"(%28) : (index) -> i32
      %121 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %122 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %123 = "arith.addi"(%122, %120) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %124 = "arith.addi"(%121, %123) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%74, %5, %124) : (f32, memref<3xf32>, i32) -> ()
      %125 = "arith.index_cast"(%25) : (index) -> i32
      %126 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %127 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %128 = "arith.addi"(%127, %125) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %129 = "arith.addi"(%126, %128) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%84, %5, %129) : (f32, memref<3xf32>, i32) -> ()
      %130 = "arith.index_cast"(%26) : (index) -> i32
      %131 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %132 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %133 = "arith.addi"(%132, %130) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %134 = "arith.addi"(%131, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%94, %13, %134) : (f32, memref<3xf32>, i32) -> ()
      %135 = "arith.index_cast"(%28) : (index) -> i32
      %136 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %137 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %138 = "arith.addi"(%137, %135) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %139 = "arith.addi"(%136, %138) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%104, %13, %139) : (f32, memref<3xf32>, i32) -> ()
      %140 = "arith.index_cast"(%25) : (index) -> i32
      %141 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %142 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %143 = "arith.addi"(%142, %140) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %144 = "arith.addi"(%141, %143) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%114, %13, %144) : (f32, memref<3xf32>, i32) -> ()
      %145 = "arith.index_cast"(%26) : (index) -> i32
      %146 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %147 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %148 = "arith.addi"(%147, %145) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %149 = "arith.addi"(%146, %148) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %150 = "aps.read_smem"(%5, %149) : (memref<3xf32>, i32) -> f32
      %151 = "arith.index_cast"(%26) : (index) -> i32
      %152 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %153 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %154 = "arith.addi"(%153, %151) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %155 = "arith.addi"(%152, %154) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %156 = "aps.read_smem"(%13, %155) : (memref<3xf32>, i32) -> f32
      %157 = "arith.subf"(%150, %156) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %158 = "arith.index_cast"(%28) : (index) -> i32
      %159 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %160 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %161 = "arith.addi"(%160, %158) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %162 = "arith.addi"(%159, %161) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %163 = "aps.read_smem"(%5, %162) : (memref<3xf32>, i32) -> f32
      %164 = "arith.index_cast"(%28) : (index) -> i32
      %165 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %166 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %167 = "arith.addi"(%166, %164) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %168 = "arith.addi"(%165, %167) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %169 = "aps.read_smem"(%13, %168) : (memref<3xf32>, i32) -> f32
      %170 = "arith.subf"(%163, %169) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %171 = "arith.index_cast"(%25) : (index) -> i32
      %172 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %173 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %174 = "arith.addi"(%173, %171) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %175 = "arith.addi"(%172, %174) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %176 = "aps.read_smem"(%5, %175) : (memref<3xf32>, i32) -> f32
      %177 = "arith.index_cast"(%25) : (index) -> i32
      %178 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %179 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %180 = "arith.addi"(%179, %177) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %181 = "arith.addi"(%178, %180) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %182 = "aps.read_smem"(%13, %181) : (memref<3xf32>, i32) -> f32
      %183 = "arith.subf"(%176, %182) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %184 = "arith.mulf"(%157, %157) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %185 = "arith.mulf"(%170, %170) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %186 = "arith.addf"(%184, %185) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %187 = "arith.mulf"(%183, %183) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %188 = "arith.addf"(%186, %187) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %189 = "arith.index_cast"(%arg3) : (index) -> i32
      %190 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %191 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %192 = "arith.addi"(%191, %189) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %193 = "arith.addi"(%190, %192) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%188, %21, %193) : (f32, memref<64xf32>, i32) -> ()
      "scf.yield"() : () -> ()
    }) {pipeline = 1 : i64, unroll = 2 : i64} : (index, index, index) -> ()
    %29 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %30 = "arith.addi"(%1, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %31 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %32 = "aps.read_smem"(%5, %31) : (memref<3xf32>, i32) -> f32
    "aps.store"(%32, %30) : (f32, i32) -> ()
    %33 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %34 = "arith.addi"(%1, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %35 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %36 = "aps.read_smem"(%5, %35) : (memref<3xf32>, i32) -> f32
    "aps.store"(%36, %34) : (f32, i32) -> ()
    %37 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %38 = "arith.addi"(%1, %37) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %39 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %40 = "aps.read_smem"(%5, %39) : (memref<3xf32>, i32) -> f32
    "aps.store"(%40, %38) : (f32, i32) -> ()
    %41 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %42 = "arith.addi"(%12, %41) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %43 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %44 = "aps.read_smem"(%13, %43) : (memref<3xf32>, i32) -> f32
    "aps.store"(%44, %42) : (f32, i32) -> ()
    %45 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %46 = "arith.addi"(%12, %45) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %47 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %48 = "aps.read_smem"(%13, %47) : (memref<3xf32>, i32) -> f32
    "aps.store"(%48, %46) : (f32, i32) -> ()
    %49 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %50 = "arith.addi"(%12, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %51 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %52 = "aps.read_smem"(%13, %51) : (memref<3xf32>, i32) -> f32
    "aps.store"(%52, %50) : (f32, i32) -> ()
    %53 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %54 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%20, %21, %54, %53) <{direction = 1 : i32}> : (i32, memref<64xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<64x6xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<3xf32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2050 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<64xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<64x6xf32>, memref<3xf32>, memref<3xf32>, memref<64xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__l0p_u0x2_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__l0p_u0x2_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
}) : () -> ()

