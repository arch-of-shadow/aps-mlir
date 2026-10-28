#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (32)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_1__l0u_buffer", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_1__l0u_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__direct_buffer_1", type = memref<1xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__direct_buffer_0", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_0__direct_buffer", type = memref<96xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_0__direct_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_0__direct"}> ({
  ^bb0(%arg4: i5, %arg5: i5, %arg6: i5):
    %112 = "aps.read_irf"(%arg4) : (i5) -> i32
    %113 = "aps.read_irf"(%arg5) : (i5) -> i32
    %114 = "memref.get_global"() <{name = @au_generated_candidate_0__direct_buffer}> : () -> memref<96xf32>
    %115 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    %116 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%112, %114, %116, %115) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %117 = "arith.bitcast"(%113) : (i32) -> f32
    %118 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__direct_csr0}> : () -> i32
    %119 = "arith.bitcast"(%118) : (i32) -> f32
    %120 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__direct_csr1}> : () -> i32
    %121 = "arith.bitcast"(%120) : (i32) -> f32
    %122 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__direct_csr2}> : () -> i32
    %123 = "arith.bitcast"(%122) : (i32) -> f32
    %124 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__direct_csr3}> : () -> i32
    %125 = "memref.get_global"() <{name = @au_generated_candidate_0__direct_buffer_0}> : () -> memref<1xi32>
    %126 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %127 = "arith.addi"(%124, %126) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %128 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %129 = "aps.load"(%127) : (i32) -> i32
    "aps.write_smem"(%129, %125, %128) : (i32, memref<1xi32>, i32) -> ()
    %130 = "aps.read_csr"() <{global_name = @au_generated_candidate_0__direct_csr4}> : () -> i32
    %131 = "memref.get_global"() <{name = @au_generated_candidate_0__direct_buffer_1}> : () -> memref<1xf32>
    %132 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %133 = "arith.addi"(%130, %132) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %134 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %135 = "aps.load"(%133) : (i32) -> f32
    "aps.write_smem"(%135, %131, %134) : (f32, memref<1xf32>, i32) -> ()
    %136 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %137 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %138 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %139 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %140 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg7: index):
      %149 = "arith.index_cast"(%arg7) : (index) -> i32
      %150 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %151 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %152 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %153 = "arith.muli"(%149, %152) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %154 = "arith.addi"(%151, %153) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %155 = "arith.addi"(%154, %137) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %156 = "arith.addi"(%150, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %157 = "aps.read_smem"(%114, %156) : (memref<96xf32>, i32) -> f32
      %158 = "arith.subf"(%157, %117) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %159 = "arith.index_cast"(%arg7) : (index) -> i32
      %160 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %161 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %162 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %163 = "arith.muli"(%159, %162) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %164 = "arith.addi"(%161, %163) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %165 = "arith.addi"(%164, %138) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %166 = "arith.addi"(%160, %165) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %167 = "aps.read_smem"(%114, %166) : (memref<96xf32>, i32) -> f32
      %168 = "arith.subf"(%167, %119) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %169 = "arith.index_cast"(%arg7) : (index) -> i32
      %170 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %171 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %172 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %173 = "arith.muli"(%169, %172) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %174 = "arith.addi"(%171, %173) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %175 = "arith.addi"(%174, %136) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %176 = "arith.addi"(%170, %175) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %177 = "aps.read_smem"(%114, %176) : (memref<96xf32>, i32) -> f32
      %178 = "arith.subf"(%177, %121) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %179 = "arith.cmpf"(%158, %140) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %180 = "scf.if"(%179) ({
        %210 = "arith.negf"(%158) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%210) : (f32) -> ()
      }, {
        "scf.yield"(%158) : (f32) -> ()
      }) : (i1) -> f32
      %181 = "arith.cmpf"(%168, %140) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %182 = "scf.if"(%181) ({
        %209 = "arith.negf"(%168) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%209) : (f32) -> ()
      }, {
        "scf.yield"(%168) : (f32) -> ()
      }) : (i1) -> f32
      %183 = "arith.addf"(%180, %182) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %184 = "arith.cmpf"(%178, %140) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %185 = "scf.if"(%184) ({
        %208 = "arith.negf"(%178) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%208) : (f32) -> ()
      }, {
        "scf.yield"(%178) : (f32) -> ()
      }) : (i1) -> f32
      %186 = "arith.addf"(%183, %185) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %187 = "arith.subf"(%186, %123) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %188 = "arith.cmpf"(%187, %140) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %189 = "scf.if"(%188) ({
        %207 = "arith.negf"(%187) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%207) : (f32) -> ()
      }, {
        "scf.yield"(%187) : (f32) -> ()
      }) : (i1) -> f32
      %190 = "arith.cmpf"(%189, %139) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      "scf.if"(%190) ({
        %199 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %200 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %201 = "arith.addi"(%199, %200) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %202 = "aps.read_smem"(%125, %201) : (memref<1xi32>, i32) -> i32
        %203 = "arith.addi"(%202, %138) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %204 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %205 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %206 = "arith.addi"(%204, %205) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%203, %125, %206) : (i32, memref<1xi32>, i32) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      %191 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %192 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %193 = "arith.addi"(%191, %192) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %194 = "aps.read_smem"(%131, %193) : (memref<1xf32>, i32) -> f32
      %195 = "arith.addf"(%194, %189) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %196 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %197 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %198 = "arith.addi"(%196, %197) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%195, %131, %198) : (f32, memref<1xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %141 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %142 = "arith.addi"(%124, %141) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %143 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %144 = "aps.read_smem"(%125, %143) : (memref<1xi32>, i32) -> i32
    "aps.store"(%144, %142) : (i32, i32) -> ()
    %145 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %146 = "arith.addi"(%130, %145) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %147 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %148 = "aps.read_smem"(%131, %147) : (memref<1xf32>, i32) -> f32
    "aps.store"(%148, %146) : (f32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr"}, {address = 2050 : i32, arg_index = 3 : i32, kind = "csr"}, {address = 2051 : i32, arg_index = 4 : i32, kind = "csr"}, {address = 2052 : i32, arg_index = 5 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}, {address = 2053 : i32, arg_index = 6 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<f32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<32x3xf32>, f32, f32, f32, f32, memref<i32>, memref<f32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_1__l0u"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "memref.get_global"() <{name = @au_generated_candidate_1__l0u_buffer}> : () -> memref<96xf32>
    %2 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %1, %3, %2) <{direction = 0 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    %4 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 8.900000e+01 : f32}> : () -> f32
    %8 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 19 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
    %12 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %13 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 1.010000e+02 : f32}> : () -> f32
    %16 = "arith.constant"() <{value = 101 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %18 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %19 = "arith.constant"() <{value = 3.000000e-01 : f32}> : () -> f32
    %20 = "arith.constant"() <{value = 3.100000e+01 : f32}> : () -> f32
    %21 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %22 = "arith.constant"() <{value = 4.000000e-01 : f32}> : () -> f32
    %23 = "arith.constant"() <{value = 1.200000e+01 : f32}> : () -> f32
    %24 = "arith.constant"() <{value = 6.000000e-01 : f32}> : () -> f32
    %25 = "arith.constant"() <{value = 6.400000e+01 : f32}> : () -> f32
    %26 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    %27 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %28 = "arith.constant"() <{value = 768 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg3: index):
      %31 = "arith.index_cast"(%arg3) : (index) -> i32
      %32 = "arith.cmpi"(%31, %28) <{predicate = 2 : i64}> : (i32, i32) -> i1
      "scf.if"(%32) ({
        %72 = "arith.remsi"(%31, %27) : (i32, i32) -> i32
        %73 = "arith.divsi"(%31, %27) : (i32, i32) -> i32
        %74 = "arith.sitofp"(%72) : (i32) -> f32
        %75 = "arith.mulf"(%74, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %76 = "arith.divf"(%75, %25) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %77 = "arith.addf"(%76, %24) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %78 = "arith.index_cast"(%arg3) : (index) -> i32
        %79 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %80 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %81 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %82 = "arith.muli"(%78, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %83 = "arith.addi"(%80, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %84 = "arith.addi"(%83, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %85 = "arith.addi"(%79, %84) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%77, %1, %85) : (f32, memref<96xf32>, i32) -> ()
        %86 = "arith.sitofp"(%73) : (i32) -> f32
        %87 = "arith.mulf"(%86, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %88 = "arith.divf"(%87, %23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %89 = "arith.addf"(%88, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %90 = "arith.index_cast"(%arg3) : (index) -> i32
        %91 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %92 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %93 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %94 = "arith.muli"(%90, %93) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %95 = "arith.addi"(%92, %94) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %96 = "arith.addi"(%95, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %97 = "arith.addi"(%91, %96) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%89, %1, %97) : (f32, memref<96xf32>, i32) -> ()
        %98 = "arith.addi"(%72, %73) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %99 = "arith.remsi"(%98, %21) : (i32, i32) -> i32
        %100 = "arith.sitofp"(%99) : (i32) -> f32
        %101 = "arith.mulf"(%100, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %102 = "arith.divf"(%101, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %103 = "arith.addf"(%102, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %104 = "arith.index_cast"(%arg3) : (index) -> i32
        %105 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %106 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %107 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %108 = "arith.muli"(%104, %107) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %109 = "arith.addi"(%106, %108) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %110 = "arith.addi"(%109, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %111 = "arith.addi"(%105, %110) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%103, %1, %111) : (f32, memref<96xf32>, i32) -> ()
        "scf.yield"() : () -> ()
      }, {
        %33 = "arith.muli"(%31, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %34 = "arith.addi"(%33, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %35 = "arith.remsi"(%34, %16) : (i32, i32) -> i32
        %36 = "arith.sitofp"(%35) : (i32) -> f32
        %37 = "arith.divf"(%36, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %38 = "arith.index_cast"(%arg3) : (index) -> i32
        %39 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %40 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %41 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %42 = "arith.muli"(%38, %41) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %43 = "arith.addi"(%40, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %44 = "arith.addi"(%43, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %45 = "arith.addi"(%39, %44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%37, %1, %45) : (f32, memref<96xf32>, i32) -> ()
        %46 = "arith.muli"(%31, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %47 = "arith.addi"(%46, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %48 = "arith.remsi"(%47, %12) : (i32, i32) -> i32
        %49 = "arith.sitofp"(%48) : (i32) -> f32
        %50 = "arith.divf"(%49, %11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %51 = "arith.index_cast"(%arg3) : (index) -> i32
        %52 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %53 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %54 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %55 = "arith.muli"(%51, %54) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %56 = "arith.addi"(%53, %55) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %57 = "arith.addi"(%56, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %58 = "arith.addi"(%52, %57) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%50, %1, %58) : (f32, memref<96xf32>, i32) -> ()
        %59 = "arith.muli"(%31, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %60 = "arith.addi"(%59, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %61 = "arith.remsi"(%60, %8) : (i32, i32) -> i32
        %62 = "arith.sitofp"(%61) : (i32) -> f32
        %63 = "arith.divf"(%62, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %64 = "arith.index_cast"(%arg3) : (index) -> i32
        %65 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %66 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %67 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %68 = "arith.muli"(%64, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %69 = "arith.addi"(%66, %68) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %70 = "arith.addi"(%69, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %71 = "arith.addi"(%65, %70) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%63, %1, %71) : (f32, memref<96xf32>, i32) -> ()
        "scf.yield"() : () -> ()
      }) : (i1) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    %29 = "arith.constant"() <{value = 96 : i32}> : () -> i32
    %30 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "aps.copy"(%0, %1, %30, %29) <{direction = 1 : i32}> : (i32, memref<96xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 2 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<32x3xf32>, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<32x3xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__direct_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__direct_csr2", sym_visibility = "private", type = memref<i32>}> {csr_address = 2051 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__direct_csr3", sym_visibility = "private", type = memref<i32>}> {csr_address = 2052 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_0__direct_csr4", sym_visibility = "private", type = memref<i32>}> {csr_address = 2053 : i32} : () -> ()
}) : () -> ()

