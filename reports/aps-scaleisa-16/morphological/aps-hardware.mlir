#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (12)>
#map2 = affine_map<() -> (5)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_9__direct_buffer_2", type = memref<144xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_9__direct_buffer_2"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_9__direct_buffer", type = memref<144xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_9__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_9__direct_local", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_9__direct_local"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_8__direct_buffer_1", type = memref<144xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_8__direct_buffer_1"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_8__direct_buffer", type = memref<144xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_8__direct_buffer"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_8__direct_local", type = memref<f32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_8__direct_local"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_40__direct_buffer_0", type = memref<1xi32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_40__direct_buffer_0"} : () -> ()
  "memref.global"() <{sym_name = "au_generated_candidate_40__direct_buffer", type = memref<144xf32>}> {scaleisa.scratchpad, var_name = "au_generated_candidate_40__direct_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_40__direct"}> ({
  ^bb0(%arg12: i5, %arg13: i5, %arg14: i5):
    %142 = "aps.read_irf"(%arg12) : (i5) -> i32
    %143 = "aps.read_irf"(%arg13) : (i5) -> i32
    %144 = "memref.get_global"() <{name = @au_generated_candidate_40__direct_buffer}> : () -> memref<144xf32>
    %145 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %146 = "arith.constant"() <{value = 144 : i32}> : () -> i32
    "aps.copy"(%142, %144, %145, %146) <{direction = 0 : i32}> : (i32, memref<144xf32>, i32, i32) -> ()
    %147 = "memref.get_global"() <{name = @au_generated_candidate_40__direct_buffer_0}> : () -> memref<1xi32>
    %148 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %149 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "aps.copy"(%143, %147, %148, %149) <{direction = 0 : i32}> : (i32, memref<1xi32>, i32, i32) -> ()
    %150 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %151 = "arith.constant"() <{value = 1.280000e+02 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg15: index):
      %152 = "arith.index_cast"(%arg15) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg16: index):
        %153 = "arith.index_cast"(%arg16) : (index) -> i32
        %154 = "arith.index_cast"(%arg15) : (index) -> i32
        %155 = "arith.index_cast"(%arg16) : (index) -> i32
        %156 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %157 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %158 = "arith.constant"() <{value = 12 : i32}> : () -> i32
        %159 = "arith.muli"(%154, %158) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %160 = "arith.addi"(%157, %159) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %161 = "arith.addi"(%160, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %162 = "arith.addi"(%156, %161) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %163 = "aps.read_smem"(%144, %162) : (memref<144xf32>, i32) -> f32
        %164 = "arith.mulf"(%163, %151) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %165 = "arith.fptosi"(%164) : (f32) -> i32
        %166 = "arith.addi"(%153, %152) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %167 = "arith.andi"(%166, %150) : (i32, i32) -> i32
        %168 = "arith.addi"(%165, %167) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %169 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %170 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %171 = "arith.addi"(%169, %170) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %172 = "aps.read_smem"(%147, %171) : (memref<1xi32>, i32) -> i32
        %173 = "arith.addi"(%172, %168) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %174 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %175 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %176 = "arith.addi"(%174, %175) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%173, %147, %176) : (i32, memref<1xi32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%143, %147, %148, %149) <{direction = 1 : i32}> : (i32, memref<1xi32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<12x12xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2", memory_mode = "scratchpad", memref_type = memref<i32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<12x12xf32>, memref<i32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_8__direct"}> ({
  ^bb0(%arg6: i5, %arg7: i5, %arg8: i5):
    %71 = "aps.read_irf"(%arg6) : (i5) -> i32
    %72 = "aps.read_irf"(%arg7) : (i5) -> i32
    %73 = "memref.get_global"() <{name = @au_generated_candidate_8__direct_buffer}> : () -> memref<144xf32>
    %74 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %75 = "arith.constant"() <{value = 144 : i32}> : () -> i32
    "aps.copy"(%71, %73, %74, %75) <{direction = 0 : i32}> : (i32, memref<144xf32>, i32, i32) -> ()
    %76 = "arith.index_castui"(%72) : (i32) -> index
    %77 = "aps.read_csr"() <{global_name = @au_generated_candidate_8__direct_csr0}> : () -> i32
    %78 = "aps.read_csr"() <{global_name = @au_generated_candidate_8__direct_csr1}> : () -> i32
    %79 = "memref.get_global"() <{name = @au_generated_candidate_8__direct_buffer_1}> : () -> memref<144xf32>
    %80 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %81 = "arith.constant"() <{value = 144 : i32}> : () -> i32
    "aps.copy"(%78, %79, %80, %81) <{direction = 0 : i32}> : (i32, memref<144xf32>, i32, i32) -> ()
    %82 = "arith.constant"() <{value = true}> : () -> i1
    %83 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %84 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %85 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg9: index):
      %86 = "arith.index_cast"(%arg9) : (index) -> i32
      %87 = "arith.index_cast"(%76) : (index) -> i32
      %88 = "arith.index_cast"(%arg9) : (index) -> i32
      %89 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %90 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %91 = "arith.constant"() <{value = 12 : i32}> : () -> i32
      %92 = "arith.muli"(%87, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %93 = "arith.addi"(%90, %92) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %94 = "arith.addi"(%93, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %95 = "arith.addi"(%89, %94) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %96 = "aps.read_smem"(%73, %95) : (memref<144xf32>, i32) -> f32
      %97 = "memref.get_global"() <{name = @au_generated_candidate_8__direct_local}> : () -> memref<f32>
      "aps.globalstore"(%96) <{global_name = @au_generated_candidate_8__direct_local}> : (f32) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg10: index):
        %108 = "arith.index_cast"(%arg10) : (index) -> i32
        %109 = "arith.addi"(%108, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg11: index):
          %110 = "arith.index_cast"(%arg11) : (index) -> i32
          %111 = "arith.addi"(%110, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %112 = "arith.addi"(%77, %109) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %113 = "arith.cmpi"(%112, %84) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %114 = "arith.cmpi"(%112, %83) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %115 = "arith.andi"(%113, %114) : (i1, i1) -> i1
          %116 = "arith.xori"(%115, %82) : (i1, i1) -> i1
          %117 = "arith.andi"(%116, %113) : (i1, i1) -> i1
          %118 = "arith.select"(%115, %83, %84) : (i1, i32, i32) -> i32
          %119 = "arith.select"(%117, %112, %118) : (i1, i32, i32) -> i32
          %120 = "arith.addi"(%86, %111) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %121 = "arith.cmpi"(%120, %84) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %122 = "arith.cmpi"(%120, %83) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %123 = "arith.andi"(%121, %122) : (i1, i1) -> i1
          %124 = "arith.xori"(%123, %82) : (i1, i1) -> i1
          %125 = "arith.andi"(%124, %121) : (i1, i1) -> i1
          %126 = "arith.select"(%123, %83, %84) : (i1, i32, i32) -> i32
          %127 = "arith.select"(%125, %120, %126) : (i1, i32, i32) -> i32
          %128 = "arith.index_cast"(%119) : (i32) -> index
          %129 = "arith.index_cast"(%127) : (i32) -> index
          %130 = "arith.index_cast"(%128) : (index) -> i32
          %131 = "arith.index_cast"(%129) : (index) -> i32
          %132 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %133 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %134 = "arith.constant"() <{value = 12 : i32}> : () -> i32
          %135 = "arith.muli"(%130, %134) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %136 = "arith.addi"(%133, %135) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %137 = "arith.addi"(%136, %131) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %138 = "arith.addi"(%132, %137) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %139 = "aps.read_smem"(%73, %138) : (memref<144xf32>, i32) -> f32
          %140 = "aps.globalload"() <{global_name = @au_generated_candidate_8__direct_local}> : () -> f32
          %141 = "arith.cmpf"(%139, %140) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
          "scf.if"(%141) ({
            "aps.globalstore"(%139) <{global_name = @au_generated_candidate_8__direct_local}> : (f32) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %98 = "aps.globalload"() <{global_name = @au_generated_candidate_8__direct_local}> : () -> f32
      %99 = "arith.index_cast"(%76) : (index) -> i32
      %100 = "arith.index_cast"(%arg9) : (index) -> i32
      %101 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %102 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %103 = "arith.constant"() <{value = 12 : i32}> : () -> i32
      %104 = "arith.muli"(%99, %103) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %105 = "arith.addi"(%102, %104) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %106 = "arith.addi"(%105, %100) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %107 = "arith.addi"(%101, %106) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%98, %79, %107) : (f32, memref<144xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%78, %79, %80, %81) <{direction = 1 : i32}> : (i32, memref<144xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 1 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<12x12xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2"}, {address = 2049 : i32, arg_index = 2 : i32, kind = "csr"}, {address = 2050 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<12x12xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> ()} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_9__direct"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "aps.read_irf"(%arg1) : (i5) -> i32
    %2 = "memref.get_global"() <{name = @au_generated_candidate_9__direct_buffer}> : () -> memref<144xf32>
    %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 144 : i32}> : () -> i32
    "aps.copy"(%0, %2, %3, %4) <{direction = 0 : i32}> : (i32, memref<144xf32>, i32, i32) -> ()
    %5 = "arith.index_castui"(%1) : (i32) -> index
    %6 = "aps.read_csr"() <{global_name = @au_generated_candidate_9__direct_csr0}> : () -> i32
    %7 = "aps.read_csr"() <{global_name = @au_generated_candidate_9__direct_csr1}> : () -> i32
    %8 = "memref.get_global"() <{name = @au_generated_candidate_9__direct_buffer_2}> : () -> memref<144xf32>
    %9 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 144 : i32}> : () -> i32
    "aps.copy"(%7, %8, %9, %10) <{direction = 0 : i32}> : (i32, memref<144xf32>, i32, i32) -> ()
    %11 = "arith.constant"() <{value = true}> : () -> i1
    %12 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %13 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg3: index):
      %15 = "arith.index_cast"(%arg3) : (index) -> i32
      %16 = "arith.index_cast"(%5) : (index) -> i32
      %17 = "arith.index_cast"(%arg3) : (index) -> i32
      %18 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %19 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %20 = "arith.constant"() <{value = 12 : i32}> : () -> i32
      %21 = "arith.muli"(%16, %20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %22 = "arith.addi"(%19, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %23 = "arith.addi"(%22, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %24 = "arith.addi"(%18, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %25 = "aps.read_smem"(%2, %24) : (memref<144xf32>, i32) -> f32
      %26 = "memref.get_global"() <{name = @au_generated_candidate_9__direct_local}> : () -> memref<f32>
      "aps.globalstore"(%25) <{global_name = @au_generated_candidate_9__direct_local}> : (f32) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg4: index):
        %37 = "arith.index_cast"(%arg4) : (index) -> i32
        %38 = "arith.addi"(%37, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg5: index):
          %39 = "arith.index_cast"(%arg5) : (index) -> i32
          %40 = "arith.addi"(%39, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %41 = "arith.addi"(%6, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %42 = "arith.cmpi"(%41, %13) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %43 = "arith.cmpi"(%41, %12) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %44 = "arith.andi"(%42, %43) : (i1, i1) -> i1
          %45 = "arith.xori"(%44, %11) : (i1, i1) -> i1
          %46 = "arith.andi"(%45, %42) : (i1, i1) -> i1
          %47 = "arith.select"(%44, %12, %13) : (i1, i32, i32) -> i32
          %48 = "arith.select"(%46, %41, %47) : (i1, i32, i32) -> i32
          %49 = "arith.addi"(%15, %40) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %50 = "arith.cmpi"(%49, %13) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %51 = "arith.cmpi"(%49, %12) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %52 = "arith.andi"(%50, %51) : (i1, i1) -> i1
          %53 = "arith.xori"(%52, %11) : (i1, i1) -> i1
          %54 = "arith.andi"(%53, %50) : (i1, i1) -> i1
          %55 = "arith.select"(%52, %12, %13) : (i1, i32, i32) -> i32
          %56 = "arith.select"(%54, %49, %55) : (i1, i32, i32) -> i32
          %57 = "arith.index_cast"(%48) : (i32) -> index
          %58 = "arith.index_cast"(%56) : (i32) -> index
          %59 = "arith.index_cast"(%57) : (index) -> i32
          %60 = "arith.index_cast"(%58) : (index) -> i32
          %61 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %62 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %63 = "arith.constant"() <{value = 12 : i32}> : () -> i32
          %64 = "arith.muli"(%59, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %65 = "arith.addi"(%62, %64) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %66 = "arith.addi"(%65, %60) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %67 = "arith.addi"(%61, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %68 = "aps.read_smem"(%2, %67) : (memref<144xf32>, i32) -> f32
          %69 = "aps.globalload"() <{global_name = @au_generated_candidate_9__direct_local}> : () -> f32
          %70 = "arith.cmpf"(%68, %69) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          "scf.if"(%70) ({
            "aps.globalstore"(%68) <{global_name = @au_generated_candidate_9__direct_local}> : (f32) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %27 = "aps.globalload"() <{global_name = @au_generated_candidate_9__direct_local}> : () -> f32
      %28 = "arith.index_cast"(%5) : (index) -> i32
      %29 = "arith.index_cast"(%arg3) : (index) -> i32
      %30 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %31 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      %32 = "arith.constant"() <{value = 12 : i32}> : () -> i32
      %33 = "arith.muli"(%28, %32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %34 = "arith.addi"(%31, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %35 = "arith.addi"(%34, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %36 = "arith.addi"(%30, %35) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "aps.write_smem"(%27, %8, %36) : (f32, memref<144xf32>, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%7, %8, %9, %10) <{direction = 1 : i32}> : (i32, memref<144xf32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 3 : i32, funct7 = 2 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<12x12xf32>, noalias, value_kind = "cpu_aligned_base"}, {arg_index = 1 : i32, kind = "rs2"}, {address = 2051 : i32, arg_index = 2 : i32, kind = "csr"}, {address = 2052 : i32, arg_index = 3 : i32, kind = "csr", memory_mode = "scratchpad", memref_type = memref<12x12xf32>, noalias, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> ()} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_8__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2049 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_8__direct_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2050 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_9__direct_csr0", sym_visibility = "private", type = memref<i32>}> {csr_address = 2051 : i32} : () -> ()
  "memref.global"() <{initial_value, sym_name = "au_generated_candidate_9__direct_csr1", sym_visibility = "private", type = memref<i32>}> {csr_address = 2052 : i32} : () -> ()
}) : () -> ()

