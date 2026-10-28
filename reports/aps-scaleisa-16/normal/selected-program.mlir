"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %78 = "arith.constant"() <{value = 4 : index}> : () -> index
    %79 = "arith.constant"() <{value = 12 : index}> : () -> index
    %80 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %81 = "arith.constant"() <{value = true}> : () -> i1
    %82 = "arith.constant"() <{value = 0 : index}> : () -> index
    %83 = "arith.constant"() <{value = 2 : index}> : () -> index
    %84 = "arith.constant"() <{value = 1 : index}> : () -> index
    %85 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %86 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %87 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %88 = "arith.constant"() <{value = 1.000000e-03 : f32}> : () -> f32
    %89 = "arith.constant"() <{value = 2.000000e+00 : f32}> : () -> f32
    %90 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %91 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %92 = "arith.constant"() <{value = 9 : i32}> : () -> i32
    %93 = "arith.constant"() <{value = 0.00999999977 : f32}> : () -> f32
    %94 = "arith.constant"() <{value = 7.000000e-02 : f32}> : () -> f32
    %95 = "arith.constant"() <{value = 1.500000e-01 : f32}> : () -> f32
    %96 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
    %97 = "arith.constant"() <{value = 5.000000e-02 : f32}> : () -> f32
    %98 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<12x12xf32>
    %99 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<12x12x3xf32>
    %100 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<12x12x3xf32>
    %101 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<12x12x3xf32>
    "scf.for"(%82, %79, %84) ({
    ^bb0(%arg15: index):
      %203 = "arith.index_cast"(%arg15) : (index) -> i32
      %204 = "arith.sitofp"(%203) : (i32) -> f32
      %205 = "arith.mulf"(%204, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %206 = "arith.mulf"(%205, %94) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "scf.for"(%82, %79, %78) ({
      ^bb0(%arg16: index):
        %207 = "arith.addi"(%arg16, %78) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        "scf.for"(%arg16, %207, %84) ({
        ^bb0(%arg17: index):
          %208 = "arith.index_cast"(%arg17) : (index) -> i32
          %209 = "arith.sitofp"(%208) : (i32) -> f32
          %210 = "arith.mulf"(%209, %97) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%210, %101, %arg15, %arg17, %82) : (f32, memref<12x12x3xf32>, index, index, index) -> ()
          "memref.store"(%205, %101, %arg15, %arg17, %84) : (f32, memref<12x12x3xf32>, index, index, index) -> ()
          %211 = "arith.mulf"(%210, %95) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %212 = "arith.addf"(%211, %206) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %213 = "arith.muli"(%208, %203) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %214 = "arith.remsi"(%213, %92) : (i32, i32) -> i32
          %215 = "arith.sitofp"(%214) : (i32) -> f32
          %216 = "arith.mulf"(%215, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %217 = "arith.addf"(%212, %216) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%217, %101, %arg15, %arg17, %83) : (f32, memref<12x12x3xf32>, index, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%82, %79, %84) ({
    ^bb0(%arg13: index):
      %105 = "arith.index_cast"(%arg13) : (index) -> i32
      "scf.for"(%82, %79, %84) ({
      ^bb0(%arg14: index):
        %106 = "arith.index_cast"(%arg14) : (index) -> i32
        %107 = "arith.addi"(%106, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %108 = "arith.cmpi"(%107, %85) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %109 = "arith.cmpi"(%107, %90) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %110 = "arith.andi"(%108, %109) : (i1, i1) -> i1
        %111 = "arith.xori"(%110, %81) : (i1, i1) -> i1
        %112 = "arith.andi"(%111, %108) : (i1, i1) -> i1
        %113 = "arith.select"(%110, %90, %85) : (i1, i32, i32) -> i32
        %114 = "arith.select"(%112, %107, %113) : (i1, i32, i32) -> i32
        %115 = "arith.addi"(%106, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %116 = "arith.cmpi"(%115, %85) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %117 = "arith.cmpi"(%115, %90) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %118 = "arith.andi"(%116, %117) : (i1, i1) -> i1
        %119 = "arith.xori"(%118, %81) : (i1, i1) -> i1
        %120 = "arith.andi"(%119, %116) : (i1, i1) -> i1
        %121 = "arith.select"(%118, %90, %85) : (i1, i32, i32) -> i32
        %122 = "arith.select"(%120, %115, %121) : (i1, i32, i32) -> i32
        %123 = "arith.addi"(%105, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %124 = "arith.cmpi"(%123, %85) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %125 = "arith.cmpi"(%123, %90) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %126 = "arith.andi"(%124, %125) : (i1, i1) -> i1
        %127 = "arith.xori"(%126, %81) : (i1, i1) -> i1
        %128 = "arith.andi"(%127, %124) : (i1, i1) -> i1
        %129 = "arith.select"(%126, %90, %85) : (i1, i32, i32) -> i32
        %130 = "arith.select"(%128, %123, %129) : (i1, i32, i32) -> i32
        %131 = "arith.addi"(%105, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %132 = "arith.cmpi"(%131, %85) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %133 = "arith.cmpi"(%131, %90) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %134 = "arith.andi"(%132, %133) : (i1, i1) -> i1
        %135 = "arith.xori"(%134, %81) : (i1, i1) -> i1
        %136 = "arith.andi"(%135, %132) : (i1, i1) -> i1
        %137 = "arith.select"(%134, %90, %85) : (i1, i32, i32) -> i32
        %138 = "arith.select"(%136, %131, %137) : (i1, i32, i32) -> i32
        %139 = "arith.index_cast"(%122) : (i32) -> index
        %140 = "memref.load"(%101, %arg13, %139, %82) : (memref<12x12x3xf32>, index, index, index) -> f32
        %141 = "arith.index_cast"(%114) : (i32) -> index
        %142 = "memref.load"(%101, %arg13, %141, %82) : (memref<12x12x3xf32>, index, index, index) -> f32
        %143 = "arith.subf"(%140, %142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %144 = "memref.load"(%101, %arg13, %139, %84) : (memref<12x12x3xf32>, index, index, index) -> f32
        %145 = "memref.load"(%101, %arg13, %141, %84) : (memref<12x12x3xf32>, index, index, index) -> f32
        %146 = "arith.subf"(%144, %145) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %147 = "memref.load"(%101, %arg13, %139, %83) : (memref<12x12x3xf32>, index, index, index) -> f32
        %148 = "memref.load"(%101, %arg13, %141, %83) : (memref<12x12x3xf32>, index, index, index) -> f32
        %149 = "arith.subf"(%147, %148) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %150 = "arith.index_cast"(%138) : (i32) -> index
        %151 = "memref.load"(%101, %150, %arg14, %82) : (memref<12x12x3xf32>, index, index, index) -> f32
        %152 = "arith.index_cast"(%130) : (i32) -> index
        %153 = "memref.load"(%101, %152, %arg14, %82) : (memref<12x12x3xf32>, index, index, index) -> f32
        %154 = "arith.subf"(%151, %153) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %155 = "memref.load"(%101, %150, %arg14, %84) : (memref<12x12x3xf32>, index, index, index) -> f32
        %156 = "memref.load"(%101, %152, %arg14, %84) : (memref<12x12x3xf32>, index, index, index) -> f32
        %157 = "arith.subf"(%155, %156) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %158 = "memref.load"(%101, %150, %arg14, %83) : (memref<12x12x3xf32>, index, index, index) -> f32
        %159 = "memref.load"(%101, %152, %arg14, %83) : (memref<12x12x3xf32>, index, index, index) -> f32
        %160 = "arith.subf"(%158, %159) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %161 = "arith.mulf"(%146, %160) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %162 = "arith.mulf"(%149, %157) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %163 = "arith.subf"(%161, %162) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %164 = "arith.mulf"(%149, %154) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %165 = "arith.mulf"(%143, %160) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %166 = "arith.subf"(%164, %165) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %167 = "arith.mulf"(%143, %157) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %168 = "arith.mulf"(%146, %154) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %169 = "arith.subf"(%167, %168) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %170 = "arith.cmpf"(%163, %80) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %171 = "scf.if"(%170) ({
          %202 = "arith.negf"(%163) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%202) : (f32) -> ()
        }, {
          "scf.yield"(%163) : (f32) -> ()
        }) : (i1) -> f32
        %172 = "arith.cmpf"(%166, %80) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %173 = "scf.if"(%172) ({
          %201 = "arith.negf"(%166) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%201) : (f32) -> ()
        }, {
          "scf.yield"(%166) : (f32) -> ()
        }) : (i1) -> f32
        %174 = "arith.addf"(%171, %173) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %175 = "arith.cmpf"(%169, %80) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %176 = "scf.if"(%175) ({
          %200 = "arith.negf"(%169) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%200) : (f32) -> ()
        }, {
          "scf.yield"(%169) : (f32) -> ()
        }) : (i1) -> f32
        %177 = "arith.addf"(%174, %176) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %178 = "arith.addf"(%177, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %179 = "arith.divf"(%87, %178) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %180 = "arith.mulf"(%163, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%180, %100, %arg13, %arg14, %82) : (f32, memref<12x12x3xf32>, index, index, index) -> ()
        %181 = "arith.mulf"(%166, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%181, %100, %arg13, %arg14, %84) : (f32, memref<12x12x3xf32>, index, index, index) -> ()
        %182 = "arith.mulf"(%169, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%182, %100, %arg13, %arg14, %83) : (f32, memref<12x12x3xf32>, index, index, index) -> ()
        %183 = "memref.load"(%101, %arg13, %139, %83) : (memref<12x12x3xf32>, index, index, index) -> f32
        %184 = "memref.load"(%101, %arg13, %arg14, %83) : (memref<12x12x3xf32>, index, index, index) -> f32
        %185 = "arith.mulf"(%184, %89) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %186 = "arith.subf"(%183, %185) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %187 = "memref.load"(%101, %arg13, %141, %83) : (memref<12x12x3xf32>, index, index, index) -> f32
        %188 = "arith.addf"(%186, %187) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %189 = "arith.cmpf"(%188, %80) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %190 = "scf.if"(%189) ({
          %199 = "arith.negf"(%188) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%199) : (f32) -> ()
        }, {
          "scf.yield"(%188) : (f32) -> ()
        }) : (i1) -> f32
        %191 = "memref.load"(%101, %150, %arg14, %83) : (memref<12x12x3xf32>, index, index, index) -> f32
        %192 = "arith.subf"(%191, %185) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %193 = "memref.load"(%101, %152, %arg14, %83) : (memref<12x12x3xf32>, index, index, index) -> f32
        %194 = "arith.addf"(%192, %193) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %195 = "arith.cmpf"(%194, %80) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %196 = "scf.if"(%195) ({
          %198 = "arith.negf"(%194) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%198) : (f32) -> ()
        }, {
          "scf.yield"(%194) : (f32) -> ()
        }) : (i1) -> f32
        %197 = "arith.addf"(%190, %196) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%197, %98, %arg13, %arg14) : (f32, memref<12x12xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%82, %79, %84) ({
    ^bb0(%arg12: index):
      %104 = "arith.index_cast"(%arg12) : (index) -> i32
      "func.call"(%104, %100, %99, %arg12) <{callee = @au_generated_candidate_12__direct_site_9554}> : (i32, memref<12x12x3xf32>, memref<12x12x3xf32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %102 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%85, %102) : (i32, memref<i32>) -> ()
    "memref.store"(%85, %102) : (i32, memref<i32>) -> ()
    "func.call"(%99, %98, %102) <{callee = @au_generated_candidate_24__direct_site_9589}> : (memref<12x12x3xf32>, memref<12x12xf32>, memref<i32>) -> ()
    %103 = "memref.load"(%102) : (memref<i32>) -> i32
    "func.call"(%101, %100, %99, %98) <{callee = @pcl_validate_outputs}> : (memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12xf32>) -> ()
    "func.return"(%103) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (i32, memref<12x12x3xf32>, memref<12x12x3xf32>, index) -> (), sym_name = "au_generated_candidate_12__direct_site_9554"}> ({
  ^bb0(%arg5: i32, %arg6: memref<12x12x3xf32>, %arg7: memref<12x12x3xf32>, %arg8: index):
    %26 = "arith.constant"() <{value = 9.000000e+00 : f32}> : () -> f32
    %27 = "arith.constant"() <{value = 2 : index}> : () -> index
    %28 = "arith.constant"() <{value = true}> : () -> i1
    %29 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %30 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %31 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %32 = "arith.constant"() <{value = 3 : index}> : () -> index
    %33 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %34 = "arith.constant"() <{value = 0 : index}> : () -> index
    %35 = "arith.constant"() <{value = 12 : index}> : () -> index
    %36 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%34, %35, %36) ({
    ^bb0(%arg9: index):
      %37 = "arith.index_cast"(%arg9) : (index) -> i32
      %38 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%33, %38) : (f32, memref<f32>) -> ()
      %39 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%33, %39) : (f32, memref<f32>) -> ()
      %40 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%33, %40) : (f32, memref<f32>) -> ()
      "scf.for"(%34, %32, %36) ({
      ^bb0(%arg10: index):
        %47 = "arith.index_cast"(%arg10) : (index) -> i32
        %48 = "arith.addi"(%47, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.for"(%34, %32, %36) ({
        ^bb0(%arg11: index):
          %49 = "arith.index_cast"(%arg11) : (index) -> i32
          %50 = "arith.addi"(%49, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %51 = "arith.addi"(%arg5, %48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %52 = "arith.cmpi"(%51, %30) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %53 = "arith.cmpi"(%51, %29) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %54 = "arith.andi"(%52, %53) : (i1, i1) -> i1
          %55 = "arith.xori"(%54, %28) : (i1, i1) -> i1
          %56 = "arith.andi"(%55, %52) : (i1, i1) -> i1
          %57 = "arith.select"(%54, %29, %30) : (i1, i32, i32) -> i32
          %58 = "arith.select"(%56, %51, %57) : (i1, i32, i32) -> i32
          %59 = "arith.addi"(%37, %50) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %60 = "arith.cmpi"(%59, %30) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %61 = "arith.cmpi"(%59, %29) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %62 = "arith.andi"(%60, %61) : (i1, i1) -> i1
          %63 = "arith.xori"(%62, %28) : (i1, i1) -> i1
          %64 = "arith.andi"(%63, %60) : (i1, i1) -> i1
          %65 = "arith.select"(%62, %29, %30) : (i1, i32, i32) -> i32
          %66 = "arith.select"(%64, %59, %65) : (i1, i32, i32) -> i32
          %67 = "arith.index_cast"(%58) : (i32) -> index
          %68 = "arith.index_cast"(%66) : (i32) -> index
          %69 = "memref.load"(%arg6, %67, %68, %34) : (memref<12x12x3xf32>, index, index, index) -> f32
          %70 = "memref.load"(%38) : (memref<f32>) -> f32
          %71 = "arith.addf"(%70, %69) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%71, %38) : (f32, memref<f32>) -> ()
          %72 = "memref.load"(%arg6, %67, %68, %36) : (memref<12x12x3xf32>, index, index, index) -> f32
          %73 = "memref.load"(%39) : (memref<f32>) -> f32
          %74 = "arith.addf"(%73, %72) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%74, %39) : (f32, memref<f32>) -> ()
          %75 = "memref.load"(%arg6, %67, %68, %27) : (memref<12x12x3xf32>, index, index, index) -> f32
          %76 = "memref.load"(%40) : (memref<f32>) -> f32
          %77 = "arith.addf"(%76, %75) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%77, %40) : (f32, memref<f32>) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %41 = "memref.load"(%38) : (memref<f32>) -> f32
      %42 = "arith.divf"(%41, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%42, %arg7, %arg8, %arg9, %34) : (f32, memref<12x12x3xf32>, index, index, index) -> ()
      %43 = "memref.load"(%39) : (memref<f32>) -> f32
      %44 = "arith.divf"(%43, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%44, %arg7, %arg8, %arg9, %36) : (f32, memref<12x12x3xf32>, index, index, index) -> ()
      %45 = "memref.load"(%40) : (memref<f32>) -> f32
      %46 = "arith.divf"(%45, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%46, %arg7, %arg8, %arg9, %27) : (f32, memref<12x12x3xf32>, index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<12x12x3xf32>, memref<12x12xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_24__direct_site_9589"}> ({
  ^bb0(%arg0: memref<12x12x3xf32>, %arg1: memref<12x12xf32>, %arg2: memref<i32>):
    %0 = "arith.constant"() <{value = 6.400000e+01 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 1.100000e+01 : f32}> : () -> f32
    %2 = "arith.constant"() <{value = 3.100000e+01 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = 2 : index}> : () -> index
    %4 = "arith.constant"() <{value = 2.300000e+01 : f32}> : () -> f32
    %5 = "arith.constant"() <{value = 1.700000e+01 : f32}> : () -> f32
    %6 = "arith.constant"() <{value = 0 : index}> : () -> index
    %7 = "arith.constant"() <{value = 10 : index}> : () -> index
    %8 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%6, %7, %8) ({
    ^bb0(%arg3: index):
      "scf.for"(%6, %7, %8) ({
      ^bb0(%arg4: index):
        %9 = "arith.addi"(%arg3, %8) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %10 = "arith.addi"(%arg4, %8) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %11 = "memref.load"(%arg0, %9, %10, %6) : (memref<12x12x3xf32>, index, index, index) -> f32
        %12 = "arith.mulf"(%11, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %13 = "memref.load"(%arg0, %9, %10, %8) : (memref<12x12x3xf32>, index, index, index) -> f32
        %14 = "arith.mulf"(%13, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %15 = "arith.addf"(%12, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %16 = "memref.load"(%arg0, %9, %10, %3) : (memref<12x12x3xf32>, index, index, index) -> f32
        %17 = "arith.mulf"(%16, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %18 = "arith.addf"(%15, %17) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %19 = "memref.load"(%arg1, %9, %10) : (memref<12x12xf32>, index, index) -> f32
        %20 = "arith.mulf"(%19, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %21 = "arith.addf"(%18, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %22 = "arith.mulf"(%21, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %23 = "arith.fptosi"(%22) : (f32) -> i32
        %24 = "memref.load"(%arg2) : (memref<i32>) -> i32
        %25 = "arith.addi"(%24, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%25, %arg2) : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12xf32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
