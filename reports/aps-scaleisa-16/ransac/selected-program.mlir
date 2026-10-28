"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %92 = "arith.constant"() <{value = 3 : index}> : () -> index
    %93 = "arith.constant"() <{value = 4 : index}> : () -> index
    %94 = "arith.constant"() <{value = 1.000000e+03 : f32}> : () -> f32
    %95 = "arith.constant"() <{value = 1.100000e+03 : f32}> : () -> f32
    %96 = "arith.constant"() <{value = 1.200000e+03 : f32}> : () -> f32
    %97 = "arith.constant"() <{value = 1.300000e+03 : f32}> : () -> f32
    %98 = "arith.constant"() <{value = 0 : index}> : () -> index
    %99 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %100 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %101 = "arith.constant"() <{value = 2 : index}> : () -> index
    %102 = "arith.constant"() <{value = 1 : index}> : () -> index
    %103 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %104 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %105 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    %106 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %107 = "arith.constant"() <{value = 32 : i32}> : () -> i32
    %108 = "arith.constant"() <{value = 23 : i32}> : () -> i32
    %109 = "arith.constant"() <{value = 29 : i32}> : () -> i32
    %110 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
    %111 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %112 = "arith.constant"() <{value = false}> : () -> i1
    %113 = "arith.constant"() <{value = true}> : () -> i1
    %114 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %115 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %116 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %117 = "arith.constant"() <{value = 19 : i32}> : () -> i32
    %118 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %119 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xi32>
    %120 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %121 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x3xf32>
    "func.call"(%121) <{callee = @au_generated_candidate_1__l0u_site_13498}> : (memref<32x3xf32>) -> ()
    %122 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi32>
    "memref.store"(%100, %122, %98) : (i32, memref<1xi32>, index) -> ()
    %123 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi32>
    "memref.store"(%100, %123, %98) : (i32, memref<1xi32>, index) -> ()
    %124 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi32>
    %125 = "memref.cast"(%124) : (memref<1xi32>) -> memref<?xi32>
    "memref.store"(%100, %124, %98) : (i32, memref<1xi32>, index) -> ()
    %126 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1xi32>
    %127 = "memref.cast"(%126) : (memref<1xi32>) -> memref<?xi32>
    "memref.store"(%100, %126, %98) : (i32, memref<1xi32>, index) -> ()
    %128 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
    "memref.store"(%99, %128) : (f32, memref<f32>) -> ()
    %129 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%100, %129) : (i32, memref<i32>) -> ()
    "memref.store"(%104, %129) : (i32, memref<i32>) -> ()
    "memref.store"(%105, %128) : (f32, memref<f32>) -> ()
    "scf.for"(%98, %93, %102) ({
    ^bb0(%arg11: index):
      %150 = "arith.index_cast"(%arg11) : (index) -> i32
      %151 = "arith.muli"(%150, %118) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %152 = "arith.addi"(%151, %106) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %153 = "arith.remsi"(%152, %107) : (i32, i32) -> i32
      "memref.store"(%153, %126, %98) : (i32, memref<1xi32>, index) -> ()
      %154 = "arith.muli"(%150, %115) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %155 = "arith.addi"(%154, %117) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %156 = "arith.remsi"(%155, %107) : (i32, i32) -> i32
      "memref.store"(%156, %124, %98) : (i32, memref<1xi32>, index) -> ()
      %157 = "arith.muli"(%150, %116) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %158 = "arith.addi"(%157, %108) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %159 = "arith.remsi"(%158, %107) : (i32, i32) -> i32
      "memref.store"(%159, %123, %98) : (i32, memref<1xi32>, index) -> ()
      %160 = "arith.muli"(%150, %109) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %161 = "arith.addi"(%160, %114) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %162 = "arith.remsi"(%161, %107) : (i32, i32) -> i32
      "memref.store"(%162, %122, %98) : (i32, memref<1xi32>, index) -> ()
      %163 = "memref.load"(%126, %98) : (memref<1xi32>, index) -> i32
      %164 = "arith.index_cast"(%163) : (i32) -> index
      %165 = "memref.load"(%121, %164, %98) : (memref<32x3xf32>, index, index) -> f32
      %166 = "memref.load"(%124, %98) : (memref<1xi32>, index) -> i32
      %167 = "arith.index_cast"(%166) : (i32) -> index
      %168 = "memref.load"(%121, %167, %98) : (memref<32x3xf32>, index, index) -> f32
      %169 = "arith.addf"(%165, %168) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %170 = "memref.load"(%123, %98) : (memref<1xi32>, index) -> i32
      %171 = "arith.index_cast"(%170) : (i32) -> index
      %172 = "memref.load"(%121, %171, %98) : (memref<32x3xf32>, index, index) -> f32
      %173 = "arith.addf"(%169, %172) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %174 = "memref.load"(%122, %98) : (memref<1xi32>, index) -> i32
      %175 = "arith.index_cast"(%174) : (i32) -> index
      %176 = "memref.load"(%121, %175, %98) : (memref<32x3xf32>, index, index) -> f32
      %177 = "arith.addf"(%173, %176) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %178 = "arith.mulf"(%177, %110) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %179 = "memref.load"(%121, %164, %102) : (memref<32x3xf32>, index, index) -> f32
      %180 = "memref.load"(%121, %167, %102) : (memref<32x3xf32>, index, index) -> f32
      %181 = "arith.addf"(%179, %180) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %182 = "memref.load"(%121, %171, %102) : (memref<32x3xf32>, index, index) -> f32
      %183 = "arith.addf"(%181, %182) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %184 = "memref.load"(%121, %175, %102) : (memref<32x3xf32>, index, index) -> f32
      %185 = "arith.addf"(%183, %184) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %186 = "arith.mulf"(%185, %110) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %187 = "memref.load"(%121, %164, %101) : (memref<32x3xf32>, index, index) -> f32
      %188 = "memref.load"(%121, %167, %101) : (memref<32x3xf32>, index, index) -> f32
      %189 = "arith.addf"(%187, %188) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %190 = "memref.load"(%121, %171, %101) : (memref<32x3xf32>, index, index) -> f32
      %191 = "arith.addf"(%189, %190) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %192 = "memref.load"(%121, %175, %101) : (memref<32x3xf32>, index, index) -> f32
      %193 = "arith.addf"(%191, %192) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %194 = "arith.mulf"(%193, %110) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %195 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%99, %195) : (f32, memref<f32>) -> ()
      %196 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "memref.store"(%100, %196) : (i32, memref<i32>) -> ()
      %197 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%99, %197) : (f32, memref<f32>) -> ()
      "scf.for"(%98, %93, %102) ({
      ^bb0(%arg12: index):
        %210 = "arith.index_cast"(%arg12) : (index) -> i32
        %211 = "arith.cmpi"(%210, %100) <{predicate = 0 : i64}> : (i32, i32) -> i1
        %212 = "scf.if"(%211) ({
          "scf.yield"(%127) : (memref<?xi32>) -> ()
        }, {
          %234 = "arith.cmpi"(%210, %103) <{predicate = 0 : i64}> : (i32, i32) -> i1
          %235 = "scf.if"(%234) ({
            "scf.yield"(%125) : (memref<?xi32>) -> ()
          }, {
            %236 = "arith.cmpi"(%210, %111) <{predicate = 0 : i64}> : (i32, i32) -> i1
            %237 = "arith.select"(%236, %123, %122) : (i1, memref<1xi32>, memref<1xi32>) -> memref<1xi32>
            %238 = "memref.cast"(%237) : (memref<1xi32>) -> memref<?xi32>
            "scf.yield"(%238) : (memref<?xi32>) -> ()
          }) : (i1) -> memref<?xi32>
          "scf.yield"(%235) : (memref<?xi32>) -> ()
        }) : (i1) -> memref<?xi32>
        %213 = "memref.load"(%212, %98) : (memref<?xi32>, index) -> i32
        %214 = "arith.index_cast"(%213) : (i32) -> index
        %215 = "memref.load"(%121, %214, %98) : (memref<32x3xf32>, index, index) -> f32
        %216 = "arith.subf"(%215, %178) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %217 = "memref.load"(%121, %214, %102) : (memref<32x3xf32>, index, index) -> f32
        %218 = "arith.subf"(%217, %186) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %219 = "memref.load"(%121, %214, %101) : (memref<32x3xf32>, index, index) -> f32
        %220 = "arith.subf"(%219, %194) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %221 = "arith.cmpf"(%216, %99) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %222 = "scf.if"(%221) ({
          %233 = "arith.negf"(%216) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%233) : (f32) -> ()
        }, {
          "scf.yield"(%216) : (f32) -> ()
        }) : (i1) -> f32
        %223 = "arith.cmpf"(%218, %99) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %224 = "scf.if"(%223) ({
          %232 = "arith.negf"(%218) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%232) : (f32) -> ()
        }, {
          "scf.yield"(%218) : (f32) -> ()
        }) : (i1) -> f32
        %225 = "arith.addf"(%222, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %226 = "arith.cmpf"(%220, %99) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %227 = "scf.if"(%226) ({
          %231 = "arith.negf"(%220) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%231) : (f32) -> ()
        }, {
          "scf.yield"(%220) : (f32) -> ()
        }) : (i1) -> f32
        %228 = "arith.addf"(%225, %227) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %229 = "memref.load"(%195) : (memref<f32>) -> f32
        %230 = "arith.addf"(%229, %228) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%230, %195) : (f32, memref<f32>) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %198 = "memref.load"(%195) : (memref<f32>) -> f32
      %199 = "arith.mulf"(%198, %110) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%199, %195) : (f32, memref<f32>) -> ()
      "func.call"(%121, %178, %186, %194, %199, %196, %197) <{callee = @au_generated_candidate_0__direct_site_13710}> : (memref<32x3xf32>, f32, f32, f32, f32, memref<i32>, memref<f32>) -> ()
      %200 = "memref.load"(%196) : (memref<i32>) -> i32
      "memref.store"(%200, %119, %arg11) : (i32, memref<4xi32>, index) -> ()
      %201 = "memref.load"(%129) : (memref<i32>) -> i32
      %202 = "arith.cmpi"(%200, %201) <{predicate = 4 : i64}> : (i32, i32) -> i1
      %203 = "scf.if"(%202) ({
        "scf.yield"(%113) : (i1) -> ()
      }, {
        %205 = "arith.cmpi"(%200, %201) <{predicate = 0 : i64}> : (i32, i32) -> i1
        %206 = "scf.if"(%205) ({
          %207 = "memref.load"(%197) : (memref<f32>) -> f32
          %208 = "memref.load"(%128) : (memref<f32>) -> f32
          %209 = "arith.cmpf"(%207, %208) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          "scf.yield"(%209) : (i1) -> ()
        }, {
          "scf.yield"(%112) : (i1) -> ()
        }) : (i1) -> i1
        "scf.yield"(%206) : (i1) -> ()
      }) : (i1) -> i1
      "scf.if"(%203) ({
        "memref.store"(%200, %129) : (i32, memref<i32>) -> ()
        %204 = "memref.load"(%197) : (memref<f32>) -> f32
        "memref.store"(%204, %128) : (f32, memref<f32>) -> ()
        "memref.store"(%178, %120, %98) : (f32, memref<4xf32>, index) -> ()
        "memref.store"(%186, %120, %102) : (f32, memref<4xf32>, index) -> ()
        "memref.store"(%194, %120, %101) : (f32, memref<4xf32>, index) -> ()
        "memref.store"(%199, %120, %92) : (f32, memref<4xf32>, index) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %130 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%100, %130) : (i32, memref<i32>) -> ()
    %131 = "memref.load"(%120, %98) : (memref<4xf32>, index) -> f32
    %132 = "arith.mulf"(%131, %94) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %133 = "arith.fptosi"(%132) : (f32) -> i32
    %134 = "memref.load"(%120, %102) : (memref<4xf32>, index) -> f32
    %135 = "arith.mulf"(%134, %95) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %136 = "arith.fptosi"(%135) : (f32) -> i32
    %137 = "arith.addi"(%133, %136) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %138 = "memref.load"(%120, %101) : (memref<4xf32>, index) -> f32
    %139 = "arith.mulf"(%138, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %140 = "arith.fptosi"(%139) : (f32) -> i32
    %141 = "arith.addi"(%137, %140) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %142 = "memref.load"(%120, %92) : (memref<4xf32>, index) -> f32
    %143 = "arith.mulf"(%142, %97) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %144 = "arith.fptosi"(%143) : (f32) -> i32
    %145 = "arith.addi"(%141, %144) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%145, %130) : (i32, memref<i32>) -> ()
    "scf.for"(%98, %93, %102) ({
    ^bb0(%arg10: index):
      %147 = "memref.load"(%119, %arg10) : (memref<4xi32>, index) -> i32
      %148 = "memref.load"(%130) : (memref<i32>) -> i32
      %149 = "arith.addi"(%148, %147) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%149, %130) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %146 = "memref.load"(%130) : (memref<i32>) -> i32
    "func.call"(%121, %120, %119) <{callee = @pcl_validate_outputs}> : (memref<32x3xf32>, memref<4xf32>, memref<4xi32>) -> ()
    "func.return"(%146) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_1__l0u_site_13498"}> ({
  ^bb0(%arg8: memref<32x3xf32>):
    %33 = "arith.constant"() <{value = 8.900000e+01 : f32}> : () -> f32
    %34 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %35 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %36 = "arith.constant"() <{value = 19 : i32}> : () -> i32
    %37 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
    %38 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %39 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %40 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %41 = "arith.constant"() <{value = 1.010000e+02 : f32}> : () -> f32
    %42 = "arith.constant"() <{value = 101 : i32}> : () -> i32
    %43 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %44 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %45 = "arith.constant"() <{value = 2 : index}> : () -> index
    %46 = "arith.constant"() <{value = 3.000000e-01 : f32}> : () -> f32
    %47 = "arith.constant"() <{value = 3.100000e+01 : f32}> : () -> f32
    %48 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %49 = "arith.constant"() <{value = 4.000000e-01 : f32}> : () -> f32
    %50 = "arith.constant"() <{value = 1.200000e+01 : f32}> : () -> f32
    %51 = "arith.constant"() <{value = 6.000000e-01 : f32}> : () -> f32
    %52 = "arith.constant"() <{value = 6.400000e+01 : f32}> : () -> f32
    %53 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    %54 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %55 = "arith.constant"() <{value = 768 : i32}> : () -> i32
    %56 = "arith.constant"() <{value = 0 : index}> : () -> index
    %57 = "arith.constant"() <{value = 32 : index}> : () -> index
    %58 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%56, %57, %58) ({
    ^bb0(%arg9: index):
      %59 = "arith.index_cast"(%arg9) : (index) -> i32
      %60 = "arith.cmpi"(%59, %55) <{predicate = 2 : i64}> : (i32, i32) -> i1
      "scf.if"(%60) ({
        %76 = "arith.remsi"(%59, %54) : (i32, i32) -> i32
        %77 = "arith.divsi"(%59, %54) : (i32, i32) -> i32
        %78 = "arith.sitofp"(%76) : (i32) -> f32
        %79 = "arith.mulf"(%78, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %80 = "arith.divf"(%79, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %81 = "arith.addf"(%80, %51) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%81, %arg8, %arg9, %56) : (f32, memref<32x3xf32>, index, index) -> ()
        %82 = "arith.sitofp"(%77) : (i32) -> f32
        %83 = "arith.mulf"(%82, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %84 = "arith.divf"(%83, %50) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %85 = "arith.addf"(%84, %49) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%85, %arg8, %arg9, %58) : (f32, memref<32x3xf32>, index, index) -> ()
        %86 = "arith.addi"(%76, %77) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %87 = "arith.remsi"(%86, %48) : (i32, i32) -> i32
        %88 = "arith.sitofp"(%87) : (i32) -> f32
        %89 = "arith.mulf"(%88, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %90 = "arith.divf"(%89, %47) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %91 = "arith.addf"(%90, %46) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%91, %arg8, %arg9, %45) : (f32, memref<32x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }, {
        %61 = "arith.muli"(%59, %44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %62 = "arith.addi"(%61, %43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %63 = "arith.remsi"(%62, %42) : (i32, i32) -> i32
        %64 = "arith.sitofp"(%63) : (i32) -> f32
        %65 = "arith.divf"(%64, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%65, %arg8, %arg9, %56) : (f32, memref<32x3xf32>, index, index) -> ()
        %66 = "arith.muli"(%59, %40) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %67 = "arith.addi"(%66, %39) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %68 = "arith.remsi"(%67, %38) : (i32, i32) -> i32
        %69 = "arith.sitofp"(%68) : (i32) -> f32
        %70 = "arith.divf"(%69, %37) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%70, %arg8, %arg9, %58) : (f32, memref<32x3xf32>, index, index) -> ()
        %71 = "arith.muli"(%59, %36) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %72 = "arith.addi"(%71, %35) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %73 = "arith.remsi"(%72, %34) : (i32, i32) -> i32
        %74 = "arith.sitofp"(%73) : (i32) -> f32
        %75 = "arith.divf"(%74, %33) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%75, %arg8, %arg9, %45) : (f32, memref<32x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (i1) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>, f32, f32, f32, f32, memref<i32>, memref<f32>) -> (), sym_name = "au_generated_candidate_0__direct_site_13710"}> ({
  ^bb0(%arg0: memref<32x3xf32>, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: memref<i32>, %arg6: memref<f32>):
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %2 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = 2 : index}> : () -> index
    %4 = "arith.constant"() <{value = 0 : index}> : () -> index
    %5 = "arith.constant"() <{value = 32 : index}> : () -> index
    %6 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%4, %5, %6) ({
    ^bb0(%arg7: index):
      %7 = "memref.load"(%arg0, %arg7, %4) : (memref<32x3xf32>, index, index) -> f32
      %8 = "arith.subf"(%7, %arg1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %9 = "memref.load"(%arg0, %arg7, %6) : (memref<32x3xf32>, index, index) -> f32
      %10 = "arith.subf"(%9, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %11 = "memref.load"(%arg0, %arg7, %3) : (memref<32x3xf32>, index, index) -> f32
      %12 = "arith.subf"(%11, %arg3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %13 = "arith.cmpf"(%8, %2) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %14 = "scf.if"(%13) ({
        %32 = "arith.negf"(%8) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%32) : (f32) -> ()
      }, {
        "scf.yield"(%8) : (f32) -> ()
      }) : (i1) -> f32
      %15 = "arith.cmpf"(%10, %2) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %16 = "scf.if"(%15) ({
        %31 = "arith.negf"(%10) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%31) : (f32) -> ()
      }, {
        "scf.yield"(%10) : (f32) -> ()
      }) : (i1) -> f32
      %17 = "arith.addf"(%14, %16) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %18 = "arith.cmpf"(%12, %2) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %19 = "scf.if"(%18) ({
        %30 = "arith.negf"(%12) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%30) : (f32) -> ()
      }, {
        "scf.yield"(%12) : (f32) -> ()
      }) : (i1) -> f32
      %20 = "arith.addf"(%17, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %21 = "arith.subf"(%20, %arg4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %22 = "arith.cmpf"(%21, %2) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %23 = "scf.if"(%22) ({
        %29 = "arith.negf"(%21) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%29) : (f32) -> ()
      }, {
        "scf.yield"(%21) : (f32) -> ()
      }) : (i1) -> f32
      %24 = "arith.cmpf"(%23, %1) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      "scf.if"(%24) ({
        %27 = "memref.load"(%arg5) : (memref<i32>) -> i32
        %28 = "arith.addi"(%27, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%28, %arg5) : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      %25 = "memref.load"(%arg6) : (memref<f32>) -> f32
      %26 = "arith.addf"(%25, %23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%26, %arg6) : (f32, memref<f32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>, memref<4xf32>, memref<4xi32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
