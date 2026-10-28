"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %124 = "arith.constant"() <{value = 5 : index}> : () -> index
    %125 = "arith.constant"() <{value = 4 : index}> : () -> index
    %126 = "arith.constant"() <{value = 3 : index}> : () -> index
    %127 = "arith.constant"() <{value = 2 : index}> : () -> index
    %128 = "arith.constant"() <{value = 24 : index}> : () -> index
    %129 = "arith.constant"() <{value = 1 : index}> : () -> index
    %130 = "arith.constant"() <{value = 216 : index}> : () -> index
    %131 = "arith.constant"() <{value = 0 : index}> : () -> index
    %132 = "arith.constant"() <{value = true}> : () -> i1
    %133 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %134 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    %135 = "arith.constant"() <{value = 6.000000e+00 : f32}> : () -> f32
    %136 = "arith.constant"() <{value = 5.000000e-01 : f32}> : () -> f32
    %137 = "arith.constant"() <{value = 8.300000e+01 : f32}> : () -> f32
    %138 = "arith.constant"() <{value = 83 : i32}> : () -> i32
    %139 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %140 = "arith.constant"() <{value = 8.900000e+01 : f32}> : () -> f32
    %141 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %142 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %143 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %144 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
    %145 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %146 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %147 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %148 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %149 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<216xf32>
    %150 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<216xf32>
    %151 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<216xf32>
    %152 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<24x6xf32>
    "scf.for"(%131, %130, %129) ({
    ^bb0(%arg19: index):
      "memref.store"(%148, %151, %arg19) : (f32, memref<216xf32>, index) -> ()
      "memref.store"(%148, %150, %arg19) : (f32, memref<216xf32>, index) -> ()
      "memref.store"(%148, %149, %arg19) : (f32, memref<216xf32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%131, %128, %129) ({
    ^bb0(%arg18: index):
      %197 = "arith.index_cast"(%arg18) : (index) -> i32
      %198 = "arith.muli"(%197, %147) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %199 = "arith.addi"(%198, %146) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %200 = "arith.remsi"(%199, %145) : (i32, i32) -> i32
      %201 = "arith.sitofp"(%200) : (i32) -> f32
      %202 = "arith.divf"(%201, %144) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%202, %152, %arg18, %131) : (f32, memref<24x6xf32>, index, index) -> ()
      %203 = "arith.muli"(%197, %143) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %204 = "arith.addi"(%203, %142) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %205 = "arith.remsi"(%204, %141) : (i32, i32) -> i32
      %206 = "arith.sitofp"(%205) : (i32) -> f32
      %207 = "arith.divf"(%206, %140) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%207, %152, %arg18, %129) : (f32, memref<24x6xf32>, index, index) -> ()
      %208 = "arith.muli"(%197, %139) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %209 = "arith.addi"(%208, %147) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %210 = "arith.remsi"(%209, %138) : (i32, i32) -> i32
      %211 = "arith.sitofp"(%210) : (i32) -> f32
      %212 = "arith.divf"(%211, %137) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%212, %152, %arg18, %127) : (f32, memref<24x6xf32>, index, index) -> ()
      %213 = "memref.load"(%152, %arg18, %131) : (memref<24x6xf32>, index, index) -> f32
      %214 = "arith.subf"(%136, %213) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%214, %152, %arg18, %126) : (f32, memref<24x6xf32>, index, index) -> ()
      %215 = "memref.load"(%152, %arg18, %129) : (memref<24x6xf32>, index, index) -> f32
      %216 = "arith.subf"(%136, %215) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%216, %152, %arg18, %125) : (f32, memref<24x6xf32>, index, index) -> ()
      %217 = "memref.load"(%152, %arg18, %127) : (memref<24x6xf32>, index, index) -> f32
      %218 = "arith.subf"(%136, %217) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%218, %152, %arg18, %124) : (f32, memref<24x6xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%131, %128, %129) ({
    ^bb0(%arg17: index):
      %155 = "memref.load"(%152, %arg17, %131) : (memref<24x6xf32>, index, index) -> f32
      %156 = "arith.mulf"(%155, %135) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %157 = "arith.fptosi"(%156) : (f32) -> i32
      %158 = "arith.cmpi"(%157, %133) <{predicate = 5 : i64}> : (i32, i32) -> i1
      %159 = "arith.cmpi"(%157, %142) <{predicate = 4 : i64}> : (i32, i32) -> i1
      %160 = "arith.andi"(%158, %159) : (i1, i1) -> i1
      %161 = "arith.subi"(%132, %160) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
      %162 = "arith.andi"(%161, %158) : (i1, i1) -> i1
      %163 = "arith.select"(%160, %142, %133) : (i1, i32, i32) -> i32
      %164 = "arith.select"(%162, %157, %163) : (i1, i32, i32) -> i32
      %165 = "memref.load"(%152, %arg17, %129) : (memref<24x6xf32>, index, index) -> f32
      %166 = "arith.mulf"(%165, %135) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %167 = "arith.fptosi"(%166) : (f32) -> i32
      %168 = "arith.cmpi"(%167, %133) <{predicate = 5 : i64}> : (i32, i32) -> i1
      %169 = "arith.cmpi"(%167, %142) <{predicate = 4 : i64}> : (i32, i32) -> i1
      %170 = "arith.andi"(%168, %169) : (i1, i1) -> i1
      %171 = "arith.xori"(%170, %132) : (i1, i1) -> i1
      %172 = "arith.andi"(%171, %168) : (i1, i1) -> i1
      %173 = "arith.select"(%170, %142, %133) : (i1, i32, i32) -> i32
      %174 = "arith.select"(%172, %167, %173) : (i1, i32, i32) -> i32
      %175 = "memref.load"(%152, %arg17, %127) : (memref<24x6xf32>, index, index) -> f32
      %176 = "arith.mulf"(%175, %135) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %177 = "arith.fptosi"(%176) : (f32) -> i32
      %178 = "arith.cmpi"(%177, %133) <{predicate = 5 : i64}> : (i32, i32) -> i1
      %179 = "arith.cmpi"(%177, %142) <{predicate = 4 : i64}> : (i32, i32) -> i1
      %180 = "arith.andi"(%178, %179) : (i1, i1) -> i1
      %181 = "arith.xori"(%180, %132) : (i1, i1) -> i1
      %182 = "arith.andi"(%181, %178) : (i1, i1) -> i1
      %183 = "arith.select"(%180, %142, %133) : (i1, i32, i32) -> i32
      %184 = "arith.select"(%182, %177, %183) : (i1, i32, i32) -> i32
      %185 = "arith.muli"(%164, %134) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %186 = "arith.addi"(%185, %174) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %187 = "arith.muli"(%186, %134) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %188 = "arith.addi"(%187, %184) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %189 = "arith.index_cast"(%188) : (i32) -> index
      %190 = "memref.load"(%152, %arg17, %126) : (memref<24x6xf32>, index, index) -> f32
      %191 = "memref.load"(%152, %arg17, %125) : (memref<24x6xf32>, index, index) -> f32
      %192 = "arith.addf"(%190, %191) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %193 = "memref.load"(%152, %arg17, %124) : (memref<24x6xf32>, index, index) -> f32
      %194 = "arith.addf"(%192, %193) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %195 = "memref.load"(%151, %189) : (memref<216xf32>, index) -> f32
      %196 = "arith.addf"(%195, %194) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%196, %151, %189) : (f32, memref<216xf32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%131, %127, %129) ({
    ^bb0(%arg16: index):
      "func.call"(%150, %151, %149) <{callee = @au_generated_candidate_9__direct_site_2843}> : (memref<216xf32>, memref<216xf32>, memref<216xf32>) -> ()
      "func.call"(%149, %150) <{callee = @au_generated_candidate_27__l0u_l1u_l2u_site_2871}> : (memref<216xf32>, memref<216xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %153 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%133, %153) : (i32, memref<i32>) -> ()
    "memref.store"(%133, %153) : (i32, memref<i32>) -> ()
    "func.call"(%150, %153) <{callee = @au_generated_candidate_11__direct_site_2938}> : (memref<216xf32>, memref<i32>) -> ()
    %154 = "memref.load"(%153) : (memref<i32>) -> i32
    "func.call"(%152, %151, %150, %149) <{callee = @pcl_validate_outputs}> : (memref<24x6xf32>, memref<216xf32>, memref<216xf32>, memref<216xf32>) -> ()
    "func.return"(%154) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<216xf32>, memref<216xf32>, memref<216xf32>) -> (), sym_name = "au_generated_candidate_9__direct_site_2843"}> ({
  ^bb0(%arg10: memref<216xf32>, %arg11: memref<216xf32>, %arg12: memref<216xf32>):
    %64 = "arith.constant"() <{value = 6.000000e+00 : f32}> : () -> f32
    %65 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %66 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %67 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    %68 = "arith.constant"() <{value = 0 : index}> : () -> index
    %69 = "arith.constant"() <{value = 4 : index}> : () -> index
    %70 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%68, %69, %70) ({
    ^bb0(%arg13: index):
      %71 = "arith.addi"(%arg13, %70) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %72 = "arith.index_cast"(%71) : (index) -> i32
      %73 = "arith.muli"(%72, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %74 = "arith.addi"(%72, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %75 = "arith.muli"(%74, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %76 = "arith.addi"(%72, %65) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %77 = "arith.muli"(%76, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.for"(%68, %69, %70) ({
      ^bb0(%arg14: index):
        %78 = "arith.addi"(%arg14, %70) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %79 = "arith.index_cast"(%78) : (index) -> i32
        "scf.for"(%68, %69, %70) ({
        ^bb0(%arg15: index):
          %80 = "arith.addi"(%arg15, %70) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %81 = "arith.index_cast"(%80) : (index) -> i32
          %82 = "arith.addi"(%73, %79) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %83 = "arith.muli"(%82, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %84 = "arith.addi"(%83, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %85 = "arith.addi"(%75, %79) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %86 = "arith.muli"(%85, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %87 = "arith.addi"(%86, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %88 = "arith.addi"(%77, %79) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %89 = "arith.muli"(%88, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %90 = "arith.addi"(%89, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %91 = "arith.addi"(%79, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %92 = "arith.addi"(%73, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %93 = "arith.muli"(%92, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %94 = "arith.addi"(%93, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %95 = "arith.addi"(%79, %65) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %96 = "arith.addi"(%73, %95) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %97 = "arith.muli"(%96, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %98 = "arith.addi"(%97, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %99 = "arith.addi"(%81, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %100 = "arith.addi"(%83, %99) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %101 = "arith.addi"(%81, %65) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %102 = "arith.addi"(%83, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %103 = "arith.index_cast"(%87) : (i32) -> index
          %104 = "memref.load"(%arg10, %103) : (memref<216xf32>, index) -> f32
          %105 = "arith.index_cast"(%90) : (i32) -> index
          %106 = "memref.load"(%arg10, %105) : (memref<216xf32>, index) -> f32
          %107 = "arith.addf"(%104, %106) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %108 = "arith.index_cast"(%94) : (i32) -> index
          %109 = "memref.load"(%arg10, %108) : (memref<216xf32>, index) -> f32
          %110 = "arith.addf"(%107, %109) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %111 = "arith.index_cast"(%98) : (i32) -> index
          %112 = "memref.load"(%arg10, %111) : (memref<216xf32>, index) -> f32
          %113 = "arith.addf"(%110, %112) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %114 = "arith.index_cast"(%100) : (i32) -> index
          %115 = "memref.load"(%arg10, %114) : (memref<216xf32>, index) -> f32
          %116 = "arith.addf"(%113, %115) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %117 = "arith.index_cast"(%102) : (i32) -> index
          %118 = "memref.load"(%arg10, %117) : (memref<216xf32>, index) -> f32
          %119 = "arith.addf"(%116, %118) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %120 = "arith.index_cast"(%84) : (i32) -> index
          %121 = "memref.load"(%arg11, %120) : (memref<216xf32>, index) -> f32
          %122 = "arith.addf"(%121, %119) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %123 = "arith.divf"(%122, %64) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%123, %arg12, %120) : (f32, memref<216xf32>, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<216xf32>, memref<216xf32>) -> (), sym_name = "au_generated_candidate_27__l0u_l1u_l2u_site_2871"}> ({
  ^bb0(%arg5: memref<216xf32>, %arg6: memref<216xf32>):
    %48 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    %49 = "arith.constant"() <{value = 0 : index}> : () -> index
    %50 = "arith.constant"() <{value = 4 : index}> : () -> index
    %51 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%49, %50, %51) ({
    ^bb0(%arg7: index):
      %52 = "arith.addi"(%arg7, %51) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %53 = "arith.index_cast"(%52) : (index) -> i32
      "scf.for"(%49, %50, %51) ({
      ^bb0(%arg8: index):
        %54 = "arith.addi"(%arg8, %51) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %55 = "arith.index_cast"(%54) : (index) -> i32
        "scf.for"(%49, %50, %51) ({
        ^bb0(%arg9: index):
          %56 = "arith.addi"(%arg9, %51) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %57 = "arith.index_cast"(%56) : (index) -> i32
          %58 = "arith.muli"(%53, %48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %59 = "arith.addi"(%58, %55) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %60 = "arith.muli"(%59, %48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %61 = "arith.addi"(%60, %57) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %62 = "arith.index_cast"(%61) : (i32) -> index
          %63 = "memref.load"(%arg5, %62) : (memref<216xf32>, index) -> f32
          "memref.store"(%63, %arg6, %62) : (f32, memref<216xf32>, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<216xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_11__direct_site_2938"}> ({
  ^bb0(%arg0: memref<216xf32>, %arg1: memref<i32>):
    %0 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 0 : index}> : () -> index
    %4 = "arith.constant"() <{value = 4 : index}> : () -> index
    %5 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%3, %4, %5) ({
    ^bb0(%arg2: index):
      %6 = "arith.addi"(%arg2, %5) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %7 = "arith.index_cast"(%6) : (index) -> i32
      "scf.for"(%3, %4, %5) ({
      ^bb0(%arg3: index):
        %8 = "arith.addi"(%arg3, %5) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %9 = "arith.index_cast"(%8) : (index) -> i32
        "scf.for"(%3, %4, %5) ({
        ^bb0(%arg4: index):
          %10 = "arith.addi"(%arg4, %5) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %11 = "arith.index_cast"(%10) : (index) -> i32
          %12 = "arith.muli"(%7, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %13 = "arith.addi"(%12, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %14 = "arith.muli"(%13, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %15 = "arith.addi"(%14, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %16 = "arith.addi"(%7, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %17 = "arith.muli"(%16, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %18 = "arith.addi"(%17, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %19 = "arith.muli"(%18, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %20 = "arith.addi"(%19, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %21 = "arith.addi"(%9, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %22 = "arith.addi"(%12, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %23 = "arith.muli"(%22, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %24 = "arith.addi"(%23, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %25 = "arith.addi"(%11, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %26 = "arith.addi"(%14, %25) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %27 = "arith.index_cast"(%15) : (i32) -> index
          %28 = "memref.load"(%arg0, %27) : (memref<216xf32>, index) -> f32
          %29 = "arith.index_cast"(%20) : (i32) -> index
          %30 = "memref.load"(%arg0, %29) : (memref<216xf32>, index) -> f32
          %31 = "arith.mulf"(%28, %30) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %32 = "arith.cmpf"(%31, %0) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %33 = "arith.extui"(%32) : (i1) -> i32
          %34 = "memref.load"(%arg1) : (memref<i32>) -> i32
          %35 = "arith.addi"(%34, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "memref.store"(%35, %arg1) : (i32, memref<i32>) -> ()
          %36 = "arith.index_cast"(%24) : (i32) -> index
          %37 = "memref.load"(%arg0, %36) : (memref<216xf32>, index) -> f32
          %38 = "arith.mulf"(%28, %37) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %39 = "arith.cmpf"(%38, %0) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %40 = "arith.extui"(%39) : (i1) -> i32
          %41 = "arith.addi"(%35, %40) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "memref.store"(%41, %arg1) : (i32, memref<i32>) -> ()
          %42 = "arith.index_cast"(%26) : (i32) -> index
          %43 = "memref.load"(%arg0, %42) : (memref<216xf32>, index) -> f32
          %44 = "arith.mulf"(%28, %43) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %45 = "arith.cmpf"(%44, %0) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %46 = "arith.extui"(%45) : (i1) -> i32
          %47 = "arith.addi"(%41, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "memref.store"(%47, %arg1) : (i32, memref<i32>) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<24x6xf32>, memref<216xf32>, memref<216xf32>, memref<216xf32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
