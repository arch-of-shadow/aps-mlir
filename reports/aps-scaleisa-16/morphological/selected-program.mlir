"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %163 = "arith.constant"() <{value = 2 : index}> : () -> index
    %164 = "arith.constant"() <{value = 1 : index}> : () -> index
    %165 = "arith.constant"() <{value = 12 : index}> : () -> index
    %166 = "arith.constant"() <{value = 0 : index}> : () -> index
    %167 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %168 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %169 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    %170 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %171 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %172 = "arith.constant"() <{value = 51 : i32}> : () -> i32
    %173 = "arith.constant"() <{value = -53 : i32}> : () -> i32
    %174 = "arith.constant"() <{value = -31 : i32}> : () -> i32
    %175 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<12x12xf32>
    %176 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<12x12xf32>
    %177 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<12x12xf32>
    %178 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<12x12xf32>
    "scf.for"(%166, %165, %164) ({
    ^bb0(%arg36: index):
      %185 = "arith.index_cast"(%arg36) : (index) -> i32
      %186 = "arith.addi"(%185, %173) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %187 = "arith.muli"(%186, %186) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %188 = "arith.muli"(%185, %171) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.for"(%166, %165, %163) ({
      ^bb0(%arg37: index):
        %189 = "arith.addi"(%arg37, %163) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        "scf.for"(%arg37, %189, %164) ({
        ^bb0(%arg38: index):
          %190 = "arith.index_cast"(%arg38) : (index) -> i32
          %191 = "arith.addi"(%190, %174) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %192 = "arith.muli"(%191, %191) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %193 = "arith.addi"(%192, %187) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %194 = "arith.remsi"(%193, %168) : (i32, i32) -> i32
          %195 = "arith.muli"(%190, %170) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %196 = "arith.addi"(%195, %188) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %197 = "arith.addi"(%196, %194) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %198 = "arith.remsi"(%197, %172) : (i32, i32) -> i32
          %199 = "arith.sitofp"(%198) : (i32) -> f32
          %200 = "arith.mulf"(%199, %169) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%200, %178, %arg36, %arg38) : (f32, memref<12x12xf32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%166, %165, %164) ({
    ^bb0(%arg35: index):
      %184 = "arith.index_cast"(%arg35) : (index) -> i32
      "func.call"(%178, %arg35, %184, %177) <{callee = @au_generated_candidate_9__direct_site_12631}> : (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%166, %165, %164) ({
    ^bb0(%arg34: index):
      %183 = "arith.index_cast"(%arg34) : (index) -> i32
      "func.call"(%177, %arg34, %183, %176) <{callee = @au_generated_candidate_8__direct_site_12690}> : (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%166, %165, %164) ({
    ^bb0(%arg33: index):
      %182 = "arith.index_cast"(%arg33) : (index) -> i32
      "func.call"(%176, %arg33, %182, %177) <{callee = @au_generated_candidate_8__direct_site_12749}> : (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%166, %165, %164) ({
    ^bb0(%arg32: index):
      %181 = "arith.index_cast"(%arg32) : (index) -> i32
      "func.call"(%177, %arg32, %181, %175) <{callee = @au_generated_candidate_9__direct_site_12808}> : (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %179 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%167, %179) : (i32, memref<i32>) -> ()
    "memref.store"(%167, %179) : (i32, memref<i32>) -> ()
    "func.call"(%175, %179) <{callee = @au_generated_candidate_40__direct_site_12836}> : (memref<12x12xf32>, memref<i32>) -> ()
    %180 = "memref.load"(%179) : (memref<i32>) -> i32
    "func.call"(%178, %177, %176, %175) <{callee = @pcl_validate_outputs}> : (memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>) -> ()
    "func.return"(%180) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> (), sym_name = "au_generated_candidate_9__direct_site_12631"}> ({
  ^bb0(%arg25: memref<12x12xf32>, %arg26: index, %arg27: i32, %arg28: memref<12x12xf32>):
    %126 = "arith.constant"() <{value = true}> : () -> i1
    %127 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %128 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %129 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %130 = "arith.constant"() <{value = 5 : index}> : () -> index
    %131 = "arith.constant"() <{value = 0 : index}> : () -> index
    %132 = "arith.constant"() <{value = 12 : index}> : () -> index
    %133 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%131, %132, %133) ({
    ^bb0(%arg29: index):
      %134 = "arith.index_cast"(%arg29) : (index) -> i32
      %135 = "memref.load"(%arg25, %arg26, %arg29) : (memref<12x12xf32>, index, index) -> f32
      %136 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%135, %136) : (f32, memref<f32>) -> ()
      "scf.for"(%131, %130, %133) ({
      ^bb0(%arg30: index):
        %138 = "arith.index_cast"(%arg30) : (index) -> i32
        %139 = "arith.addi"(%138, %129) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.for"(%131, %130, %133) ({
        ^bb0(%arg31: index):
          %140 = "arith.index_cast"(%arg31) : (index) -> i32
          %141 = "arith.addi"(%140, %129) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %142 = "arith.addi"(%arg27, %139) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %143 = "arith.cmpi"(%142, %128) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %144 = "arith.cmpi"(%142, %127) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %145 = "arith.andi"(%143, %144) : (i1, i1) -> i1
          %146 = "arith.xori"(%145, %126) : (i1, i1) -> i1
          %147 = "arith.andi"(%146, %143) : (i1, i1) -> i1
          %148 = "arith.select"(%145, %127, %128) : (i1, i32, i32) -> i32
          %149 = "arith.select"(%147, %142, %148) : (i1, i32, i32) -> i32
          %150 = "arith.addi"(%134, %141) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %151 = "arith.cmpi"(%150, %128) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %152 = "arith.cmpi"(%150, %127) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %153 = "arith.andi"(%151, %152) : (i1, i1) -> i1
          %154 = "arith.xori"(%153, %126) : (i1, i1) -> i1
          %155 = "arith.andi"(%154, %151) : (i1, i1) -> i1
          %156 = "arith.select"(%153, %127, %128) : (i1, i32, i32) -> i32
          %157 = "arith.select"(%155, %150, %156) : (i1, i32, i32) -> i32
          %158 = "arith.index_cast"(%149) : (i32) -> index
          %159 = "arith.index_cast"(%157) : (i32) -> index
          %160 = "memref.load"(%arg25, %158, %159) : (memref<12x12xf32>, index, index) -> f32
          %161 = "memref.load"(%136) : (memref<f32>) -> f32
          %162 = "arith.cmpf"(%160, %161) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          "scf.if"(%162) ({
            "memref.store"(%160, %136) : (f32, memref<f32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %137 = "memref.load"(%136) : (memref<f32>) -> f32
      "memref.store"(%137, %arg28, %arg26, %arg29) : (f32, memref<12x12xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> (), sym_name = "au_generated_candidate_8__direct_site_12690"}> ({
  ^bb0(%arg18: memref<12x12xf32>, %arg19: index, %arg20: i32, %arg21: memref<12x12xf32>):
    %89 = "arith.constant"() <{value = true}> : () -> i1
    %90 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %91 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %92 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %93 = "arith.constant"() <{value = 5 : index}> : () -> index
    %94 = "arith.constant"() <{value = 0 : index}> : () -> index
    %95 = "arith.constant"() <{value = 12 : index}> : () -> index
    %96 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%94, %95, %96) ({
    ^bb0(%arg22: index):
      %97 = "arith.index_cast"(%arg22) : (index) -> i32
      %98 = "memref.load"(%arg18, %arg19, %arg22) : (memref<12x12xf32>, index, index) -> f32
      %99 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%98, %99) : (f32, memref<f32>) -> ()
      "scf.for"(%94, %93, %96) ({
      ^bb0(%arg23: index):
        %101 = "arith.index_cast"(%arg23) : (index) -> i32
        %102 = "arith.addi"(%101, %92) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.for"(%94, %93, %96) ({
        ^bb0(%arg24: index):
          %103 = "arith.index_cast"(%arg24) : (index) -> i32
          %104 = "arith.addi"(%103, %92) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %105 = "arith.addi"(%arg20, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %106 = "arith.cmpi"(%105, %91) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %107 = "arith.cmpi"(%105, %90) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %108 = "arith.andi"(%106, %107) : (i1, i1) -> i1
          %109 = "arith.xori"(%108, %89) : (i1, i1) -> i1
          %110 = "arith.andi"(%109, %106) : (i1, i1) -> i1
          %111 = "arith.select"(%108, %90, %91) : (i1, i32, i32) -> i32
          %112 = "arith.select"(%110, %105, %111) : (i1, i32, i32) -> i32
          %113 = "arith.addi"(%97, %104) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %114 = "arith.cmpi"(%113, %91) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %115 = "arith.cmpi"(%113, %90) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %116 = "arith.andi"(%114, %115) : (i1, i1) -> i1
          %117 = "arith.xori"(%116, %89) : (i1, i1) -> i1
          %118 = "arith.andi"(%117, %114) : (i1, i1) -> i1
          %119 = "arith.select"(%116, %90, %91) : (i1, i32, i32) -> i32
          %120 = "arith.select"(%118, %113, %119) : (i1, i32, i32) -> i32
          %121 = "arith.index_cast"(%112) : (i32) -> index
          %122 = "arith.index_cast"(%120) : (i32) -> index
          %123 = "memref.load"(%arg18, %121, %122) : (memref<12x12xf32>, index, index) -> f32
          %124 = "memref.load"(%99) : (memref<f32>) -> f32
          %125 = "arith.cmpf"(%123, %124) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
          "scf.if"(%125) ({
            "memref.store"(%123, %99) : (f32, memref<f32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %100 = "memref.load"(%99) : (memref<f32>) -> f32
      "memref.store"(%100, %arg21, %arg19, %arg22) : (f32, memref<12x12xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> (), sym_name = "au_generated_candidate_8__direct_site_12749"}> ({
  ^bb0(%arg11: memref<12x12xf32>, %arg12: index, %arg13: i32, %arg14: memref<12x12xf32>):
    %52 = "arith.constant"() <{value = true}> : () -> i1
    %53 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %54 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %55 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %56 = "arith.constant"() <{value = 5 : index}> : () -> index
    %57 = "arith.constant"() <{value = 0 : index}> : () -> index
    %58 = "arith.constant"() <{value = 12 : index}> : () -> index
    %59 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%57, %58, %59) ({
    ^bb0(%arg15: index):
      %60 = "arith.index_cast"(%arg15) : (index) -> i32
      %61 = "memref.load"(%arg11, %arg12, %arg15) : (memref<12x12xf32>, index, index) -> f32
      %62 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%61, %62) : (f32, memref<f32>) -> ()
      "scf.for"(%57, %56, %59) ({
      ^bb0(%arg16: index):
        %64 = "arith.index_cast"(%arg16) : (index) -> i32
        %65 = "arith.addi"(%64, %55) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.for"(%57, %56, %59) ({
        ^bb0(%arg17: index):
          %66 = "arith.index_cast"(%arg17) : (index) -> i32
          %67 = "arith.addi"(%66, %55) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %68 = "arith.addi"(%arg13, %65) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %69 = "arith.cmpi"(%68, %54) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %70 = "arith.cmpi"(%68, %53) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %71 = "arith.andi"(%69, %70) : (i1, i1) -> i1
          %72 = "arith.xori"(%71, %52) : (i1, i1) -> i1
          %73 = "arith.andi"(%72, %69) : (i1, i1) -> i1
          %74 = "arith.select"(%71, %53, %54) : (i1, i32, i32) -> i32
          %75 = "arith.select"(%73, %68, %74) : (i1, i32, i32) -> i32
          %76 = "arith.addi"(%60, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %77 = "arith.cmpi"(%76, %54) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %78 = "arith.cmpi"(%76, %53) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %79 = "arith.andi"(%77, %78) : (i1, i1) -> i1
          %80 = "arith.xori"(%79, %52) : (i1, i1) -> i1
          %81 = "arith.andi"(%80, %77) : (i1, i1) -> i1
          %82 = "arith.select"(%79, %53, %54) : (i1, i32, i32) -> i32
          %83 = "arith.select"(%81, %76, %82) : (i1, i32, i32) -> i32
          %84 = "arith.index_cast"(%75) : (i32) -> index
          %85 = "arith.index_cast"(%83) : (i32) -> index
          %86 = "memref.load"(%arg11, %84, %85) : (memref<12x12xf32>, index, index) -> f32
          %87 = "memref.load"(%62) : (memref<f32>) -> f32
          %88 = "arith.cmpf"(%86, %87) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
          "scf.if"(%88) ({
            "memref.store"(%86, %62) : (f32, memref<f32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %63 = "memref.load"(%62) : (memref<f32>) -> f32
      "memref.store"(%63, %arg14, %arg12, %arg15) : (f32, memref<12x12xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> (), sym_name = "au_generated_candidate_9__direct_site_12808"}> ({
  ^bb0(%arg4: memref<12x12xf32>, %arg5: index, %arg6: i32, %arg7: memref<12x12xf32>):
    %15 = "arith.constant"() <{value = true}> : () -> i1
    %16 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %18 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %19 = "arith.constant"() <{value = 5 : index}> : () -> index
    %20 = "arith.constant"() <{value = 0 : index}> : () -> index
    %21 = "arith.constant"() <{value = 12 : index}> : () -> index
    %22 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%20, %21, %22) ({
    ^bb0(%arg8: index):
      %23 = "arith.index_cast"(%arg8) : (index) -> i32
      %24 = "memref.load"(%arg4, %arg5, %arg8) : (memref<12x12xf32>, index, index) -> f32
      %25 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%24, %25) : (f32, memref<f32>) -> ()
      "scf.for"(%20, %19, %22) ({
      ^bb0(%arg9: index):
        %27 = "arith.index_cast"(%arg9) : (index) -> i32
        %28 = "arith.addi"(%27, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.for"(%20, %19, %22) ({
        ^bb0(%arg10: index):
          %29 = "arith.index_cast"(%arg10) : (index) -> i32
          %30 = "arith.addi"(%29, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %31 = "arith.addi"(%arg6, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %32 = "arith.cmpi"(%31, %17) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %33 = "arith.cmpi"(%31, %16) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %34 = "arith.andi"(%32, %33) : (i1, i1) -> i1
          %35 = "arith.xori"(%34, %15) : (i1, i1) -> i1
          %36 = "arith.andi"(%35, %32) : (i1, i1) -> i1
          %37 = "arith.select"(%34, %16, %17) : (i1, i32, i32) -> i32
          %38 = "arith.select"(%36, %31, %37) : (i1, i32, i32) -> i32
          %39 = "arith.addi"(%23, %30) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %40 = "arith.cmpi"(%39, %17) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %41 = "arith.cmpi"(%39, %16) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %42 = "arith.andi"(%40, %41) : (i1, i1) -> i1
          %43 = "arith.xori"(%42, %15) : (i1, i1) -> i1
          %44 = "arith.andi"(%43, %40) : (i1, i1) -> i1
          %45 = "arith.select"(%42, %16, %17) : (i1, i32, i32) -> i32
          %46 = "arith.select"(%44, %39, %45) : (i1, i32, i32) -> i32
          %47 = "arith.index_cast"(%38) : (i32) -> index
          %48 = "arith.index_cast"(%46) : (i32) -> index
          %49 = "memref.load"(%arg4, %47, %48) : (memref<12x12xf32>, index, index) -> f32
          %50 = "memref.load"(%25) : (memref<f32>) -> f32
          %51 = "arith.cmpf"(%49, %50) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          "scf.if"(%51) ({
            "memref.store"(%49, %25) : (f32, memref<f32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %26 = "memref.load"(%25) : (memref<f32>) -> f32
      "memref.store"(%26, %arg7, %arg5, %arg8) : (f32, memref<12x12xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<12x12xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_40__direct_site_12836"}> ({
  ^bb0(%arg0: memref<12x12xf32>, %arg1: memref<i32>):
    %0 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1.280000e+02 : f32}> : () -> f32
    %2 = "arith.constant"() <{value = 0 : index}> : () -> index
    %3 = "arith.constant"() <{value = 12 : index}> : () -> index
    %4 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%2, %3, %4) ({
    ^bb0(%arg2: index):
      %5 = "arith.index_cast"(%arg2) : (index) -> i32
      "scf.for"(%2, %3, %4) ({
      ^bb0(%arg3: index):
        %6 = "arith.index_cast"(%arg3) : (index) -> i32
        %7 = "memref.load"(%arg0, %arg2, %arg3) : (memref<12x12xf32>, index, index) -> f32
        %8 = "arith.mulf"(%7, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %9 = "arith.fptosi"(%8) : (f32) -> i32
        %10 = "arith.addi"(%6, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %11 = "arith.andi"(%10, %0) : (i32, i32) -> i32
        %12 = "arith.addi"(%9, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %13 = "memref.load"(%arg1) : (memref<i32>) -> i32
        %14 = "arith.addi"(%13, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%14, %arg1) : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
