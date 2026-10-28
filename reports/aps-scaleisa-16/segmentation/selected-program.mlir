"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %156 = "arith.constant"() <{value = 2 : index}> : () -> index
    %157 = "arith.constant"() <{value = 1 : index}> : () -> index
    %158 = "arith.constant"() <{value = 8 : index}> : () -> index
    %159 = "arith.constant"() <{value = 0 : index}> : () -> index
    %160 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %161 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x8xi32>
    %162 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x8xf32>
    %163 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8x8xf32>
    "func.call"(%163, %161) <{callee = @au_generated_candidate_21__l0u_l1u_site_255}> : (memref<8x8xf32>, memref<8x8xi32>) -> ()
    "func.call"(%163, %162) <{callee = @au_generated_candidate_10__direct_site_89}> : (memref<8x8xf32>, memref<8x8xf32>) -> ()
    "scf.for"(%159, %156, %157) ({
    ^bb0(%arg20: index):
      "scf.for"(%159, %158, %157) ({
      ^bb0(%arg21: index):
        %166 = "arith.index_cast"(%arg21) : (index) -> i32
        "func.call"(%161, %arg21, %166, %163, %162) <{callee = @au_generated_candidate_0__direct_site_193}> : (memref<8x8xi32>, index, i32, memref<8x8xf32>, memref<8x8xf32>) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %164 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%160, %164) : (i32, memref<i32>) -> ()
    "memref.store"(%160, %164) : (i32, memref<i32>) -> ()
    "func.call"(%161, %164) <{callee = @au_generated_candidate_35__l0u_l1u_l2u_site_221}> : (memref<8x8xi32>, memref<i32>) -> ()
    %165 = "memref.load"(%164) : (memref<i32>) -> i32
    "func.call"(%163, %162, %161) <{callee = @pcl_validate_outputs}> : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xi32>) -> ()
    "func.return"(%165) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<8x8xf32>, memref<8x8xf32>) -> (), sym_name = "au_generated_candidate_10__direct_site_89"}> ({
  ^bb0(%arg16: memref<8x8xf32>, %arg17: memref<8x8xf32>):
    %96 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %97 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %98 = "arith.constant"() <{value = true}> : () -> i1
    %99 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %100 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %101 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %102 = "arith.constant"() <{value = 0 : index}> : () -> index
    %103 = "arith.constant"() <{value = 8 : index}> : () -> index
    %104 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%102, %103, %104) ({
    ^bb0(%arg18: index):
      %105 = "arith.index_cast"(%arg18) : (index) -> i32
      "scf.for"(%102, %103, %104) ({
      ^bb0(%arg19: index):
        %106 = "arith.index_cast"(%arg19) : (index) -> i32
        %107 = "arith.addi"(%106, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %108 = "arith.cmpi"(%107, %100) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %109 = "arith.cmpi"(%107, %99) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %110 = "arith.andi"(%108, %109) : (i1, i1) -> i1
        %111 = "arith.xori"(%110, %98) : (i1, i1) -> i1
        %112 = "arith.andi"(%111, %108) : (i1, i1) -> i1
        %113 = "arith.select"(%110, %99, %100) : (i1, i32, i32) -> i32
        %114 = "arith.select"(%112, %107, %113) : (i1, i32, i32) -> i32
        %115 = "arith.addi"(%106, %97) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %116 = "arith.cmpi"(%115, %100) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %117 = "arith.cmpi"(%115, %99) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %118 = "arith.andi"(%116, %117) : (i1, i1) -> i1
        %119 = "arith.xori"(%118, %98) : (i1, i1) -> i1
        %120 = "arith.andi"(%119, %116) : (i1, i1) -> i1
        %121 = "arith.select"(%118, %99, %100) : (i1, i32, i32) -> i32
        %122 = "arith.select"(%120, %115, %121) : (i1, i32, i32) -> i32
        %123 = "arith.addi"(%105, %101) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %124 = "arith.cmpi"(%123, %100) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %125 = "arith.cmpi"(%123, %99) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %126 = "arith.andi"(%124, %125) : (i1, i1) -> i1
        %127 = "arith.xori"(%126, %98) : (i1, i1) -> i1
        %128 = "arith.andi"(%127, %124) : (i1, i1) -> i1
        %129 = "arith.select"(%126, %99, %100) : (i1, i32, i32) -> i32
        %130 = "arith.select"(%128, %123, %129) : (i1, i32, i32) -> i32
        %131 = "arith.addi"(%105, %97) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %132 = "arith.cmpi"(%131, %100) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %133 = "arith.cmpi"(%131, %99) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %134 = "arith.andi"(%132, %133) : (i1, i1) -> i1
        %135 = "arith.subi"(%98, %134) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
        %136 = "arith.andi"(%135, %132) : (i1, i1) -> i1
        %137 = "arith.select"(%134, %99, %100) : (i1, i32, i32) -> i32
        %138 = "arith.select"(%136, %131, %137) : (i1, i32, i32) -> i32
        %139 = "arith.index_cast"(%122) : (i32) -> index
        %140 = "memref.load"(%arg16, %arg18, %139) : (memref<8x8xf32>, index, index) -> f32
        %141 = "arith.index_cast"(%114) : (i32) -> index
        %142 = "memref.load"(%arg16, %arg18, %141) : (memref<8x8xf32>, index, index) -> f32
        %143 = "arith.subf"(%140, %142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %144 = "arith.cmpf"(%143, %96) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %145 = "scf.if"(%144) ({
          %155 = "arith.negf"(%143) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%155) : (f32) -> ()
        }, {
          "scf.yield"(%143) : (f32) -> ()
        }) : (i1) -> f32
        %146 = "arith.index_cast"(%138) : (i32) -> index
        %147 = "memref.load"(%arg16, %146, %arg19) : (memref<8x8xf32>, index, index) -> f32
        %148 = "arith.index_cast"(%130) : (i32) -> index
        %149 = "memref.load"(%arg16, %148, %arg19) : (memref<8x8xf32>, index, index) -> f32
        %150 = "arith.subf"(%147, %149) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %151 = "arith.cmpf"(%150, %96) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %152 = "scf.if"(%151) ({
          %154 = "arith.negf"(%150) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%154) : (f32) -> ()
        }, {
          "scf.yield"(%150) : (f32) -> ()
        }) : (i1) -> f32
        %153 = "arith.addf"(%145, %152) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%153, %arg17, %arg18, %arg19) : (f32, memref<8x8xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<8x8xi32>, index, i32, memref<8x8xf32>, memref<8x8xf32>) -> (), sym_name = "au_generated_candidate_0__direct_site_193"}> ({
  ^bb0(%arg8: memref<8x8xi32>, %arg9: index, %arg10: i32, %arg11: memref<8x8xf32>, %arg12: memref<8x8xf32>):
    %41 = "arith.constant"() <{value = false}> : () -> i1
    %42 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %43 = "arith.constant"() <{value = true}> : () -> i1
    %44 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %45 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %46 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %47 = "arith.constant"() <{value = 3 : index}> : () -> index
    %48 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    %49 = "arith.constant"() <{value = 0 : index}> : () -> index
    %50 = "arith.constant"() <{value = 8 : index}> : () -> index
    %51 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%49, %50, %51) ({
    ^bb0(%arg13: index):
      %52 = "arith.index_cast"(%arg13) : (index) -> i32
      %53 = "memref.load"(%arg8, %arg9, %arg13) : (memref<8x8xi32>, index, index) -> i32
      %54 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "memref.store"(%53, %54) : (i32, memref<i32>) -> ()
      %55 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%48, %55) : (f32, memref<f32>) -> ()
      "scf.for"(%49, %47, %51) ({
      ^bb0(%arg14: index):
        %57 = "arith.index_cast"(%arg14) : (index) -> i32
        %58 = "arith.addi"(%57, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.for"(%49, %47, %51) ({
        ^bb0(%arg15: index):
          %59 = "arith.index_cast"(%arg15) : (index) -> i32
          %60 = "arith.addi"(%59, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %61 = "arith.addi"(%arg10, %58) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %62 = "arith.cmpi"(%61, %45) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %63 = "arith.cmpi"(%61, %44) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %64 = "arith.andi"(%62, %63) : (i1, i1) -> i1
          %65 = "arith.xori"(%64, %43) : (i1, i1) -> i1
          %66 = "arith.andi"(%65, %62) : (i1, i1) -> i1
          %67 = "arith.select"(%64, %44, %45) : (i1, i32, i32) -> i32
          %68 = "arith.select"(%66, %61, %67) : (i1, i32, i32) -> i32
          %69 = "arith.addi"(%52, %60) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %70 = "arith.cmpi"(%69, %45) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %71 = "arith.cmpi"(%69, %44) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %72 = "arith.andi"(%70, %71) : (i1, i1) -> i1
          %73 = "arith.xori"(%72, %43) : (i1, i1) -> i1
          %74 = "arith.andi"(%73, %70) : (i1, i1) -> i1
          %75 = "arith.select"(%72, %44, %45) : (i1, i32, i32) -> i32
          %76 = "arith.select"(%74, %69, %75) : (i1, i32, i32) -> i32
          %77 = "memref.load"(%arg11, %arg9, %arg13) : (memref<8x8xf32>, index, index) -> f32
          %78 = "arith.index_cast"(%68) : (i32) -> index
          %79 = "arith.index_cast"(%76) : (i32) -> index
          %80 = "memref.load"(%arg11, %78, %79) : (memref<8x8xf32>, index, index) -> f32
          %81 = "arith.subf"(%77, %80) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %82 = "arith.cmpf"(%81, %42) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %83 = "scf.if"(%82) ({
            %95 = "arith.negf"(%81) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
            "scf.yield"(%95) : (f32) -> ()
          }, {
            "scf.yield"(%81) : (f32) -> ()
          }) : (i1) -> f32
          %84 = "memref.load"(%arg12, %78, %79) : (memref<8x8xf32>, index, index) -> f32
          %85 = "arith.addf"(%83, %84) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %86 = "memref.load"(%55) : (memref<f32>) -> f32
          %87 = "arith.cmpf"(%85, %86) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %88 = "scf.if"(%87) ({
            "scf.yield"(%43) : (i1) -> ()
          }, {
            %90 = "arith.cmpf"(%85, %86) <{fastmath = #arith.fastmath<none>, predicate = 1 : i64}> : (f32, f32) -> i1
            %91 = "scf.if"(%90) ({
              %92 = "memref.load"(%arg8, %78, %79) : (memref<8x8xi32>, index, index) -> i32
              %93 = "memref.load"(%54) : (memref<i32>) -> i32
              %94 = "arith.cmpi"(%92, %93) <{predicate = 2 : i64}> : (i32, i32) -> i1
              "scf.yield"(%94) : (i1) -> ()
            }, {
              "scf.yield"(%41) : (i1) -> ()
            }) : (i1) -> i1
            "scf.yield"(%91) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%88) ({
            "memref.store"(%85, %55) : (f32, memref<f32>) -> ()
            %89 = "memref.load"(%arg8, %78, %79) : (memref<8x8xi32>, index, index) -> i32
            "memref.store"(%89, %54) : (i32, memref<i32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %56 = "memref.load"(%54) : (memref<i32>) -> i32
      "memref.store"(%56, %arg8, %arg9, %arg13) : (i32, memref<8x8xi32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<8x8xi32>, memref<i32>) -> (), sym_name = "au_generated_candidate_35__l0u_l1u_l2u_site_221"}> ({
  ^bb0(%arg4: memref<8x8xi32>, %arg5: memref<i32>):
    %33 = "arith.constant"() <{value = 255 : i32}> : () -> i32
    %34 = "arith.constant"() <{value = 0 : index}> : () -> index
    %35 = "arith.constant"() <{value = 8 : index}> : () -> index
    %36 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%34, %35, %36) ({
    ^bb0(%arg6: index):
      "scf.for"(%34, %35, %36) ({
      ^bb0(%arg7: index):
        %37 = "memref.load"(%arg4, %arg6, %arg7) : (memref<8x8xi32>, index, index) -> i32
        %38 = "arith.andi"(%37, %33) : (i32, i32) -> i32
        %39 = "memref.load"(%arg5) : (memref<i32>) -> i32
        %40 = "arith.addi"(%39, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%40, %arg5) : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<8x8xf32>, memref<8x8xi32>) -> (), sym_name = "au_generated_candidate_21__l0u_l1u_site_255"}> ({
  ^bb0(%arg0: memref<8x8xf32>, %arg1: memref<8x8xi32>):
    %0 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 5.000000e-03 : f32}> : () -> f32
    %2 = "arith.constant"() <{value = 9 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 1.000000e-01 : f32}> : () -> f32
    %6 = "arith.constant"() <{value = 30 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 56 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 0 : index}> : () -> index
    %10 = "arith.constant"() <{value = 8 : index}> : () -> index
    %11 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%9, %10, %11) ({
    ^bb0(%arg2: index):
      %12 = "arith.index_cast"(%arg2) : (index) -> i32
      "scf.for"(%9, %10, %11) ({
      ^bb0(%arg3: index):
        %13 = "arith.index_cast"(%arg3) : (index) -> i32
        %14 = "arith.cmpi"(%13, %8) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %15 = "arith.extui"(%14) : (i1) -> i32
        %16 = "arith.cmpi"(%13, %7) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %17 = "arith.extui"(%16) : (i1) -> i32
        %18 = "arith.addi"(%15, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %19 = "arith.cmpi"(%12, %6) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %20 = "arith.extui"(%19) : (i1) -> i32
        %21 = "arith.addi"(%18, %20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %22 = "arith.sitofp"(%21) : (i32) -> f32
        %23 = "arith.mulf"(%22, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %24 = "arith.muli"(%13, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %25 = "arith.muli"(%12, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %26 = "arith.addi"(%24, %25) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %27 = "arith.remsi"(%26, %2) : (i32, i32) -> i32
        %28 = "arith.sitofp"(%27) : (i32) -> f32
        %29 = "arith.mulf"(%28, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %30 = "arith.addf"(%23, %29) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%30, %arg0, %arg2, %arg3) : (f32, memref<8x8xf32>, index, index) -> ()
        %31 = "arith.muli"(%12, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %32 = "arith.addi"(%31, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%32, %arg1, %arg2, %arg3) : (i32, memref<8x8xi32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xi32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
