"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %119 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %120 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %121 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %122 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x8xi32>
    %123 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x8xf32>
    %124 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x3xf32>
    %125 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x3xf32>
    "func.call"(%125) <{callee = @au_generated_candidate_24__l0u_site_2883}> : (memref<32x3xf32>) -> ()
    "func.call"(%124) <{callee = @au_generated_candidate_25__l0u_site_2907}> : (memref<4x3xf32>) -> ()
    %126 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8xi32>
    %127 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<8xf32>
    "func.call"(%123, %122, %125, %124, %127, %126) <{callee = @au_generated_candidate_2__direct_site_2987}> : (memref<4x8xf32>, memref<4x8xi32>, memref<32x3xf32>, memref<4x3xf32>, memref<8xf32>, memref<8xi32>) -> ()
    %128 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%120, %128) : (i32, memref<i32>) -> ()
    "memref.store"(%120, %128) : (i32, memref<i32>) -> ()
    "func.call"(%122, %123, %128) <{callee = @au_generated_candidate_11__direct_site_3018}> : (memref<4x8xi32>, memref<4x8xf32>, memref<i32>) -> ()
    %129 = "memref.load"(%128) : (memref<i32>) -> i32
    %130 = "arith.sitofp"(%129) : (i32) -> f32
    %131 = "arith.cmpi"(%129, %120) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %132 = "scf.if"(%131) ({
      %135 = "arith.negf"(%130) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "scf.yield"(%135) : (f32) -> ()
    }, {
      "scf.yield"(%130) : (f32) -> ()
    }) : (i1) -> f32
    %133 = "arith.cmpf"(%132, %119) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
    %134 = "arith.select"(%133, %129, %121) : (i1, i32, i32) -> i32
    "func.call"(%125, %124, %123, %122) <{callee = @pcl_validate_outputs}> : (memref<32x3xf32>, memref<4x3xf32>, memref<4x8xf32>, memref<4x8xi32>) -> ()
    "func.return"(%134) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_24__l0u_site_2883"}> ({
  ^bb0(%arg18: memref<32x3xf32>):
    %87 = "arith.constant"() <{value = 2 : index}> : () -> index
    %88 = "arith.constant"() <{value = 2.700000e-02 : f32}> : () -> f32
    %89 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %90 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %91 = "arith.constant"() <{value = 19 : i32}> : () -> i32
    %92 = "arith.constant"() <{value = 2.900000e-02 : f32}> : () -> f32
    %93 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %94 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %95 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %96 = "arith.constant"() <{value = 3.100000e-02 : f32}> : () -> f32
    %97 = "arith.constant"() <{value = 101 : i32}> : () -> i32
    %98 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %99 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %100 = "arith.constant"() <{value = 0 : index}> : () -> index
    %101 = "arith.constant"() <{value = 32 : index}> : () -> index
    %102 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%100, %101, %102) ({
    ^bb0(%arg19: index):
      %103 = "arith.index_cast"(%arg19) : (index) -> i32
      %104 = "arith.muli"(%103, %99) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %105 = "arith.addi"(%104, %98) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %106 = "arith.remsi"(%105, %97) : (i32, i32) -> i32
      %107 = "arith.sitofp"(%106) : (i32) -> f32
      %108 = "arith.mulf"(%107, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%108, %arg18, %arg19, %100) : (f32, memref<32x3xf32>, index, index) -> ()
      %109 = "arith.muli"(%103, %95) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %110 = "arith.addi"(%109, %94) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %111 = "arith.remsi"(%110, %93) : (i32, i32) -> i32
      %112 = "arith.sitofp"(%111) : (i32) -> f32
      %113 = "arith.mulf"(%112, %92) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%113, %arg18, %arg19, %102) : (f32, memref<32x3xf32>, index, index) -> ()
      %114 = "arith.muli"(%103, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %115 = "arith.addi"(%114, %90) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %116 = "arith.remsi"(%115, %89) : (i32, i32) -> i32
      %117 = "arith.sitofp"(%116) : (i32) -> f32
      %118 = "arith.mulf"(%117, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%118, %arg18, %arg19, %87) : (f32, memref<32x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<4x3xf32>) -> (), sym_name = "au_generated_candidate_25__l0u_site_2907"}> ({
  ^bb0(%arg16: memref<4x3xf32>):
    %55 = "arith.constant"() <{value = 2 : index}> : () -> index
    %56 = "arith.constant"() <{value = 2.700000e-02 : f32}> : () -> f32
    %57 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %58 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %59 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %60 = "arith.constant"() <{value = 2.900000e-02 : f32}> : () -> f32
    %61 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %62 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %63 = "arith.constant"() <{value = 29 : i32}> : () -> i32
    %64 = "arith.constant"() <{value = 3.100000e-02 : f32}> : () -> f32
    %65 = "arith.constant"() <{value = 101 : i32}> : () -> i32
    %66 = "arith.constant"() <{value = 9 : i32}> : () -> i32
    %67 = "arith.constant"() <{value = 23 : i32}> : () -> i32
    %68 = "arith.constant"() <{value = 0 : index}> : () -> index
    %69 = "arith.constant"() <{value = 4 : index}> : () -> index
    %70 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%68, %69, %70) ({
    ^bb0(%arg17: index):
      %71 = "arith.index_cast"(%arg17) : (index) -> i32
      %72 = "arith.muli"(%71, %67) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %73 = "arith.addi"(%72, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %74 = "arith.remsi"(%73, %65) : (i32, i32) -> i32
      %75 = "arith.sitofp"(%74) : (i32) -> f32
      %76 = "arith.mulf"(%75, %64) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%76, %arg16, %arg17, %68) : (f32, memref<4x3xf32>, index, index) -> ()
      %77 = "arith.muli"(%71, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %78 = "arith.addi"(%77, %62) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %79 = "arith.remsi"(%78, %61) : (i32, i32) -> i32
      %80 = "arith.sitofp"(%79) : (i32) -> f32
      %81 = "arith.mulf"(%80, %60) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%81, %arg16, %arg17, %70) : (f32, memref<4x3xf32>, index, index) -> ()
      %82 = "arith.muli"(%71, %59) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %83 = "arith.addi"(%82, %58) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %84 = "arith.remsi"(%83, %57) : (i32, i32) -> i32
      %85 = "arith.sitofp"(%84) : (i32) -> f32
      %86 = "arith.mulf"(%85, %56) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%86, %arg16, %arg17, %55) : (f32, memref<4x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<4x8xf32>, memref<4x8xi32>, memref<32x3xf32>, memref<4x3xf32>, memref<8xf32>, memref<8xi32>) -> (), sym_name = "au_generated_candidate_2__direct_site_2987"}> ({
  ^bb0(%arg5: memref<4x8xf32>, %arg6: memref<4x8xi32>, %arg7: memref<32x3xf32>, %arg8: memref<4x3xf32>, %arg9: memref<8xf32>, %arg10: memref<8xi32>):
    %16 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = false}> : () -> i1
    %18 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %19 = "arith.constant"() <{value = 2 : index}> : () -> index
    %20 = "arith.constant"() <{value = 32 : index}> : () -> index
    %21 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %22 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    %23 = "arith.constant"() <{value = 8 : index}> : () -> index
    %24 = "arith.constant"() <{value = 0 : index}> : () -> index
    %25 = "arith.constant"() <{value = 4 : index}> : () -> index
    %26 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%24, %25, %26) ({
    ^bb0(%arg11: index):
      "scf.for"(%24, %23, %26) ({
      ^bb0(%arg15: index):
        "memref.store"(%22, %arg5, %arg11, %arg15) : (f32, memref<4x8xf32>, index, index) -> ()
        "memref.store"(%21, %arg6, %arg11, %arg15) : (i32, memref<4x8xi32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%24, %20, %26) ({
      ^bb0(%arg12: index):
        %27 = "arith.index_cast"(%arg12) : (index) -> i32
        %28 = "memref.load"(%arg7, %arg12, %24) : (memref<32x3xf32>, index, index) -> f32
        %29 = "memref.load"(%arg8, %arg11, %24) : (memref<4x3xf32>, index, index) -> f32
        %30 = "arith.subf"(%28, %29) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %31 = "memref.load"(%arg7, %arg12, %26) : (memref<32x3xf32>, index, index) -> f32
        %32 = "memref.load"(%arg8, %arg11, %26) : (memref<4x3xf32>, index, index) -> f32
        %33 = "arith.subf"(%31, %32) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %34 = "memref.load"(%arg7, %arg12, %19) : (memref<32x3xf32>, index, index) -> f32
        %35 = "memref.load"(%arg8, %arg11, %19) : (memref<4x3xf32>, index, index) -> f32
        %36 = "arith.subf"(%34, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %37 = "arith.mulf"(%30, %30) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %38 = "arith.mulf"(%33, %33) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %39 = "arith.addf"(%37, %38) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %40 = "arith.mulf"(%36, %36) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %41 = "arith.addf"(%39, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %42 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
        "memref.store"(%18, %42) : (i32, memref<i32>) -> ()
        "scf.for"(%24, %23, %26) ({
        ^bb0(%arg14: index):
          %45 = "arith.index_cast"(%arg14) : (index) -> i32
          %46 = "memref.load"(%42) : (memref<i32>) -> i32
          %47 = "arith.cmpi"(%46, %18) <{predicate = 0 : i64}> : (i32, i32) -> i1
          %48 = "scf.if"(%47) ({
            %53 = "memref.load"(%arg5, %arg11, %arg14) : (memref<4x8xf32>, index, index) -> f32
            %54 = "arith.cmpf"(%41, %53) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
            "scf.yield"(%54) : (i1) -> ()
          }, {
            "scf.yield"(%17) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%48) ({
            "memref.store"(%41, %arg9, %arg14) : (f32, memref<8xf32>, index) -> ()
            "memref.store"(%27, %arg10, %arg14) : (i32, memref<8xi32>, index) -> ()
            "memref.store"(%16, %42) : (i32, memref<i32>) -> ()
            "scf.yield"() : () -> ()
          }, {
            %49 = "arith.subi"(%45, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %50 = "arith.index_cast"(%49) : (i32) -> index
            %51 = "memref.load"(%arg5, %arg11, %50) : (memref<4x8xf32>, index, index) -> f32
            "memref.store"(%51, %arg9, %arg14) : (f32, memref<8xf32>, index) -> ()
            %52 = "memref.load"(%arg6, %arg11, %50) : (memref<4x8xi32>, index, index) -> i32
            "memref.store"(%52, %arg10, %arg14) : (i32, memref<8xi32>, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i1) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.for"(%24, %23, %26) ({
        ^bb0(%arg13: index):
          %43 = "memref.load"(%arg9, %arg13) : (memref<8xf32>, index) -> f32
          "memref.store"(%43, %arg5, %arg11, %arg13) : (f32, memref<4x8xf32>, index, index) -> ()
          %44 = "memref.load"(%arg10, %arg13) : (memref<8xi32>, index) -> i32
          "memref.store"(%44, %arg6, %arg11, %arg13) : (i32, memref<4x8xi32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<4x8xi32>, memref<4x8xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_11__direct_site_3018"}> ({
  ^bb0(%arg0: memref<4x8xi32>, %arg1: memref<4x8xf32>, %arg2: memref<i32>):
    %0 = "arith.constant"() <{value = 2.560000e+02 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 8 : index}> : () -> index
    %3 = "arith.constant"() <{value = 0 : index}> : () -> index
    %4 = "arith.constant"() <{value = 4 : index}> : () -> index
    %5 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%3, %4, %5) ({
    ^bb0(%arg3: index):
      "scf.for"(%3, %2, %5) ({
      ^bb0(%arg4: index):
        %6 = "memref.load"(%arg0, %arg3, %arg4) : (memref<4x8xi32>, index, index) -> i32
        %7 = "arith.cmpi"(%6, %1) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %8 = "scf.if"(%7) ({
          %11 = "memref.load"(%arg0, %arg3, %arg4) : (memref<4x8xi32>, index, index) -> i32
          %12 = "memref.load"(%arg1, %arg3, %arg4) : (memref<4x8xf32>, index, index) -> f32
          %13 = "arith.mulf"(%12, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %14 = "arith.fptosi"(%13) : (f32) -> i32
          %15 = "arith.addi"(%11, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "scf.yield"(%15) : (i32) -> ()
        }, {
          "scf.yield"(%1) : (i32) -> ()
        }) : (i1) -> i32
        %9 = "memref.load"(%arg2) : (memref<i32>) -> i32
        %10 = "arith.addi"(%9, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%10, %arg2) : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>, memref<4x3xf32>, memref<4x8xf32>, memref<4x8xi32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
