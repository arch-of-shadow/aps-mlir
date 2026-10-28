"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %89 = "arith.constant"() <{value = 4 : index}> : () -> index
    %90 = "arith.constant"() <{value = 1.000000e+03 : f32}> : () -> f32
    %91 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %92 = "arith.constant"() <{value = 1.200000e+03 : f32}> : () -> f32
    %93 = "arith.constant"() <{value = 1.400000e+03 : f32}> : () -> f32
    %94 = "arith.constant"() <{value = 5.000000e+02 : f32}> : () -> f32
    %95 = "arith.constant"() <{value = 0 : index}> : () -> index
    %96 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %97 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %98 = "arith.constant"() <{value = 5.000000e-01 : f32}> : () -> f32
    %99 = "arith.constant"() <{value = 3.200000e+01 : f32}> : () -> f32
    %100 = "arith.constant"() <{value = 1 : index}> : () -> index
    %101 = "arith.constant"() <{value = 2 : index}> : () -> index
    %102 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4xf32>
    %103 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<3xf32>
    %104 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x3xf32>
    %105 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x3xf32>
    "func.call"(%105, %104) <{callee = @au_generated_candidate_8__l0u_site_2884}> : (memref<32x3xf32>, memref<32x3xf32>) -> ()
    %106 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
    "memref.store"(%96, %106) : (f32, memref<f32>) -> ()
    %107 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
    "memref.store"(%96, %107) : (f32, memref<f32>) -> ()
    %108 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
    "memref.store"(%96, %108) : (f32, memref<f32>) -> ()
    "memref.store"(%96, %108) : (f32, memref<f32>) -> ()
    "memref.store"(%96, %107) : (f32, memref<f32>) -> ()
    "memref.store"(%96, %106) : (f32, memref<f32>) -> ()
    "scf.for"(%95, %89, %100) ({
    ^bb0(%arg15: index):
      %136 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%96, %136) : (f32, memref<f32>) -> ()
      %137 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%96, %137) : (f32, memref<f32>) -> ()
      %138 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%96, %138) : (f32, memref<f32>) -> ()
      %139 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%96, %139) : (f32, memref<f32>) -> ()
      "func.call"(%108, %107, %106, %105, %104, %136, %137, %138, %139) <{callee = @au_generated_candidate_1__direct_site_2977}> : (memref<f32>, memref<f32>, memref<f32>, memref<32x3xf32>, memref<32x3xf32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) -> ()
      %140 = "memref.load"(%136) : (memref<f32>) -> f32
      %141 = "arith.mulf"(%140, %98) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %142 = "arith.divf"(%141, %99) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %143 = "memref.load"(%108) : (memref<f32>) -> f32
      %144 = "arith.addf"(%143, %142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%144, %108) : (f32, memref<f32>) -> ()
      %145 = "memref.load"(%137) : (memref<f32>) -> f32
      %146 = "arith.mulf"(%145, %98) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %147 = "arith.divf"(%146, %99) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %148 = "memref.load"(%107) : (memref<f32>) -> f32
      %149 = "arith.addf"(%148, %147) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%149, %107) : (f32, memref<f32>) -> ()
      %150 = "memref.load"(%138) : (memref<f32>) -> f32
      %151 = "arith.mulf"(%150, %98) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %152 = "arith.divf"(%151, %99) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %153 = "memref.load"(%106) : (memref<f32>) -> f32
      %154 = "arith.addf"(%153, %152) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%154, %106) : (f32, memref<f32>) -> ()
      %155 = "memref.load"(%139) : (memref<f32>) -> f32
      %156 = "arith.divf"(%155, %99) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%156, %102, %arg15) : (f32, memref<4xf32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %109 = "memref.load"(%108) : (memref<f32>) -> f32
    "memref.store"(%109, %103, %95) : (f32, memref<3xf32>, index) -> ()
    %110 = "memref.load"(%107) : (memref<f32>) -> f32
    "memref.store"(%110, %103, %100) : (f32, memref<3xf32>, index) -> ()
    %111 = "memref.load"(%106) : (memref<f32>) -> f32
    "memref.store"(%111, %103, %101) : (f32, memref<3xf32>, index) -> ()
    %112 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%97, %112) : (i32, memref<i32>) -> ()
    "memref.store"(%97, %112) : (i32, memref<i32>) -> ()
    %113 = "memref.load"(%103, %95) : (memref<3xf32>, index) -> f32
    %114 = "arith.mulf"(%113, %90) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %115 = "arith.fptosi"(%114) : (f32) -> i32
    "memref.store"(%115, %112) : (i32, memref<i32>) -> ()
    %116 = "memref.load"(%103, %100) : (memref<3xf32>, index) -> f32
    %117 = "arith.mulf"(%116, %92) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %118 = "arith.fptosi"(%117) : (f32) -> i32
    %119 = "arith.addi"(%115, %118) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%119, %112) : (i32, memref<i32>) -> ()
    %120 = "memref.load"(%103, %101) : (memref<3xf32>, index) -> f32
    %121 = "arith.mulf"(%120, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %122 = "arith.fptosi"(%121) : (f32) -> i32
    %123 = "arith.addi"(%119, %122) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%123, %112) : (i32, memref<i32>) -> ()
    "scf.for"(%95, %89, %100) ({
    ^bb0(%arg14: index):
      %131 = "memref.load"(%102, %arg14) : (memref<4xf32>, index) -> f32
      %132 = "arith.mulf"(%131, %94) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %133 = "arith.fptosi"(%132) : (f32) -> i32
      %134 = "memref.load"(%112) : (memref<i32>) -> i32
      %135 = "arith.addi"(%134, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%135, %112) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %124 = "memref.load"(%112) : (memref<i32>) -> i32
    %125 = "arith.sitofp"(%124) : (i32) -> f32
    %126 = "arith.cmpi"(%124, %97) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %127 = "scf.if"(%126) ({
      %130 = "arith.negf"(%125) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "scf.yield"(%130) : (f32) -> ()
    }, {
      "scf.yield"(%125) : (f32) -> ()
    }) : (i1) -> f32
    %128 = "arith.cmpf"(%127, %96) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
    %129 = "arith.select"(%128, %124, %91) : (i1, i32, i32) -> i32
    "func.call"(%105, %104, %103, %102) <{callee = @pcl_validate_outputs}> : (memref<32x3xf32>, memref<32x3xf32>, memref<3xf32>, memref<4xf32>) -> ()
    "func.return"(%129) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>, memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_8__l0u_site_2884"}> ({
  ^bb0(%arg11: memref<32x3xf32>, %arg12: memref<32x3xf32>):
    %57 = "arith.constant"() <{value = 6.000000e-02 : f32}> : () -> f32
    %58 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
    %59 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %60 = "arith.constant"() <{value = 2 : index}> : () -> index
    %61 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    %62 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %63 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %64 = "arith.constant"() <{value = 2.500000e-02 : f32}> : () -> f32
    %65 = "arith.constant"() <{value = 37 : i32}> : () -> i32
    %66 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %67 = "arith.constant"() <{value = 3.000000e-02 : f32}> : () -> f32
    %68 = "arith.constant"() <{value = 41 : i32}> : () -> i32
    %69 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %70 = "arith.constant"() <{value = 0 : index}> : () -> index
    %71 = "arith.constant"() <{value = 32 : index}> : () -> index
    %72 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%70, %71, %72) ({
    ^bb0(%arg13: index):
      %73 = "arith.index_cast"(%arg13) : (index) -> i32
      %74 = "arith.muli"(%73, %69) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %75 = "arith.remsi"(%74, %68) : (i32, i32) -> i32
      %76 = "arith.sitofp"(%75) : (i32) -> f32
      %77 = "arith.mulf"(%76, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %78 = "arith.muli"(%73, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %79 = "arith.remsi"(%78, %65) : (i32, i32) -> i32
      %80 = "arith.sitofp"(%79) : (i32) -> f32
      %81 = "arith.mulf"(%80, %64) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %82 = "arith.muli"(%73, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %83 = "arith.remsi"(%82, %62) : (i32, i32) -> i32
      %84 = "arith.sitofp"(%83) : (i32) -> f32
      %85 = "arith.mulf"(%84, %61) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%77, %arg11, %arg13, %70) : (f32, memref<32x3xf32>, index, index) -> ()
      "memref.store"(%81, %arg11, %arg13, %72) : (f32, memref<32x3xf32>, index, index) -> ()
      "memref.store"(%85, %arg11, %arg13, %60) : (f32, memref<32x3xf32>, index, index) -> ()
      %86 = "arith.addf"(%77, %59) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%86, %arg12, %arg13, %70) : (f32, memref<32x3xf32>, index, index) -> ()
      %87 = "arith.subf"(%81, %58) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%87, %arg12, %arg13, %72) : (f32, memref<32x3xf32>, index, index) -> ()
      %88 = "arith.addf"(%85, %57) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%88, %arg12, %arg13, %60) : (f32, memref<32x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<f32>, memref<f32>, memref<f32>, memref<32x3xf32>, memref<32x3xf32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) -> (), sym_name = "au_generated_candidate_1__direct_site_2977"}> ({
  ^bb0(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<32x3xf32>, %arg4: memref<32x3xf32>, %arg5: memref<f32>, %arg6: memref<f32>, %arg7: memref<f32>, %arg8: memref<f32>):
    %0 = "arith.constant"() <{value = 2 : index}> : () -> index
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = 0 : index}> : () -> index
    %4 = "arith.constant"() <{value = 32 : index}> : () -> index
    %5 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%3, %4, %5) ({
    ^bb0(%arg9: index):
      %6 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%2, %6) : (f32, memref<f32>) -> ()
      %7 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "memref.store"(%1, %7) : (i32, memref<i32>) -> ()
      "scf.for"(%3, %4, %5) ({
      ^bb0(%arg10: index):
        %34 = "arith.index_cast"(%arg10) : (index) -> i32
        %35 = "memref.load"(%arg0) : (memref<f32>) -> f32
        %36 = "memref.load"(%arg1) : (memref<f32>) -> f32
        %37 = "memref.load"(%arg2) : (memref<f32>) -> f32
        %38 = "memref.load"(%arg3, %arg9, %3) : (memref<32x3xf32>, index, index) -> f32
        %39 = "arith.addf"(%38, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %40 = "memref.load"(%arg4, %arg10, %3) : (memref<32x3xf32>, index, index) -> f32
        %41 = "arith.subf"(%39, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %42 = "memref.load"(%arg3, %arg9, %5) : (memref<32x3xf32>, index, index) -> f32
        %43 = "arith.addf"(%42, %36) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %44 = "memref.load"(%arg4, %arg10, %5) : (memref<32x3xf32>, index, index) -> f32
        %45 = "arith.subf"(%43, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %46 = "memref.load"(%arg3, %arg9, %0) : (memref<32x3xf32>, index, index) -> f32
        %47 = "arith.addf"(%46, %37) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %48 = "memref.load"(%arg4, %arg10, %0) : (memref<32x3xf32>, index, index) -> f32
        %49 = "arith.subf"(%47, %48) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %50 = "arith.mulf"(%41, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %51 = "arith.mulf"(%45, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %52 = "arith.addf"(%50, %51) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %53 = "arith.mulf"(%49, %49) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %54 = "arith.addf"(%52, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %55 = "memref.load"(%6) : (memref<f32>) -> f32
        %56 = "arith.cmpf"(%54, %55) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        "scf.if"(%56) ({
          "memref.store"(%54, %6) : (f32, memref<f32>) -> ()
          "memref.store"(%34, %7) : (i32, memref<i32>) -> ()
          "scf.yield"() : () -> ()
        }, {
        }) : (i1) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %8 = "memref.load"(%7) : (memref<i32>) -> i32
      %9 = "arith.index_cast"(%8) : (i32) -> index
      %10 = "memref.load"(%arg4, %9, %3) : (memref<32x3xf32>, index, index) -> f32
      %11 = "memref.load"(%arg3, %arg9, %3) : (memref<32x3xf32>, index, index) -> f32
      %12 = "memref.load"(%arg0) : (memref<f32>) -> f32
      %13 = "arith.addf"(%11, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %14 = "arith.subf"(%10, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %15 = "memref.load"(%arg5) : (memref<f32>) -> f32
      %16 = "arith.addf"(%15, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%16, %arg5) : (f32, memref<f32>) -> ()
      %17 = "memref.load"(%arg4, %9, %5) : (memref<32x3xf32>, index, index) -> f32
      %18 = "memref.load"(%arg3, %arg9, %5) : (memref<32x3xf32>, index, index) -> f32
      %19 = "memref.load"(%arg1) : (memref<f32>) -> f32
      %20 = "arith.addf"(%18, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %21 = "arith.subf"(%17, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %22 = "memref.load"(%arg6) : (memref<f32>) -> f32
      %23 = "arith.addf"(%22, %21) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%23, %arg6) : (f32, memref<f32>) -> ()
      %24 = "memref.load"(%arg4, %9, %0) : (memref<32x3xf32>, index, index) -> f32
      %25 = "memref.load"(%arg3, %arg9, %0) : (memref<32x3xf32>, index, index) -> f32
      %26 = "memref.load"(%arg2) : (memref<f32>) -> f32
      %27 = "arith.addf"(%25, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %28 = "arith.subf"(%24, %27) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %29 = "memref.load"(%arg7) : (memref<f32>) -> f32
      %30 = "arith.addf"(%29, %28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%30, %arg7) : (f32, memref<f32>) -> ()
      %31 = "memref.load"(%6) : (memref<f32>) -> f32
      %32 = "memref.load"(%arg8) : (memref<f32>) -> f32
      %33 = "arith.addf"(%32, %31) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%33, %arg8) : (f32, memref<f32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>, memref<32x3xf32>, memref<3xf32>, memref<4xf32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
