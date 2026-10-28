"builtin.module"() ({
  "memref.global"() <{initial_value, sym_name = "bench_input_0", type = memref<40xi32>}> : () -> ()
  "memref.global"() <{initial_value, sym_name = "bench_input_1", type = memref<40xi32>}> : () -> ()
  "memref.global"() <{initial_value, sym_name = "bench_input_2", type = memref<40x3xf32>}> : () -> ()
  "memref.global"() <{initial_value, sym_name = "bench_input_3", type = memref<40x3xf32>}> : () -> ()
  "memref.global"() <{initial_value, sym_name = "bench_input_4", type = memref<24x3xf32>}> : () -> ()
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %74 = "arith.constant"() <{value = 2 : index}> : () -> index
    %75 = "arith.constant"() <{value = 8 : index}> : () -> index
    %76 = "arith.constant"() <{value = 24 : index}> : () -> index
    %77 = "arith.constant"() <{value = 3 : index}> : () -> index
    %78 = "arith.constant"() <{value = 1 : index}> : () -> index
    %79 = "arith.constant"() <{value = 40 : index}> : () -> index
    %80 = "arith.constant"() <{value = 0 : index}> : () -> index
    %81 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %82 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %83 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %84 = "arith.constant"() <{value = 1.000000e-03 : f32}> : () -> f32
    %85 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<24x3xf32>
    %86 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<40x3xf32>
    %87 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<40x3xf32>
    %88 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<40xi32>
    %89 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<40xi32>
    %90 = "memref.get_global"() <{name = @bench_input_0}> : () -> memref<40xi32>
    "scf.for"(%80, %79, %78) ({
    ^bb0(%arg33: index):
      %106 = "memref.load"(%90, %arg33) : (memref<40xi32>, index) -> i32
      "memref.store"(%106, %89, %arg33) : (i32, memref<40xi32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %91 = "memref.get_global"() <{name = @bench_input_1}> : () -> memref<40xi32>
    "scf.for"(%80, %79, %78) ({
    ^bb0(%arg32: index):
      %105 = "memref.load"(%91, %arg32) : (memref<40xi32>, index) -> i32
      "memref.store"(%105, %88, %arg32) : (i32, memref<40xi32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %92 = "memref.get_global"() <{name = @bench_input_2}> : () -> memref<40x3xf32>
    "func.call"(%92, %87) <{callee = @au_generated_candidate_36__l0u_l1u_site_7891}> : (memref<40x3xf32>, memref<40x3xf32>) -> ()
    %93 = "memref.get_global"() <{name = @bench_input_3}> : () -> memref<40x3xf32>
    "func.call"(%93, %86) <{callee = @au_generated_candidate_36__l0u_l1u_site_7904}> : (memref<40x3xf32>, memref<40x3xf32>) -> ()
    %94 = "memref.get_global"() <{name = @bench_input_4}> : () -> memref<24x3xf32>
    "func.call"(%94, %85) <{callee = @au_generated_candidate_38__l0u_l1u_site_7917}> : (memref<24x3xf32>, memref<24x3xf32>) -> ()
    %95 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<24x3xf32>
    %96 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<24x3xf32>
    "scf.for"(%80, %75, %78) ({
    ^bb0(%arg29: index):
      "scf.for"(%80, %77, %78) ({
      ^bb0(%arg30: index):
        "scf.for"(%80, %76, %78) ({
        ^bb0(%arg31: index):
          "memref.store"(%83, %96, %arg31, %arg30) : (f32, memref<24x3xf32>, index, index) -> ()
          "memref.store"(%84, %95, %arg31, %arg30) : (f32, memref<24x3xf32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "func.call"(%89, %88, %85, %87, %86, %96, %95) <{callee = @au_generated_candidate_5__l0p_l1u_site_7967}> : (memref<40xi32>, memref<40xi32>, memref<24x3xf32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> ()
      "memref.store"(%83, %85, %80, %80) : (f32, memref<24x3xf32>, index, index) -> ()
      "memref.store"(%83, %85, %80, %78) : (f32, memref<24x3xf32>, index, index) -> ()
      "memref.store"(%83, %85, %80, %74) : (f32, memref<24x3xf32>, index, index) -> ()
      "func.call"(%96, %95, %85) <{callee = @au_generated_candidate_31__l1p_site_7988}> : (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %97 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%81, %97) : (i32, memref<i32>) -> ()
    "memref.store"(%81, %97) : (i32, memref<i32>) -> ()
    "func.call"(%85, %97) <{callee = @au_generated_candidate_21__direct_site_8018}> : (memref<24x3xf32>, memref<i32>) -> ()
    %98 = "memref.load"(%97) : (memref<i32>) -> i32
    %99 = "arith.sitofp"(%98) : (i32) -> f32
    %100 = "arith.cmpi"(%98, %81) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %101 = "scf.if"(%100) ({
      %104 = "arith.negf"(%99) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "scf.yield"(%104) : (f32) -> ()
    }, {
      "scf.yield"(%99) : (f32) -> ()
    }) : (i1) -> f32
    %102 = "arith.cmpf"(%101, %83) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
    %103 = "arith.select"(%102, %98, %82) : (i1, i32, i32) -> i32
    "func.call"(%89, %88, %87, %86, %85) <{callee = @pcl_validate_outputs}> : (memref<40xi32>, memref<40xi32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>) -> ()
    "func.return"(%103) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<40x3xf32>, memref<40x3xf32>) -> (), sym_name = "au_generated_candidate_36__l0u_l1u_site_7891"}> ({
  ^bb0(%arg25: memref<40x3xf32>, %arg26: memref<40x3xf32>):
    %69 = "arith.constant"() <{value = 3 : index}> : () -> index
    %70 = "arith.constant"() <{value = 0 : index}> : () -> index
    %71 = "arith.constant"() <{value = 40 : index}> : () -> index
    %72 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%70, %71, %72) ({
    ^bb0(%arg27: index):
      "scf.for"(%70, %69, %72) ({
      ^bb0(%arg28: index):
        %73 = "memref.load"(%arg25, %arg27, %arg28) : (memref<40x3xf32>, index, index) -> f32
        "memref.store"(%73, %arg26, %arg27, %arg28) : (f32, memref<40x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<40x3xf32>, memref<40x3xf32>) -> (), sym_name = "au_generated_candidate_36__l0u_l1u_site_7904"}> ({
  ^bb0(%arg21: memref<40x3xf32>, %arg22: memref<40x3xf32>):
    %64 = "arith.constant"() <{value = 3 : index}> : () -> index
    %65 = "arith.constant"() <{value = 0 : index}> : () -> index
    %66 = "arith.constant"() <{value = 40 : index}> : () -> index
    %67 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%65, %66, %67) ({
    ^bb0(%arg23: index):
      "scf.for"(%65, %64, %67) ({
      ^bb0(%arg24: index):
        %68 = "memref.load"(%arg21, %arg23, %arg24) : (memref<40x3xf32>, index, index) -> f32
        "memref.store"(%68, %arg22, %arg23, %arg24) : (f32, memref<40x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_38__l0u_l1u_site_7917"}> ({
  ^bb0(%arg17: memref<24x3xf32>, %arg18: memref<24x3xf32>):
    %59 = "arith.constant"() <{value = 3 : index}> : () -> index
    %60 = "arith.constant"() <{value = 0 : index}> : () -> index
    %61 = "arith.constant"() <{value = 24 : index}> : () -> index
    %62 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%60, %61, %62) ({
    ^bb0(%arg19: index):
      "scf.for"(%60, %59, %62) ({
      ^bb0(%arg20: index):
        %63 = "memref.load"(%arg17, %arg19, %arg20) : (memref<24x3xf32>, index, index) -> f32
        "memref.store"(%63, %arg18, %arg19, %arg20) : (f32, memref<24x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<40xi32>, memref<40xi32>, memref<24x3xf32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_5__l0p_l1u_site_7967"}> ({
  ^bb0(%arg8: memref<40xi32>, %arg9: memref<40xi32>, %arg10: memref<24x3xf32>, %arg11: memref<40x3xf32>, %arg12: memref<40x3xf32>, %arg13: memref<24x3xf32>, %arg14: memref<24x3xf32>):
    %34 = "arith.constant"() <{value = 3 : index}> : () -> index
    %35 = "arith.constant"() <{value = 0 : index}> : () -> index
    %36 = "arith.constant"() <{value = 40 : index}> : () -> index
    %37 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%35, %36, %37) ({
    ^bb0(%arg15: index):
      %38 = "memref.load"(%arg8, %arg15) : (memref<40xi32>, index) -> i32
      %39 = "memref.load"(%arg9, %arg15) : (memref<40xi32>, index) -> i32
      "scf.for"(%35, %34, %37) ({
      ^bb0(%arg16: index):
        %40 = "arith.index_cast"(%39) : (i32) -> index
        %41 = "memref.load"(%arg10, %40, %arg16) : (memref<24x3xf32>, index, index) -> f32
        %42 = "arith.index_cast"(%38) : (i32) -> index
        %43 = "memref.load"(%arg10, %42, %arg16) : (memref<24x3xf32>, index, index) -> f32
        %44 = "arith.subf"(%41, %43) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %45 = "memref.load"(%arg11, %arg15, %arg16) : (memref<40x3xf32>, index, index) -> f32
        %46 = "arith.subf"(%44, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %47 = "memref.load"(%arg12, %arg15, %arg16) : (memref<40x3xf32>, index, index) -> f32
        %48 = "arith.mulf"(%47, %46) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %49 = "memref.load"(%arg13, %42, %arg16) : (memref<24x3xf32>, index, index) -> f32
        %50 = "arith.addf"(%49, %48) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%50, %arg13, %42, %arg16) : (f32, memref<24x3xf32>, index, index) -> ()
        %51 = "memref.load"(%arg13, %40, %arg16) : (memref<24x3xf32>, index, index) -> f32
        %52 = "arith.subf"(%51, %48) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%52, %arg13, %40, %arg16) : (f32, memref<24x3xf32>, index, index) -> ()
        %53 = "memref.load"(%arg12, %arg15, %arg16) : (memref<40x3xf32>, index, index) -> f32
        %54 = "memref.load"(%arg14, %42, %arg16) : (memref<24x3xf32>, index, index) -> f32
        %55 = "arith.addf"(%54, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%55, %arg14, %42, %arg16) : (f32, memref<24x3xf32>, index, index) -> ()
        %56 = "memref.load"(%arg12, %arg15, %arg16) : (memref<40x3xf32>, index, index) -> f32
        %57 = "memref.load"(%arg14, %40, %arg16) : (memref<24x3xf32>, index, index) -> f32
        %58 = "arith.addf"(%57, %56) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%58, %arg14, %40, %arg16) : (f32, memref<24x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_31__l1p_site_7988"}> ({
  ^bb0(%arg3: memref<24x3xf32>, %arg4: memref<24x3xf32>, %arg5: memref<24x3xf32>):
    %22 = "arith.constant"() <{value = 2.800000e-01 : f32}> : () -> f32
    %23 = "arith.constant"() <{value = 23 : index}> : () -> index
    %24 = "arith.constant"() <{value = 0 : index}> : () -> index
    %25 = "arith.constant"() <{value = 3 : index}> : () -> index
    %26 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%24, %25, %26) ({
    ^bb0(%arg6: index):
      "scf.for"(%24, %23, %26) ({
      ^bb0(%arg7: index):
        %27 = "arith.addi"(%arg7, %26) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %28 = "memref.load"(%arg3, %27, %arg6) : (memref<24x3xf32>, index, index) -> f32
        %29 = "arith.mulf"(%28, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %30 = "memref.load"(%arg4, %27, %arg6) : (memref<24x3xf32>, index, index) -> f32
        %31 = "arith.divf"(%29, %30) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %32 = "memref.load"(%arg5, %27, %arg6) : (memref<24x3xf32>, index, index) -> f32
        %33 = "arith.subf"(%32, %31) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%33, %arg5, %27, %arg6) : (f32, memref<24x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<24x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_21__direct_site_8018"}> ({
  ^bb0(%arg0: memref<24x3xf32>, %arg1: memref<i32>):
    %0 = "arith.constant"() <{value = 2.570000e+02 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 2 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1.730000e+02 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = 1.010000e+02 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 0 : index}> : () -> index
    %5 = "arith.constant"() <{value = 24 : index}> : () -> index
    %6 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%4, %5, %6) ({
    ^bb0(%arg2: index):
      %7 = "memref.load"(%arg0, %arg2, %4) : (memref<24x3xf32>, index, index) -> f32
      %8 = "arith.mulf"(%7, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %9 = "arith.fptosi"(%8) : (f32) -> i32
      %10 = "memref.load"(%arg1) : (memref<i32>) -> i32
      %11 = "arith.addi"(%10, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%11, %arg1) : (i32, memref<i32>) -> ()
      %12 = "memref.load"(%arg0, %arg2, %6) : (memref<24x3xf32>, index, index) -> f32
      %13 = "arith.mulf"(%12, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %14 = "arith.fptosi"(%13) : (f32) -> i32
      %15 = "arith.addi"(%11, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%15, %arg1) : (i32, memref<i32>) -> ()
      %16 = "memref.load"(%arg0, %arg2, %1) : (memref<24x3xf32>, index, index) -> f32
      %17 = "arith.mulf"(%16, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %18 = "arith.fptosi"(%17) : (f32) -> i32
      %19 = "arith.andi"(%15, %18) : (i32, i32) -> i32
      %20 = "arith.ori"(%15, %18) : (i32, i32) -> i32
      %21 = "arith.addi"(%19, %20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%21, %arg1) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<40xi32>, memref<40xi32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
