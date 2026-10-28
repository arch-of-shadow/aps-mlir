"builtin.module"() ({
  "memref.global"() <{initial_value, sym_name = "bench_input_0", type = memref<18x6xf32>}> : () -> ()
  "memref.global"() <{initial_value, sym_name = "bench_input_1", type = memref<30xi32>}> : () -> ()
  "memref.global"() <{initial_value, sym_name = "bench_input_2", type = memref<30xi32>}> : () -> ()
  "memref.global"() <{initial_value, sym_name = "bench_input_3", type = memref<30x6xf32>}> : () -> ()
  "memref.global"() <{initial_value, sym_name = "bench_input_4", type = memref<30x6xf32>}> : () -> ()
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %73 = "arith.constant"() <{value = 30 : index}> : () -> index
    %74 = "arith.constant"() <{value = 1 : index}> : () -> index
    %75 = "arith.constant"() <{value = 0 : index}> : () -> index
    %76 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %77 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %78 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %79 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<30x6xf32>
    %80 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<30x6xf32>
    %81 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<30xi32>
    %82 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<30xi32>
    %83 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<18x6xf32>
    %84 = "memref.get_global"() <{name = @bench_input_0}> : () -> memref<18x6xf32>
    "func.call"(%84, %83) <{callee = @au_generated_candidate_43__l0u_l1u_site_2709}> : (memref<18x6xf32>, memref<18x6xf32>) -> ()
    %85 = "memref.get_global"() <{name = @bench_input_1}> : () -> memref<30xi32>
    "scf.for"(%75, %73, %74) ({
    ^bb0(%arg32: index):
      %100 = "memref.load"(%85, %arg32) : (memref<30xi32>, index) -> i32
      "memref.store"(%100, %82, %arg32) : (i32, memref<30xi32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %86 = "memref.get_global"() <{name = @bench_input_2}> : () -> memref<30xi32>
    "scf.for"(%75, %73, %74) ({
    ^bb0(%arg31: index):
      %99 = "memref.load"(%86, %arg31) : (memref<30xi32>, index) -> i32
      "memref.store"(%99, %81, %arg31) : (i32, memref<30xi32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %87 = "memref.get_global"() <{name = @bench_input_3}> : () -> memref<30x6xf32>
    "func.call"(%87, %80) <{callee = @au_generated_candidate_40__l0u_l1u_site_2738}> : (memref<30x6xf32>, memref<30x6xf32>) -> ()
    %88 = "memref.get_global"() <{name = @bench_input_4}> : () -> memref<30x6xf32>
    "func.call"(%88, %79) <{callee = @au_generated_candidate_40__l0u_l1u_site_2751}> : (memref<30x6xf32>, memref<30x6xf32>) -> ()
    %89 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<18x6xf32>
    %90 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<18x6xf32>
    "func.call"(%90, %89, %82, %81, %83, %80, %79) <{callee = @au_generated_candidate_5__l4p_site_2832}> : (memref<18x6xf32>, memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<18x6xf32>, memref<30x6xf32>, memref<30x6xf32>) -> ()
    %91 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%76, %91) : (i32, memref<i32>) -> ()
    "memref.store"(%76, %91) : (i32, memref<i32>) -> ()
    "func.call"(%83, %91) <{callee = @au_generated_candidate_32__direct_site_2855}> : (memref<18x6xf32>, memref<i32>) -> ()
    %92 = "memref.load"(%91) : (memref<i32>) -> i32
    %93 = "arith.sitofp"(%92) : (i32) -> f32
    %94 = "arith.cmpi"(%92, %76) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %95 = "scf.if"(%94) ({
      %98 = "arith.negf"(%93) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "scf.yield"(%98) : (f32) -> ()
    }, {
      "scf.yield"(%93) : (f32) -> ()
    }) : (i1) -> f32
    %96 = "arith.cmpf"(%95, %78) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
    %97 = "arith.select"(%96, %92, %77) : (i1, i32, i32) -> i32
    "func.call"(%83, %82, %81, %80, %79) <{callee = @pcl_validate_outputs}> : (memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<30x6xf32>, memref<30x6xf32>) -> ()
    "func.return"(%97) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<18x6xf32>, memref<18x6xf32>) -> (), sym_name = "au_generated_candidate_43__l0u_l1u_site_2709"}> ({
  ^bb0(%arg27: memref<18x6xf32>, %arg28: memref<18x6xf32>):
    %68 = "arith.constant"() <{value = 6 : index}> : () -> index
    %69 = "arith.constant"() <{value = 0 : index}> : () -> index
    %70 = "arith.constant"() <{value = 18 : index}> : () -> index
    %71 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%69, %70, %71) ({
    ^bb0(%arg29: index):
      "scf.for"(%69, %68, %71) ({
      ^bb0(%arg30: index):
        %72 = "memref.load"(%arg27, %arg29, %arg30) : (memref<18x6xf32>, index, index) -> f32
        "memref.store"(%72, %arg28, %arg29, %arg30) : (f32, memref<18x6xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<30x6xf32>, memref<30x6xf32>) -> (), sym_name = "au_generated_candidate_40__l0u_l1u_site_2738"}> ({
  ^bb0(%arg23: memref<30x6xf32>, %arg24: memref<30x6xf32>):
    %63 = "arith.constant"() <{value = 6 : index}> : () -> index
    %64 = "arith.constant"() <{value = 0 : index}> : () -> index
    %65 = "arith.constant"() <{value = 30 : index}> : () -> index
    %66 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%64, %65, %66) ({
    ^bb0(%arg25: index):
      "scf.for"(%64, %63, %66) ({
      ^bb0(%arg26: index):
        %67 = "memref.load"(%arg23, %arg25, %arg26) : (memref<30x6xf32>, index, index) -> f32
        "memref.store"(%67, %arg24, %arg25, %arg26) : (f32, memref<30x6xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<30x6xf32>, memref<30x6xf32>) -> (), sym_name = "au_generated_candidate_40__l0u_l1u_site_2751"}> ({
  ^bb0(%arg19: memref<30x6xf32>, %arg20: memref<30x6xf32>):
    %58 = "arith.constant"() <{value = 6 : index}> : () -> index
    %59 = "arith.constant"() <{value = 0 : index}> : () -> index
    %60 = "arith.constant"() <{value = 30 : index}> : () -> index
    %61 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%59, %60, %61) ({
    ^bb0(%arg21: index):
      "scf.for"(%59, %58, %61) ({
      ^bb0(%arg22: index):
        %62 = "memref.load"(%arg19, %arg21, %arg22) : (memref<30x6xf32>, index, index) -> f32
        "memref.store"(%62, %arg20, %arg21, %arg22) : (f32, memref<30x6xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<18x6xf32>, memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<18x6xf32>, memref<30x6xf32>, memref<30x6xf32>) -> (), sym_name = "au_generated_candidate_5__l4p_site_2832"}> ({
  ^bb0(%arg4: memref<18x6xf32>, %arg5: memref<18x6xf32>, %arg6: memref<30xi32>, %arg7: memref<30xi32>, %arg8: memref<18x6xf32>, %arg9: memref<30x6xf32>, %arg10: memref<30x6xf32>):
    %15 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %16 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    %17 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %18 = "arith.constant"() <{value = 17 : index}> : () -> index
    %19 = "arith.constant"() <{value = 30 : index}> : () -> index
    %20 = "arith.constant"() <{value = 1.000000e-03 : f32}> : () -> f32
    %21 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %22 = "arith.constant"() <{value = 18 : index}> : () -> index
    %23 = "arith.constant"() <{value = 6 : index}> : () -> index
    %24 = "arith.constant"() <{value = 0 : index}> : () -> index
    %25 = "arith.constant"() <{value = 7 : index}> : () -> index
    %26 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%24, %25, %26) ({
    ^bb0(%arg11: index):
      "scf.for"(%24, %23, %26) ({
      ^bb0(%arg17: index):
        "scf.for"(%24, %22, %26) ({
        ^bb0(%arg18: index):
          "memref.store"(%21, %arg4, %arg18, %arg17) : (f32, memref<18x6xf32>, index, index) -> ()
          "memref.store"(%20, %arg5, %arg18, %arg17) : (f32, memref<18x6xf32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%24, %19, %26) ({
      ^bb0(%arg15: index):
        %37 = "memref.load"(%arg6, %arg15) : (memref<30xi32>, index) -> i32
        %38 = "memref.load"(%arg7, %arg15) : (memref<30xi32>, index) -> i32
        "scf.for"(%24, %23, %26) ({
        ^bb0(%arg16: index):
          %39 = "arith.index_cast"(%38) : (i32) -> index
          %40 = "memref.load"(%arg8, %39, %arg16) : (memref<18x6xf32>, index, index) -> f32
          %41 = "arith.index_cast"(%37) : (i32) -> index
          %42 = "memref.load"(%arg8, %41, %arg16) : (memref<18x6xf32>, index, index) -> f32
          %43 = "arith.subf"(%40, %42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %44 = "memref.load"(%arg9, %arg15, %arg16) : (memref<30x6xf32>, index, index) -> f32
          %45 = "arith.subf"(%43, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %46 = "memref.load"(%arg10, %arg15, %arg16) : (memref<30x6xf32>, index, index) -> f32
          %47 = "arith.mulf"(%46, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %48 = "memref.load"(%arg4, %41, %arg16) : (memref<18x6xf32>, index, index) -> f32
          %49 = "arith.addf"(%48, %47) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%49, %arg4, %41, %arg16) : (f32, memref<18x6xf32>, index, index) -> ()
          %50 = "memref.load"(%arg4, %39, %arg16) : (memref<18x6xf32>, index, index) -> f32
          %51 = "arith.subf"(%50, %47) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%51, %arg4, %39, %arg16) : (f32, memref<18x6xf32>, index, index) -> ()
          %52 = "memref.load"(%arg10, %arg15, %arg16) : (memref<30x6xf32>, index, index) -> f32
          %53 = "memref.load"(%arg5, %41, %arg16) : (memref<18x6xf32>, index, index) -> f32
          %54 = "arith.addf"(%53, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%54, %arg5, %41, %arg16) : (f32, memref<18x6xf32>, index, index) -> ()
          %55 = "memref.load"(%arg10, %arg15, %arg16) : (memref<30x6xf32>, index, index) -> f32
          %56 = "memref.load"(%arg5, %39, %arg16) : (memref<18x6xf32>, index, index) -> f32
          %57 = "arith.addf"(%56, %55) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%57, %arg5, %39, %arg16) : (f32, memref<18x6xf32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%24, %23, %26) ({
      ^bb0(%arg14: index):
        "memref.store"(%21, %arg8, %24, %arg14) : (f32, memref<18x6xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%24, %18, %26) ({
      ^bb0(%arg12: index):
        "scf.for"(%24, %23, %26) ({
        ^bb0(%arg13: index):
          %27 = "arith.index_cast"(%arg13) : (index) -> i32
          %28 = "arith.cmpi"(%27, %17) <{predicate = 2 : i64}> : (i32, i32) -> i1
          %29 = "arith.select"(%28, %16, %15) : (i1, f32, f32) -> f32
          %30 = "arith.addi"(%arg12, %26) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %31 = "memref.load"(%arg4, %30, %arg13) : (memref<18x6xf32>, index, index) -> f32
          %32 = "arith.mulf"(%29, %31) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %33 = "memref.load"(%arg5, %30, %arg13) : (memref<18x6xf32>, index, index) -> f32
          %34 = "arith.divf"(%32, %33) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %35 = "memref.load"(%arg8, %30, %arg13) : (memref<18x6xf32>, index, index) -> f32
          %36 = "arith.subf"(%35, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%36, %arg8, %30, %arg13) : (f32, memref<18x6xf32>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<18x6xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_32__direct_site_2855"}> ({
  ^bb0(%arg0: memref<18x6xf32>, %arg1: memref<i32>):
    %0 = "arith.constant"() <{value = 8.300000e+01 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 1.900000e+01 : f32}> : () -> f32
    %2 = "arith.constant"() <{value = 6 : index}> : () -> index
    %3 = "arith.constant"() <{value = 0 : index}> : () -> index
    %4 = "arith.constant"() <{value = 18 : index}> : () -> index
    %5 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%3, %4, %5) ({
    ^bb0(%arg2: index):
      "scf.for"(%3, %2, %5) ({
      ^bb0(%arg3: index):
        %6 = "arith.index_cast"(%arg3) : (index) -> i32
        %7 = "memref.load"(%arg0, %arg2, %arg3) : (memref<18x6xf32>, index, index) -> f32
        %8 = "arith.sitofp"(%6) : (i32) -> f32
        %9 = "arith.mulf"(%8, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %10 = "arith.addf"(%9, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %11 = "arith.mulf"(%7, %10) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %12 = "arith.fptosi"(%11) : (f32) -> i32
        %13 = "memref.load"(%arg1) : (memref<i32>) -> i32
        %14 = "arith.addi"(%12, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%14, %arg1) : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<30x6xf32>, memref<30x6xf32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
