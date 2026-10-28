#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<(d0) -> (d0, 1)>
#map3 = affine_map<(d0) -> (d0, 2)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<() -> (24)>
#map6 = affine_map<() -> (8)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<() -> (3)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_22__direct"}> ({
  ^bb0(%arg16: memref<24x3xf32>, %arg17: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg18: index):
      %78 = "affine.load"(%arg16, %arg18) <{map = #map}> : (memref<24x3xf32>, index) -> f32
      %79 = "arith.constant"() <{value = 8.300000e+01 : f32}> : () -> f32
      %80 = "arith.mulf"(%78, %79) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %81 = "arith.fptosi"(%80) : (f32) -> i32
      %82 = "affine.load"(%arg17) <{map = #map1}> : (memref<i32>) -> i32
      %83 = "arith.addi"(%82, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%83, %arg17) <{map = #map1}> : (i32, memref<i32>) -> ()
      %84 = "affine.load"(%arg16, %arg18) <{map = #map2}> : (memref<24x3xf32>, index) -> f32
      %85 = "arith.constant"() <{value = 1.090000e+02 : f32}> : () -> f32
      %86 = "arith.mulf"(%84, %85) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %87 = "arith.fptosi"(%86) : (f32) -> i32
      %88 = "arith.addi"(%83, %87) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%88, %arg17) <{map = #map1}> : (i32, memref<i32>) -> ()
      %89 = "affine.load"(%arg16, %arg18) <{map = #map3}> : (memref<24x3xf32>, index) -> f32
      %90 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
      %91 = "arith.mulf"(%89, %90) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %92 = "arith.fptosi"(%91) : (f32) -> i32
      %93 = "arith.addi"(%88, %92) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%93, %arg17) <{map = #map1}> : (i32, memref<i32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_38__l0u"}> ({
  ^bb0(%arg14: memref<24x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg15: index):
      %57 = "arith.index_cast"(%arg15) : (index) -> i32
      %58 = "arith.constant"() <{value = 6 : i32}> : () -> i32
      %59 = "arith.remsi"(%57, %58) : (i32, i32) -> i32
      %60 = "arith.sitofp"(%59) : (i32) -> f32
      %61 = "arith.constant"() <{value = 4.500000e-01 : f32}> : () -> f32
      %62 = "arith.mulf"(%60, %61) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %63 = "arith.constant"() <{value = -1.200000e+00 : f32}> : () -> f32
      %64 = "arith.addf"(%62, %63) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%64, %arg14, %arg15) <{map = #map}> : (f32, memref<24x3xf32>, index) -> ()
      %65 = "arith.divsi"(%57, %58) : (i32, i32) -> i32
      %66 = "arith.sitofp"(%65) : (i32) -> f32
      %67 = "arith.constant"() <{value = 6.000000e-01 : f32}> : () -> f32
      %68 = "arith.mulf"(%66, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %69 = "arith.constant"() <{value = -0.899999976 : f32}> : () -> f32
      %70 = "arith.addf"(%68, %69) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%70, %arg14, %arg15) <{map = #map2}> : (f32, memref<24x3xf32>, index) -> ()
      %71 = "arith.constant"() <{value = 5 : i32}> : () -> i32
      %72 = "arith.remsi"(%57, %71) : (i32, i32) -> i32
      %73 = "arith.sitofp"(%72) : (i32) -> f32
      %74 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
      %75 = "arith.mulf"(%73, %74) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %76 = "arith.constant"() <{value = 2.800000e+00 : f32}> : () -> f32
      %77 = "arith.addf"(%75, %76) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%77, %arg14, %arg15) <{map = #map3}> : (f32, memref<24x3xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<8x3xf32>) -> (), sym_name = "au_generated_candidate_44__l0u"}> ({
  ^bb0(%arg12: memref<8x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map6}> ({
    ^bb0(%arg13: index):
      %39 = "arith.index_cast"(%arg13) : (index) -> i32
      %40 = "arith.sitofp"(%39) : (i32) -> f32
      %41 = "arith.constant"() <{value = 5.500000e-01 : f32}> : () -> f32
      %42 = "arith.mulf"(%40, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %43 = "arith.constant"() <{value = -1.800000e+00 : f32}> : () -> f32
      %44 = "arith.addf"(%42, %43) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%44, %arg12, %arg13) <{map = #map}> : (f32, memref<8x3xf32>, index) -> ()
      %45 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %46 = "arith.remsi"(%39, %45) : (i32, i32) -> i32
      %47 = "arith.constant"() <{value = -1 : i32}> : () -> i32
      %48 = "arith.addi"(%46, %47) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %49 = "arith.sitofp"(%48) : (i32) -> f32
      %50 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
      %51 = "arith.mulf"(%49, %50) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%51, %arg12, %arg13) <{map = #map2}> : (f32, memref<8x3xf32>, index) -> ()
      %52 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %53 = "arith.remsi"(%39, %52) : (i32, i32) -> i32
      %54 = "arith.sitofp"(%53) : (i32) -> f32
      %55 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
      %56 = "arith.mulf"(%54, %55) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%56, %arg12, %arg13) <{map = #map3}> : (f32, memref<8x3xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_48__l0p_l1u"}> ({
  ^bb0(%arg7: memref<24x3xf32>, %arg8: memref<24x3xf32>, %arg9: memref<24x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg10: index):
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map8}> ({
      ^bb0(%arg11: index):
        %32 = "affine.load"(%arg7, %arg10, %arg11) <{map = #map7}> : (memref<24x3xf32>, index, index) -> f32
        %33 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
        %34 = "arith.mulf"(%32, %33) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %35 = "affine.load"(%arg8, %arg10, %arg11) <{map = #map7}> : (memref<24x3xf32>, index, index) -> f32
        %36 = "arith.divf"(%34, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %37 = "affine.load"(%arg9, %arg10, %arg11) <{map = #map7}> : (memref<24x3xf32>, index, index) -> f32
        %38 = "arith.subf"(%37, %36) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%38, %arg9, %arg10, %arg11) <{map = #map7}> : (f32, memref<24x3xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {pipeline = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x3xf32>, memref<8x3xf32>, memref<8x3xf32>) -> (), sym_name = "au_generated_candidate_49__l1p"}> ({
  ^bb0(%arg2: memref<8x3xf32>, %arg3: memref<8x3xf32>, %arg4: memref<8x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map6}> ({
    ^bb0(%arg5: index):
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map8}> ({
      ^bb0(%arg6: index):
        %25 = "affine.load"(%arg2, %arg5, %arg6) <{map = #map7}> : (memref<8x3xf32>, index, index) -> f32
        %26 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
        %27 = "arith.mulf"(%25, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %28 = "affine.load"(%arg3, %arg5, %arg6) <{map = #map7}> : (memref<8x3xf32>, index, index) -> f32
        %29 = "arith.divf"(%27, %28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %30 = "affine.load"(%arg4, %arg5, %arg6) <{map = #map7}> : (memref<8x3xf32>, index, index) -> f32
        %31 = "arith.subf"(%30, %29) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%31, %arg4, %arg5, %arg6) <{map = #map7}> : (f32, memref<8x3xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {pipeline = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<8x3xf32>) -> (), sym_name = "au_generated_candidate_9__l0u"}> ({
  ^bb0(%arg0: memref<8x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map6}> ({
    ^bb0(%arg1: index):
      %0 = "arith.index_cast"(%arg1) : (index) -> i32
      %1 = "arith.constant"() <{value = 4 : i32}> : () -> i32
      %2 = "arith.remsi"(%0, %1) : (i32, i32) -> i32
      %3 = "arith.constant"() <{value = -1 : i32}> : () -> i32
      %4 = "arith.addi"(%2, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %5 = "arith.sitofp"(%4) : (i32) -> f32
      %6 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
      %7 = "arith.mulf"(%5, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %8 = "affine.load"(%arg0, %arg1) <{map = #map}> : (memref<8x3xf32>, index) -> f32
      %9 = "arith.addf"(%8, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%9, %arg0, %arg1) <{map = #map}> : (f32, memref<8x3xf32>, index) -> ()
      %10 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %11 = "arith.remsi"(%0, %10) : (i32, i32) -> i32
      %12 = "arith.sitofp"(%11) : (i32) -> f32
      %13 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
      %14 = "arith.mulf"(%12, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %15 = "affine.load"(%arg0, %arg1) <{map = #map2}> : (memref<8x3xf32>, index) -> f32
      %16 = "arith.subf"(%15, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%16, %arg0, %arg1) <{map = #map2}> : (f32, memref<8x3xf32>, index) -> ()
      %17 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %18 = "arith.remsi"(%0, %17) : (i32, i32) -> i32
      %19 = "arith.addi"(%18, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %20 = "arith.sitofp"(%19) : (i32) -> f32
      %21 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
      %22 = "arith.mulf"(%20, %21) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %23 = "affine.load"(%arg0, %arg1) <{map = #map3}> : (memref<8x3xf32>, index) -> f32
      %24 = "arith.addf"(%23, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%24, %arg0, %arg1) <{map = #map3}> : (f32, memref<8x3xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
