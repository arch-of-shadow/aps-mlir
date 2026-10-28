#map = affine_map<(d0) -> (d0 + 1, 0)>
#map1 = affine_map<(d0) -> (d0 + 1, 1)>
#map2 = affine_map<(d0) -> (d0 + 1, 2)>
#map3 = affine_map<() -> (0)>
#map4 = affine_map<() -> (15)>
#map5 = affine_map<(d0) -> (d0, 0)>
#map6 = affine_map<() -> ()>
#map7 = affine_map<(d0) -> (d0, 1)>
#map8 = affine_map<(d0) -> (d0, 2)>
#map9 = affine_map<() -> (16)>
#map10 = affine_map<() -> (20)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<16x3xf32>, memref<16x3xf32>, memref<16x3xf32>) -> (), sym_name = "au_generated_candidate_12__direct"}> ({
  ^bb0(%arg14: memref<16x3xf32>, %arg15: memref<16x3xf32>, %arg16: memref<16x3xf32>):
    "affine.for"() <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg17: index):
      %76 = "affine.load"(%arg14, %arg17) <{map = #map}> : (memref<16x3xf32>, index) -> f32
      %77 = "arith.constant"() <{value = 2.200000e-01 : f32}> : () -> f32
      %78 = "arith.mulf"(%76, %77) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %79 = "affine.load"(%arg15, %arg17) <{map = #map}> : (memref<16x3xf32>, index) -> f32
      %80 = "arith.divf"(%78, %79) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %81 = "affine.load"(%arg16, %arg17) <{map = #map}> : (memref<16x3xf32>, index) -> f32
      %82 = "arith.subf"(%81, %80) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%82, %arg16, %arg17) <{map = #map}> : (f32, memref<16x3xf32>, index) -> ()
      %83 = "affine.load"(%arg14, %arg17) <{map = #map1}> : (memref<16x3xf32>, index) -> f32
      %84 = "arith.mulf"(%83, %77) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %85 = "affine.load"(%arg15, %arg17) <{map = #map1}> : (memref<16x3xf32>, index) -> f32
      %86 = "arith.divf"(%84, %85) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %87 = "affine.load"(%arg16, %arg17) <{map = #map1}> : (memref<16x3xf32>, index) -> f32
      %88 = "arith.subf"(%87, %86) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%88, %arg16, %arg17) <{map = #map1}> : (f32, memref<16x3xf32>, index) -> ()
      %89 = "affine.load"(%arg14, %arg17) <{map = #map2}> : (memref<16x3xf32>, index) -> f32
      %90 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
      %91 = "arith.mulf"(%89, %90) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %92 = "affine.load"(%arg15, %arg17) <{map = #map2}> : (memref<16x3xf32>, index) -> f32
      %93 = "arith.divf"(%91, %92) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %94 = "affine.load"(%arg16, %arg17) <{map = #map2}> : (memref<16x3xf32>, index) -> f32
      %95 = "arith.subf"(%94, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%95, %arg16, %arg17) <{map = #map2}> : (f32, memref<16x3xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<16x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_21__direct"}> ({
  ^bb0(%arg11: memref<16x3xf32>, %arg12: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map9}> ({
    ^bb0(%arg13: index):
      %60 = "affine.load"(%arg11, %arg13) <{map = #map5}> : (memref<16x3xf32>, index) -> f32
      %61 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
      %62 = "arith.mulf"(%60, %61) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %63 = "arith.fptosi"(%62) : (f32) -> i32
      %64 = "affine.load"(%arg12) <{map = #map6}> : (memref<i32>) -> i32
      %65 = "arith.addi"(%64, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%65, %arg12) <{map = #map6}> : (i32, memref<i32>) -> ()
      %66 = "affine.load"(%arg11, %arg13) <{map = #map7}> : (memref<16x3xf32>, index) -> f32
      %67 = "arith.constant"() <{value = 1.310000e+02 : f32}> : () -> f32
      %68 = "arith.mulf"(%66, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %69 = "arith.fptosi"(%68) : (f32) -> i32
      %70 = "arith.addi"(%65, %69) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%70, %arg12) <{map = #map6}> : (i32, memref<i32>) -> ()
      %71 = "affine.load"(%arg11, %arg13) <{map = #map8}> : (memref<16x3xf32>, index) -> f32
      %72 = "arith.constant"() <{value = 1.970000e+02 : f32}> : () -> f32
      %73 = "arith.mulf"(%71, %72) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %74 = "arith.fptosi"(%73) : (f32) -> i32
      %75 = "arith.addi"(%70, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%75, %arg12) <{map = #map6}> : (i32, memref<i32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<16x3xf32>) -> (), sym_name = "au_generated_candidate_34__l0u"}> ({
  ^bb0(%arg9: memref<16x3xf32>):
    "affine.for"() <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map9}> ({
    ^bb0(%arg10: index):
      %38 = "arith.index_cast"(%arg10) : (index) -> i32
      %39 = "arith.sitofp"(%38) : (i32) -> f32
      %40 = "arith.constant"() <{value = 7.500000e-01 : f32}> : () -> f32
      %41 = "arith.mulf"(%39, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %42 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %43 = "arith.remsi"(%38, %42) : (i32, i32) -> i32
      %44 = "arith.sitofp"(%43) : (i32) -> f32
      %45 = "arith.constant"() <{value = 5.000000e-02 : f32}> : () -> f32
      %46 = "arith.mulf"(%44, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %47 = "arith.addf"(%41, %46) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%47, %arg9, %arg10) <{map = #map5}> : (f32, memref<16x3xf32>, index) -> ()
      %48 = "arith.constant"() <{value = 2 : i32}> : () -> i32
      %49 = "arith.remsi"(%38, %48) : (i32, i32) -> i32
      %50 = "arith.sitofp"(%49) : (i32) -> f32
      %51 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
      %52 = "arith.mulf"(%50, %51) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%52, %arg9, %arg10) <{map = #map7}> : (f32, memref<16x3xf32>, index) -> ()
      %53 = "arith.constant"() <{value = 5 : i32}> : () -> i32
      %54 = "arith.remsi"(%38, %53) : (i32, i32) -> i32
      %55 = "arith.constant"() <{value = -2 : i32}> : () -> i32
      %56 = "arith.addi"(%54, %55) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %57 = "arith.sitofp"(%56) : (i32) -> f32
      %58 = "arith.constant"() <{value = 1.500000e-02 : f32}> : () -> f32
      %59 = "arith.mulf"(%57, %58) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%59, %arg9, %arg10) <{map = #map8}> : (f32, memref<16x3xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<20x2xf32>, memref<20x2xf32>, memref<20x2xf32>) -> (), sym_name = "au_generated_candidate_41__direct"}> ({
  ^bb0(%arg5: memref<20x2xf32>, %arg6: memref<20x2xf32>, %arg7: memref<20x2xf32>):
    "affine.for"() <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map10}> ({
    ^bb0(%arg8: index):
      %25 = "affine.load"(%arg5, %arg8) <{map = #map5}> : (memref<20x2xf32>, index) -> f32
      %26 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
      %27 = "arith.mulf"(%25, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %28 = "affine.load"(%arg6, %arg8) <{map = #map5}> : (memref<20x2xf32>, index) -> f32
      %29 = "arith.divf"(%27, %28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %30 = "affine.load"(%arg7, %arg8) <{map = #map5}> : (memref<20x2xf32>, index) -> f32
      %31 = "arith.subf"(%30, %29) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%31, %arg7, %arg8) <{map = #map5}> : (f32, memref<20x2xf32>, index) -> ()
      %32 = "affine.load"(%arg5, %arg8) <{map = #map7}> : (memref<20x2xf32>, index) -> f32
      %33 = "arith.mulf"(%32, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %34 = "affine.load"(%arg6, %arg8) <{map = #map7}> : (memref<20x2xf32>, index) -> f32
      %35 = "arith.divf"(%33, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %36 = "affine.load"(%arg7, %arg8) <{map = #map7}> : (memref<20x2xf32>, index) -> f32
      %37 = "arith.subf"(%36, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%37, %arg7, %arg8) <{map = #map7}> : (f32, memref<20x2xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<20x2xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_51__direct"}> ({
  ^bb0(%arg2: memref<20x2xf32>, %arg3: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map10}> ({
    ^bb0(%arg4: index):
      %14 = "affine.load"(%arg2, %arg4) <{map = #map5}> : (memref<20x2xf32>, index) -> f32
      %15 = "arith.constant"() <{value = 1.510000e+02 : f32}> : () -> f32
      %16 = "arith.mulf"(%14, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %17 = "arith.fptosi"(%16) : (f32) -> i32
      %18 = "affine.load"(%arg3) <{map = #map6}> : (memref<i32>) -> i32
      %19 = "arith.addi"(%18, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%19, %arg3) <{map = #map6}> : (i32, memref<i32>) -> ()
      %20 = "affine.load"(%arg2, %arg4) <{map = #map7}> : (memref<20x2xf32>, index) -> f32
      %21 = "arith.constant"() <{value = 1.810000e+02 : f32}> : () -> f32
      %22 = "arith.mulf"(%20, %21) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %23 = "arith.fptosi"(%22) : (f32) -> i32
      %24 = "arith.addi"(%19, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%24, %arg3) <{map = #map6}> : (i32, memref<i32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<20x2xf32>) -> (), sym_name = "au_generated_candidate_56__l0u"}> ({
  ^bb0(%arg0: memref<20x2xf32>):
    "affine.for"() <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map10}> ({
    ^bb0(%arg1: index):
      %0 = "arith.index_cast"(%arg1) : (index) -> i32
      %1 = "arith.constant"() <{value = 5 : i32}> : () -> i32
      %2 = "arith.remsi"(%0, %1) : (i32, i32) -> i32
      %3 = "arith.sitofp"(%2) : (i32) -> f32
      %4 = "arith.constant"() <{value = 3.800000e-01 : f32}> : () -> f32
      %5 = "arith.mulf"(%3, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %6 = "arith.constant"() <{value = 1.600000e+00 : f32}> : () -> f32
      %7 = "arith.addf"(%5, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%7, %arg0, %arg1) <{map = #map5}> : (f32, memref<20x2xf32>, index) -> ()
      %8 = "arith.divsi"(%0, %1) : (i32, i32) -> i32
      %9 = "arith.sitofp"(%8) : (i32) -> f32
      %10 = "arith.constant"() <{value = 2.700000e-01 : f32}> : () -> f32
      %11 = "arith.mulf"(%9, %10) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %12 = "arith.constant"() <{value = 1.200000e+00 : f32}> : () -> f32
      %13 = "arith.addf"(%11, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%13, %arg0, %arg1) <{map = #map7}> : (f32, memref<20x2xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [2]} : () -> ()
}) : () -> ()
