#map = affine_map<() -> ()>
#map1 = affine_map<(d0) -> (d0, 0)>
#map2 = affine_map<(d0) -> (d0, 1)>
#map3 = affine_map<(d0) -> (d0, 2)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<f32>, memref<f32>, memref<f32>, memref<32x3xf32>, memref<32x3xf32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) -> (), sym_name = "au_generated_candidate_1__direct"}> ({
  ^bb0(%arg3: memref<f32>, %arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<32x3xf32>, %arg7: memref<32x3xf32>, %arg8: memref<f32>, %arg9: memref<f32>, %arg10: memref<f32>, %arg11: memref<f32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg12: index):
      %28 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      %29 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
      "affine.store"(%29, %28) <{map = #map}> : (f32, memref<f32>) -> ()
      %30 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      %31 = "arith.constant"() <{value = 0 : i32}> : () -> i32
      "affine.store"(%31, %30) <{map = #map}> : (i32, memref<i32>) -> ()
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
      ^bb0(%arg13: index):
        %61 = "arith.index_cast"(%arg13) : (index) -> i32
        %62 = "affine.load"(%arg3) <{map = #map}> : (memref<f32>) -> f32
        %63 = "affine.load"(%arg4) <{map = #map}> : (memref<f32>) -> f32
        %64 = "affine.load"(%arg5) <{map = #map}> : (memref<f32>) -> f32
        %65 = "affine.load"(%arg6, %arg12) <{map = #map1}> : (memref<32x3xf32>, index) -> f32
        %66 = "arith.addf"(%65, %62) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %67 = "affine.load"(%arg7, %arg13) <{map = #map1}> : (memref<32x3xf32>, index) -> f32
        %68 = "arith.subf"(%66, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %69 = "affine.load"(%arg6, %arg12) <{map = #map2}> : (memref<32x3xf32>, index) -> f32
        %70 = "arith.addf"(%69, %63) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %71 = "affine.load"(%arg7, %arg13) <{map = #map2}> : (memref<32x3xf32>, index) -> f32
        %72 = "arith.subf"(%70, %71) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %73 = "affine.load"(%arg6, %arg12) <{map = #map3}> : (memref<32x3xf32>, index) -> f32
        %74 = "arith.addf"(%73, %64) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %75 = "affine.load"(%arg7, %arg13) <{map = #map3}> : (memref<32x3xf32>, index) -> f32
        %76 = "arith.subf"(%74, %75) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %77 = "arith.mulf"(%68, %68) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %78 = "arith.mulf"(%72, %72) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %79 = "arith.addf"(%77, %78) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %80 = "arith.mulf"(%76, %76) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %81 = "arith.addf"(%79, %80) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %82 = "affine.load"(%28) <{map = #map}> : (memref<f32>) -> f32
        %83 = "arith.cmpf"(%81, %82) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        "scf.if"(%83) ({
          "affine.store"(%81, %28) <{map = #map}> : (f32, memref<f32>) -> ()
          "affine.store"(%61, %30) <{map = #map}> : (i32, memref<i32>) -> ()
          "scf.yield"() : () -> ()
        }, {
        }) : (i1) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %32 = "affine.load"(%30) <{map = #map}> : (memref<i32>) -> i32
      %33 = "arith.index_cast"(%32) : (i32) -> index
      %34 = "arith.constant"() <{value = 0 : index}> : () -> index
      %35 = "memref.load"(%arg7, %33, %34) : (memref<32x3xf32>, index, index) -> f32
      %36 = "affine.load"(%arg6, %arg12) <{map = #map1}> : (memref<32x3xf32>, index) -> f32
      %37 = "affine.load"(%arg3) <{map = #map}> : (memref<f32>) -> f32
      %38 = "arith.addf"(%36, %37) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %39 = "arith.subf"(%35, %38) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %40 = "affine.load"(%arg8) <{map = #map}> : (memref<f32>) -> f32
      %41 = "arith.addf"(%40, %39) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%41, %arg8) <{map = #map}> : (f32, memref<f32>) -> ()
      %42 = "arith.constant"() <{value = 1 : index}> : () -> index
      %43 = "memref.load"(%arg7, %33, %42) : (memref<32x3xf32>, index, index) -> f32
      %44 = "affine.load"(%arg6, %arg12) <{map = #map2}> : (memref<32x3xf32>, index) -> f32
      %45 = "affine.load"(%arg4) <{map = #map}> : (memref<f32>) -> f32
      %46 = "arith.addf"(%44, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %47 = "arith.subf"(%43, %46) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %48 = "affine.load"(%arg9) <{map = #map}> : (memref<f32>) -> f32
      %49 = "arith.addf"(%48, %47) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%49, %arg9) <{map = #map}> : (f32, memref<f32>) -> ()
      %50 = "arith.constant"() <{value = 2 : index}> : () -> index
      %51 = "memref.load"(%arg7, %33, %50) : (memref<32x3xf32>, index, index) -> f32
      %52 = "affine.load"(%arg6, %arg12) <{map = #map3}> : (memref<32x3xf32>, index) -> f32
      %53 = "affine.load"(%arg5) <{map = #map}> : (memref<f32>) -> f32
      %54 = "arith.addf"(%52, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %55 = "arith.subf"(%51, %54) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %56 = "affine.load"(%arg10) <{map = #map}> : (memref<f32>) -> f32
      %57 = "arith.addf"(%56, %55) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%57, %arg10) <{map = #map}> : (f32, memref<f32>) -> ()
      %58 = "affine.load"(%28) <{map = #map}> : (memref<f32>) -> f32
      %59 = "affine.load"(%arg11) <{map = #map}> : (memref<f32>) -> f32
      %60 = "arith.addf"(%59, %58) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%60, %arg11) <{map = #map}> : (f32, memref<f32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<32x3xf32>, memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_8__l0u"}> ({
  ^bb0(%arg0: memref<32x3xf32>, %arg1: memref<32x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg2: index):
      %0 = "arith.index_cast"(%arg2) : (index) -> i32
      %1 = "arith.constant"() <{value = 7 : i32}> : () -> i32
      %2 = "arith.muli"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %3 = "arith.constant"() <{value = 41 : i32}> : () -> i32
      %4 = "arith.remsi"(%2, %3) : (i32, i32) -> i32
      %5 = "arith.sitofp"(%4) : (i32) -> f32
      %6 = "arith.constant"() <{value = 3.000000e-02 : f32}> : () -> f32
      %7 = "arith.mulf"(%5, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %8 = "arith.constant"() <{value = 11 : i32}> : () -> i32
      %9 = "arith.muli"(%0, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %10 = "arith.constant"() <{value = 37 : i32}> : () -> i32
      %11 = "arith.remsi"(%9, %10) : (i32, i32) -> i32
      %12 = "arith.sitofp"(%11) : (i32) -> f32
      %13 = "arith.constant"() <{value = 2.500000e-02 : f32}> : () -> f32
      %14 = "arith.mulf"(%12, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %15 = "arith.constant"() <{value = 13 : i32}> : () -> i32
      %16 = "arith.muli"(%0, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %17 = "arith.constant"() <{value = 31 : i32}> : () -> i32
      %18 = "arith.remsi"(%16, %17) : (i32, i32) -> i32
      %19 = "arith.sitofp"(%18) : (i32) -> f32
      %20 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
      %21 = "arith.mulf"(%19, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%7, %arg0, %arg2) <{map = #map1}> : (f32, memref<32x3xf32>, index) -> ()
      "affine.store"(%14, %arg0, %arg2) <{map = #map2}> : (f32, memref<32x3xf32>, index) -> ()
      "affine.store"(%21, %arg0, %arg2) <{map = #map3}> : (f32, memref<32x3xf32>, index) -> ()
      %22 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
      %23 = "arith.addf"(%7, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%23, %arg1, %arg2) <{map = #map1}> : (f32, memref<32x3xf32>, index) -> ()
      %24 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
      %25 = "arith.subf"(%14, %24) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%25, %arg1, %arg2) <{map = #map2}> : (f32, memref<32x3xf32>, index) -> ()
      %26 = "arith.constant"() <{value = 6.000000e-02 : f32}> : () -> f32
      %27 = "arith.addf"(%21, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%27, %arg1, %arg2) <{map = #map3}> : (f32, memref<32x3xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
