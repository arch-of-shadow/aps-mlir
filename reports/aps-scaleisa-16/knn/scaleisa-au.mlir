#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (8)>
#map4 = affine_map<() -> (4)>
#map5 = affine_map<(d0) -> (d0, 0)>
#map6 = affine_map<(d0) -> (d0, 1)>
#map7 = affine_map<(d0) -> (d0, 2)>
#map8 = affine_map<(d0) -> (d0)>
#map9 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<4x8xi32>, memref<4x8xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_11__direct"}> ({
  ^bb0(%arg16: memref<4x8xi32>, %arg17: memref<4x8xf32>, %arg18: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg19: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg20: index):
        %89 = "affine.load"(%arg16, %arg19, %arg20) <{map = #map}> : (memref<4x8xi32>, index, index) -> i32
        %90 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %91 = "arith.cmpi"(%89, %90) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %92 = "scf.if"(%91) ({
          %95 = "affine.load"(%arg16, %arg19, %arg20) <{map = #map}> : (memref<4x8xi32>, index, index) -> i32
          %96 = "affine.load"(%arg17, %arg19, %arg20) <{map = #map}> : (memref<4x8xf32>, index, index) -> f32
          %97 = "arith.constant"() <{value = 2.560000e+02 : f32}> : () -> f32
          %98 = "arith.mulf"(%96, %97) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %99 = "arith.fptosi"(%98) : (f32) -> i32
          %100 = "arith.addi"(%95, %99) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "scf.yield"(%100) : (i32) -> ()
        }, {
          "scf.yield"(%90) : (i32) -> ()
        }) : (i1) -> i32
        %93 = "affine.load"(%arg18) <{map = #map1}> : (memref<i32>) -> i32
        %94 = "arith.addi"(%93, %92) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%94, %arg18) <{map = #map1}> : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<4x8xf32>, memref<4x8xi32>, memref<32x3xf32>, memref<4x3xf32>, memref<8xf32>, memref<8xi32>) -> (), sym_name = "au_generated_candidate_2__direct"}> ({
  ^bb0(%arg4: memref<4x8xf32>, %arg5: memref<4x8xi32>, %arg6: memref<32x3xf32>, %arg7: memref<4x3xf32>, %arg8: memref<8xf32>, %arg9: memref<8xi32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg10: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 8 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg14: index):
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg15: index):
          %87 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
          "affine.store"(%87, %arg4, %arg10, %arg15) <{map = #map}> : (f32, memref<4x8xf32>, index, index) -> ()
          %88 = "arith.constant"() <{value = -1 : i32}> : () -> i32
          "affine.store"(%88, %arg5, %arg10, %arg15) <{map = #map}> : (i32, memref<4x8xi32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map9}> ({
      ^bb0(%arg11: index):
        %56 = "arith.index_cast"(%arg11) : (index) -> i32
        %57 = "affine.load"(%arg6, %arg11) <{map = #map5}> : (memref<32x3xf32>, index) -> f32
        %58 = "affine.load"(%arg7, %arg10) <{map = #map5}> : (memref<4x3xf32>, index) -> f32
        %59 = "arith.subf"(%57, %58) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %60 = "affine.load"(%arg6, %arg11) <{map = #map6}> : (memref<32x3xf32>, index) -> f32
        %61 = "affine.load"(%arg7, %arg10) <{map = #map6}> : (memref<4x3xf32>, index) -> f32
        %62 = "arith.subf"(%60, %61) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %63 = "affine.load"(%arg6, %arg11) <{map = #map7}> : (memref<32x3xf32>, index) -> f32
        %64 = "affine.load"(%arg7, %arg10) <{map = #map7}> : (memref<4x3xf32>, index) -> f32
        %65 = "arith.subf"(%63, %64) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %66 = "arith.mulf"(%59, %59) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %67 = "arith.mulf"(%62, %62) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %68 = "arith.addf"(%66, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %69 = "arith.mulf"(%65, %65) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %70 = "arith.addf"(%68, %69) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %71 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
        %72 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        "affine.store"(%72, %71) <{map = #map1}> : (i32, memref<i32>) -> ()
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg13: index):
          %75 = "arith.index_cast"(%arg13) : (index) -> i32
          %76 = "affine.load"(%71) <{map = #map1}> : (memref<i32>) -> i32
          %77 = "arith.cmpi"(%76, %72) <{predicate = 0 : i64}> : (i32, i32) -> i1
          %78 = "scf.if"(%77) ({
            %85 = "affine.load"(%arg4, %arg10, %arg13) <{map = #map}> : (memref<4x8xf32>, index, index) -> f32
            %86 = "arith.cmpf"(%70, %85) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
            "scf.yield"(%86) : (i1) -> ()
          }, {
            %84 = "arith.constant"() <{value = false}> : () -> i1
            "scf.yield"(%84) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%78) ({
            "affine.store"(%70, %arg8, %arg13) <{map = #map8}> : (f32, memref<8xf32>, index) -> ()
            "affine.store"(%56, %arg9, %arg13) <{map = #map8}> : (i32, memref<8xi32>, index) -> ()
            %83 = "arith.constant"() <{value = 1 : i32}> : () -> i32
            "affine.store"(%83, %71) <{map = #map1}> : (i32, memref<i32>) -> ()
            "scf.yield"() : () -> ()
          }, {
            %79 = "arith.subi"(%75, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %80 = "arith.index_cast"(%79) : (i32) -> index
            %81 = "memref.load"(%arg4, %arg10, %80) : (memref<4x8xf32>, index, index) -> f32
            "affine.store"(%81, %arg8, %arg13) <{map = #map8}> : (f32, memref<8xf32>, index) -> ()
            %82 = "memref.load"(%arg5, %arg10, %80) : (memref<4x8xi32>, index, index) -> i32
            "affine.store"(%82, %arg9, %arg13) <{map = #map8}> : (i32, memref<8xi32>, index) -> ()
            "scf.yield"() : () -> ()
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg12: index):
          %73 = "affine.load"(%arg8, %arg12) <{map = #map8}> : (memref<8xf32>, index) -> f32
          "affine.store"(%73, %arg4, %arg10, %arg12) <{map = #map}> : (f32, memref<4x8xf32>, index, index) -> ()
          %74 = "affine.load"(%arg9, %arg12) <{map = #map8}> : (memref<8xi32>, index) -> i32
          "affine.store"(%74, %arg5, %arg10, %arg12) <{map = #map}> : (i32, memref<4x8xi32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_24__l0u"}> ({
  ^bb0(%arg2: memref<32x3xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map9}> ({
    ^bb0(%arg3: index):
      %28 = "arith.index_cast"(%arg3) : (index) -> i32
      %29 = "arith.constant"() <{value = 11 : i32}> : () -> i32
      %30 = "arith.muli"(%28, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %31 = "arith.constant"() <{value = 3 : i32}> : () -> i32
      %32 = "arith.addi"(%30, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %33 = "arith.constant"() <{value = 101 : i32}> : () -> i32
      %34 = "arith.remsi"(%32, %33) : (i32, i32) -> i32
      %35 = "arith.sitofp"(%34) : (i32) -> f32
      %36 = "arith.constant"() <{value = 3.100000e-02 : f32}> : () -> f32
      %37 = "arith.mulf"(%35, %36) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%37, %arg2, %arg3) <{map = #map5}> : (f32, memref<32x3xf32>, index) -> ()
      %38 = "arith.constant"() <{value = 17 : i32}> : () -> i32
      %39 = "arith.muli"(%28, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %40 = "arith.constant"() <{value = 5 : i32}> : () -> i32
      %41 = "arith.addi"(%39, %40) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %42 = "arith.constant"() <{value = 97 : i32}> : () -> i32
      %43 = "arith.remsi"(%41, %42) : (i32, i32) -> i32
      %44 = "arith.sitofp"(%43) : (i32) -> f32
      %45 = "arith.constant"() <{value = 2.900000e-02 : f32}> : () -> f32
      %46 = "arith.mulf"(%44, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%46, %arg2, %arg3) <{map = #map6}> : (f32, memref<32x3xf32>, index) -> ()
      %47 = "arith.constant"() <{value = 19 : i32}> : () -> i32
      %48 = "arith.muli"(%28, %47) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %49 = "arith.constant"() <{value = 7 : i32}> : () -> i32
      %50 = "arith.addi"(%48, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %51 = "arith.constant"() <{value = 89 : i32}> : () -> i32
      %52 = "arith.remsi"(%50, %51) : (i32, i32) -> i32
      %53 = "arith.sitofp"(%52) : (i32) -> f32
      %54 = "arith.constant"() <{value = 2.700000e-02 : f32}> : () -> f32
      %55 = "arith.mulf"(%53, %54) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%55, %arg2, %arg3) <{map = #map7}> : (f32, memref<32x3xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<4x3xf32>) -> (), sym_name = "au_generated_candidate_25__l0u"}> ({
  ^bb0(%arg0: memref<4x3xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg1: index):
      %0 = "arith.index_cast"(%arg1) : (index) -> i32
      %1 = "arith.constant"() <{value = 23 : i32}> : () -> i32
      %2 = "arith.muli"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %3 = "arith.constant"() <{value = 9 : i32}> : () -> i32
      %4 = "arith.addi"(%2, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %5 = "arith.constant"() <{value = 101 : i32}> : () -> i32
      %6 = "arith.remsi"(%4, %5) : (i32, i32) -> i32
      %7 = "arith.sitofp"(%6) : (i32) -> f32
      %8 = "arith.constant"() <{value = 3.100000e-02 : f32}> : () -> f32
      %9 = "arith.mulf"(%7, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%9, %arg0, %arg1) <{map = #map5}> : (f32, memref<4x3xf32>, index) -> ()
      %10 = "arith.constant"() <{value = 29 : i32}> : () -> i32
      %11 = "arith.muli"(%0, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %12 = "arith.constant"() <{value = 13 : i32}> : () -> i32
      %13 = "arith.addi"(%11, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %14 = "arith.constant"() <{value = 97 : i32}> : () -> i32
      %15 = "arith.remsi"(%13, %14) : (i32, i32) -> i32
      %16 = "arith.sitofp"(%15) : (i32) -> f32
      %17 = "arith.constant"() <{value = 2.900000e-02 : f32}> : () -> f32
      %18 = "arith.mulf"(%16, %17) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%18, %arg0, %arg1) <{map = #map6}> : (f32, memref<4x3xf32>, index) -> ()
      %19 = "arith.constant"() <{value = 31 : i32}> : () -> i32
      %20 = "arith.muli"(%0, %19) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %21 = "arith.constant"() <{value = 17 : i32}> : () -> i32
      %22 = "arith.addi"(%20, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %23 = "arith.constant"() <{value = 89 : i32}> : () -> i32
      %24 = "arith.remsi"(%22, %23) : (i32, i32) -> i32
      %25 = "arith.sitofp"(%24) : (i32) -> f32
      %26 = "arith.constant"() <{value = 2.700000e-02 : f32}> : () -> f32
      %27 = "arith.mulf"(%25, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%27, %arg0, %arg1) <{map = #map7}> : (f32, memref<4x3xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
