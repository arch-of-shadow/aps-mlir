#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (4)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<216xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_11__direct"}> ({
  ^bb0(%arg11: memref<216xf32>, %arg12: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg13: index):
      %70 = "affine.apply"(%arg13) <{map = #map}> : (index) -> index
      %71 = "arith.index_cast"(%70) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg14: index):
        %72 = "affine.apply"(%arg14) <{map = #map}> : (index) -> index
        %73 = "arith.index_cast"(%72) : (index) -> i32
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg15: index):
          %74 = "affine.apply"(%arg15) <{map = #map}> : (index) -> index
          %75 = "arith.index_cast"(%74) : (index) -> i32
          %76 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %77 = "arith.muli"(%71, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %78 = "arith.addi"(%77, %73) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %79 = "arith.muli"(%78, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %80 = "arith.addi"(%79, %75) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %81 = "arith.constant"() <{value = 1 : i32}> : () -> i32
          %82 = "arith.addi"(%71, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %83 = "arith.muli"(%82, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %84 = "arith.addi"(%83, %73) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %85 = "arith.muli"(%84, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %86 = "arith.addi"(%85, %75) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %87 = "arith.addi"(%73, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %88 = "arith.addi"(%77, %87) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %89 = "arith.muli"(%88, %76) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %90 = "arith.addi"(%89, %75) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %91 = "arith.addi"(%75, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %92 = "arith.addi"(%79, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %93 = "arith.index_cast"(%80) : (i32) -> index
          %94 = "memref.load"(%arg11, %93) : (memref<216xf32>, index) -> f32
          %95 = "arith.index_cast"(%86) : (i32) -> index
          %96 = "memref.load"(%arg11, %95) : (memref<216xf32>, index) -> f32
          %97 = "arith.mulf"(%94, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %98 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
          %99 = "arith.cmpf"(%97, %98) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %100 = "arith.extui"(%99) : (i1) -> i32
          %101 = "affine.load"(%arg12) <{map = #map1}> : (memref<i32>) -> i32
          %102 = "arith.addi"(%101, %100) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "affine.store"(%102, %arg12) <{map = #map1}> : (i32, memref<i32>) -> ()
          %103 = "arith.index_cast"(%90) : (i32) -> index
          %104 = "memref.load"(%arg11, %103) : (memref<216xf32>, index) -> f32
          %105 = "arith.mulf"(%94, %104) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %106 = "arith.cmpf"(%105, %98) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %107 = "arith.extui"(%106) : (i1) -> i32
          %108 = "arith.addi"(%102, %107) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "affine.store"(%108, %arg12) <{map = #map1}> : (i32, memref<i32>) -> ()
          %109 = "arith.index_cast"(%92) : (i32) -> index
          %110 = "memref.load"(%arg11, %109) : (memref<216xf32>, index) -> f32
          %111 = "arith.mulf"(%94, %110) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %112 = "arith.cmpf"(%111, %98) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %113 = "arith.extui"(%112) : (i1) -> i32
          %114 = "arith.addi"(%108, %113) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "affine.store"(%114, %arg12) <{map = #map1}> : (i32, memref<i32>) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<216xf32>, memref<216xf32>) -> (), sym_name = "au_generated_candidate_27__l0u_l1u_l2u"}> ({
  ^bb0(%arg6: memref<216xf32>, %arg7: memref<216xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg8: index):
      %57 = "affine.apply"(%arg8) <{map = #map}> : (index) -> index
      %58 = "arith.index_cast"(%57) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg9: index):
        %59 = "affine.apply"(%arg9) <{map = #map}> : (index) -> index
        %60 = "arith.index_cast"(%59) : (index) -> i32
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg10: index):
          %61 = "affine.apply"(%arg10) <{map = #map}> : (index) -> index
          %62 = "arith.index_cast"(%61) : (index) -> i32
          %63 = "arith.constant"() <{value = 6 : i32}> : () -> i32
          %64 = "arith.muli"(%58, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %65 = "arith.addi"(%64, %60) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %66 = "arith.muli"(%65, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %67 = "arith.addi"(%66, %62) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %68 = "arith.index_cast"(%67) : (i32) -> index
          %69 = "memref.load"(%arg6, %68) : (memref<216xf32>, index) -> f32
          "memref.store"(%69, %arg7, %68) : (f32, memref<216xf32>, index) -> ()
          "affine.yield"() : () -> ()
        }) {unroll = 0 : i64} : () -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [0], partition_dim_array_1 = [0], partition_factor_array_0 = [4], partition_factor_array_1 = [4]} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<216xf32>, memref<216xf32>, memref<216xf32>) -> (), sym_name = "au_generated_candidate_9__direct"}> ({
  ^bb0(%arg0: memref<216xf32>, %arg1: memref<216xf32>, %arg2: memref<216xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg3: index):
      %0 = "affine.apply"(%arg3) <{map = #map}> : (index) -> index
      %1 = "arith.index_cast"(%0) : (index) -> i32
      %2 = "arith.constant"() <{value = 6 : i32}> : () -> i32
      %3 = "arith.muli"(%1, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %4 = "arith.constant"() <{value = -1 : i32}> : () -> i32
      %5 = "arith.addi"(%1, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %6 = "arith.muli"(%5, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %7 = "arith.constant"() <{value = 1 : i32}> : () -> i32
      %8 = "arith.addi"(%1, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %9 = "arith.muli"(%8, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg4: index):
        %10 = "affine.apply"(%arg4) <{map = #map}> : (index) -> index
        %11 = "arith.index_cast"(%10) : (index) -> i32
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg5: index):
          %12 = "affine.apply"(%arg5) <{map = #map}> : (index) -> index
          %13 = "arith.index_cast"(%12) : (index) -> i32
          %14 = "arith.addi"(%3, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %15 = "arith.muli"(%14, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %16 = "arith.addi"(%15, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %17 = "arith.addi"(%6, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %18 = "arith.muli"(%17, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %19 = "arith.addi"(%18, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %20 = "arith.addi"(%9, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %21 = "arith.muli"(%20, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %22 = "arith.addi"(%21, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %23 = "arith.addi"(%11, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %24 = "arith.addi"(%3, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %25 = "arith.muli"(%24, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %26 = "arith.addi"(%25, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %27 = "arith.addi"(%11, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %28 = "arith.addi"(%3, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %29 = "arith.muli"(%28, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %30 = "arith.addi"(%29, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %31 = "arith.addi"(%13, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %32 = "arith.addi"(%15, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %33 = "arith.addi"(%13, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %34 = "arith.addi"(%15, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %35 = "arith.index_cast"(%19) : (i32) -> index
          %36 = "memref.load"(%arg0, %35) : (memref<216xf32>, index) -> f32
          %37 = "arith.index_cast"(%22) : (i32) -> index
          %38 = "memref.load"(%arg0, %37) : (memref<216xf32>, index) -> f32
          %39 = "arith.addf"(%36, %38) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %40 = "arith.index_cast"(%26) : (i32) -> index
          %41 = "memref.load"(%arg0, %40) : (memref<216xf32>, index) -> f32
          %42 = "arith.addf"(%39, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %43 = "arith.index_cast"(%30) : (i32) -> index
          %44 = "memref.load"(%arg0, %43) : (memref<216xf32>, index) -> f32
          %45 = "arith.addf"(%42, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %46 = "arith.index_cast"(%32) : (i32) -> index
          %47 = "memref.load"(%arg0, %46) : (memref<216xf32>, index) -> f32
          %48 = "arith.addf"(%45, %47) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %49 = "arith.index_cast"(%34) : (i32) -> index
          %50 = "memref.load"(%arg0, %49) : (memref<216xf32>, index) -> f32
          %51 = "arith.addf"(%48, %50) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %52 = "arith.index_cast"(%16) : (i32) -> index
          %53 = "memref.load"(%arg1, %52) : (memref<216xf32>, index) -> f32
          %54 = "arith.addf"(%53, %51) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %55 = "arith.constant"() <{value = 6.000000e+00 : f32}> : () -> f32
          %56 = "arith.divf"(%54, %55) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%56, %arg2, %52) : (f32, memref<216xf32>, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
