#map = affine_map<() -> ()>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (3)>
#map3 = affine_map<(d0, d1) -> (d0, d1, 0)>
#map4 = affine_map<(d0, d1) -> (d0, d1, 1)>
#map5 = affine_map<(d0, d1) -> (d0, d1, 2)>
#map6 = affine_map<() -> (12)>
#map7 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1, 0)>
#map8 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1, 1)>
#map9 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1, 2)>
#map10 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>
#map11 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{}, {scaleisa.noalias}, {scaleisa.noalias}, {}], function_type = (i32, memref<12x12x3xf32>, memref<12x12x3xf32>, index) -> (), sym_name = "au_generated_candidate_12__direct"}> ({
  ^bb0(%arg5: i32, %arg6: memref<12x12x3xf32>, %arg7: memref<12x12x3xf32>, %arg8: index):
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map6}> ({
    ^bb0(%arg9: index):
      %20 = "arith.index_cast"(%arg9) : (index) -> i32
      %21 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      %22 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      "affine.store"(%22, %21) <{map = #map}> : (f32, memref<f32>) -> ()
      %23 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "affine.store"(%22, %23) <{map = #map}> : (f32, memref<f32>) -> ()
      %24 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "affine.store"(%22, %24) <{map = #map}> : (f32, memref<f32>) -> ()
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg10: index):
        %32 = "arith.index_cast"(%arg10) : (index) -> i32
        %33 = "arith.constant"() <{value = -1 : i32}> : () -> i32
        %34 = "arith.addi"(%32, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg11: index):
          %35 = "arith.index_cast"(%arg11) : (index) -> i32
          %36 = "arith.addi"(%35, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %37 = "arith.addi"(%arg5, %34) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %38 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %39 = "arith.cmpi"(%37, %38) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %40 = "arith.constant"() <{value = 11 : i32}> : () -> i32
          %41 = "arith.cmpi"(%37, %40) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %42 = "arith.andi"(%39, %41) : (i1, i1) -> i1
          %43 = "arith.constant"() <{value = true}> : () -> i1
          %44 = "arith.xori"(%42, %43) : (i1, i1) -> i1
          %45 = "arith.andi"(%44, %39) : (i1, i1) -> i1
          %46 = "arith.select"(%42, %40, %38) : (i1, i32, i32) -> i32
          %47 = "arith.select"(%45, %37, %46) : (i1, i32, i32) -> i32
          %48 = "arith.addi"(%20, %36) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %49 = "arith.cmpi"(%48, %38) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %50 = "arith.cmpi"(%48, %40) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %51 = "arith.andi"(%49, %50) : (i1, i1) -> i1
          %52 = "arith.xori"(%51, %43) : (i1, i1) -> i1
          %53 = "arith.andi"(%52, %49) : (i1, i1) -> i1
          %54 = "arith.select"(%51, %40, %38) : (i1, i32, i32) -> i32
          %55 = "arith.select"(%53, %48, %54) : (i1, i32, i32) -> i32
          %56 = "arith.index_cast"(%47) : (i32) -> index
          %57 = "arith.index_cast"(%55) : (i32) -> index
          %58 = "arith.constant"() <{value = 0 : index}> : () -> index
          %59 = "memref.load"(%arg6, %56, %57, %58) : (memref<12x12x3xf32>, index, index, index) -> f32
          %60 = "affine.load"(%21) <{map = #map}> : (memref<f32>) -> f32
          %61 = "arith.addf"(%60, %59) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%61, %21) <{map = #map}> : (f32, memref<f32>) -> ()
          %62 = "arith.constant"() <{value = 1 : index}> : () -> index
          %63 = "memref.load"(%arg6, %56, %57, %62) : (memref<12x12x3xf32>, index, index, index) -> f32
          %64 = "affine.load"(%23) <{map = #map}> : (memref<f32>) -> f32
          %65 = "arith.addf"(%64, %63) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%65, %23) <{map = #map}> : (f32, memref<f32>) -> ()
          %66 = "arith.constant"() <{value = 2 : index}> : () -> index
          %67 = "memref.load"(%arg6, %56, %57, %66) : (memref<12x12x3xf32>, index, index, index) -> f32
          %68 = "affine.load"(%24) <{map = #map}> : (memref<f32>) -> f32
          %69 = "arith.addf"(%68, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%69, %24) <{map = #map}> : (f32, memref<f32>) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %25 = "affine.load"(%21) <{map = #map}> : (memref<f32>) -> f32
      %26 = "arith.constant"() <{value = 9.000000e+00 : f32}> : () -> f32
      %27 = "arith.divf"(%25, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%27, %arg7, %arg8, %arg9) <{map = #map3}> : (f32, memref<12x12x3xf32>, index, index) -> ()
      %28 = "affine.load"(%23) <{map = #map}> : (memref<f32>) -> f32
      %29 = "arith.divf"(%28, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%29, %arg7, %arg8, %arg9) <{map = #map4}> : (f32, memref<12x12x3xf32>, index, index) -> ()
      %30 = "affine.load"(%24) <{map = #map}> : (memref<f32>) -> f32
      %31 = "arith.divf"(%30, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%31, %arg7, %arg8, %arg9) <{map = #map5}> : (f32, memref<12x12x3xf32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<12x12x3xf32>, memref<12x12xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_24__direct"}> ({
  ^bb0(%arg0: memref<12x12x3xf32>, %arg1: memref<12x12xf32>, %arg2: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map11}> ({
    ^bb0(%arg3: index):
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map11}> ({
      ^bb0(%arg4: index):
        %0 = "affine.load"(%arg0, %arg3, %arg4) <{map = #map7}> : (memref<12x12x3xf32>, index, index) -> f32
        %1 = "arith.constant"() <{value = 1.700000e+01 : f32}> : () -> f32
        %2 = "arith.mulf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %3 = "affine.load"(%arg0, %arg3, %arg4) <{map = #map8}> : (memref<12x12x3xf32>, index, index) -> f32
        %4 = "arith.constant"() <{value = 2.300000e+01 : f32}> : () -> f32
        %5 = "arith.mulf"(%3, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %6 = "arith.addf"(%2, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %7 = "affine.load"(%arg0, %arg3, %arg4) <{map = #map9}> : (memref<12x12x3xf32>, index, index) -> f32
        %8 = "arith.constant"() <{value = 3.100000e+01 : f32}> : () -> f32
        %9 = "arith.mulf"(%7, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %10 = "arith.addf"(%6, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %11 = "affine.load"(%arg1, %arg3, %arg4) <{map = #map10}> : (memref<12x12xf32>, index, index) -> f32
        %12 = "arith.constant"() <{value = 1.100000e+01 : f32}> : () -> f32
        %13 = "arith.mulf"(%11, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %14 = "arith.addf"(%10, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %15 = "arith.constant"() <{value = 6.400000e+01 : f32}> : () -> f32
        %16 = "arith.mulf"(%14, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %17 = "arith.fptosi"(%16) : (f32) -> i32
        %18 = "affine.load"(%arg2) <{map = #map}> : (memref<i32>) -> i32
        %19 = "arith.addi"(%18, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%19, %arg2) <{map = #map}> : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
