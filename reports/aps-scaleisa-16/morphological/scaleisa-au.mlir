#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (12)>
#map4 = affine_map<() -> (5)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<12x12xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_40__direct"}> ({
  ^bb0(%arg14: memref<12x12xf32>, %arg15: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg16: index):
      %66 = "arith.index_cast"(%arg16) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg17: index):
        %67 = "arith.index_cast"(%arg17) : (index) -> i32
        %68 = "affine.load"(%arg14, %arg16, %arg17) <{map = #map}> : (memref<12x12xf32>, index, index) -> f32
        %69 = "arith.constant"() <{value = 1.280000e+02 : f32}> : () -> f32
        %70 = "arith.mulf"(%68, %69) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %71 = "arith.fptosi"(%70) : (f32) -> i32
        %72 = "arith.addi"(%67, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %73 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %74 = "arith.andi"(%72, %73) : (i32, i32) -> i32
        %75 = "arith.addi"(%71, %74) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %76 = "affine.load"(%arg15) <{map = #map1}> : (memref<i32>) -> i32
        %77 = "arith.addi"(%76, %75) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%77, %arg15) <{map = #map1}> : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}], function_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> (), sym_name = "au_generated_candidate_8__direct"}> ({
  ^bb0(%arg7: memref<12x12xf32>, %arg8: index, %arg9: i32, %arg10: memref<12x12xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg11: index):
      %33 = "arith.index_cast"(%arg11) : (index) -> i32
      %34 = "affine.load"(%arg7, %arg8, %arg11) <{map = #map}> : (memref<12x12xf32>, index, index) -> f32
      %35 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "affine.store"(%34, %35) <{map = #map1}> : (f32, memref<f32>) -> ()
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
      ^bb0(%arg12: index):
        %37 = "arith.index_cast"(%arg12) : (index) -> i32
        %38 = "arith.constant"() <{value = -2 : i32}> : () -> i32
        %39 = "arith.addi"(%37, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg13: index):
          %40 = "arith.index_cast"(%arg13) : (index) -> i32
          %41 = "arith.addi"(%40, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %42 = "arith.addi"(%arg9, %39) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %43 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %44 = "arith.cmpi"(%42, %43) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %45 = "arith.constant"() <{value = 11 : i32}> : () -> i32
          %46 = "arith.cmpi"(%42, %45) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %47 = "arith.andi"(%44, %46) : (i1, i1) -> i1
          %48 = "arith.constant"() <{value = true}> : () -> i1
          %49 = "arith.xori"(%47, %48) : (i1, i1) -> i1
          %50 = "arith.andi"(%49, %44) : (i1, i1) -> i1
          %51 = "arith.select"(%47, %45, %43) : (i1, i32, i32) -> i32
          %52 = "arith.select"(%50, %42, %51) : (i1, i32, i32) -> i32
          %53 = "arith.addi"(%33, %41) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %54 = "arith.cmpi"(%53, %43) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %55 = "arith.cmpi"(%53, %45) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %56 = "arith.andi"(%54, %55) : (i1, i1) -> i1
          %57 = "arith.xori"(%56, %48) : (i1, i1) -> i1
          %58 = "arith.andi"(%57, %54) : (i1, i1) -> i1
          %59 = "arith.select"(%56, %45, %43) : (i1, i32, i32) -> i32
          %60 = "arith.select"(%58, %53, %59) : (i1, i32, i32) -> i32
          %61 = "arith.index_cast"(%52) : (i32) -> index
          %62 = "arith.index_cast"(%60) : (i32) -> index
          %63 = "memref.load"(%arg7, %61, %62) : (memref<12x12xf32>, index, index) -> f32
          %64 = "affine.load"(%35) <{map = #map1}> : (memref<f32>) -> f32
          %65 = "arith.cmpf"(%63, %64) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
          "scf.if"(%65) ({
            "affine.store"(%63, %35) <{map = #map1}> : (f32, memref<f32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %36 = "affine.load"(%35) <{map = #map1}> : (memref<f32>) -> f32
      "affine.store"(%36, %arg10, %arg8, %arg11) <{map = #map}> : (f32, memref<12x12xf32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}], function_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> (), sym_name = "au_generated_candidate_9__direct"}> ({
  ^bb0(%arg0: memref<12x12xf32>, %arg1: index, %arg2: i32, %arg3: memref<12x12xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg4: index):
      %0 = "arith.index_cast"(%arg4) : (index) -> i32
      %1 = "affine.load"(%arg0, %arg1, %arg4) <{map = #map}> : (memref<12x12xf32>, index, index) -> f32
      %2 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "affine.store"(%1, %2) <{map = #map1}> : (f32, memref<f32>) -> ()
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
      ^bb0(%arg5: index):
        %4 = "arith.index_cast"(%arg5) : (index) -> i32
        %5 = "arith.constant"() <{value = -2 : i32}> : () -> i32
        %6 = "arith.addi"(%4, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg6: index):
          %7 = "arith.index_cast"(%arg6) : (index) -> i32
          %8 = "arith.addi"(%7, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %9 = "arith.addi"(%arg2, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %10 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %11 = "arith.cmpi"(%9, %10) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %12 = "arith.constant"() <{value = 11 : i32}> : () -> i32
          %13 = "arith.cmpi"(%9, %12) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %14 = "arith.andi"(%11, %13) : (i1, i1) -> i1
          %15 = "arith.constant"() <{value = true}> : () -> i1
          %16 = "arith.xori"(%14, %15) : (i1, i1) -> i1
          %17 = "arith.andi"(%16, %11) : (i1, i1) -> i1
          %18 = "arith.select"(%14, %12, %10) : (i1, i32, i32) -> i32
          %19 = "arith.select"(%17, %9, %18) : (i1, i32, i32) -> i32
          %20 = "arith.addi"(%0, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %21 = "arith.cmpi"(%20, %10) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %22 = "arith.cmpi"(%20, %12) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %23 = "arith.andi"(%21, %22) : (i1, i1) -> i1
          %24 = "arith.xori"(%23, %15) : (i1, i1) -> i1
          %25 = "arith.andi"(%24, %21) : (i1, i1) -> i1
          %26 = "arith.select"(%23, %12, %10) : (i1, i32, i32) -> i32
          %27 = "arith.select"(%25, %20, %26) : (i1, i32, i32) -> i32
          %28 = "arith.index_cast"(%19) : (i32) -> index
          %29 = "arith.index_cast"(%27) : (i32) -> index
          %30 = "memref.load"(%arg0, %28, %29) : (memref<12x12xf32>, index, index) -> f32
          %31 = "affine.load"(%2) <{map = #map1}> : (memref<f32>) -> f32
          %32 = "arith.cmpf"(%30, %31) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          "scf.if"(%32) ({
            "affine.store"(%30, %2) <{map = #map1}> : (f32, memref<f32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %3 = "affine.load"(%2) <{map = #map1}> : (memref<f32>) -> f32
      "affine.store"(%3, %arg3, %arg1, %arg4) <{map = #map}> : (f32, memref<12x12xf32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
