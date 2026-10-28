#map = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1, 1)>
#map3 = affine_map<(d0, d1) -> (d0, d1, 2)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<() -> (5)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<() -> (16)>
#map8 = affine_map<() -> (3)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}], function_type = (memref<16x16x3xf32>, index, i32, memref<16x16xf32>) -> (), sym_name = "au_generated_candidate_10__direct"}> ({
  ^bb0(%arg12: memref<16x16x3xf32>, %arg13: index, %arg14: i32, %arg15: memref<16x16xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map7}> ({
    ^bb0(%arg16: index):
      %52 = "arith.index_cast"(%arg16) : (index) -> i32
      %53 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      %54 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      "affine.store"(%54, %53) <{map = #map}> : (f32, memref<f32>) -> ()
      %55 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "affine.store"(%54, %55) <{map = #map}> : (f32, memref<f32>) -> ()
      %56 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "affine.store"(%54, %56) <{map = #map}> : (f32, memref<f32>) -> ()
      %57 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "affine.store"(%54, %57) <{map = #map}> : (f32, memref<f32>) -> ()
      %58 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "affine.store"(%54, %58) <{map = #map}> : (f32, memref<f32>) -> ()
      %59 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "affine.store"(%54, %59) <{map = #map}> : (f32, memref<f32>) -> ()
      %60 = "affine.load"(%arg12, %arg13, %arg16) <{map = #map1}> : (memref<16x16x3xf32>, index, index) -> f32
      %61 = "affine.load"(%arg12, %arg13, %arg16) <{map = #map2}> : (memref<16x16x3xf32>, index, index) -> f32
      %62 = "affine.load"(%arg12, %arg13, %arg16) <{map = #map3}> : (memref<16x16x3xf32>, index, index) -> f32
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
      ^bb0(%arg17: index):
        %85 = "arith.index_cast"(%arg17) : (index) -> i32
        %86 = "arith.constant"() <{value = -2 : i32}> : () -> i32
        %87 = "arith.addi"(%85, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
        ^bb0(%arg18: index):
          %88 = "arith.index_cast"(%arg18) : (index) -> i32
          %89 = "arith.addi"(%88, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %90 = "arith.addi"(%arg14, %87) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %91 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %92 = "arith.cmpi"(%90, %91) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %93 = "arith.constant"() <{value = 15 : i32}> : () -> i32
          %94 = "arith.cmpi"(%90, %93) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %95 = "arith.andi"(%92, %94) : (i1, i1) -> i1
          %96 = "arith.constant"() <{value = true}> : () -> i1
          %97 = "arith.xori"(%95, %96) : (i1, i1) -> i1
          %98 = "arith.andi"(%97, %92) : (i1, i1) -> i1
          %99 = "arith.select"(%95, %93, %91) : (i1, i32, i32) -> i32
          %100 = "arith.select"(%98, %90, %99) : (i1, i32, i32) -> i32
          %101 = "arith.addi"(%52, %89) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %102 = "arith.cmpi"(%101, %91) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %103 = "arith.cmpi"(%101, %93) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %104 = "arith.andi"(%102, %103) : (i1, i1) -> i1
          %105 = "arith.subi"(%96, %104) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
          %106 = "arith.andi"(%105, %102) : (i1, i1) -> i1
          %107 = "arith.select"(%104, %93, %91) : (i1, i32, i32) -> i32
          %108 = "arith.select"(%106, %101, %107) : (i1, i32, i32) -> i32
          %109 = "arith.index_cast"(%100) : (i32) -> index
          %110 = "arith.index_cast"(%108) : (i32) -> index
          %111 = "arith.constant"() <{value = 0 : index}> : () -> index
          %112 = "memref.load"(%arg12, %109, %110, %111) : (memref<16x16x3xf32>, index, index, index) -> f32
          %113 = "arith.subf"(%112, %60) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %114 = "arith.constant"() <{value = 1 : index}> : () -> index
          %115 = "memref.load"(%arg12, %109, %110, %114) : (memref<16x16x3xf32>, index, index, index) -> f32
          %116 = "arith.subf"(%115, %61) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %117 = "arith.constant"() <{value = 2 : index}> : () -> index
          %118 = "memref.load"(%arg12, %109, %110, %117) : (memref<16x16x3xf32>, index, index, index) -> f32
          %119 = "arith.subf"(%118, %62) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %120 = "arith.mulf"(%113, %113) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %121 = "affine.load"(%53) <{map = #map}> : (memref<f32>) -> f32
          %122 = "arith.addf"(%121, %120) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%122, %53) <{map = #map}> : (f32, memref<f32>) -> ()
          %123 = "arith.mulf"(%116, %116) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %124 = "affine.load"(%55) <{map = #map}> : (memref<f32>) -> f32
          %125 = "arith.addf"(%124, %123) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%125, %55) <{map = #map}> : (f32, memref<f32>) -> ()
          %126 = "arith.mulf"(%119, %119) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %127 = "affine.load"(%56) <{map = #map}> : (memref<f32>) -> f32
          %128 = "arith.addf"(%127, %126) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%128, %56) <{map = #map}> : (f32, memref<f32>) -> ()
          %129 = "arith.mulf"(%113, %116) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %130 = "affine.load"(%57) <{map = #map}> : (memref<f32>) -> f32
          %131 = "arith.addf"(%130, %129) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%131, %57) <{map = #map}> : (f32, memref<f32>) -> ()
          %132 = "arith.mulf"(%113, %119) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %133 = "affine.load"(%58) <{map = #map}> : (memref<f32>) -> f32
          %134 = "arith.addf"(%133, %132) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%134, %58) <{map = #map}> : (f32, memref<f32>) -> ()
          %135 = "arith.mulf"(%116, %119) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %136 = "affine.load"(%59) <{map = #map}> : (memref<f32>) -> f32
          %137 = "arith.addf"(%136, %135) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%137, %59) <{map = #map}> : (f32, memref<f32>) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %63 = "affine.load"(%53) <{map = #map}> : (memref<f32>) -> f32
      %64 = "affine.load"(%55) <{map = #map}> : (memref<f32>) -> f32
      %65 = "arith.addf"(%63, %64) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %66 = "affine.load"(%56) <{map = #map}> : (memref<f32>) -> f32
      %67 = "arith.addf"(%65, %66) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %68 = "affine.load"(%57) <{map = #map}> : (memref<f32>) -> f32
      %69 = "arith.cmpf"(%68, %54) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %70 = "scf.if"(%69) ({
        %84 = "arith.negf"(%68) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%84) : (f32) -> ()
      }, {
        "scf.yield"(%68) : (f32) -> ()
      }) : (i1) -> f32
      %71 = "affine.load"(%58) <{map = #map}> : (memref<f32>) -> f32
      %72 = "arith.cmpf"(%71, %54) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %73 = "scf.if"(%72) ({
        %83 = "arith.negf"(%71) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%83) : (f32) -> ()
      }, {
        "scf.yield"(%71) : (f32) -> ()
      }) : (i1) -> f32
      %74 = "arith.addf"(%70, %73) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %75 = "affine.load"(%59) <{map = #map}> : (memref<f32>) -> f32
      %76 = "arith.cmpf"(%75, %54) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %77 = "scf.if"(%76) ({
        %82 = "arith.negf"(%75) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%82) : (f32) -> ()
      }, {
        "scf.yield"(%75) : (f32) -> ()
      }) : (i1) -> f32
      %78 = "arith.addf"(%74, %77) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %79 = "arith.constant"() <{value = 5.000000e-01 : f32}> : () -> f32
      %80 = "arith.mulf"(%78, %79) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %81 = "arith.subf"(%67, %80) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%81, %arg15, %arg13, %arg16) <{map = #map6}> : (f32, memref<16x16xf32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}], function_type = (memref<16x16xf32>, index, i32, memref<16x16xi32>) -> (), sym_name = "au_generated_candidate_24__direct"}> ({
  ^bb0(%arg5: memref<16x16xf32>, %arg6: index, %arg7: i32, %arg8: memref<16x16xi32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map7}> ({
    ^bb0(%arg9: index):
      %14 = "arith.index_cast"(%arg9) : (index) -> i32
      %15 = "affine.load"(%arg5, %arg6, %arg9) <{map = #map6}> : (memref<16x16xf32>, index, index) -> f32
      %16 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
      %17 = "arith.cmpf"(%15, %16) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
      %18 = "arith.extui"(%17) : (i1) -> i32
      %19 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "affine.store"(%18, %19) <{map = #map}> : (i32, memref<i32>) -> ()
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map8}> ({
      ^bb0(%arg10: index):
        %21 = "arith.index_cast"(%arg10) : (index) -> i32
        %22 = "arith.constant"() <{value = -1 : i32}> : () -> i32
        %23 = "arith.addi"(%21, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map8}> ({
        ^bb0(%arg11: index):
          %24 = "arith.index_cast"(%arg11) : (index) -> i32
          %25 = "arith.addi"(%24, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %26 = "arith.addi"(%arg7, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %27 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %28 = "arith.cmpi"(%26, %27) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %29 = "arith.constant"() <{value = 15 : i32}> : () -> i32
          %30 = "arith.cmpi"(%26, %29) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %31 = "arith.andi"(%28, %30) : (i1, i1) -> i1
          %32 = "arith.constant"() <{value = true}> : () -> i1
          %33 = "arith.xori"(%31, %32) : (i1, i1) -> i1
          %34 = "arith.andi"(%33, %28) : (i1, i1) -> i1
          %35 = "arith.select"(%31, %29, %27) : (i1, i32, i32) -> i32
          %36 = "arith.select"(%34, %26, %35) : (i1, i32, i32) -> i32
          %37 = "arith.addi"(%14, %25) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %38 = "arith.cmpi"(%37, %27) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %39 = "arith.cmpi"(%37, %29) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %40 = "arith.andi"(%38, %39) : (i1, i1) -> i1
          %41 = "arith.xori"(%40, %32) : (i1, i1) -> i1
          %42 = "arith.andi"(%41, %38) : (i1, i1) -> i1
          %43 = "arith.select"(%40, %29, %27) : (i1, i32, i32) -> i32
          %44 = "arith.select"(%42, %37, %43) : (i1, i32, i32) -> i32
          %45 = "arith.cmpi"(%23, %27) <{predicate = 1 : i64}> : (i32, i32) -> i1
          %46 = "scf.if"(%45) ({
            "scf.yield"(%32) : (i1) -> ()
          }, {
            %51 = "arith.cmpi"(%25, %27) <{predicate = 1 : i64}> : (i32, i32) -> i1
            "scf.yield"(%51) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%46) ({
            %47 = "arith.index_cast"(%36) : (i32) -> index
            %48 = "arith.index_cast"(%44) : (i32) -> index
            %49 = "memref.load"(%arg5, %47, %48) : (memref<16x16xf32>, index, index) -> f32
            %50 = "arith.cmpf"(%49, %15) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
            "scf.if"(%50) ({
              "affine.store"(%27, %19) <{map = #map}> : (i32, memref<i32>) -> ()
              "scf.yield"() : () -> ()
            }, {
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %20 = "affine.load"(%19) <{map = #map}> : (memref<i32>) -> i32
      "affine.store"(%20, %arg8, %arg6, %arg9) <{map = #map6}> : (i32, memref<16x16xi32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<16x16xi32>, memref<16x16xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_27__direct"}> ({
  ^bb0(%arg0: memref<16x16xi32>, %arg1: memref<16x16xf32>, %arg2: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map7}> ({
    ^bb0(%arg3: index):
      %0 = "arith.index_cast"(%arg3) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map7}> ({
      ^bb0(%arg4: index):
        %1 = "arith.index_cast"(%arg4) : (index) -> i32
        %2 = "affine.load"(%arg0, %arg3, %arg4) <{map = #map6}> : (memref<16x16xi32>, index, index) -> i32
        %3 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %4 = "arith.cmpi"(%2, %3) <{predicate = 1 : i64}> : (i32, i32) -> i1
        %5 = "scf.if"(%4) ({
          %8 = "affine.load"(%arg1, %arg3, %arg4) <{map = #map6}> : (memref<16x16xf32>, index, index) -> f32
          %9 = "arith.constant"() <{value = 3.200000e+01 : f32}> : () -> f32
          %10 = "arith.mulf"(%8, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %11 = "arith.fptosi"(%10) : (f32) -> i32
          %12 = "arith.addi"(%11, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %13 = "arith.addi"(%12, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "scf.yield"(%13) : (i32) -> ()
        }, {
          "scf.yield"(%3) : (i32) -> ()
        }) : (i1) -> i32
        %6 = "affine.load"(%arg2) <{map = #map}> : (memref<i32>) -> i32
        %7 = "arith.addi"(%6, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%7, %arg2) <{map = #map}> : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
