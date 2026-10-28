#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (3)>
#map4 = affine_map<() -> (8)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x8xi32>, index, i32, memref<8x8xf32>, memref<8x8xf32>) -> (), sym_name = "au_generated_candidate_0__direct"}> ({
  ^bb0(%arg13: memref<8x8xi32>, %arg14: index, %arg15: i32, %arg16: memref<8x8xf32>, %arg17: memref<8x8xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg18: index):
      %92 = "arith.index_cast"(%arg18) : (index) -> i32
      %93 = "affine.load"(%arg13, %arg14, %arg18) <{map = #map}> : (memref<8x8xi32>, index, index) -> i32
      %94 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "affine.store"(%93, %94) <{map = #map1}> : (i32, memref<i32>) -> ()
      %95 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      %96 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
      "affine.store"(%96, %95) <{map = #map1}> : (f32, memref<f32>) -> ()
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg19: index):
        %98 = "arith.index_cast"(%arg19) : (index) -> i32
        %99 = "arith.constant"() <{value = -1 : i32}> : () -> i32
        %100 = "arith.addi"(%98, %99) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg20: index):
          %101 = "arith.index_cast"(%arg20) : (index) -> i32
          %102 = "arith.addi"(%101, %99) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %103 = "arith.addi"(%arg15, %100) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %104 = "arith.constant"() <{value = 0 : i32}> : () -> i32
          %105 = "arith.cmpi"(%103, %104) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %106 = "arith.constant"() <{value = 7 : i32}> : () -> i32
          %107 = "arith.cmpi"(%103, %106) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %108 = "arith.andi"(%105, %107) : (i1, i1) -> i1
          %109 = "arith.constant"() <{value = true}> : () -> i1
          %110 = "arith.xori"(%108, %109) : (i1, i1) -> i1
          %111 = "arith.andi"(%110, %105) : (i1, i1) -> i1
          %112 = "arith.select"(%108, %106, %104) : (i1, i32, i32) -> i32
          %113 = "arith.select"(%111, %103, %112) : (i1, i32, i32) -> i32
          %114 = "arith.addi"(%92, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %115 = "arith.cmpi"(%114, %104) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %116 = "arith.cmpi"(%114, %106) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %117 = "arith.andi"(%115, %116) : (i1, i1) -> i1
          %118 = "arith.xori"(%117, %109) : (i1, i1) -> i1
          %119 = "arith.andi"(%118, %115) : (i1, i1) -> i1
          %120 = "arith.select"(%117, %106, %104) : (i1, i32, i32) -> i32
          %121 = "arith.select"(%119, %114, %120) : (i1, i32, i32) -> i32
          %122 = "affine.load"(%arg16, %arg14, %arg18) <{map = #map}> : (memref<8x8xf32>, index, index) -> f32
          %123 = "arith.index_cast"(%113) : (i32) -> index
          %124 = "arith.index_cast"(%121) : (i32) -> index
          %125 = "memref.load"(%arg16, %123, %124) : (memref<8x8xf32>, index, index) -> f32
          %126 = "arith.subf"(%122, %125) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %127 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
          %128 = "arith.cmpf"(%126, %127) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %129 = "scf.if"(%128) ({
            %142 = "arith.negf"(%126) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
            "scf.yield"(%142) : (f32) -> ()
          }, {
            "scf.yield"(%126) : (f32) -> ()
          }) : (i1) -> f32
          %130 = "memref.load"(%arg17, %123, %124) : (memref<8x8xf32>, index, index) -> f32
          %131 = "arith.addf"(%129, %130) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %132 = "affine.load"(%95) <{map = #map1}> : (memref<f32>) -> f32
          %133 = "arith.cmpf"(%131, %132) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %134 = "scf.if"(%133) ({
            "scf.yield"(%109) : (i1) -> ()
          }, {
            %136 = "arith.cmpf"(%131, %132) <{fastmath = #arith.fastmath<none>, predicate = 1 : i64}> : (f32, f32) -> i1
            %137 = "scf.if"(%136) ({
              %139 = "memref.load"(%arg13, %123, %124) : (memref<8x8xi32>, index, index) -> i32
              %140 = "affine.load"(%94) <{map = #map1}> : (memref<i32>) -> i32
              %141 = "arith.cmpi"(%139, %140) <{predicate = 2 : i64}> : (i32, i32) -> i1
              "scf.yield"(%141) : (i1) -> ()
            }, {
              %138 = "arith.constant"() <{value = false}> : () -> i1
              "scf.yield"(%138) : (i1) -> ()
            }) : (i1) -> i1
            "scf.yield"(%137) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%134) ({
            "affine.store"(%131, %95) <{map = #map1}> : (f32, memref<f32>) -> ()
            %135 = "memref.load"(%arg13, %123, %124) : (memref<8x8xi32>, index, index) -> i32
            "affine.store"(%135, %94) <{map = #map1}> : (i32, memref<i32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %97 = "affine.load"(%94) <{map = #map1}> : (memref<i32>) -> i32
      "affine.store"(%97, %arg13, %arg14, %arg18) <{map = #map}> : (i32, memref<8x8xi32>, index, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x8xf32>, memref<8x8xf32>) -> (), sym_name = "au_generated_candidate_10__direct"}> ({
  ^bb0(%arg9: memref<8x8xf32>, %arg10: memref<8x8xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg11: index):
      %35 = "arith.index_cast"(%arg11) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
      ^bb0(%arg12: index):
        %36 = "arith.index_cast"(%arg12) : (index) -> i32
        %37 = "arith.constant"() <{value = -1 : i32}> : () -> i32
        %38 = "arith.addi"(%36, %37) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %39 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %40 = "arith.cmpi"(%38, %39) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %41 = "arith.constant"() <{value = 7 : i32}> : () -> i32
        %42 = "arith.cmpi"(%38, %41) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %43 = "arith.andi"(%40, %42) : (i1, i1) -> i1
        %44 = "arith.constant"() <{value = true}> : () -> i1
        %45 = "arith.xori"(%43, %44) : (i1, i1) -> i1
        %46 = "arith.andi"(%45, %40) : (i1, i1) -> i1
        %47 = "arith.select"(%43, %41, %39) : (i1, i32, i32) -> i32
        %48 = "arith.select"(%46, %38, %47) : (i1, i32, i32) -> i32
        %49 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        %50 = "arith.addi"(%36, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %51 = "arith.cmpi"(%50, %39) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %52 = "arith.cmpi"(%50, %41) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %53 = "arith.andi"(%51, %52) : (i1, i1) -> i1
        %54 = "arith.xori"(%53, %44) : (i1, i1) -> i1
        %55 = "arith.andi"(%54, %51) : (i1, i1) -> i1
        %56 = "arith.select"(%53, %41, %39) : (i1, i32, i32) -> i32
        %57 = "arith.select"(%55, %50, %56) : (i1, i32, i32) -> i32
        %58 = "arith.addi"(%35, %37) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %59 = "arith.cmpi"(%58, %39) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %60 = "arith.cmpi"(%58, %41) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %61 = "arith.andi"(%59, %60) : (i1, i1) -> i1
        %62 = "arith.xori"(%61, %44) : (i1, i1) -> i1
        %63 = "arith.andi"(%62, %59) : (i1, i1) -> i1
        %64 = "arith.select"(%61, %41, %39) : (i1, i32, i32) -> i32
        %65 = "arith.select"(%63, %58, %64) : (i1, i32, i32) -> i32
        %66 = "arith.addi"(%35, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %67 = "arith.cmpi"(%66, %39) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %68 = "arith.cmpi"(%66, %41) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %69 = "arith.andi"(%67, %68) : (i1, i1) -> i1
        %70 = "arith.subi"(%44, %69) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
        %71 = "arith.andi"(%70, %67) : (i1, i1) -> i1
        %72 = "arith.select"(%69, %41, %39) : (i1, i32, i32) -> i32
        %73 = "arith.select"(%71, %66, %72) : (i1, i32, i32) -> i32
        %74 = "arith.index_cast"(%57) : (i32) -> index
        %75 = "memref.load"(%arg9, %arg11, %74) : (memref<8x8xf32>, index, index) -> f32
        %76 = "arith.index_cast"(%48) : (i32) -> index
        %77 = "memref.load"(%arg9, %arg11, %76) : (memref<8x8xf32>, index, index) -> f32
        %78 = "arith.subf"(%75, %77) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %79 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
        %80 = "arith.cmpf"(%78, %79) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %81 = "scf.if"(%80) ({
          %91 = "arith.negf"(%78) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%91) : (f32) -> ()
        }, {
          "scf.yield"(%78) : (f32) -> ()
        }) : (i1) -> f32
        %82 = "arith.index_cast"(%73) : (i32) -> index
        %83 = "memref.load"(%arg9, %82, %arg12) : (memref<8x8xf32>, index, index) -> f32
        %84 = "arith.index_cast"(%65) : (i32) -> index
        %85 = "memref.load"(%arg9, %84, %arg12) : (memref<8x8xf32>, index, index) -> f32
        %86 = "arith.subf"(%83, %85) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %87 = "arith.cmpf"(%86, %79) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %88 = "scf.if"(%87) ({
          %90 = "arith.negf"(%86) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%90) : (f32) -> ()
        }, {
          "scf.yield"(%86) : (f32) -> ()
        }) : (i1) -> f32
        %89 = "arith.addf"(%81, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%89, %arg10, %arg11, %arg12) <{map = #map}> : (f32, memref<8x8xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x8xf32>, memref<8x8xi32>) -> (), sym_name = "au_generated_candidate_21__l0u_l1u"}> ({
  ^bb0(%arg5: memref<8x8xf32>, %arg6: memref<8x8xi32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg7: index):
      %5 = "arith.index_cast"(%arg7) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
      ^bb0(%arg8: index):
        %6 = "arith.index_cast"(%arg8) : (index) -> i32
        %7 = "arith.constant"() <{value = 24 : i32}> : () -> i32
        %8 = "arith.cmpi"(%6, %7) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %9 = "arith.extui"(%8) : (i1) -> i32
        %10 = "arith.constant"() <{value = 56 : i32}> : () -> i32
        %11 = "arith.cmpi"(%6, %10) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %12 = "arith.extui"(%11) : (i1) -> i32
        %13 = "arith.addi"(%9, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %14 = "arith.constant"() <{value = 30 : i32}> : () -> i32
        %15 = "arith.cmpi"(%5, %14) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %16 = "arith.extui"(%15) : (i1) -> i32
        %17 = "arith.addi"(%13, %16) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %18 = "arith.sitofp"(%17) : (i32) -> f32
        %19 = "arith.constant"() <{value = 1.000000e-01 : f32}> : () -> f32
        %20 = "arith.mulf"(%18, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %21 = "arith.constant"() <{value = 7 : i32}> : () -> i32
        %22 = "arith.muli"(%6, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %23 = "arith.constant"() <{value = 5 : i32}> : () -> i32
        %24 = "arith.muli"(%5, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %25 = "arith.addi"(%22, %24) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %26 = "arith.constant"() <{value = 9 : i32}> : () -> i32
        %27 = "arith.remsi"(%25, %26) : (i32, i32) -> i32
        %28 = "arith.sitofp"(%27) : (i32) -> f32
        %29 = "arith.constant"() <{value = 5.000000e-03 : f32}> : () -> f32
        %30 = "arith.mulf"(%28, %29) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %31 = "arith.addf"(%20, %30) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%31, %arg5, %arg7, %arg8) <{map = #map}> : (f32, memref<8x8xf32>, index, index) -> ()
        %32 = "arith.constant"() <{value = 8 : i32}> : () -> i32
        %33 = "arith.muli"(%5, %32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %34 = "arith.addi"(%33, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%34, %arg6, %arg7, %arg8) <{map = #map}> : (i32, memref<8x8xi32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [1], partition_dim_array_1 = [1], partition_factor_array_0 = [8], partition_factor_array_1 = [8]} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x8xi32>, memref<i32>) -> (), sym_name = "au_generated_candidate_35__l0u_l1u_l2u"}> ({
  ^bb0(%arg0: memref<8x8xi32>, %arg1: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg2: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 8 : index, upperBoundMap = #map4}> ({
      ^bb0(%arg3: index):
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg4: index):
          %0 = "affine.load"(%arg0, %arg2, %arg4) <{map = #map}> : (memref<8x8xi32>, index, index) -> i32
          %1 = "arith.constant"() <{value = 255 : i32}> : () -> i32
          %2 = "arith.andi"(%0, %1) : (i32, i32) -> i32
          %3 = "affine.load"(%arg1) <{map = #map1}> : (memref<i32>) -> i32
          %4 = "arith.addi"(%3, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "affine.store"(%4, %arg1) <{map = #map1}> : (i32, memref<i32>) -> ()
          "affine.yield"() : () -> ()
        }) {unroll = 0 : i64} : () -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [8]} : () -> ()
}) : () -> ()
