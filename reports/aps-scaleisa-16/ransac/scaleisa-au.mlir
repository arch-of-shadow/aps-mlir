#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<(d0) -> (d0, 1)>
#map2 = affine_map<(d0) -> (d0, 2)>
#map3 = affine_map<() -> ()>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {}, {}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<32x3xf32>, f32, f32, f32, f32, memref<i32>, memref<f32>) -> (), sym_name = "au_generated_candidate_0__direct"}> ({
  ^bb0(%arg2: memref<32x3xf32>, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: memref<i32>, %arg8: memref<f32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg9: index):
      %55 = "affine.load"(%arg2, %arg9) <{map = #map}> : (memref<32x3xf32>, index) -> f32
      %56 = "arith.subf"(%55, %arg3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %57 = "affine.load"(%arg2, %arg9) <{map = #map1}> : (memref<32x3xf32>, index) -> f32
      %58 = "arith.subf"(%57, %arg4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %59 = "affine.load"(%arg2, %arg9) <{map = #map2}> : (memref<32x3xf32>, index) -> f32
      %60 = "arith.subf"(%59, %arg5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %61 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
      %62 = "arith.cmpf"(%56, %61) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %63 = "scf.if"(%62) ({
        %83 = "arith.negf"(%56) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%83) : (f32) -> ()
      }, {
        "scf.yield"(%56) : (f32) -> ()
      }) : (i1) -> f32
      %64 = "arith.cmpf"(%58, %61) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %65 = "scf.if"(%64) ({
        %82 = "arith.negf"(%58) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%82) : (f32) -> ()
      }, {
        "scf.yield"(%58) : (f32) -> ()
      }) : (i1) -> f32
      %66 = "arith.addf"(%63, %65) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %67 = "arith.cmpf"(%60, %61) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %68 = "scf.if"(%67) ({
        %81 = "arith.negf"(%60) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%81) : (f32) -> ()
      }, {
        "scf.yield"(%60) : (f32) -> ()
      }) : (i1) -> f32
      %69 = "arith.addf"(%66, %68) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %70 = "arith.subf"(%69, %arg6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %71 = "arith.cmpf"(%70, %61) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %72 = "scf.if"(%71) ({
        %80 = "arith.negf"(%70) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%80) : (f32) -> ()
      }, {
        "scf.yield"(%70) : (f32) -> ()
      }) : (i1) -> f32
      %73 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
      %74 = "arith.cmpf"(%72, %73) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      "scf.if"(%74) ({
        %77 = "affine.load"(%arg7) <{map = #map3}> : (memref<i32>) -> i32
        %78 = "arith.constant"() <{value = 1 : i32}> : () -> i32
        %79 = "arith.addi"(%77, %78) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%79, %arg7) <{map = #map3}> : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      %75 = "affine.load"(%arg8) <{map = #map3}> : (memref<f32>) -> f32
      %76 = "arith.addf"(%75, %72) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%76, %arg8) <{map = #map3}> : (f32, memref<f32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_1__l0u"}> ({
  ^bb0(%arg0: memref<32x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg1: index):
      %0 = "arith.index_cast"(%arg1) : (index) -> i32
      %1 = "arith.constant"() <{value = 768 : i32}> : () -> i32
      %2 = "arith.cmpi"(%0, %1) <{predicate = 2 : i64}> : (i32, i32) -> i1
      "scf.if"(%2) ({
        %30 = "arith.constant"() <{value = 64 : i32}> : () -> i32
        %31 = "arith.remsi"(%0, %30) : (i32, i32) -> i32
        %32 = "arith.divsi"(%0, %30) : (i32, i32) -> i32
        %33 = "arith.sitofp"(%31) : (i32) -> f32
        %34 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
        %35 = "arith.mulf"(%33, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %36 = "arith.constant"() <{value = 6.400000e+01 : f32}> : () -> f32
        %37 = "arith.divf"(%35, %36) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %38 = "arith.constant"() <{value = 6.000000e-01 : f32}> : () -> f32
        %39 = "arith.addf"(%37, %38) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%39, %arg0, %arg1) <{map = #map}> : (f32, memref<32x3xf32>, index) -> ()
        %40 = "arith.sitofp"(%32) : (i32) -> f32
        %41 = "arith.mulf"(%40, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %42 = "arith.constant"() <{value = 1.200000e+01 : f32}> : () -> f32
        %43 = "arith.divf"(%41, %42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %44 = "arith.constant"() <{value = 4.000000e-01 : f32}> : () -> f32
        %45 = "arith.addf"(%43, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%45, %arg0, %arg1) <{map = #map1}> : (f32, memref<32x3xf32>, index) -> ()
        %46 = "arith.addi"(%31, %32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %47 = "arith.constant"() <{value = 31 : i32}> : () -> i32
        %48 = "arith.remsi"(%46, %47) : (i32, i32) -> i32
        %49 = "arith.sitofp"(%48) : (i32) -> f32
        %50 = "arith.mulf"(%49, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %51 = "arith.constant"() <{value = 3.100000e+01 : f32}> : () -> f32
        %52 = "arith.divf"(%50, %51) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %53 = "arith.constant"() <{value = 3.000000e-01 : f32}> : () -> f32
        %54 = "arith.addf"(%52, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%54, %arg0, %arg1) <{map = #map2}> : (f32, memref<32x3xf32>, index) -> ()
        "scf.yield"() : () -> ()
      }, {
        %3 = "arith.constant"() <{value = 13 : i32}> : () -> i32
        %4 = "arith.muli"(%0, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %5 = "arith.constant"() <{value = 3 : i32}> : () -> i32
        %6 = "arith.addi"(%4, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %7 = "arith.constant"() <{value = 101 : i32}> : () -> i32
        %8 = "arith.remsi"(%6, %7) : (i32, i32) -> i32
        %9 = "arith.sitofp"(%8) : (i32) -> f32
        %10 = "arith.constant"() <{value = 1.010000e+02 : f32}> : () -> f32
        %11 = "arith.divf"(%9, %10) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%11, %arg0, %arg1) <{map = #map}> : (f32, memref<32x3xf32>, index) -> ()
        %12 = "arith.constant"() <{value = 17 : i32}> : () -> i32
        %13 = "arith.muli"(%0, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %14 = "arith.constant"() <{value = 5 : i32}> : () -> i32
        %15 = "arith.addi"(%13, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %16 = "arith.constant"() <{value = 97 : i32}> : () -> i32
        %17 = "arith.remsi"(%15, %16) : (i32, i32) -> i32
        %18 = "arith.sitofp"(%17) : (i32) -> f32
        %19 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
        %20 = "arith.divf"(%18, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%20, %arg0, %arg1) <{map = #map1}> : (f32, memref<32x3xf32>, index) -> ()
        %21 = "arith.constant"() <{value = 19 : i32}> : () -> i32
        %22 = "arith.muli"(%0, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %23 = "arith.constant"() <{value = 7 : i32}> : () -> i32
        %24 = "arith.addi"(%22, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %25 = "arith.constant"() <{value = 89 : i32}> : () -> i32
        %26 = "arith.remsi"(%24, %25) : (i32, i32) -> i32
        %27 = "arith.sitofp"(%26) : (i32) -> f32
        %28 = "arith.constant"() <{value = 8.900000e+01 : f32}> : () -> f32
        %29 = "arith.divf"(%27, %28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%29, %arg0, %arg1) <{map = #map2}> : (f32, memref<32x3xf32>, index) -> ()
        "scf.yield"() : () -> ()
      }) : (i1) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
