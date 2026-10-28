#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<(d0) -> (d0, 1)>
#map3 = affine_map<(d0) -> (d0, 2)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<() -> (24)>
#map6 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map7 = affine_map<() -> (23)>
#map8 = affine_map<() -> (3)>
#map9 = affine_map<(d0, d1) -> (d0, d1)>
#map10 = affine_map<() -> (40)>
#map11 = affine_map<(d0) -> (d0)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_21__direct"}> ({
  ^bb0(%arg22: memref<24x3xf32>, %arg23: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg24: index):
      %30 = "affine.load"(%arg22, %arg24) <{map = #map}> : (memref<24x3xf32>, index) -> f32
      %31 = "arith.constant"() <{value = 1.010000e+02 : f32}> : () -> f32
      %32 = "arith.mulf"(%30, %31) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %33 = "arith.fptosi"(%32) : (f32) -> i32
      %34 = "affine.load"(%arg23) <{map = #map1}> : (memref<i32>) -> i32
      %35 = "arith.addi"(%34, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%35, %arg23) <{map = #map1}> : (i32, memref<i32>) -> ()
      %36 = "affine.load"(%arg22, %arg24) <{map = #map2}> : (memref<24x3xf32>, index) -> f32
      %37 = "arith.constant"() <{value = 1.730000e+02 : f32}> : () -> f32
      %38 = "arith.mulf"(%36, %37) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %39 = "arith.fptosi"(%38) : (f32) -> i32
      %40 = "arith.addi"(%35, %39) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%40, %arg23) <{map = #map1}> : (i32, memref<i32>) -> ()
      %41 = "affine.load"(%arg22, %arg24) <{map = #map3}> : (memref<24x3xf32>, index) -> f32
      %42 = "arith.constant"() <{value = 2.570000e+02 : f32}> : () -> f32
      %43 = "arith.mulf"(%41, %42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %44 = "arith.fptosi"(%43) : (f32) -> i32
      %45 = "arith.andi"(%40, %44) : (i32, i32) -> i32
      %46 = "arith.ori"(%40, %44) : (i32, i32) -> i32
      %47 = "arith.addi"(%45, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.store"(%47, %arg23) <{map = #map1}> : (i32, memref<i32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_31__l1p"}> ({
  ^bb0(%arg17: memref<24x3xf32>, %arg18: memref<24x3xf32>, %arg19: memref<24x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map8}> ({
    ^bb0(%arg20: index):
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map7}> ({
      ^bb0(%arg21: index):
        %23 = "affine.load"(%arg17, %arg21, %arg20) <{map = #map6}> : (memref<24x3xf32>, index, index) -> f32
        %24 = "arith.constant"() <{value = 2.800000e-01 : f32}> : () -> f32
        %25 = "arith.mulf"(%23, %24) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %26 = "affine.load"(%arg18, %arg21, %arg20) <{map = #map6}> : (memref<24x3xf32>, index, index) -> f32
        %27 = "arith.divf"(%25, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %28 = "affine.load"(%arg19, %arg21, %arg20) <{map = #map6}> : (memref<24x3xf32>, index, index) -> f32
        %29 = "arith.subf"(%28, %27) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "affine.store"(%29, %arg19, %arg21, %arg20) <{map = #map6}> : (f32, memref<24x3xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {pipeline = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<40x3xf32>, memref<40x3xf32>) -> (), sym_name = "au_generated_candidate_36__l0u_l1u"}> ({
  ^bb0(%arg13: memref<40x3xf32>, %arg14: memref<40x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map10}> ({
    ^bb0(%arg15: index):
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map8}> ({
      ^bb0(%arg16: index):
        %22 = "affine.load"(%arg13, %arg15, %arg16) <{map = #map9}> : (memref<40x3xf32>, index, index) -> f32
        "affine.store"(%22, %arg14, %arg15, %arg16) <{map = #map9}> : (f32, memref<40x3xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_38__l0u_l1u"}> ({
  ^bb0(%arg9: memref<24x3xf32>, %arg10: memref<24x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg11: index):
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map8}> ({
      ^bb0(%arg12: index):
        %21 = "affine.load"(%arg9, %arg11, %arg12) <{map = #map9}> : (memref<24x3xf32>, index, index) -> f32
        "affine.store"(%21, %arg10, %arg11, %arg12) <{map = #map9}> : (f32, memref<24x3xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<40xi32>, memref<40xi32>, memref<24x3xf32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_5__l0p_l1u"}> ({
  ^bb0(%arg0: memref<40xi32>, %arg1: memref<40xi32>, %arg2: memref<24x3xf32>, %arg3: memref<40x3xf32>, %arg4: memref<40x3xf32>, %arg5: memref<24x3xf32>, %arg6: memref<24x3xf32>):
    "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map10}> ({
    ^bb0(%arg7: index):
      %0 = "affine.load"(%arg0, %arg7) <{map = #map11}> : (memref<40xi32>, index) -> i32
      %1 = "affine.load"(%arg1, %arg7) <{map = #map11}> : (memref<40xi32>, index) -> i32
      "affine.for"() <{lowerBoundMap = #map4, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map8}> ({
      ^bb0(%arg8: index):
        %2 = "arith.index_cast"(%1) : (i32) -> index
        %3 = "memref.load"(%arg2, %2, %arg8) : (memref<24x3xf32>, index, index) -> f32
        %4 = "arith.index_cast"(%0) : (i32) -> index
        %5 = "memref.load"(%arg2, %4, %arg8) : (memref<24x3xf32>, index, index) -> f32
        %6 = "arith.subf"(%3, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %7 = "affine.load"(%arg3, %arg7, %arg8) <{map = #map9}> : (memref<40x3xf32>, index, index) -> f32
        %8 = "arith.subf"(%6, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %9 = "affine.load"(%arg4, %arg7, %arg8) <{map = #map9}> : (memref<40x3xf32>, index, index) -> f32
        %10 = "arith.mulf"(%9, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %11 = "memref.load"(%arg5, %4, %arg8) : (memref<24x3xf32>, index, index) -> f32
        %12 = "arith.addf"(%11, %10) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%12, %arg5, %4, %arg8) : (f32, memref<24x3xf32>, index, index) -> ()
        %13 = "memref.load"(%arg5, %2, %arg8) : (memref<24x3xf32>, index, index) -> f32
        %14 = "arith.subf"(%13, %10) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%14, %arg5, %2, %arg8) : (f32, memref<24x3xf32>, index, index) -> ()
        %15 = "affine.load"(%arg4, %arg7, %arg8) <{map = #map9}> : (memref<40x3xf32>, index, index) -> f32
        %16 = "memref.load"(%arg6, %4, %arg8) : (memref<24x3xf32>, index, index) -> f32
        %17 = "arith.addf"(%16, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%17, %arg6, %4, %arg8) : (f32, memref<24x3xf32>, index, index) -> ()
        %18 = "affine.load"(%arg4, %arg7, %arg8) <{map = #map9}> : (memref<40x3xf32>, index, index) -> f32
        %19 = "memref.load"(%arg6, %2, %arg8) : (memref<24x3xf32>, index, index) -> f32
        %20 = "arith.addf"(%19, %18) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%20, %arg6, %2, %arg8) : (f32, memref<24x3xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {pipeline = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [0], partition_dim_array_1 = [0], partition_factor_array_0 = [2], partition_factor_array_1 = [2]} : () -> ()
}) : () -> ()
