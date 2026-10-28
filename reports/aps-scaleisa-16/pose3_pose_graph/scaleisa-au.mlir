#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (6)>
#map4 = affine_map<() -> (18)>
#map5 = affine_map<() -> (30)>
#map6 = affine_map<(d0) -> (d0)>
#map7 = affine_map<(d0) -> (0, d0)>
#map8 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map9 = affine_map<() -> (17)>
#map10 = affine_map<() -> (7)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<18x6xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_32__direct"}> ({
  ^bb0(%arg23: memref<18x6xf32>, %arg24: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg25: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg26: index):
        %38 = "arith.index_cast"(%arg26) : (index) -> i32
        %39 = "affine.load"(%arg23, %arg25, %arg26) <{map = #map}> : (memref<18x6xf32>, index, index) -> f32
        %40 = "arith.sitofp"(%38) : (i32) -> f32
        %41 = "arith.constant"() <{value = 1.900000e+01 : f32}> : () -> f32
        %42 = "arith.mulf"(%40, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %43 = "arith.constant"() <{value = 8.300000e+01 : f32}> : () -> f32
        %44 = "arith.addf"(%42, %43) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %45 = "arith.mulf"(%39, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %46 = "arith.fptosi"(%45) : (f32) -> i32
        %47 = "affine.load"(%arg24) <{map = #map1}> : (memref<i32>) -> i32
        %48 = "arith.addi"(%46, %47) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%48, %arg24) <{map = #map1}> : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<30x6xf32>, memref<30x6xf32>) -> (), sym_name = "au_generated_candidate_40__l0u_l1u"}> ({
  ^bb0(%arg19: memref<30x6xf32>, %arg20: memref<30x6xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg21: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg22: index):
        %37 = "affine.load"(%arg19, %arg21, %arg22) <{map = #map}> : (memref<30x6xf32>, index, index) -> f32
        "affine.store"(%37, %arg20, %arg21, %arg22) <{map = #map}> : (f32, memref<30x6xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [1], partition_dim_array_1 = [1], partition_factor_array_0 = [2], partition_factor_array_1 = [2]} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<18x6xf32>, memref<18x6xf32>) -> (), sym_name = "au_generated_candidate_43__l0u_l1u"}> ({
  ^bb0(%arg15: memref<18x6xf32>, %arg16: memref<18x6xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg17: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg18: index):
        %36 = "affine.load"(%arg15, %arg17, %arg18) <{map = #map}> : (memref<18x6xf32>, index, index) -> f32
        "affine.store"(%36, %arg16, %arg17, %arg18) <{map = #map}> : (f32, memref<18x6xf32>, index, index) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [1], partition_dim_array_1 = [1], partition_factor_array_0 = [2], partition_factor_array_1 = [2]} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<18x6xf32>, memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<18x6xf32>, memref<30x6xf32>, memref<30x6xf32>) -> (), sym_name = "au_generated_candidate_5__l4p"}> ({
  ^bb0(%arg0: memref<18x6xf32>, %arg1: memref<18x6xf32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<18x6xf32>, %arg5: memref<30x6xf32>, %arg6: memref<30x6xf32>):
    "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map10}> ({
    ^bb0(%arg7: index):
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg13: index):
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
        ^bb0(%arg14: index):
          %34 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
          "affine.store"(%34, %arg0, %arg14, %arg13) <{map = #map}> : (f32, memref<18x6xf32>, index, index) -> ()
          %35 = "arith.constant"() <{value = 1.000000e-03 : f32}> : () -> f32
          "affine.store"(%35, %arg1, %arg14, %arg13) <{map = #map}> : (f32, memref<18x6xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
      ^bb0(%arg11: index):
        %13 = "affine.load"(%arg2, %arg11) <{map = #map6}> : (memref<30xi32>, index) -> i32
        %14 = "affine.load"(%arg3, %arg11) <{map = #map6}> : (memref<30xi32>, index) -> i32
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg12: index):
          %15 = "arith.index_cast"(%14) : (i32) -> index
          %16 = "memref.load"(%arg4, %15, %arg12) : (memref<18x6xf32>, index, index) -> f32
          %17 = "arith.index_cast"(%13) : (i32) -> index
          %18 = "memref.load"(%arg4, %17, %arg12) : (memref<18x6xf32>, index, index) -> f32
          %19 = "arith.subf"(%16, %18) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %20 = "affine.load"(%arg5, %arg11, %arg12) <{map = #map}> : (memref<30x6xf32>, index, index) -> f32
          %21 = "arith.subf"(%19, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %22 = "affine.load"(%arg6, %arg11, %arg12) <{map = #map}> : (memref<30x6xf32>, index, index) -> f32
          %23 = "arith.mulf"(%22, %21) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %24 = "memref.load"(%arg0, %17, %arg12) : (memref<18x6xf32>, index, index) -> f32
          %25 = "arith.addf"(%24, %23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%25, %arg0, %17, %arg12) : (f32, memref<18x6xf32>, index, index) -> ()
          %26 = "memref.load"(%arg0, %15, %arg12) : (memref<18x6xf32>, index, index) -> f32
          %27 = "arith.subf"(%26, %23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%27, %arg0, %15, %arg12) : (f32, memref<18x6xf32>, index, index) -> ()
          %28 = "affine.load"(%arg6, %arg11, %arg12) <{map = #map}> : (memref<30x6xf32>, index, index) -> f32
          %29 = "memref.load"(%arg1, %17, %arg12) : (memref<18x6xf32>, index, index) -> f32
          %30 = "arith.addf"(%29, %28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%30, %arg1, %17, %arg12) : (f32, memref<18x6xf32>, index, index) -> ()
          %31 = "affine.load"(%arg6, %arg11, %arg12) <{map = #map}> : (memref<30x6xf32>, index, index) -> f32
          %32 = "memref.load"(%arg1, %15, %arg12) : (memref<18x6xf32>, index, index) -> f32
          %33 = "arith.addf"(%32, %31) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%33, %arg1, %15, %arg12) : (f32, memref<18x6xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) {pipeline = 0 : i64} : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg10: index):
        %12 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
        "affine.store"(%12, %arg4, %arg10) <{map = #map7}> : (f32, memref<18x6xf32>, index) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map9}> ({
      ^bb0(%arg8: index):
        "affine.for"() <{lowerBoundMap = #map2, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg9: index):
          %0 = "arith.index_cast"(%arg9) : (index) -> i32
          %1 = "arith.constant"() <{value = 3 : i32}> : () -> i32
          %2 = "arith.cmpi"(%0, %1) <{predicate = 2 : i64}> : (i32, i32) -> i1
          %3 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
          %4 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
          %5 = "arith.select"(%2, %3, %4) : (i1, f32, f32) -> f32
          %6 = "affine.load"(%arg0, %arg8, %arg9) <{map = #map8}> : (memref<18x6xf32>, index, index) -> f32
          %7 = "arith.mulf"(%5, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %8 = "affine.load"(%arg1, %arg8, %arg9) <{map = #map8}> : (memref<18x6xf32>, index, index) -> f32
          %9 = "arith.divf"(%7, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %10 = "affine.load"(%arg4, %arg8, %arg9) <{map = #map8}> : (memref<18x6xf32>, index, index) -> f32
          %11 = "arith.subf"(%10, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "affine.store"(%11, %arg4, %arg8, %arg9) <{map = #map8}> : (f32, memref<18x6xf32>, index, index) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
