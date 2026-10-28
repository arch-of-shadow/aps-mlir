#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<(d0) -> (d0, 1)>
#map2 = affine_map<(d0) -> (d0, 2)>
#map3 = affine_map<() -> (0)>
#map4 = affine_map<() -> (32)>
#map5 = affine_map<() -> ()>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_17__direct"}> ({
  ^bb0(%arg4: memref<32x3xf32>, %arg5: memref<32x3xf32>, %arg6: memref<32x3xf32>):
    "affine.for"() <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg7: index):
      %16 = "affine.load"(%arg4, %arg7) <{map = #map}> : (memref<32x3xf32>, index) -> f32
      %17 = "arith.constant"() <{value = 1.600000e-01 : f32}> : () -> f32
      %18 = "arith.mulf"(%16, %17) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %19 = "affine.load"(%arg5, %arg7) <{map = #map}> : (memref<32x3xf32>, index) -> f32
      %20 = "arith.divf"(%18, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %21 = "affine.load"(%arg6, %arg7) <{map = #map}> : (memref<32x3xf32>, index) -> f32
      %22 = "arith.subf"(%21, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%22, %arg6, %arg7) <{map = #map}> : (f32, memref<32x3xf32>, index) -> ()
      %23 = "affine.load"(%arg4, %arg7) <{map = #map1}> : (memref<32x3xf32>, index) -> f32
      %24 = "arith.mulf"(%23, %17) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %25 = "affine.load"(%arg5, %arg7) <{map = #map1}> : (memref<32x3xf32>, index) -> f32
      %26 = "arith.divf"(%24, %25) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %27 = "affine.load"(%arg6, %arg7) <{map = #map1}> : (memref<32x3xf32>, index) -> f32
      %28 = "arith.subf"(%27, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%28, %arg6, %arg7) <{map = #map1}> : (f32, memref<32x3xf32>, index) -> ()
      %29 = "affine.load"(%arg4, %arg7) <{map = #map2}> : (memref<32x3xf32>, index) -> f32
      %30 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
      %31 = "arith.mulf"(%29, %30) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %32 = "affine.load"(%arg5, %arg7) <{map = #map2}> : (memref<32x3xf32>, index) -> f32
      %33 = "arith.divf"(%31, %32) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %34 = "affine.load"(%arg6, %arg7) <{map = #map2}> : (memref<32x3xf32>, index) -> f32
      %35 = "arith.subf"(%34, %33) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "affine.store"(%35, %arg6, %arg7) <{map = #map2}> : (f32, memref<32x3xf32>, index) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<32x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_26__direct"}> ({
  ^bb0(%arg0: memref<32x3xf32>, %arg1: memref<i32>):
    "affine.for"() <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 0, 0>, step = 32 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg2: index):
      "affine.for"() <{lowerBoundMap = #map3, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
      ^bb0(%arg3: index):
        %0 = "affine.load"(%arg0, %arg3) <{map = #map}> : (memref<32x3xf32>, index) -> f32
        %1 = "arith.constant"() <{value = 7.900000e+01 : f32}> : () -> f32
        %2 = "arith.mulf"(%0, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %3 = "arith.fptosi"(%2) : (f32) -> i32
        %4 = "affine.load"(%arg1) <{map = #map5}> : (memref<i32>) -> i32
        %5 = "arith.addi"(%4, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%5, %arg1) <{map = #map5}> : (i32, memref<i32>) -> ()
        %6 = "affine.load"(%arg0, %arg3) <{map = #map1}> : (memref<32x3xf32>, index) -> f32
        %7 = "arith.constant"() <{value = 1.030000e+02 : f32}> : () -> f32
        %8 = "arith.mulf"(%6, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %9 = "arith.fptosi"(%8) : (f32) -> i32
        %10 = "arith.addi"(%5, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%10, %arg1) <{map = #map5}> : (i32, memref<i32>) -> ()
        %11 = "affine.load"(%arg0, %arg3) <{map = #map2}> : (memref<32x3xf32>, index) -> f32
        %12 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
        %13 = "arith.mulf"(%11, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %14 = "arith.fptosi"(%13) : (f32) -> i32
        %15 = "arith.addi"(%10, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.store"(%15, %arg1) <{map = #map5}> : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
}) : () -> ()
