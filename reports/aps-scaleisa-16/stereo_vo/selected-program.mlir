"builtin.module"() ({
  "memref.global"() <{initial_value = dense<2.000000e-01> : tensor<1xf32>, sym_name = "_ZL8BASELINE", sym_visibility = "private", type = memref<1xf32>}> : () -> ()
  "memref.global"() <{initial_value = dense<1.780000e+02> : tensor<1xf32>, sym_name = "_ZL2FY", sym_visibility = "private", type = memref<1xf32>}> : () -> ()
  "memref.global"() <{initial_value = dense<1.800000e+02> : tensor<1xf32>, sym_name = "_ZL2FX", sym_visibility = "private", type = memref<1xf32>}> : () -> ()
  "memref.global"() <{initial_value = dense<4.200000e+00> : tensor<1xf32>, sym_name = "_ZL10DEPTH_BIAS", sym_visibility = "private", type = memref<1xf32>}> : () -> ()
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %44 = "arith.constant"() <{value = 10 : index}> : () -> index
    %45 = "arith.constant"() <{value = 32 : index}> : () -> index
    %46 = "arith.constant"() <{value = 2 : index}> : () -> index
    %47 = "arith.constant"() <{value = 1 : index}> : () -> index
    %48 = "arith.constant"() <{value = 0 : index}> : () -> index
    %49 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %50 = "arith.constant"() <{value = 2.110000e+02 : f32}> : () -> f32
    %51 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %52 = "arith.constant"() <{value = 2.570000e+02 : f32}> : () -> f32
    %53 = "arith.constant"() <{value = 3.070000e+02 : f32}> : () -> f32
    %54 = "arith.constant"() <{value = 1.000000e-03 : f32}> : () -> f32
    %55 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
    %56 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %57 = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32
    %58 = "arith.constant"() <{value = 3.500000e-01 : f32}> : () -> f32
    %59 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
    %60 = "arith.constant"() <{value = -4.000000e-02 : f32}> : () -> f32
    %61 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %62 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %63 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %64 = "arith.constant"() <{value = 3.000000e-02 : f32}> : () -> f32
    %65 = "arith.constant"() <{value = -2.000000e-02 : f32}> : () -> f32
    %66 = "arith.constant"() <{value = -1.000000e+00 : f32}> : () -> f32
    %67 = "arith.constant"() <{value = 4.200000e-01 : f32}> : () -> f32
    %68 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %69 = "arith.constant"() <{value = -6.000000e-01 : f32}> : () -> f32
    %70 = "arith.constant"() <{value = 3.100000e-01 : f32}> : () -> f32
    %71 = "arith.constant"() <{value = 1.600000e+00 : f32}> : () -> f32
    %72 = "arith.constant"() <{value = 1.100000e-01 : f32}> : () -> f32
    %73 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %74 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %75 = "arith.constant"() <{value = 0.00999999977 : f32}> : () -> f32
    %76 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %77 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    %78 = "arith.constant"() <{value = 1.500000e-02 : f32}> : () -> f32
    %79 = "arith.constant"() <{value = 1.200000e-02 : f32}> : () -> f32
    %80 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    %81 = "arith.constant"() <{value = 8.000000e-03 : f32}> : () -> f32
    %82 = "arith.constant"() <{value = 6.000000e-02 : f32}> : () -> f32
    %83 = "arith.constant"() <{value = 5.000000e-02 : f32}> : () -> f32
    %84 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %85 = "arith.constant"() <{value = -3 : i32}> : () -> i32
    %86 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %87 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x3xf32>
    %88 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x3xf32>
    %89 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x3xf32>
    %90 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<3xf32>
    "memref.store"(%63, %90, %48) : (f32, memref<3xf32>, index) -> ()
    "memref.store"(%64, %90, %47) : (f32, memref<3xf32>, index) -> ()
    "memref.store"(%65, %90, %46) : (f32, memref<3xf32>, index) -> ()
    "scf.for"(%48, %45, %47) ({
    ^bb0(%arg10: index):
      %303 = "arith.index_cast"(%arg10) : (index) -> i32
      %304 = "arith.remsi"(%303, %68) : (i32, i32) -> i32
      %305 = "arith.sitofp"(%304) : (i32) -> f32
      %306 = "arith.mulf"(%305, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %307 = "arith.addf"(%306, %66) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%307, %89, %arg10, %48) : (f32, memref<32x3xf32>, index, index) -> ()
      %308 = "arith.divsi"(%303, %68) : (i32, i32) -> i32
      %309 = "arith.sitofp"(%308) : (i32) -> f32
      %310 = "arith.mulf"(%309, %70) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %311 = "arith.addf"(%310, %69) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%311, %89, %arg10, %47) : (f32, memref<32x3xf32>, index, index) -> ()
      %312 = "arith.remsi"(%303, %73) : (i32, i32) -> i32
      %313 = "arith.sitofp"(%312) : (i32) -> f32
      %314 = "arith.mulf"(%313, %72) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %315 = "arith.addf"(%314, %71) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%315, %89, %arg10, %46) : (f32, memref<32x3xf32>, index, index) -> ()
      %316 = "memref.load"(%89, %arg10, %48) : (memref<32x3xf32>, index, index) -> f32
      %317 = "memref.load"(%89, %arg10, %47) : (memref<32x3xf32>, index, index) -> f32
      %318 = "memref.load"(%89, %arg10, %46) : (memref<32x3xf32>, index, index) -> f32
      %319 = "memref.get_global"() <{name = @_ZL10DEPTH_BIAS}> : () -> memref<1xf32>
      %320 = "memref.load"(%319, %48) : (memref<1xf32>, index) -> f32
      %321 = "arith.addf"(%318, %320) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %322 = "arith.divf"(%57, %321) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %323 = "memref.get_global"() <{name = @_ZL2FX}> : () -> memref<1xf32>
      %324 = "memref.load"(%323, %48) : (memref<1xf32>, index) -> f32
      %325 = "arith.mulf"(%324, %316) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %326 = "arith.mulf"(%325, %322) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%326, %88, %arg10, %48) : (f32, memref<32x3xf32>, index, index) -> ()
      %327 = "memref.get_global"() <{name = @_ZL2FY}> : () -> memref<1xf32>
      %328 = "memref.load"(%327, %48) : (memref<1xf32>, index) -> f32
      %329 = "arith.mulf"(%328, %317) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %330 = "arith.mulf"(%329, %322) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%330, %88, %arg10, %47) : (f32, memref<32x3xf32>, index, index) -> ()
      %331 = "memref.load"(%323, %48) : (memref<1xf32>, index) -> f32
      %332 = "memref.get_global"() <{name = @_ZL8BASELINE}> : () -> memref<1xf32>
      %333 = "memref.load"(%332, %48) : (memref<1xf32>, index) -> f32
      %334 = "arith.mulf"(%331, %333) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %335 = "arith.mulf"(%334, %322) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%335, %88, %arg10, %46) : (f32, memref<32x3xf32>, index, index) -> ()
      %336 = "memref.load"(%89, %arg10, %48) : (memref<32x3xf32>, index, index) -> f32
      %337 = "arith.subf"(%336, %58) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %338 = "memref.load"(%89, %arg10, %47) : (memref<32x3xf32>, index, index) -> f32
      %339 = "arith.subf"(%338, %60) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %340 = "memref.load"(%89, %arg10, %46) : (memref<32x3xf32>, index, index) -> f32
      %341 = "arith.subf"(%340, %62) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %342 = "memref.load"(%319, %48) : (memref<1xf32>, index) -> f32
      %343 = "arith.addf"(%341, %342) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %344 = "arith.divf"(%57, %343) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %345 = "memref.load"(%323, %48) : (memref<1xf32>, index) -> f32
      %346 = "arith.mulf"(%345, %337) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %347 = "arith.mulf"(%346, %344) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%347, %87, %arg10, %48) : (f32, memref<32x3xf32>, index, index) -> ()
      %348 = "memref.load"(%327, %48) : (memref<1xf32>, index) -> f32
      %349 = "arith.mulf"(%348, %339) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %350 = "arith.mulf"(%349, %344) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%350, %87, %arg10, %47) : (f32, memref<32x3xf32>, index, index) -> ()
      %351 = "memref.load"(%323, %48) : (memref<1xf32>, index) -> f32
      %352 = "memref.load"(%332, %48) : (memref<1xf32>, index) -> f32
      %353 = "arith.mulf"(%351, %352) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %354 = "arith.mulf"(%353, %344) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%354, %87, %arg10, %46) : (f32, memref<32x3xf32>, index, index) -> ()
      %355 = "arith.remsi"(%303, %76) : (i32, i32) -> i32
      %356 = "arith.addi"(%355, %84) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %357 = "arith.sitofp"(%356) : (i32) -> f32
      %358 = "arith.mulf"(%357, %75) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %359 = "memref.load"(%88, %arg10, %48) : (memref<32x3xf32>, index, index) -> f32
      %360 = "arith.addf"(%359, %358) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%360, %88, %arg10, %48) : (f32, memref<32x3xf32>, index, index) -> ()
      %361 = "arith.remsi"(%303, %77) : (i32, i32) -> i32
      %362 = "arith.addi"(%361, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %363 = "arith.sitofp"(%362) : (i32) -> f32
      %364 = "arith.mulf"(%363, %75) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %365 = "memref.load"(%88, %arg10, %47) : (memref<32x3xf32>, index, index) -> f32
      %366 = "arith.subf"(%365, %364) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%366, %88, %arg10, %47) : (f32, memref<32x3xf32>, index, index) -> ()
      %367 = "arith.addi"(%312, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %368 = "arith.sitofp"(%367) : (i32) -> f32
      %369 = "arith.mulf"(%368, %78) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %370 = "memref.load"(%87, %arg10, %48) : (memref<32x3xf32>, index, index) -> f32
      %371 = "arith.addf"(%370, %369) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%371, %87, %arg10, %48) : (f32, memref<32x3xf32>, index, index) -> ()
      %372 = "arith.remsi"(%303, %80) : (i32, i32) -> i32
      %373 = "arith.addi"(%372, %85) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %374 = "arith.sitofp"(%373) : (i32) -> f32
      %375 = "arith.mulf"(%374, %79) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %376 = "memref.load"(%87, %arg10, %47) : (memref<32x3xf32>, index, index) -> f32
      %377 = "arith.subf"(%376, %375) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%377, %87, %arg10, %47) : (f32, memref<32x3xf32>, index, index) -> ()
      %378 = "arith.addi"(%361, %84) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %379 = "arith.sitofp"(%378) : (i32) -> f32
      %380 = "arith.mulf"(%379, %81) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %381 = "memref.load"(%87, %arg10, %46) : (memref<32x3xf32>, index, index) -> f32
      %382 = "arith.addf"(%381, %380) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%382, %87, %arg10, %46) : (f32, memref<32x3xf32>, index, index) -> ()
      %383 = "arith.remsi"(%303, %61) : (i32, i32) -> i32
      %384 = "arith.sitofp"(%383) : (i32) -> f32
      %385 = "arith.mulf"(%384, %82) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %386 = "memref.load"(%89, %arg10, %48) : (memref<32x3xf32>, index, index) -> f32
      %387 = "arith.addf"(%386, %385) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%387, %89, %arg10, %48) : (f32, memref<32x3xf32>, index, index) -> ()
      %388 = "arith.sitofp"(%355) : (i32) -> f32
      %389 = "arith.mulf"(%388, %83) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %390 = "memref.load"(%89, %arg10, %47) : (memref<32x3xf32>, index, index) -> f32
      %391 = "arith.subf"(%390, %389) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%391, %89, %arg10, %47) : (f32, memref<32x3xf32>, index, index) -> ()
      %392 = "arith.mulf"(%379, %59) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %393 = "memref.load"(%89, %arg10, %46) : (memref<32x3xf32>, index, index) -> f32
      %394 = "arith.addf"(%393, %392) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%394, %89, %arg10, %46) : (f32, memref<32x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %91 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<3xf32>
    %92 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<3xf32>
    %93 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x3xf32>
    %94 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<32x3xf32>
    %95 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<3xf32>
    %96 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<3xf32>
    %97 = "memref.get_global"() <{name = @_ZL10DEPTH_BIAS}> : () -> memref<1xf32>
    %98 = "memref.get_global"() <{name = @_ZL2FX}> : () -> memref<1xf32>
    %99 = "memref.get_global"() <{name = @_ZL2FY}> : () -> memref<1xf32>
    %100 = "memref.get_global"() <{name = @_ZL8BASELINE}> : () -> memref<1xf32>
    "scf.for"(%48, %44, %47) ({
    ^bb0(%arg7: index):
      "memref.store"(%74, %96, %48) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%74, %96, %47) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%74, %96, %46) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%54, %95, %48) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%54, %95, %47) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%54, %95, %46) : (f32, memref<3xf32>, index) -> ()
      "scf.for"(%48, %45, %47) ({
      ^bb0(%arg9: index):
        "memref.store"(%74, %94, %arg9, %48) : (f32, memref<32x3xf32>, index, index) -> ()
        "memref.store"(%74, %94, %arg9, %47) : (f32, memref<32x3xf32>, index, index) -> ()
        "memref.store"(%74, %94, %arg9, %46) : (f32, memref<32x3xf32>, index, index) -> ()
        "memref.store"(%54, %93, %arg9, %48) : (f32, memref<32x3xf32>, index, index) -> ()
        "memref.store"(%54, %93, %arg9, %47) : (f32, memref<32x3xf32>, index, index) -> ()
        "memref.store"(%54, %93, %arg9, %46) : (f32, memref<32x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.for"(%48, %45, %47) ({
      ^bb0(%arg8: index):
        %138 = "memref.load"(%89, %arg8, %48) : (memref<32x3xf32>, index, index) -> f32
        %139 = "memref.load"(%89, %arg8, %47) : (memref<32x3xf32>, index, index) -> f32
        %140 = "memref.load"(%89, %arg8, %46) : (memref<32x3xf32>, index, index) -> f32
        %141 = "memref.load"(%97, %48) : (memref<1xf32>, index) -> f32
        %142 = "arith.addf"(%140, %141) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %143 = "arith.divf"(%57, %142) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %144 = "memref.load"(%98, %48) : (memref<1xf32>, index) -> f32
        %145 = "arith.mulf"(%144, %138) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %146 = "arith.mulf"(%145, %143) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%146, %92, %48) : (f32, memref<3xf32>, index) -> ()
        %147 = "memref.load"(%99, %48) : (memref<1xf32>, index) -> f32
        %148 = "arith.mulf"(%147, %139) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %149 = "arith.mulf"(%148, %143) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%149, %92, %47) : (f32, memref<3xf32>, index) -> ()
        %150 = "memref.load"(%98, %48) : (memref<1xf32>, index) -> f32
        %151 = "memref.load"(%100, %48) : (memref<1xf32>, index) -> f32
        %152 = "arith.mulf"(%150, %151) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %153 = "arith.mulf"(%152, %143) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%153, %92, %46) : (f32, memref<3xf32>, index) -> ()
        %154 = "memref.load"(%90, %48) : (memref<3xf32>, index) -> f32
        %155 = "memref.load"(%90, %47) : (memref<3xf32>, index) -> f32
        %156 = "memref.load"(%90, %46) : (memref<3xf32>, index) -> f32
        %157 = "memref.load"(%89, %arg8, %48) : (memref<32x3xf32>, index, index) -> f32
        %158 = "arith.subf"(%157, %154) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %159 = "memref.load"(%89, %arg8, %47) : (memref<32x3xf32>, index, index) -> f32
        %160 = "arith.subf"(%159, %155) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %161 = "memref.load"(%89, %arg8, %46) : (memref<32x3xf32>, index, index) -> f32
        %162 = "arith.subf"(%161, %156) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %163 = "memref.load"(%97, %48) : (memref<1xf32>, index) -> f32
        %164 = "arith.addf"(%162, %163) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %165 = "arith.divf"(%57, %164) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %166 = "memref.load"(%98, %48) : (memref<1xf32>, index) -> f32
        %167 = "arith.mulf"(%166, %158) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %168 = "arith.mulf"(%167, %165) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%168, %91, %48) : (f32, memref<3xf32>, index) -> ()
        %169 = "memref.load"(%99, %48) : (memref<1xf32>, index) -> f32
        %170 = "arith.mulf"(%169, %160) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %171 = "arith.mulf"(%170, %165) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%171, %91, %47) : (f32, memref<3xf32>, index) -> ()
        %172 = "memref.load"(%98, %48) : (memref<1xf32>, index) -> f32
        %173 = "memref.load"(%100, %48) : (memref<1xf32>, index) -> f32
        %174 = "arith.mulf"(%172, %173) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %175 = "arith.mulf"(%174, %165) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%175, %91, %46) : (f32, memref<3xf32>, index) -> ()
        %176 = "memref.load"(%89, %arg8, %46) : (memref<32x3xf32>, index, index) -> f32
        %177 = "memref.load"(%97, %48) : (memref<1xf32>, index) -> f32
        %178 = "arith.addf"(%176, %177) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %179 = "arith.divf"(%57, %178) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %180 = "memref.load"(%92, %48) : (memref<3xf32>, index) -> f32
        %181 = "memref.load"(%88, %arg8, %48) : (memref<32x3xf32>, index, index) -> f32
        %182 = "arith.subf"(%180, %181) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %183 = "memref.load"(%92, %47) : (memref<3xf32>, index) -> f32
        %184 = "memref.load"(%88, %arg8, %47) : (memref<32x3xf32>, index, index) -> f32
        %185 = "arith.subf"(%183, %184) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %186 = "memref.load"(%92, %46) : (memref<3xf32>, index) -> f32
        %187 = "memref.load"(%88, %arg8, %46) : (memref<32x3xf32>, index, index) -> f32
        %188 = "arith.subf"(%186, %187) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %189 = "memref.load"(%98, %48) : (memref<1xf32>, index) -> f32
        %190 = "arith.mulf"(%189, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %191 = "memref.load"(%99, %48) : (memref<1xf32>, index) -> f32
        %192 = "arith.mulf"(%191, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %193 = "arith.negf"(%189) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %194 = "memref.load"(%89, %arg8, %48) : (memref<32x3xf32>, index, index) -> f32
        %195 = "arith.mulf"(%193, %194) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %196 = "arith.mulf"(%195, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %197 = "arith.mulf"(%196, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %198 = "arith.negf"(%191) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %199 = "memref.load"(%89, %arg8, %47) : (memref<32x3xf32>, index, index) -> f32
        %200 = "arith.mulf"(%198, %199) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %201 = "arith.mulf"(%200, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %202 = "arith.mulf"(%201, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %203 = "memref.load"(%100, %48) : (memref<1xf32>, index) -> f32
        %204 = "arith.mulf"(%193, %203) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %205 = "arith.mulf"(%204, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %206 = "arith.mulf"(%205, %179) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %207 = "memref.load"(%90, %48) : (memref<3xf32>, index) -> f32
        %208 = "arith.subf"(%194, %207) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %209 = "memref.load"(%90, %47) : (memref<3xf32>, index) -> f32
        %210 = "arith.subf"(%199, %209) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %211 = "memref.load"(%90, %46) : (memref<3xf32>, index) -> f32
        %212 = "arith.subf"(%176, %211) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %213 = "arith.addf"(%212, %177) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %214 = "arith.divf"(%57, %213) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %215 = "memref.load"(%91, %48) : (memref<3xf32>, index) -> f32
        %216 = "memref.load"(%87, %arg8, %48) : (memref<32x3xf32>, index, index) -> f32
        %217 = "arith.subf"(%215, %216) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %218 = "memref.load"(%91, %47) : (memref<3xf32>, index) -> f32
        %219 = "memref.load"(%87, %arg8, %47) : (memref<32x3xf32>, index, index) -> f32
        %220 = "arith.subf"(%218, %219) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %221 = "memref.load"(%91, %46) : (memref<3xf32>, index) -> f32
        %222 = "memref.load"(%87, %arg8, %46) : (memref<32x3xf32>, index, index) -> f32
        %223 = "arith.subf"(%221, %222) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %224 = "arith.mulf"(%189, %214) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %225 = "arith.mulf"(%191, %214) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %226 = "arith.mulf"(%193, %208) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %227 = "arith.mulf"(%226, %214) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %228 = "arith.mulf"(%227, %214) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %229 = "arith.mulf"(%198, %210) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %230 = "arith.mulf"(%229, %214) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %231 = "arith.mulf"(%230, %214) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %232 = "arith.mulf"(%204, %214) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %233 = "arith.mulf"(%232, %214) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %234 = "arith.negf"(%217) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %235 = "arith.mulf"(%234, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %236 = "memref.load"(%96, %48) : (memref<3xf32>, index) -> f32
        %237 = "arith.addf"(%236, %235) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%237, %96, %48) : (f32, memref<3xf32>, index) -> ()
        %238 = "arith.negf"(%220) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %239 = "arith.mulf"(%238, %225) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %240 = "memref.load"(%96, %47) : (memref<3xf32>, index) -> f32
        %241 = "arith.addf"(%240, %239) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%241, %96, %47) : (f32, memref<3xf32>, index) -> ()
        %242 = "arith.mulf"(%217, %228) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %243 = "arith.mulf"(%220, %231) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %244 = "arith.addf"(%242, %243) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %245 = "arith.mulf"(%223, %233) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %246 = "arith.addf"(%244, %245) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %247 = "arith.negf"(%246) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        %248 = "memref.load"(%96, %46) : (memref<3xf32>, index) -> f32
        %249 = "arith.addf"(%248, %247) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%249, %96, %46) : (f32, memref<3xf32>, index) -> ()
        %250 = "arith.mulf"(%224, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %251 = "memref.load"(%95, %48) : (memref<3xf32>, index) -> f32
        %252 = "arith.addf"(%251, %250) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%252, %95, %48) : (f32, memref<3xf32>, index) -> ()
        %253 = "arith.mulf"(%225, %225) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %254 = "memref.load"(%95, %47) : (memref<3xf32>, index) -> f32
        %255 = "arith.addf"(%254, %253) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%255, %95, %47) : (f32, memref<3xf32>, index) -> ()
        %256 = "arith.mulf"(%228, %228) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %257 = "arith.mulf"(%231, %231) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %258 = "arith.addf"(%256, %257) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %259 = "arith.mulf"(%233, %233) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %260 = "arith.addf"(%258, %259) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %261 = "memref.load"(%95, %46) : (memref<3xf32>, index) -> f32
        %262 = "arith.addf"(%261, %260) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%262, %95, %46) : (f32, memref<3xf32>, index) -> ()
        %263 = "arith.mulf"(%182, %190) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %264 = "memref.load"(%94, %arg8, %48) : (memref<32x3xf32>, index, index) -> f32
        %265 = "arith.addf"(%264, %263) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%265, %94, %arg8, %48) : (f32, memref<32x3xf32>, index, index) -> ()
        %266 = "arith.mulf"(%185, %192) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %267 = "memref.load"(%94, %arg8, %47) : (memref<32x3xf32>, index, index) -> f32
        %268 = "arith.addf"(%267, %266) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%268, %94, %arg8, %47) : (f32, memref<32x3xf32>, index, index) -> ()
        %269 = "arith.mulf"(%182, %197) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %270 = "arith.mulf"(%185, %202) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %271 = "arith.addf"(%269, %270) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %272 = "arith.mulf"(%188, %206) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %273 = "arith.addf"(%271, %272) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %274 = "memref.load"(%94, %arg8, %46) : (memref<32x3xf32>, index, index) -> f32
        %275 = "arith.addf"(%274, %273) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%275, %94, %arg8, %46) : (f32, memref<32x3xf32>, index, index) -> ()
        %276 = "arith.mulf"(%190, %190) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %277 = "memref.load"(%93, %arg8, %48) : (memref<32x3xf32>, index, index) -> f32
        %278 = "arith.addf"(%277, %276) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%278, %93, %arg8, %48) : (f32, memref<32x3xf32>, index, index) -> ()
        %279 = "arith.mulf"(%192, %192) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %280 = "memref.load"(%93, %arg8, %47) : (memref<32x3xf32>, index, index) -> f32
        %281 = "arith.addf"(%280, %279) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%281, %93, %arg8, %47) : (f32, memref<32x3xf32>, index, index) -> ()
        %282 = "arith.mulf"(%197, %197) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %283 = "arith.mulf"(%202, %202) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %284 = "arith.addf"(%282, %283) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %285 = "arith.mulf"(%206, %206) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %286 = "arith.addf"(%284, %285) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %287 = "memref.load"(%93, %arg8, %46) : (memref<32x3xf32>, index, index) -> f32
        %288 = "arith.addf"(%287, %286) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%288, %93, %arg8, %46) : (f32, memref<32x3xf32>, index, index) -> ()
        %289 = "arith.mulf"(%217, %224) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %290 = "memref.load"(%94, %arg8, %48) : (memref<32x3xf32>, index, index) -> f32
        %291 = "arith.addf"(%290, %289) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%291, %94, %arg8, %48) : (f32, memref<32x3xf32>, index, index) -> ()
        %292 = "arith.mulf"(%220, %225) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %293 = "memref.load"(%94, %arg8, %47) : (memref<32x3xf32>, index, index) -> f32
        %294 = "arith.addf"(%293, %292) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%294, %94, %arg8, %47) : (f32, memref<32x3xf32>, index, index) -> ()
        %295 = "memref.load"(%94, %arg8, %46) : (memref<32x3xf32>, index, index) -> f32
        %296 = "arith.addf"(%295, %246) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%296, %94, %arg8, %46) : (f32, memref<32x3xf32>, index, index) -> ()
        %297 = "memref.load"(%93, %arg8, %48) : (memref<32x3xf32>, index, index) -> f32
        %298 = "arith.addf"(%297, %250) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%298, %93, %arg8, %48) : (f32, memref<32x3xf32>, index, index) -> ()
        %299 = "memref.load"(%93, %arg8, %47) : (memref<32x3xf32>, index, index) -> f32
        %300 = "arith.addf"(%299, %253) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%300, %93, %arg8, %47) : (f32, memref<32x3xf32>, index, index) -> ()
        %301 = "memref.load"(%93, %arg8, %46) : (memref<32x3xf32>, index, index) -> f32
        %302 = "arith.addf"(%301, %260) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "memref.store"(%302, %93, %arg8, %46) : (f32, memref<32x3xf32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %120 = "memref.load"(%96, %48) : (memref<3xf32>, index) -> f32
      %121 = "arith.mulf"(%120, %55) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %122 = "memref.load"(%95, %48) : (memref<3xf32>, index) -> f32
      %123 = "arith.divf"(%121, %122) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %124 = "memref.load"(%90, %48) : (memref<3xf32>, index) -> f32
      %125 = "arith.subf"(%124, %123) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%125, %90, %48) : (f32, memref<3xf32>, index) -> ()
      %126 = "memref.load"(%96, %47) : (memref<3xf32>, index) -> f32
      %127 = "arith.mulf"(%126, %55) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %128 = "memref.load"(%95, %47) : (memref<3xf32>, index) -> f32
      %129 = "arith.divf"(%127, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %130 = "memref.load"(%90, %47) : (memref<3xf32>, index) -> f32
      %131 = "arith.subf"(%130, %129) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%131, %90, %47) : (f32, memref<3xf32>, index) -> ()
      %132 = "memref.load"(%96, %46) : (memref<3xf32>, index) -> f32
      %133 = "arith.mulf"(%132, %56) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %134 = "memref.load"(%95, %46) : (memref<3xf32>, index) -> f32
      %135 = "arith.divf"(%133, %134) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %136 = "memref.load"(%90, %46) : (memref<3xf32>, index) -> f32
      %137 = "arith.subf"(%136, %135) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%137, %90, %46) : (f32, memref<3xf32>, index) -> ()
      "func.call"(%94, %93, %89) <{callee = @au_generated_candidate_17__direct_site_23551}> : (memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %101 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%49, %101) : (i32, memref<i32>) -> ()
    %102 = "memref.load"(%90, %48) : (memref<3xf32>, index) -> f32
    %103 = "arith.mulf"(%102, %50) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %104 = "arith.fptosi"(%103) : (f32) -> i32
    %105 = "memref.load"(%90, %47) : (memref<3xf32>, index) -> f32
    %106 = "arith.mulf"(%105, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %107 = "arith.fptosi"(%106) : (f32) -> i32
    %108 = "arith.addi"(%104, %107) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %109 = "memref.load"(%90, %46) : (memref<3xf32>, index) -> f32
    %110 = "arith.mulf"(%109, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %111 = "arith.fptosi"(%110) : (f32) -> i32
    %112 = "arith.addi"(%108, %111) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    "memref.store"(%112, %101) : (i32, memref<i32>) -> ()
    "func.call"(%89, %101) <{callee = @au_generated_candidate_26__direct_site_23590}> : (memref<32x3xf32>, memref<i32>) -> ()
    %113 = "memref.load"(%101) : (memref<i32>) -> i32
    %114 = "arith.sitofp"(%113) : (i32) -> f32
    %115 = "arith.cmpi"(%113, %49) <{predicate = 2 : i64}> : (i32, i32) -> i1
    %116 = "scf.if"(%115) ({
      %119 = "arith.negf"(%114) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
      "scf.yield"(%119) : (f32) -> ()
    }, {
      "scf.yield"(%114) : (f32) -> ()
    }) : (i1) -> f32
    %117 = "arith.cmpf"(%116, %74) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
    %118 = "arith.select"(%117, %113, %51) : (i1, i32, i32) -> i32
    "func.call"(%90, %89, %88, %87) <{callee = @pcl_validate_outputs}> : (memref<3xf32>, memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>) -> ()
    "func.return"(%118) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_17__direct_site_23551"}> ({
  ^bb0(%arg3: memref<32x3xf32>, %arg4: memref<32x3xf32>, %arg5: memref<32x3xf32>):
    %20 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %21 = "arith.constant"() <{value = 2 : index}> : () -> index
    %22 = "arith.constant"() <{value = 1.600000e-01 : f32}> : () -> f32
    %23 = "arith.constant"() <{value = 0 : index}> : () -> index
    %24 = "arith.constant"() <{value = 32 : index}> : () -> index
    %25 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%23, %24, %25) ({
    ^bb0(%arg6: index):
      %26 = "memref.load"(%arg3, %arg6, %23) : (memref<32x3xf32>, index, index) -> f32
      %27 = "arith.mulf"(%26, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %28 = "memref.load"(%arg4, %arg6, %23) : (memref<32x3xf32>, index, index) -> f32
      %29 = "arith.divf"(%27, %28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %30 = "memref.load"(%arg5, %arg6, %23) : (memref<32x3xf32>, index, index) -> f32
      %31 = "arith.subf"(%30, %29) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%31, %arg5, %arg6, %23) : (f32, memref<32x3xf32>, index, index) -> ()
      %32 = "memref.load"(%arg3, %arg6, %25) : (memref<32x3xf32>, index, index) -> f32
      %33 = "arith.mulf"(%32, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %34 = "memref.load"(%arg4, %arg6, %25) : (memref<32x3xf32>, index, index) -> f32
      %35 = "arith.divf"(%33, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %36 = "memref.load"(%arg5, %arg6, %25) : (memref<32x3xf32>, index, index) -> f32
      %37 = "arith.subf"(%36, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%37, %arg5, %arg6, %25) : (f32, memref<32x3xf32>, index, index) -> ()
      %38 = "memref.load"(%arg3, %arg6, %21) : (memref<32x3xf32>, index, index) -> f32
      %39 = "arith.mulf"(%38, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %40 = "memref.load"(%arg4, %arg6, %21) : (memref<32x3xf32>, index, index) -> f32
      %41 = "arith.divf"(%39, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %42 = "memref.load"(%arg5, %arg6, %21) : (memref<32x3xf32>, index, index) -> f32
      %43 = "arith.subf"(%42, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%43, %arg5, %arg6, %21) : (f32, memref<32x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_26__direct_site_23590"}> ({
  ^bb0(%arg0: memref<32x3xf32>, %arg1: memref<i32>):
    %0 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 2 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1.030000e+02 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = 7.900000e+01 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 0 : index}> : () -> index
    %5 = "arith.constant"() <{value = 32 : index}> : () -> index
    %6 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%4, %5, %6) ({
    ^bb0(%arg2: index):
      %7 = "memref.load"(%arg0, %arg2, %4) : (memref<32x3xf32>, index, index) -> f32
      %8 = "arith.mulf"(%7, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %9 = "arith.fptosi"(%8) : (f32) -> i32
      %10 = "memref.load"(%arg1) : (memref<i32>) -> i32
      %11 = "arith.addi"(%10, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%11, %arg1) : (i32, memref<i32>) -> ()
      %12 = "memref.load"(%arg0, %arg2, %6) : (memref<32x3xf32>, index, index) -> f32
      %13 = "arith.mulf"(%12, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %14 = "arith.fptosi"(%13) : (f32) -> i32
      %15 = "arith.addi"(%11, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%15, %arg1) : (i32, memref<i32>) -> ()
      %16 = "memref.load"(%arg0, %arg2, %1) : (memref<32x3xf32>, index, index) -> f32
      %17 = "arith.mulf"(%16, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %18 = "arith.fptosi"(%17) : (f32) -> i32
      %19 = "arith.addi"(%15, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%19, %arg1) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<3xf32>, memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
