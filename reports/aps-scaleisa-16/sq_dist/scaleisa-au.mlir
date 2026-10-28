"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<64x6xf32>, memref<3xf32>, memref<3xf32>, memref<64xf32>) -> (), sym_name = "au_generated_candidate_0__l0p_u0x2"}> ({
  ^bb0(%arg0: memref<64x6xf32>, %arg1: memref<3xf32>, %arg2: memref<3xf32>, %arg3: memref<64xf32>):
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 64 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%0, %1, %2) ({
    ^bb0(%arg4: index):
      %3 = "memref.load"(%arg0, %arg4, %0) : (memref<64x6xf32>, index, index) -> f32
      %4 = "memref.load"(%arg0, %arg4, %2) : (memref<64x6xf32>, index, index) -> f32
      %5 = "arith.constant"() <{value = 2 : index}> : () -> index
      %6 = "memref.load"(%arg0, %arg4, %5) : (memref<64x6xf32>, index, index) -> f32
      %7 = "arith.constant"() <{value = 3 : index}> : () -> index
      %8 = "memref.load"(%arg0, %arg4, %7) : (memref<64x6xf32>, index, index) -> f32
      %9 = "arith.constant"() <{value = 4 : index}> : () -> index
      %10 = "memref.load"(%arg0, %arg4, %9) : (memref<64x6xf32>, index, index) -> f32
      %11 = "arith.constant"() <{value = 5 : index}> : () -> index
      %12 = "memref.load"(%arg0, %arg4, %11) : (memref<64x6xf32>, index, index) -> f32
      "memref.store"(%3, %arg1, %0) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%4, %arg1, %2) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%6, %arg1, %5) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%8, %arg2, %0) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%10, %arg2, %2) : (f32, memref<3xf32>, index) -> ()
      "memref.store"(%12, %arg2, %5) : (f32, memref<3xf32>, index) -> ()
      %13 = "memref.load"(%arg1, %0) : (memref<3xf32>, index) -> f32
      %14 = "memref.load"(%arg2, %0) : (memref<3xf32>, index) -> f32
      %15 = "arith.subf"(%13, %14) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %16 = "memref.load"(%arg1, %2) : (memref<3xf32>, index) -> f32
      %17 = "memref.load"(%arg2, %2) : (memref<3xf32>, index) -> f32
      %18 = "arith.subf"(%16, %17) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %19 = "memref.load"(%arg1, %5) : (memref<3xf32>, index) -> f32
      %20 = "memref.load"(%arg2, %5) : (memref<3xf32>, index) -> f32
      %21 = "arith.subf"(%19, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %22 = "arith.mulf"(%15, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %23 = "arith.mulf"(%18, %18) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %24 = "arith.addf"(%22, %23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %25 = "arith.mulf"(%21, %21) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %26 = "arith.addf"(%24, %25) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%26, %arg3, %arg4) : (f32, memref<64xf32>, index) -> ()
      "scf.yield"() : () -> ()
    }) {pipeline = 0 : i64, unroll = 2 : i64} : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_3 = [true], partition_dim_array_0 = [1], partition_dim_array_3 = [0], partition_factor_array_0 = [2], partition_factor_array_3 = [2]} : () -> ()
}) : () -> ()
