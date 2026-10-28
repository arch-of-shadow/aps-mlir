"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<64x6xf32>, memref<3xf32>, memref<3xf32>, memref<64xf32>) -> (), sym_name = "au_generated_candidate_0__l0p_u0x2"}> ({
  ^bb0(%arg0: memref<64x6xf32>, %arg1: memref<3xf32>, %arg2: memref<3xf32>, %arg3: memref<64xf32>):
    %0 = "arith.constant"() <{value = 5 : index}> : () -> index
    %1 = "arith.constant"() <{value = 4 : index}> : () -> index
    %2 = "arith.constant"() <{value = 3 : index}> : () -> index
    %3 = "arith.constant"() <{value = 2 : index}> : () -> index
    %4 = "arith.constant"() <{value = 0 : index}> : () -> index
    %5 = "arith.constant"() <{value = 64 : index}> : () -> index
    %6 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%4, %5, %6) ({
    ^bb0(%arg4: index):
      %7 = "arith.index_cast"(%arg4) : (index) -> i32
      %8 = "arith.index_cast"(%4) : (index) -> i32
      %9 = "aps.read_smem"(%arg0, %7, %8) : (memref<64x6xf32>, i32, i32) -> f32
      %10 = "arith.index_cast"(%arg4) : (index) -> i32
      %11 = "arith.index_cast"(%6) : (index) -> i32
      %12 = "aps.read_smem"(%arg0, %10, %11) : (memref<64x6xf32>, i32, i32) -> f32
      %13 = "arith.index_cast"(%arg4) : (index) -> i32
      %14 = "arith.index_cast"(%3) : (index) -> i32
      %15 = "aps.read_smem"(%arg0, %13, %14) : (memref<64x6xf32>, i32, i32) -> f32
      %16 = "arith.index_cast"(%arg4) : (index) -> i32
      %17 = "arith.index_cast"(%2) : (index) -> i32
      %18 = "aps.read_smem"(%arg0, %16, %17) : (memref<64x6xf32>, i32, i32) -> f32
      %19 = "arith.index_cast"(%arg4) : (index) -> i32
      %20 = "arith.index_cast"(%1) : (index) -> i32
      %21 = "aps.read_smem"(%arg0, %19, %20) : (memref<64x6xf32>, i32, i32) -> f32
      %22 = "arith.index_cast"(%arg4) : (index) -> i32
      %23 = "arith.index_cast"(%0) : (index) -> i32
      %24 = "aps.read_smem"(%arg0, %22, %23) : (memref<64x6xf32>, i32, i32) -> f32
      %25 = "arith.index_cast"(%4) : (index) -> i32
      "aps.write_smem"(%9, %arg1, %25) : (f32, memref<3xf32>, i32) -> ()
      %26 = "arith.index_cast"(%6) : (index) -> i32
      "aps.write_smem"(%12, %arg1, %26) : (f32, memref<3xf32>, i32) -> ()
      %27 = "arith.index_cast"(%3) : (index) -> i32
      "aps.write_smem"(%15, %arg1, %27) : (f32, memref<3xf32>, i32) -> ()
      %28 = "arith.index_cast"(%4) : (index) -> i32
      "aps.write_smem"(%18, %arg2, %28) : (f32, memref<3xf32>, i32) -> ()
      %29 = "arith.index_cast"(%6) : (index) -> i32
      "aps.write_smem"(%21, %arg2, %29) : (f32, memref<3xf32>, i32) -> ()
      %30 = "arith.index_cast"(%3) : (index) -> i32
      "aps.write_smem"(%24, %arg2, %30) : (f32, memref<3xf32>, i32) -> ()
      %31 = "arith.index_cast"(%4) : (index) -> i32
      %32 = "aps.read_smem"(%arg1, %31) : (memref<3xf32>, i32) -> f32
      %33 = "arith.index_cast"(%4) : (index) -> i32
      %34 = "aps.read_smem"(%arg2, %33) : (memref<3xf32>, i32) -> f32
      %35 = "arith.subf"(%32, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %36 = "arith.index_cast"(%6) : (index) -> i32
      %37 = "aps.read_smem"(%arg1, %36) : (memref<3xf32>, i32) -> f32
      %38 = "arith.index_cast"(%6) : (index) -> i32
      %39 = "aps.read_smem"(%arg2, %38) : (memref<3xf32>, i32) -> f32
      %40 = "arith.subf"(%37, %39) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %41 = "arith.index_cast"(%3) : (index) -> i32
      %42 = "aps.read_smem"(%arg1, %41) : (memref<3xf32>, i32) -> f32
      %43 = "arith.index_cast"(%3) : (index) -> i32
      %44 = "aps.read_smem"(%arg2, %43) : (memref<3xf32>, i32) -> f32
      %45 = "arith.subf"(%42, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %46 = "arith.mulf"(%35, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %47 = "arith.mulf"(%40, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %48 = "arith.addf"(%46, %47) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %49 = "arith.mulf"(%45, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %50 = "arith.addf"(%48, %49) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %51 = "arith.index_cast"(%arg4) : (index) -> i32
      "aps.write_smem"(%50, %arg3, %51) : (f32, memref<64xf32>, i32) -> ()
      "scf.yield"() : () -> ()
    }) {pipeline = 1 : i64, unroll = 2 : i64} : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_3 = [true], partition_dim_array_0 = [1], partition_dim_array_3 = [0], partition_factor_array_0 = [2], partition_factor_array_3 = [2], scaleisa.converted, scaleisa.csr_count = 2 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 4 : i32} : () -> ()
}) : () -> ()

