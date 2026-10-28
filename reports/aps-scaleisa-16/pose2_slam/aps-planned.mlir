#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (24)>
#map2 = affine_map<() -> (23)>
#map3 = affine_map<() -> (3)>
#map4 = affine_map<() -> (40)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_21__direct"}> ({
  ^bb0(%arg22: memref<24x3xf32>, %arg23: memref<i32>):
    %81 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %82 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %83 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %84 = "arith.constant"() <{value = 2.570000e+02 : f32}> : () -> f32
    %85 = "arith.constant"() <{value = 1.730000e+02 : f32}> : () -> f32
    %86 = "arith.constant"() <{value = 1.010000e+02 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg24: index):
      %87 = "arith.index_cast"(%arg24) : (index) -> i32
      %88 = "aps.read_smem"(%arg22, %87, %83) : (memref<24x3xf32>, i32, i32) -> f32
      %89 = "arith.mulf"(%88, %86) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %90 = "arith.fptosi"(%89) : (f32) -> i32
      %91 = "memref.load"(%arg23) : (memref<i32>) -> i32
      %92 = "arith.addi"(%91, %90) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%92, %arg23) : (i32, memref<i32>) -> ()
      %93 = "arith.index_cast"(%arg24) : (index) -> i32
      %94 = "aps.read_smem"(%arg22, %93, %82) : (memref<24x3xf32>, i32, i32) -> f32
      %95 = "arith.mulf"(%94, %85) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %96 = "arith.fptosi"(%95) : (f32) -> i32
      %97 = "arith.addi"(%92, %96) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%97, %arg23) : (i32, memref<i32>) -> ()
      %98 = "arith.index_cast"(%arg24) : (index) -> i32
      %99 = "aps.read_smem"(%arg22, %98, %81) : (memref<24x3xf32>, i32, i32) -> f32
      %100 = "arith.mulf"(%99, %84) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %101 = "arith.fptosi"(%100) : (f32) -> i32
      %102 = "arith.andi"(%97, %101) : (i32, i32) -> i32
      %103 = "arith.ori"(%97, %101) : (i32, i32) -> i32
      %104 = "arith.addi"(%102, %103) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%104, %arg23) : (i32, memref<i32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_31__l1p"}> ({
  ^bb0(%arg17: memref<24x3xf32>, %arg18: memref<24x3xf32>, %arg19: memref<24x3xf32>):
    %61 = "arith.constant"() <{value = 1 : index}> : () -> index
    %62 = "arith.constant"() <{value = 2.800000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg20: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg21: index):
        %63 = "arith.addi"(%arg21, %61) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %64 = "arith.index_cast"(%63) : (index) -> i32
        %65 = "arith.index_cast"(%arg20) : (index) -> i32
        %66 = "aps.read_smem"(%arg17, %64, %65) : (memref<24x3xf32>, i32, i32) -> f32
        %67 = "arith.mulf"(%66, %62) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %68 = "arith.addi"(%arg21, %61) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %69 = "arith.index_cast"(%68) : (index) -> i32
        %70 = "arith.index_cast"(%arg20) : (index) -> i32
        %71 = "aps.read_smem"(%arg18, %69, %70) : (memref<24x3xf32>, i32, i32) -> f32
        %72 = "arith.divf"(%67, %71) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %73 = "arith.addi"(%arg21, %61) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %74 = "arith.index_cast"(%73) : (index) -> i32
        %75 = "arith.index_cast"(%arg20) : (index) -> i32
        %76 = "aps.read_smem"(%arg19, %74, %75) : (memref<24x3xf32>, i32, i32) -> f32
        %77 = "arith.subf"(%76, %72) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %78 = "arith.addi"(%arg21, %61) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %79 = "arith.index_cast"(%78) : (index) -> i32
        %80 = "arith.index_cast"(%arg20) : (index) -> i32
        "aps.write_smem"(%77, %arg19, %79, %80) : (f32, memref<24x3xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {pipeline = 1 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<40x3xf32>, memref<40x3xf32>) -> (), sym_name = "au_generated_candidate_36__l0u_l1u"}> ({
  ^bb0(%arg13: memref<40x3xf32>, %arg14: memref<40x3xf32>):
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg15: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg16: index):
        %56 = "arith.index_cast"(%arg15) : (index) -> i32
        %57 = "arith.index_cast"(%arg16) : (index) -> i32
        %58 = "aps.read_smem"(%arg13, %56, %57) : (memref<40x3xf32>, i32, i32) -> f32
        %59 = "arith.index_cast"(%arg15) : (index) -> i32
        %60 = "arith.index_cast"(%arg16) : (index) -> i32
        "aps.write_smem"(%58, %arg14, %59, %60) : (f32, memref<40x3xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_38__l0u_l1u"}> ({
  ^bb0(%arg9: memref<24x3xf32>, %arg10: memref<24x3xf32>):
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg11: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg12: index):
        %51 = "arith.index_cast"(%arg11) : (index) -> i32
        %52 = "arith.index_cast"(%arg12) : (index) -> i32
        %53 = "aps.read_smem"(%arg9, %51, %52) : (memref<24x3xf32>, i32, i32) -> f32
        %54 = "arith.index_cast"(%arg11) : (index) -> i32
        %55 = "arith.index_cast"(%arg12) : (index) -> i32
        "aps.write_smem"(%53, %arg10, %54, %55) : (f32, memref<24x3xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<40xi32>, memref<40xi32>, memref<24x3xf32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_5__l0p_l1u"}> ({
  ^bb0(%arg0: memref<40xi32>, %arg1: memref<40xi32>, %arg2: memref<24x3xf32>, %arg3: memref<40x3xf32>, %arg4: memref<40x3xf32>, %arg5: memref<24x3xf32>, %arg6: memref<24x3xf32>):
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
    ^bb0(%arg7: index):
      %0 = "arith.index_cast"(%arg7) : (index) -> i32
      %1 = "aps.read_smem"(%arg0, %0) : (memref<40xi32>, i32) -> i32
      %2 = "arith.index_cast"(%arg7) : (index) -> i32
      %3 = "aps.read_smem"(%arg1, %2) : (memref<40xi32>, i32) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg8: index):
        %4 = "arith.index_cast"(%3) : (i32) -> index
        %5 = "arith.index_cast"(%4) : (index) -> i32
        %6 = "arith.index_cast"(%arg8) : (index) -> i32
        %7 = "aps.read_smem"(%arg2, %5, %6) : (memref<24x3xf32>, i32, i32) -> f32
        %8 = "arith.index_cast"(%1) : (i32) -> index
        %9 = "arith.index_cast"(%8) : (index) -> i32
        %10 = "arith.index_cast"(%arg8) : (index) -> i32
        %11 = "aps.read_smem"(%arg2, %9, %10) : (memref<24x3xf32>, i32, i32) -> f32
        %12 = "arith.subf"(%7, %11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %13 = "arith.index_cast"(%arg7) : (index) -> i32
        %14 = "arith.index_cast"(%arg8) : (index) -> i32
        %15 = "aps.read_smem"(%arg3, %13, %14) : (memref<40x3xf32>, i32, i32) -> f32
        %16 = "arith.subf"(%12, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %17 = "arith.index_cast"(%arg7) : (index) -> i32
        %18 = "arith.index_cast"(%arg8) : (index) -> i32
        %19 = "aps.read_smem"(%arg4, %17, %18) : (memref<40x3xf32>, i32, i32) -> f32
        %20 = "arith.mulf"(%19, %16) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %21 = "arith.index_cast"(%8) : (index) -> i32
        %22 = "arith.index_cast"(%arg8) : (index) -> i32
        %23 = "aps.read_smem"(%arg5, %21, %22) : (memref<24x3xf32>, i32, i32) -> f32
        %24 = "arith.addf"(%23, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %25 = "arith.index_cast"(%8) : (index) -> i32
        %26 = "arith.index_cast"(%arg8) : (index) -> i32
        "aps.write_smem"(%24, %arg5, %25, %26) : (f32, memref<24x3xf32>, i32, i32) -> ()
        %27 = "arith.index_cast"(%4) : (index) -> i32
        %28 = "arith.index_cast"(%arg8) : (index) -> i32
        %29 = "aps.read_smem"(%arg5, %27, %28) : (memref<24x3xf32>, i32, i32) -> f32
        %30 = "arith.subf"(%29, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %31 = "arith.index_cast"(%4) : (index) -> i32
        %32 = "arith.index_cast"(%arg8) : (index) -> i32
        "aps.write_smem"(%30, %arg5, %31, %32) : (f32, memref<24x3xf32>, i32, i32) -> ()
        %33 = "arith.index_cast"(%arg7) : (index) -> i32
        %34 = "arith.index_cast"(%arg8) : (index) -> i32
        %35 = "aps.read_smem"(%arg4, %33, %34) : (memref<40x3xf32>, i32, i32) -> f32
        %36 = "arith.index_cast"(%8) : (index) -> i32
        %37 = "arith.index_cast"(%arg8) : (index) -> i32
        %38 = "aps.read_smem"(%arg6, %36, %37) : (memref<24x3xf32>, i32, i32) -> f32
        %39 = "arith.addf"(%38, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %40 = "arith.index_cast"(%8) : (index) -> i32
        %41 = "arith.index_cast"(%arg8) : (index) -> i32
        "aps.write_smem"(%39, %arg6, %40, %41) : (f32, memref<24x3xf32>, i32, i32) -> ()
        %42 = "arith.index_cast"(%arg7) : (index) -> i32
        %43 = "arith.index_cast"(%arg8) : (index) -> i32
        %44 = "aps.read_smem"(%arg4, %42, %43) : (memref<40x3xf32>, i32, i32) -> f32
        %45 = "arith.index_cast"(%4) : (index) -> i32
        %46 = "arith.index_cast"(%arg8) : (index) -> i32
        %47 = "aps.read_smem"(%arg6, %45, %46) : (memref<24x3xf32>, i32, i32) -> f32
        %48 = "arith.addf"(%47, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %49 = "arith.index_cast"(%4) : (index) -> i32
        %50 = "arith.index_cast"(%arg8) : (index) -> i32
        "aps.write_smem"(%48, %arg6, %49, %50) : (f32, memref<24x3xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {pipeline = 1 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [0], partition_dim_array_1 = [0], partition_factor_array_0 = [2], partition_factor_array_1 = [2], scaleisa.converted, scaleisa.csr_count = 5 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 7 : i32} : () -> ()
}) : () -> ()

