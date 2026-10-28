#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (6)>
#map2 = affine_map<() -> (18)>
#map3 = affine_map<() -> (30)>
#map4 = affine_map<() -> (17)>
#map5 = affine_map<() -> (7)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<18x6xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_32__direct"}> ({
  ^bb0(%arg23: memref<18x6xf32>, %arg24: memref<i32>):
    %94 = "arith.constant"() <{value = 8.300000e+01 : f32}> : () -> f32
    %95 = "arith.constant"() <{value = 1.900000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg25: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg26: index):
        %96 = "arith.index_cast"(%arg26) : (index) -> i32
        %97 = "arith.index_cast"(%arg25) : (index) -> i32
        %98 = "arith.index_cast"(%arg26) : (index) -> i32
        %99 = "aps.read_smem"(%arg23, %97, %98) : (memref<18x6xf32>, i32, i32) -> f32
        %100 = "arith.sitofp"(%96) : (i32) -> f32
        %101 = "arith.mulf"(%100, %95) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %102 = "arith.addf"(%101, %94) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %103 = "arith.mulf"(%99, %102) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %104 = "arith.fptosi"(%103) : (f32) -> i32
        %105 = "memref.load"(%arg24) : (memref<i32>) -> i32
        %106 = "arith.addi"(%104, %105) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%106, %arg24) : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<30x6xf32>, memref<30x6xf32>) -> (), sym_name = "au_generated_candidate_40__l0u_l1u"}> ({
  ^bb0(%arg19: memref<30x6xf32>, %arg20: memref<30x6xf32>):
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg21: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg22: index):
        %89 = "arith.index_cast"(%arg21) : (index) -> i32
        %90 = "arith.index_cast"(%arg22) : (index) -> i32
        %91 = "aps.read_smem"(%arg19, %89, %90) : (memref<30x6xf32>, i32, i32) -> f32
        %92 = "arith.index_cast"(%arg21) : (index) -> i32
        %93 = "arith.index_cast"(%arg22) : (index) -> i32
        "aps.write_smem"(%91, %arg20, %92, %93) : (f32, memref<30x6xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [1], partition_dim_array_1 = [1], partition_factor_array_0 = [2], partition_factor_array_1 = [2], scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<18x6xf32>, memref<18x6xf32>) -> (), sym_name = "au_generated_candidate_43__l0u_l1u"}> ({
  ^bb0(%arg15: memref<18x6xf32>, %arg16: memref<18x6xf32>):
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg17: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg18: index):
        %84 = "arith.index_cast"(%arg17) : (index) -> i32
        %85 = "arith.index_cast"(%arg18) : (index) -> i32
        %86 = "aps.read_smem"(%arg15, %84, %85) : (memref<18x6xf32>, i32, i32) -> f32
        %87 = "arith.index_cast"(%arg17) : (index) -> i32
        %88 = "arith.index_cast"(%arg18) : (index) -> i32
        "aps.write_smem"(%86, %arg16, %87, %88) : (f32, memref<18x6xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [1], partition_dim_array_1 = [1], partition_factor_array_0 = [2], partition_factor_array_1 = [2], scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<18x6xf32>, memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<18x6xf32>, memref<30x6xf32>, memref<30x6xf32>) -> (), sym_name = "au_generated_candidate_5__l4p"}> ({
  ^bb0(%arg0: memref<18x6xf32>, %arg1: memref<18x6xf32>, %arg2: memref<30xi32>, %arg3: memref<30xi32>, %arg4: memref<18x6xf32>, %arg5: memref<30x6xf32>, %arg6: memref<30x6xf32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %2 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 1.000000e-03 : f32}> : () -> f32
    %6 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map5}> ({
    ^bb0(%arg7: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg13: index):
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg14: index):
          %80 = "arith.index_cast"(%arg14) : (index) -> i32
          %81 = "arith.index_cast"(%arg13) : (index) -> i32
          "aps.write_smem"(%6, %arg0, %80, %81) : (f32, memref<18x6xf32>, i32, i32) -> ()
          %82 = "arith.index_cast"(%arg14) : (index) -> i32
          %83 = "arith.index_cast"(%arg13) : (index) -> i32
          "aps.write_smem"(%5, %arg1, %82, %83) : (f32, memref<18x6xf32>, i32, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg11: index):
        %29 = "arith.index_cast"(%arg11) : (index) -> i32
        %30 = "aps.read_smem"(%arg2, %29) : (memref<30xi32>, i32) -> i32
        %31 = "arith.index_cast"(%arg11) : (index) -> i32
        %32 = "aps.read_smem"(%arg3, %31) : (memref<30xi32>, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg12: index):
          %33 = "arith.index_cast"(%32) : (i32) -> index
          %34 = "arith.index_cast"(%33) : (index) -> i32
          %35 = "arith.index_cast"(%arg12) : (index) -> i32
          %36 = "aps.read_smem"(%arg4, %34, %35) : (memref<18x6xf32>, i32, i32) -> f32
          %37 = "arith.index_cast"(%30) : (i32) -> index
          %38 = "arith.index_cast"(%37) : (index) -> i32
          %39 = "arith.index_cast"(%arg12) : (index) -> i32
          %40 = "aps.read_smem"(%arg4, %38, %39) : (memref<18x6xf32>, i32, i32) -> f32
          %41 = "arith.subf"(%36, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %42 = "arith.index_cast"(%arg11) : (index) -> i32
          %43 = "arith.index_cast"(%arg12) : (index) -> i32
          %44 = "aps.read_smem"(%arg5, %42, %43) : (memref<30x6xf32>, i32, i32) -> f32
          %45 = "arith.subf"(%41, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %46 = "arith.index_cast"(%arg11) : (index) -> i32
          %47 = "arith.index_cast"(%arg12) : (index) -> i32
          %48 = "aps.read_smem"(%arg6, %46, %47) : (memref<30x6xf32>, i32, i32) -> f32
          %49 = "arith.mulf"(%48, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %50 = "arith.index_cast"(%37) : (index) -> i32
          %51 = "arith.index_cast"(%arg12) : (index) -> i32
          %52 = "aps.read_smem"(%arg0, %50, %51) : (memref<18x6xf32>, i32, i32) -> f32
          %53 = "arith.addf"(%52, %49) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %54 = "arith.index_cast"(%37) : (index) -> i32
          %55 = "arith.index_cast"(%arg12) : (index) -> i32
          "aps.write_smem"(%53, %arg0, %54, %55) : (f32, memref<18x6xf32>, i32, i32) -> ()
          %56 = "arith.index_cast"(%33) : (index) -> i32
          %57 = "arith.index_cast"(%arg12) : (index) -> i32
          %58 = "aps.read_smem"(%arg0, %56, %57) : (memref<18x6xf32>, i32, i32) -> f32
          %59 = "arith.subf"(%58, %49) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %60 = "arith.index_cast"(%33) : (index) -> i32
          %61 = "arith.index_cast"(%arg12) : (index) -> i32
          "aps.write_smem"(%59, %arg0, %60, %61) : (f32, memref<18x6xf32>, i32, i32) -> ()
          %62 = "arith.index_cast"(%arg11) : (index) -> i32
          %63 = "arith.index_cast"(%arg12) : (index) -> i32
          %64 = "aps.read_smem"(%arg6, %62, %63) : (memref<30x6xf32>, i32, i32) -> f32
          %65 = "arith.index_cast"(%37) : (index) -> i32
          %66 = "arith.index_cast"(%arg12) : (index) -> i32
          %67 = "aps.read_smem"(%arg1, %65, %66) : (memref<18x6xf32>, i32, i32) -> f32
          %68 = "arith.addf"(%67, %64) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %69 = "arith.index_cast"(%37) : (index) -> i32
          %70 = "arith.index_cast"(%arg12) : (index) -> i32
          "aps.write_smem"(%68, %arg1, %69, %70) : (f32, memref<18x6xf32>, i32, i32) -> ()
          %71 = "arith.index_cast"(%arg11) : (index) -> i32
          %72 = "arith.index_cast"(%arg12) : (index) -> i32
          %73 = "aps.read_smem"(%arg6, %71, %72) : (memref<30x6xf32>, i32, i32) -> f32
          %74 = "arith.index_cast"(%33) : (index) -> i32
          %75 = "arith.index_cast"(%arg12) : (index) -> i32
          %76 = "aps.read_smem"(%arg1, %74, %75) : (memref<18x6xf32>, i32, i32) -> f32
          %77 = "arith.addf"(%76, %73) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %78 = "arith.index_cast"(%33) : (index) -> i32
          %79 = "arith.index_cast"(%arg12) : (index) -> i32
          "aps.write_smem"(%77, %arg1, %78, %79) : (f32, memref<18x6xf32>, i32, i32) -> ()
          "affine.yield"() : () -> ()
        }) {pipeline = 1 : i64} : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg10: index):
        %28 = "arith.index_cast"(%arg10) : (index) -> i32
        "aps.write_smem"(%6, %arg4, %0, %28) : (f32, memref<18x6xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map4}> ({
      ^bb0(%arg8: index):
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg9: index):
          %7 = "arith.index_cast"(%arg9) : (index) -> i32
          %8 = "arith.cmpi"(%7, %4) <{predicate = 2 : i64}> : (i32, i32) -> i1
          %9 = "arith.select"(%8, %3, %2) : (i1, f32, f32) -> f32
          %10 = "arith.addi"(%arg8, %1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %11 = "arith.index_cast"(%10) : (index) -> i32
          %12 = "arith.index_cast"(%arg9) : (index) -> i32
          %13 = "aps.read_smem"(%arg0, %11, %12) : (memref<18x6xf32>, i32, i32) -> f32
          %14 = "arith.mulf"(%9, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %15 = "arith.addi"(%arg8, %1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %16 = "arith.index_cast"(%15) : (index) -> i32
          %17 = "arith.index_cast"(%arg9) : (index) -> i32
          %18 = "aps.read_smem"(%arg1, %16, %17) : (memref<18x6xf32>, i32, i32) -> f32
          %19 = "arith.divf"(%14, %18) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %20 = "arith.addi"(%arg8, %1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %21 = "arith.index_cast"(%20) : (index) -> i32
          %22 = "arith.index_cast"(%arg9) : (index) -> i32
          %23 = "aps.read_smem"(%arg4, %21, %22) : (memref<18x6xf32>, i32, i32) -> f32
          %24 = "arith.subf"(%23, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %25 = "arith.addi"(%arg8, %1) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
          %26 = "arith.index_cast"(%25) : (index) -> i32
          %27 = "arith.index_cast"(%arg9) : (index) -> i32
          "aps.write_smem"(%24, %arg4, %26, %27) : (f32, memref<18x6xf32>, i32, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 5 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 7 : i32} : () -> ()
}) : () -> ()

