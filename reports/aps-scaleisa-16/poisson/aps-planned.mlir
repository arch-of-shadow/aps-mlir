#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<() -> (4)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<216xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_11__direct"}> ({
  ^bb0(%arg11: memref<216xf32>, %arg12: memref<i32>):
    %80 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %81 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %82 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg13: index):
      %83 = "affine.apply"(%arg13) <{map = #map}> : (index) -> index
      %84 = "arith.index_cast"(%83) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg14: index):
        %85 = "affine.apply"(%arg14) <{map = #map}> : (index) -> index
        %86 = "arith.index_cast"(%85) : (index) -> i32
        "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg15: index):
          %87 = "affine.apply"(%arg15) <{map = #map}> : (index) -> index
          %88 = "arith.index_cast"(%87) : (index) -> i32
          %89 = "arith.muli"(%84, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %90 = "arith.addi"(%89, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %91 = "arith.muli"(%90, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %92 = "arith.addi"(%91, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %93 = "arith.addi"(%84, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %94 = "arith.muli"(%93, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %95 = "arith.addi"(%94, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %96 = "arith.muli"(%95, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %97 = "arith.addi"(%96, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %98 = "arith.addi"(%86, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %99 = "arith.addi"(%89, %98) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %100 = "arith.muli"(%99, %82) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %101 = "arith.addi"(%100, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %102 = "arith.addi"(%88, %81) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %103 = "arith.addi"(%91, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %104 = "arith.index_cast"(%92) : (i32) -> index
          %105 = "arith.index_cast"(%104) : (index) -> i32
          %106 = "aps.read_smem"(%arg11, %105) : (memref<216xf32>, i32) -> f32
          %107 = "arith.index_cast"(%97) : (i32) -> index
          %108 = "arith.index_cast"(%107) : (index) -> i32
          %109 = "aps.read_smem"(%arg11, %108) : (memref<216xf32>, i32) -> f32
          %110 = "arith.mulf"(%106, %109) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %111 = "arith.cmpf"(%110, %80) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %112 = "arith.extui"(%111) : (i1) -> i32
          %113 = "memref.load"(%arg12) : (memref<i32>) -> i32
          %114 = "arith.addi"(%113, %112) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "memref.store"(%114, %arg12) : (i32, memref<i32>) -> ()
          %115 = "arith.index_cast"(%101) : (i32) -> index
          %116 = "arith.index_cast"(%115) : (index) -> i32
          %117 = "aps.read_smem"(%arg11, %116) : (memref<216xf32>, i32) -> f32
          %118 = "arith.mulf"(%106, %117) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %119 = "arith.cmpf"(%118, %80) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %120 = "arith.extui"(%119) : (i1) -> i32
          %121 = "arith.addi"(%114, %120) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "memref.store"(%121, %arg12) : (i32, memref<i32>) -> ()
          %122 = "arith.index_cast"(%103) : (i32) -> index
          %123 = "arith.index_cast"(%122) : (index) -> i32
          %124 = "aps.read_smem"(%arg11, %123) : (memref<216xf32>, i32) -> f32
          %125 = "arith.mulf"(%106, %124) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %126 = "arith.cmpf"(%125, %80) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %127 = "arith.extui"(%126) : (i1) -> i32
          %128 = "arith.addi"(%121, %127) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "memref.store"(%128, %arg12) : (i32, memref<i32>) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<216xf32>, memref<216xf32>) -> (), sym_name = "au_generated_candidate_27__l0u_l1u_l2u"}> ({
  ^bb0(%arg6: memref<216xf32>, %arg7: memref<216xf32>):
    %65 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg8: index):
      %66 = "affine.apply"(%arg8) <{map = #map}> : (index) -> index
      %67 = "arith.index_cast"(%66) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg9: index):
        %68 = "affine.apply"(%arg9) <{map = #map}> : (index) -> index
        %69 = "arith.index_cast"(%68) : (index) -> i32
        "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg10: index):
          %70 = "affine.apply"(%arg10) <{map = #map}> : (index) -> index
          %71 = "arith.index_cast"(%70) : (index) -> i32
          %72 = "arith.muli"(%67, %65) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %73 = "arith.addi"(%72, %69) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %74 = "arith.muli"(%73, %65) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %75 = "arith.addi"(%74, %71) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %76 = "arith.index_cast"(%75) : (i32) -> index
          %77 = "arith.index_cast"(%76) : (index) -> i32
          %78 = "aps.read_smem"(%arg6, %77) : (memref<216xf32>, i32) -> f32
          %79 = "arith.index_cast"(%76) : (index) -> i32
          "aps.write_smem"(%78, %arg7, %79) : (f32, memref<216xf32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) {unroll = 0 : i64} : () -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [0], partition_dim_array_1 = [0], partition_factor_array_0 = [4], partition_factor_array_1 = [4], scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<216xf32>, memref<216xf32>, memref<216xf32>) -> (), sym_name = "au_generated_candidate_9__direct"}> ({
  ^bb0(%arg0: memref<216xf32>, %arg1: memref<216xf32>, %arg2: memref<216xf32>):
    %0 = "arith.constant"() <{value = 6.000000e+00 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg3: index):
      %4 = "affine.apply"(%arg3) <{map = #map}> : (index) -> index
      %5 = "arith.index_cast"(%4) : (index) -> i32
      %6 = "arith.muli"(%5, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %7 = "arith.addi"(%5, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %8 = "arith.muli"(%7, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %9 = "arith.addi"(%5, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %10 = "arith.muli"(%9, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg4: index):
        %11 = "affine.apply"(%arg4) <{map = #map}> : (index) -> index
        %12 = "arith.index_cast"(%11) : (index) -> i32
        "affine.for"() <{lowerBoundMap = #map1, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg5: index):
          %13 = "affine.apply"(%arg5) <{map = #map}> : (index) -> index
          %14 = "arith.index_cast"(%13) : (index) -> i32
          %15 = "arith.addi"(%6, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %16 = "arith.muli"(%15, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %17 = "arith.addi"(%16, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %18 = "arith.addi"(%8, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %19 = "arith.muli"(%18, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %20 = "arith.addi"(%19, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %21 = "arith.addi"(%10, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %22 = "arith.muli"(%21, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %23 = "arith.addi"(%22, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %24 = "arith.addi"(%12, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %25 = "arith.addi"(%6, %24) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %26 = "arith.muli"(%25, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %27 = "arith.addi"(%26, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %28 = "arith.addi"(%12, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %29 = "arith.addi"(%6, %28) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %30 = "arith.muli"(%29, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %31 = "arith.addi"(%30, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %32 = "arith.addi"(%14, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %33 = "arith.addi"(%16, %32) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %34 = "arith.addi"(%14, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %35 = "arith.addi"(%16, %34) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %36 = "arith.index_cast"(%20) : (i32) -> index
          %37 = "arith.index_cast"(%36) : (index) -> i32
          %38 = "aps.read_smem"(%arg0, %37) : (memref<216xf32>, i32) -> f32
          %39 = "arith.index_cast"(%23) : (i32) -> index
          %40 = "arith.index_cast"(%39) : (index) -> i32
          %41 = "aps.read_smem"(%arg0, %40) : (memref<216xf32>, i32) -> f32
          %42 = "arith.addf"(%38, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %43 = "arith.index_cast"(%27) : (i32) -> index
          %44 = "arith.index_cast"(%43) : (index) -> i32
          %45 = "aps.read_smem"(%arg0, %44) : (memref<216xf32>, i32) -> f32
          %46 = "arith.addf"(%42, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %47 = "arith.index_cast"(%31) : (i32) -> index
          %48 = "arith.index_cast"(%47) : (index) -> i32
          %49 = "aps.read_smem"(%arg0, %48) : (memref<216xf32>, i32) -> f32
          %50 = "arith.addf"(%46, %49) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %51 = "arith.index_cast"(%33) : (i32) -> index
          %52 = "arith.index_cast"(%51) : (index) -> i32
          %53 = "aps.read_smem"(%arg0, %52) : (memref<216xf32>, i32) -> f32
          %54 = "arith.addf"(%50, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %55 = "arith.index_cast"(%35) : (i32) -> index
          %56 = "arith.index_cast"(%55) : (index) -> i32
          %57 = "aps.read_smem"(%arg0, %56) : (memref<216xf32>, i32) -> f32
          %58 = "arith.addf"(%54, %57) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %59 = "arith.index_cast"(%17) : (i32) -> index
          %60 = "arith.index_cast"(%59) : (index) -> i32
          %61 = "aps.read_smem"(%arg1, %60) : (memref<216xf32>, i32) -> f32
          %62 = "arith.addf"(%61, %58) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %63 = "arith.divf"(%62, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %64 = "arith.index_cast"(%59) : (index) -> i32
          "aps.write_smem"(%63, %arg2, %64) : (f32, memref<216xf32>, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
}) : () -> ()

