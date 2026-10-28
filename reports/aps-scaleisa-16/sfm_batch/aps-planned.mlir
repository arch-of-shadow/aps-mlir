#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (24)>
#map2 = affine_map<() -> (8)>
#map3 = affine_map<() -> (3)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_22__direct"}> ({
  ^bb0(%arg16: memref<24x3xf32>, %arg17: memref<i32>):
    %113 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %114 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %115 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %116 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
    %117 = "arith.constant"() <{value = 1.090000e+02 : f32}> : () -> f32
    %118 = "arith.constant"() <{value = 8.300000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg18: index):
      %119 = "arith.index_cast"(%arg18) : (index) -> i32
      %120 = "aps.read_smem"(%arg16, %119, %115) : (memref<24x3xf32>, i32, i32) -> f32
      %121 = "arith.mulf"(%120, %118) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %122 = "arith.fptosi"(%121) : (f32) -> i32
      %123 = "memref.load"(%arg17) : (memref<i32>) -> i32
      %124 = "arith.addi"(%123, %122) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%124, %arg17) : (i32, memref<i32>) -> ()
      %125 = "arith.index_cast"(%arg18) : (index) -> i32
      %126 = "aps.read_smem"(%arg16, %125, %114) : (memref<24x3xf32>, i32, i32) -> f32
      %127 = "arith.mulf"(%126, %117) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %128 = "arith.fptosi"(%127) : (f32) -> i32
      %129 = "arith.addi"(%124, %128) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%129, %arg17) : (i32, memref<i32>) -> ()
      %130 = "arith.index_cast"(%arg18) : (index) -> i32
      %131 = "aps.read_smem"(%arg16, %130, %113) : (memref<24x3xf32>, i32, i32) -> f32
      %132 = "arith.mulf"(%131, %116) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %133 = "arith.fptosi"(%132) : (f32) -> i32
      %134 = "arith.addi"(%129, %133) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%134, %arg17) : (i32, memref<i32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{function_type = (memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_38__l0u"}> ({
  ^bb0(%arg14: memref<24x3xf32>):
    %86 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %87 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %88 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %89 = "arith.constant"() <{value = 2.800000e+00 : f32}> : () -> f32
    %90 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %91 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %92 = "arith.constant"() <{value = -0.899999976 : f32}> : () -> f32
    %93 = "arith.constant"() <{value = 6.000000e-01 : f32}> : () -> f32
    %94 = "arith.constant"() <{value = -1.200000e+00 : f32}> : () -> f32
    %95 = "arith.constant"() <{value = 4.500000e-01 : f32}> : () -> f32
    %96 = "arith.constant"() <{value = 6 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg15: index):
      %97 = "arith.index_cast"(%arg15) : (index) -> i32
      %98 = "arith.remsi"(%97, %96) : (i32, i32) -> i32
      %99 = "arith.sitofp"(%98) : (i32) -> f32
      %100 = "arith.mulf"(%99, %95) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %101 = "arith.addf"(%100, %94) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %102 = "arith.index_cast"(%arg15) : (index) -> i32
      "aps.write_smem"(%101, %arg14, %102, %88) : (f32, memref<24x3xf32>, i32, i32) -> ()
      %103 = "arith.divsi"(%97, %96) : (i32, i32) -> i32
      %104 = "arith.sitofp"(%103) : (i32) -> f32
      %105 = "arith.mulf"(%104, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %106 = "arith.addf"(%105, %92) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %107 = "arith.index_cast"(%arg15) : (index) -> i32
      "aps.write_smem"(%106, %arg14, %107, %87) : (f32, memref<24x3xf32>, i32, i32) -> ()
      %108 = "arith.remsi"(%97, %91) : (i32, i32) -> i32
      %109 = "arith.sitofp"(%108) : (i32) -> f32
      %110 = "arith.mulf"(%109, %90) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %111 = "arith.addf"(%110, %89) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %112 = "arith.index_cast"(%arg15) : (index) -> i32
      "aps.write_smem"(%111, %arg14, %112, %86) : (f32, memref<24x3xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 1 : i32} : () -> ()
  "func.func"() <{function_type = (memref<8x3xf32>) -> (), sym_name = "au_generated_candidate_44__l0u"}> ({
  ^bb0(%arg12: memref<8x3xf32>):
    %63 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %64 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %65 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %66 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %67 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
    %68 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %69 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %70 = "arith.constant"() <{value = -1.800000e+00 : f32}> : () -> f32
    %71 = "arith.constant"() <{value = 5.500000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg13: index):
      %72 = "arith.index_cast"(%arg13) : (index) -> i32
      %73 = "arith.sitofp"(%72) : (i32) -> f32
      %74 = "arith.mulf"(%73, %71) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %75 = "arith.addf"(%74, %70) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %76 = "arith.index_cast"(%arg13) : (index) -> i32
      "aps.write_smem"(%75, %arg12, %76, %64) : (f32, memref<8x3xf32>, i32, i32) -> ()
      %77 = "arith.remsi"(%72, %69) : (i32, i32) -> i32
      %78 = "arith.addi"(%77, %68) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %79 = "arith.sitofp"(%78) : (i32) -> f32
      %80 = "arith.mulf"(%79, %67) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %81 = "arith.index_cast"(%arg13) : (index) -> i32
      "aps.write_smem"(%80, %arg12, %81, %63) : (f32, memref<8x3xf32>, i32, i32) -> ()
      %82 = "arith.remsi"(%72, %66) : (i32, i32) -> i32
      %83 = "arith.sitofp"(%82) : (i32) -> f32
      %84 = "arith.mulf"(%83, %65) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %85 = "arith.index_cast"(%arg13) : (index) -> i32
      "aps.write_smem"(%84, %arg12, %85, %66) : (f32, memref<8x3xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 1 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<24x3xf32>, memref<24x3xf32>, memref<24x3xf32>) -> (), sym_name = "au_generated_candidate_48__l0p_l1u"}> ({
  ^bb0(%arg7: memref<24x3xf32>, %arg8: memref<24x3xf32>, %arg9: memref<24x3xf32>):
    %48 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg10: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg11: index):
        %49 = "arith.index_cast"(%arg10) : (index) -> i32
        %50 = "arith.index_cast"(%arg11) : (index) -> i32
        %51 = "aps.read_smem"(%arg7, %49, %50) : (memref<24x3xf32>, i32, i32) -> f32
        %52 = "arith.mulf"(%51, %48) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %53 = "arith.index_cast"(%arg10) : (index) -> i32
        %54 = "arith.index_cast"(%arg11) : (index) -> i32
        %55 = "aps.read_smem"(%arg8, %53, %54) : (memref<24x3xf32>, i32, i32) -> f32
        %56 = "arith.divf"(%52, %55) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %57 = "arith.index_cast"(%arg10) : (index) -> i32
        %58 = "arith.index_cast"(%arg11) : (index) -> i32
        %59 = "aps.read_smem"(%arg9, %57, %58) : (memref<24x3xf32>, i32, i32) -> f32
        %60 = "arith.subf"(%59, %56) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %61 = "arith.index_cast"(%arg10) : (index) -> i32
        %62 = "arith.index_cast"(%arg11) : (index) -> i32
        "aps.write_smem"(%60, %arg9, %61, %62) : (f32, memref<24x3xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {pipeline = 1 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x3xf32>, memref<8x3xf32>, memref<8x3xf32>) -> (), sym_name = "au_generated_candidate_49__l1p"}> ({
  ^bb0(%arg2: memref<8x3xf32>, %arg3: memref<8x3xf32>, %arg4: memref<8x3xf32>):
    %33 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg5: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg6: index):
        %34 = "arith.index_cast"(%arg5) : (index) -> i32
        %35 = "arith.index_cast"(%arg6) : (index) -> i32
        %36 = "aps.read_smem"(%arg2, %34, %35) : (memref<8x3xf32>, i32, i32) -> f32
        %37 = "arith.mulf"(%36, %33) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %38 = "arith.index_cast"(%arg5) : (index) -> i32
        %39 = "arith.index_cast"(%arg6) : (index) -> i32
        %40 = "aps.read_smem"(%arg3, %38, %39) : (memref<8x3xf32>, i32, i32) -> f32
        %41 = "arith.divf"(%37, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %42 = "arith.index_cast"(%arg5) : (index) -> i32
        %43 = "arith.index_cast"(%arg6) : (index) -> i32
        %44 = "aps.read_smem"(%arg4, %42, %43) : (memref<8x3xf32>, i32, i32) -> f32
        %45 = "arith.subf"(%44, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %46 = "arith.index_cast"(%arg5) : (index) -> i32
        %47 = "arith.index_cast"(%arg6) : (index) -> i32
        "aps.write_smem"(%45, %arg4, %46, %47) : (f32, memref<8x3xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {pipeline = 1 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
  "func.func"() <{function_type = (memref<8x3xf32>) -> (), sym_name = "au_generated_candidate_9__l0u"}> ({
  ^bb0(%arg0: memref<8x3xf32>):
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %5 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %7 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 4 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg1: index):
      %9 = "arith.index_cast"(%arg1) : (index) -> i32
      %10 = "arith.remsi"(%9, %8) : (i32, i32) -> i32
      %11 = "arith.addi"(%10, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %12 = "arith.sitofp"(%11) : (i32) -> f32
      %13 = "arith.mulf"(%12, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %14 = "arith.index_cast"(%arg1) : (index) -> i32
      %15 = "aps.read_smem"(%arg0, %14, %1) : (memref<8x3xf32>, i32, i32) -> f32
      %16 = "arith.addf"(%15, %13) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %17 = "arith.index_cast"(%arg1) : (index) -> i32
      "aps.write_smem"(%16, %arg0, %17, %1) : (f32, memref<8x3xf32>, i32, i32) -> ()
      %18 = "arith.remsi"(%9, %5) : (i32, i32) -> i32
      %19 = "arith.sitofp"(%18) : (i32) -> f32
      %20 = "arith.mulf"(%19, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %21 = "arith.index_cast"(%arg1) : (index) -> i32
      %22 = "aps.read_smem"(%arg0, %21, %0) : (memref<8x3xf32>, i32, i32) -> f32
      %23 = "arith.subf"(%22, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %24 = "arith.index_cast"(%arg1) : (index) -> i32
      "aps.write_smem"(%23, %arg0, %24, %0) : (f32, memref<8x3xf32>, i32, i32) -> ()
      %25 = "arith.remsi"(%9, %3) : (i32, i32) -> i32
      %26 = "arith.addi"(%25, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %27 = "arith.sitofp"(%26) : (i32) -> f32
      %28 = "arith.mulf"(%27, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %29 = "arith.index_cast"(%arg1) : (index) -> i32
      %30 = "aps.read_smem"(%arg0, %29, %5) : (memref<8x3xf32>, i32, i32) -> f32
      %31 = "arith.addf"(%30, %28) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %32 = "arith.index_cast"(%arg1) : (index) -> i32
      "aps.write_smem"(%31, %arg0, %32, %5) : (f32, memref<8x3xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 1 : i32} : () -> ()
}) : () -> ()

