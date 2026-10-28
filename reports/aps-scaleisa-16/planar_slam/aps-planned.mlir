#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (15)>
#map2 = affine_map<() -> (16)>
#map3 = affine_map<() -> (20)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<16x3xf32>, memref<16x3xf32>, memref<16x3xf32>) -> (), sym_name = "au_generated_candidate_12__direct"}> ({
  ^bb0(%arg14: memref<16x3xf32>, %arg15: memref<16x3xf32>, %arg16: memref<16x3xf32>):
    %105 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %106 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %107 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %108 = "arith.constant"() <{value = 1 : index}> : () -> index
    %109 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %110 = "arith.constant"() <{value = 2.200000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg17: index):
      %111 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %112 = "arith.index_cast"(%111) : (index) -> i32
      %113 = "aps.read_smem"(%arg14, %112, %107) : (memref<16x3xf32>, i32, i32) -> f32
      %114 = "arith.mulf"(%113, %110) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %115 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %116 = "arith.index_cast"(%115) : (index) -> i32
      %117 = "aps.read_smem"(%arg15, %116, %107) : (memref<16x3xf32>, i32, i32) -> f32
      %118 = "arith.divf"(%114, %117) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %119 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %120 = "arith.index_cast"(%119) : (index) -> i32
      %121 = "aps.read_smem"(%arg16, %120, %107) : (memref<16x3xf32>, i32, i32) -> f32
      %122 = "arith.subf"(%121, %118) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %123 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %124 = "arith.index_cast"(%123) : (index) -> i32
      "aps.write_smem"(%122, %arg16, %124, %107) : (f32, memref<16x3xf32>, i32, i32) -> ()
      %125 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %126 = "arith.index_cast"(%125) : (index) -> i32
      %127 = "aps.read_smem"(%arg14, %126, %106) : (memref<16x3xf32>, i32, i32) -> f32
      %128 = "arith.mulf"(%127, %110) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %129 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %130 = "arith.index_cast"(%129) : (index) -> i32
      %131 = "aps.read_smem"(%arg15, %130, %106) : (memref<16x3xf32>, i32, i32) -> f32
      %132 = "arith.divf"(%128, %131) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %133 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %134 = "arith.index_cast"(%133) : (index) -> i32
      %135 = "aps.read_smem"(%arg16, %134, %106) : (memref<16x3xf32>, i32, i32) -> f32
      %136 = "arith.subf"(%135, %132) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %137 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %138 = "arith.index_cast"(%137) : (index) -> i32
      "aps.write_smem"(%136, %arg16, %138, %106) : (f32, memref<16x3xf32>, i32, i32) -> ()
      %139 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %140 = "arith.index_cast"(%139) : (index) -> i32
      %141 = "aps.read_smem"(%arg14, %140, %105) : (memref<16x3xf32>, i32, i32) -> f32
      %142 = "arith.mulf"(%141, %109) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %143 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %144 = "arith.index_cast"(%143) : (index) -> i32
      %145 = "aps.read_smem"(%arg15, %144, %105) : (memref<16x3xf32>, i32, i32) -> f32
      %146 = "arith.divf"(%142, %145) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %147 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %148 = "arith.index_cast"(%147) : (index) -> i32
      %149 = "aps.read_smem"(%arg16, %148, %105) : (memref<16x3xf32>, i32, i32) -> f32
      %150 = "arith.subf"(%149, %146) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %151 = "arith.addi"(%arg17, %108) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
      %152 = "arith.index_cast"(%151) : (index) -> i32
      "aps.write_smem"(%150, %arg16, %152, %105) : (f32, memref<16x3xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<16x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_21__direct"}> ({
  ^bb0(%arg11: memref<16x3xf32>, %arg12: memref<i32>):
    %83 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %84 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %85 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %86 = "arith.constant"() <{value = 1.970000e+02 : f32}> : () -> f32
    %87 = "arith.constant"() <{value = 1.310000e+02 : f32}> : () -> f32
    %88 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg13: index):
      %89 = "arith.index_cast"(%arg13) : (index) -> i32
      %90 = "aps.read_smem"(%arg11, %89, %85) : (memref<16x3xf32>, i32, i32) -> f32
      %91 = "arith.mulf"(%90, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %92 = "arith.fptosi"(%91) : (f32) -> i32
      %93 = "memref.load"(%arg12) : (memref<i32>) -> i32
      %94 = "arith.addi"(%93, %92) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%94, %arg12) : (i32, memref<i32>) -> ()
      %95 = "arith.index_cast"(%arg13) : (index) -> i32
      %96 = "aps.read_smem"(%arg11, %95, %84) : (memref<16x3xf32>, i32, i32) -> f32
      %97 = "arith.mulf"(%96, %87) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %98 = "arith.fptosi"(%97) : (f32) -> i32
      %99 = "arith.addi"(%94, %98) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%99, %arg12) : (i32, memref<i32>) -> ()
      %100 = "arith.index_cast"(%arg13) : (index) -> i32
      %101 = "aps.read_smem"(%arg11, %100, %83) : (memref<16x3xf32>, i32, i32) -> f32
      %102 = "arith.mulf"(%101, %86) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %103 = "arith.fptosi"(%102) : (f32) -> i32
      %104 = "arith.addi"(%99, %103) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%104, %arg12) : (i32, memref<i32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{function_type = (memref<16x3xf32>) -> (), sym_name = "au_generated_candidate_34__l0u"}> ({
  ^bb0(%arg9: memref<16x3xf32>):
    %56 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %57 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %58 = "arith.constant"() <{value = 1.500000e-02 : f32}> : () -> f32
    %59 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %60 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %61 = "arith.constant"() <{value = 1.800000e-01 : f32}> : () -> f32
    %62 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %63 = "arith.constant"() <{value = 5.000000e-02 : f32}> : () -> f32
    %64 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %65 = "arith.constant"() <{value = 7.500000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg10: index):
      %66 = "arith.index_cast"(%arg10) : (index) -> i32
      %67 = "arith.sitofp"(%66) : (i32) -> f32
      %68 = "arith.mulf"(%67, %65) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %69 = "arith.remsi"(%66, %64) : (i32, i32) -> i32
      %70 = "arith.sitofp"(%69) : (i32) -> f32
      %71 = "arith.mulf"(%70, %63) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %72 = "arith.addf"(%68, %71) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %73 = "arith.index_cast"(%arg10) : (index) -> i32
      "aps.write_smem"(%72, %arg9, %73, %57) : (f32, memref<16x3xf32>, i32, i32) -> ()
      %74 = "arith.remsi"(%66, %62) : (i32, i32) -> i32
      %75 = "arith.sitofp"(%74) : (i32) -> f32
      %76 = "arith.mulf"(%75, %61) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %77 = "arith.index_cast"(%arg10) : (index) -> i32
      "aps.write_smem"(%76, %arg9, %77, %56) : (f32, memref<16x3xf32>, i32, i32) -> ()
      %78 = "arith.remsi"(%66, %60) : (i32, i32) -> i32
      %79 = "arith.addi"(%78, %59) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %80 = "arith.sitofp"(%79) : (i32) -> f32
      %81 = "arith.mulf"(%80, %58) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %82 = "arith.index_cast"(%arg10) : (index) -> i32
      "aps.write_smem"(%81, %arg9, %82, %62) : (f32, memref<16x3xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 1 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<20x2xf32>, memref<20x2xf32>, memref<20x2xf32>) -> (), sym_name = "au_generated_candidate_41__direct"}> ({
  ^bb0(%arg5: memref<20x2xf32>, %arg6: memref<20x2xf32>, %arg7: memref<20x2xf32>):
    %33 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %34 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %35 = "arith.constant"() <{value = 2.500000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg8: index):
      %36 = "arith.index_cast"(%arg8) : (index) -> i32
      %37 = "aps.read_smem"(%arg5, %36, %34) : (memref<20x2xf32>, i32, i32) -> f32
      %38 = "arith.mulf"(%37, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %39 = "arith.index_cast"(%arg8) : (index) -> i32
      %40 = "aps.read_smem"(%arg6, %39, %34) : (memref<20x2xf32>, i32, i32) -> f32
      %41 = "arith.divf"(%38, %40) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %42 = "arith.index_cast"(%arg8) : (index) -> i32
      %43 = "aps.read_smem"(%arg7, %42, %34) : (memref<20x2xf32>, i32, i32) -> f32
      %44 = "arith.subf"(%43, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %45 = "arith.index_cast"(%arg8) : (index) -> i32
      "aps.write_smem"(%44, %arg7, %45, %34) : (f32, memref<20x2xf32>, i32, i32) -> ()
      %46 = "arith.index_cast"(%arg8) : (index) -> i32
      %47 = "aps.read_smem"(%arg5, %46, %33) : (memref<20x2xf32>, i32, i32) -> f32
      %48 = "arith.mulf"(%47, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %49 = "arith.index_cast"(%arg8) : (index) -> i32
      %50 = "aps.read_smem"(%arg6, %49, %33) : (memref<20x2xf32>, i32, i32) -> f32
      %51 = "arith.divf"(%48, %50) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %52 = "arith.index_cast"(%arg8) : (index) -> i32
      %53 = "aps.read_smem"(%arg7, %52, %33) : (memref<20x2xf32>, i32, i32) -> f32
      %54 = "arith.subf"(%53, %51) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %55 = "arith.index_cast"(%arg8) : (index) -> i32
      "aps.write_smem"(%54, %arg7, %55, %33) : (f32, memref<20x2xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<20x2xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_51__direct"}> ({
  ^bb0(%arg2: memref<20x2xf32>, %arg3: memref<i32>):
    %18 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %19 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %20 = "arith.constant"() <{value = 1.810000e+02 : f32}> : () -> f32
    %21 = "arith.constant"() <{value = 1.510000e+02 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg4: index):
      %22 = "arith.index_cast"(%arg4) : (index) -> i32
      %23 = "aps.read_smem"(%arg2, %22, %19) : (memref<20x2xf32>, i32, i32) -> f32
      %24 = "arith.mulf"(%23, %21) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %25 = "arith.fptosi"(%24) : (f32) -> i32
      %26 = "memref.load"(%arg3) : (memref<i32>) -> i32
      %27 = "arith.addi"(%26, %25) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%27, %arg3) : (i32, memref<i32>) -> ()
      %28 = "arith.index_cast"(%arg4) : (index) -> i32
      %29 = "aps.read_smem"(%arg2, %28, %18) : (memref<20x2xf32>, i32, i32) -> f32
      %30 = "arith.mulf"(%29, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %31 = "arith.fptosi"(%30) : (f32) -> i32
      %32 = "arith.addi"(%27, %31) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%32, %arg3) : (i32, memref<i32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{function_type = (memref<20x2xf32>) -> (), sym_name = "au_generated_candidate_56__l0u"}> ({
  ^bb0(%arg0: memref<20x2xf32>):
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 1.200000e+00 : f32}> : () -> f32
    %3 = "arith.constant"() <{value = 2.700000e-01 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 1.600000e+00 : f32}> : () -> f32
    %5 = "arith.constant"() <{value = 3.800000e-01 : f32}> : () -> f32
    %6 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg1: index):
      %7 = "arith.index_cast"(%arg1) : (index) -> i32
      %8 = "arith.remsi"(%7, %6) : (i32, i32) -> i32
      %9 = "arith.sitofp"(%8) : (i32) -> f32
      %10 = "arith.mulf"(%9, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %11 = "arith.addf"(%10, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %12 = "arith.index_cast"(%arg1) : (index) -> i32
      "aps.write_smem"(%11, %arg0, %12, %1) : (f32, memref<20x2xf32>, i32, i32) -> ()
      %13 = "arith.divsi"(%7, %6) : (i32, i32) -> i32
      %14 = "arith.sitofp"(%13) : (i32) -> f32
      %15 = "arith.mulf"(%14, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %16 = "arith.addf"(%15, %2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %17 = "arith.index_cast"(%arg1) : (index) -> i32
      "aps.write_smem"(%16, %arg0, %17, %0) : (f32, memref<20x2xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [2], scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 1 : i32} : () -> ()
}) : () -> ()

