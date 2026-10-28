#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<f32>, memref<f32>, memref<f32>, memref<32x3xf32>, memref<32x3xf32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) -> (), sym_name = "au_generated_candidate_1__direct"}> ({
  ^bb0(%arg3: memref<f32>, %arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<32x3xf32>, %arg7: memref<32x3xf32>, %arg8: memref<f32>, %arg9: memref<f32>, %arg10: memref<f32>, %arg11: memref<f32>):
    %37 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %38 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %39 = "arith.constant"() <{value = 2 : index}> : () -> index
    %40 = "arith.constant"() <{value = 1 : index}> : () -> index
    %41 = "arith.constant"() <{value = 0 : index}> : () -> index
    %42 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %43 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg12: index):
      %44 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%43, %44) : (f32, memref<f32>) -> ()
      %45 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "memref.store"(%42, %45) : (i32, memref<i32>) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg13: index):
        %81 = "arith.index_cast"(%arg13) : (index) -> i32
        %82 = "memref.load"(%arg3) : (memref<f32>) -> f32
        %83 = "memref.load"(%arg4) : (memref<f32>) -> f32
        %84 = "memref.load"(%arg5) : (memref<f32>) -> f32
        %85 = "arith.index_cast"(%arg12) : (index) -> i32
        %86 = "aps.read_smem"(%arg6, %85, %42) : (memref<32x3xf32>, i32, i32) -> f32
        %87 = "arith.addf"(%86, %82) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %88 = "arith.index_cast"(%arg13) : (index) -> i32
        %89 = "aps.read_smem"(%arg7, %88, %42) : (memref<32x3xf32>, i32, i32) -> f32
        %90 = "arith.subf"(%87, %89) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %91 = "arith.index_cast"(%arg12) : (index) -> i32
        %92 = "aps.read_smem"(%arg6, %91, %38) : (memref<32x3xf32>, i32, i32) -> f32
        %93 = "arith.addf"(%92, %83) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %94 = "arith.index_cast"(%arg13) : (index) -> i32
        %95 = "aps.read_smem"(%arg7, %94, %38) : (memref<32x3xf32>, i32, i32) -> f32
        %96 = "arith.subf"(%93, %95) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %97 = "arith.index_cast"(%arg12) : (index) -> i32
        %98 = "aps.read_smem"(%arg6, %97, %37) : (memref<32x3xf32>, i32, i32) -> f32
        %99 = "arith.addf"(%98, %84) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %100 = "arith.index_cast"(%arg13) : (index) -> i32
        %101 = "aps.read_smem"(%arg7, %100, %37) : (memref<32x3xf32>, i32, i32) -> f32
        %102 = "arith.subf"(%99, %101) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %103 = "arith.mulf"(%90, %90) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %104 = "arith.mulf"(%96, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %105 = "arith.addf"(%103, %104) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %106 = "arith.mulf"(%102, %102) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %107 = "arith.addf"(%105, %106) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %108 = "memref.load"(%44) : (memref<f32>) -> f32
        %109 = "arith.cmpf"(%107, %108) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        "scf.if"(%109) ({
          "memref.store"(%107, %44) : (f32, memref<f32>) -> ()
          "memref.store"(%81, %45) : (i32, memref<i32>) -> ()
          "scf.yield"() : () -> ()
        }, {
        }) : (i1) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %46 = "memref.load"(%45) : (memref<i32>) -> i32
      %47 = "arith.index_cast"(%46) : (i32) -> index
      %48 = "arith.index_cast"(%47) : (index) -> i32
      %49 = "arith.index_cast"(%41) : (index) -> i32
      %50 = "aps.read_smem"(%arg7, %48, %49) : (memref<32x3xf32>, i32, i32) -> f32
      %51 = "arith.index_cast"(%arg12) : (index) -> i32
      %52 = "aps.read_smem"(%arg6, %51, %42) : (memref<32x3xf32>, i32, i32) -> f32
      %53 = "memref.load"(%arg3) : (memref<f32>) -> f32
      %54 = "arith.addf"(%52, %53) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %55 = "arith.subf"(%50, %54) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %56 = "memref.load"(%arg8) : (memref<f32>) -> f32
      %57 = "arith.addf"(%56, %55) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%57, %arg8) : (f32, memref<f32>) -> ()
      %58 = "arith.index_cast"(%47) : (index) -> i32
      %59 = "arith.index_cast"(%40) : (index) -> i32
      %60 = "aps.read_smem"(%arg7, %58, %59) : (memref<32x3xf32>, i32, i32) -> f32
      %61 = "arith.index_cast"(%arg12) : (index) -> i32
      %62 = "aps.read_smem"(%arg6, %61, %38) : (memref<32x3xf32>, i32, i32) -> f32
      %63 = "memref.load"(%arg4) : (memref<f32>) -> f32
      %64 = "arith.addf"(%62, %63) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %65 = "arith.subf"(%60, %64) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %66 = "memref.load"(%arg9) : (memref<f32>) -> f32
      %67 = "arith.addf"(%66, %65) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%67, %arg9) : (f32, memref<f32>) -> ()
      %68 = "arith.index_cast"(%47) : (index) -> i32
      %69 = "arith.index_cast"(%39) : (index) -> i32
      %70 = "aps.read_smem"(%arg7, %68, %69) : (memref<32x3xf32>, i32, i32) -> f32
      %71 = "arith.index_cast"(%arg12) : (index) -> i32
      %72 = "aps.read_smem"(%arg6, %71, %37) : (memref<32x3xf32>, i32, i32) -> f32
      %73 = "memref.load"(%arg5) : (memref<f32>) -> f32
      %74 = "arith.addf"(%72, %73) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %75 = "arith.subf"(%70, %74) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %76 = "memref.load"(%arg10) : (memref<f32>) -> f32
      %77 = "arith.addf"(%76, %75) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%77, %arg10) : (f32, memref<f32>) -> ()
      %78 = "memref.load"(%44) : (memref<f32>) -> f32
      %79 = "memref.load"(%arg11) : (memref<f32>) -> f32
      %80 = "arith.addf"(%79, %78) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%80, %arg11) : (f32, memref<f32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 7 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 9 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<32x3xf32>, memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_8__l0u"}> ({
  ^bb0(%arg0: memref<32x3xf32>, %arg1: memref<32x3xf32>):
    %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 6.000000e-02 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 4.000000e-02 : f32}> : () -> f32
    %5 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %6 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    %7 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 2.500000e-02 : f32}> : () -> f32
    %10 = "arith.constant"() <{value = 37 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %12 = "arith.constant"() <{value = 3.000000e-02 : f32}> : () -> f32
    %13 = "arith.constant"() <{value = 41 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg2: index):
      %15 = "arith.index_cast"(%arg2) : (index) -> i32
      %16 = "arith.muli"(%15, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %17 = "arith.remsi"(%16, %13) : (i32, i32) -> i32
      %18 = "arith.sitofp"(%17) : (i32) -> f32
      %19 = "arith.mulf"(%18, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %20 = "arith.muli"(%15, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %21 = "arith.remsi"(%20, %10) : (i32, i32) -> i32
      %22 = "arith.sitofp"(%21) : (i32) -> f32
      %23 = "arith.mulf"(%22, %9) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %24 = "arith.muli"(%15, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %25 = "arith.remsi"(%24, %7) : (i32, i32) -> i32
      %26 = "arith.sitofp"(%25) : (i32) -> f32
      %27 = "arith.mulf"(%26, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %28 = "arith.index_cast"(%arg2) : (index) -> i32
      "aps.write_smem"(%19, %arg0, %28, %2) : (f32, memref<32x3xf32>, i32, i32) -> ()
      %29 = "arith.index_cast"(%arg2) : (index) -> i32
      "aps.write_smem"(%23, %arg0, %29, %1) : (f32, memref<32x3xf32>, i32, i32) -> ()
      %30 = "arith.index_cast"(%arg2) : (index) -> i32
      "aps.write_smem"(%27, %arg0, %30, %0) : (f32, memref<32x3xf32>, i32, i32) -> ()
      %31 = "arith.addf"(%19, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %32 = "arith.index_cast"(%arg2) : (index) -> i32
      "aps.write_smem"(%31, %arg1, %32, %2) : (f32, memref<32x3xf32>, i32, i32) -> ()
      %33 = "arith.subf"(%23, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %34 = "arith.index_cast"(%arg2) : (index) -> i32
      "aps.write_smem"(%33, %arg1, %34, %1) : (f32, memref<32x3xf32>, i32, i32) -> ()
      %35 = "arith.addf"(%27, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %36 = "arith.index_cast"(%arg2) : (index) -> i32
      "aps.write_smem"(%35, %arg1, %36, %0) : (f32, memref<32x3xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
}) : () -> ()

