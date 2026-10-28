#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {}, {}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<32x3xf32>, f32, f32, f32, f32, memref<i32>, memref<f32>) -> (), sym_name = "au_generated_candidate_0__direct"}> ({
  ^bb0(%arg2: memref<32x3xf32>, %arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32, %arg7: memref<i32>, %arg8: memref<f32>):
    %64 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %65 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %66 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %67 = "arith.constant"() <{value = 8.000000e-02 : f32}> : () -> f32
    %68 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg9: index):
      %69 = "arith.index_cast"(%arg9) : (index) -> i32
      %70 = "aps.read_smem"(%arg2, %69, %65) : (memref<32x3xf32>, i32, i32) -> f32
      %71 = "arith.subf"(%70, %arg3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %72 = "arith.index_cast"(%arg9) : (index) -> i32
      %73 = "aps.read_smem"(%arg2, %72, %66) : (memref<32x3xf32>, i32, i32) -> f32
      %74 = "arith.subf"(%73, %arg4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %75 = "arith.index_cast"(%arg9) : (index) -> i32
      %76 = "aps.read_smem"(%arg2, %75, %64) : (memref<32x3xf32>, i32, i32) -> f32
      %77 = "arith.subf"(%76, %arg5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %78 = "arith.cmpf"(%71, %68) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %79 = "scf.if"(%78) ({
        %97 = "arith.negf"(%71) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%97) : (f32) -> ()
      }, {
        "scf.yield"(%71) : (f32) -> ()
      }) : (i1) -> f32
      %80 = "arith.cmpf"(%74, %68) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %81 = "scf.if"(%80) ({
        %96 = "arith.negf"(%74) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%96) : (f32) -> ()
      }, {
        "scf.yield"(%74) : (f32) -> ()
      }) : (i1) -> f32
      %82 = "arith.addf"(%79, %81) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %83 = "arith.cmpf"(%77, %68) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %84 = "scf.if"(%83) ({
        %95 = "arith.negf"(%77) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%95) : (f32) -> ()
      }, {
        "scf.yield"(%77) : (f32) -> ()
      }) : (i1) -> f32
      %85 = "arith.addf"(%82, %84) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %86 = "arith.subf"(%85, %arg6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %87 = "arith.cmpf"(%86, %68) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %88 = "scf.if"(%87) ({
        %94 = "arith.negf"(%86) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%94) : (f32) -> ()
      }, {
        "scf.yield"(%86) : (f32) -> ()
      }) : (i1) -> f32
      %89 = "arith.cmpf"(%88, %67) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      "scf.if"(%89) ({
        %92 = "memref.load"(%arg7) : (memref<i32>) -> i32
        %93 = "arith.addi"(%92, %66) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%93, %arg7) : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }, {
      }) : (i1) -> ()
      %90 = "memref.load"(%arg8) : (memref<f32>) -> f32
      %91 = "arith.addf"(%90, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%91, %arg8) : (f32, memref<f32>) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 5 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 7 : i32} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_1__l0u"}> ({
  ^bb0(%arg0: memref<32x3xf32>):
    %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 8.900000e+01 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 19 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 9.700000e+01 : f32}> : () -> f32
    %8 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 1.010000e+02 : f32}> : () -> f32
    %12 = "arith.constant"() <{value = 101 : i32}> : () -> i32
    %13 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 3.000000e-01 : f32}> : () -> f32
    %16 = "arith.constant"() <{value = 3.100000e+01 : f32}> : () -> f32
    %17 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %18 = "arith.constant"() <{value = 4.000000e-01 : f32}> : () -> f32
    %19 = "arith.constant"() <{value = 1.200000e+01 : f32}> : () -> f32
    %20 = "arith.constant"() <{value = 6.000000e-01 : f32}> : () -> f32
    %21 = "arith.constant"() <{value = 6.400000e+01 : f32}> : () -> f32
    %22 = "arith.constant"() <{value = 2.000000e-01 : f32}> : () -> f32
    %23 = "arith.constant"() <{value = 64 : i32}> : () -> i32
    %24 = "arith.constant"() <{value = 768 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg1: index):
      %25 = "arith.index_cast"(%arg1) : (index) -> i32
      %26 = "arith.cmpi"(%25, %24) <{predicate = 2 : i64}> : (i32, i32) -> i1
      "scf.if"(%26) ({
        %45 = "arith.remsi"(%25, %23) : (i32, i32) -> i32
        %46 = "arith.divsi"(%25, %23) : (i32, i32) -> i32
        %47 = "arith.sitofp"(%45) : (i32) -> f32
        %48 = "arith.mulf"(%47, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %49 = "arith.divf"(%48, %21) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %50 = "arith.addf"(%49, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %51 = "arith.index_cast"(%arg1) : (index) -> i32
        "aps.write_smem"(%50, %arg0, %51, %2) : (f32, memref<32x3xf32>, i32, i32) -> ()
        %52 = "arith.sitofp"(%46) : (i32) -> f32
        %53 = "arith.mulf"(%52, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %54 = "arith.divf"(%53, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %55 = "arith.addf"(%54, %18) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %56 = "arith.index_cast"(%arg1) : (index) -> i32
        "aps.write_smem"(%55, %arg0, %56, %1) : (f32, memref<32x3xf32>, i32, i32) -> ()
        %57 = "arith.addi"(%45, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %58 = "arith.remsi"(%57, %17) : (i32, i32) -> i32
        %59 = "arith.sitofp"(%58) : (i32) -> f32
        %60 = "arith.mulf"(%59, %22) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %61 = "arith.divf"(%60, %16) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %62 = "arith.addf"(%61, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %63 = "arith.index_cast"(%arg1) : (index) -> i32
        "aps.write_smem"(%62, %arg0, %63, %0) : (f32, memref<32x3xf32>, i32, i32) -> ()
        "scf.yield"() : () -> ()
      }, {
        %27 = "arith.muli"(%25, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %28 = "arith.addi"(%27, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %29 = "arith.remsi"(%28, %12) : (i32, i32) -> i32
        %30 = "arith.sitofp"(%29) : (i32) -> f32
        %31 = "arith.divf"(%30, %11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %32 = "arith.index_cast"(%arg1) : (index) -> i32
        "aps.write_smem"(%31, %arg0, %32, %2) : (f32, memref<32x3xf32>, i32, i32) -> ()
        %33 = "arith.muli"(%25, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %34 = "arith.addi"(%33, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %35 = "arith.remsi"(%34, %8) : (i32, i32) -> i32
        %36 = "arith.sitofp"(%35) : (i32) -> f32
        %37 = "arith.divf"(%36, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %38 = "arith.index_cast"(%arg1) : (index) -> i32
        "aps.write_smem"(%37, %arg0, %38, %1) : (f32, memref<32x3xf32>, i32, i32) -> ()
        %39 = "arith.muli"(%25, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %40 = "arith.addi"(%39, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %41 = "arith.remsi"(%40, %4) : (i32, i32) -> i32
        %42 = "arith.sitofp"(%41) : (i32) -> f32
        %43 = "arith.divf"(%42, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %44 = "arith.index_cast"(%arg1) : (index) -> i32
        "aps.write_smem"(%43, %arg0, %44, %0) : (f32, memref<32x3xf32>, i32, i32) -> ()
        "scf.yield"() : () -> ()
      }) : (i1) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 1 : i32} : () -> ()
}) : () -> ()

