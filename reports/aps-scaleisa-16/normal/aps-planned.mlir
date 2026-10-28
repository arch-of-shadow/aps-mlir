#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (3)>
#map2 = affine_map<() -> (12)>
#map3 = affine_map<() -> (10)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{}, {scaleisa.noalias}, {scaleisa.noalias}, {}], function_type = (i32, memref<12x12x3xf32>, memref<12x12x3xf32>, index) -> (), sym_name = "au_generated_candidate_12__direct"}> ({
  ^bb0(%arg5: i32, %arg6: memref<12x12x3xf32>, %arg7: memref<12x12x3xf32>, %arg8: index):
    %40 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %41 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %42 = "arith.constant"() <{value = 9.000000e+00 : f32}> : () -> f32
    %43 = "arith.constant"() <{value = 2 : index}> : () -> index
    %44 = "arith.constant"() <{value = 1 : index}> : () -> index
    %45 = "arith.constant"() <{value = 0 : index}> : () -> index
    %46 = "arith.constant"() <{value = true}> : () -> i1
    %47 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %48 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %49 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %50 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg9: index):
      %51 = "arith.index_cast"(%arg9) : (index) -> i32
      %52 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%50, %52) : (f32, memref<f32>) -> ()
      %53 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%50, %53) : (f32, memref<f32>) -> ()
      %54 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%50, %54) : (f32, memref<f32>) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg10: index):
        %67 = "arith.index_cast"(%arg10) : (index) -> i32
        %68 = "arith.addi"(%67, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg11: index):
          %69 = "arith.index_cast"(%arg11) : (index) -> i32
          %70 = "arith.addi"(%69, %49) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %71 = "arith.addi"(%arg5, %68) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %72 = "arith.cmpi"(%71, %48) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %73 = "arith.cmpi"(%71, %47) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %74 = "arith.andi"(%72, %73) : (i1, i1) -> i1
          %75 = "arith.xori"(%74, %46) : (i1, i1) -> i1
          %76 = "arith.andi"(%75, %72) : (i1, i1) -> i1
          %77 = "arith.select"(%74, %47, %48) : (i1, i32, i32) -> i32
          %78 = "arith.select"(%76, %71, %77) : (i1, i32, i32) -> i32
          %79 = "arith.addi"(%51, %70) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %80 = "arith.cmpi"(%79, %48) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %81 = "arith.cmpi"(%79, %47) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %82 = "arith.andi"(%80, %81) : (i1, i1) -> i1
          %83 = "arith.xori"(%82, %46) : (i1, i1) -> i1
          %84 = "arith.andi"(%83, %80) : (i1, i1) -> i1
          %85 = "arith.select"(%82, %47, %48) : (i1, i32, i32) -> i32
          %86 = "arith.select"(%84, %79, %85) : (i1, i32, i32) -> i32
          %87 = "arith.index_cast"(%78) : (i32) -> index
          %88 = "arith.index_cast"(%86) : (i32) -> index
          %89 = "arith.index_cast"(%87) : (index) -> i32
          %90 = "arith.index_cast"(%88) : (index) -> i32
          %91 = "arith.index_cast"(%45) : (index) -> i32
          %92 = "aps.read_smem"(%arg6, %89, %90, %91) : (memref<12x12x3xf32>, i32, i32, i32) -> f32
          %93 = "memref.load"(%52) : (memref<f32>) -> f32
          %94 = "arith.addf"(%93, %92) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%94, %52) : (f32, memref<f32>) -> ()
          %95 = "arith.index_cast"(%87) : (index) -> i32
          %96 = "arith.index_cast"(%88) : (index) -> i32
          %97 = "arith.index_cast"(%44) : (index) -> i32
          %98 = "aps.read_smem"(%arg6, %95, %96, %97) : (memref<12x12x3xf32>, i32, i32, i32) -> f32
          %99 = "memref.load"(%53) : (memref<f32>) -> f32
          %100 = "arith.addf"(%99, %98) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%100, %53) : (f32, memref<f32>) -> ()
          %101 = "arith.index_cast"(%87) : (index) -> i32
          %102 = "arith.index_cast"(%88) : (index) -> i32
          %103 = "arith.index_cast"(%43) : (index) -> i32
          %104 = "aps.read_smem"(%arg6, %101, %102, %103) : (memref<12x12x3xf32>, i32, i32, i32) -> f32
          %105 = "memref.load"(%54) : (memref<f32>) -> f32
          %106 = "arith.addf"(%105, %104) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%106, %54) : (f32, memref<f32>) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %55 = "memref.load"(%52) : (memref<f32>) -> f32
      %56 = "arith.divf"(%55, %42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %57 = "arith.index_cast"(%arg8) : (index) -> i32
      %58 = "arith.index_cast"(%arg9) : (index) -> i32
      "aps.write_smem"(%56, %arg7, %57, %58, %48) : (f32, memref<12x12x3xf32>, i32, i32, i32) -> ()
      %59 = "memref.load"(%53) : (memref<f32>) -> f32
      %60 = "arith.divf"(%59, %42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %61 = "arith.index_cast"(%arg8) : (index) -> i32
      %62 = "arith.index_cast"(%arg9) : (index) -> i32
      "aps.write_smem"(%60, %arg7, %61, %62, %41) : (f32, memref<12x12x3xf32>, i32, i32, i32) -> ()
      %63 = "memref.load"(%54) : (memref<f32>) -> f32
      %64 = "arith.divf"(%63, %42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %65 = "arith.index_cast"(%arg8) : (index) -> i32
      %66 = "arith.index_cast"(%arg9) : (index) -> i32
      "aps.write_smem"(%64, %arg7, %65, %66, %40) : (f32, memref<12x12x3xf32>, i32, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 2 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 4 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<12x12x3xf32>, memref<12x12xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_24__direct"}> ({
  ^bb0(%arg0: memref<12x12x3xf32>, %arg1: memref<12x12xf32>, %arg2: memref<i32>):
    %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 1 : index}> : () -> index
    %4 = "arith.constant"() <{value = 6.400000e+01 : f32}> : () -> f32
    %5 = "arith.constant"() <{value = 1.100000e+01 : f32}> : () -> f32
    %6 = "arith.constant"() <{value = 3.100000e+01 : f32}> : () -> f32
    %7 = "arith.constant"() <{value = 2.300000e+01 : f32}> : () -> f32
    %8 = "arith.constant"() <{value = 1.700000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg3: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg4: index):
        %9 = "arith.addi"(%arg3, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %10 = "arith.addi"(%arg4, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %11 = "arith.index_cast"(%9) : (index) -> i32
        %12 = "arith.index_cast"(%10) : (index) -> i32
        %13 = "aps.read_smem"(%arg0, %11, %12, %2) : (memref<12x12x3xf32>, i32, i32, i32) -> f32
        %14 = "arith.mulf"(%13, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %15 = "arith.addi"(%arg3, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %16 = "arith.addi"(%arg4, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %17 = "arith.index_cast"(%15) : (index) -> i32
        %18 = "arith.index_cast"(%16) : (index) -> i32
        %19 = "aps.read_smem"(%arg0, %17, %18, %1) : (memref<12x12x3xf32>, i32, i32, i32) -> f32
        %20 = "arith.mulf"(%19, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %21 = "arith.addf"(%14, %20) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %22 = "arith.addi"(%arg3, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %23 = "arith.addi"(%arg4, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %24 = "arith.index_cast"(%22) : (index) -> i32
        %25 = "arith.index_cast"(%23) : (index) -> i32
        %26 = "aps.read_smem"(%arg0, %24, %25, %0) : (memref<12x12x3xf32>, i32, i32, i32) -> f32
        %27 = "arith.mulf"(%26, %6) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %28 = "arith.addf"(%21, %27) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %29 = "arith.addi"(%arg3, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %30 = "arith.addi"(%arg4, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        %31 = "arith.index_cast"(%29) : (index) -> i32
        %32 = "arith.index_cast"(%30) : (index) -> i32
        %33 = "aps.read_smem"(%arg1, %31, %32) : (memref<12x12xf32>, i32, i32) -> f32
        %34 = "arith.mulf"(%33, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %35 = "arith.addf"(%28, %34) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %36 = "arith.mulf"(%35, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %37 = "arith.fptosi"(%36) : (f32) -> i32
        %38 = "memref.load"(%arg2) : (memref<i32>) -> i32
        %39 = "arith.addi"(%38, %37) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%39, %arg2) : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
}) : () -> ()

