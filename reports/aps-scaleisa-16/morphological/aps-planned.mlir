#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (12)>
#map2 = affine_map<() -> (5)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<12x12xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_40__direct"}> ({
  ^bb0(%arg14: memref<12x12xf32>, %arg15: memref<i32>):
    %78 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %79 = "arith.constant"() <{value = 1.280000e+02 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg16: index):
      %80 = "arith.index_cast"(%arg16) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg17: index):
        %81 = "arith.index_cast"(%arg17) : (index) -> i32
        %82 = "arith.index_cast"(%arg16) : (index) -> i32
        %83 = "arith.index_cast"(%arg17) : (index) -> i32
        %84 = "aps.read_smem"(%arg14, %82, %83) : (memref<12x12xf32>, i32, i32) -> f32
        %85 = "arith.mulf"(%84, %79) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %86 = "arith.fptosi"(%85) : (f32) -> i32
        %87 = "arith.addi"(%81, %80) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %88 = "arith.andi"(%87, %78) : (i32, i32) -> i32
        %89 = "arith.addi"(%86, %88) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %90 = "memref.load"(%arg15) : (memref<i32>) -> i32
        %91 = "arith.addi"(%90, %89) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%91, %arg15) : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}], function_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> (), sym_name = "au_generated_candidate_8__direct"}> ({
  ^bb0(%arg7: memref<12x12xf32>, %arg8: index, %arg9: i32, %arg10: memref<12x12xf32>):
    %39 = "arith.constant"() <{value = true}> : () -> i1
    %40 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %41 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %42 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg11: index):
      %43 = "arith.index_cast"(%arg11) : (index) -> i32
      %44 = "arith.index_cast"(%arg8) : (index) -> i32
      %45 = "arith.index_cast"(%arg11) : (index) -> i32
      %46 = "aps.read_smem"(%arg7, %44, %45) : (memref<12x12xf32>, i32, i32) -> f32
      %47 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%46, %47) : (f32, memref<f32>) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg12: index):
        %51 = "arith.index_cast"(%arg12) : (index) -> i32
        %52 = "arith.addi"(%51, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg13: index):
          %53 = "arith.index_cast"(%arg13) : (index) -> i32
          %54 = "arith.addi"(%53, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %55 = "arith.addi"(%arg9, %52) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %56 = "arith.cmpi"(%55, %41) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %57 = "arith.cmpi"(%55, %40) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %58 = "arith.andi"(%56, %57) : (i1, i1) -> i1
          %59 = "arith.xori"(%58, %39) : (i1, i1) -> i1
          %60 = "arith.andi"(%59, %56) : (i1, i1) -> i1
          %61 = "arith.select"(%58, %40, %41) : (i1, i32, i32) -> i32
          %62 = "arith.select"(%60, %55, %61) : (i1, i32, i32) -> i32
          %63 = "arith.addi"(%43, %54) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %64 = "arith.cmpi"(%63, %41) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %65 = "arith.cmpi"(%63, %40) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %66 = "arith.andi"(%64, %65) : (i1, i1) -> i1
          %67 = "arith.xori"(%66, %39) : (i1, i1) -> i1
          %68 = "arith.andi"(%67, %64) : (i1, i1) -> i1
          %69 = "arith.select"(%66, %40, %41) : (i1, i32, i32) -> i32
          %70 = "arith.select"(%68, %63, %69) : (i1, i32, i32) -> i32
          %71 = "arith.index_cast"(%62) : (i32) -> index
          %72 = "arith.index_cast"(%70) : (i32) -> index
          %73 = "arith.index_cast"(%71) : (index) -> i32
          %74 = "arith.index_cast"(%72) : (index) -> i32
          %75 = "aps.read_smem"(%arg7, %73, %74) : (memref<12x12xf32>, i32, i32) -> f32
          %76 = "memref.load"(%47) : (memref<f32>) -> f32
          %77 = "arith.cmpf"(%75, %76) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
          "scf.if"(%77) ({
            "memref.store"(%75, %47) : (f32, memref<f32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %48 = "memref.load"(%47) : (memref<f32>) -> f32
      %49 = "arith.index_cast"(%arg8) : (index) -> i32
      %50 = "arith.index_cast"(%arg11) : (index) -> i32
      "aps.write_smem"(%48, %arg10, %49, %50) : (f32, memref<12x12xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 2 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 4 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}], function_type = (memref<12x12xf32>, index, i32, memref<12x12xf32>) -> (), sym_name = "au_generated_candidate_9__direct"}> ({
  ^bb0(%arg0: memref<12x12xf32>, %arg1: index, %arg2: i32, %arg3: memref<12x12xf32>):
    %0 = "arith.constant"() <{value = true}> : () -> i1
    %1 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg4: index):
      %4 = "arith.index_cast"(%arg4) : (index) -> i32
      %5 = "arith.index_cast"(%arg1) : (index) -> i32
      %6 = "arith.index_cast"(%arg4) : (index) -> i32
      %7 = "aps.read_smem"(%arg0, %5, %6) : (memref<12x12xf32>, i32, i32) -> f32
      %8 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%7, %8) : (f32, memref<f32>) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg5: index):
        %12 = "arith.index_cast"(%arg5) : (index) -> i32
        %13 = "arith.addi"(%12, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg6: index):
          %14 = "arith.index_cast"(%arg6) : (index) -> i32
          %15 = "arith.addi"(%14, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %16 = "arith.addi"(%arg2, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %17 = "arith.cmpi"(%16, %2) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %18 = "arith.cmpi"(%16, %1) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %19 = "arith.andi"(%17, %18) : (i1, i1) -> i1
          %20 = "arith.xori"(%19, %0) : (i1, i1) -> i1
          %21 = "arith.andi"(%20, %17) : (i1, i1) -> i1
          %22 = "arith.select"(%19, %1, %2) : (i1, i32, i32) -> i32
          %23 = "arith.select"(%21, %16, %22) : (i1, i32, i32) -> i32
          %24 = "arith.addi"(%4, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %25 = "arith.cmpi"(%24, %2) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %26 = "arith.cmpi"(%24, %1) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %27 = "arith.andi"(%25, %26) : (i1, i1) -> i1
          %28 = "arith.xori"(%27, %0) : (i1, i1) -> i1
          %29 = "arith.andi"(%28, %25) : (i1, i1) -> i1
          %30 = "arith.select"(%27, %1, %2) : (i1, i32, i32) -> i32
          %31 = "arith.select"(%29, %24, %30) : (i1, i32, i32) -> i32
          %32 = "arith.index_cast"(%23) : (i32) -> index
          %33 = "arith.index_cast"(%31) : (i32) -> index
          %34 = "arith.index_cast"(%32) : (index) -> i32
          %35 = "arith.index_cast"(%33) : (index) -> i32
          %36 = "aps.read_smem"(%arg0, %34, %35) : (memref<12x12xf32>, i32, i32) -> f32
          %37 = "memref.load"(%8) : (memref<f32>) -> f32
          %38 = "arith.cmpf"(%36, %37) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          "scf.if"(%38) ({
            "memref.store"(%36, %8) : (f32, memref<f32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %9 = "memref.load"(%8) : (memref<f32>) -> f32
      %10 = "arith.index_cast"(%arg1) : (index) -> i32
      %11 = "arith.index_cast"(%arg4) : (index) -> i32
      "aps.write_smem"(%9, %arg3, %10, %11) : (f32, memref<12x12xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 2 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 4 : i32} : () -> ()
}) : () -> ()

