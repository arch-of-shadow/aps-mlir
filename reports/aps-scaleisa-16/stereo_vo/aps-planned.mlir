#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_17__direct"}> ({
  ^bb0(%arg4: memref<32x3xf32>, %arg5: memref<32x3xf32>, %arg6: memref<32x3xf32>):
    %22 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %23 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %24 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %25 = "arith.constant"() <{value = 1.200000e-01 : f32}> : () -> f32
    %26 = "arith.constant"() <{value = 1.600000e-01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg7: index):
      %27 = "arith.index_cast"(%arg7) : (index) -> i32
      %28 = "aps.read_smem"(%arg4, %27, %24) : (memref<32x3xf32>, i32, i32) -> f32
      %29 = "arith.mulf"(%28, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %30 = "arith.index_cast"(%arg7) : (index) -> i32
      %31 = "aps.read_smem"(%arg5, %30, %24) : (memref<32x3xf32>, i32, i32) -> f32
      %32 = "arith.divf"(%29, %31) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %33 = "arith.index_cast"(%arg7) : (index) -> i32
      %34 = "aps.read_smem"(%arg6, %33, %24) : (memref<32x3xf32>, i32, i32) -> f32
      %35 = "arith.subf"(%34, %32) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %36 = "arith.index_cast"(%arg7) : (index) -> i32
      "aps.write_smem"(%35, %arg6, %36, %24) : (f32, memref<32x3xf32>, i32, i32) -> ()
      %37 = "arith.index_cast"(%arg7) : (index) -> i32
      %38 = "aps.read_smem"(%arg4, %37, %23) : (memref<32x3xf32>, i32, i32) -> f32
      %39 = "arith.mulf"(%38, %26) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %40 = "arith.index_cast"(%arg7) : (index) -> i32
      %41 = "aps.read_smem"(%arg5, %40, %23) : (memref<32x3xf32>, i32, i32) -> f32
      %42 = "arith.divf"(%39, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %43 = "arith.index_cast"(%arg7) : (index) -> i32
      %44 = "aps.read_smem"(%arg6, %43, %23) : (memref<32x3xf32>, i32, i32) -> f32
      %45 = "arith.subf"(%44, %42) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %46 = "arith.index_cast"(%arg7) : (index) -> i32
      "aps.write_smem"(%45, %arg6, %46, %23) : (f32, memref<32x3xf32>, i32, i32) -> ()
      %47 = "arith.index_cast"(%arg7) : (index) -> i32
      %48 = "aps.read_smem"(%arg4, %47, %22) : (memref<32x3xf32>, i32, i32) -> f32
      %49 = "arith.mulf"(%48, %25) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %50 = "arith.index_cast"(%arg7) : (index) -> i32
      %51 = "aps.read_smem"(%arg5, %50, %22) : (memref<32x3xf32>, i32, i32) -> f32
      %52 = "arith.divf"(%49, %51) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %53 = "arith.index_cast"(%arg7) : (index) -> i32
      %54 = "aps.read_smem"(%arg6, %53, %22) : (memref<32x3xf32>, i32, i32) -> f32
      %55 = "arith.subf"(%54, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %56 = "arith.index_cast"(%arg7) : (index) -> i32
      "aps.write_smem"(%55, %arg6, %56, %22) : (f32, memref<32x3xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<32x3xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_26__direct"}> ({
  ^bb0(%arg0: memref<32x3xf32>, %arg1: memref<i32>):
    %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 1.030000e+02 : f32}> : () -> f32
    %5 = "arith.constant"() <{value = 7.900000e+01 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 32 : index, upperBoundMap = #map1}> ({
    ^bb0(%arg2: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg3: index):
        %6 = "arith.index_cast"(%arg3) : (index) -> i32
        %7 = "aps.read_smem"(%arg0, %6, %2) : (memref<32x3xf32>, i32, i32) -> f32
        %8 = "arith.mulf"(%7, %5) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %9 = "arith.fptosi"(%8) : (f32) -> i32
        %10 = "memref.load"(%arg1) : (memref<i32>) -> i32
        %11 = "arith.addi"(%10, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%11, %arg1) : (i32, memref<i32>) -> ()
        %12 = "arith.index_cast"(%arg3) : (index) -> i32
        %13 = "aps.read_smem"(%arg0, %12, %1) : (memref<32x3xf32>, i32, i32) -> f32
        %14 = "arith.mulf"(%13, %4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %15 = "arith.fptosi"(%14) : (f32) -> i32
        %16 = "arith.addi"(%11, %15) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%16, %arg1) : (i32, memref<i32>) -> ()
        %17 = "arith.index_cast"(%arg3) : (index) -> i32
        %18 = "aps.read_smem"(%arg0, %17, %0) : (memref<32x3xf32>, i32, i32) -> f32
        %19 = "arith.mulf"(%18, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %20 = "arith.fptosi"(%19) : (f32) -> i32
        %21 = "arith.addi"(%16, %20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%21, %arg1) : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
}) : () -> ()

