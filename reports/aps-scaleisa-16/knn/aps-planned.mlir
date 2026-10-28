#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (8)>
#map2 = affine_map<() -> (4)>
#map3 = affine_map<() -> (32)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<4x8xi32>, memref<4x8xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_11__direct"}> ({
  ^bb0(%arg16: memref<4x8xi32>, %arg17: memref<4x8xf32>, %arg18: memref<i32>):
    %128 = "arith.constant"() <{value = 2.560000e+02 : f32}> : () -> f32
    %129 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg19: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg20: index):
        %130 = "arith.index_cast"(%arg19) : (index) -> i32
        %131 = "arith.index_cast"(%arg20) : (index) -> i32
        %132 = "aps.read_smem"(%arg16, %130, %131) : (memref<4x8xi32>, i32, i32) -> i32
        %133 = "arith.cmpi"(%132, %129) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %134 = "scf.if"(%133) ({
          %137 = "arith.index_cast"(%arg19) : (index) -> i32
          %138 = "arith.index_cast"(%arg20) : (index) -> i32
          %139 = "aps.read_smem"(%arg16, %137, %138) : (memref<4x8xi32>, i32, i32) -> i32
          %140 = "arith.index_cast"(%arg19) : (index) -> i32
          %141 = "arith.index_cast"(%arg20) : (index) -> i32
          %142 = "aps.read_smem"(%arg17, %140, %141) : (memref<4x8xf32>, i32, i32) -> f32
          %143 = "arith.mulf"(%142, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %144 = "arith.fptosi"(%143) : (f32) -> i32
          %145 = "arith.addi"(%139, %144) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "scf.yield"(%145) : (i32) -> ()
        }, {
          "scf.yield"(%129) : (i32) -> ()
        }) : (i1) -> i32
        %135 = "memref.load"(%arg18) : (memref<i32>) -> i32
        %136 = "arith.addi"(%135, %134) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%136, %arg18) : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<4x8xf32>, memref<4x8xi32>, memref<32x3xf32>, memref<4x3xf32>, memref<8xf32>, memref<8xi32>) -> (), sym_name = "au_generated_candidate_2__direct"}> ({
  ^bb0(%arg4: memref<4x8xf32>, %arg5: memref<4x8xi32>, %arg6: memref<32x3xf32>, %arg7: memref<4x3xf32>, %arg8: memref<8xf32>, %arg9: memref<8xi32>):
    %68 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %69 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %70 = "arith.constant"() <{value = false}> : () -> i1
    %71 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %72 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %73 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg10: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 8 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg14: index):
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg15: index):
          %124 = "arith.index_cast"(%arg10) : (index) -> i32
          %125 = "arith.index_cast"(%arg15) : (index) -> i32
          "aps.write_smem"(%73, %arg4, %124, %125) : (f32, memref<4x8xf32>, i32, i32) -> ()
          %126 = "arith.index_cast"(%arg10) : (index) -> i32
          %127 = "arith.index_cast"(%arg15) : (index) -> i32
          "aps.write_smem"(%72, %arg5, %126, %127) : (i32, memref<4x8xi32>, i32, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg11: index):
        %74 = "arith.index_cast"(%arg11) : (index) -> i32
        %75 = "arith.index_cast"(%arg11) : (index) -> i32
        %76 = "aps.read_smem"(%arg6, %75, %71) : (memref<32x3xf32>, i32, i32) -> f32
        %77 = "arith.index_cast"(%arg10) : (index) -> i32
        %78 = "aps.read_smem"(%arg7, %77, %71) : (memref<4x3xf32>, i32, i32) -> f32
        %79 = "arith.subf"(%76, %78) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %80 = "arith.index_cast"(%arg11) : (index) -> i32
        %81 = "aps.read_smem"(%arg6, %80, %69) : (memref<32x3xf32>, i32, i32) -> f32
        %82 = "arith.index_cast"(%arg10) : (index) -> i32
        %83 = "aps.read_smem"(%arg7, %82, %69) : (memref<4x3xf32>, i32, i32) -> f32
        %84 = "arith.subf"(%81, %83) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %85 = "arith.index_cast"(%arg11) : (index) -> i32
        %86 = "aps.read_smem"(%arg6, %85, %68) : (memref<32x3xf32>, i32, i32) -> f32
        %87 = "arith.index_cast"(%arg10) : (index) -> i32
        %88 = "aps.read_smem"(%arg7, %87, %68) : (memref<4x3xf32>, i32, i32) -> f32
        %89 = "arith.subf"(%86, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %90 = "arith.mulf"(%79, %79) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %91 = "arith.mulf"(%84, %84) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %92 = "arith.addf"(%90, %91) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %93 = "arith.mulf"(%89, %89) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %94 = "arith.addf"(%92, %93) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %95 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
        "memref.store"(%71, %95) : (i32, memref<i32>) -> ()
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg13: index):
          %104 = "arith.index_cast"(%arg13) : (index) -> i32
          %105 = "memref.load"(%95) : (memref<i32>) -> i32
          %106 = "arith.cmpi"(%105, %71) <{predicate = 0 : i64}> : (i32, i32) -> i1
          %107 = "scf.if"(%106) ({
            %120 = "arith.index_cast"(%arg10) : (index) -> i32
            %121 = "arith.index_cast"(%arg13) : (index) -> i32
            %122 = "aps.read_smem"(%arg4, %120, %121) : (memref<4x8xf32>, i32, i32) -> f32
            %123 = "arith.cmpf"(%94, %122) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
            "scf.yield"(%123) : (i1) -> ()
          }, {
            "scf.yield"(%70) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%107) ({
            %118 = "arith.index_cast"(%arg13) : (index) -> i32
            "aps.write_smem"(%94, %arg8, %118) : (f32, memref<8xf32>, i32) -> ()
            %119 = "arith.index_cast"(%arg13) : (index) -> i32
            "aps.write_smem"(%74, %arg9, %119) : (i32, memref<8xi32>, i32) -> ()
            "memref.store"(%69, %95) : (i32, memref<i32>) -> ()
            "scf.yield"() : () -> ()
          }, {
            %108 = "arith.subi"(%104, %105) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            %109 = "arith.index_cast"(%108) : (i32) -> index
            %110 = "arith.index_cast"(%arg10) : (index) -> i32
            %111 = "arith.index_cast"(%109) : (index) -> i32
            %112 = "aps.read_smem"(%arg4, %110, %111) : (memref<4x8xf32>, i32, i32) -> f32
            %113 = "arith.index_cast"(%arg13) : (index) -> i32
            "aps.write_smem"(%112, %arg8, %113) : (f32, memref<8xf32>, i32) -> ()
            %114 = "arith.index_cast"(%arg10) : (index) -> i32
            %115 = "arith.index_cast"(%109) : (index) -> i32
            %116 = "aps.read_smem"(%arg5, %114, %115) : (memref<4x8xi32>, i32, i32) -> i32
            %117 = "arith.index_cast"(%arg13) : (index) -> i32
            "aps.write_smem"(%116, %arg9, %117) : (i32, memref<8xi32>, i32) -> ()
            "scf.yield"() : () -> ()
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg12: index):
          %96 = "arith.index_cast"(%arg12) : (index) -> i32
          %97 = "aps.read_smem"(%arg8, %96) : (memref<8xf32>, i32) -> f32
          %98 = "arith.index_cast"(%arg10) : (index) -> i32
          %99 = "arith.index_cast"(%arg12) : (index) -> i32
          "aps.write_smem"(%97, %arg4, %98, %99) : (f32, memref<4x8xf32>, i32, i32) -> ()
          %100 = "arith.index_cast"(%arg12) : (index) -> i32
          %101 = "aps.read_smem"(%arg9, %100) : (memref<8xi32>, i32) -> i32
          %102 = "arith.index_cast"(%arg10) : (index) -> i32
          %103 = "arith.index_cast"(%arg12) : (index) -> i32
          "aps.write_smem"(%101, %arg5, %102, %103) : (i32, memref<4x8xi32>, i32, i32) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 4 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 6 : i32} : () -> ()
  "func.func"() <{function_type = (memref<32x3xf32>) -> (), sym_name = "au_generated_candidate_24__l0u"}> ({
  ^bb0(%arg2: memref<32x3xf32>):
    %34 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %35 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %36 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %37 = "arith.constant"() <{value = 2.700000e-02 : f32}> : () -> f32
    %38 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %39 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %40 = "arith.constant"() <{value = 19 : i32}> : () -> i32
    %41 = "arith.constant"() <{value = 2.900000e-02 : f32}> : () -> f32
    %42 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %43 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %44 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %45 = "arith.constant"() <{value = 3.100000e-02 : f32}> : () -> f32
    %46 = "arith.constant"() <{value = 101 : i32}> : () -> i32
    %47 = "arith.constant"() <{value = 3 : i32}> : () -> i32
    %48 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
    ^bb0(%arg3: index):
      %49 = "arith.index_cast"(%arg3) : (index) -> i32
      %50 = "arith.muli"(%49, %48) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %51 = "arith.addi"(%50, %47) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %52 = "arith.remsi"(%51, %46) : (i32, i32) -> i32
      %53 = "arith.sitofp"(%52) : (i32) -> f32
      %54 = "arith.mulf"(%53, %45) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %55 = "arith.index_cast"(%arg3) : (index) -> i32
      "aps.write_smem"(%54, %arg2, %55, %36) : (f32, memref<32x3xf32>, i32, i32) -> ()
      %56 = "arith.muli"(%49, %44) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %57 = "arith.addi"(%56, %43) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %58 = "arith.remsi"(%57, %42) : (i32, i32) -> i32
      %59 = "arith.sitofp"(%58) : (i32) -> f32
      %60 = "arith.mulf"(%59, %41) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %61 = "arith.index_cast"(%arg3) : (index) -> i32
      "aps.write_smem"(%60, %arg2, %61, %35) : (f32, memref<32x3xf32>, i32, i32) -> ()
      %62 = "arith.muli"(%49, %40) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %63 = "arith.addi"(%62, %39) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %64 = "arith.remsi"(%63, %38) : (i32, i32) -> i32
      %65 = "arith.sitofp"(%64) : (i32) -> f32
      %66 = "arith.mulf"(%65, %37) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %67 = "arith.index_cast"(%arg3) : (index) -> i32
      "aps.write_smem"(%66, %arg2, %67, %34) : (f32, memref<32x3xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 1 : i32} : () -> ()
  "func.func"() <{function_type = (memref<4x3xf32>) -> (), sym_name = "au_generated_candidate_25__l0u"}> ({
  ^bb0(%arg0: memref<4x3xf32>):
    %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 2.700000e-02 : f32}> : () -> f32
    %4 = "arith.constant"() <{value = 89 : i32}> : () -> i32
    %5 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %6 = "arith.constant"() <{value = 31 : i32}> : () -> i32
    %7 = "arith.constant"() <{value = 2.900000e-02 : f32}> : () -> f32
    %8 = "arith.constant"() <{value = 97 : i32}> : () -> i32
    %9 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 29 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 3.100000e-02 : f32}> : () -> f32
    %12 = "arith.constant"() <{value = 101 : i32}> : () -> i32
    %13 = "arith.constant"() <{value = 9 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 23 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg1: index):
      %15 = "arith.index_cast"(%arg1) : (index) -> i32
      %16 = "arith.muli"(%15, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %17 = "arith.addi"(%16, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %18 = "arith.remsi"(%17, %12) : (i32, i32) -> i32
      %19 = "arith.sitofp"(%18) : (i32) -> f32
      %20 = "arith.mulf"(%19, %11) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %21 = "arith.index_cast"(%arg1) : (index) -> i32
      "aps.write_smem"(%20, %arg0, %21, %2) : (f32, memref<4x3xf32>, i32, i32) -> ()
      %22 = "arith.muli"(%15, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %23 = "arith.addi"(%22, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %24 = "arith.remsi"(%23, %8) : (i32, i32) -> i32
      %25 = "arith.sitofp"(%24) : (i32) -> f32
      %26 = "arith.mulf"(%25, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %27 = "arith.index_cast"(%arg1) : (index) -> i32
      "aps.write_smem"(%26, %arg0, %27, %1) : (f32, memref<4x3xf32>, i32, i32) -> ()
      %28 = "arith.muli"(%15, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %29 = "arith.addi"(%28, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %30 = "arith.remsi"(%29, %4) : (i32, i32) -> i32
      %31 = "arith.sitofp"(%30) : (i32) -> f32
      %32 = "arith.mulf"(%31, %3) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %33 = "arith.index_cast"(%arg1) : (index) -> i32
      "aps.write_smem"(%32, %arg0, %33, %0) : (f32, memref<4x3xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 1 : i32} : () -> ()
}) : () -> ()

