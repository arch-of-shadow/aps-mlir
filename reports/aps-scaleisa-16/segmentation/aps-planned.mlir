#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (3)>
#map2 = affine_map<() -> (8)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x8xi32>, index, i32, memref<8x8xf32>, memref<8x8xf32>) -> (), sym_name = "au_generated_candidate_0__direct"}> ({
  ^bb0(%arg13: memref<8x8xi32>, %arg14: index, %arg15: i32, %arg16: memref<8x8xf32>, %arg17: memref<8x8xf32>):
    %108 = "arith.constant"() <{value = false}> : () -> i1
    %109 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %110 = "arith.constant"() <{value = true}> : () -> i1
    %111 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %112 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %113 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %114 = "arith.constant"() <{value = 1.000000e+30 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg18: index):
      %115 = "arith.index_cast"(%arg18) : (index) -> i32
      %116 = "arith.index_cast"(%arg14) : (index) -> i32
      %117 = "arith.index_cast"(%arg18) : (index) -> i32
      %118 = "aps.read_smem"(%arg13, %116, %117) : (memref<8x8xi32>, i32, i32) -> i32
      %119 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "memref.store"(%118, %119) : (i32, memref<i32>) -> ()
      %120 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%114, %120) : (f32, memref<f32>) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg19: index):
        %124 = "arith.index_cast"(%arg19) : (index) -> i32
        %125 = "arith.addi"(%124, %113) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg20: index):
          %126 = "arith.index_cast"(%arg20) : (index) -> i32
          %127 = "arith.addi"(%126, %113) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %128 = "arith.addi"(%arg15, %125) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %129 = "arith.cmpi"(%128, %112) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %130 = "arith.cmpi"(%128, %111) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %131 = "arith.andi"(%129, %130) : (i1, i1) -> i1
          %132 = "arith.xori"(%131, %110) : (i1, i1) -> i1
          %133 = "arith.andi"(%132, %129) : (i1, i1) -> i1
          %134 = "arith.select"(%131, %111, %112) : (i1, i32, i32) -> i32
          %135 = "arith.select"(%133, %128, %134) : (i1, i32, i32) -> i32
          %136 = "arith.addi"(%115, %127) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %137 = "arith.cmpi"(%136, %112) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %138 = "arith.cmpi"(%136, %111) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %139 = "arith.andi"(%137, %138) : (i1, i1) -> i1
          %140 = "arith.xori"(%139, %110) : (i1, i1) -> i1
          %141 = "arith.andi"(%140, %137) : (i1, i1) -> i1
          %142 = "arith.select"(%139, %111, %112) : (i1, i32, i32) -> i32
          %143 = "arith.select"(%141, %136, %142) : (i1, i32, i32) -> i32
          %144 = "arith.index_cast"(%arg14) : (index) -> i32
          %145 = "arith.index_cast"(%arg18) : (index) -> i32
          %146 = "aps.read_smem"(%arg16, %144, %145) : (memref<8x8xf32>, i32, i32) -> f32
          %147 = "arith.index_cast"(%135) : (i32) -> index
          %148 = "arith.index_cast"(%143) : (i32) -> index
          %149 = "arith.index_cast"(%147) : (index) -> i32
          %150 = "arith.index_cast"(%148) : (index) -> i32
          %151 = "aps.read_smem"(%arg16, %149, %150) : (memref<8x8xf32>, i32, i32) -> f32
          %152 = "arith.subf"(%146, %151) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %153 = "arith.cmpf"(%152, %109) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %154 = "scf.if"(%153) ({
            %172 = "arith.negf"(%152) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
            "scf.yield"(%172) : (f32) -> ()
          }, {
            "scf.yield"(%152) : (f32) -> ()
          }) : (i1) -> f32
          %155 = "arith.index_cast"(%147) : (index) -> i32
          %156 = "arith.index_cast"(%148) : (index) -> i32
          %157 = "aps.read_smem"(%arg17, %155, %156) : (memref<8x8xf32>, i32, i32) -> f32
          %158 = "arith.addf"(%154, %157) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %159 = "memref.load"(%120) : (memref<f32>) -> f32
          %160 = "arith.cmpf"(%158, %159) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
          %161 = "scf.if"(%160) ({
            "scf.yield"(%110) : (i1) -> ()
          }, {
            %165 = "arith.cmpf"(%158, %159) <{fastmath = #arith.fastmath<none>, predicate = 1 : i64}> : (f32, f32) -> i1
            %166 = "scf.if"(%165) ({
              %167 = "arith.index_cast"(%147) : (index) -> i32
              %168 = "arith.index_cast"(%148) : (index) -> i32
              %169 = "aps.read_smem"(%arg13, %167, %168) : (memref<8x8xi32>, i32, i32) -> i32
              %170 = "memref.load"(%119) : (memref<i32>) -> i32
              %171 = "arith.cmpi"(%169, %170) <{predicate = 2 : i64}> : (i32, i32) -> i1
              "scf.yield"(%171) : (i1) -> ()
            }, {
              "scf.yield"(%108) : (i1) -> ()
            }) : (i1) -> i1
            "scf.yield"(%166) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%161) ({
            "memref.store"(%158, %120) : (f32, memref<f32>) -> ()
            %162 = "arith.index_cast"(%147) : (index) -> i32
            %163 = "arith.index_cast"(%148) : (index) -> i32
            %164 = "aps.read_smem"(%arg13, %162, %163) : (memref<8x8xi32>, i32, i32) -> i32
            "memref.store"(%164, %119) : (i32, memref<i32>) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %121 = "memref.load"(%119) : (memref<i32>) -> i32
      %122 = "arith.index_cast"(%arg14) : (index) -> i32
      %123 = "arith.index_cast"(%arg18) : (index) -> i32
      "aps.write_smem"(%121, %arg13, %122, %123) : (i32, memref<8x8xi32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 3 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 5 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x8xf32>, memref<8x8xf32>) -> (), sym_name = "au_generated_candidate_10__direct"}> ({
  ^bb0(%arg9: memref<8x8xf32>, %arg10: memref<8x8xf32>):
    %41 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %42 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %43 = "arith.constant"() <{value = true}> : () -> i1
    %44 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %45 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %46 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg11: index):
      %47 = "arith.index_cast"(%arg11) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg12: index):
        %48 = "arith.index_cast"(%arg12) : (index) -> i32
        %49 = "arith.addi"(%48, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %50 = "arith.cmpi"(%49, %45) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %51 = "arith.cmpi"(%49, %44) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %52 = "arith.andi"(%50, %51) : (i1, i1) -> i1
        %53 = "arith.xori"(%52, %43) : (i1, i1) -> i1
        %54 = "arith.andi"(%53, %50) : (i1, i1) -> i1
        %55 = "arith.select"(%52, %44, %45) : (i1, i32, i32) -> i32
        %56 = "arith.select"(%54, %49, %55) : (i1, i32, i32) -> i32
        %57 = "arith.addi"(%48, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %58 = "arith.cmpi"(%57, %45) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %59 = "arith.cmpi"(%57, %44) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %60 = "arith.andi"(%58, %59) : (i1, i1) -> i1
        %61 = "arith.xori"(%60, %43) : (i1, i1) -> i1
        %62 = "arith.andi"(%61, %58) : (i1, i1) -> i1
        %63 = "arith.select"(%60, %44, %45) : (i1, i32, i32) -> i32
        %64 = "arith.select"(%62, %57, %63) : (i1, i32, i32) -> i32
        %65 = "arith.addi"(%47, %46) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %66 = "arith.cmpi"(%65, %45) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %67 = "arith.cmpi"(%65, %44) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %68 = "arith.andi"(%66, %67) : (i1, i1) -> i1
        %69 = "arith.xori"(%68, %43) : (i1, i1) -> i1
        %70 = "arith.andi"(%69, %66) : (i1, i1) -> i1
        %71 = "arith.select"(%68, %44, %45) : (i1, i32, i32) -> i32
        %72 = "arith.select"(%70, %65, %71) : (i1, i32, i32) -> i32
        %73 = "arith.addi"(%47, %42) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %74 = "arith.cmpi"(%73, %45) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %75 = "arith.cmpi"(%73, %44) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %76 = "arith.andi"(%74, %75) : (i1, i1) -> i1
        %77 = "arith.subi"(%43, %76) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
        %78 = "arith.andi"(%77, %74) : (i1, i1) -> i1
        %79 = "arith.select"(%76, %44, %45) : (i1, i32, i32) -> i32
        %80 = "arith.select"(%78, %73, %79) : (i1, i32, i32) -> i32
        %81 = "arith.index_cast"(%64) : (i32) -> index
        %82 = "arith.index_cast"(%arg11) : (index) -> i32
        %83 = "arith.index_cast"(%81) : (index) -> i32
        %84 = "aps.read_smem"(%arg9, %82, %83) : (memref<8x8xf32>, i32, i32) -> f32
        %85 = "arith.index_cast"(%56) : (i32) -> index
        %86 = "arith.index_cast"(%arg11) : (index) -> i32
        %87 = "arith.index_cast"(%85) : (index) -> i32
        %88 = "aps.read_smem"(%arg9, %86, %87) : (memref<8x8xf32>, i32, i32) -> f32
        %89 = "arith.subf"(%84, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %90 = "arith.cmpf"(%89, %41) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %91 = "scf.if"(%90) ({
          %107 = "arith.negf"(%89) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%107) : (f32) -> ()
        }, {
          "scf.yield"(%89) : (f32) -> ()
        }) : (i1) -> f32
        %92 = "arith.index_cast"(%80) : (i32) -> index
        %93 = "arith.index_cast"(%92) : (index) -> i32
        %94 = "arith.index_cast"(%arg12) : (index) -> i32
        %95 = "aps.read_smem"(%arg9, %93, %94) : (memref<8x8xf32>, i32, i32) -> f32
        %96 = "arith.index_cast"(%72) : (i32) -> index
        %97 = "arith.index_cast"(%96) : (index) -> i32
        %98 = "arith.index_cast"(%arg12) : (index) -> i32
        %99 = "aps.read_smem"(%arg9, %97, %98) : (memref<8x8xf32>, i32, i32) -> f32
        %100 = "arith.subf"(%95, %99) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %101 = "arith.cmpf"(%100, %41) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
        %102 = "scf.if"(%101) ({
          %106 = "arith.negf"(%100) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
          "scf.yield"(%106) : (f32) -> ()
        }, {
          "scf.yield"(%100) : (f32) -> ()
        }) : (i1) -> f32
        %103 = "arith.addf"(%91, %102) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %104 = "arith.index_cast"(%arg11) : (index) -> i32
        %105 = "arith.index_cast"(%arg12) : (index) -> i32
        "aps.write_smem"(%103, %arg10, %104, %105) : (f32, memref<8x8xf32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x8xf32>, memref<8x8xi32>) -> (), sym_name = "au_generated_candidate_21__l0u_l1u"}> ({
  ^bb0(%arg5: memref<8x8xf32>, %arg6: memref<8x8xi32>):
    %7 = "arith.constant"() <{value = 8 : i32}> : () -> i32
    %8 = "arith.constant"() <{value = 5.000000e-03 : f32}> : () -> f32
    %9 = "arith.constant"() <{value = 9 : i32}> : () -> i32
    %10 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %11 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %12 = "arith.constant"() <{value = 1.000000e-01 : f32}> : () -> f32
    %13 = "arith.constant"() <{value = 30 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = 56 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 24 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg7: index):
      %16 = "arith.index_cast"(%arg7) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg8: index):
        %17 = "arith.index_cast"(%arg8) : (index) -> i32
        %18 = "arith.cmpi"(%17, %15) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %19 = "arith.extui"(%18) : (i1) -> i32
        %20 = "arith.cmpi"(%17, %14) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %21 = "arith.extui"(%20) : (i1) -> i32
        %22 = "arith.addi"(%19, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %23 = "arith.cmpi"(%16, %13) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %24 = "arith.extui"(%23) : (i1) -> i32
        %25 = "arith.addi"(%22, %24) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %26 = "arith.sitofp"(%25) : (i32) -> f32
        %27 = "arith.mulf"(%26, %12) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %28 = "arith.muli"(%17, %11) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %29 = "arith.muli"(%16, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %30 = "arith.addi"(%28, %29) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %31 = "arith.remsi"(%30, %9) : (i32, i32) -> i32
        %32 = "arith.sitofp"(%31) : (i32) -> f32
        %33 = "arith.mulf"(%32, %8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %34 = "arith.addf"(%27, %33) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %35 = "arith.index_cast"(%arg7) : (index) -> i32
        %36 = "arith.index_cast"(%arg8) : (index) -> i32
        "aps.write_smem"(%34, %arg5, %35, %36) : (f32, memref<8x8xf32>, i32, i32) -> ()
        %37 = "arith.muli"(%16, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %38 = "arith.addi"(%37, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %39 = "arith.index_cast"(%arg7) : (index) -> i32
        %40 = "arith.index_cast"(%arg8) : (index) -> i32
        "aps.write_smem"(%38, %arg6, %39, %40) : (i32, memref<8x8xi32>, i32, i32) -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_cyclic_array_1 = [true], partition_dim_array_0 = [1], partition_dim_array_1 = [1], partition_factor_array_0 = [8], partition_factor_array_1 = [8], scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<8x8xi32>, memref<i32>) -> (), sym_name = "au_generated_candidate_35__l0u_l1u_l2u"}> ({
  ^bb0(%arg0: memref<8x8xi32>, %arg1: memref<i32>):
    %0 = "arith.constant"() <{value = 255 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg2: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 8 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg3: index):
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
        ^bb0(%arg4: index):
          %1 = "arith.index_cast"(%arg2) : (index) -> i32
          %2 = "arith.index_cast"(%arg4) : (index) -> i32
          %3 = "aps.read_smem"(%arg0, %1, %2) : (memref<8x8xi32>, i32, i32) -> i32
          %4 = "arith.andi"(%3, %0) : (i32, i32) -> i32
          %5 = "memref.load"(%arg1) : (memref<i32>) -> i32
          %6 = "arith.addi"(%5, %4) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "memref.store"(%6, %arg1) : (i32, memref<i32>) -> ()
          "affine.yield"() : () -> ()
        }) {unroll = 0 : i64} : () -> ()
        "affine.yield"() : () -> ()
      }) {unroll = 0 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) {unroll = 0 : i64} : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [8], scaleisa.converted, scaleisa.csr_count = 0 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 8 : i32, scaleisa.operand_count = 2 : i32} : () -> ()
}) : () -> ()

