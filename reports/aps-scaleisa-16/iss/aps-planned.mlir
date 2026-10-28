#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (5)>
#map2 = affine_map<() -> (16)>
#map3 = affine_map<() -> (3)>
"builtin.module"() ({
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}], function_type = (memref<16x16x3xf32>, index, i32, memref<16x16xf32>) -> (), sym_name = "au_generated_candidate_10__direct"}> ({
  ^bb0(%arg12: memref<16x16x3xf32>, %arg13: index, %arg14: i32, %arg15: memref<16x16xf32>):
    %62 = "arith.constant"() <{value = 2 : i32}> : () -> i32
    %63 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %64 = "arith.constant"() <{value = 5.000000e-01 : f32}> : () -> f32
    %65 = "arith.constant"() <{value = 2 : index}> : () -> index
    %66 = "arith.constant"() <{value = 1 : index}> : () -> index
    %67 = "arith.constant"() <{value = 0 : index}> : () -> index
    %68 = "arith.constant"() <{value = true}> : () -> i1
    %69 = "arith.constant"() <{value = 15 : i32}> : () -> i32
    %70 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %71 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %72 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg16: index):
      %73 = "arith.index_cast"(%arg16) : (index) -> i32
      %74 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%72, %74) : (f32, memref<f32>) -> ()
      %75 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%72, %75) : (f32, memref<f32>) -> ()
      %76 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%72, %76) : (f32, memref<f32>) -> ()
      %77 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%72, %77) : (f32, memref<f32>) -> ()
      %78 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%72, %78) : (f32, memref<f32>) -> ()
      %79 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%72, %79) : (f32, memref<f32>) -> ()
      %80 = "arith.index_cast"(%arg13) : (index) -> i32
      %81 = "arith.index_cast"(%arg16) : (index) -> i32
      %82 = "aps.read_smem"(%arg12, %80, %81, %70) : (memref<16x16x3xf32>, i32, i32, i32) -> f32
      %83 = "arith.index_cast"(%arg13) : (index) -> i32
      %84 = "arith.index_cast"(%arg16) : (index) -> i32
      %85 = "aps.read_smem"(%arg12, %83, %84, %63) : (memref<16x16x3xf32>, i32, i32, i32) -> f32
      %86 = "arith.index_cast"(%arg13) : (index) -> i32
      %87 = "arith.index_cast"(%arg16) : (index) -> i32
      %88 = "aps.read_smem"(%arg12, %86, %87, %62) : (memref<16x16x3xf32>, i32, i32, i32) -> f32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg17: index):
        %112 = "arith.index_cast"(%arg17) : (index) -> i32
        %113 = "arith.addi"(%112, %71) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
        ^bb0(%arg18: index):
          %114 = "arith.index_cast"(%arg18) : (index) -> i32
          %115 = "arith.addi"(%114, %71) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %116 = "arith.addi"(%arg14, %113) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %117 = "arith.cmpi"(%116, %70) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %118 = "arith.cmpi"(%116, %69) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %119 = "arith.andi"(%117, %118) : (i1, i1) -> i1
          %120 = "arith.xori"(%119, %68) : (i1, i1) -> i1
          %121 = "arith.andi"(%120, %117) : (i1, i1) -> i1
          %122 = "arith.select"(%119, %69, %70) : (i1, i32, i32) -> i32
          %123 = "arith.select"(%121, %116, %122) : (i1, i32, i32) -> i32
          %124 = "arith.addi"(%73, %115) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %125 = "arith.cmpi"(%124, %70) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %126 = "arith.cmpi"(%124, %69) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %127 = "arith.andi"(%125, %126) : (i1, i1) -> i1
          %128 = "arith.subi"(%68, %127) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
          %129 = "arith.andi"(%128, %125) : (i1, i1) -> i1
          %130 = "arith.select"(%127, %69, %70) : (i1, i32, i32) -> i32
          %131 = "arith.select"(%129, %124, %130) : (i1, i32, i32) -> i32
          %132 = "arith.index_cast"(%123) : (i32) -> index
          %133 = "arith.index_cast"(%131) : (i32) -> index
          %134 = "arith.index_cast"(%132) : (index) -> i32
          %135 = "arith.index_cast"(%133) : (index) -> i32
          %136 = "arith.index_cast"(%67) : (index) -> i32
          %137 = "aps.read_smem"(%arg12, %134, %135, %136) : (memref<16x16x3xf32>, i32, i32, i32) -> f32
          %138 = "arith.subf"(%137, %82) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %139 = "arith.index_cast"(%132) : (index) -> i32
          %140 = "arith.index_cast"(%133) : (index) -> i32
          %141 = "arith.index_cast"(%66) : (index) -> i32
          %142 = "aps.read_smem"(%arg12, %139, %140, %141) : (memref<16x16x3xf32>, i32, i32, i32) -> f32
          %143 = "arith.subf"(%142, %85) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %144 = "arith.index_cast"(%132) : (index) -> i32
          %145 = "arith.index_cast"(%133) : (index) -> i32
          %146 = "arith.index_cast"(%65) : (index) -> i32
          %147 = "aps.read_smem"(%arg12, %144, %145, %146) : (memref<16x16x3xf32>, i32, i32, i32) -> f32
          %148 = "arith.subf"(%147, %88) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %149 = "arith.mulf"(%138, %138) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %150 = "memref.load"(%74) : (memref<f32>) -> f32
          %151 = "arith.addf"(%150, %149) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%151, %74) : (f32, memref<f32>) -> ()
          %152 = "arith.mulf"(%143, %143) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %153 = "memref.load"(%75) : (memref<f32>) -> f32
          %154 = "arith.addf"(%153, %152) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%154, %75) : (f32, memref<f32>) -> ()
          %155 = "arith.mulf"(%148, %148) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %156 = "memref.load"(%76) : (memref<f32>) -> f32
          %157 = "arith.addf"(%156, %155) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%157, %76) : (f32, memref<f32>) -> ()
          %158 = "arith.mulf"(%138, %143) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %159 = "memref.load"(%77) : (memref<f32>) -> f32
          %160 = "arith.addf"(%159, %158) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%160, %77) : (f32, memref<f32>) -> ()
          %161 = "arith.mulf"(%138, %148) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %162 = "memref.load"(%78) : (memref<f32>) -> f32
          %163 = "arith.addf"(%162, %161) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%163, %78) : (f32, memref<f32>) -> ()
          %164 = "arith.mulf"(%143, %148) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %165 = "memref.load"(%79) : (memref<f32>) -> f32
          %166 = "arith.addf"(%165, %164) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%166, %79) : (f32, memref<f32>) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %89 = "memref.load"(%74) : (memref<f32>) -> f32
      %90 = "memref.load"(%75) : (memref<f32>) -> f32
      %91 = "arith.addf"(%89, %90) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %92 = "memref.load"(%76) : (memref<f32>) -> f32
      %93 = "arith.addf"(%91, %92) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %94 = "memref.load"(%77) : (memref<f32>) -> f32
      %95 = "arith.cmpf"(%94, %72) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %96 = "scf.if"(%95) ({
        %111 = "arith.negf"(%94) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%111) : (f32) -> ()
      }, {
        "scf.yield"(%94) : (f32) -> ()
      }) : (i1) -> f32
      %97 = "memref.load"(%78) : (memref<f32>) -> f32
      %98 = "arith.cmpf"(%97, %72) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %99 = "scf.if"(%98) ({
        %110 = "arith.negf"(%97) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%110) : (f32) -> ()
      }, {
        "scf.yield"(%97) : (f32) -> ()
      }) : (i1) -> f32
      %100 = "arith.addf"(%96, %99) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %101 = "memref.load"(%79) : (memref<f32>) -> f32
      %102 = "arith.cmpf"(%101, %72) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %103 = "scf.if"(%102) ({
        %109 = "arith.negf"(%101) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%109) : (f32) -> ()
      }, {
        "scf.yield"(%101) : (f32) -> ()
      }) : (i1) -> f32
      %104 = "arith.addf"(%100, %103) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %105 = "arith.mulf"(%104, %64) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %106 = "arith.subf"(%93, %105) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %107 = "arith.index_cast"(%arg13) : (index) -> i32
      %108 = "arith.index_cast"(%arg16) : (index) -> i32
      "aps.write_smem"(%106, %arg15, %107, %108) : (f32, memref<16x16xf32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 2 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 4 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {}, {}, {scaleisa.noalias}], function_type = (memref<16x16xf32>, index, i32, memref<16x16xi32>) -> (), sym_name = "au_generated_candidate_24__direct"}> ({
  ^bb0(%arg5: memref<16x16xf32>, %arg6: index, %arg7: i32, %arg8: memref<16x16xi32>):
    %18 = "arith.constant"() <{value = true}> : () -> i1
    %19 = "arith.constant"() <{value = 15 : i32}> : () -> i32
    %20 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %21 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %22 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg9: index):
      %23 = "arith.index_cast"(%arg9) : (index) -> i32
      %24 = "arith.index_cast"(%arg6) : (index) -> i32
      %25 = "arith.index_cast"(%arg9) : (index) -> i32
      %26 = "aps.read_smem"(%arg5, %24, %25) : (memref<16x16xf32>, i32, i32) -> f32
      %27 = "arith.cmpf"(%26, %22) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
      %28 = "arith.extui"(%27) : (i1) -> i32
      %29 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "memref.store"(%28, %29) : (i32, memref<i32>) -> ()
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
      ^bb0(%arg10: index):
        %33 = "arith.index_cast"(%arg10) : (index) -> i32
        %34 = "arith.addi"(%33, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map3}> ({
        ^bb0(%arg11: index):
          %35 = "arith.index_cast"(%arg11) : (index) -> i32
          %36 = "arith.addi"(%35, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %37 = "arith.addi"(%arg7, %34) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %38 = "arith.cmpi"(%37, %20) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %39 = "arith.cmpi"(%37, %19) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %40 = "arith.andi"(%38, %39) : (i1, i1) -> i1
          %41 = "arith.xori"(%40, %18) : (i1, i1) -> i1
          %42 = "arith.andi"(%41, %38) : (i1, i1) -> i1
          %43 = "arith.select"(%40, %19, %20) : (i1, i32, i32) -> i32
          %44 = "arith.select"(%42, %37, %43) : (i1, i32, i32) -> i32
          %45 = "arith.addi"(%23, %36) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %46 = "arith.cmpi"(%45, %20) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %47 = "arith.cmpi"(%45, %19) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %48 = "arith.andi"(%46, %47) : (i1, i1) -> i1
          %49 = "arith.xori"(%48, %18) : (i1, i1) -> i1
          %50 = "arith.andi"(%49, %46) : (i1, i1) -> i1
          %51 = "arith.select"(%48, %19, %20) : (i1, i32, i32) -> i32
          %52 = "arith.select"(%50, %45, %51) : (i1, i32, i32) -> i32
          %53 = "arith.cmpi"(%34, %20) <{predicate = 1 : i64}> : (i32, i32) -> i1
          %54 = "scf.if"(%53) ({
            "scf.yield"(%18) : (i1) -> ()
          }, {
            %61 = "arith.cmpi"(%36, %20) <{predicate = 1 : i64}> : (i32, i32) -> i1
            "scf.yield"(%61) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%54) ({
            %55 = "arith.index_cast"(%44) : (i32) -> index
            %56 = "arith.index_cast"(%52) : (i32) -> index
            %57 = "arith.index_cast"(%55) : (index) -> i32
            %58 = "arith.index_cast"(%56) : (index) -> i32
            %59 = "aps.read_smem"(%arg5, %57, %58) : (memref<16x16xf32>, i32, i32) -> f32
            %60 = "arith.cmpf"(%59, %26) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
            "scf.if"(%60) ({
              "memref.store"(%20, %29) : (i32, memref<i32>) -> ()
              "scf.yield"() : () -> ()
            }, {
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      %30 = "memref.load"(%29) : (memref<i32>) -> i32
      %31 = "arith.index_cast"(%arg6) : (index) -> i32
      %32 = "arith.index_cast"(%arg9) : (index) -> i32
      "aps.write_smem"(%30, %arg8, %31, %32) : (i32, memref<16x16xi32>, i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 2 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 4 : i32} : () -> ()
  "func.func"() <{arg_attrs = [{scaleisa.noalias}, {scaleisa.noalias}, {scaleisa.noalias}], function_type = (memref<16x16xi32>, memref<16x16xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_27__direct"}> ({
  ^bb0(%arg0: memref<16x16xi32>, %arg1: memref<16x16xf32>, %arg2: memref<i32>):
    %0 = "arith.constant"() <{value = 3.200000e+01 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg3: index):
      %2 = "arith.index_cast"(%arg3) : (index) -> i32
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
      ^bb0(%arg4: index):
        %3 = "arith.index_cast"(%arg4) : (index) -> i32
        %4 = "arith.index_cast"(%arg3) : (index) -> i32
        %5 = "arith.index_cast"(%arg4) : (index) -> i32
        %6 = "aps.read_smem"(%arg0, %4, %5) : (memref<16x16xi32>, i32, i32) -> i32
        %7 = "arith.cmpi"(%6, %1) <{predicate = 1 : i64}> : (i32, i32) -> i1
        %8 = "scf.if"(%7) ({
          %11 = "arith.index_cast"(%arg3) : (index) -> i32
          %12 = "arith.index_cast"(%arg4) : (index) -> i32
          %13 = "aps.read_smem"(%arg1, %11, %12) : (memref<16x16xf32>, i32, i32) -> f32
          %14 = "arith.mulf"(%13, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %15 = "arith.fptosi"(%14) : (f32) -> i32
          %16 = "arith.addi"(%15, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %17 = "arith.addi"(%16, %2) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "scf.yield"(%17) : (i32) -> ()
        }, {
          "scf.yield"(%1) : (i32) -> ()
        }) : (i1) -> i32
        %9 = "memref.load"(%arg2) : (memref<i32>) -> i32
        %10 = "arith.addi"(%9, %8) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%10, %arg2) : (i32, memref<i32>) -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "func.return"() : () -> ()
  }) {au.generated = true, scaleisa.converted, scaleisa.csr_count = 1 : i32, scaleisa.instruction, scaleisa.max_extra_csrs = 5 : i32, scaleisa.operand_count = 3 : i32} : () -> ()
}) : () -> ()

