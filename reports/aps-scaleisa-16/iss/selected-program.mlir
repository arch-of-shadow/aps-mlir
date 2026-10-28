"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %147 = "arith.constant"() <{value = 4 : index}> : () -> index
    %148 = "arith.constant"() <{value = 16 : index}> : () -> index
    %149 = "arith.constant"() <{value = 0 : index}> : () -> index
    %150 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %151 = "arith.constant"() <{value = 2 : index}> : () -> index
    %152 = "arith.constant"() <{value = 1 : index}> : () -> index
    %153 = "arith.constant"() <{value = 19 : i32}> : () -> i32
    %154 = "arith.constant"() <{value = 9 : i32}> : () -> i32
    %155 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %156 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    %157 = "arith.constant"() <{value = 3.000000e-02 : f32}> : () -> f32
    %158 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x16xi32>
    %159 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x16xf32>
    %160 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<16x16x3xf32>
    "scf.for"(%149, %148, %152) ({
    ^bb0(%arg21: index):
      %165 = "arith.index_cast"(%arg21) : (index) -> i32
      %166 = "arith.sitofp"(%165) : (i32) -> f32
      %167 = "arith.mulf"(%166, %157) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %168 = "arith.muli"(%165, %154) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "scf.for"(%149, %148, %147) ({
      ^bb0(%arg22: index):
        %169 = "arith.addi"(%arg22, %147) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        "scf.for"(%arg22, %169, %152) ({
        ^bb0(%arg23: index):
          %170 = "arith.index_cast"(%arg23) : (index) -> i32
          %171 = "arith.sitofp"(%170) : (i32) -> f32
          %172 = "arith.mulf"(%171, %157) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%172, %160, %arg21, %arg23, %149) : (f32, memref<16x16x3xf32>, index, index, index) -> ()
          "memref.store"(%167, %160, %arg21, %arg23, %152) : (f32, memref<16x16x3xf32>, index, index, index) -> ()
          %173 = "arith.muli"(%170, %155) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %174 = "arith.addi"(%173, %168) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %175 = "arith.xori"(%170, %165) : (i32, i32) -> i32
          %176 = "arith.addi"(%174, %175) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %177 = "arith.remsi"(%176, %153) : (i32, i32) -> i32
          %178 = "arith.sitofp"(%177) : (i32) -> f32
          %179 = "arith.mulf"(%178, %156) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%179, %160, %arg21, %arg23, %151) : (f32, memref<16x16x3xf32>, index, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%149, %148, %152) ({
    ^bb0(%arg20: index):
      %164 = "arith.index_cast"(%arg20) : (index) -> i32
      "func.call"(%160, %arg20, %164, %159) <{callee = @au_generated_candidate_10__direct_site_6842}> : (memref<16x16x3xf32>, index, i32, memref<16x16xf32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%149, %148, %152) ({
    ^bb0(%arg19: index):
      %163 = "arith.index_cast"(%arg19) : (index) -> i32
      "func.call"(%159, %arg19, %163, %158) <{callee = @au_generated_candidate_24__direct_site_6918}> : (memref<16x16xf32>, index, i32, memref<16x16xi32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %161 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%150, %161) : (i32, memref<i32>) -> ()
    "memref.store"(%150, %161) : (i32, memref<i32>) -> ()
    "func.call"(%158, %159, %161) <{callee = @au_generated_candidate_27__direct_site_6955}> : (memref<16x16xi32>, memref<16x16xf32>, memref<i32>) -> ()
    %162 = "memref.load"(%161) : (memref<i32>) -> i32
    "func.call"(%160, %159, %158) <{callee = @pcl_validate_outputs}> : (memref<16x16x3xf32>, memref<16x16xf32>, memref<16x16xi32>) -> ()
    "func.return"(%162) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<16x16x3xf32>, index, i32, memref<16x16xf32>) -> (), sym_name = "au_generated_candidate_10__direct_site_6842"}> ({
  ^bb0(%arg12: memref<16x16x3xf32>, %arg13: index, %arg14: i32, %arg15: memref<16x16xf32>):
    %59 = "arith.constant"() <{value = 5.000000e-01 : f32}> : () -> f32
    %60 = "arith.constant"() <{value = true}> : () -> i1
    %61 = "arith.constant"() <{value = 15 : i32}> : () -> i32
    %62 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %63 = "arith.constant"() <{value = -2 : i32}> : () -> i32
    %64 = "arith.constant"() <{value = 5 : index}> : () -> index
    %65 = "arith.constant"() <{value = 2 : index}> : () -> index
    %66 = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %67 = "arith.constant"() <{value = 0 : index}> : () -> index
    %68 = "arith.constant"() <{value = 16 : index}> : () -> index
    %69 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%67, %68, %69) ({
    ^bb0(%arg16: index):
      %70 = "arith.index_cast"(%arg16) : (index) -> i32
      %71 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%66, %71) : (f32, memref<f32>) -> ()
      %72 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%66, %72) : (f32, memref<f32>) -> ()
      %73 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%66, %73) : (f32, memref<f32>) -> ()
      %74 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%66, %74) : (f32, memref<f32>) -> ()
      %75 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%66, %75) : (f32, memref<f32>) -> ()
      %76 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<f32>
      "memref.store"(%66, %76) : (f32, memref<f32>) -> ()
      %77 = "memref.load"(%arg12, %arg13, %arg16, %67) : (memref<16x16x3xf32>, index, index, index) -> f32
      %78 = "memref.load"(%arg12, %arg13, %arg16, %69) : (memref<16x16x3xf32>, index, index, index) -> f32
      %79 = "memref.load"(%arg12, %arg13, %arg16, %65) : (memref<16x16x3xf32>, index, index, index) -> f32
      "scf.for"(%67, %64, %69) ({
      ^bb0(%arg17: index):
        %101 = "arith.index_cast"(%arg17) : (index) -> i32
        %102 = "arith.addi"(%101, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.for"(%67, %64, %69) ({
        ^bb0(%arg18: index):
          %103 = "arith.index_cast"(%arg18) : (index) -> i32
          %104 = "arith.addi"(%103, %63) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %105 = "arith.addi"(%arg14, %102) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %106 = "arith.cmpi"(%105, %62) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %107 = "arith.cmpi"(%105, %61) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %108 = "arith.andi"(%106, %107) : (i1, i1) -> i1
          %109 = "arith.xori"(%108, %60) : (i1, i1) -> i1
          %110 = "arith.andi"(%109, %106) : (i1, i1) -> i1
          %111 = "arith.select"(%108, %61, %62) : (i1, i32, i32) -> i32
          %112 = "arith.select"(%110, %105, %111) : (i1, i32, i32) -> i32
          %113 = "arith.addi"(%70, %104) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %114 = "arith.cmpi"(%113, %62) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %115 = "arith.cmpi"(%113, %61) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %116 = "arith.andi"(%114, %115) : (i1, i1) -> i1
          %117 = "arith.subi"(%60, %116) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
          %118 = "arith.andi"(%117, %114) : (i1, i1) -> i1
          %119 = "arith.select"(%116, %61, %62) : (i1, i32, i32) -> i32
          %120 = "arith.select"(%118, %113, %119) : (i1, i32, i32) -> i32
          %121 = "arith.index_cast"(%112) : (i32) -> index
          %122 = "arith.index_cast"(%120) : (i32) -> index
          %123 = "memref.load"(%arg12, %121, %122, %67) : (memref<16x16x3xf32>, index, index, index) -> f32
          %124 = "arith.subf"(%123, %77) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %125 = "memref.load"(%arg12, %121, %122, %69) : (memref<16x16x3xf32>, index, index, index) -> f32
          %126 = "arith.subf"(%125, %78) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %127 = "memref.load"(%arg12, %121, %122, %65) : (memref<16x16x3xf32>, index, index, index) -> f32
          %128 = "arith.subf"(%127, %79) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %129 = "arith.mulf"(%124, %124) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %130 = "memref.load"(%71) : (memref<f32>) -> f32
          %131 = "arith.addf"(%130, %129) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%131, %71) : (f32, memref<f32>) -> ()
          %132 = "arith.mulf"(%126, %126) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %133 = "memref.load"(%72) : (memref<f32>) -> f32
          %134 = "arith.addf"(%133, %132) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%134, %72) : (f32, memref<f32>) -> ()
          %135 = "arith.mulf"(%128, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %136 = "memref.load"(%73) : (memref<f32>) -> f32
          %137 = "arith.addf"(%136, %135) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%137, %73) : (f32, memref<f32>) -> ()
          %138 = "arith.mulf"(%124, %126) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %139 = "memref.load"(%74) : (memref<f32>) -> f32
          %140 = "arith.addf"(%139, %138) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%140, %74) : (f32, memref<f32>) -> ()
          %141 = "arith.mulf"(%124, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %142 = "memref.load"(%75) : (memref<f32>) -> f32
          %143 = "arith.addf"(%142, %141) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%143, %75) : (f32, memref<f32>) -> ()
          %144 = "arith.mulf"(%126, %128) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %145 = "memref.load"(%76) : (memref<f32>) -> f32
          %146 = "arith.addf"(%145, %144) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          "memref.store"(%146, %76) : (f32, memref<f32>) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %80 = "memref.load"(%71) : (memref<f32>) -> f32
      %81 = "memref.load"(%72) : (memref<f32>) -> f32
      %82 = "arith.addf"(%80, %81) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %83 = "memref.load"(%73) : (memref<f32>) -> f32
      %84 = "arith.addf"(%82, %83) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %85 = "memref.load"(%74) : (memref<f32>) -> f32
      %86 = "arith.cmpf"(%85, %66) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %87 = "scf.if"(%86) ({
        %100 = "arith.negf"(%85) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%100) : (f32) -> ()
      }, {
        "scf.yield"(%85) : (f32) -> ()
      }) : (i1) -> f32
      %88 = "memref.load"(%75) : (memref<f32>) -> f32
      %89 = "arith.cmpf"(%88, %66) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %90 = "scf.if"(%89) ({
        %99 = "arith.negf"(%88) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%99) : (f32) -> ()
      }, {
        "scf.yield"(%88) : (f32) -> ()
      }) : (i1) -> f32
      %91 = "arith.addf"(%87, %90) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %92 = "memref.load"(%76) : (memref<f32>) -> f32
      %93 = "arith.cmpf"(%92, %66) <{fastmath = #arith.fastmath<none>, predicate = 4 : i64}> : (f32, f32) -> i1
      %94 = "scf.if"(%93) ({
        %98 = "arith.negf"(%92) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        "scf.yield"(%98) : (f32) -> ()
      }, {
        "scf.yield"(%92) : (f32) -> ()
      }) : (i1) -> f32
      %95 = "arith.addf"(%91, %94) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %96 = "arith.mulf"(%95, %59) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      %97 = "arith.subf"(%84, %96) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%97, %arg15, %arg13, %arg16) : (f32, memref<16x16xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<16x16xf32>, index, i32, memref<16x16xi32>) -> (), sym_name = "au_generated_candidate_24__direct_site_6918"}> ({
  ^bb0(%arg5: memref<16x16xf32>, %arg6: index, %arg7: i32, %arg8: memref<16x16xi32>):
    %17 = "arith.constant"() <{value = true}> : () -> i1
    %18 = "arith.constant"() <{value = 15 : i32}> : () -> i32
    %19 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %20 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %21 = "arith.constant"() <{value = 3 : index}> : () -> index
    %22 = "arith.constant"() <{value = 2.000000e-02 : f32}> : () -> f32
    %23 = "arith.constant"() <{value = 0 : index}> : () -> index
    %24 = "arith.constant"() <{value = 16 : index}> : () -> index
    %25 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%23, %24, %25) ({
    ^bb0(%arg9: index):
      %26 = "arith.index_cast"(%arg9) : (index) -> i32
      %27 = "memref.load"(%arg5, %arg6, %arg9) : (memref<16x16xf32>, index, index) -> f32
      %28 = "arith.cmpf"(%27, %22) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
      %29 = "arith.extui"(%28) : (i1) -> i32
      %30 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
      "memref.store"(%29, %30) : (i32, memref<i32>) -> ()
      "scf.for"(%23, %21, %25) ({
      ^bb0(%arg10: index):
        %32 = "arith.index_cast"(%arg10) : (index) -> i32
        %33 = "arith.addi"(%32, %20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "scf.for"(%23, %21, %25) ({
        ^bb0(%arg11: index):
          %34 = "arith.index_cast"(%arg11) : (index) -> i32
          %35 = "arith.addi"(%34, %20) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %36 = "arith.addi"(%arg7, %33) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %37 = "arith.cmpi"(%36, %19) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %38 = "arith.cmpi"(%36, %18) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %39 = "arith.andi"(%37, %38) : (i1, i1) -> i1
          %40 = "arith.xori"(%39, %17) : (i1, i1) -> i1
          %41 = "arith.andi"(%40, %37) : (i1, i1) -> i1
          %42 = "arith.select"(%39, %18, %19) : (i1, i32, i32) -> i32
          %43 = "arith.select"(%41, %36, %42) : (i1, i32, i32) -> i32
          %44 = "arith.addi"(%26, %35) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %45 = "arith.cmpi"(%44, %19) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %46 = "arith.cmpi"(%44, %18) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %47 = "arith.andi"(%45, %46) : (i1, i1) -> i1
          %48 = "arith.xori"(%47, %17) : (i1, i1) -> i1
          %49 = "arith.andi"(%48, %45) : (i1, i1) -> i1
          %50 = "arith.select"(%47, %18, %19) : (i1, i32, i32) -> i32
          %51 = "arith.select"(%49, %44, %50) : (i1, i32, i32) -> i32
          %52 = "arith.cmpi"(%33, %19) <{predicate = 1 : i64}> : (i32, i32) -> i1
          %53 = "scf.if"(%52) ({
            "scf.yield"(%17) : (i1) -> ()
          }, {
            %58 = "arith.cmpi"(%35, %19) <{predicate = 1 : i64}> : (i32, i32) -> i1
            "scf.yield"(%58) : (i1) -> ()
          }) : (i1) -> i1
          "scf.if"(%53) ({
            %54 = "arith.index_cast"(%43) : (i32) -> index
            %55 = "arith.index_cast"(%51) : (i32) -> index
            %56 = "memref.load"(%arg5, %54, %55) : (memref<16x16xf32>, index, index) -> f32
            %57 = "arith.cmpf"(%56, %27) <{fastmath = #arith.fastmath<none>, predicate = 2 : i64}> : (f32, f32) -> i1
            "scf.if"(%57) ({
              "memref.store"(%19, %30) : (i32, memref<i32>) -> ()
              "scf.yield"() : () -> ()
            }, {
            }) : (i1) -> ()
            "scf.yield"() : () -> ()
          }, {
          }) : (i1) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %31 = "memref.load"(%30) : (memref<i32>) -> i32
      "memref.store"(%31, %arg8, %arg6, %arg9) : (i32, memref<16x16xi32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<16x16xi32>, memref<16x16xf32>, memref<i32>) -> (), sym_name = "au_generated_candidate_27__direct_site_6955"}> ({
  ^bb0(%arg0: memref<16x16xi32>, %arg1: memref<16x16xf32>, %arg2: memref<i32>):
    %0 = "arith.constant"() <{value = 3.200000e+01 : f32}> : () -> f32
    %1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "arith.constant"() <{value = 0 : index}> : () -> index
    %3 = "arith.constant"() <{value = 16 : index}> : () -> index
    %4 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%2, %3, %4) ({
    ^bb0(%arg3: index):
      %5 = "arith.index_cast"(%arg3) : (index) -> i32
      "scf.for"(%2, %3, %4) ({
      ^bb0(%arg4: index):
        %6 = "arith.index_cast"(%arg4) : (index) -> i32
        %7 = "memref.load"(%arg0, %arg3, %arg4) : (memref<16x16xi32>, index, index) -> i32
        %8 = "arith.cmpi"(%7, %1) <{predicate = 1 : i64}> : (i32, i32) -> i1
        %9 = "scf.if"(%8) ({
          %12 = "memref.load"(%arg1, %arg3, %arg4) : (memref<16x16xf32>, index, index) -> f32
          %13 = "arith.mulf"(%12, %0) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
          %14 = "arith.fptosi"(%13) : (f32) -> i32
          %15 = "arith.addi"(%14, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %16 = "arith.addi"(%15, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "scf.yield"(%16) : (i32) -> ()
        }, {
          "scf.yield"(%1) : (i32) -> ()
        }) : (i1) -> i32
        %10 = "memref.load"(%arg2) : (memref<i32>) -> i32
        %11 = "arith.addi"(%10, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%11, %arg2) : (i32, memref<i32>) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<16x16x3xf32>, memref<16x16xf32>, memref<16x16xi32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
