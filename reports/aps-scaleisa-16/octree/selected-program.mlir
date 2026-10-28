"builtin.module"() ({
  "func.func"() <{function_type = () -> i32, sym_name = "pcl_selected_main"}> ({
    %5 = "arith.constant"() <{value = 3 : index}> : () -> index
    %6 = "arith.constant"() <{value = 128 : index}> : () -> index
    %7 = "arith.constant"() <{value = 2 : index}> : () -> index
    %8 = "arith.constant"() <{value = 2048 : index}> : () -> index
    %9 = "arith.constant"() <{value = 1 : index}> : () -> index
    %10 = "arith.constant"() <{value = 4 : index}> : () -> index
    %11 = "arith.constant"() <{value = 0 : index}> : () -> index
    %12 = "arith.constant"() <{value = true}> : () -> i1
    %13 = "arith.constant"() <{value = 16 : i32}> : () -> i32
    %14 = "arith.constant"() <{value = -1 : i32}> : () -> i32
    %15 = "arith.constant"() <{value = 1.370000e+02 : f32}> : () -> f32
    %16 = "arith.constant"() <{value = 137 : i32}> : () -> i32
    %17 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %18 = "arith.constant"() <{value = 19 : i32}> : () -> i32
    %19 = "arith.constant"() <{value = 1.310000e+02 : f32}> : () -> f32
    %20 = "arith.constant"() <{value = 131 : i32}> : () -> i32
    %21 = "arith.constant"() <{value = 11 : i32}> : () -> i32
    %22 = "arith.constant"() <{value = 17 : i32}> : () -> i32
    %23 = "arith.constant"() <{value = 1.270000e+02 : f32}> : () -> f32
    %24 = "arith.constant"() <{value = 127 : i32}> : () -> i32
    %25 = "arith.constant"() <{value = 7 : i32}> : () -> i32
    %26 = "arith.constant"() <{value = 13 : i32}> : () -> i32
    %27 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %28 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %29 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%28, %29) : (i32, memref<i32>) -> ()
    %30 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<128xi32>
    %31 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<4x4096xi32>
    %32 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<2048x3xf32>
    "memref.store"(%28, %29) : (i32, memref<i32>) -> ()
    "func.call"(%31) <{callee = @au_generated_candidate_53__l1p_u1x8_site_5230}> : (memref<4x4096xi32>) -> ()
    "scf.for"(%11, %8, %9) ({
    ^bb0(%arg11: index):
      %158 = "arith.index_cast"(%arg11) : (index) -> i32
      %159 = "arith.muli"(%158, %26) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %160 = "arith.addi"(%159, %25) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %161 = "arith.remsi"(%160, %24) : (i32, i32) -> i32
      %162 = "arith.sitofp"(%161) : (i32) -> f32
      %163 = "arith.divf"(%162, %23) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%163, %32, %arg11, %11) : (f32, memref<2048x3xf32>, index, index) -> ()
      %164 = "arith.muli"(%158, %22) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %165 = "arith.addi"(%164, %21) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %166 = "arith.remsi"(%165, %20) : (i32, i32) -> i32
      %167 = "arith.sitofp"(%166) : (i32) -> f32
      %168 = "arith.divf"(%167, %19) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%168, %32, %arg11, %9) : (f32, memref<2048x3xf32>, index, index) -> ()
      %169 = "arith.muli"(%158, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %170 = "arith.addi"(%169, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %171 = "arith.remsi"(%170, %16) : (i32, i32) -> i32
      %172 = "arith.sitofp"(%171) : (i32) -> f32
      %173 = "arith.divf"(%172, %15) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "memref.store"(%173, %32, %arg11, %7) : (f32, memref<2048x3xf32>, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%11, %10, %9) ({
    ^bb0(%arg9: index):
      %117 = "arith.index_cast"(%arg9) : (index) -> i32
      %118 = "arith.shrsi"(%13, %117) : (i32, i32) -> i32
      "scf.for"(%11, %8, %9) ({
      ^bb0(%arg10: index):
        %119 = "memref.load"(%32, %arg10, %11) : (memref<2048x3xf32>, index, index) -> f32
        %120 = "arith.sitofp"(%118) : (i32) -> f32
        %121 = "arith.mulf"(%119, %120) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %122 = "arith.fptosi"(%121) : (f32) -> i32
        %123 = "arith.addi"(%118, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %124 = "arith.cmpi"(%122, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %125 = "arith.cmpi"(%122, %123) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %126 = "arith.andi"(%124, %125) : (i1, i1) -> i1
        %127 = "arith.xori"(%126, %12) : (i1, i1) -> i1
        %128 = "arith.andi"(%127, %124) : (i1, i1) -> i1
        %129 = "arith.select"(%126, %123, %28) : (i1, i32, i32) -> i32
        %130 = "arith.select"(%128, %122, %129) : (i1, i32, i32) -> i32
        %131 = "memref.load"(%32, %arg10, %9) : (memref<2048x3xf32>, index, index) -> f32
        %132 = "arith.mulf"(%131, %120) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %133 = "arith.fptosi"(%132) : (f32) -> i32
        %134 = "arith.cmpi"(%133, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %135 = "arith.cmpi"(%133, %123) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %136 = "arith.andi"(%134, %135) : (i1, i1) -> i1
        %137 = "arith.xori"(%136, %12) : (i1, i1) -> i1
        %138 = "arith.andi"(%137, %134) : (i1, i1) -> i1
        %139 = "arith.select"(%136, %123, %28) : (i1, i32, i32) -> i32
        %140 = "arith.select"(%138, %133, %139) : (i1, i32, i32) -> i32
        %141 = "memref.load"(%32, %arg10, %7) : (memref<2048x3xf32>, index, index) -> f32
        %142 = "arith.mulf"(%141, %120) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %143 = "arith.fptosi"(%142) : (f32) -> i32
        %144 = "arith.cmpi"(%143, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %145 = "arith.cmpi"(%143, %123) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %146 = "arith.andi"(%144, %145) : (i1, i1) -> i1
        %147 = "arith.xori"(%146, %12) : (i1, i1) -> i1
        %148 = "arith.andi"(%147, %144) : (i1, i1) -> i1
        %149 = "arith.select"(%146, %123, %28) : (i1, i32, i32) -> i32
        %150 = "arith.select"(%148, %143, %149) : (i1, i32, i32) -> i32
        %151 = "arith.muli"(%130, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %152 = "arith.addi"(%151, %140) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %153 = "arith.muli"(%152, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %154 = "arith.addi"(%153, %150) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %155 = "arith.index_cast"(%154) : (i32) -> index
        %156 = "memref.load"(%31, %arg9, %155) : (memref<4x4096xi32>, index, index) -> i32
        %157 = "arith.addi"(%156, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "memref.store"(%157, %31, %arg9, %155) : (i32, memref<4x4096xi32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %33 = "memref.alloca"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<i32>
    "memref.store"(%28, %33) : (i32, memref<i32>) -> ()
    "scf.for"(%11, %6, %9) ({
    ^bb0(%arg4: index):
      "memref.store"(%28, %33) : (i32, memref<i32>) -> ()
      %42 = "memref.load"(%32, %arg4, %11) : (memref<2048x3xf32>, index, index) -> f32
      %43 = "memref.load"(%32, %arg4, %9) : (memref<2048x3xf32>, index, index) -> f32
      %44 = "memref.load"(%32, %arg4, %7) : (memref<2048x3xf32>, index, index) -> f32
      "scf.for"(%11, %10, %9) ({
      ^bb0(%arg5: index):
        %46 = "arith.index_cast"(%arg5) : (index) -> i32
        %47 = "arith.shrsi"(%13, %46) : (i32, i32) -> i32
        %48 = "arith.sitofp"(%47) : (i32) -> f32
        %49 = "arith.mulf"(%42, %48) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %50 = "arith.fptosi"(%49) : (f32) -> i32
        %51 = "arith.addi"(%47, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %52 = "arith.cmpi"(%50, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %53 = "arith.cmpi"(%50, %51) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %54 = "arith.andi"(%52, %53) : (i1, i1) -> i1
        %55 = "arith.xori"(%54, %12) : (i1, i1) -> i1
        %56 = "arith.andi"(%55, %52) : (i1, i1) -> i1
        %57 = "arith.select"(%54, %51, %28) : (i1, i32, i32) -> i32
        %58 = "arith.select"(%56, %50, %57) : (i1, i32, i32) -> i32
        %59 = "arith.mulf"(%43, %48) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %60 = "arith.fptosi"(%59) : (f32) -> i32
        %61 = "arith.cmpi"(%60, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %62 = "arith.cmpi"(%60, %51) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %63 = "arith.andi"(%61, %62) : (i1, i1) -> i1
        %64 = "arith.xori"(%63, %12) : (i1, i1) -> i1
        %65 = "arith.andi"(%64, %61) : (i1, i1) -> i1
        %66 = "arith.select"(%63, %51, %28) : (i1, i32, i32) -> i32
        %67 = "arith.select"(%65, %60, %66) : (i1, i32, i32) -> i32
        %68 = "arith.mulf"(%44, %48) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        %69 = "arith.fptosi"(%68) : (f32) -> i32
        %70 = "arith.cmpi"(%69, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
        %71 = "arith.cmpi"(%69, %51) <{predicate = 4 : i64}> : (i32, i32) -> i1
        %72 = "arith.andi"(%70, %71) : (i1, i1) -> i1
        %73 = "arith.xori"(%72, %12) : (i1, i1) -> i1
        %74 = "arith.andi"(%73, %70) : (i1, i1) -> i1
        %75 = "arith.select"(%72, %51, %28) : (i1, i32, i32) -> i32
        %76 = "arith.select"(%74, %69, %75) : (i1, i32, i32) -> i32
        "scf.for"(%11, %5, %9) ({
        ^bb0(%arg6: index):
          %77 = "arith.index_cast"(%arg6) : (index) -> i32
          %78 = "arith.addi"(%77, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %79 = "arith.addi"(%76, %78) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          %80 = "arith.cmpi"(%79, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
          %81 = "arith.cmpi"(%79, %51) <{predicate = 4 : i64}> : (i32, i32) -> i1
          %82 = "arith.andi"(%80, %81) : (i1, i1) -> i1
          %83 = "arith.xori"(%82, %12) : (i1, i1) -> i1
          %84 = "arith.andi"(%83, %80) : (i1, i1) -> i1
          %85 = "arith.select"(%82, %51, %28) : (i1, i32, i32) -> i32
          %86 = "arith.select"(%84, %79, %85) : (i1, i32, i32) -> i32
          %87 = "arith.addi"(%46, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
          "scf.for"(%11, %5, %9) ({
          ^bb0(%arg7: index):
            %88 = "arith.index_cast"(%arg7) : (index) -> i32
            %89 = "arith.addi"(%88, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
            "scf.for"(%11, %5, %9) ({
            ^bb0(%arg8: index):
              %90 = "arith.index_cast"(%arg8) : (index) -> i32
              %91 = "arith.addi"(%90, %14) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %92 = "arith.addi"(%58, %91) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %93 = "arith.cmpi"(%92, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
              %94 = "arith.cmpi"(%92, %51) <{predicate = 4 : i64}> : (i32, i32) -> i1
              %95 = "arith.andi"(%93, %94) : (i1, i1) -> i1
              %96 = "arith.xori"(%95, %12) : (i1, i1) -> i1
              %97 = "arith.andi"(%96, %93) : (i1, i1) -> i1
              %98 = "arith.select"(%95, %51, %28) : (i1, i32, i32) -> i32
              %99 = "arith.select"(%97, %92, %98) : (i1, i32, i32) -> i32
              %100 = "arith.addi"(%67, %89) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %101 = "arith.cmpi"(%100, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
              %102 = "arith.cmpi"(%100, %51) <{predicate = 4 : i64}> : (i32, i32) -> i1
              %103 = "arith.andi"(%101, %102) : (i1, i1) -> i1
              %104 = "arith.xori"(%103, %12) : (i1, i1) -> i1
              %105 = "arith.andi"(%104, %101) : (i1, i1) -> i1
              %106 = "arith.select"(%103, %51, %28) : (i1, i32, i32) -> i32
              %107 = "arith.select"(%105, %100, %106) : (i1, i32, i32) -> i32
              %108 = "arith.muli"(%99, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %109 = "arith.addi"(%108, %107) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %110 = "arith.muli"(%109, %13) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %111 = "arith.addi"(%110, %86) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %112 = "arith.index_cast"(%111) : (i32) -> index
              %113 = "memref.load"(%31, %arg5, %112) : (memref<4x4096xi32>, index, index) -> i32
              %114 = "arith.muli"(%113, %87) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              %115 = "memref.load"(%33) : (memref<i32>) -> i32
              %116 = "arith.addi"(%115, %114) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
              "memref.store"(%116, %33) : (i32, memref<i32>) -> ()
              "scf.yield"() : () -> ()
            }) : (index, index, index) -> ()
            "scf.yield"() : () -> ()
          }) : (index, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      %45 = "memref.load"(%33) : (memref<i32>) -> i32
      "memref.store"(%45, %30, %arg4) : (i32, memref<128xi32>, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "scf.for"(%11, %6, %9) ({
    ^bb0(%arg3: index):
      %35 = "arith.index_cast"(%arg3) : (index) -> i32
      %36 = "memref.load"(%30, %arg3) : (memref<128xi32>, index) -> i32
      %37 = "arith.andi"(%35, %25) : (i32, i32) -> i32
      %38 = "arith.addi"(%37, %27) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %39 = "arith.muli"(%36, %38) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      %40 = "memref.load"(%29) : (memref<i32>) -> i32
      %41 = "arith.addi"(%40, %39) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
      "memref.store"(%41, %29) : (i32, memref<i32>) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    %34 = "memref.load"(%29) : (memref<i32>) -> i32
    "func.call"(%32, %31, %30) <{callee = @pcl_validate_outputs}> : (memref<2048x3xf32>, memref<4x4096xi32>, memref<128xi32>) -> ()
    "func.return"(%34) : (i32) -> ()
  }) {llvm.linkage = #llvm.linkage<external>} : () -> ()
  "func.func"() <{function_type = (memref<4x4096xi32>) -> (), sym_name = "au_generated_candidate_53__l1p_u1x8_site_5230"}> ({
  ^bb0(%arg0: memref<4x4096xi32>):
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 4096 : index}> : () -> index
    %2 = "arith.constant"() <{value = 0 : index}> : () -> index
    %3 = "arith.constant"() <{value = 4 : index}> : () -> index
    %4 = "arith.constant"() <{value = 1 : index}> : () -> index
    "scf.for"(%2, %3, %4) ({
    ^bb0(%arg1: index):
      "scf.for"(%2, %1, %4) ({
      ^bb0(%arg2: index):
        "memref.store"(%0, %arg0, %arg1, %arg2) : (i32, memref<4x4096xi32>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {au.generated = true} : () -> ()
  "func.func"() <{function_type = (memref<2048x3xf32>, memref<4x4096xi32>, memref<128xi32>) -> (), sym_name = "pcl_validate_outputs", sym_visibility = "private"}> ({
  }) : () -> ()
}) : () -> ()
