module attributes {scaleisa.ci_instructions = 2 : i64, scaleisa.ci_sites = 2 : i64} {
  memref.global "private" @_ZL8BASELINE : memref<1xf32> = dense<2.000000e-01>
  memref.global "private" @_ZL2FY : memref<1xf32> = dense<1.780000e+02>
  memref.global "private" @_ZL2FX : memref<1xf32> = dense<1.800000e+02>
  memref.global "private" @_ZL10DEPTH_BIAS : memref<1xf32> = dense<4.200000e+00>
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c10 = arith.constant 10 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 2.110000e+02 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 2.570000e+02 : f32
    %cst_1 = arith.constant 3.070000e+02 : f32
    %cst_2 = arith.constant 1.000000e-03 : f32
    %cst_3 = arith.constant 2.500000e-01 : f32
    %cst_4 = arith.constant 1.800000e-01 : f32
    %cst_5 = arith.constant 1.000000e+00 : f32
    %cst_6 = arith.constant 3.500000e-01 : f32
    %cst_7 = arith.constant 4.000000e-02 : f32
    %cst_8 = arith.constant -4.000000e-02 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_9 = arith.constant 8.000000e-02 : f32
    %cst_10 = arith.constant 1.200000e-01 : f32
    %cst_11 = arith.constant 3.000000e-02 : f32
    %cst_12 = arith.constant -2.000000e-02 : f32
    %cst_13 = arith.constant -1.000000e+00 : f32
    %cst_14 = arith.constant 4.200000e-01 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_15 = arith.constant -6.000000e-01 : f32
    %cst_16 = arith.constant 3.100000e-01 : f32
    %cst_17 = arith.constant 1.600000e+00 : f32
    %cst_18 = arith.constant 1.100000e-01 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_19 = arith.constant 0.000000e+00 : f32
    %cst_20 = arith.constant 0.00999999977 : f32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_21 = arith.constant 1.500000e-02 : f32
    %cst_22 = arith.constant 1.200000e-02 : f32
    %c6_i32 = arith.constant 6 : i32
    %cst_23 = arith.constant 8.000000e-03 : f32
    %cst_24 = arith.constant 6.000000e-02 : f32
    %cst_25 = arith.constant 5.000000e-02 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-2_i32 = arith.constant -2 : i32
    %alloca = memref.alloca() : memref<32x3xf32>
    %alloca_26 = memref.alloca() : memref<32x3xf32>
    %alloca_27 = memref.alloca() {alignment = 64 : i64} : memref<32x3xf32>
    %alloca_28 = memref.alloca() : memref<3xf32>
    memref.store %cst_10, %alloca_28[%c0] : memref<3xf32>
    memref.store %cst_11, %alloca_28[%c1] : memref<3xf32>
    memref.store %cst_12, %alloca_28[%c2] : memref<3xf32>
    scf.for %arg0 = %c0 to %c32 step %c1 {
      %23 = arith.index_cast %arg0 : index to i32
      %24 = arith.remsi %23, %c8_i32 : i32
      %25 = arith.sitofp %24 : i32 to f32
      %26 = arith.mulf %25, %cst_14 : f32
      %27 = arith.addf %26, %cst_13 : f32
      memref.store %27, %alloca_27[%arg0, %c0] : memref<32x3xf32>
      %28 = arith.divsi %23, %c8_i32 : i32
      %29 = arith.sitofp %28 : i32 to f32
      %30 = arith.mulf %29, %cst_16 : f32
      %31 = arith.addf %30, %cst_15 : f32
      memref.store %31, %alloca_27[%arg0, %c1] : memref<32x3xf32>
      %32 = arith.remsi %23, %c5_i32 : i32
      %33 = arith.sitofp %32 : i32 to f32
      %34 = arith.mulf %33, %cst_18 : f32
      %35 = arith.addf %34, %cst_17 : f32
      memref.store %35, %alloca_27[%arg0, %c2] : memref<32x3xf32>
      %36 = memref.load %alloca_27[%arg0, %c0] : memref<32x3xf32>
      %37 = memref.load %alloca_27[%arg0, %c1] : memref<32x3xf32>
      %38 = memref.load %alloca_27[%arg0, %c2] : memref<32x3xf32>
      %39 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
      %40 = memref.load %39[%c0] : memref<1xf32>
      %41 = arith.addf %38, %40 : f32
      %42 = arith.divf %cst_5, %41 : f32
      %43 = memref.get_global @_ZL2FX : memref<1xf32>
      %44 = memref.load %43[%c0] : memref<1xf32>
      %45 = arith.mulf %44, %36 : f32
      %46 = arith.mulf %45, %42 : f32
      memref.store %46, %alloca_26[%arg0, %c0] : memref<32x3xf32>
      %47 = memref.get_global @_ZL2FY : memref<1xf32>
      %48 = memref.load %47[%c0] : memref<1xf32>
      %49 = arith.mulf %48, %37 : f32
      %50 = arith.mulf %49, %42 : f32
      memref.store %50, %alloca_26[%arg0, %c1] : memref<32x3xf32>
      %51 = memref.load %43[%c0] : memref<1xf32>
      %52 = memref.get_global @_ZL8BASELINE : memref<1xf32>
      %53 = memref.load %52[%c0] : memref<1xf32>
      %54 = arith.mulf %51, %53 : f32
      %55 = arith.mulf %54, %42 : f32
      memref.store %55, %alloca_26[%arg0, %c2] : memref<32x3xf32>
      %56 = memref.load %alloca_27[%arg0, %c0] : memref<32x3xf32>
      %57 = arith.subf %56, %cst_6 : f32
      %58 = memref.load %alloca_27[%arg0, %c1] : memref<32x3xf32>
      %59 = arith.subf %58, %cst_8 : f32
      %60 = memref.load %alloca_27[%arg0, %c2] : memref<32x3xf32>
      %61 = arith.subf %60, %cst_9 : f32
      %62 = memref.load %39[%c0] : memref<1xf32>
      %63 = arith.addf %61, %62 : f32
      %64 = arith.divf %cst_5, %63 : f32
      %65 = memref.load %43[%c0] : memref<1xf32>
      %66 = arith.mulf %65, %57 : f32
      %67 = arith.mulf %66, %64 : f32
      memref.store %67, %alloca[%arg0, %c0] : memref<32x3xf32>
      %68 = memref.load %47[%c0] : memref<1xf32>
      %69 = arith.mulf %68, %59 : f32
      %70 = arith.mulf %69, %64 : f32
      memref.store %70, %alloca[%arg0, %c1] : memref<32x3xf32>
      %71 = memref.load %43[%c0] : memref<1xf32>
      %72 = memref.load %52[%c0] : memref<1xf32>
      %73 = arith.mulf %71, %72 : f32
      %74 = arith.mulf %73, %64 : f32
      memref.store %74, %alloca[%arg0, %c2] : memref<32x3xf32>
      %75 = arith.remsi %23, %c3_i32 : i32
      %76 = arith.addi %75, %c-1_i32 : i32
      %77 = arith.sitofp %76 : i32 to f32
      %78 = arith.mulf %77, %cst_20 : f32
      %79 = memref.load %alloca_26[%arg0, %c0] : memref<32x3xf32>
      %80 = arith.addf %79, %78 : f32
      memref.store %80, %alloca_26[%arg0, %c0] : memref<32x3xf32>
      %81 = arith.remsi %23, %c4_i32 : i32
      %82 = arith.addi %81, %c-2_i32 : i32
      %83 = arith.sitofp %82 : i32 to f32
      %84 = arith.mulf %83, %cst_20 : f32
      %85 = memref.load %alloca_26[%arg0, %c1] : memref<32x3xf32>
      %86 = arith.subf %85, %84 : f32
      memref.store %86, %alloca_26[%arg0, %c1] : memref<32x3xf32>
      %87 = arith.addi %32, %c-2_i32 : i32
      %88 = arith.sitofp %87 : i32 to f32
      %89 = arith.mulf %88, %cst_21 : f32
      %90 = memref.load %alloca[%arg0, %c0] : memref<32x3xf32>
      %91 = arith.addf %90, %89 : f32
      memref.store %91, %alloca[%arg0, %c0] : memref<32x3xf32>
      %92 = arith.remsi %23, %c6_i32 : i32
      %93 = arith.addi %92, %c-3_i32 : i32
      %94 = arith.sitofp %93 : i32 to f32
      %95 = arith.mulf %94, %cst_22 : f32
      %96 = memref.load %alloca[%arg0, %c1] : memref<32x3xf32>
      %97 = arith.subf %96, %95 : f32
      memref.store %97, %alloca[%arg0, %c1] : memref<32x3xf32>
      %98 = arith.addi %81, %c-1_i32 : i32
      %99 = arith.sitofp %98 : i32 to f32
      %100 = arith.mulf %99, %cst_23 : f32
      %101 = memref.load %alloca[%arg0, %c2] : memref<32x3xf32>
      %102 = arith.addf %101, %100 : f32
      memref.store %102, %alloca[%arg0, %c2] : memref<32x3xf32>
      %103 = arith.remsi %23, %c2_i32 : i32
      %104 = arith.sitofp %103 : i32 to f32
      %105 = arith.mulf %104, %cst_24 : f32
      %106 = memref.load %alloca_27[%arg0, %c0] : memref<32x3xf32>
      %107 = arith.addf %106, %105 : f32
      memref.store %107, %alloca_27[%arg0, %c0] : memref<32x3xf32>
      %108 = arith.sitofp %75 : i32 to f32
      %109 = arith.mulf %108, %cst_25 : f32
      %110 = memref.load %alloca_27[%arg0, %c1] : memref<32x3xf32>
      %111 = arith.subf %110, %109 : f32
      memref.store %111, %alloca_27[%arg0, %c1] : memref<32x3xf32>
      %112 = arith.mulf %99, %cst_7 : f32
      %113 = memref.load %alloca_27[%arg0, %c2] : memref<32x3xf32>
      %114 = arith.addf %113, %112 : f32
      memref.store %114, %alloca_27[%arg0, %c2] : memref<32x3xf32>
    }
    %alloca_29 = memref.alloca() : memref<3xf32>
    %alloca_30 = memref.alloca() : memref<3xf32>
    %alloca_31 = memref.alloca() {alignment = 64 : i64} : memref<32x3xf32>
    %alloca_32 = memref.alloca() {alignment = 64 : i64} : memref<32x3xf32>
    %alloca_33 = memref.alloca() : memref<3xf32>
    %alloca_34 = memref.alloca() : memref<3xf32>
    %0 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
    %1 = memref.get_global @_ZL2FX : memref<1xf32>
    %2 = memref.get_global @_ZL2FY : memref<1xf32>
    %3 = memref.get_global @_ZL8BASELINE : memref<1xf32>
    scf.for %arg0 = %c0 to %c10 step %c1 {
      memref.store %cst_19, %alloca_34[%c0] : memref<3xf32>
      memref.store %cst_19, %alloca_34[%c1] : memref<3xf32>
      memref.store %cst_19, %alloca_34[%c2] : memref<3xf32>
      memref.store %cst_2, %alloca_33[%c0] : memref<3xf32>
      memref.store %cst_2, %alloca_33[%c1] : memref<3xf32>
      memref.store %cst_2, %alloca_33[%c2] : memref<3xf32>
      scf.for %arg1 = %c0 to %c32 step %c1 {
        memref.store %cst_19, %alloca_32[%arg1, %c0] : memref<32x3xf32>
        memref.store %cst_19, %alloca_32[%arg1, %c1] : memref<32x3xf32>
        memref.store %cst_19, %alloca_32[%arg1, %c2] : memref<32x3xf32>
        memref.store %cst_2, %alloca_31[%arg1, %c0] : memref<32x3xf32>
        memref.store %cst_2, %alloca_31[%arg1, %c1] : memref<32x3xf32>
        memref.store %cst_2, %alloca_31[%arg1, %c2] : memref<32x3xf32>
      }
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %44 = memref.load %alloca_27[%arg1, %c0] : memref<32x3xf32>
        %45 = memref.load %alloca_27[%arg1, %c1] : memref<32x3xf32>
        %46 = memref.load %alloca_27[%arg1, %c2] : memref<32x3xf32>
        %47 = memref.load %0[%c0] : memref<1xf32>
        %48 = arith.addf %46, %47 : f32
        %49 = arith.divf %cst_5, %48 : f32
        %50 = memref.load %1[%c0] : memref<1xf32>
        %51 = arith.mulf %50, %44 : f32
        %52 = arith.mulf %51, %49 : f32
        memref.store %52, %alloca_30[%c0] : memref<3xf32>
        %53 = memref.load %2[%c0] : memref<1xf32>
        %54 = arith.mulf %53, %45 : f32
        %55 = arith.mulf %54, %49 : f32
        memref.store %55, %alloca_30[%c1] : memref<3xf32>
        %56 = memref.load %1[%c0] : memref<1xf32>
        %57 = memref.load %3[%c0] : memref<1xf32>
        %58 = arith.mulf %56, %57 : f32
        %59 = arith.mulf %58, %49 : f32
        memref.store %59, %alloca_30[%c2] : memref<3xf32>
        %60 = memref.load %alloca_28[%c0] : memref<3xf32>
        %61 = memref.load %alloca_28[%c1] : memref<3xf32>
        %62 = memref.load %alloca_28[%c2] : memref<3xf32>
        %63 = memref.load %alloca_27[%arg1, %c0] : memref<32x3xf32>
        %64 = arith.subf %63, %60 : f32
        %65 = memref.load %alloca_27[%arg1, %c1] : memref<32x3xf32>
        %66 = arith.subf %65, %61 : f32
        %67 = memref.load %alloca_27[%arg1, %c2] : memref<32x3xf32>
        %68 = arith.subf %67, %62 : f32
        %69 = memref.load %0[%c0] : memref<1xf32>
        %70 = arith.addf %68, %69 : f32
        %71 = arith.divf %cst_5, %70 : f32
        %72 = memref.load %1[%c0] : memref<1xf32>
        %73 = arith.mulf %72, %64 : f32
        %74 = arith.mulf %73, %71 : f32
        memref.store %74, %alloca_29[%c0] : memref<3xf32>
        %75 = memref.load %2[%c0] : memref<1xf32>
        %76 = arith.mulf %75, %66 : f32
        %77 = arith.mulf %76, %71 : f32
        memref.store %77, %alloca_29[%c1] : memref<3xf32>
        %78 = memref.load %1[%c0] : memref<1xf32>
        %79 = memref.load %3[%c0] : memref<1xf32>
        %80 = arith.mulf %78, %79 : f32
        %81 = arith.mulf %80, %71 : f32
        memref.store %81, %alloca_29[%c2] : memref<3xf32>
        %82 = memref.load %alloca_27[%arg1, %c2] : memref<32x3xf32>
        %83 = memref.load %0[%c0] : memref<1xf32>
        %84 = arith.addf %82, %83 : f32
        %85 = arith.divf %cst_5, %84 : f32
        %86 = memref.load %alloca_30[%c0] : memref<3xf32>
        %87 = memref.load %alloca_26[%arg1, %c0] : memref<32x3xf32>
        %88 = arith.subf %86, %87 : f32
        %89 = memref.load %alloca_30[%c1] : memref<3xf32>
        %90 = memref.load %alloca_26[%arg1, %c1] : memref<32x3xf32>
        %91 = arith.subf %89, %90 : f32
        %92 = memref.load %alloca_30[%c2] : memref<3xf32>
        %93 = memref.load %alloca_26[%arg1, %c2] : memref<32x3xf32>
        %94 = arith.subf %92, %93 : f32
        %95 = memref.load %1[%c0] : memref<1xf32>
        %96 = arith.mulf %95, %85 : f32
        %97 = memref.load %2[%c0] : memref<1xf32>
        %98 = arith.mulf %97, %85 : f32
        %99 = arith.negf %95 : f32
        %100 = memref.load %alloca_27[%arg1, %c0] : memref<32x3xf32>
        %101 = arith.mulf %99, %100 : f32
        %102 = arith.mulf %101, %85 : f32
        %103 = arith.mulf %102, %85 : f32
        %104 = arith.negf %97 : f32
        %105 = memref.load %alloca_27[%arg1, %c1] : memref<32x3xf32>
        %106 = arith.mulf %104, %105 : f32
        %107 = arith.mulf %106, %85 : f32
        %108 = arith.mulf %107, %85 : f32
        %109 = memref.load %3[%c0] : memref<1xf32>
        %110 = arith.mulf %99, %109 : f32
        %111 = arith.mulf %110, %85 : f32
        %112 = arith.mulf %111, %85 : f32
        %113 = memref.load %alloca_28[%c0] : memref<3xf32>
        %114 = arith.subf %100, %113 : f32
        %115 = memref.load %alloca_28[%c1] : memref<3xf32>
        %116 = arith.subf %105, %115 : f32
        %117 = memref.load %alloca_28[%c2] : memref<3xf32>
        %118 = arith.subf %82, %117 : f32
        %119 = arith.addf %118, %83 : f32
        %120 = arith.divf %cst_5, %119 : f32
        %121 = memref.load %alloca_29[%c0] : memref<3xf32>
        %122 = memref.load %alloca[%arg1, %c0] : memref<32x3xf32>
        %123 = arith.subf %121, %122 : f32
        %124 = memref.load %alloca_29[%c1] : memref<3xf32>
        %125 = memref.load %alloca[%arg1, %c1] : memref<32x3xf32>
        %126 = arith.subf %124, %125 : f32
        %127 = memref.load %alloca_29[%c2] : memref<3xf32>
        %128 = memref.load %alloca[%arg1, %c2] : memref<32x3xf32>
        %129 = arith.subf %127, %128 : f32
        %130 = arith.mulf %95, %120 : f32
        %131 = arith.mulf %97, %120 : f32
        %132 = arith.mulf %99, %114 : f32
        %133 = arith.mulf %132, %120 : f32
        %134 = arith.mulf %133, %120 : f32
        %135 = arith.mulf %104, %116 : f32
        %136 = arith.mulf %135, %120 : f32
        %137 = arith.mulf %136, %120 : f32
        %138 = arith.mulf %110, %120 : f32
        %139 = arith.mulf %138, %120 : f32
        %140 = arith.negf %123 : f32
        %141 = arith.mulf %140, %130 : f32
        %142 = memref.load %alloca_34[%c0] : memref<3xf32>
        %143 = arith.addf %142, %141 : f32
        memref.store %143, %alloca_34[%c0] : memref<3xf32>
        %144 = arith.negf %126 : f32
        %145 = arith.mulf %144, %131 : f32
        %146 = memref.load %alloca_34[%c1] : memref<3xf32>
        %147 = arith.addf %146, %145 : f32
        memref.store %147, %alloca_34[%c1] : memref<3xf32>
        %148 = arith.mulf %123, %134 : f32
        %149 = arith.mulf %126, %137 : f32
        %150 = arith.addf %148, %149 : f32
        %151 = arith.mulf %129, %139 : f32
        %152 = arith.addf %150, %151 : f32
        %153 = arith.negf %152 : f32
        %154 = memref.load %alloca_34[%c2] : memref<3xf32>
        %155 = arith.addf %154, %153 : f32
        memref.store %155, %alloca_34[%c2] : memref<3xf32>
        %156 = arith.mulf %130, %130 : f32
        %157 = memref.load %alloca_33[%c0] : memref<3xf32>
        %158 = arith.addf %157, %156 : f32
        memref.store %158, %alloca_33[%c0] : memref<3xf32>
        %159 = arith.mulf %131, %131 : f32
        %160 = memref.load %alloca_33[%c1] : memref<3xf32>
        %161 = arith.addf %160, %159 : f32
        memref.store %161, %alloca_33[%c1] : memref<3xf32>
        %162 = arith.mulf %134, %134 : f32
        %163 = arith.mulf %137, %137 : f32
        %164 = arith.addf %162, %163 : f32
        %165 = arith.mulf %139, %139 : f32
        %166 = arith.addf %164, %165 : f32
        %167 = memref.load %alloca_33[%c2] : memref<3xf32>
        %168 = arith.addf %167, %166 : f32
        memref.store %168, %alloca_33[%c2] : memref<3xf32>
        %169 = arith.mulf %88, %96 : f32
        %170 = memref.load %alloca_32[%arg1, %c0] : memref<32x3xf32>
        %171 = arith.addf %170, %169 : f32
        memref.store %171, %alloca_32[%arg1, %c0] : memref<32x3xf32>
        %172 = arith.mulf %91, %98 : f32
        %173 = memref.load %alloca_32[%arg1, %c1] : memref<32x3xf32>
        %174 = arith.addf %173, %172 : f32
        memref.store %174, %alloca_32[%arg1, %c1] : memref<32x3xf32>
        %175 = arith.mulf %88, %103 : f32
        %176 = arith.mulf %91, %108 : f32
        %177 = arith.addf %175, %176 : f32
        %178 = arith.mulf %94, %112 : f32
        %179 = arith.addf %177, %178 : f32
        %180 = memref.load %alloca_32[%arg1, %c2] : memref<32x3xf32>
        %181 = arith.addf %180, %179 : f32
        memref.store %181, %alloca_32[%arg1, %c2] : memref<32x3xf32>
        %182 = arith.mulf %96, %96 : f32
        %183 = memref.load %alloca_31[%arg1, %c0] : memref<32x3xf32>
        %184 = arith.addf %183, %182 : f32
        memref.store %184, %alloca_31[%arg1, %c0] : memref<32x3xf32>
        %185 = arith.mulf %98, %98 : f32
        %186 = memref.load %alloca_31[%arg1, %c1] : memref<32x3xf32>
        %187 = arith.addf %186, %185 : f32
        memref.store %187, %alloca_31[%arg1, %c1] : memref<32x3xf32>
        %188 = arith.mulf %103, %103 : f32
        %189 = arith.mulf %108, %108 : f32
        %190 = arith.addf %188, %189 : f32
        %191 = arith.mulf %112, %112 : f32
        %192 = arith.addf %190, %191 : f32
        %193 = memref.load %alloca_31[%arg1, %c2] : memref<32x3xf32>
        %194 = arith.addf %193, %192 : f32
        memref.store %194, %alloca_31[%arg1, %c2] : memref<32x3xf32>
        %195 = arith.mulf %123, %130 : f32
        %196 = memref.load %alloca_32[%arg1, %c0] : memref<32x3xf32>
        %197 = arith.addf %196, %195 : f32
        memref.store %197, %alloca_32[%arg1, %c0] : memref<32x3xf32>
        %198 = arith.mulf %126, %131 : f32
        %199 = memref.load %alloca_32[%arg1, %c1] : memref<32x3xf32>
        %200 = arith.addf %199, %198 : f32
        memref.store %200, %alloca_32[%arg1, %c1] : memref<32x3xf32>
        %201 = memref.load %alloca_32[%arg1, %c2] : memref<32x3xf32>
        %202 = arith.addf %201, %152 : f32
        memref.store %202, %alloca_32[%arg1, %c2] : memref<32x3xf32>
        %203 = memref.load %alloca_31[%arg1, %c0] : memref<32x3xf32>
        %204 = arith.addf %203, %156 : f32
        memref.store %204, %alloca_31[%arg1, %c0] : memref<32x3xf32>
        %205 = memref.load %alloca_31[%arg1, %c1] : memref<32x3xf32>
        %206 = arith.addf %205, %159 : f32
        memref.store %206, %alloca_31[%arg1, %c1] : memref<32x3xf32>
        %207 = memref.load %alloca_31[%arg1, %c2] : memref<32x3xf32>
        %208 = arith.addf %207, %166 : f32
        memref.store %208, %alloca_31[%arg1, %c2] : memref<32x3xf32>
      }
      %23 = memref.load %alloca_34[%c0] : memref<3xf32>
      %24 = arith.mulf %23, %cst_3 : f32
      %25 = memref.load %alloca_33[%c0] : memref<3xf32>
      %26 = arith.divf %24, %25 : f32
      %27 = memref.load %alloca_28[%c0] : memref<3xf32>
      %28 = arith.subf %27, %26 : f32
      memref.store %28, %alloca_28[%c0] : memref<3xf32>
      %29 = memref.load %alloca_34[%c1] : memref<3xf32>
      %30 = arith.mulf %29, %cst_3 : f32
      %31 = memref.load %alloca_33[%c1] : memref<3xf32>
      %32 = arith.divf %30, %31 : f32
      %33 = memref.load %alloca_28[%c1] : memref<3xf32>
      %34 = arith.subf %33, %32 : f32
      memref.store %34, %alloca_28[%c1] : memref<3xf32>
      %35 = memref.load %alloca_34[%c2] : memref<3xf32>
      %36 = arith.mulf %35, %cst_4 : f32
      %37 = memref.load %alloca_33[%c2] : memref<3xf32>
      %38 = arith.divf %36, %37 : f32
      %39 = memref.load %alloca_28[%c2] : memref<3xf32>
      %40 = arith.subf %39, %38 : f32
      memref.store %40, %alloca_28[%c2] : memref<3xf32>
      %intptr_37 = memref.extract_aligned_pointer_as_index %alloca_32 : memref<32x3xf32> -> index
      %41 = arith.index_castui %intptr_37 : index to i32
      %intptr_38 = memref.extract_aligned_pointer_as_index %alloca_31 : memref<32x3xf32> -> index
      %42 = arith.index_castui %intptr_38 : index to i32
      %intptr_39 = memref.extract_aligned_pointer_as_index %alloca_27 : memref<32x3xf32> -> index
      %43 = arith.index_castui %intptr_39 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %41, %42, %43 : (i32, i32, i32) -> ()
    }
    %alloca_35 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_35[] : memref<i32>
    %4 = memref.load %alloca_28[%c0] : memref<3xf32>
    %5 = arith.mulf %4, %cst : f32
    %6 = arith.fptosi %5 : f32 to i32
    %7 = memref.load %alloca_28[%c1] : memref<3xf32>
    %8 = arith.mulf %7, %cst_0 : f32
    %9 = arith.fptosi %8 : f32 to i32
    %10 = arith.addi %6, %9 : i32
    %11 = memref.load %alloca_28[%c2] : memref<3xf32>
    %12 = arith.mulf %11, %cst_1 : f32
    %13 = arith.fptosi %12 : f32 to i32
    %14 = arith.addi %10, %13 : i32
    memref.store %14, %alloca_35[] : memref<i32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_27 : memref<32x3xf32> -> index
    %15 = arith.index_castui %intptr : index to i32
    %intptr_36 = memref.extract_aligned_pointer_as_index %alloca_35 : memref<i32> -> index
    %16 = arith.index_castui %intptr_36 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %15, %16 : (i32, i32) -> ()
    %17 = memref.load %alloca_35[] : memref<i32>
    %18 = arith.sitofp %17 : i32 to f32
    %19 = arith.cmpi slt, %17, %c0_i32 : i32
    %20 = scf.if %19 -> (f32) {
      %23 = arith.negf %18 : f32
      scf.yield %23 : f32
    } else {
      scf.yield %18 : f32
    }
    %21 = arith.cmpf ogt, %20, %cst_19 : f32
    %22 = arith.select %21, %17, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_28, %alloca_27, %alloca_26, %alloca) : (memref<3xf32>, memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>) -> ()
    return %22 : i32
  }
  func.func private @pcl_validate_outputs(memref<3xf32>, memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>)
}

