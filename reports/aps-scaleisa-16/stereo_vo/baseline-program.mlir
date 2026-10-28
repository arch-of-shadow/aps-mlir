module {
  memref.global "private" @_ZL8BASELINE : memref<1xf32> = dense<2.000000e-01>
  memref.global "private" @_ZL2FY : memref<1xf32> = dense<1.780000e+02>
  memref.global "private" @_ZL2FX : memref<1xf32> = dense<1.800000e+02>
  memref.global "private" @_ZL10DEPTH_BIAS : memref<1xf32> = dense<4.200000e+00>
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 2.110000e+02 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 2.570000e+02 : f32
    %cst_1 = arith.constant 3.070000e+02 : f32
    %cst_2 = arith.constant 7.900000e+01 : f32
    %cst_3 = arith.constant 1.030000e+02 : f32
    %cst_4 = arith.constant 1.270000e+02 : f32
    %cst_5 = arith.constant 1.000000e-03 : f32
    %cst_6 = arith.constant 2.500000e-01 : f32
    %cst_7 = arith.constant 1.800000e-01 : f32
    %cst_8 = arith.constant 1.600000e-01 : f32
    %cst_9 = arith.constant 1.000000e+00 : f32
    %cst_10 = arith.constant 3.500000e-01 : f32
    %cst_11 = arith.constant 4.000000e-02 : f32
    %cst_12 = arith.constant -4.000000e-02 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_13 = arith.constant 8.000000e-02 : f32
    %cst_14 = arith.constant 1.200000e-01 : f32
    %cst_15 = arith.constant 3.000000e-02 : f32
    %cst_16 = arith.constant -2.000000e-02 : f32
    %cst_17 = arith.constant -1.000000e+00 : f32
    %cst_18 = arith.constant 4.200000e-01 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_19 = arith.constant -6.000000e-01 : f32
    %cst_20 = arith.constant 3.100000e-01 : f32
    %cst_21 = arith.constant 1.600000e+00 : f32
    %cst_22 = arith.constant 1.100000e-01 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_23 = arith.constant 0.000000e+00 : f32
    %cst_24 = arith.constant 0.00999999977 : f32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_25 = arith.constant 1.500000e-02 : f32
    %cst_26 = arith.constant 1.200000e-02 : f32
    %c6_i32 = arith.constant 6 : i32
    %cst_27 = arith.constant 8.000000e-03 : f32
    %cst_28 = arith.constant 6.000000e-02 : f32
    %cst_29 = arith.constant 5.000000e-02 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-2_i32 = arith.constant -2 : i32
    %alloca = memref.alloca() : memref<32x3xf32>
    %alloca_30 = memref.alloca() : memref<32x3xf32>
    %alloca_31 = memref.alloca() : memref<32x3xf32>
    %alloca_32 = memref.alloca() : memref<3xf32>
    affine.store %cst_14, %alloca_32[0] : memref<3xf32>
    affine.store %cst_15, %alloca_32[1] : memref<3xf32>
    affine.store %cst_16, %alloca_32[2] : memref<3xf32>
    affine.for %arg0 = 0 to 32 {
      %17 = arith.index_cast %arg0 : index to i32
      %18 = arith.remsi %17, %c8_i32 : i32
      %19 = arith.sitofp %18 : i32 to f32
      %20 = arith.mulf %19, %cst_18 : f32
      %21 = arith.addf %20, %cst_17 : f32
      affine.store %21, %alloca_31[%arg0, 0] : memref<32x3xf32>
      %22 = arith.divsi %17, %c8_i32 : i32
      %23 = arith.sitofp %22 : i32 to f32
      %24 = arith.mulf %23, %cst_20 : f32
      %25 = arith.addf %24, %cst_19 : f32
      affine.store %25, %alloca_31[%arg0, 1] : memref<32x3xf32>
      %26 = arith.remsi %17, %c5_i32 : i32
      %27 = arith.sitofp %26 : i32 to f32
      %28 = arith.mulf %27, %cst_22 : f32
      %29 = arith.addf %28, %cst_21 : f32
      affine.store %29, %alloca_31[%arg0, 2] : memref<32x3xf32>
      %30 = affine.load %alloca_31[%arg0, 0] : memref<32x3xf32>
      %31 = affine.load %alloca_31[%arg0, 1] : memref<32x3xf32>
      %32 = affine.load %alloca_31[%arg0, 2] : memref<32x3xf32>
      %33 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
      %34 = affine.load %33[0] : memref<1xf32>
      %35 = arith.addf %32, %34 : f32
      %36 = arith.divf %cst_9, %35 : f32
      %37 = memref.get_global @_ZL2FX : memref<1xf32>
      %38 = affine.load %37[0] : memref<1xf32>
      %39 = arith.mulf %38, %30 : f32
      %40 = arith.mulf %39, %36 : f32
      affine.store %40, %alloca_30[%arg0, 0] : memref<32x3xf32>
      %41 = memref.get_global @_ZL2FY : memref<1xf32>
      %42 = affine.load %41[0] : memref<1xf32>
      %43 = arith.mulf %42, %31 : f32
      %44 = arith.mulf %43, %36 : f32
      affine.store %44, %alloca_30[%arg0, 1] : memref<32x3xf32>
      %45 = affine.load %37[0] : memref<1xf32>
      %46 = memref.get_global @_ZL8BASELINE : memref<1xf32>
      %47 = affine.load %46[0] : memref<1xf32>
      %48 = arith.mulf %45, %47 : f32
      %49 = arith.mulf %48, %36 : f32
      affine.store %49, %alloca_30[%arg0, 2] : memref<32x3xf32>
      %50 = affine.load %alloca_31[%arg0, 0] : memref<32x3xf32>
      %51 = arith.subf %50, %cst_10 : f32
      %52 = affine.load %alloca_31[%arg0, 1] : memref<32x3xf32>
      %53 = arith.subf %52, %cst_12 : f32
      %54 = affine.load %alloca_31[%arg0, 2] : memref<32x3xf32>
      %55 = arith.subf %54, %cst_13 : f32
      %56 = affine.load %33[0] : memref<1xf32>
      %57 = arith.addf %55, %56 : f32
      %58 = arith.divf %cst_9, %57 : f32
      %59 = affine.load %37[0] : memref<1xf32>
      %60 = arith.mulf %59, %51 : f32
      %61 = arith.mulf %60, %58 : f32
      affine.store %61, %alloca[%arg0, 0] : memref<32x3xf32>
      %62 = affine.load %41[0] : memref<1xf32>
      %63 = arith.mulf %62, %53 : f32
      %64 = arith.mulf %63, %58 : f32
      affine.store %64, %alloca[%arg0, 1] : memref<32x3xf32>
      %65 = affine.load %37[0] : memref<1xf32>
      %66 = affine.load %46[0] : memref<1xf32>
      %67 = arith.mulf %65, %66 : f32
      %68 = arith.mulf %67, %58 : f32
      affine.store %68, %alloca[%arg0, 2] : memref<32x3xf32>
      %69 = arith.remsi %17, %c3_i32 : i32
      %70 = arith.addi %69, %c-1_i32 : i32
      %71 = arith.sitofp %70 : i32 to f32
      %72 = arith.mulf %71, %cst_24 : f32
      %73 = affine.load %alloca_30[%arg0, 0] : memref<32x3xf32>
      %74 = arith.addf %73, %72 : f32
      affine.store %74, %alloca_30[%arg0, 0] : memref<32x3xf32>
      %75 = arith.remsi %17, %c4_i32 : i32
      %76 = arith.addi %75, %c-2_i32 : i32
      %77 = arith.sitofp %76 : i32 to f32
      %78 = arith.mulf %77, %cst_24 : f32
      %79 = affine.load %alloca_30[%arg0, 1] : memref<32x3xf32>
      %80 = arith.subf %79, %78 : f32
      affine.store %80, %alloca_30[%arg0, 1] : memref<32x3xf32>
      %81 = arith.addi %26, %c-2_i32 : i32
      %82 = arith.sitofp %81 : i32 to f32
      %83 = arith.mulf %82, %cst_25 : f32
      %84 = affine.load %alloca[%arg0, 0] : memref<32x3xf32>
      %85 = arith.addf %84, %83 : f32
      affine.store %85, %alloca[%arg0, 0] : memref<32x3xf32>
      %86 = arith.remsi %17, %c6_i32 : i32
      %87 = arith.addi %86, %c-3_i32 : i32
      %88 = arith.sitofp %87 : i32 to f32
      %89 = arith.mulf %88, %cst_26 : f32
      %90 = affine.load %alloca[%arg0, 1] : memref<32x3xf32>
      %91 = arith.subf %90, %89 : f32
      affine.store %91, %alloca[%arg0, 1] : memref<32x3xf32>
      %92 = arith.addi %75, %c-1_i32 : i32
      %93 = arith.sitofp %92 : i32 to f32
      %94 = arith.mulf %93, %cst_27 : f32
      %95 = affine.load %alloca[%arg0, 2] : memref<32x3xf32>
      %96 = arith.addf %95, %94 : f32
      affine.store %96, %alloca[%arg0, 2] : memref<32x3xf32>
      %97 = arith.remsi %17, %c2_i32 : i32
      %98 = arith.sitofp %97 : i32 to f32
      %99 = arith.mulf %98, %cst_28 : f32
      %100 = affine.load %alloca_31[%arg0, 0] : memref<32x3xf32>
      %101 = arith.addf %100, %99 : f32
      affine.store %101, %alloca_31[%arg0, 0] : memref<32x3xf32>
      %102 = arith.sitofp %69 : i32 to f32
      %103 = arith.mulf %102, %cst_29 : f32
      %104 = affine.load %alloca_31[%arg0, 1] : memref<32x3xf32>
      %105 = arith.subf %104, %103 : f32
      affine.store %105, %alloca_31[%arg0, 1] : memref<32x3xf32>
      %106 = arith.mulf %93, %cst_11 : f32
      %107 = affine.load %alloca_31[%arg0, 2] : memref<32x3xf32>
      %108 = arith.addf %107, %106 : f32
      affine.store %108, %alloca_31[%arg0, 2] : memref<32x3xf32>
    }
    %alloca_33 = memref.alloca() : memref<3xf32>
    %alloca_34 = memref.alloca() : memref<3xf32>
    %alloca_35 = memref.alloca() : memref<32x3xf32>
    %alloca_36 = memref.alloca() : memref<32x3xf32>
    %alloca_37 = memref.alloca() : memref<3xf32>
    %alloca_38 = memref.alloca() : memref<3xf32>
    affine.for %arg0 = 0 to 10 {
      affine.store %cst_23, %alloca_38[0] : memref<3xf32>
      affine.store %cst_23, %alloca_38[1] : memref<3xf32>
      affine.store %cst_23, %alloca_38[2] : memref<3xf32>
      affine.store %cst_5, %alloca_37[0] : memref<3xf32>
      affine.store %cst_5, %alloca_37[1] : memref<3xf32>
      affine.store %cst_5, %alloca_37[2] : memref<3xf32>
      affine.for %arg1 = 0 to 32 {
        affine.store %cst_23, %alloca_36[%arg1, 0] : memref<32x3xf32>
        affine.store %cst_23, %alloca_36[%arg1, 1] : memref<32x3xf32>
        affine.store %cst_23, %alloca_36[%arg1, 2] : memref<32x3xf32>
        affine.store %cst_5, %alloca_35[%arg1, 0] : memref<32x3xf32>
        affine.store %cst_5, %alloca_35[%arg1, 1] : memref<32x3xf32>
        affine.store %cst_5, %alloca_35[%arg1, 2] : memref<32x3xf32>
      }
      affine.for %arg1 = 0 to 32 {
        %35 = affine.load %alloca_31[%arg1, 0] : memref<32x3xf32>
        %36 = affine.load %alloca_31[%arg1, 1] : memref<32x3xf32>
        %37 = affine.load %alloca_31[%arg1, 2] : memref<32x3xf32>
        %38 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
        %39 = affine.load %38[0] : memref<1xf32>
        %40 = arith.addf %37, %39 : f32
        %41 = arith.divf %cst_9, %40 : f32
        %42 = memref.get_global @_ZL2FX : memref<1xf32>
        %43 = affine.load %42[0] : memref<1xf32>
        %44 = arith.mulf %43, %35 : f32
        %45 = arith.mulf %44, %41 : f32
        affine.store %45, %alloca_34[0] : memref<3xf32>
        %46 = memref.get_global @_ZL2FY : memref<1xf32>
        %47 = affine.load %46[0] : memref<1xf32>
        %48 = arith.mulf %47, %36 : f32
        %49 = arith.mulf %48, %41 : f32
        affine.store %49, %alloca_34[1] : memref<3xf32>
        %50 = affine.load %42[0] : memref<1xf32>
        %51 = memref.get_global @_ZL8BASELINE : memref<1xf32>
        %52 = affine.load %51[0] : memref<1xf32>
        %53 = arith.mulf %50, %52 : f32
        %54 = arith.mulf %53, %41 : f32
        affine.store %54, %alloca_34[2] : memref<3xf32>
        %55 = affine.load %alloca_32[0] : memref<3xf32>
        %56 = affine.load %alloca_32[1] : memref<3xf32>
        %57 = affine.load %alloca_32[2] : memref<3xf32>
        %58 = affine.load %alloca_31[%arg1, 0] : memref<32x3xf32>
        %59 = arith.subf %58, %55 : f32
        %60 = affine.load %alloca_31[%arg1, 1] : memref<32x3xf32>
        %61 = arith.subf %60, %56 : f32
        %62 = affine.load %alloca_31[%arg1, 2] : memref<32x3xf32>
        %63 = arith.subf %62, %57 : f32
        %64 = affine.load %38[0] : memref<1xf32>
        %65 = arith.addf %63, %64 : f32
        %66 = arith.divf %cst_9, %65 : f32
        %67 = affine.load %42[0] : memref<1xf32>
        %68 = arith.mulf %67, %59 : f32
        %69 = arith.mulf %68, %66 : f32
        affine.store %69, %alloca_33[0] : memref<3xf32>
        %70 = affine.load %46[0] : memref<1xf32>
        %71 = arith.mulf %70, %61 : f32
        %72 = arith.mulf %71, %66 : f32
        affine.store %72, %alloca_33[1] : memref<3xf32>
        %73 = affine.load %42[0] : memref<1xf32>
        %74 = affine.load %51[0] : memref<1xf32>
        %75 = arith.mulf %73, %74 : f32
        %76 = arith.mulf %75, %66 : f32
        affine.store %76, %alloca_33[2] : memref<3xf32>
        %77 = affine.load %alloca_31[%arg1, 2] : memref<32x3xf32>
        %78 = affine.load %38[0] : memref<1xf32>
        %79 = arith.addf %77, %78 : f32
        %80 = arith.divf %cst_9, %79 : f32
        %81 = affine.load %alloca_34[0] : memref<3xf32>
        %82 = affine.load %alloca_30[%arg1, 0] : memref<32x3xf32>
        %83 = arith.subf %81, %82 : f32
        %84 = affine.load %alloca_34[1] : memref<3xf32>
        %85 = affine.load %alloca_30[%arg1, 1] : memref<32x3xf32>
        %86 = arith.subf %84, %85 : f32
        %87 = affine.load %alloca_34[2] : memref<3xf32>
        %88 = affine.load %alloca_30[%arg1, 2] : memref<32x3xf32>
        %89 = arith.subf %87, %88 : f32
        %90 = affine.load %42[0] : memref<1xf32>
        %91 = arith.mulf %90, %80 : f32
        %92 = affine.load %46[0] : memref<1xf32>
        %93 = arith.mulf %92, %80 : f32
        %94 = arith.negf %90 : f32
        %95 = affine.load %alloca_31[%arg1, 0] : memref<32x3xf32>
        %96 = arith.mulf %94, %95 : f32
        %97 = arith.mulf %96, %80 : f32
        %98 = arith.mulf %97, %80 : f32
        %99 = arith.negf %92 : f32
        %100 = affine.load %alloca_31[%arg1, 1] : memref<32x3xf32>
        %101 = arith.mulf %99, %100 : f32
        %102 = arith.mulf %101, %80 : f32
        %103 = arith.mulf %102, %80 : f32
        %104 = affine.load %51[0] : memref<1xf32>
        %105 = arith.mulf %94, %104 : f32
        %106 = arith.mulf %105, %80 : f32
        %107 = arith.mulf %106, %80 : f32
        %108 = affine.load %alloca_32[0] : memref<3xf32>
        %109 = arith.subf %95, %108 : f32
        %110 = affine.load %alloca_32[1] : memref<3xf32>
        %111 = arith.subf %100, %110 : f32
        %112 = affine.load %alloca_32[2] : memref<3xf32>
        %113 = arith.subf %77, %112 : f32
        %114 = arith.addf %113, %78 : f32
        %115 = arith.divf %cst_9, %114 : f32
        %116 = affine.load %alloca_33[0] : memref<3xf32>
        %117 = affine.load %alloca[%arg1, 0] : memref<32x3xf32>
        %118 = arith.subf %116, %117 : f32
        %119 = affine.load %alloca_33[1] : memref<3xf32>
        %120 = affine.load %alloca[%arg1, 1] : memref<32x3xf32>
        %121 = arith.subf %119, %120 : f32
        %122 = affine.load %alloca_33[2] : memref<3xf32>
        %123 = affine.load %alloca[%arg1, 2] : memref<32x3xf32>
        %124 = arith.subf %122, %123 : f32
        %125 = arith.mulf %90, %115 : f32
        %126 = arith.mulf %92, %115 : f32
        %127 = arith.mulf %94, %109 : f32
        %128 = arith.mulf %127, %115 : f32
        %129 = arith.mulf %128, %115 : f32
        %130 = arith.mulf %99, %111 : f32
        %131 = arith.mulf %130, %115 : f32
        %132 = arith.mulf %131, %115 : f32
        %133 = arith.mulf %105, %115 : f32
        %134 = arith.mulf %133, %115 : f32
        %135 = arith.negf %118 : f32
        %136 = arith.mulf %135, %125 : f32
        %137 = affine.load %alloca_38[0] : memref<3xf32>
        %138 = arith.addf %137, %136 : f32
        affine.store %138, %alloca_38[0] : memref<3xf32>
        %139 = arith.negf %121 : f32
        %140 = arith.mulf %139, %126 : f32
        %141 = affine.load %alloca_38[1] : memref<3xf32>
        %142 = arith.addf %141, %140 : f32
        affine.store %142, %alloca_38[1] : memref<3xf32>
        %143 = arith.mulf %118, %129 : f32
        %144 = arith.mulf %121, %132 : f32
        %145 = arith.addf %143, %144 : f32
        %146 = arith.mulf %124, %134 : f32
        %147 = arith.addf %145, %146 : f32
        %148 = arith.negf %147 : f32
        %149 = affine.load %alloca_38[2] : memref<3xf32>
        %150 = arith.addf %149, %148 : f32
        affine.store %150, %alloca_38[2] : memref<3xf32>
        %151 = arith.mulf %125, %125 : f32
        %152 = affine.load %alloca_37[0] : memref<3xf32>
        %153 = arith.addf %152, %151 : f32
        affine.store %153, %alloca_37[0] : memref<3xf32>
        %154 = arith.mulf %126, %126 : f32
        %155 = affine.load %alloca_37[1] : memref<3xf32>
        %156 = arith.addf %155, %154 : f32
        affine.store %156, %alloca_37[1] : memref<3xf32>
        %157 = arith.mulf %129, %129 : f32
        %158 = arith.mulf %132, %132 : f32
        %159 = arith.addf %157, %158 : f32
        %160 = arith.mulf %134, %134 : f32
        %161 = arith.addf %159, %160 : f32
        %162 = affine.load %alloca_37[2] : memref<3xf32>
        %163 = arith.addf %162, %161 : f32
        affine.store %163, %alloca_37[2] : memref<3xf32>
        %164 = arith.mulf %83, %91 : f32
        %165 = affine.load %alloca_36[%arg1, 0] : memref<32x3xf32>
        %166 = arith.addf %165, %164 : f32
        affine.store %166, %alloca_36[%arg1, 0] : memref<32x3xf32>
        %167 = arith.mulf %86, %93 : f32
        %168 = affine.load %alloca_36[%arg1, 1] : memref<32x3xf32>
        %169 = arith.addf %168, %167 : f32
        affine.store %169, %alloca_36[%arg1, 1] : memref<32x3xf32>
        %170 = arith.mulf %83, %98 : f32
        %171 = arith.mulf %86, %103 : f32
        %172 = arith.addf %170, %171 : f32
        %173 = arith.mulf %89, %107 : f32
        %174 = arith.addf %172, %173 : f32
        %175 = affine.load %alloca_36[%arg1, 2] : memref<32x3xf32>
        %176 = arith.addf %175, %174 : f32
        affine.store %176, %alloca_36[%arg1, 2] : memref<32x3xf32>
        %177 = arith.mulf %91, %91 : f32
        %178 = affine.load %alloca_35[%arg1, 0] : memref<32x3xf32>
        %179 = arith.addf %178, %177 : f32
        affine.store %179, %alloca_35[%arg1, 0] : memref<32x3xf32>
        %180 = arith.mulf %93, %93 : f32
        %181 = affine.load %alloca_35[%arg1, 1] : memref<32x3xf32>
        %182 = arith.addf %181, %180 : f32
        affine.store %182, %alloca_35[%arg1, 1] : memref<32x3xf32>
        %183 = arith.mulf %98, %98 : f32
        %184 = arith.mulf %103, %103 : f32
        %185 = arith.addf %183, %184 : f32
        %186 = arith.mulf %107, %107 : f32
        %187 = arith.addf %185, %186 : f32
        %188 = affine.load %alloca_35[%arg1, 2] : memref<32x3xf32>
        %189 = arith.addf %188, %187 : f32
        affine.store %189, %alloca_35[%arg1, 2] : memref<32x3xf32>
        %190 = arith.mulf %118, %125 : f32
        %191 = affine.load %alloca_36[%arg1, 0] : memref<32x3xf32>
        %192 = arith.addf %191, %190 : f32
        affine.store %192, %alloca_36[%arg1, 0] : memref<32x3xf32>
        %193 = arith.mulf %121, %126 : f32
        %194 = affine.load %alloca_36[%arg1, 1] : memref<32x3xf32>
        %195 = arith.addf %194, %193 : f32
        affine.store %195, %alloca_36[%arg1, 1] : memref<32x3xf32>
        %196 = affine.load %alloca_36[%arg1, 2] : memref<32x3xf32>
        %197 = arith.addf %196, %147 : f32
        affine.store %197, %alloca_36[%arg1, 2] : memref<32x3xf32>
        %198 = affine.load %alloca_35[%arg1, 0] : memref<32x3xf32>
        %199 = arith.addf %198, %151 : f32
        affine.store %199, %alloca_35[%arg1, 0] : memref<32x3xf32>
        %200 = affine.load %alloca_35[%arg1, 1] : memref<32x3xf32>
        %201 = arith.addf %200, %154 : f32
        affine.store %201, %alloca_35[%arg1, 1] : memref<32x3xf32>
        %202 = affine.load %alloca_35[%arg1, 2] : memref<32x3xf32>
        %203 = arith.addf %202, %161 : f32
        affine.store %203, %alloca_35[%arg1, 2] : memref<32x3xf32>
      }
      %17 = affine.load %alloca_38[0] : memref<3xf32>
      %18 = arith.mulf %17, %cst_6 : f32
      %19 = affine.load %alloca_37[0] : memref<3xf32>
      %20 = arith.divf %18, %19 : f32
      %21 = affine.load %alloca_32[0] : memref<3xf32>
      %22 = arith.subf %21, %20 : f32
      affine.store %22, %alloca_32[0] : memref<3xf32>
      %23 = affine.load %alloca_38[1] : memref<3xf32>
      %24 = arith.mulf %23, %cst_6 : f32
      %25 = affine.load %alloca_37[1] : memref<3xf32>
      %26 = arith.divf %24, %25 : f32
      %27 = affine.load %alloca_32[1] : memref<3xf32>
      %28 = arith.subf %27, %26 : f32
      affine.store %28, %alloca_32[1] : memref<3xf32>
      %29 = affine.load %alloca_38[2] : memref<3xf32>
      %30 = arith.mulf %29, %cst_7 : f32
      %31 = affine.load %alloca_37[2] : memref<3xf32>
      %32 = arith.divf %30, %31 : f32
      %33 = affine.load %alloca_32[2] : memref<3xf32>
      %34 = arith.subf %33, %32 : f32
      affine.store %34, %alloca_32[2] : memref<3xf32>
      affine.for %arg1 = 0 to 32 {
        %35 = affine.load %alloca_36[%arg1, 0] : memref<32x3xf32>
        %36 = arith.mulf %35, %cst_8 : f32
        %37 = affine.load %alloca_35[%arg1, 0] : memref<32x3xf32>
        %38 = arith.divf %36, %37 : f32
        %39 = affine.load %alloca_31[%arg1, 0] : memref<32x3xf32>
        %40 = arith.subf %39, %38 : f32
        affine.store %40, %alloca_31[%arg1, 0] : memref<32x3xf32>
        %41 = affine.load %alloca_36[%arg1, 1] : memref<32x3xf32>
        %42 = arith.mulf %41, %cst_8 : f32
        %43 = affine.load %alloca_35[%arg1, 1] : memref<32x3xf32>
        %44 = arith.divf %42, %43 : f32
        %45 = affine.load %alloca_31[%arg1, 1] : memref<32x3xf32>
        %46 = arith.subf %45, %44 : f32
        affine.store %46, %alloca_31[%arg1, 1] : memref<32x3xf32>
        %47 = affine.load %alloca_36[%arg1, 2] : memref<32x3xf32>
        %48 = arith.mulf %47, %cst_14 : f32
        %49 = affine.load %alloca_35[%arg1, 2] : memref<32x3xf32>
        %50 = arith.divf %48, %49 : f32
        %51 = affine.load %alloca_31[%arg1, 2] : memref<32x3xf32>
        %52 = arith.subf %51, %50 : f32
        affine.store %52, %alloca_31[%arg1, 2] : memref<32x3xf32>
      }
    }
    %alloca_39 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_39[] : memref<i32>
    %0 = affine.load %alloca_32[0] : memref<3xf32>
    %1 = arith.mulf %0, %cst : f32
    %2 = arith.fptosi %1 : f32 to i32
    %3 = affine.load %alloca_32[1] : memref<3xf32>
    %4 = arith.mulf %3, %cst_0 : f32
    %5 = arith.fptosi %4 : f32 to i32
    %6 = arith.addi %2, %5 : i32
    %7 = affine.load %alloca_32[2] : memref<3xf32>
    %8 = arith.mulf %7, %cst_1 : f32
    %9 = arith.fptosi %8 : f32 to i32
    %10 = arith.addi %6, %9 : i32
    affine.store %10, %alloca_39[] : memref<i32>
    affine.for %arg0 = 0 to 32 {
      %17 = affine.load %alloca_31[%arg0, 0] : memref<32x3xf32>
      %18 = arith.mulf %17, %cst_2 : f32
      %19 = arith.fptosi %18 : f32 to i32
      %20 = affine.load %alloca_39[] : memref<i32>
      %21 = arith.addi %20, %19 : i32
      affine.store %21, %alloca_39[] : memref<i32>
      %22 = affine.load %alloca_31[%arg0, 1] : memref<32x3xf32>
      %23 = arith.mulf %22, %cst_3 : f32
      %24 = arith.fptosi %23 : f32 to i32
      %25 = arith.addi %21, %24 : i32
      affine.store %25, %alloca_39[] : memref<i32>
      %26 = affine.load %alloca_31[%arg0, 2] : memref<32x3xf32>
      %27 = arith.mulf %26, %cst_4 : f32
      %28 = arith.fptosi %27 : f32 to i32
      %29 = arith.addi %25, %28 : i32
      affine.store %29, %alloca_39[] : memref<i32>
    }
    %11 = affine.load %alloca_39[] : memref<i32>
    %12 = arith.sitofp %11 : i32 to f32
    %13 = arith.cmpi slt, %11, %c0_i32 : i32
    %14 = scf.if %13 -> (f32) {
      %17 = arith.negf %12 : f32
      scf.yield %17 : f32
    } else {
      scf.yield %12 : f32
    }
    %15 = arith.cmpf ogt, %14, %cst_23 : f32
    %16 = arith.select %15, %11, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_32, %alloca_31, %alloca_30, %alloca) : (memref<3xf32>, memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>) -> ()
    return %16 : i32
  }
  func.func private @pcl_validate_outputs(memref<3xf32>, memref<32x3xf32>, memref<32x3xf32>, memref<32x3xf32>)
}

