module {
  memref.global "private" @_ZL8BASELINE : memref<1xf32> = dense<2.000000e-01>
  memref.global "private" @_ZL2FY : memref<1xf32> = dense<1.780000e+02>
  memref.global "private" @_ZL2FX : memref<1xf32> = dense<1.800000e+02>
  memref.global "private" @_ZL10DEPTH_BIAS : memref<1xf32> = dense<4.200000e+00>
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<32x3xf32>
    %alloca_0 = memref.alloca() : memref<32x3xf32>
    %alloca_1 = memref.alloca() : memref<32x3xf32>
    %alloca_2 = memref.alloca() : memref<3xf32>
    %cast = memref.cast %alloca_2 : memref<3xf32> to memref<?xf32>
    %cast_3 = memref.cast %alloca_1 : memref<32x3xf32> to memref<?x3xf32>
    %cast_4 = memref.cast %alloca_0 : memref<32x3xf32> to memref<?x3xf32>
    %cast_5 = memref.cast %alloca : memref<32x3xf32> to memref<?x3xf32>
    call @_ZL19init_stereo_problemPfPA3_fS1_S1_(%cast, %cast_3, %cast_4, %cast_5) : (memref<?xf32>, memref<?x3xf32>, memref<?x3xf32>, memref<?x3xf32>) -> ()
    call @_ZL15optimize_stereoPfPA3_fS1_S1_(%cast, %cast_3, %cast_4, %cast_5) : (memref<?xf32>, memref<?x3xf32>, memref<?x3xf32>, memref<?x3xf32>) -> ()
    %0 = call @_ZL9summarizePfPA3_f(%cast, %cast_3) : (memref<?xf32>, memref<?x3xf32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL19init_stereo_problemPfPA3_fS1_S1_(%arg0: memref<?xf32>, %arg1: memref<?x3xf32>, %arg2: memref<?x3xf32>, %arg3: memref<?x3xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-2_i32 = arith.constant -2 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 5.000000e-02 : f32
    %cst_0 = arith.constant 6.000000e-02 : f32
    %cst_1 = arith.constant 8.000000e-03 : f32
    %c6_i32 = arith.constant 6 : i32
    %cst_2 = arith.constant 1.200000e-02 : f32
    %cst_3 = arith.constant 1.500000e-02 : f32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %cst_4 = arith.constant 0.00999999977 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_6 = arith.constant 1.100000e-01 : f32
    %cst_7 = arith.constant 1.600000e+00 : f32
    %cst_8 = arith.constant 3.100000e-01 : f32
    %cst_9 = arith.constant -6.000000e-01 : f32
    %c8_i32 = arith.constant 8 : i32
    %cst_10 = arith.constant 4.200000e-01 : f32
    %cst_11 = arith.constant -1.000000e+00 : f32
    %cst_12 = arith.constant -2.000000e-02 : f32
    %cst_13 = arith.constant 3.000000e-02 : f32
    %cst_14 = arith.constant 1.200000e-01 : f32
    %cst_15 = arith.constant 8.000000e-02 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_16 = arith.constant -4.000000e-02 : f32
    %cst_17 = arith.constant 4.000000e-02 : f32
    %cst_18 = arith.constant 3.500000e-01 : f32
    affine.store %cst_14, %arg0[0] : memref<?xf32>
    affine.store %cst_13, %arg0[1] : memref<?xf32>
    affine.store %cst_12, %arg0[2] : memref<?xf32>
    affine.for %arg4 = 0 to 32 {
      %0 = arith.index_cast %arg4 : index to i32
      %1 = arith.remsi %0, %c8_i32 : i32
      %2 = arith.sitofp %1 : i32 to f32
      %3 = arith.mulf %2, %cst_10 : f32
      %4 = arith.addf %3, %cst_11 : f32
      affine.store %4, %arg1[%arg4, 0] : memref<?x3xf32>
      %5 = arith.divsi %0, %c8_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.mulf %6, %cst_8 : f32
      %8 = arith.addf %7, %cst_9 : f32
      affine.store %8, %arg1[%arg4, 1] : memref<?x3xf32>
      %9 = arith.remsi %0, %c5_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.mulf %10, %cst_6 : f32
      %12 = arith.addf %11, %cst_7 : f32
      affine.store %12, %arg1[%arg4, 2] : memref<?x3xf32>
      %13 = "polygeist.subindex"(%arg1, %arg4) : (memref<?x3xf32>, index) -> memref<?xf32>
      %14 = "polygeist.subindex"(%arg2, %arg4) : (memref<?x3xf32>, index) -> memref<?xf32>
      func.call @_ZL14stereo_projectfffPfS_(%cst_5, %cst_5, %cst_5, %13, %14) : (f32, f32, f32, memref<?xf32>, memref<?xf32>) -> ()
      %15 = "polygeist.subindex"(%arg3, %arg4) : (memref<?x3xf32>, index) -> memref<?xf32>
      func.call @_ZL14stereo_projectfffPfS_(%cst_18, %cst_16, %cst_15, %13, %15) : (f32, f32, f32, memref<?xf32>, memref<?xf32>) -> ()
      %16 = arith.remsi %0, %c3_i32 : i32
      %17 = arith.addi %16, %c-1_i32 : i32
      %18 = arith.sitofp %17 : i32 to f32
      %19 = arith.mulf %18, %cst_4 : f32
      %20 = affine.load %arg2[%arg4, 0] : memref<?x3xf32>
      %21 = arith.addf %20, %19 : f32
      affine.store %21, %arg2[%arg4, 0] : memref<?x3xf32>
      %22 = arith.remsi %0, %c4_i32 : i32
      %23 = arith.addi %22, %c-2_i32 : i32
      %24 = arith.sitofp %23 : i32 to f32
      %25 = arith.mulf %24, %cst_4 : f32
      %26 = affine.load %arg2[%arg4, 1] : memref<?x3xf32>
      %27 = arith.subf %26, %25 : f32
      affine.store %27, %arg2[%arg4, 1] : memref<?x3xf32>
      %28 = arith.addi %9, %c-2_i32 : i32
      %29 = arith.sitofp %28 : i32 to f32
      %30 = arith.mulf %29, %cst_3 : f32
      %31 = affine.load %arg3[%arg4, 0] : memref<?x3xf32>
      %32 = arith.addf %31, %30 : f32
      affine.store %32, %arg3[%arg4, 0] : memref<?x3xf32>
      %33 = arith.remsi %0, %c6_i32 : i32
      %34 = arith.addi %33, %c-3_i32 : i32
      %35 = arith.sitofp %34 : i32 to f32
      %36 = arith.mulf %35, %cst_2 : f32
      %37 = affine.load %arg3[%arg4, 1] : memref<?x3xf32>
      %38 = arith.subf %37, %36 : f32
      affine.store %38, %arg3[%arg4, 1] : memref<?x3xf32>
      %39 = arith.addi %22, %c-1_i32 : i32
      %40 = arith.sitofp %39 : i32 to f32
      %41 = arith.mulf %40, %cst_1 : f32
      %42 = affine.load %arg3[%arg4, 2] : memref<?x3xf32>
      %43 = arith.addf %42, %41 : f32
      affine.store %43, %arg3[%arg4, 2] : memref<?x3xf32>
      %44 = arith.remsi %0, %c2_i32 : i32
      %45 = arith.sitofp %44 : i32 to f32
      %46 = arith.mulf %45, %cst_0 : f32
      %47 = affine.load %arg1[%arg4, 0] : memref<?x3xf32>
      %48 = arith.addf %47, %46 : f32
      affine.store %48, %arg1[%arg4, 0] : memref<?x3xf32>
      %49 = arith.sitofp %16 : i32 to f32
      %50 = arith.mulf %49, %cst : f32
      %51 = affine.load %arg1[%arg4, 1] : memref<?x3xf32>
      %52 = arith.subf %51, %50 : f32
      affine.store %52, %arg1[%arg4, 1] : memref<?x3xf32>
      %53 = arith.mulf %40, %cst_17 : f32
      %54 = affine.load %arg1[%arg4, 2] : memref<?x3xf32>
      %55 = arith.addf %54, %53 : f32
      affine.store %55, %arg1[%arg4, 2] : memref<?x3xf32>
    }
    return
  }
  func.func private @_ZL15optimize_stereoPfPA3_fS1_S1_(%arg0: memref<?xf32>, %arg1: memref<?x3xf32>, %arg2: memref<?x3xf32>, %arg3: memref<?x3xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 1.200000e-01 : f32
    %cst_0 = arith.constant 1.600000e-01 : f32
    %cst_1 = arith.constant 1.800000e-01 : f32
    %cst_2 = arith.constant 2.500000e-01 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant 1.000000e-03 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<3xf32>
    %alloca_6 = memref.alloca() : memref<3xf32>
    %alloca_7 = memref.alloca() : memref<32x3xf32>
    %alloca_8 = memref.alloca() : memref<32x3xf32>
    %alloca_9 = memref.alloca() : memref<3xf32>
    %alloca_10 = memref.alloca() : memref<3xf32>
    affine.for %arg4 = 0 to 10 {
      affine.store %cst_5, %alloca_10[0] : memref<3xf32>
      affine.store %cst_5, %alloca_10[1] : memref<3xf32>
      affine.store %cst_5, %alloca_10[2] : memref<3xf32>
      affine.store %cst_4, %alloca_9[0] : memref<3xf32>
      affine.store %cst_4, %alloca_9[1] : memref<3xf32>
      affine.store %cst_4, %alloca_9[2] : memref<3xf32>
      affine.for %arg5 = 0 to 32 {
        affine.store %cst_5, %alloca_8[%arg5, 0] : memref<32x3xf32>
        affine.store %cst_5, %alloca_8[%arg5, 1] : memref<32x3xf32>
        affine.store %cst_5, %alloca_8[%arg5, 2] : memref<32x3xf32>
        affine.store %cst_4, %alloca_7[%arg5, 0] : memref<32x3xf32>
        affine.store %cst_4, %alloca_7[%arg5, 1] : memref<32x3xf32>
        affine.store %cst_4, %alloca_7[%arg5, 2] : memref<32x3xf32>
      }
      affine.for %arg5 = 0 to 32 {
        %18 = "polygeist.subindex"(%arg1, %arg5) : (memref<?x3xf32>, index) -> memref<?xf32>
        %cast = memref.cast %alloca_6 : memref<3xf32> to memref<?xf32>
        func.call @_ZL14stereo_projectfffPfS_(%cst_5, %cst_5, %cst_5, %18, %cast) : (f32, f32, f32, memref<?xf32>, memref<?xf32>) -> ()
        %19 = affine.load %arg0[0] : memref<?xf32>
        %20 = affine.load %arg0[1] : memref<?xf32>
        %21 = affine.load %arg0[2] : memref<?xf32>
        %cast_11 = memref.cast %alloca : memref<3xf32> to memref<?xf32>
        func.call @_ZL14stereo_projectfffPfS_(%19, %20, %21, %18, %cast_11) : (f32, f32, f32, memref<?xf32>, memref<?xf32>) -> ()
        %22 = affine.load %arg1[%arg5, 2] : memref<?x3xf32>
        %23 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
        %24 = affine.load %23[0] : memref<1xf32>
        %25 = arith.addf %22, %24 : f32
        %26 = arith.divf %cst_3, %25 : f32
        %27 = affine.load %alloca_6[0] : memref<3xf32>
        %28 = affine.load %arg2[%arg5, 0] : memref<?x3xf32>
        %29 = arith.subf %27, %28 : f32
        %30 = affine.load %alloca_6[1] : memref<3xf32>
        %31 = affine.load %arg2[%arg5, 1] : memref<?x3xf32>
        %32 = arith.subf %30, %31 : f32
        %33 = affine.load %alloca_6[2] : memref<3xf32>
        %34 = affine.load %arg2[%arg5, 2] : memref<?x3xf32>
        %35 = arith.subf %33, %34 : f32
        %36 = memref.get_global @_ZL2FX : memref<1xf32>
        %37 = affine.load %36[0] : memref<1xf32>
        %38 = arith.mulf %37, %26 : f32
        %39 = memref.get_global @_ZL2FY : memref<1xf32>
        %40 = affine.load %39[0] : memref<1xf32>
        %41 = arith.mulf %40, %26 : f32
        %42 = arith.negf %37 : f32
        %43 = affine.load %arg1[%arg5, 0] : memref<?x3xf32>
        %44 = arith.mulf %42, %43 : f32
        %45 = arith.mulf %44, %26 : f32
        %46 = arith.mulf %45, %26 : f32
        %47 = arith.negf %40 : f32
        %48 = affine.load %arg1[%arg5, 1] : memref<?x3xf32>
        %49 = arith.mulf %47, %48 : f32
        %50 = arith.mulf %49, %26 : f32
        %51 = arith.mulf %50, %26 : f32
        %52 = memref.get_global @_ZL8BASELINE : memref<1xf32>
        %53 = affine.load %52[0] : memref<1xf32>
        %54 = arith.mulf %42, %53 : f32
        %55 = arith.mulf %54, %26 : f32
        %56 = arith.mulf %55, %26 : f32
        %57 = affine.load %arg0[0] : memref<?xf32>
        %58 = arith.subf %43, %57 : f32
        %59 = affine.load %arg0[1] : memref<?xf32>
        %60 = arith.subf %48, %59 : f32
        %61 = affine.load %arg0[2] : memref<?xf32>
        %62 = arith.subf %22, %61 : f32
        %63 = arith.addf %62, %24 : f32
        %64 = arith.divf %cst_3, %63 : f32
        %65 = affine.load %alloca[0] : memref<3xf32>
        %66 = affine.load %arg3[%arg5, 0] : memref<?x3xf32>
        %67 = arith.subf %65, %66 : f32
        %68 = affine.load %alloca[1] : memref<3xf32>
        %69 = affine.load %arg3[%arg5, 1] : memref<?x3xf32>
        %70 = arith.subf %68, %69 : f32
        %71 = affine.load %alloca[2] : memref<3xf32>
        %72 = affine.load %arg3[%arg5, 2] : memref<?x3xf32>
        %73 = arith.subf %71, %72 : f32
        %74 = arith.mulf %37, %64 : f32
        %75 = arith.mulf %40, %64 : f32
        %76 = arith.mulf %42, %58 : f32
        %77 = arith.mulf %76, %64 : f32
        %78 = arith.mulf %77, %64 : f32
        %79 = arith.mulf %47, %60 : f32
        %80 = arith.mulf %79, %64 : f32
        %81 = arith.mulf %80, %64 : f32
        %82 = arith.mulf %54, %64 : f32
        %83 = arith.mulf %82, %64 : f32
        %84 = arith.negf %67 : f32
        %85 = arith.mulf %84, %74 : f32
        %86 = affine.load %alloca_10[0] : memref<3xf32>
        %87 = arith.addf %86, %85 : f32
        affine.store %87, %alloca_10[0] : memref<3xf32>
        %88 = arith.negf %70 : f32
        %89 = arith.mulf %88, %75 : f32
        %90 = affine.load %alloca_10[1] : memref<3xf32>
        %91 = arith.addf %90, %89 : f32
        affine.store %91, %alloca_10[1] : memref<3xf32>
        %92 = arith.mulf %67, %78 : f32
        %93 = arith.mulf %70, %81 : f32
        %94 = arith.addf %92, %93 : f32
        %95 = arith.mulf %73, %83 : f32
        %96 = arith.addf %94, %95 : f32
        %97 = arith.negf %96 : f32
        %98 = affine.load %alloca_10[2] : memref<3xf32>
        %99 = arith.addf %98, %97 : f32
        affine.store %99, %alloca_10[2] : memref<3xf32>
        %100 = arith.mulf %74, %74 : f32
        %101 = affine.load %alloca_9[0] : memref<3xf32>
        %102 = arith.addf %101, %100 : f32
        affine.store %102, %alloca_9[0] : memref<3xf32>
        %103 = arith.mulf %75, %75 : f32
        %104 = affine.load %alloca_9[1] : memref<3xf32>
        %105 = arith.addf %104, %103 : f32
        affine.store %105, %alloca_9[1] : memref<3xf32>
        %106 = arith.mulf %78, %78 : f32
        %107 = arith.mulf %81, %81 : f32
        %108 = arith.addf %106, %107 : f32
        %109 = arith.mulf %83, %83 : f32
        %110 = arith.addf %108, %109 : f32
        %111 = affine.load %alloca_9[2] : memref<3xf32>
        %112 = arith.addf %111, %110 : f32
        affine.store %112, %alloca_9[2] : memref<3xf32>
        %113 = arith.mulf %29, %38 : f32
        %114 = affine.load %alloca_8[%arg5, 0] : memref<32x3xf32>
        %115 = arith.addf %114, %113 : f32
        affine.store %115, %alloca_8[%arg5, 0] : memref<32x3xf32>
        %116 = arith.mulf %32, %41 : f32
        %117 = affine.load %alloca_8[%arg5, 1] : memref<32x3xf32>
        %118 = arith.addf %117, %116 : f32
        affine.store %118, %alloca_8[%arg5, 1] : memref<32x3xf32>
        %119 = arith.mulf %29, %46 : f32
        %120 = arith.mulf %32, %51 : f32
        %121 = arith.addf %119, %120 : f32
        %122 = arith.mulf %35, %56 : f32
        %123 = arith.addf %121, %122 : f32
        %124 = affine.load %alloca_8[%arg5, 2] : memref<32x3xf32>
        %125 = arith.addf %124, %123 : f32
        affine.store %125, %alloca_8[%arg5, 2] : memref<32x3xf32>
        %126 = arith.mulf %38, %38 : f32
        %127 = affine.load %alloca_7[%arg5, 0] : memref<32x3xf32>
        %128 = arith.addf %127, %126 : f32
        affine.store %128, %alloca_7[%arg5, 0] : memref<32x3xf32>
        %129 = arith.mulf %41, %41 : f32
        %130 = affine.load %alloca_7[%arg5, 1] : memref<32x3xf32>
        %131 = arith.addf %130, %129 : f32
        affine.store %131, %alloca_7[%arg5, 1] : memref<32x3xf32>
        %132 = arith.mulf %46, %46 : f32
        %133 = arith.mulf %51, %51 : f32
        %134 = arith.addf %132, %133 : f32
        %135 = arith.mulf %56, %56 : f32
        %136 = arith.addf %134, %135 : f32
        %137 = affine.load %alloca_7[%arg5, 2] : memref<32x3xf32>
        %138 = arith.addf %137, %136 : f32
        affine.store %138, %alloca_7[%arg5, 2] : memref<32x3xf32>
        %139 = arith.mulf %67, %74 : f32
        %140 = affine.load %alloca_8[%arg5, 0] : memref<32x3xf32>
        %141 = arith.addf %140, %139 : f32
        affine.store %141, %alloca_8[%arg5, 0] : memref<32x3xf32>
        %142 = arith.mulf %70, %75 : f32
        %143 = affine.load %alloca_8[%arg5, 1] : memref<32x3xf32>
        %144 = arith.addf %143, %142 : f32
        affine.store %144, %alloca_8[%arg5, 1] : memref<32x3xf32>
        %145 = affine.load %alloca_8[%arg5, 2] : memref<32x3xf32>
        %146 = arith.addf %145, %96 : f32
        affine.store %146, %alloca_8[%arg5, 2] : memref<32x3xf32>
        %147 = affine.load %alloca_7[%arg5, 0] : memref<32x3xf32>
        %148 = arith.addf %147, %100 : f32
        affine.store %148, %alloca_7[%arg5, 0] : memref<32x3xf32>
        %149 = affine.load %alloca_7[%arg5, 1] : memref<32x3xf32>
        %150 = arith.addf %149, %103 : f32
        affine.store %150, %alloca_7[%arg5, 1] : memref<32x3xf32>
        %151 = affine.load %alloca_7[%arg5, 2] : memref<32x3xf32>
        %152 = arith.addf %151, %110 : f32
        affine.store %152, %alloca_7[%arg5, 2] : memref<32x3xf32>
      }
      %0 = affine.load %alloca_10[0] : memref<3xf32>
      %1 = arith.mulf %0, %cst_2 : f32
      %2 = affine.load %alloca_9[0] : memref<3xf32>
      %3 = arith.divf %1, %2 : f32
      %4 = affine.load %arg0[0] : memref<?xf32>
      %5 = arith.subf %4, %3 : f32
      affine.store %5, %arg0[0] : memref<?xf32>
      %6 = affine.load %alloca_10[1] : memref<3xf32>
      %7 = arith.mulf %6, %cst_2 : f32
      %8 = affine.load %alloca_9[1] : memref<3xf32>
      %9 = arith.divf %7, %8 : f32
      %10 = affine.load %arg0[1] : memref<?xf32>
      %11 = arith.subf %10, %9 : f32
      affine.store %11, %arg0[1] : memref<?xf32>
      %12 = affine.load %alloca_10[2] : memref<3xf32>
      %13 = arith.mulf %12, %cst_1 : f32
      %14 = affine.load %alloca_9[2] : memref<3xf32>
      %15 = arith.divf %13, %14 : f32
      %16 = affine.load %arg0[2] : memref<?xf32>
      %17 = arith.subf %16, %15 : f32
      affine.store %17, %arg0[2] : memref<?xf32>
      affine.for %arg5 = 0 to 32 {
        %18 = affine.load %alloca_8[%arg5, 0] : memref<32x3xf32>
        %19 = arith.mulf %18, %cst_0 : f32
        %20 = affine.load %alloca_7[%arg5, 0] : memref<32x3xf32>
        %21 = arith.divf %19, %20 : f32
        %22 = affine.load %arg1[%arg5, 0] : memref<?x3xf32>
        %23 = arith.subf %22, %21 : f32
        affine.store %23, %arg1[%arg5, 0] : memref<?x3xf32>
        %24 = affine.load %alloca_8[%arg5, 1] : memref<32x3xf32>
        %25 = arith.mulf %24, %cst_0 : f32
        %26 = affine.load %alloca_7[%arg5, 1] : memref<32x3xf32>
        %27 = arith.divf %25, %26 : f32
        %28 = affine.load %arg1[%arg5, 1] : memref<?x3xf32>
        %29 = arith.subf %28, %27 : f32
        affine.store %29, %arg1[%arg5, 1] : memref<?x3xf32>
        %30 = affine.load %alloca_8[%arg5, 2] : memref<32x3xf32>
        %31 = arith.mulf %30, %cst : f32
        %32 = affine.load %alloca_7[%arg5, 2] : memref<32x3xf32>
        %33 = arith.divf %31, %32 : f32
        %34 = affine.load %arg1[%arg5, 2] : memref<?x3xf32>
        %35 = arith.subf %34, %33 : f32
        affine.store %35, %arg1[%arg5, 2] : memref<?x3xf32>
      }
    }
    return
  }
  func.func private @_ZL9summarizePfPA3_f(%arg0: memref<?xf32>, %arg1: memref<?x3xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.270000e+02 : f32
    %cst_1 = arith.constant 1.030000e+02 : f32
    %cst_2 = arith.constant 7.900000e+01 : f32
    %cst_3 = arith.constant 3.070000e+02 : f32
    %cst_4 = arith.constant 2.570000e+02 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_5 = arith.constant 2.110000e+02 : f32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %0 = affine.load %arg0[0] : memref<?xf32>
    %1 = arith.mulf %0, %cst_5 : f32
    %2 = arith.fptosi %1 : f32 to i32
    %3 = affine.load %arg0[1] : memref<?xf32>
    %4 = arith.mulf %3, %cst_4 : f32
    %5 = arith.fptosi %4 : f32 to i32
    %6 = arith.addi %2, %5 : i32
    %7 = affine.load %arg0[2] : memref<?xf32>
    %8 = arith.mulf %7, %cst_3 : f32
    %9 = arith.fptosi %8 : f32 to i32
    %10 = arith.addi %6, %9 : i32
    affine.store %10, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 32 {
      %16 = affine.load %arg1[%arg2, 0] : memref<?x3xf32>
      %17 = arith.mulf %16, %cst_2 : f32
      %18 = arith.fptosi %17 : f32 to i32
      %19 = affine.load %alloca[] : memref<i32>
      %20 = arith.addi %19, %18 : i32
      affine.store %20, %alloca[] : memref<i32>
      %21 = affine.load %arg1[%arg2, 1] : memref<?x3xf32>
      %22 = arith.mulf %21, %cst_1 : f32
      %23 = arith.fptosi %22 : f32 to i32
      %24 = arith.addi %20, %23 : i32
      affine.store %24, %alloca[] : memref<i32>
      %25 = affine.load %arg1[%arg2, 2] : memref<?x3xf32>
      %26 = arith.mulf %25, %cst_0 : f32
      %27 = arith.fptosi %26 : f32 to i32
      %28 = arith.addi %24, %27 : i32
      affine.store %28, %alloca[] : memref<i32>
    }
    %11 = affine.load %alloca[] : memref<i32>
    %12 = arith.sitofp %11 : i32 to f32
    %13 = call @_ZL5abs_ff(%12) : (f32) -> f32
    %14 = arith.cmpf ogt, %13, %cst : f32
    %15 = arith.select %14, %11, %c1_i32 : i32
    return %15 : i32
  }
  func.func private @_ZL14stereo_projectfffPfS_(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = affine.load %arg3[0] : memref<?xf32>
    %1 = arith.subf %0, %arg0 : f32
    %2 = affine.load %arg3[1] : memref<?xf32>
    %3 = arith.subf %2, %arg1 : f32
    %4 = affine.load %arg3[2] : memref<?xf32>
    %5 = arith.subf %4, %arg2 : f32
    %6 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
    %7 = affine.load %6[0] : memref<1xf32>
    %8 = arith.addf %5, %7 : f32
    %9 = arith.divf %cst, %8 : f32
    %10 = memref.get_global @_ZL2FX : memref<1xf32>
    %11 = affine.load %10[0] : memref<1xf32>
    %12 = arith.mulf %11, %1 : f32
    %13 = arith.mulf %12, %9 : f32
    affine.store %13, %arg4[0] : memref<?xf32>
    %14 = memref.get_global @_ZL2FY : memref<1xf32>
    %15 = affine.load %14[0] : memref<1xf32>
    %16 = arith.mulf %15, %3 : f32
    %17 = arith.mulf %16, %9 : f32
    affine.store %17, %arg4[1] : memref<?xf32>
    %18 = affine.load %10[0] : memref<1xf32>
    %19 = memref.get_global @_ZL8BASELINE : memref<1xf32>
    %20 = affine.load %19[0] : memref<1xf32>
    %21 = arith.mulf %18, %20 : f32
    %22 = arith.mulf %21, %9 : f32
    affine.store %22, %arg4[2] : memref<?xf32>
    return
  }
  func.func private @_ZL5abs_ff(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.cmpf olt, %arg0, %cst : f32
    %1 = scf.if %0 -> (f32) {
      %2 = arith.negf %arg0 : f32
      scf.yield %2 : f32
    } else {
      scf.yield %arg0 : f32
    }
    return %1 : f32
  }
}

