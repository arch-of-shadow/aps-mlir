module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c9_i32 = arith.constant 9 : i32
    %cst = arith.constant 0.00999999977 : f32
    %cst_0 = arith.constant 7.000000e-02 : f32
    %cst_1 = arith.constant 1.500000e-01 : f32
    %cst_2 = arith.constant 4.000000e-02 : f32
    %cst_3 = arith.constant 5.000000e-02 : f32
    %alloca = memref.alloca() : memref<12x12xf32>
    %alloca_4 = memref.alloca() : memref<12x12x3xf32>
    %alloca_5 = memref.alloca() : memref<12x12x3xf32>
    %alloca_6 = memref.alloca() : memref<12x12x3xf32>
    affine.for %arg0 = 0 to 12 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 12 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.sitofp %2 : i32 to f32
        %4 = arith.mulf %3, %cst_3 : f32
        %5 = arith.sitofp %1 : i32 to f32
        %6 = arith.mulf %5, %cst_2 : f32
        affine.store %4, %alloca_6[%arg0, %arg1, 0] : memref<12x12x3xf32>
        affine.store %6, %alloca_6[%arg0, %arg1, 1] : memref<12x12x3xf32>
        %7 = arith.mulf %4, %cst_1 : f32
        %8 = arith.mulf %6, %cst_0 : f32
        %9 = arith.addf %7, %8 : f32
        %10 = arith.muli %2, %1 : i32
        %11 = arith.remsi %10, %c9_i32 : i32
        %12 = arith.sitofp %11 : i32 to f32
        %13 = arith.mulf %12, %cst : f32
        %14 = arith.addf %9, %13 : f32
        affine.store %14, %alloca_6[%arg0, %arg1, 2] : memref<12x12x3xf32>
      }
    }
    %cast = memref.cast %alloca_6 : memref<12x12x3xf32> to memref<?x12x3xf32>
    %cast_7 = memref.cast %alloca_5 : memref<12x12x3xf32> to memref<?x12x3xf32>
    %cast_8 = memref.cast %alloca : memref<12x12xf32> to memref<?x12xf32>
    call @_ZL15compute_normalsPA12_A3_fS1_PA12_f(%cast, %cast_7, %cast_8) : (memref<?x12x3xf32>, memref<?x12x3xf32>, memref<?x12xf32>) -> ()
    %cast_9 = memref.cast %alloca_4 : memref<12x12x3xf32> to memref<?x12x3xf32>
    call @_ZL14smooth_normalsPA12_A3_fS1_(%cast_7, %cast_9) : (memref<?x12x3xf32>, memref<?x12x3xf32>) -> ()
    %0 = call @_ZL20accumulate_signaturePA12_A3_fPA12_f(%cast_9, %cast_8) : (memref<?x12x3xf32>, memref<?x12xf32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL15compute_normalsPA12_A3_fS1_PA12_f(%arg0: memref<?x12x3xf32>, %arg1: memref<?x12x3xf32>, %arg2: memref<?x12xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-1_i32 = arith.constant -1 : i32
    %c47_i32 = arith.constant 11 : i32
    %c39_i32 = arith.constant 11 : i32
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-03 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    affine.for %arg3 = 0 to 12 {
      %0 = arith.index_cast %arg3 : index to i32
      affine.for %arg4 = 0 to 12 {
        %1 = arith.index_cast %arg4 : index to i32
        %2 = arith.addi %1, %c-1_i32 : i32
        %3 = func.call @_ZL7clamp_iiii(%2, %c0_i32, %c47_i32) : (i32, i32, i32) -> i32
        %4 = arith.addi %1, %c1_i32 : i32
        %5 = func.call @_ZL7clamp_iiii(%4, %c0_i32, %c47_i32) : (i32, i32, i32) -> i32
        %6 = arith.addi %0, %c-1_i32 : i32
        %7 = func.call @_ZL7clamp_iiii(%6, %c0_i32, %c39_i32) : (i32, i32, i32) -> i32
        %8 = arith.addi %0, %c1_i32 : i32
        %9 = func.call @_ZL7clamp_iiii(%8, %c0_i32, %c39_i32) : (i32, i32, i32) -> i32
        %10 = arith.index_cast %5 : i32 to index
        %11 = memref.load %arg0[%arg3, %10, %c0] : memref<?x12x3xf32>
        %12 = arith.index_cast %3 : i32 to index
        %13 = memref.load %arg0[%arg3, %12, %c0] : memref<?x12x3xf32>
        %14 = arith.subf %11, %13 : f32
        %15 = memref.load %arg0[%arg3, %10, %c1] : memref<?x12x3xf32>
        %16 = memref.load %arg0[%arg3, %12, %c1] : memref<?x12x3xf32>
        %17 = arith.subf %15, %16 : f32
        %18 = memref.load %arg0[%arg3, %10, %c2] : memref<?x12x3xf32>
        %19 = memref.load %arg0[%arg3, %12, %c2] : memref<?x12x3xf32>
        %20 = arith.subf %18, %19 : f32
        %21 = arith.index_cast %9 : i32 to index
        %22 = memref.load %arg0[%21, %arg4, %c0] : memref<?x12x3xf32>
        %23 = arith.index_cast %7 : i32 to index
        %24 = memref.load %arg0[%23, %arg4, %c0] : memref<?x12x3xf32>
        %25 = arith.subf %22, %24 : f32
        %26 = memref.load %arg0[%21, %arg4, %c1] : memref<?x12x3xf32>
        %27 = memref.load %arg0[%23, %arg4, %c1] : memref<?x12x3xf32>
        %28 = arith.subf %26, %27 : f32
        %29 = memref.load %arg0[%21, %arg4, %c2] : memref<?x12x3xf32>
        %30 = memref.load %arg0[%23, %arg4, %c2] : memref<?x12x3xf32>
        %31 = arith.subf %29, %30 : f32
        %32 = arith.mulf %17, %31 : f32
        %33 = arith.mulf %20, %28 : f32
        %34 = arith.subf %32, %33 : f32
        %35 = arith.mulf %20, %25 : f32
        %36 = arith.mulf %14, %31 : f32
        %37 = arith.subf %35, %36 : f32
        %38 = arith.mulf %14, %28 : f32
        %39 = arith.mulf %17, %25 : f32
        %40 = arith.subf %38, %39 : f32
        %41 = func.call @_ZL5abs_ff(%34) : (f32) -> f32
        %42 = func.call @_ZL5abs_ff(%37) : (f32) -> f32
        %43 = arith.addf %41, %42 : f32
        %44 = func.call @_ZL5abs_ff(%40) : (f32) -> f32
        %45 = arith.addf %43, %44 : f32
        %46 = arith.addf %45, %cst_0 : f32
        %47 = arith.divf %cst_1, %46 : f32
        %48 = arith.mulf %34, %47 : f32
        affine.store %48, %arg1[%arg3, %arg4, 0] : memref<?x12x3xf32>
        %49 = arith.mulf %37, %47 : f32
        affine.store %49, %arg1[%arg3, %arg4, 1] : memref<?x12x3xf32>
        %50 = arith.mulf %40, %47 : f32
        affine.store %50, %arg1[%arg3, %arg4, 2] : memref<?x12x3xf32>
        %51 = memref.load %arg0[%arg3, %10, %c2] : memref<?x12x3xf32>
        %52 = affine.load %arg0[%arg3, %arg4, 2] : memref<?x12x3xf32>
        %53 = arith.mulf %52, %cst : f32
        %54 = arith.subf %51, %53 : f32
        %55 = memref.load %arg0[%arg3, %12, %c2] : memref<?x12x3xf32>
        %56 = arith.addf %54, %55 : f32
        %57 = func.call @_ZL5abs_ff(%56) : (f32) -> f32
        %58 = memref.load %arg0[%21, %arg4, %c2] : memref<?x12x3xf32>
        %59 = affine.load %arg0[%arg3, %arg4, 2] : memref<?x12x3xf32>
        %60 = arith.mulf %59, %cst : f32
        %61 = arith.subf %58, %60 : f32
        %62 = memref.load %arg0[%23, %arg4, %c2] : memref<?x12x3xf32>
        %63 = arith.addf %61, %62 : f32
        %64 = func.call @_ZL5abs_ff(%63) : (f32) -> f32
        %65 = arith.addf %57, %64 : f32
        affine.store %65, %arg2[%arg3, %arg4] : memref<?x12xf32>
      }
    }
    return
  }
  func.func private @_ZL14smooth_normalsPA12_A3_fS1_(%arg0: memref<?x12x3xf32>, %arg1: memref<?x12x3xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-1_i32 = arith.constant -1 : i32
    %c39_i32 = arith.constant 11 : i32
    %c47_i32 = arith.constant 11 : i32
    %cst = arith.constant 9.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca_2[] : memref<f32>
    affine.for %arg2 = 0 to 12 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 12 {
        %1 = arith.index_cast %arg3 : index to i32
        affine.store %cst_0, %alloca_2[] : memref<f32>
        affine.store %cst_0, %alloca_1[] : memref<f32>
        affine.store %cst_0, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3 {
          %8 = arith.index_cast %arg4 : index to i32
          %9 = arith.addi %8, %c-1_i32 : i32
          affine.for %arg5 = 0 to 3 {
            %10 = arith.index_cast %arg5 : index to i32
            %11 = arith.addi %10, %c-1_i32 : i32
            %12 = arith.addi %0, %9 : i32
            %13 = func.call @_ZL7clamp_iiii(%12, %c0_i32, %c39_i32) : (i32, i32, i32) -> i32
            %14 = arith.addi %1, %11 : i32
            %15 = func.call @_ZL7clamp_iiii(%14, %c0_i32, %c47_i32) : (i32, i32, i32) -> i32
            %16 = arith.index_cast %13 : i32 to index
            %17 = arith.index_cast %15 : i32 to index
            %18 = memref.load %arg0[%16, %17, %c0] : memref<?x12x3xf32>
            %19 = affine.load %alloca_2[] : memref<f32>
            %20 = arith.addf %19, %18 : f32
            affine.store %20, %alloca_2[] : memref<f32>
            %21 = memref.load %arg0[%16, %17, %c1] : memref<?x12x3xf32>
            %22 = affine.load %alloca_1[] : memref<f32>
            %23 = arith.addf %22, %21 : f32
            affine.store %23, %alloca_1[] : memref<f32>
            %24 = memref.load %arg0[%16, %17, %c2] : memref<?x12x3xf32>
            %25 = affine.load %alloca[] : memref<f32>
            %26 = arith.addf %25, %24 : f32
            affine.store %26, %alloca[] : memref<f32>
          }
        }
        %2 = affine.load %alloca_2[] : memref<f32>
        %3 = arith.divf %2, %cst : f32
        affine.store %3, %arg1[%arg2, %arg3, 0] : memref<?x12x3xf32>
        %4 = affine.load %alloca_1[] : memref<f32>
        %5 = arith.divf %4, %cst : f32
        affine.store %5, %arg1[%arg2, %arg3, 1] : memref<?x12x3xf32>
        %6 = affine.load %alloca[] : memref<f32>
        %7 = arith.divf %6, %cst : f32
        affine.store %7, %arg1[%arg2, %arg3, 2] : memref<?x12x3xf32>
      }
    }
    return
  }
  func.func private @_ZL20accumulate_signaturePA12_A3_fPA12_f(%arg0: memref<?x12x3xf32>, %arg1: memref<?x12xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 6.400000e+01 : f32
    %cst_0 = arith.constant 1.100000e+01 : f32
    %cst_1 = arith.constant 3.100000e+01 : f32
    %cst_2 = arith.constant 2.300000e+01 : f32
    %cst_3 = arith.constant 1.700000e+01 : f32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 10 {
      affine.for %arg3 = 0 to 10 {
        %1 = affine.load %arg0[%arg2 + 1, %arg3 + 1, 0] : memref<?x12x3xf32>
        %2 = arith.mulf %1, %cst_3 : f32
        %3 = affine.load %arg0[%arg2 + 1, %arg3 + 1, 1] : memref<?x12x3xf32>
        %4 = arith.mulf %3, %cst_2 : f32
        %5 = arith.addf %2, %4 : f32
        %6 = affine.load %arg0[%arg2 + 1, %arg3 + 1, 2] : memref<?x12x3xf32>
        %7 = arith.mulf %6, %cst_1 : f32
        %8 = arith.addf %5, %7 : f32
        %9 = affine.load %arg1[%arg2 + 1, %arg3 + 1] : memref<?x12xf32>
        %10 = arith.mulf %9, %cst_0 : f32
        %11 = arith.addf %8, %10 : f32
        %12 = arith.mulf %11, %cst : f32
        %13 = arith.fptosi %12 : f32 to i32
        %14 = affine.load %alloca[] : memref<i32>
        %15 = arith.addi %14, %13 : i32
        affine.store %15, %alloca[] : memref<i32>
      }
    }
    %0 = affine.load %alloca[] : memref<i32>
    return %0 : i32
  }
  func.func private @_ZL7clamp_iiii(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi slt, %arg0, %arg1 : i32
    %1 = arith.cmpi sge, %arg0, %arg1 : i32
    %2 = arith.select %0, %arg1, %c0_i32 : i32
    %3 = arith.cmpi sgt, %arg0, %arg2 : i32
    %4 = arith.andi %1, %3 : i1
    %5 = arith.xori %4, %true : i1
    %6 = arith.andi %5, %1 : i1
    %7 = arith.select %4, %arg2, %2 : i32
    %8 = arith.select %6, %arg0, %7 : i32
    return %8 : i32
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

