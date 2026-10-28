module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c19_i32 = arith.constant 19 : i32
    %c9_i32 = arith.constant 9 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst = arith.constant 2.000000e-02 : f32
    %cst_0 = arith.constant 3.000000e-02 : f32
    %alloca = memref.alloca() : memref<16x16xi32>
    %alloca_1 = memref.alloca() : memref<16x16xf32>
    %alloca_2 = memref.alloca() : memref<16x16x3xf32>
    affine.for %arg0 = 0 to 16 {
      %1 = arith.index_cast %arg0 : index to i32
      affine.for %arg1 = 0 to 16 {
        %2 = arith.index_cast %arg1 : index to i32
        %3 = arith.sitofp %2 : i32 to f32
        %4 = arith.mulf %3, %cst_0 : f32
        affine.store %4, %alloca_2[%arg0, %arg1, 0] : memref<16x16x3xf32>
        %5 = arith.sitofp %1 : i32 to f32
        %6 = arith.mulf %5, %cst_0 : f32
        affine.store %6, %alloca_2[%arg0, %arg1, 1] : memref<16x16x3xf32>
        %7 = arith.muli %2, %c5_i32 : i32
        %8 = arith.muli %1, %c9_i32 : i32
        %9 = arith.addi %7, %8 : i32
        %10 = arith.xori %2, %1 : i32
        %11 = arith.addi %9, %10 : i32
        %12 = arith.remsi %11, %c19_i32 : i32
        %13 = arith.sitofp %12 : i32 to f32
        %14 = arith.mulf %13, %cst : f32
        affine.store %14, %alloca_2[%arg0, %arg1, 2] : memref<16x16x3xf32>
      }
    }
    %cast = memref.cast %alloca_2 : memref<16x16x3xf32> to memref<?x16x3xf32>
    %cast_3 = memref.cast %alloca_1 : memref<16x16xf32> to memref<?x16xf32>
    call @_ZL20compute_iss_responsePA16_A3_fPA16_f(%cast, %cast_3) : (memref<?x16x3xf32>, memref<?x16xf32>) -> ()
    %cast_4 = memref.cast %alloca : memref<16x16xi32> to memref<?x16xi32>
    call @_ZL15nonmax_suppressPA16_fPA16_i(%cast_3, %cast_4) : (memref<?x16xf32>, memref<?x16xi32>) -> ()
    %0 = call @_ZL9summarizePA16_fPA16_i(%cast_3, %cast_4) : (memref<?x16xf32>, memref<?x16xi32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL20compute_iss_responsePA16_A3_fPA16_f(%arg0: memref<?x16x3xf32>, %arg1: memref<?x16xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-2_i32 = arith.constant -2 : i32
    %c39_i32 = arith.constant 15 : i32
    %cst = arith.constant 5.000000e-01 : f32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca[] : memref<f32>
    %alloca_1 = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca_1[] : memref<f32>
    %alloca_2 = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca_2[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca_3[] : memref<f32>
    %alloca_4 = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<f32>
    affine.store %cst_0, %alloca_5[] : memref<f32>
    affine.for %arg2 = 0 to 16 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 16 {
        %1 = arith.index_cast %arg3 : index to i32
        affine.store %cst_0, %alloca_5[] : memref<f32>
        affine.store %cst_0, %alloca_4[] : memref<f32>
        affine.store %cst_0, %alloca_3[] : memref<f32>
        affine.store %cst_0, %alloca_2[] : memref<f32>
        affine.store %cst_0, %alloca_1[] : memref<f32>
        affine.store %cst_0, %alloca[] : memref<f32>
        %2 = affine.load %arg0[%arg2, %arg3, 0] : memref<?x16x3xf32>
        %3 = affine.load %arg0[%arg2, %arg3, 1] : memref<?x16x3xf32>
        %4 = affine.load %arg0[%arg2, %arg3, 2] : memref<?x16x3xf32>
        affine.for %arg4 = 0 to 5 {
          %20 = arith.index_cast %arg4 : index to i32
          %21 = arith.addi %20, %c-2_i32 : i32
          affine.for %arg5 = 0 to 5 {
            %22 = arith.index_cast %arg5 : index to i32
            %23 = arith.addi %22, %c-2_i32 : i32
            %24 = arith.addi %0, %21 : i32
            %25 = func.call @_ZL7clamp_iiii(%24, %c0_i32, %c39_i32) : (i32, i32, i32) -> i32
            %26 = arith.addi %1, %23 : i32
            %27 = func.call @_ZL7clamp_iiii(%26, %c0_i32, %c39_i32) : (i32, i32, i32) -> i32
            %28 = arith.index_cast %25 : i32 to index
            %29 = arith.index_cast %27 : i32 to index
            %30 = memref.load %arg0[%28, %29, %c0] : memref<?x16x3xf32>
            %31 = arith.subf %30, %2 : f32
            %32 = memref.load %arg0[%28, %29, %c1] : memref<?x16x3xf32>
            %33 = arith.subf %32, %3 : f32
            %34 = memref.load %arg0[%28, %29, %c2] : memref<?x16x3xf32>
            %35 = arith.subf %34, %4 : f32
            %36 = arith.mulf %31, %31 : f32
            %37 = affine.load %alloca_5[] : memref<f32>
            %38 = arith.addf %37, %36 : f32
            affine.store %38, %alloca_5[] : memref<f32>
            %39 = arith.mulf %33, %33 : f32
            %40 = affine.load %alloca_4[] : memref<f32>
            %41 = arith.addf %40, %39 : f32
            affine.store %41, %alloca_4[] : memref<f32>
            %42 = arith.mulf %35, %35 : f32
            %43 = affine.load %alloca_3[] : memref<f32>
            %44 = arith.addf %43, %42 : f32
            affine.store %44, %alloca_3[] : memref<f32>
            %45 = arith.mulf %31, %33 : f32
            %46 = affine.load %alloca_2[] : memref<f32>
            %47 = arith.addf %46, %45 : f32
            affine.store %47, %alloca_2[] : memref<f32>
            %48 = arith.mulf %31, %35 : f32
            %49 = affine.load %alloca_1[] : memref<f32>
            %50 = arith.addf %49, %48 : f32
            affine.store %50, %alloca_1[] : memref<f32>
            %51 = arith.mulf %33, %35 : f32
            %52 = affine.load %alloca[] : memref<f32>
            %53 = arith.addf %52, %51 : f32
            affine.store %53, %alloca[] : memref<f32>
          }
        }
        %5 = affine.load %alloca_5[] : memref<f32>
        %6 = affine.load %alloca_4[] : memref<f32>
        %7 = arith.addf %5, %6 : f32
        %8 = affine.load %alloca_3[] : memref<f32>
        %9 = arith.addf %7, %8 : f32
        %10 = affine.load %alloca_2[] : memref<f32>
        %11 = func.call @_ZL5abs_ff(%10) : (f32) -> f32
        %12 = affine.load %alloca_1[] : memref<f32>
        %13 = func.call @_ZL5abs_ff(%12) : (f32) -> f32
        %14 = arith.addf %11, %13 : f32
        %15 = affine.load %alloca[] : memref<f32>
        %16 = func.call @_ZL5abs_ff(%15) : (f32) -> f32
        %17 = arith.addf %14, %16 : f32
        %18 = arith.mulf %17, %cst : f32
        %19 = arith.subf %9, %18 : f32
        affine.store %19, %arg1[%arg2, %arg3] : memref<?x16xf32>
      }
    }
    return
  }
  func.func private @_ZL15nonmax_suppressPA16_fPA16_i(%arg0: memref<?x16xf32>, %arg1: memref<?x16xi32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-1_i32 = arith.constant -1 : i32
    %true = arith.constant true
    %c39_i32 = arith.constant 15 : i32
    %cst = arith.constant 2.000000e-02 : f32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 16 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 16 {
        %1 = arith.index_cast %arg3 : index to i32
        %2 = affine.load %arg0[%arg2, %arg3] : memref<?x16xf32>
        %3 = arith.cmpf ogt, %2, %cst : f32
        %4 = arith.extui %3 : i1 to i32
        affine.store %4, %alloca[] : memref<i32>
        affine.for %arg4 = 0 to 3 {
          %6 = arith.index_cast %arg4 : index to i32
          %7 = arith.addi %6, %c-1_i32 : i32
          affine.for %arg5 = 0 to 3 {
            %8 = arith.index_cast %arg5 : index to i32
            %9 = arith.addi %8, %c-1_i32 : i32
            %10 = arith.addi %0, %7 : i32
            %11 = func.call @_ZL7clamp_iiii(%10, %c0_i32, %c39_i32) : (i32, i32, i32) -> i32
            %12 = arith.addi %1, %9 : i32
            %13 = func.call @_ZL7clamp_iiii(%12, %c0_i32, %c39_i32) : (i32, i32, i32) -> i32
            %14 = arith.cmpi ne, %7, %c0_i32 : i32
            %15 = scf.if %14 -> (i1) {
              scf.yield %true : i1
            } else {
              %16 = arith.cmpi ne, %9, %c0_i32 : i32
              scf.yield %16 : i1
            }
            scf.if %15 {
              %16 = arith.index_cast %11 : i32 to index
              %17 = arith.index_cast %13 : i32 to index
              %18 = memref.load %arg0[%16, %17] : memref<?x16xf32>
              %19 = arith.cmpf ogt, %18, %2 : f32
              scf.if %19 {
                affine.store %c0_i32, %alloca[] : memref<i32>
              }
            }
          }
        }
        %5 = affine.load %alloca[] : memref<i32>
        affine.store %5, %arg1[%arg2, %arg3] : memref<?x16xi32>
      }
    }
    return
  }
  func.func private @_ZL9summarizePA16_fPA16_i(%arg0: memref<?x16xf32>, %arg1: memref<?x16xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 3.200000e+01 : f32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 16 {
      %1 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 16 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = affine.load %arg1[%arg2, %arg3] : memref<?x16xi32>
        %4 = arith.cmpi ne, %3, %c0_i32 : i32
        %5 = scf.if %4 -> (i32) {
          %8 = affine.load %arg0[%arg2, %arg3] : memref<?x16xf32>
          %9 = arith.mulf %8, %cst : f32
          %10 = arith.fptosi %9 : f32 to i32
          %11 = arith.addi %10, %2 : i32
          %12 = arith.addi %11, %1 : i32
          scf.yield %12 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %6 = affine.load %alloca[] : memref<i32>
        %7 = arith.addi %6, %5 : i32
        affine.store %7, %alloca[] : memref<i32>
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

