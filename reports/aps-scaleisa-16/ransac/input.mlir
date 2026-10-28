module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<4xi32>
    %alloca_0 = memref.alloca() : memref<4xf32>
    %alloca_1 = memref.alloca() : memref<32x3xf32>
    %cast = memref.cast %alloca_1 : memref<32x3xf32> to memref<?x3xf32>
    call @_ZL10init_cloudPA3_f(%cast) : (memref<?x3xf32>) -> ()
    %cast_2 = memref.cast %alloca_0 : memref<4xf32> to memref<?xf32>
    %cast_3 = memref.cast %alloca : memref<4xi32> to memref<?xi32>
    call @_ZL13ransac_spherePA3_fPfPi(%cast, %cast_2, %cast_3) : (memref<?x3xf32>, memref<?xf32>, memref<?xi32>) -> ()
    %0 = call @_ZL9summarizePfPi(%cast_2, %cast_3) : (memref<?xf32>, memref<?xi32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL10init_cloudPA3_f(%arg0: memref<?x3xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 8.900000e+01 : f32
    %c89_i32 = arith.constant 89 : i32
    %c7_i32 = arith.constant 7 : i32
    %c19_i32 = arith.constant 19 : i32
    %cst_0 = arith.constant 9.700000e+01 : f32
    %c97_i32 = arith.constant 97 : i32
    %c5_i32 = arith.constant 5 : i32
    %c17_i32 = arith.constant 17 : i32
    %cst_1 = arith.constant 1.010000e+02 : f32
    %c101_i32 = arith.constant 101 : i32
    %c3_i32 = arith.constant 3 : i32
    %c13_i32 = arith.constant 13 : i32
    %cst_2 = arith.constant 3.100000e+01 : f32
    %c31_i32 = arith.constant 31 : i32
    %cst_3 = arith.constant 3.000000e-01 : f32
    %cst_4 = arith.constant 1.200000e+01 : f32
    %cst_5 = arith.constant 4.000000e-01 : f32
    %cst_6 = arith.constant 6.400000e+01 : f32
    %cst_7 = arith.constant 2.000000e-01 : f32
    %cst_8 = arith.constant 6.000000e-01 : f32
    %c64_i32 = arith.constant 64 : i32
    %c768_i32 = arith.constant 768 : i32
    affine.for %arg1 = 0 to 32 {
      %0 = arith.index_cast %arg1 : index to i32
      %1 = arith.cmpi slt, %0, %c768_i32 : i32
      scf.if %1 {
        %2 = arith.remsi %0, %c64_i32 : i32
        %3 = arith.divsi %0, %c64_i32 : i32
        %4 = arith.sitofp %2 : i32 to f32
        %5 = arith.mulf %4, %cst_7 : f32
        %6 = arith.divf %5, %cst_6 : f32
        %7 = arith.addf %6, %cst_8 : f32
        affine.store %7, %arg0[%arg1, 0] : memref<?x3xf32>
        %8 = arith.sitofp %3 : i32 to f32
        %9 = arith.mulf %8, %cst_7 : f32
        %10 = arith.divf %9, %cst_4 : f32
        %11 = arith.addf %10, %cst_5 : f32
        affine.store %11, %arg0[%arg1, 1] : memref<?x3xf32>
        %12 = arith.addi %2, %3 : i32
        %13 = arith.remsi %12, %c31_i32 : i32
        %14 = arith.sitofp %13 : i32 to f32
        %15 = arith.mulf %14, %cst_7 : f32
        %16 = arith.divf %15, %cst_2 : f32
        %17 = arith.addf %16, %cst_3 : f32
        affine.store %17, %arg0[%arg1, 2] : memref<?x3xf32>
      } else {
        %2 = arith.muli %0, %c13_i32 : i32
        %3 = arith.addi %2, %c3_i32 : i32
        %4 = arith.remsi %3, %c101_i32 : i32
        %5 = arith.sitofp %4 : i32 to f32
        %6 = arith.divf %5, %cst_1 : f32
        affine.store %6, %arg0[%arg1, 0] : memref<?x3xf32>
        %7 = arith.muli %0, %c17_i32 : i32
        %8 = arith.addi %7, %c5_i32 : i32
        %9 = arith.remsi %8, %c97_i32 : i32
        %10 = arith.sitofp %9 : i32 to f32
        %11 = arith.divf %10, %cst_0 : f32
        affine.store %11, %arg0[%arg1, 1] : memref<?x3xf32>
        %12 = arith.muli %0, %c19_i32 : i32
        %13 = arith.addi %12, %c7_i32 : i32
        %14 = arith.remsi %13, %c89_i32 : i32
        %15 = arith.sitofp %14 : i32 to f32
        %16 = arith.divf %15, %cst : f32
        affine.store %16, %arg0[%arg1, 2] : memref<?x3xf32>
      }
    }
    return
  }
  func.func private @_ZL13ransac_spherePA3_fPfPi(%arg0: memref<?x3xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant 8.000000e-02 : f32
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant 2.500000e-01 : f32
    %c31_i32 = arith.constant 31 : i32
    %c29_i32 = arith.constant 29 : i32
    %c23_i32 = arith.constant 23 : i32
    %c17_i32 = arith.constant 17 : i32
    %c19_i32 = arith.constant 19 : i32
    %c13_i32 = arith.constant 13 : i32
    %c1024_i32 = arith.constant 32 : i32
    %c11_i32 = arith.constant 11 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst_1 = arith.constant 1.000000e+30 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %alloca = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca[] : memref<f32>
    %alloca_3 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_3[] : memref<i32>
    %alloca_4 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_4[] : memref<f32>
    %alloca_5 = memref.alloca() : memref<1xi32>
    affine.store %c0_i32, %alloca_5[0] : memref<1xi32>
    %alloca_6 = memref.alloca() : memref<1xi32>
    affine.store %c0_i32, %alloca_6[0] : memref<1xi32>
    %alloca_7 = memref.alloca() : memref<1xi32>
    %cast = memref.cast %alloca_7 : memref<1xi32> to memref<?xi32>
    affine.store %c0_i32, %alloca_7[0] : memref<1xi32>
    %alloca_8 = memref.alloca() : memref<1xi32>
    %cast_9 = memref.cast %alloca_8 : memref<1xi32> to memref<?xi32>
    affine.store %c0_i32, %alloca_8[0] : memref<1xi32>
    %alloca_10 = memref.alloca() : memref<f32>
    affine.store %cst_2, %alloca_10[] : memref<f32>
    %alloca_11 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_11[] : memref<i32>
    affine.store %c-1_i32, %alloca_11[] : memref<i32>
    affine.store %cst_1, %alloca_10[] : memref<f32>
    affine.for %arg3 = 0 to 4 {
      %0 = arith.index_cast %arg3 : index to i32
      %1 = arith.muli %0, %c7_i32 : i32
      %2 = arith.addi %1, %c11_i32 : i32
      %3 = arith.remsi %2, %c1024_i32 : i32
      affine.store %3, %alloca_8[0] : memref<1xi32>
      %4 = arith.muli %0, %c13_i32 : i32
      %5 = arith.addi %4, %c19_i32 : i32
      %6 = arith.remsi %5, %c1024_i32 : i32
      affine.store %6, %alloca_7[0] : memref<1xi32>
      %7 = arith.muli %0, %c17_i32 : i32
      %8 = arith.addi %7, %c23_i32 : i32
      %9 = arith.remsi %8, %c1024_i32 : i32
      affine.store %9, %alloca_6[0] : memref<1xi32>
      %10 = arith.muli %0, %c29_i32 : i32
      %11 = arith.addi %10, %c31_i32 : i32
      %12 = arith.remsi %11, %c1024_i32 : i32
      affine.store %12, %alloca_5[0] : memref<1xi32>
      %13 = affine.load %alloca_8[0] : memref<1xi32>
      %14 = arith.index_cast %13 : i32 to index
      %15 = memref.load %arg0[%14, %c0] : memref<?x3xf32>
      %16 = affine.load %alloca_7[0] : memref<1xi32>
      %17 = arith.index_cast %16 : i32 to index
      %18 = memref.load %arg0[%17, %c0] : memref<?x3xf32>
      %19 = arith.addf %15, %18 : f32
      %20 = affine.load %alloca_6[0] : memref<1xi32>
      %21 = arith.index_cast %20 : i32 to index
      %22 = memref.load %arg0[%21, %c0] : memref<?x3xf32>
      %23 = arith.addf %19, %22 : f32
      %24 = affine.load %alloca_5[0] : memref<1xi32>
      %25 = arith.index_cast %24 : i32 to index
      %26 = memref.load %arg0[%25, %c0] : memref<?x3xf32>
      %27 = arith.addf %23, %26 : f32
      %28 = arith.mulf %27, %cst_0 : f32
      %29 = memref.load %arg0[%14, %c1] : memref<?x3xf32>
      %30 = memref.load %arg0[%17, %c1] : memref<?x3xf32>
      %31 = arith.addf %29, %30 : f32
      %32 = memref.load %arg0[%21, %c1] : memref<?x3xf32>
      %33 = arith.addf %31, %32 : f32
      %34 = memref.load %arg0[%25, %c1] : memref<?x3xf32>
      %35 = arith.addf %33, %34 : f32
      %36 = arith.mulf %35, %cst_0 : f32
      %37 = memref.load %arg0[%14, %c2] : memref<?x3xf32>
      %38 = memref.load %arg0[%17, %c2] : memref<?x3xf32>
      %39 = arith.addf %37, %38 : f32
      %40 = memref.load %arg0[%21, %c2] : memref<?x3xf32>
      %41 = arith.addf %39, %40 : f32
      %42 = memref.load %arg0[%25, %c2] : memref<?x3xf32>
      %43 = arith.addf %41, %42 : f32
      %44 = arith.mulf %43, %cst_0 : f32
      affine.store %cst_2, %alloca_4[] : memref<f32>
      affine.store %c0_i32, %alloca_3[] : memref<i32>
      affine.store %cst_2, %alloca[] : memref<f32>
      affine.for %arg4 = 0 to 4 {
        %51 = arith.index_cast %arg4 : index to i32
        %52 = arith.cmpi eq, %51, %c0_i32 : i32
        %53 = scf.if %52 -> (memref<?xi32>) {
          scf.yield %cast_9 : memref<?xi32>
        } else {
          %69 = arith.cmpi eq, %51, %c1_i32 : i32
          %70 = scf.if %69 -> (memref<?xi32>) {
            scf.yield %cast : memref<?xi32>
          } else {
            %71 = arith.cmpi eq, %51, %c2_i32 : i32
            %72 = arith.select %71, %alloca_6, %alloca_5 : memref<1xi32>
            %cast_12 = memref.cast %72 : memref<1xi32> to memref<?xi32>
            scf.yield %cast_12 : memref<?xi32>
          }
          scf.yield %70 : memref<?xi32>
        }
        %54 = affine.load %53[0] : memref<?xi32>
        %55 = arith.index_cast %54 : i32 to index
        %56 = memref.load %arg0[%55, %c0] : memref<?x3xf32>
        %57 = arith.subf %56, %28 : f32
        %58 = memref.load %arg0[%55, %c1] : memref<?x3xf32>
        %59 = arith.subf %58, %36 : f32
        %60 = memref.load %arg0[%55, %c2] : memref<?x3xf32>
        %61 = arith.subf %60, %44 : f32
        %62 = func.call @_ZL5abs_ff(%57) : (f32) -> f32
        %63 = func.call @_ZL5abs_ff(%59) : (f32) -> f32
        %64 = arith.addf %62, %63 : f32
        %65 = func.call @_ZL5abs_ff(%61) : (f32) -> f32
        %66 = arith.addf %64, %65 : f32
        %67 = affine.load %alloca_4[] : memref<f32>
        %68 = arith.addf %67, %66 : f32
        affine.store %68, %alloca_4[] : memref<f32>
      }
      %45 = affine.load %alloca_4[] : memref<f32>
      %46 = arith.mulf %45, %cst_0 : f32
      affine.store %46, %alloca_4[] : memref<f32>
      affine.for %arg4 = 0 to 32 {
        %51 = affine.load %arg0[%arg4, 0] : memref<?x3xf32>
        %52 = arith.subf %51, %28 : f32
        %53 = affine.load %arg0[%arg4, 1] : memref<?x3xf32>
        %54 = arith.subf %53, %36 : f32
        %55 = affine.load %arg0[%arg4, 2] : memref<?x3xf32>
        %56 = arith.subf %55, %44 : f32
        %57 = func.call @_ZL5abs_ff(%52) : (f32) -> f32
        %58 = func.call @_ZL5abs_ff(%54) : (f32) -> f32
        %59 = arith.addf %57, %58 : f32
        %60 = func.call @_ZL5abs_ff(%56) : (f32) -> f32
        %61 = arith.addf %59, %60 : f32
        %62 = arith.subf %61, %46 : f32
        %63 = func.call @_ZL5abs_ff(%62) : (f32) -> f32
        %64 = arith.cmpf olt, %63, %cst : f32
        scf.if %64 {
          %67 = affine.load %alloca_3[] : memref<i32>
          %68 = arith.addi %67, %c1_i32 : i32
          affine.store %68, %alloca_3[] : memref<i32>
        }
        %65 = affine.load %alloca[] : memref<f32>
        %66 = arith.addf %65, %63 : f32
        affine.store %66, %alloca[] : memref<f32>
      }
      %47 = affine.load %alloca_3[] : memref<i32>
      affine.store %47, %arg2[%arg3] : memref<?xi32>
      %48 = affine.load %alloca_11[] : memref<i32>
      %49 = arith.cmpi sgt, %47, %48 : i32
      %50 = scf.if %49 -> (i1) {
        scf.yield %true : i1
      } else {
        %51 = arith.cmpi eq, %47, %48 : i32
        %52 = scf.if %51 -> (i1) {
          %53 = affine.load %alloca[] : memref<f32>
          %54 = affine.load %alloca_10[] : memref<f32>
          %55 = arith.cmpf olt, %53, %54 : f32
          scf.yield %55 : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %52 : i1
      }
      scf.if %50 {
        affine.store %47, %alloca_11[] : memref<i32>
        %51 = affine.load %alloca[] : memref<f32>
        affine.store %51, %alloca_10[] : memref<f32>
        affine.store %28, %arg1[0] : memref<?xf32>
        affine.store %36, %arg1[1] : memref<?xf32>
        affine.store %44, %arg1[2] : memref<?xf32>
        affine.store %46, %arg1[3] : memref<?xf32>
      }
    }
    return
  }
  func.func private @_ZL9summarizePfPi(%arg0: memref<?xf32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 1.300000e+03 : f32
    %cst_0 = arith.constant 1.200000e+03 : f32
    %cst_1 = arith.constant 1.100000e+03 : f32
    %cst_2 = arith.constant 1.000000e+03 : f32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    %0 = affine.load %arg0[0] : memref<?xf32>
    %1 = arith.mulf %0, %cst_2 : f32
    %2 = arith.fptosi %1 : f32 to i32
    %3 = affine.load %arg0[1] : memref<?xf32>
    %4 = arith.mulf %3, %cst_1 : f32
    %5 = arith.fptosi %4 : f32 to i32
    %6 = arith.addi %2, %5 : i32
    %7 = affine.load %arg0[2] : memref<?xf32>
    %8 = arith.mulf %7, %cst_0 : f32
    %9 = arith.fptosi %8 : f32 to i32
    %10 = arith.addi %6, %9 : i32
    %11 = affine.load %arg0[3] : memref<?xf32>
    %12 = arith.mulf %11, %cst : f32
    %13 = arith.fptosi %12 : f32 to i32
    %14 = arith.addi %10, %13 : i32
    affine.store %14, %alloca[] : memref<i32>
    affine.for %arg2 = 0 to 4 {
      %16 = affine.load %arg1[%arg2] : memref<?xi32>
      %17 = affine.load %alloca[] : memref<i32>
      %18 = arith.addi %17, %16 : i32
      affine.store %18, %alloca[] : memref<i32>
    }
    %15 = affine.load %alloca[] : memref<i32>
    return %15 : i32
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

