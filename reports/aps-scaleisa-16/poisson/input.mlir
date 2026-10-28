#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 8.300000e+01 : f32
    %c83_i32 = arith.constant 83 : i32
    %c13_i32 = arith.constant 13 : i32
    %cst_1 = arith.constant 8.900000e+01 : f32
    %c89_i32 = arith.constant 89 : i32
    %c5_i32 = arith.constant 5 : i32
    %c11_i32 = arith.constant 11 : i32
    %cst_2 = arith.constant 9.700000e+01 : f32
    %c97_i32 = arith.constant 97 : i32
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst_3 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<216xf32>
    %alloca_4 = memref.alloca() : memref<216xf32>
    %alloca_5 = memref.alloca() : memref<216xf32>
    %alloca_6 = memref.alloca() : memref<24x6xf32>
    affine.for %arg0 = 0 to 216 {
      affine.store %cst_3, %alloca_5[%arg0] : memref<216xf32>
      affine.store %cst_3, %alloca_4[%arg0] : memref<216xf32>
      affine.store %cst_3, %alloca[%arg0] : memref<216xf32>
    }
    affine.for %arg0 = 0 to 24 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.muli %1, %c7_i32 : i32
      %3 = arith.addi %2, %c3_i32 : i32
      %4 = arith.remsi %3, %c97_i32 : i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.divf %5, %cst_2 : f32
      affine.store %6, %alloca_6[%arg0, 0] : memref<24x6xf32>
      %7 = arith.muli %1, %c11_i32 : i32
      %8 = arith.addi %7, %c5_i32 : i32
      %9 = arith.remsi %8, %c89_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.divf %10, %cst_1 : f32
      affine.store %11, %alloca_6[%arg0, 1] : memref<24x6xf32>
      %12 = arith.muli %1, %c13_i32 : i32
      %13 = arith.addi %12, %c7_i32 : i32
      %14 = arith.remsi %13, %c83_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.divf %15, %cst_0 : f32
      affine.store %16, %alloca_6[%arg0, 2] : memref<24x6xf32>
      %17 = affine.load %alloca_6[%arg0, 0] : memref<24x6xf32>
      %18 = arith.subf %cst, %17 : f32
      affine.store %18, %alloca_6[%arg0, 3] : memref<24x6xf32>
      %19 = affine.load %alloca_6[%arg0, 1] : memref<24x6xf32>
      %20 = arith.subf %cst, %19 : f32
      affine.store %20, %alloca_6[%arg0, 4] : memref<24x6xf32>
      %21 = affine.load %alloca_6[%arg0, 2] : memref<24x6xf32>
      %22 = arith.subf %cst, %21 : f32
      affine.store %22, %alloca_6[%arg0, 5] : memref<24x6xf32>
    }
    %cast = memref.cast %alloca_6 : memref<24x6xf32> to memref<?x6xf32>
    %cast_7 = memref.cast %alloca_5 : memref<216xf32> to memref<?xf32>
    call @_ZL16splat_divergencePA6_fPf(%cast, %cast_7) : (memref<?x6xf32>, memref<?xf32>) -> ()
    %cast_8 = memref.cast %alloca_4 : memref<216xf32> to memref<?xf32>
    %cast_9 = memref.cast %alloca : memref<216xf32> to memref<?xf32>
    call @_ZL13solve_poissonPfS_S_(%cast_7, %cast_8, %cast_9) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    %0 = call @_ZL21extract_surface_proxyPf(%cast_8) : (memref<?xf32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL16splat_divergencePA6_fPf(%arg0: memref<?x6xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %cst = arith.constant 6.000000e+00 : f32
    %c23_i32 = arith.constant 5 : i32
    %c24_i32 = arith.constant 6 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg2 = 0 to 24 {
      %0 = affine.load %arg0[%arg2, 0] : memref<?x6xf32>
      %1 = arith.mulf %0, %cst : f32
      %2 = arith.fptosi %1 : f32 to i32
      %3 = func.call @_ZL7clamp_iiii(%2, %c0_i32, %c23_i32) : (i32, i32, i32) -> i32
      %4 = affine.load %arg0[%arg2, 1] : memref<?x6xf32>
      %5 = arith.mulf %4, %cst : f32
      %6 = arith.fptosi %5 : f32 to i32
      %7 = func.call @_ZL7clamp_iiii(%6, %c0_i32, %c23_i32) : (i32, i32, i32) -> i32
      %8 = affine.load %arg0[%arg2, 2] : memref<?x6xf32>
      %9 = arith.mulf %8, %cst : f32
      %10 = arith.fptosi %9 : f32 to i32
      %11 = func.call @_ZL7clamp_iiii(%10, %c0_i32, %c23_i32) : (i32, i32, i32) -> i32
      %12 = arith.muli %3, %c24_i32 : i32
      %13 = arith.addi %12, %7 : i32
      %14 = arith.muli %13, %c24_i32 : i32
      %15 = arith.addi %14, %11 : i32
      %16 = arith.index_cast %15 : i32 to index
      %17 = affine.load %arg0[%arg2, 3] : memref<?x6xf32>
      %18 = affine.load %arg0[%arg2, 4] : memref<?x6xf32>
      %19 = arith.addf %17, %18 : f32
      %20 = affine.load %arg0[%arg2, 5] : memref<?x6xf32>
      %21 = arith.addf %19, %20 : f32
      %22 = memref.load %arg1[%16] : memref<?xf32>
      %23 = arith.addf %22, %21 : f32
      memref.store %23, %arg1[%16] : memref<?xf32>
    }
    return
  }
  func.func private @_ZL13solve_poissonPfS_S_(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 6.000000e+00 : f32
    %c24_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 4 {
        %0 = affine.apply #map(%arg4)
        %1 = arith.index_cast %0 : index to i32
        affine.for %arg5 = 0 to 4 {
          %2 = affine.apply #map(%arg5)
          %3 = arith.index_cast %2 : index to i32
          affine.for %arg6 = 0 to 4 {
            %4 = affine.apply #map(%arg6)
            %5 = arith.index_cast %4 : index to i32
            %6 = arith.muli %1, %c24_i32 : i32
            %7 = arith.addi %6, %3 : i32
            %8 = arith.muli %7, %c24_i32 : i32
            %9 = arith.addi %8, %5 : i32
            %10 = arith.addi %1, %c-1_i32 : i32
            %11 = arith.muli %10, %c24_i32 : i32
            %12 = arith.addi %11, %3 : i32
            %13 = arith.muli %12, %c24_i32 : i32
            %14 = arith.addi %13, %5 : i32
            %15 = arith.addi %1, %c1_i32 : i32
            %16 = arith.muli %15, %c24_i32 : i32
            %17 = arith.addi %16, %3 : i32
            %18 = arith.muli %17, %c24_i32 : i32
            %19 = arith.addi %18, %5 : i32
            %20 = arith.addi %3, %c-1_i32 : i32
            %21 = arith.addi %6, %20 : i32
            %22 = arith.muli %21, %c24_i32 : i32
            %23 = arith.addi %22, %5 : i32
            %24 = arith.addi %3, %c1_i32 : i32
            %25 = arith.addi %6, %24 : i32
            %26 = arith.muli %25, %c24_i32 : i32
            %27 = arith.addi %26, %5 : i32
            %28 = arith.addi %5, %c-1_i32 : i32
            %29 = arith.addi %8, %28 : i32
            %30 = arith.addi %5, %c1_i32 : i32
            %31 = arith.addi %8, %30 : i32
            %32 = arith.index_cast %14 : i32 to index
            %33 = memref.load %arg1[%32] : memref<?xf32>
            %34 = arith.index_cast %19 : i32 to index
            %35 = memref.load %arg1[%34] : memref<?xf32>
            %36 = arith.addf %33, %35 : f32
            %37 = arith.index_cast %23 : i32 to index
            %38 = memref.load %arg1[%37] : memref<?xf32>
            %39 = arith.addf %36, %38 : f32
            %40 = arith.index_cast %27 : i32 to index
            %41 = memref.load %arg1[%40] : memref<?xf32>
            %42 = arith.addf %39, %41 : f32
            %43 = arith.index_cast %29 : i32 to index
            %44 = memref.load %arg1[%43] : memref<?xf32>
            %45 = arith.addf %42, %44 : f32
            %46 = arith.index_cast %31 : i32 to index
            %47 = memref.load %arg1[%46] : memref<?xf32>
            %48 = arith.addf %45, %47 : f32
            %49 = arith.index_cast %9 : i32 to index
            %50 = memref.load %arg0[%49] : memref<?xf32>
            %51 = arith.addf %50, %48 : f32
            %52 = arith.divf %51, %cst : f32
            memref.store %52, %arg2[%49] : memref<?xf32>
          }
        }
      }
      affine.for %arg4 = 0 to 4 {
        %0 = affine.apply #map(%arg4)
        %1 = arith.index_cast %0 : index to i32
        affine.for %arg5 = 0 to 4 {
          %2 = affine.apply #map(%arg5)
          %3 = arith.index_cast %2 : index to i32
          affine.for %arg6 = 0 to 4 {
            %4 = affine.apply #map(%arg6)
            %5 = arith.index_cast %4 : index to i32
            %6 = arith.muli %1, %c24_i32 : i32
            %7 = arith.addi %6, %3 : i32
            %8 = arith.muli %7, %c24_i32 : i32
            %9 = arith.addi %8, %5 : i32
            %10 = arith.index_cast %9 : i32 to index
            %11 = memref.load %arg2[%10] : memref<?xf32>
            memref.store %11, %arg1[%10] : memref<?xf32>
          }
        }
      }
    }
    return
  }
  func.func private @_ZL21extract_surface_proxyPf(%arg0: memref<?xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c24_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg1 = 0 to 4 {
      %1 = affine.apply #map(%arg1)
      %2 = arith.index_cast %1 : index to i32
      affine.for %arg2 = 0 to 4 {
        %3 = affine.apply #map(%arg2)
        %4 = arith.index_cast %3 : index to i32
        affine.for %arg3 = 0 to 4 {
          %5 = affine.apply #map(%arg3)
          %6 = arith.index_cast %5 : index to i32
          %7 = arith.muli %2, %c24_i32 : i32
          %8 = arith.addi %7, %4 : i32
          %9 = arith.muli %8, %c24_i32 : i32
          %10 = arith.addi %9, %6 : i32
          %11 = arith.addi %2, %c1_i32 : i32
          %12 = arith.muli %11, %c24_i32 : i32
          %13 = arith.addi %12, %4 : i32
          %14 = arith.muli %13, %c24_i32 : i32
          %15 = arith.addi %14, %6 : i32
          %16 = arith.addi %4, %c1_i32 : i32
          %17 = arith.addi %7, %16 : i32
          %18 = arith.muli %17, %c24_i32 : i32
          %19 = arith.addi %18, %6 : i32
          %20 = arith.addi %6, %c1_i32 : i32
          %21 = arith.addi %9, %20 : i32
          %22 = arith.index_cast %10 : i32 to index
          %23 = memref.load %arg0[%22] : memref<?xf32>
          %24 = arith.index_cast %15 : i32 to index
          %25 = memref.load %arg0[%24] : memref<?xf32>
          %26 = arith.mulf %23, %25 : f32
          %27 = arith.cmpf olt, %26, %cst : f32
          %28 = arith.extui %27 : i1 to i32
          %29 = affine.load %alloca[] : memref<i32>
          %30 = arith.addi %29, %28 : i32
          affine.store %30, %alloca[] : memref<i32>
          %31 = arith.index_cast %19 : i32 to index
          %32 = memref.load %arg0[%31] : memref<?xf32>
          %33 = arith.mulf %23, %32 : f32
          %34 = arith.cmpf olt, %33, %cst : f32
          %35 = arith.extui %34 : i1 to i32
          %36 = arith.addi %30, %35 : i32
          affine.store %36, %alloca[] : memref<i32>
          %37 = arith.index_cast %21 : i32 to index
          %38 = memref.load %arg0[%37] : memref<?xf32>
          %39 = arith.mulf %23, %38 : f32
          %40 = arith.cmpf olt, %39, %cst : f32
          %41 = arith.extui %40 : i1 to i32
          %42 = arith.addi %36, %41 : i32
          affine.store %42, %alloca[] : memref<i32>
        }
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
}

