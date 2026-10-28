#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @au_generated_candidate_9__direct(%arg0: memref<216xf32>, %arg1: memref<216xf32>, %arg2: memref<216xf32>) attributes {au.generated = true} {
    affine.for %arg3 = 0 to 4 {
      %0 = affine.apply #map(%arg3)
      %1 = arith.index_cast %0 : index to i32
      %c6_i32 = arith.constant 6 : i32
      %2 = arith.muli %1, %c6_i32 : i32
      %c-1_i32 = arith.constant -1 : i32
      %3 = arith.addi %1, %c-1_i32 : i32
      %4 = arith.muli %3, %c6_i32 : i32
      %c1_i32 = arith.constant 1 : i32
      %5 = arith.addi %1, %c1_i32 : i32
      %6 = arith.muli %5, %c6_i32 : i32
      affine.for %arg4 = 0 to 4 {
        %7 = affine.apply #map(%arg4)
        %8 = arith.index_cast %7 : index to i32
        affine.for %arg5 = 0 to 4 {
          %9 = affine.apply #map(%arg5)
          %10 = arith.index_cast %9 : index to i32
          %11 = arith.addi %2, %8 : i32
          %12 = arith.muli %11, %c6_i32 : i32
          %13 = arith.addi %12, %10 : i32
          %14 = arith.addi %4, %8 : i32
          %15 = arith.muli %14, %c6_i32 : i32
          %16 = arith.addi %15, %10 : i32
          %17 = arith.addi %6, %8 : i32
          %18 = arith.muli %17, %c6_i32 : i32
          %19 = arith.addi %18, %10 : i32
          %20 = arith.addi %8, %c-1_i32 : i32
          %21 = arith.addi %2, %20 : i32
          %22 = arith.muli %21, %c6_i32 : i32
          %23 = arith.addi %22, %10 : i32
          %24 = arith.addi %8, %c1_i32 : i32
          %25 = arith.addi %2, %24 : i32
          %26 = arith.muli %25, %c6_i32 : i32
          %27 = arith.addi %26, %10 : i32
          %28 = arith.addi %10, %c-1_i32 : i32
          %29 = arith.addi %12, %28 : i32
          %30 = arith.addi %10, %c1_i32 : i32
          %31 = arith.addi %12, %30 : i32
          %32 = arith.index_cast %16 : i32 to index
          %33 = memref.load %arg0[%32] : memref<216xf32>
          %34 = arith.index_cast %19 : i32 to index
          %35 = memref.load %arg0[%34] : memref<216xf32>
          %36 = arith.addf %33, %35 : f32
          %37 = arith.index_cast %23 : i32 to index
          %38 = memref.load %arg0[%37] : memref<216xf32>
          %39 = arith.addf %36, %38 : f32
          %40 = arith.index_cast %27 : i32 to index
          %41 = memref.load %arg0[%40] : memref<216xf32>
          %42 = arith.addf %39, %41 : f32
          %43 = arith.index_cast %29 : i32 to index
          %44 = memref.load %arg0[%43] : memref<216xf32>
          %45 = arith.addf %42, %44 : f32
          %46 = arith.index_cast %31 : i32 to index
          %47 = memref.load %arg0[%46] : memref<216xf32>
          %48 = arith.addf %45, %47 : f32
          %49 = arith.index_cast %13 : i32 to index
          %50 = memref.load %arg1[%49] : memref<216xf32>
          %51 = arith.addf %50, %48 : f32
          %cst = arith.constant 6.000000e+00 : f32
          %52 = arith.divf %51, %cst : f32
          memref.store %52, %arg2[%49] : memref<216xf32>
        }
      }
    }
    return
  }
}

