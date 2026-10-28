#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @au_generated_candidate_11__direct(%arg0: memref<216xf32>, %arg1: memref<i32>) attributes {au.generated = true} {
    affine.for %arg2 = 0 to 4 {
      %0 = affine.apply #map(%arg2)
      %1 = arith.index_cast %0 : index to i32
      affine.for %arg3 = 0 to 4 {
        %2 = affine.apply #map(%arg3)
        %3 = arith.index_cast %2 : index to i32
        affine.for %arg4 = 0 to 4 {
          %4 = affine.apply #map(%arg4)
          %5 = arith.index_cast %4 : index to i32
          %c6_i32 = arith.constant 6 : i32
          %6 = arith.muli %1, %c6_i32 : i32
          %7 = arith.addi %6, %3 : i32
          %8 = arith.muli %7, %c6_i32 : i32
          %9 = arith.addi %8, %5 : i32
          %c1_i32 = arith.constant 1 : i32
          %10 = arith.addi %1, %c1_i32 : i32
          %11 = arith.muli %10, %c6_i32 : i32
          %12 = arith.addi %11, %3 : i32
          %13 = arith.muli %12, %c6_i32 : i32
          %14 = arith.addi %13, %5 : i32
          %15 = arith.addi %3, %c1_i32 : i32
          %16 = arith.addi %6, %15 : i32
          %17 = arith.muli %16, %c6_i32 : i32
          %18 = arith.addi %17, %5 : i32
          %19 = arith.addi %5, %c1_i32 : i32
          %20 = arith.addi %8, %19 : i32
          %21 = arith.index_cast %9 : i32 to index
          %22 = memref.load %arg0[%21] : memref<216xf32>
          %23 = arith.index_cast %14 : i32 to index
          %24 = memref.load %arg0[%23] : memref<216xf32>
          %25 = arith.mulf %22, %24 : f32
          %cst = arith.constant 0.000000e+00 : f32
          %26 = arith.cmpf olt, %25, %cst : f32
          %27 = arith.extui %26 : i1 to i32
          %28 = affine.load %arg1[] : memref<i32>
          %29 = arith.addi %28, %27 : i32
          affine.store %29, %arg1[] : memref<i32>
          %30 = arith.index_cast %18 : i32 to index
          %31 = memref.load %arg0[%30] : memref<216xf32>
          %32 = arith.mulf %22, %31 : f32
          %33 = arith.cmpf olt, %32, %cst : f32
          %34 = arith.extui %33 : i1 to i32
          %35 = arith.addi %29, %34 : i32
          affine.store %35, %arg1[] : memref<i32>
          %36 = arith.index_cast %20 : i32 to index
          %37 = memref.load %arg0[%36] : memref<216xf32>
          %38 = arith.mulf %22, %37 : f32
          %39 = arith.cmpf olt, %38, %cst : f32
          %40 = arith.extui %39 : i1 to i32
          %41 = arith.addi %35, %40 : i32
          affine.store %41, %arg1[] : memref<i32>
        }
      }
    }
    return
  }
}

