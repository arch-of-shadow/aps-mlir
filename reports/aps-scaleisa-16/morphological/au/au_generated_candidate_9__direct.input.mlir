module {
  func.func @au_generated_candidate_9__direct(%arg0: memref<12x12xf32>, %arg1: index, %arg2: i32, %arg3: memref<12x12xf32>) attributes {au.generated = true} {
    affine.for %arg4 = 0 to 12 {
      %0 = arith.index_cast %arg4 : index to i32
      %1 = affine.load %arg0[%arg1, %arg4] : memref<12x12xf32>
      %alloca = memref.alloca() : memref<f32>
      affine.store %1, %alloca[] : memref<f32>
      affine.for %arg5 = 0 to 5 {
        %3 = arith.index_cast %arg5 : index to i32
        %c-2_i32 = arith.constant -2 : i32
        %4 = arith.addi %3, %c-2_i32 : i32
        affine.for %arg6 = 0 to 5 {
          %5 = arith.index_cast %arg6 : index to i32
          %6 = arith.addi %5, %c-2_i32 : i32
          %7 = arith.addi %arg2, %4 : i32
          %c0_i32 = arith.constant 0 : i32
          %8 = arith.cmpi sge, %7, %c0_i32 : i32
          %c11_i32 = arith.constant 11 : i32
          %9 = arith.cmpi sgt, %7, %c11_i32 : i32
          %10 = arith.andi %8, %9 : i1
          %true = arith.constant true
          %11 = arith.xori %10, %true : i1
          %12 = arith.andi %11, %8 : i1
          %13 = arith.select %10, %c11_i32, %c0_i32 : i32
          %14 = arith.select %12, %7, %13 : i32
          %15 = arith.addi %0, %6 : i32
          %16 = arith.cmpi sge, %15, %c0_i32 : i32
          %17 = arith.cmpi sgt, %15, %c11_i32 : i32
          %18 = arith.andi %16, %17 : i1
          %19 = arith.xori %18, %true : i1
          %20 = arith.andi %19, %16 : i1
          %21 = arith.select %18, %c11_i32, %c0_i32 : i32
          %22 = arith.select %20, %15, %21 : i32
          %23 = arith.index_cast %14 : i32 to index
          %24 = arith.index_cast %22 : i32 to index
          %25 = memref.load %arg0[%23, %24] : memref<12x12xf32>
          %26 = affine.load %alloca[] : memref<f32>
          %27 = arith.cmpf olt, %25, %26 : f32
          scf.if %27 {
            affine.store %25, %alloca[] : memref<f32>
          }
        }
      }
      %2 = affine.load %alloca[] : memref<f32>
      affine.store %2, %arg3[%arg1, %arg4] : memref<12x12xf32>
    }
    return
  }
}

