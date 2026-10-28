module {
  func.func @au_generated_candidate_24__direct(%arg0: memref<16x16xf32>, %arg1: index, %arg2: i32, %arg3: memref<16x16xi32>) attributes {au.generated = true} {
    affine.for %arg4 = 0 to 16 {
      %0 = arith.index_cast %arg4 : index to i32
      %1 = affine.load %arg0[%arg1, %arg4] : memref<16x16xf32>
      %cst = arith.constant 2.000000e-02 : f32
      %2 = arith.cmpf ogt, %1, %cst : f32
      %3 = arith.extui %2 : i1 to i32
      %alloca = memref.alloca() : memref<i32>
      affine.store %3, %alloca[] : memref<i32>
      affine.for %arg5 = 0 to 3 {
        %5 = arith.index_cast %arg5 : index to i32
        %c-1_i32 = arith.constant -1 : i32
        %6 = arith.addi %5, %c-1_i32 : i32
        affine.for %arg6 = 0 to 3 {
          %7 = arith.index_cast %arg6 : index to i32
          %8 = arith.addi %7, %c-1_i32 : i32
          %9 = arith.addi %arg2, %6 : i32
          %c0_i32 = arith.constant 0 : i32
          %10 = arith.cmpi sge, %9, %c0_i32 : i32
          %c15_i32 = arith.constant 15 : i32
          %11 = arith.cmpi sgt, %9, %c15_i32 : i32
          %12 = arith.andi %10, %11 : i1
          %true = arith.constant true
          %13 = arith.xori %12, %true : i1
          %14 = arith.andi %13, %10 : i1
          %15 = arith.select %12, %c15_i32, %c0_i32 : i32
          %16 = arith.select %14, %9, %15 : i32
          %17 = arith.addi %0, %8 : i32
          %18 = arith.cmpi sge, %17, %c0_i32 : i32
          %19 = arith.cmpi sgt, %17, %c15_i32 : i32
          %20 = arith.andi %18, %19 : i1
          %21 = arith.xori %20, %true : i1
          %22 = arith.andi %21, %18 : i1
          %23 = arith.select %20, %c15_i32, %c0_i32 : i32
          %24 = arith.select %22, %17, %23 : i32
          %25 = arith.cmpi ne, %6, %c0_i32 : i32
          %26 = scf.if %25 -> (i1) {
            scf.yield %true : i1
          } else {
            %27 = arith.cmpi ne, %8, %c0_i32 : i32
            scf.yield %27 : i1
          }
          scf.if %26 {
            %27 = arith.index_cast %16 : i32 to index
            %28 = arith.index_cast %24 : i32 to index
            %29 = memref.load %arg0[%27, %28] : memref<16x16xf32>
            %30 = arith.cmpf ogt, %29, %1 : f32
            scf.if %30 {
              affine.store %c0_i32, %alloca[] : memref<i32>
            }
          }
        }
      }
      %4 = affine.load %alloca[] : memref<i32>
      affine.store %4, %arg3[%arg1, %arg4] : memref<16x16xi32>
    }
    return
  }
}

