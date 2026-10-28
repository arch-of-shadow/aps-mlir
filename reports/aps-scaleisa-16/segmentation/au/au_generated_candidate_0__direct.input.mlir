module {
  func.func @au_generated_candidate_0__direct(%arg0: memref<8x8xi32>, %arg1: index, %arg2: i32, %arg3: memref<8x8xf32>, %arg4: memref<8x8xf32>) attributes {au.generated = true} {
    affine.for %arg5 = 0 to 8 {
      %0 = arith.index_cast %arg5 : index to i32
      %1 = affine.load %arg0[%arg1, %arg5] : memref<8x8xi32>
      %alloca = memref.alloca() : memref<i32>
      affine.store %1, %alloca[] : memref<i32>
      %alloca_0 = memref.alloca() : memref<f32>
      %cst = arith.constant 1.000000e+30 : f32
      affine.store %cst, %alloca_0[] : memref<f32>
      affine.for %arg6 = 0 to 3 {
        %3 = arith.index_cast %arg6 : index to i32
        %c-1_i32 = arith.constant -1 : i32
        %4 = arith.addi %3, %c-1_i32 : i32
        affine.for %arg7 = 0 to 3 {
          %5 = arith.index_cast %arg7 : index to i32
          %6 = arith.addi %5, %c-1_i32 : i32
          %7 = arith.addi %arg2, %4 : i32
          %c0_i32 = arith.constant 0 : i32
          %8 = arith.cmpi sge, %7, %c0_i32 : i32
          %c7_i32 = arith.constant 7 : i32
          %9 = arith.cmpi sgt, %7, %c7_i32 : i32
          %10 = arith.andi %8, %9 : i1
          %true = arith.constant true
          %11 = arith.xori %10, %true : i1
          %12 = arith.andi %11, %8 : i1
          %13 = arith.select %10, %c7_i32, %c0_i32 : i32
          %14 = arith.select %12, %7, %13 : i32
          %15 = arith.addi %0, %6 : i32
          %16 = arith.cmpi sge, %15, %c0_i32 : i32
          %17 = arith.cmpi sgt, %15, %c7_i32 : i32
          %18 = arith.andi %16, %17 : i1
          %19 = arith.xori %18, %true : i1
          %20 = arith.andi %19, %16 : i1
          %21 = arith.select %18, %c7_i32, %c0_i32 : i32
          %22 = arith.select %20, %15, %21 : i32
          %23 = affine.load %arg3[%arg1, %arg5] : memref<8x8xf32>
          %24 = arith.index_cast %14 : i32 to index
          %25 = arith.index_cast %22 : i32 to index
          %26 = memref.load %arg3[%24, %25] : memref<8x8xf32>
          %27 = arith.subf %23, %26 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %28 = arith.cmpf olt, %27, %cst_1 : f32
          %29 = scf.if %28 -> (f32) {
            %35 = arith.negf %27 : f32
            scf.yield %35 : f32
          } else {
            scf.yield %27 : f32
          }
          %30 = memref.load %arg4[%24, %25] : memref<8x8xf32>
          %31 = arith.addf %29, %30 : f32
          %32 = affine.load %alloca_0[] : memref<f32>
          %33 = arith.cmpf olt, %31, %32 : f32
          %34 = scf.if %33 -> (i1) {
            scf.yield %true : i1
          } else {
            %35 = arith.cmpf oeq, %31, %32 : f32
            %36 = scf.if %35 -> (i1) {
              %37 = memref.load %arg0[%24, %25] : memref<8x8xi32>
              %38 = affine.load %alloca[] : memref<i32>
              %39 = arith.cmpi slt, %37, %38 : i32
              scf.yield %39 : i1
            } else {
              %false = arith.constant false
              scf.yield %false : i1
            }
            scf.yield %36 : i1
          }
          scf.if %34 {
            affine.store %31, %alloca_0[] : memref<f32>
            %35 = memref.load %arg0[%24, %25] : memref<8x8xi32>
            affine.store %35, %alloca[] : memref<i32>
          }
        }
      }
      %2 = affine.load %alloca[] : memref<i32>
      affine.store %2, %arg0[%arg1, %arg5] : memref<8x8xi32>
    }
    return
  }
}

