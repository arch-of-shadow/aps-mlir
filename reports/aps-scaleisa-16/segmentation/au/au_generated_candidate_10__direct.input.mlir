module {
  func.func @au_generated_candidate_10__direct(%arg0: memref<8x8xf32>, %arg1: memref<8x8xf32>) attributes {au.generated = true} {
    affine.for %arg2 = 0 to 8 {
      %0 = arith.index_cast %arg2 : index to i32
      affine.for %arg3 = 0 to 8 {
        %1 = arith.index_cast %arg3 : index to i32
        %c-1_i32 = arith.constant -1 : i32
        %2 = arith.addi %1, %c-1_i32 : i32
        %c0_i32 = arith.constant 0 : i32
        %3 = arith.cmpi sge, %2, %c0_i32 : i32
        %c7_i32 = arith.constant 7 : i32
        %4 = arith.cmpi sgt, %2, %c7_i32 : i32
        %5 = arith.andi %3, %4 : i1
        %true = arith.constant true
        %6 = arith.xori %5, %true : i1
        %7 = arith.andi %6, %3 : i1
        %8 = arith.select %5, %c7_i32, %c0_i32 : i32
        %9 = arith.select %7, %2, %8 : i32
        %c1_i32 = arith.constant 1 : i32
        %10 = arith.addi %1, %c1_i32 : i32
        %11 = arith.cmpi sge, %10, %c0_i32 : i32
        %12 = arith.cmpi sgt, %10, %c7_i32 : i32
        %13 = arith.andi %11, %12 : i1
        %14 = arith.xori %13, %true : i1
        %15 = arith.andi %14, %11 : i1
        %16 = arith.select %13, %c7_i32, %c0_i32 : i32
        %17 = arith.select %15, %10, %16 : i32
        %18 = arith.addi %0, %c-1_i32 : i32
        %19 = arith.cmpi sge, %18, %c0_i32 : i32
        %20 = arith.cmpi sgt, %18, %c7_i32 : i32
        %21 = arith.andi %19, %20 : i1
        %22 = arith.xori %21, %true : i1
        %23 = arith.andi %22, %19 : i1
        %24 = arith.select %21, %c7_i32, %c0_i32 : i32
        %25 = arith.select %23, %18, %24 : i32
        %26 = arith.addi %0, %c1_i32 : i32
        %27 = arith.cmpi sge, %26, %c0_i32 : i32
        %28 = arith.cmpi sgt, %26, %c7_i32 : i32
        %29 = arith.andi %27, %28 : i1
        %30 = arith.subi %true, %29 : i1
        %31 = arith.andi %30, %27 : i1
        %32 = arith.select %29, %c7_i32, %c0_i32 : i32
        %33 = arith.select %31, %26, %32 : i32
        %34 = arith.index_cast %17 : i32 to index
        %35 = memref.load %arg0[%arg2, %34] : memref<8x8xf32>
        %36 = arith.index_cast %9 : i32 to index
        %37 = memref.load %arg0[%arg2, %36] : memref<8x8xf32>
        %38 = arith.subf %35, %37 : f32
        %cst = arith.constant 0.000000e+00 : f32
        %39 = arith.cmpf olt, %38, %cst : f32
        %40 = scf.if %39 -> (f32) {
          %49 = arith.negf %38 : f32
          scf.yield %49 : f32
        } else {
          scf.yield %38 : f32
        }
        %41 = arith.index_cast %33 : i32 to index
        %42 = memref.load %arg0[%41, %arg3] : memref<8x8xf32>
        %43 = arith.index_cast %25 : i32 to index
        %44 = memref.load %arg0[%43, %arg3] : memref<8x8xf32>
        %45 = arith.subf %42, %44 : f32
        %46 = arith.cmpf olt, %45, %cst : f32
        %47 = scf.if %46 -> (f32) {
          %49 = arith.negf %45 : f32
          scf.yield %49 : f32
        } else {
          scf.yield %45 : f32
        }
        %48 = arith.addf %40, %47 : f32
        affine.store %48, %arg1[%arg2, %arg3] : memref<8x8xf32>
      }
    }
    return
  }
}

