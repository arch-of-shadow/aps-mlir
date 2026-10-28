module attributes {scaleisa.ci_instructions = 2 : i64, scaleisa.ci_sites = 2 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %cst = arith.constant 1.000000e+03 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 1.200000e+03 : f32
    %cst_1 = arith.constant 1.400000e+03 : f32
    %cst_2 = arith.constant 5.000000e+02 : f32
    %c0 = arith.constant 0 : index
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_4 = arith.constant 5.000000e-01 : f32
    %cst_5 = arith.constant 3.200000e+01 : f32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloca = memref.alloca() : memref<4xf32>
    %alloca_6 = memref.alloca() : memref<3xf32>
    %alloca_7 = memref.alloca() {alignment = 64 : i64} : memref<32x3xf32>
    %alloca_8 = memref.alloca() {alignment = 64 : i64} : memref<32x3xf32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_8 : memref<32x3xf32> -> index
    %0 = arith.index_castui %intptr : index to i32
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloca_7 : memref<32x3xf32> -> index
    %1 = arith.index_castui %intptr_9 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %0, %1 : (i32, i32) -> ()
    %alloca_10 = memref.alloca() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_3, %alloca_10[] : memref<f32>
    %alloca_11 = memref.alloca() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_3, %alloca_11[] : memref<f32>
    %alloca_12 = memref.alloca() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_3, %alloca_12[] : memref<f32>
    memref.store %cst_3, %alloca_12[] : memref<f32>
    memref.store %cst_3, %alloca_11[] : memref<f32>
    memref.store %cst_3, %alloca_10[] : memref<f32>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      %alloca_14 = memref.alloca() {alignment = 64 : i64} : memref<f32>
      memref.store %cst_3, %alloca_14[] : memref<f32>
      %alloca_15 = memref.alloca() {alignment = 64 : i64} : memref<f32>
      memref.store %cst_3, %alloca_15[] : memref<f32>
      %alloca_16 = memref.alloca() {alignment = 64 : i64} : memref<f32>
      memref.store %cst_3, %alloca_16[] : memref<f32>
      %alloca_17 = memref.alloca() {alignment = 64 : i64} : memref<f32>
      memref.store %cst_3, %alloca_17[] : memref<f32>
      %intptr_18 = memref.extract_aligned_pointer_as_index %alloca_12 : memref<f32> -> index
      %22 = arith.index_castui %intptr_18 : index to i32
      %intptr_19 = memref.extract_aligned_pointer_as_index %alloca_11 : memref<f32> -> index
      %23 = arith.index_castui %intptr_19 : index to i32
      %intptr_20 = memref.extract_aligned_pointer_as_index %alloca_10 : memref<f32> -> index
      %24 = arith.index_castui %intptr_20 : index to i32
      %intptr_21 = memref.extract_aligned_pointer_as_index %alloca_8 : memref<32x3xf32> -> index
      %25 = arith.index_castui %intptr_21 : index to i32
      %intptr_22 = memref.extract_aligned_pointer_as_index %alloca_7 : memref<32x3xf32> -> index
      %26 = arith.index_castui %intptr_22 : index to i32
      %intptr_23 = memref.extract_aligned_pointer_as_index %alloca_14 : memref<f32> -> index
      %27 = arith.index_castui %intptr_23 : index to i32
      %intptr_24 = memref.extract_aligned_pointer_as_index %alloca_15 : memref<f32> -> index
      %28 = arith.index_castui %intptr_24 : index to i32
      %intptr_25 = memref.extract_aligned_pointer_as_index %alloca_16 : memref<f32> -> index
      %29 = arith.index_castui %intptr_25 : index to i32
      %intptr_26 = memref.extract_aligned_pointer_as_index %alloca_17 : memref<f32> -> index
      %30 = arith.index_castui %intptr_26 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0Acsrrw x0, 2050, $3\0Acsrrw x0, 2051, $4\0Acsrrw x0, 2052, $5\0Acsrrw x0, 2053, $6\0Acsrrw x0, 2054, $7\0Acsrrw x0, 2055, $8\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,r,r,r,r,r,r,r,~{memory}" %22, %23, %24, %25, %26, %27, %28, %29, %30 : (i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
      %31 = memref.load %alloca_14[] : memref<f32>
      %32 = arith.mulf %31, %cst_4 : f32
      %33 = arith.divf %32, %cst_5 : f32
      %34 = memref.load %alloca_12[] : memref<f32>
      %35 = arith.addf %34, %33 : f32
      memref.store %35, %alloca_12[] : memref<f32>
      %36 = memref.load %alloca_15[] : memref<f32>
      %37 = arith.mulf %36, %cst_4 : f32
      %38 = arith.divf %37, %cst_5 : f32
      %39 = memref.load %alloca_11[] : memref<f32>
      %40 = arith.addf %39, %38 : f32
      memref.store %40, %alloca_11[] : memref<f32>
      %41 = memref.load %alloca_16[] : memref<f32>
      %42 = arith.mulf %41, %cst_4 : f32
      %43 = arith.divf %42, %cst_5 : f32
      %44 = memref.load %alloca_10[] : memref<f32>
      %45 = arith.addf %44, %43 : f32
      memref.store %45, %alloca_10[] : memref<f32>
      %46 = memref.load %alloca_17[] : memref<f32>
      %47 = arith.divf %46, %cst_5 : f32
      memref.store %47, %alloca[%arg0] : memref<4xf32>
    }
    %2 = memref.load %alloca_12[] : memref<f32>
    memref.store %2, %alloca_6[%c0] : memref<3xf32>
    %3 = memref.load %alloca_11[] : memref<f32>
    memref.store %3, %alloca_6[%c1] : memref<3xf32>
    %4 = memref.load %alloca_10[] : memref<f32>
    memref.store %4, %alloca_6[%c2] : memref<3xf32>
    %alloca_13 = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca_13[] : memref<i32>
    memref.store %c0_i32, %alloca_13[] : memref<i32>
    %5 = memref.load %alloca_6[%c0] : memref<3xf32>
    %6 = arith.mulf %5, %cst : f32
    %7 = arith.fptosi %6 : f32 to i32
    memref.store %7, %alloca_13[] : memref<i32>
    %8 = memref.load %alloca_6[%c1] : memref<3xf32>
    %9 = arith.mulf %8, %cst_0 : f32
    %10 = arith.fptosi %9 : f32 to i32
    %11 = arith.addi %7, %10 : i32
    memref.store %11, %alloca_13[] : memref<i32>
    %12 = memref.load %alloca_6[%c2] : memref<3xf32>
    %13 = arith.mulf %12, %cst_1 : f32
    %14 = arith.fptosi %13 : f32 to i32
    %15 = arith.addi %11, %14 : i32
    memref.store %15, %alloca_13[] : memref<i32>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      %22 = memref.load %alloca[%arg0] : memref<4xf32>
      %23 = arith.mulf %22, %cst_2 : f32
      %24 = arith.fptosi %23 : f32 to i32
      %25 = memref.load %alloca_13[] : memref<i32>
      %26 = arith.addi %25, %24 : i32
      memref.store %26, %alloca_13[] : memref<i32>
    }
    %16 = memref.load %alloca_13[] : memref<i32>
    %17 = arith.sitofp %16 : i32 to f32
    %18 = arith.cmpi slt, %16, %c0_i32 : i32
    %19 = scf.if %18 -> (f32) {
      %22 = arith.negf %17 : f32
      scf.yield %22 : f32
    } else {
      scf.yield %17 : f32
    }
    %20 = arith.cmpf ogt, %19, %cst_3 : f32
    %21 = arith.select %20, %16, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_8, %alloca_7, %alloca_6, %alloca) : (memref<32x3xf32>, memref<32x3xf32>, memref<3xf32>, memref<4xf32>) -> ()
    return %21 : i32
  }
  func.func private @pcl_validate_outputs(memref<32x3xf32>, memref<32x3xf32>, memref<3xf32>, memref<4xf32>)
}

