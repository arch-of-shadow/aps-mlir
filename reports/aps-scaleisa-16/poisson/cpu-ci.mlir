module attributes {scaleisa.ci_instructions = 3 : i64, scaleisa.ci_sites = 3 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c24 = arith.constant 24 : index
    %c1 = arith.constant 1 : index
    %c216 = arith.constant 216 : index
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst = arith.constant 6.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 8.300000e+01 : f32
    %c83_i32 = arith.constant 83 : i32
    %c13_i32 = arith.constant 13 : i32
    %cst_2 = arith.constant 8.900000e+01 : f32
    %c89_i32 = arith.constant 89 : i32
    %c5_i32 = arith.constant 5 : i32
    %c11_i32 = arith.constant 11 : i32
    %cst_3 = arith.constant 9.700000e+01 : f32
    %c97_i32 = arith.constant 97 : i32
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<216xf32>
    %alloca_5 = memref.alloca() {alignment = 64 : i64} : memref<216xf32>
    %alloca_6 = memref.alloca() {alignment = 64 : i64} : memref<216xf32>
    %alloca_7 = memref.alloca() : memref<24x6xf32>
    scf.for %arg0 = %c0 to %c216 step %c1 {
      memref.store %cst_4, %alloca_6[%arg0] : memref<216xf32>
      memref.store %cst_4, %alloca_5[%arg0] : memref<216xf32>
      memref.store %cst_4, %alloca[%arg0] : memref<216xf32>
    }
    scf.for %arg0 = %c0 to %c24 step %c1 {
      %3 = arith.index_cast %arg0 : index to i32
      %4 = arith.muli %3, %c7_i32 : i32
      %5 = arith.addi %4, %c3_i32 : i32
      %6 = arith.remsi %5, %c97_i32 : i32
      %7 = arith.sitofp %6 : i32 to f32
      %8 = arith.divf %7, %cst_3 : f32
      memref.store %8, %alloca_7[%arg0, %c0] : memref<24x6xf32>
      %9 = arith.muli %3, %c11_i32 : i32
      %10 = arith.addi %9, %c5_i32 : i32
      %11 = arith.remsi %10, %c89_i32 : i32
      %12 = arith.sitofp %11 : i32 to f32
      %13 = arith.divf %12, %cst_2 : f32
      memref.store %13, %alloca_7[%arg0, %c1] : memref<24x6xf32>
      %14 = arith.muli %3, %c13_i32 : i32
      %15 = arith.addi %14, %c7_i32 : i32
      %16 = arith.remsi %15, %c83_i32 : i32
      %17 = arith.sitofp %16 : i32 to f32
      %18 = arith.divf %17, %cst_1 : f32
      memref.store %18, %alloca_7[%arg0, %c2] : memref<24x6xf32>
      %19 = memref.load %alloca_7[%arg0, %c0] : memref<24x6xf32>
      %20 = arith.subf %cst_0, %19 : f32
      memref.store %20, %alloca_7[%arg0, %c3] : memref<24x6xf32>
      %21 = memref.load %alloca_7[%arg0, %c1] : memref<24x6xf32>
      %22 = arith.subf %cst_0, %21 : f32
      memref.store %22, %alloca_7[%arg0, %c4] : memref<24x6xf32>
      %23 = memref.load %alloca_7[%arg0, %c2] : memref<24x6xf32>
      %24 = arith.subf %cst_0, %23 : f32
      memref.store %24, %alloca_7[%arg0, %c5] : memref<24x6xf32>
    }
    scf.for %arg0 = %c0 to %c24 step %c1 {
      %3 = memref.load %alloca_7[%arg0, %c0] : memref<24x6xf32>
      %4 = arith.mulf %3, %cst : f32
      %5 = arith.fptosi %4 : f32 to i32
      %6 = arith.cmpi sge, %5, %c0_i32 : i32
      %7 = arith.cmpi sgt, %5, %c5_i32 : i32
      %8 = arith.andi %6, %7 : i1
      %9 = arith.subi %true, %8 : i1
      %10 = arith.andi %9, %6 : i1
      %11 = arith.select %8, %c5_i32, %c0_i32 : i32
      %12 = arith.select %10, %5, %11 : i32
      %13 = memref.load %alloca_7[%arg0, %c1] : memref<24x6xf32>
      %14 = arith.mulf %13, %cst : f32
      %15 = arith.fptosi %14 : f32 to i32
      %16 = arith.cmpi sge, %15, %c0_i32 : i32
      %17 = arith.cmpi sgt, %15, %c5_i32 : i32
      %18 = arith.andi %16, %17 : i1
      %19 = arith.xori %18, %true : i1
      %20 = arith.andi %19, %16 : i1
      %21 = arith.select %18, %c5_i32, %c0_i32 : i32
      %22 = arith.select %20, %15, %21 : i32
      %23 = memref.load %alloca_7[%arg0, %c2] : memref<24x6xf32>
      %24 = arith.mulf %23, %cst : f32
      %25 = arith.fptosi %24 : f32 to i32
      %26 = arith.cmpi sge, %25, %c0_i32 : i32
      %27 = arith.cmpi sgt, %25, %c5_i32 : i32
      %28 = arith.andi %26, %27 : i1
      %29 = arith.xori %28, %true : i1
      %30 = arith.andi %29, %26 : i1
      %31 = arith.select %28, %c5_i32, %c0_i32 : i32
      %32 = arith.select %30, %25, %31 : i32
      %33 = arith.muli %12, %c6_i32 : i32
      %34 = arith.addi %33, %22 : i32
      %35 = arith.muli %34, %c6_i32 : i32
      %36 = arith.addi %35, %32 : i32
      %37 = arith.index_cast %36 : i32 to index
      %38 = memref.load %alloca_7[%arg0, %c3] : memref<24x6xf32>
      %39 = memref.load %alloca_7[%arg0, %c4] : memref<24x6xf32>
      %40 = arith.addf %38, %39 : f32
      %41 = memref.load %alloca_7[%arg0, %c5] : memref<24x6xf32>
      %42 = arith.addf %40, %41 : f32
      %43 = memref.load %alloca_6[%37] : memref<216xf32>
      %44 = arith.addf %43, %42 : f32
      memref.store %44, %alloca_6[%37] : memref<216xf32>
    }
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %intptr_10 = memref.extract_aligned_pointer_as_index %alloca_5 : memref<216xf32> -> index
      %3 = arith.index_castui %intptr_10 : index to i32
      %intptr_11 = memref.extract_aligned_pointer_as_index %alloca_6 : memref<216xf32> -> index
      %4 = arith.index_castui %intptr_11 : index to i32
      %intptr_12 = memref.extract_aligned_pointer_as_index %alloca : memref<216xf32> -> index
      %5 = arith.index_castui %intptr_12 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0A.insn r 43, 3, 2, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %3, %4, %5 : (i32, i32, i32) -> ()
      %intptr_13 = memref.extract_aligned_pointer_as_index %alloca : memref<216xf32> -> index
      %6 = arith.index_castui %intptr_13 : index to i32
      %intptr_14 = memref.extract_aligned_pointer_as_index %alloca_5 : memref<216xf32> -> index
      %7 = arith.index_castui %intptr_14 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %6, %7 : (i32, i32) -> ()
    }
    %alloca_8 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_8[] : memref<i32>
    memref.store %c0_i32, %alloca_8[] : memref<i32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_5 : memref<216xf32> -> index
    %0 = arith.index_castui %intptr : index to i32
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloca_8 : memref<i32> -> index
    %1 = arith.index_castui %intptr_9 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %0, %1 : (i32, i32) -> ()
    %2 = memref.load %alloca_8[] : memref<i32>
    call @pcl_validate_outputs(%alloca_7, %alloca_6, %alloca_5, %alloca) : (memref<24x6xf32>, memref<216xf32>, memref<216xf32>, memref<216xf32>) -> ()
    return %2 : i32
  }
  func.func private @pcl_validate_outputs(memref<24x6xf32>, memref<216xf32>, memref<216xf32>, memref<216xf32>)
}

