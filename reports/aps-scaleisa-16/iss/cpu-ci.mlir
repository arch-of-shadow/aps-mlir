module attributes {scaleisa.ci_instructions = 3 : i64, scaleisa.ci_sites = 3 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c19_i32 = arith.constant 19 : i32
    %c9_i32 = arith.constant 9 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst = arith.constant 2.000000e-02 : f32
    %cst_0 = arith.constant 3.000000e-02 : f32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<16x16xi32>
    %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<16x16xf32>
    %alloca_2 = memref.alloca() {alignment = 64 : i64} : memref<16x16x3xf32>
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %4 = arith.index_cast %arg0 : index to i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.mulf %5, %cst_0 : f32
      %7 = arith.muli %4, %c9_i32 : i32
      scf.for %arg1 = %c0 to %c16 step %c4 {
        %8 = arith.addi %arg1, %c4 : index
        scf.for %arg2 = %arg1 to %8 step %c1 {
          %9 = arith.index_cast %arg2 : index to i32
          %10 = arith.sitofp %9 : i32 to f32
          %11 = arith.mulf %10, %cst_0 : f32
          memref.store %11, %alloca_2[%arg0, %arg2, %c0] : memref<16x16x3xf32>
          memref.store %6, %alloca_2[%arg0, %arg2, %c1] : memref<16x16x3xf32>
          %12 = arith.muli %9, %c5_i32 : i32
          %13 = arith.addi %12, %7 : i32
          %14 = arith.xori %9, %4 : i32
          %15 = arith.addi %13, %14 : i32
          %16 = arith.remsi %15, %c19_i32 : i32
          %17 = arith.sitofp %16 : i32 to f32
          %18 = arith.mulf %17, %cst : f32
          memref.store %18, %alloca_2[%arg0, %arg2, %c2] : memref<16x16x3xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %4 = arith.index_cast %arg0 : index to i32
      %intptr_6 = memref.extract_aligned_pointer_as_index %alloca_2 : memref<16x16x3xf32> -> index
      %5 = arith.index_castui %intptr_6 : index to i32
      %6 = arith.index_castui %arg0 : index to i32
      %intptr_7 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<16x16xf32> -> index
      %7 = arith.index_castui %intptr_7 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0Acsrrw x0, 2050, $3\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,r,r,~{memory}" %5, %6, %4, %7 : (i32, i32, i32, i32) -> ()
    }
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %4 = arith.index_cast %arg0 : index to i32
      %intptr_6 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<16x16xf32> -> index
      %5 = arith.index_castui %intptr_6 : index to i32
      %6 = arith.index_castui %arg0 : index to i32
      %intptr_7 = memref.extract_aligned_pointer_as_index %alloca : memref<16x16xi32> -> index
      %7 = arith.index_castui %intptr_7 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2051, $2\0Acsrrw x0, 2052, $3\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,r,r,~{memory}" %5, %6, %4, %7 : (i32, i32, i32, i32) -> ()
    }
    %alloca_3 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_3[] : memref<i32>
    memref.store %c0_i32, %alloca_3[] : memref<i32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca : memref<16x16xi32> -> index
    %0 = arith.index_castui %intptr : index to i32
    %intptr_4 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<16x16xf32> -> index
    %1 = arith.index_castui %intptr_4 : index to i32
    %intptr_5 = memref.extract_aligned_pointer_as_index %alloca_3 : memref<i32> -> index
    %2 = arith.index_castui %intptr_5 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2053, $2\0A.insn r 43, 3, 2, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %0, %1, %2 : (i32, i32, i32) -> ()
    %3 = memref.load %alloca_3[] : memref<i32>
    call @pcl_validate_outputs(%alloca_2, %alloca_1, %alloca) : (memref<16x16x3xf32>, memref<16x16xf32>, memref<16x16xi32>) -> ()
    return %3 : i32
  }
  func.func private @pcl_validate_outputs(memref<16x16x3xf32>, memref<16x16xf32>, memref<16x16xi32>)
}

