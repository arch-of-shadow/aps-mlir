module attributes {scaleisa.ci_instructions = 5 : i64, scaleisa.ci_sites = 5 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c17_i32 = arith.constant 17 : i32
    %cst = arith.constant 2.000000e-02 : f32
    %c7_i32 = arith.constant 7 : i32
    %c13_i32 = arith.constant 13 : i32
    %c51_i32 = arith.constant 51 : i32
    %c-53_i32 = arith.constant -53 : i32
    %c-31_i32 = arith.constant -31 : i32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<12x12xf32>
    %alloca_0 = memref.alloca() {alignment = 64 : i64} : memref<12x12xf32>
    %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<12x12xf32>
    %alloca_2 = memref.alloca() {alignment = 64 : i64} : memref<12x12xf32>
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %3 = arith.index_cast %arg0 : index to i32
      %4 = arith.addi %3, %c-53_i32 : i32
      %5 = arith.muli %4, %4 : i32
      %6 = arith.muli %3, %c13_i32 : i32
      scf.for %arg1 = %c0 to %c12 step %c2 {
        %7 = arith.addi %arg1, %c2 : index
        scf.for %arg2 = %arg1 to %7 step %c1 {
          %8 = arith.index_cast %arg2 : index to i32
          %9 = arith.addi %8, %c-31_i32 : i32
          %10 = arith.muli %9, %9 : i32
          %11 = arith.addi %10, %5 : i32
          %12 = arith.remsi %11, %c17_i32 : i32
          %13 = arith.muli %8, %c7_i32 : i32
          %14 = arith.addi %13, %6 : i32
          %15 = arith.addi %14, %12 : i32
          %16 = arith.remsi %15, %c51_i32 : i32
          %17 = arith.sitofp %16 : i32 to f32
          %18 = arith.mulf %17, %cst : f32
          memref.store %18, %alloca_2[%arg0, %arg2] : memref<12x12xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %3 = arith.index_cast %arg0 : index to i32
      %intptr_5 = memref.extract_aligned_pointer_as_index %alloca_2 : memref<12x12xf32> -> index
      %4 = arith.index_castui %intptr_5 : index to i32
      %5 = arith.index_castui %arg0 : index to i32
      %intptr_6 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<12x12xf32> -> index
      %6 = arith.index_castui %intptr_6 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2051, $2\0Acsrrw x0, 2052, $3\0A.insn r 43, 3, 2, x0, $0, $1\0Afence rw, rw", "r,r,r,r,~{memory}" %4, %5, %3, %6 : (i32, i32, i32, i32) -> ()
    }
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %3 = arith.index_cast %arg0 : index to i32
      %intptr_5 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<12x12xf32> -> index
      %4 = arith.index_castui %intptr_5 : index to i32
      %5 = arith.index_castui %arg0 : index to i32
      %intptr_6 = memref.extract_aligned_pointer_as_index %alloca_0 : memref<12x12xf32> -> index
      %6 = arith.index_castui %intptr_6 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0Acsrrw x0, 2050, $3\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,r,r,~{memory}" %4, %5, %3, %6 : (i32, i32, i32, i32) -> ()
    }
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %3 = arith.index_cast %arg0 : index to i32
      %intptr_5 = memref.extract_aligned_pointer_as_index %alloca_0 : memref<12x12xf32> -> index
      %4 = arith.index_castui %intptr_5 : index to i32
      %5 = arith.index_castui %arg0 : index to i32
      %intptr_6 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<12x12xf32> -> index
      %6 = arith.index_castui %intptr_6 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0Acsrrw x0, 2050, $3\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,r,r,~{memory}" %4, %5, %3, %6 : (i32, i32, i32, i32) -> ()
    }
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %3 = arith.index_cast %arg0 : index to i32
      %intptr_5 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<12x12xf32> -> index
      %4 = arith.index_castui %intptr_5 : index to i32
      %5 = arith.index_castui %arg0 : index to i32
      %intptr_6 = memref.extract_aligned_pointer_as_index %alloca : memref<12x12xf32> -> index
      %6 = arith.index_castui %intptr_6 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2051, $2\0Acsrrw x0, 2052, $3\0A.insn r 43, 3, 2, x0, $0, $1\0Afence rw, rw", "r,r,r,r,~{memory}" %4, %5, %3, %6 : (i32, i32, i32, i32) -> ()
    }
    %alloca_3 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_3[] : memref<i32>
    memref.store %c0_i32, %alloca_3[] : memref<i32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca : memref<12x12xf32> -> index
    %0 = arith.index_castui %intptr : index to i32
    %intptr_4 = memref.extract_aligned_pointer_as_index %alloca_3 : memref<i32> -> index
    %1 = arith.index_castui %intptr_4 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %0, %1 : (i32, i32) -> ()
    %2 = memref.load %alloca_3[] : memref<i32>
    call @pcl_validate_outputs(%alloca_2, %alloca_1, %alloca_0, %alloca) : (memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>) -> ()
    return %2 : i32
  }
  func.func private @pcl_validate_outputs(memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>, memref<12x12xf32>)
}

