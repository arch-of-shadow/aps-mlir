module attributes {scaleisa.ci_instructions = 4 : i64, scaleisa.ci_sites = 4 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<8x8xi32>
    %alloca_0 = memref.alloca() {alignment = 64 : i64} : memref<8x8xf32>
    %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<8x8xf32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_1 : memref<8x8xf32> -> index
    %0 = arith.index_castui %intptr : index to i32
    %intptr_2 = memref.extract_aligned_pointer_as_index %alloca : memref<8x8xi32> -> index
    %1 = arith.index_castui %intptr_2 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 2, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %0, %1 : (i32, i32) -> ()
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<8x8xf32> -> index
    %2 = arith.index_castui %intptr_3 : index to i32
    %intptr_4 = memref.extract_aligned_pointer_as_index %alloca_0 : memref<8x8xf32> -> index
    %3 = arith.index_castui %intptr_4 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %2, %3 : (i32, i32) -> ()
    scf.for %arg0 = %c0 to %c2 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        %7 = arith.index_cast %arg1 : index to i32
        %intptr_8 = memref.extract_aligned_pointer_as_index %alloca : memref<8x8xi32> -> index
        %8 = arith.index_castui %intptr_8 : index to i32
        %9 = arith.index_castui %arg1 : index to i32
        %intptr_9 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<8x8xf32> -> index
        %10 = arith.index_castui %intptr_9 : index to i32
        %intptr_10 = memref.extract_aligned_pointer_as_index %alloca_0 : memref<8x8xf32> -> index
        %11 = arith.index_castui %intptr_10 : index to i32
        llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0Acsrrw x0, 2050, $3\0Acsrrw x0, 2051, $4\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,r,r,r,~{memory}" %8, %9, %7, %10, %11 : (i32, i32, i32, i32, i32) -> ()
      }
    }
    %alloca_5 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_5[] : memref<i32>
    memref.store %c0_i32, %alloca_5[] : memref<i32>
    %intptr_6 = memref.extract_aligned_pointer_as_index %alloca : memref<8x8xi32> -> index
    %4 = arith.index_castui %intptr_6 : index to i32
    %intptr_7 = memref.extract_aligned_pointer_as_index %alloca_5 : memref<i32> -> index
    %5 = arith.index_castui %intptr_7 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 3, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %4, %5 : (i32, i32) -> ()
    %6 = memref.load %alloca_5[] : memref<i32>
    call @pcl_validate_outputs(%alloca_1, %alloca_0, %alloca) : (memref<8x8xf32>, memref<8x8xf32>, memref<8x8xi32>) -> ()
    return %6 : i32
  }
  func.func private @pcl_validate_outputs(memref<8x8xf32>, memref<8x8xf32>, memref<8x8xi32>)
}

