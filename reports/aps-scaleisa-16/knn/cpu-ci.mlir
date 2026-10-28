module attributes {scaleisa.ci_instructions = 4 : i64, scaleisa.ci_sites = 4 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<4x8xi32>
    %alloca_0 = memref.alloca() {alignment = 64 : i64} : memref<4x8xf32>
    %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<4x3xf32>
    %alloca_2 = memref.alloca() {alignment = 64 : i64} : memref<32x3xf32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_2 : memref<32x3xf32> -> index
    %0 = arith.index_castui %intptr : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 2, 2, x0, $0, x0\0Afence rw, rw", "r,~{memory}" %0 : (i32) -> ()
    %intptr_3 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<4x3xf32> -> index
    %1 = arith.index_castui %intptr_3 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 2, 3, x0, $0, x0\0Afence rw, rw", "r,~{memory}" %1 : (i32) -> ()
    %alloca_4 = memref.alloca() {alignment = 64 : i64} : memref<8xi32>
    %alloca_5 = memref.alloca() {alignment = 64 : i64} : memref<8xf32>
    %intptr_6 = memref.extract_aligned_pointer_as_index %alloca_0 : memref<4x8xf32> -> index
    %2 = arith.index_castui %intptr_6 : index to i32
    %intptr_7 = memref.extract_aligned_pointer_as_index %alloca : memref<4x8xi32> -> index
    %3 = arith.index_castui %intptr_7 : index to i32
    %intptr_8 = memref.extract_aligned_pointer_as_index %alloca_2 : memref<32x3xf32> -> index
    %4 = arith.index_castui %intptr_8 : index to i32
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<4x3xf32> -> index
    %5 = arith.index_castui %intptr_9 : index to i32
    %intptr_10 = memref.extract_aligned_pointer_as_index %alloca_5 : memref<8xf32> -> index
    %6 = arith.index_castui %intptr_10 : index to i32
    %intptr_11 = memref.extract_aligned_pointer_as_index %alloca_4 : memref<8xi32> -> index
    %7 = arith.index_castui %intptr_11 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2050, $2\0Acsrrw x0, 2051, $3\0Acsrrw x0, 2052, $4\0Acsrrw x0, 2053, $5\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,r,r,r,r,~{memory}" %2, %3, %4, %5, %6, %7 : (i32, i32, i32, i32, i32, i32) -> ()
    %alloca_12 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_12[] : memref<i32>
    memref.store %c0_i32, %alloca_12[] : memref<i32>
    %intptr_13 = memref.extract_aligned_pointer_as_index %alloca : memref<4x8xi32> -> index
    %8 = arith.index_castui %intptr_13 : index to i32
    %intptr_14 = memref.extract_aligned_pointer_as_index %alloca_0 : memref<4x8xf32> -> index
    %9 = arith.index_castui %intptr_14 : index to i32
    %intptr_15 = memref.extract_aligned_pointer_as_index %alloca_12 : memref<i32> -> index
    %10 = arith.index_castui %intptr_15 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %8, %9, %10 : (i32, i32, i32) -> ()
    %11 = memref.load %alloca_12[] : memref<i32>
    %12 = arith.sitofp %11 : i32 to f32
    %13 = arith.cmpi slt, %11, %c0_i32 : i32
    %14 = scf.if %13 -> (f32) {
      %17 = arith.negf %12 : f32
      scf.yield %17 : f32
    } else {
      scf.yield %12 : f32
    }
    %15 = arith.cmpf ogt, %14, %cst : f32
    %16 = arith.select %15, %11, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_2, %alloca_1, %alloca_0, %alloca) : (memref<32x3xf32>, memref<4x3xf32>, memref<4x8xf32>, memref<4x8xi32>) -> ()
    return %16 : i32
  }
  func.func private @pcl_validate_outputs(memref<32x3xf32>, memref<4x3xf32>, memref<4x8xf32>, memref<4x8xi32>)
}

