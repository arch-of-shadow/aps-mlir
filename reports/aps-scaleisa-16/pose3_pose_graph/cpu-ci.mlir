module attributes {scaleisa.ci_instructions = 5 : i64, scaleisa.ci_sites = 5 : i64} {
  memref.global @bench_input_0 : memref<18x6xf32> = uninitialized {alignment = 64 : i64}
  memref.global @bench_input_1 : memref<30xi32> = uninitialized
  memref.global @bench_input_2 : memref<30xi32> = uninitialized
  memref.global @bench_input_3 : memref<30x6xf32> = uninitialized {alignment = 64 : i64}
  memref.global @bench_input_4 : memref<30x6xf32> = uninitialized {alignment = 64 : i64}
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c30 = arith.constant 30 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<30x6xf32>
    %alloca_0 = memref.alloca() {alignment = 64 : i64} : memref<30x6xf32>
    %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<30xi32>
    %alloca_2 = memref.alloca() {alignment = 64 : i64} : memref<30xi32>
    %alloca_3 = memref.alloca() {alignment = 64 : i64} : memref<18x6xf32>
    %0 = memref.get_global @bench_input_0 : memref<18x6xf32>
    %intptr = memref.extract_aligned_pointer_as_index %0 : memref<18x6xf32> -> index
    %1 = arith.index_castui %intptr : index to i32
    %intptr_4 = memref.extract_aligned_pointer_as_index %alloca_3 : memref<18x6xf32> -> index
    %2 = arith.index_castui %intptr_4 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 2, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %1, %2 : (i32, i32) -> ()
    %3 = memref.get_global @bench_input_1 : memref<30xi32>
    scf.for %arg0 = %c0 to %c30 step %c1 {
      %26 = memref.load %3[%arg0] : memref<30xi32>
      memref.store %26, %alloca_2[%arg0] : memref<30xi32>
    }
    %4 = memref.get_global @bench_input_2 : memref<30xi32>
    scf.for %arg0 = %c0 to %c30 step %c1 {
      %26 = memref.load %4[%arg0] : memref<30xi32>
      memref.store %26, %alloca_1[%arg0] : memref<30xi32>
    }
    %5 = memref.get_global @bench_input_3 : memref<30x6xf32>
    %intptr_5 = memref.extract_aligned_pointer_as_index %5 : memref<30x6xf32> -> index
    %6 = arith.index_castui %intptr_5 : index to i32
    %intptr_6 = memref.extract_aligned_pointer_as_index %alloca_0 : memref<30x6xf32> -> index
    %7 = arith.index_castui %intptr_6 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %6, %7 : (i32, i32) -> ()
    %8 = memref.get_global @bench_input_4 : memref<30x6xf32>
    %intptr_7 = memref.extract_aligned_pointer_as_index %8 : memref<30x6xf32> -> index
    %9 = arith.index_castui %intptr_7 : index to i32
    %intptr_8 = memref.extract_aligned_pointer_as_index %alloca : memref<30x6xf32> -> index
    %10 = arith.index_castui %intptr_8 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %9, %10 : (i32, i32) -> ()
    %alloca_9 = memref.alloca() {alignment = 64 : i64} : memref<18x6xf32>
    %alloca_10 = memref.alloca() {alignment = 64 : i64} : memref<18x6xf32>
    %intptr_11 = memref.extract_aligned_pointer_as_index %alloca_10 : memref<18x6xf32> -> index
    %11 = arith.index_castui %intptr_11 : index to i32
    %intptr_12 = memref.extract_aligned_pointer_as_index %alloca_9 : memref<18x6xf32> -> index
    %12 = arith.index_castui %intptr_12 : index to i32
    %intptr_13 = memref.extract_aligned_pointer_as_index %alloca_2 : memref<30xi32> -> index
    %13 = arith.index_castui %intptr_13 : index to i32
    %intptr_14 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<30xi32> -> index
    %14 = arith.index_castui %intptr_14 : index to i32
    %intptr_15 = memref.extract_aligned_pointer_as_index %alloca_3 : memref<18x6xf32> -> index
    %15 = arith.index_castui %intptr_15 : index to i32
    %intptr_16 = memref.extract_aligned_pointer_as_index %alloca_0 : memref<30x6xf32> -> index
    %16 = arith.index_castui %intptr_16 : index to i32
    %intptr_17 = memref.extract_aligned_pointer_as_index %alloca : memref<30x6xf32> -> index
    %17 = arith.index_castui %intptr_17 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0Acsrrw x0, 2050, $3\0Acsrrw x0, 2051, $4\0Acsrrw x0, 2052, $5\0Acsrrw x0, 2053, $6\0A.insn r 43, 3, 3, x0, $0, $1\0Afence rw, rw", "r,r,r,r,r,r,r,~{memory}" %11, %12, %13, %14, %15, %16, %17 : (i32, i32, i32, i32, i32, i32, i32) -> ()
    %alloca_18 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_18[] : memref<i32>
    memref.store %c0_i32, %alloca_18[] : memref<i32>
    %intptr_19 = memref.extract_aligned_pointer_as_index %alloca_3 : memref<18x6xf32> -> index
    %18 = arith.index_castui %intptr_19 : index to i32
    %intptr_20 = memref.extract_aligned_pointer_as_index %alloca_18 : memref<i32> -> index
    %19 = arith.index_castui %intptr_20 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %18, %19 : (i32, i32) -> ()
    %20 = memref.load %alloca_18[] : memref<i32>
    %21 = arith.sitofp %20 : i32 to f32
    %22 = arith.cmpi slt, %20, %c0_i32 : i32
    %23 = scf.if %22 -> (f32) {
      %26 = arith.negf %21 : f32
      scf.yield %26 : f32
    } else {
      scf.yield %21 : f32
    }
    %24 = arith.cmpf ogt, %23, %cst : f32
    %25 = arith.select %24, %20, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_3, %alloca_2, %alloca_1, %alloca_0, %alloca) : (memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<30x6xf32>, memref<30x6xf32>) -> ()
    return %25 : i32
  }
  func.func private @pcl_validate_outputs(memref<18x6xf32>, memref<30xi32>, memref<30xi32>, memref<30x6xf32>, memref<30x6xf32>)
}

