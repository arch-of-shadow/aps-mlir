module attributes {scaleisa.ci_instructions = 6 : i64, scaleisa.ci_sites = 6 : i64} {
  memref.global @bench_input_0 : memref<40xi32> = uninitialized
  memref.global @bench_input_1 : memref<40xi32> = uninitialized
  memref.global @bench_input_2 : memref<40x3xf32> = uninitialized {alignment = 64 : i64}
  memref.global @bench_input_3 : memref<40x3xf32> = uninitialized {alignment = 64 : i64}
  memref.global @bench_input_4 : memref<24x3xf32> = uninitialized {alignment = 64 : i64}
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c24 = arith.constant 24 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c40 = arith.constant 40 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-03 : f32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<24x3xf32>
    %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<40x3xf32>
    %alloca_2 = memref.alloca() {alignment = 64 : i64} : memref<40x3xf32>
    %alloca_3 = memref.alloca() {alignment = 64 : i64} : memref<40xi32>
    %alloca_4 = memref.alloca() {alignment = 64 : i64} : memref<40xi32>
    %0 = memref.get_global @bench_input_0 : memref<40xi32>
    scf.for %arg0 = %c0 to %c40 step %c1 {
      %19 = memref.load %0[%arg0] : memref<40xi32>
      memref.store %19, %alloca_4[%arg0] : memref<40xi32>
    }
    %1 = memref.get_global @bench_input_1 : memref<40xi32>
    scf.for %arg0 = %c0 to %c40 step %c1 {
      %19 = memref.load %1[%arg0] : memref<40xi32>
      memref.store %19, %alloca_3[%arg0] : memref<40xi32>
    }
    %2 = memref.get_global @bench_input_2 : memref<40x3xf32>
    %intptr = memref.extract_aligned_pointer_as_index %2 : memref<40x3xf32> -> index
    %3 = arith.index_castui %intptr : index to i32
    %intptr_5 = memref.extract_aligned_pointer_as_index %alloca_2 : memref<40x3xf32> -> index
    %4 = arith.index_castui %intptr_5 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 2, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %3, %4 : (i32, i32) -> ()
    %5 = memref.get_global @bench_input_3 : memref<40x3xf32>
    %intptr_6 = memref.extract_aligned_pointer_as_index %5 : memref<40x3xf32> -> index
    %6 = arith.index_castui %intptr_6 : index to i32
    %intptr_7 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<40x3xf32> -> index
    %7 = arith.index_castui %intptr_7 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 2, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %6, %7 : (i32, i32) -> ()
    %8 = memref.get_global @bench_input_4 : memref<24x3xf32>
    %intptr_8 = memref.extract_aligned_pointer_as_index %8 : memref<24x3xf32> -> index
    %9 = arith.index_castui %intptr_8 : index to i32
    %intptr_9 = memref.extract_aligned_pointer_as_index %alloca : memref<24x3xf32> -> index
    %10 = arith.index_castui %intptr_9 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 3, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %9, %10 : (i32, i32) -> ()
    %alloca_10 = memref.alloca() {alignment = 64 : i64} : memref<24x3xf32>
    %alloca_11 = memref.alloca() {alignment = 64 : i64} : memref<24x3xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c3 step %c1 {
        scf.for %arg2 = %c0 to %c24 step %c1 {
          memref.store %cst, %alloca_11[%arg2, %arg1] : memref<24x3xf32>
          memref.store %cst_0, %alloca_10[%arg2, %arg1] : memref<24x3xf32>
        }
      }
      %intptr_15 = memref.extract_aligned_pointer_as_index %alloca_4 : memref<40xi32> -> index
      %19 = arith.index_castui %intptr_15 : index to i32
      %intptr_16 = memref.extract_aligned_pointer_as_index %alloca_3 : memref<40xi32> -> index
      %20 = arith.index_castui %intptr_16 : index to i32
      %intptr_17 = memref.extract_aligned_pointer_as_index %alloca : memref<24x3xf32> -> index
      %21 = arith.index_castui %intptr_17 : index to i32
      %intptr_18 = memref.extract_aligned_pointer_as_index %alloca_2 : memref<40x3xf32> -> index
      %22 = arith.index_castui %intptr_18 : index to i32
      %intptr_19 = memref.extract_aligned_pointer_as_index %alloca_1 : memref<40x3xf32> -> index
      %23 = arith.index_castui %intptr_19 : index to i32
      %intptr_20 = memref.extract_aligned_pointer_as_index %alloca_11 : memref<24x3xf32> -> index
      %24 = arith.index_castui %intptr_20 : index to i32
      %intptr_21 = memref.extract_aligned_pointer_as_index %alloca_10 : memref<24x3xf32> -> index
      %25 = arith.index_castui %intptr_21 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2050, $2\0Acsrrw x0, 2051, $3\0Acsrrw x0, 2052, $4\0Acsrrw x0, 2053, $5\0Acsrrw x0, 2054, $6\0A.insn r 43, 3, 4, x0, $0, $1\0Afence rw, rw", "r,r,r,r,r,r,r,~{memory}" %19, %20, %21, %22, %23, %24, %25 : (i32, i32, i32, i32, i32, i32, i32) -> ()
      memref.store %cst, %alloca[%c0, %c0] : memref<24x3xf32>
      memref.store %cst, %alloca[%c0, %c1] : memref<24x3xf32>
      memref.store %cst, %alloca[%c0, %c2] : memref<24x3xf32>
      %intptr_22 = memref.extract_aligned_pointer_as_index %alloca_11 : memref<24x3xf32> -> index
      %26 = arith.index_castui %intptr_22 : index to i32
      %intptr_23 = memref.extract_aligned_pointer_as_index %alloca_10 : memref<24x3xf32> -> index
      %27 = arith.index_castui %intptr_23 : index to i32
      %intptr_24 = memref.extract_aligned_pointer_as_index %alloca : memref<24x3xf32> -> index
      %28 = arith.index_castui %intptr_24 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %26, %27, %28 : (i32, i32, i32) -> ()
    }
    %alloca_12 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_12[] : memref<i32>
    memref.store %c0_i32, %alloca_12[] : memref<i32>
    %intptr_13 = memref.extract_aligned_pointer_as_index %alloca : memref<24x3xf32> -> index
    %11 = arith.index_castui %intptr_13 : index to i32
    %intptr_14 = memref.extract_aligned_pointer_as_index %alloca_12 : memref<i32> -> index
    %12 = arith.index_castui %intptr_14 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %11, %12 : (i32, i32) -> ()
    %13 = memref.load %alloca_12[] : memref<i32>
    %14 = arith.sitofp %13 : i32 to f32
    %15 = arith.cmpi slt, %13, %c0_i32 : i32
    %16 = scf.if %15 -> (f32) {
      %19 = arith.negf %14 : f32
      scf.yield %19 : f32
    } else {
      scf.yield %14 : f32
    }
    %17 = arith.cmpf ogt, %16, %cst : f32
    %18 = arith.select %17, %13, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_4, %alloca_3, %alloca_2, %alloca_1, %alloca) : (memref<40xi32>, memref<40xi32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>) -> ()
    return %18 : i32
  }
  func.func private @pcl_validate_outputs(memref<40xi32>, memref<40xi32>, memref<40x3xf32>, memref<40x3xf32>, memref<24x3xf32>)
}

