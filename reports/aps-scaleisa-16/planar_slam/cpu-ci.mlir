module attributes {scaleisa.ci_instructions = 6 : i64, scaleisa.ci_sites = 6 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c7 = arith.constant 7 : index
    %c64 = arith.constant 64 : index
    %c20 = arith.constant 20 : index
    %c16 = arith.constant 16 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e-03 : f32
    %cst_1 = arith.constant -0.899999976 : f32
    %cst_2 = arith.constant 2.000000e-01 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %cst_4 = arith.constant 4.000000e-01 : f32
    %cst_5 = arith.constant 1.000000e-01 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c16_i32 = arith.constant 16 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_6 = arith.constant 1.500000e-02 : f32
    %c20_i32 = arith.constant 20 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst_7 = arith.constant 2.000000e-02 : f32
    %c4_i32 = arith.constant 4 : i32
    %cst_8 = arith.constant 1.200000e-01 : f32
    %cst_9 = arith.constant 7.000000e-02 : f32
    %cst_10 = arith.constant 0.00999999977 : f32
    %cst_11 = arith.constant 9.000000e-02 : f32
    %cst_12 = arith.constant 6.000000e-02 : f32
    %c-1_i32 = arith.constant -1 : i32
    %alloca = memref.alloca() : memref<64x2xf32>
    %alloca_13 = memref.alloca() : memref<64xi32>
    %alloca_14 = memref.alloca() : memref<64xi32>
    %alloca_15 = memref.alloca() {alignment = 64 : i64} : memref<20x2xf32>
    %alloca_16 = memref.alloca() {alignment = 64 : i64} : memref<16x3xf32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_16 : memref<16x3xf32> -> index
    %0 = arith.index_castui %intptr : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 2, 2, x0, $0, x0\0Afence rw, rw", "r,~{memory}" %0 : (i32) -> ()
    %intptr_17 = memref.extract_aligned_pointer_as_index %alloca_15 : memref<20x2xf32> -> index
    %1 = arith.index_castui %intptr_17 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 2, 5, x0, $0, x0\0Afence rw, rw", "r,~{memory}" %1 : (i32) -> ()
    scf.for %arg0 = %c0 to %c64 step %c1 {
      %12 = arith.index_cast %arg0 : index to i32
      %13 = arith.remsi %12, %c16_i32 : i32
      %14 = arith.muli %12, %c3_i32 : i32
      %15 = arith.divsi %12, %c7_i32 : i32
      %16 = arith.addi %14, %15 : i32
      %17 = arith.remsi %16, %c20_i32 : i32
      %18 = arith.index_cast %17 : i32 to index
      %19 = memref.load %alloca_15[%18, %c0] : memref<20x2xf32>
      %20 = arith.index_cast %13 : i32 to index
      %21 = memref.load %alloca_16[%20, %c0] : memref<16x3xf32>
      %22 = arith.subf %19, %21 : f32
      %23 = memref.load %alloca_15[%18, %c1] : memref<20x2xf32>
      %24 = memref.load %alloca_16[%20, %c1] : memref<16x3xf32>
      %25 = arith.subf %23, %24 : f32
      %26 = memref.load %alloca_16[%20, %c2] : memref<16x3xf32>
      memref.store %13, %alloca_14[%arg0] : memref<64xi32>
      memref.store %17, %alloca_13[%arg0] : memref<64xi32>
      %27 = arith.mulf %26, %25 : f32
      %28 = arith.addf %22, %27 : f32
      %29 = arith.remsi %12, %c3_i32 : i32
      %30 = arith.sitofp %29 : i32 to f32
      %31 = arith.mulf %30, %cst_7 : f32
      %32 = arith.addf %28, %31 : f32
      memref.store %32, %alloca[%arg0, %c0] : memref<64x2xf32>
      %33 = arith.mulf %26, %22 : f32
      %34 = arith.subf %25, %33 : f32
      %35 = arith.remsi %12, %c4_i32 : i32
      %36 = arith.sitofp %35 : i32 to f32
      %37 = arith.mulf %36, %cst_6 : f32
      %38 = arith.subf %34, %37 : f32
      memref.store %38, %alloca[%arg0, %c1] : memref<64x2xf32>
    }
    scf.for %arg0 = %c0 to %c16 step %c1 {
      %12 = arith.index_cast %arg0 : index to i32
      %13 = arith.remsi %12, %c4_i32 : i32
      %14 = arith.addi %13, %c-1_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.mulf %15, %cst_8 : f32
      %17 = memref.load %alloca_16[%arg0, %c0] : memref<16x3xf32>
      %18 = arith.addf %17, %16 : f32
      memref.store %18, %alloca_16[%arg0, %c0] : memref<16x3xf32>
      %19 = arith.remsi %12, %c3_i32 : i32
      %20 = arith.sitofp %19 : i32 to f32
      %21 = arith.mulf %20, %cst_9 : f32
      %22 = memref.load %alloca_16[%arg0, %c1] : memref<16x3xf32>
      %23 = arith.subf %22, %21 : f32
      memref.store %23, %alloca_16[%arg0, %c1] : memref<16x3xf32>
      %24 = arith.remsi %12, %c2_i32 : i32
      %25 = arith.sitofp %24 : i32 to f32
      %26 = arith.mulf %25, %cst_10 : f32
      %27 = memref.load %alloca_16[%arg0, %c2] : memref<16x3xf32>
      %28 = arith.addf %27, %26 : f32
      memref.store %28, %alloca_16[%arg0, %c2] : memref<16x3xf32>
    }
    scf.for %arg0 = %c0 to %c20 step %c1 {
      %12 = arith.index_cast %arg0 : index to i32
      %13 = arith.remsi %12, %c3_i32 : i32
      %14 = arith.sitofp %13 : i32 to f32
      %15 = arith.mulf %14, %cst_11 : f32
      %16 = memref.load %alloca_15[%arg0, %c0] : memref<20x2xf32>
      %17 = arith.subf %16, %15 : f32
      memref.store %17, %alloca_15[%arg0, %c0] : memref<20x2xf32>
      %18 = arith.remsi %12, %c2_i32 : i32
      %19 = arith.sitofp %18 : i32 to f32
      %20 = arith.mulf %19, %cst_12 : f32
      %21 = memref.load %alloca_15[%arg0, %c1] : memref<20x2xf32>
      %22 = arith.addf %21, %20 : f32
      memref.store %22, %alloca_15[%arg0, %c1] : memref<20x2xf32>
    }
    %alloca_18 = memref.alloca() {alignment = 64 : i64} : memref<20x2xf32>
    %alloca_19 = memref.alloca() {alignment = 64 : i64} : memref<20x2xf32>
    %alloca_20 = memref.alloca() {alignment = 64 : i64} : memref<16x3xf32>
    %alloca_21 = memref.alloca() {alignment = 64 : i64} : memref<16x3xf32>
    scf.for %arg0 = %c0 to %c7 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        memref.store %cst, %alloca_21[%arg1, %c0] : memref<16x3xf32>
        memref.store %cst, %alloca_21[%arg1, %c1] : memref<16x3xf32>
        memref.store %cst, %alloca_21[%arg1, %c2] : memref<16x3xf32>
        memref.store %cst_0, %alloca_20[%arg1, %c0] : memref<16x3xf32>
        memref.store %cst_0, %alloca_20[%arg1, %c1] : memref<16x3xf32>
        memref.store %cst_0, %alloca_20[%arg1, %c2] : memref<16x3xf32>
      }
      scf.for %arg1 = %c0 to %c20 step %c1 {
        memref.store %cst, %alloca_19[%arg1, %c0] : memref<20x2xf32>
        memref.store %cst, %alloca_19[%arg1, %c1] : memref<20x2xf32>
        memref.store %cst_0, %alloca_18[%arg1, %c0] : memref<20x2xf32>
        memref.store %cst_0, %alloca_18[%arg1, %c1] : memref<20x2xf32>
      }
      scf.for %arg1 = %c0 to %c64 step %c1 {
        %18 = memref.load %alloca_14[%arg1] : memref<64xi32>
        %19 = memref.load %alloca_13[%arg1] : memref<64xi32>
        %20 = arith.index_cast %19 : i32 to index
        %21 = memref.load %alloca_15[%20, %c0] : memref<20x2xf32>
        %22 = arith.index_cast %18 : i32 to index
        %23 = memref.load %alloca_16[%22, %c0] : memref<16x3xf32>
        %24 = arith.subf %21, %23 : f32
        %25 = memref.load %alloca_15[%20, %c1] : memref<20x2xf32>
        %26 = memref.load %alloca_16[%22, %c1] : memref<16x3xf32>
        %27 = arith.subf %25, %26 : f32
        %28 = memref.load %alloca_16[%22, %c2] : memref<16x3xf32>
        %29 = arith.mulf %28, %27 : f32
        %30 = arith.addf %24, %29 : f32
        %31 = arith.mulf %28, %24 : f32
        %32 = arith.subf %27, %31 : f32
        %33 = memref.load %alloca[%arg1, %c0] : memref<64x2xf32>
        %34 = arith.subf %30, %33 : f32
        %35 = memref.load %alloca[%arg1, %c1] : memref<64x2xf32>
        %36 = arith.subf %32, %35 : f32
        %37 = arith.mulf %34, %cst_1 : f32
        %38 = arith.mulf %28, %cst_2 : f32
        %39 = arith.mulf %38, %36 : f32
        %40 = arith.addf %37, %39 : f32
        %41 = memref.load %alloca_21[%22, %c0] : memref<16x3xf32>
        %42 = arith.addf %41, %40 : f32
        memref.store %42, %alloca_21[%22, %c0] : memref<16x3xf32>
        %43 = arith.mulf %36, %cst_1 : f32
        %44 = arith.mulf %38, %34 : f32
        %45 = arith.subf %43, %44 : f32
        %46 = memref.load %alloca_21[%22, %c1] : memref<16x3xf32>
        %47 = arith.addf %46, %45 : f32
        memref.store %47, %alloca_21[%22, %c1] : memref<16x3xf32>
        %48 = arith.mulf %27, %34 : f32
        %49 = arith.mulf %24, %36 : f32
        %50 = arith.subf %48, %49 : f32
        %51 = memref.load %alloca_21[%22, %c2] : memref<16x3xf32>
        %52 = arith.addf %51, %50 : f32
        memref.store %52, %alloca_21[%22, %c2] : memref<16x3xf32>
        %53 = memref.load %alloca_20[%22, %c0] : memref<16x3xf32>
        %54 = arith.addf %53, %cst_3 : f32
        memref.store %54, %alloca_20[%22, %c0] : memref<16x3xf32>
        %55 = memref.load %alloca_20[%22, %c1] : memref<16x3xf32>
        %56 = arith.addf %55, %cst_3 : f32
        memref.store %56, %alloca_20[%22, %c1] : memref<16x3xf32>
        %57 = arith.mulf %24, %24 : f32
        %58 = arith.mulf %27, %27 : f32
        %59 = arith.addf %57, %58 : f32
        %60 = arith.mulf %59, %cst_4 : f32
        %61 = arith.addf %60, %cst_3 : f32
        %62 = memref.load %alloca_20[%22, %c2] : memref<16x3xf32>
        %63 = arith.addf %62, %61 : f32
        memref.store %63, %alloca_20[%22, %c2] : memref<16x3xf32>
        %64 = arith.mulf %28, %cst_5 : f32
        %65 = arith.mulf %64, %36 : f32
        %66 = arith.subf %34, %65 : f32
        %67 = memref.load %alloca_19[%20, %c0] : memref<20x2xf32>
        %68 = arith.addf %67, %66 : f32
        memref.store %68, %alloca_19[%20, %c0] : memref<20x2xf32>
        %69 = arith.mulf %64, %34 : f32
        %70 = arith.addf %36, %69 : f32
        %71 = memref.load %alloca_19[%20, %c1] : memref<20x2xf32>
        %72 = arith.addf %71, %70 : f32
        memref.store %72, %alloca_19[%20, %c1] : memref<20x2xf32>
        %73 = memref.load %alloca_18[%20, %c0] : memref<20x2xf32>
        %74 = arith.addf %73, %cst_3 : f32
        memref.store %74, %alloca_18[%20, %c0] : memref<20x2xf32>
        %75 = memref.load %alloca_18[%20, %c1] : memref<20x2xf32>
        %76 = arith.addf %75, %cst_3 : f32
        memref.store %76, %alloca_18[%20, %c1] : memref<20x2xf32>
      }
      memref.store %cst, %alloca_16[%c0, %c0] : memref<16x3xf32>
      memref.store %cst, %alloca_16[%c0, %c1] : memref<16x3xf32>
      memref.store %cst, %alloca_16[%c0, %c2] : memref<16x3xf32>
      %intptr_27 = memref.extract_aligned_pointer_as_index %alloca_21 : memref<16x3xf32> -> index
      %12 = arith.index_castui %intptr_27 : index to i32
      %intptr_28 = memref.extract_aligned_pointer_as_index %alloca_20 : memref<16x3xf32> -> index
      %13 = arith.index_castui %intptr_28 : index to i32
      %intptr_29 = memref.extract_aligned_pointer_as_index %alloca_16 : memref<16x3xf32> -> index
      %14 = arith.index_castui %intptr_29 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %12, %13, %14 : (i32, i32, i32) -> ()
      %intptr_30 = memref.extract_aligned_pointer_as_index %alloca_19 : memref<20x2xf32> -> index
      %15 = arith.index_castui %intptr_30 : index to i32
      %intptr_31 = memref.extract_aligned_pointer_as_index %alloca_18 : memref<20x2xf32> -> index
      %16 = arith.index_castui %intptr_31 : index to i32
      %intptr_32 = memref.extract_aligned_pointer_as_index %alloca_15 : memref<20x2xf32> -> index
      %17 = arith.index_castui %intptr_32 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2050, $2\0A.insn r 43, 3, 3, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %15, %16, %17 : (i32, i32, i32) -> ()
    }
    %alloca_22 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_22[] : memref<i32>
    memref.store %c0_i32, %alloca_22[] : memref<i32>
    %intptr_23 = memref.extract_aligned_pointer_as_index %alloca_16 : memref<16x3xf32> -> index
    %2 = arith.index_castui %intptr_23 : index to i32
    %intptr_24 = memref.extract_aligned_pointer_as_index %alloca_22 : memref<i32> -> index
    %3 = arith.index_castui %intptr_24 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %2, %3 : (i32, i32) -> ()
    %intptr_25 = memref.extract_aligned_pointer_as_index %alloca_15 : memref<20x2xf32> -> index
    %4 = arith.index_castui %intptr_25 : index to i32
    %intptr_26 = memref.extract_aligned_pointer_as_index %alloca_22 : memref<i32> -> index
    %5 = arith.index_castui %intptr_26 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 4, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %4, %5 : (i32, i32) -> ()
    %6 = memref.load %alloca_22[] : memref<i32>
    %7 = arith.sitofp %6 : i32 to f32
    %8 = arith.cmpi slt, %6, %c0_i32 : i32
    %9 = scf.if %8 -> (f32) {
      %12 = arith.negf %7 : f32
      scf.yield %12 : f32
    } else {
      scf.yield %7 : f32
    }
    %10 = arith.cmpf ogt, %9, %cst : f32
    %11 = arith.select %10, %6, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_16, %alloca_15, %alloca_14, %alloca_13, %alloca) : (memref<16x3xf32>, memref<20x2xf32>, memref<64xi32>, memref<64xi32>, memref<64x2xf32>) -> ()
    return %11 : i32
  }
  func.func private @pcl_validate_outputs(memref<16x3xf32>, memref<20x2xf32>, memref<64xi32>, memref<64xi32>, memref<64x2xf32>)
}

