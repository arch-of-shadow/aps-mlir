module attributes {scaleisa.ci_instructions = 6 : i64, scaleisa.ci_sites = 6 : i64} {
  memref.global "private" @_ZL10DEPTH_BIAS : memref<1xf32> = dense<6.000000e+00>
  memref.global "private" @_ZL2FY : memref<1xf32> = dense<2.160000e+02>
  memref.global "private" @_ZL2FX : memref<1xf32> = dense<2.200000e+02>
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c192 = arith.constant 192 : index
    %c3 = arith.constant 3 : index
    %c6 = arith.constant 6 : index
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 9.700000e+01 : f32
    %cst_0 = arith.constant 1.310000e+02 : f32
    %cst_1 = arith.constant 1.490000e+02 : f32
    %c2 = arith.constant 2 : index
    %cst_2 = arith.constant 1.000000e-03 : f32
    %cst_3 = arith.constant 1.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_5 = arith.constant 3.000000e-02 : f32
    %cst_6 = arith.constant 2.000000e-02 : f32
    %c7_i32 = arith.constant 7 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_7 = arith.constant 1.000000e-01 : f32
    %cst_8 = arith.constant 7.000000e-02 : f32
    %cst_9 = arith.constant 5.000000e-02 : f32
    %c-1_i32 = arith.constant -1 : i32
    %c-3_i32 = arith.constant -3 : i32
    %c-2_i32 = arith.constant -2 : i32
    %alloca = memref.alloca() : memref<192x2xf32>
    %alloca_10 = memref.alloca() : memref<192xi32>
    %alloca_11 = memref.alloca() : memref<192xi32>
    %alloca_12 = memref.alloca() {alignment = 64 : i64} : memref<24x3xf32>
    %alloca_13 = memref.alloca() {alignment = 64 : i64} : memref<8x3xf32>
    %alloca_14 = memref.alloca() : memref<1xf32>
    memref.store %cst_4, %alloca_14[%c0] : memref<1xf32>
    %alloca_15 = memref.alloca() : memref<1xf32>
    memref.store %cst_4, %alloca_15[%c0] : memref<1xf32>
    %alloca_16 = memref.alloca() : memref<1xf32>
    memref.store %cst_4, %alloca_16[%c0] : memref<1xf32>
    %alloca_17 = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca_17[] : memref<i32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_13 : memref<8x3xf32> -> index
    %0 = arith.index_castui %intptr : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 2, 2, x0, $0, x0\0Afence rw, rw", "r,~{memory}" %0 : (i32) -> ()
    %intptr_18 = memref.extract_aligned_pointer_as_index %alloca_12 : memref<24x3xf32> -> index
    %1 = arith.index_castui %intptr_18 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 2, 1, x0, $0, x0\0Afence rw, rw", "r,~{memory}" %1 : (i32) -> ()
    memref.store %c0_i32, %alloca_17[] : memref<i32>
    %2 = memref.get_global @_ZL10DEPTH_BIAS : memref<1xf32>
    %3 = memref.get_global @_ZL2FX : memref<1xf32>
    %4 = memref.get_global @_ZL2FY : memref<1xf32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %14 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c24 step %c1 {
        %15 = arith.index_cast %arg1 : index to i32
        %16 = memref.load %alloca_17[] : memref<i32>
        %17 = arith.index_cast %16 : i32 to index
        memref.store %14, %alloca_11[%17] : memref<192xi32>
        memref.store %15, %alloca_10[%17] : memref<192xi32>
        %18 = memref.load %alloca_12[%arg1, %c0] : memref<24x3xf32>
        %19 = memref.load %alloca_13[%arg0, %c0] : memref<8x3xf32>
        %20 = arith.subf %18, %19 : f32
        %21 = memref.load %alloca_12[%arg1, %c1] : memref<24x3xf32>
        %22 = memref.load %alloca_13[%arg0, %c1] : memref<8x3xf32>
        %23 = arith.subf %21, %22 : f32
        %24 = memref.load %alloca_12[%arg1, %c2] : memref<24x3xf32>
        %25 = memref.load %alloca_13[%arg0, %c2] : memref<8x3xf32>
        %26 = arith.subf %24, %25 : f32
        %27 = memref.load %2[%c0] : memref<1xf32>
        %28 = arith.addf %26, %27 : f32
        %29 = arith.divf %cst_3, %28 : f32
        memref.store %29, %alloca_14[%c0] : memref<1xf32>
        %30 = memref.load %3[%c0] : memref<1xf32>
        %31 = arith.mulf %30, %20 : f32
        %32 = memref.load %alloca_14[%c0] : memref<1xf32>
        %33 = arith.mulf %31, %32 : f32
        memref.store %33, %alloca_16[%c0] : memref<1xf32>
        %34 = memref.load %4[%c0] : memref<1xf32>
        %35 = arith.mulf %34, %23 : f32
        %36 = memref.load %alloca_14[%c0] : memref<1xf32>
        %37 = arith.mulf %35, %36 : f32
        memref.store %37, %alloca_15[%c0] : memref<1xf32>
        %38 = memref.load %alloca_16[%c0] : memref<1xf32>
        %39 = arith.remsi %16, %c5_i32 : i32
        %40 = arith.addi %39, %c-2_i32 : i32
        %41 = arith.sitofp %40 : i32 to f32
        %42 = arith.mulf %41, %cst_5 : f32
        %43 = arith.addf %38, %42 : f32
        memref.store %43, %alloca[%17, %c0] : memref<192x2xf32>
        %44 = memref.load %alloca_15[%c0] : memref<1xf32>
        %45 = arith.remsi %16, %c7_i32 : i32
        %46 = arith.addi %45, %c-3_i32 : i32
        %47 = arith.sitofp %46 : i32 to f32
        %48 = arith.mulf %47, %cst_6 : f32
        %49 = arith.subf %44, %48 : f32
        memref.store %49, %alloca[%17, %c1] : memref<192x2xf32>
        %50 = arith.addi %16, %c1_i32 : i32
        memref.store %50, %alloca_17[] : memref<i32>
      }
    }
    %intptr_19 = memref.extract_aligned_pointer_as_index %alloca_13 : memref<8x3xf32> -> index
    %5 = arith.index_castui %intptr_19 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 2, 5, x0, $0, x0\0Afence rw, rw", "r,~{memory}" %5 : (i32) -> ()
    scf.for %arg0 = %c0 to %c24 step %c1 {
      %14 = arith.index_cast %arg0 : index to i32
      %15 = arith.remsi %14, %c3_i32 : i32
      %16 = arith.sitofp %15 : i32 to f32
      %17 = arith.mulf %16, %cst_7 : f32
      %18 = memref.load %alloca_12[%arg0, %c0] : memref<24x3xf32>
      %19 = arith.subf %18, %17 : f32
      memref.store %19, %alloca_12[%arg0, %c0] : memref<24x3xf32>
      %20 = arith.remsi %14, %c2_i32 : i32
      %21 = arith.sitofp %20 : i32 to f32
      %22 = arith.mulf %21, %cst_8 : f32
      %23 = memref.load %alloca_12[%arg0, %c1] : memref<24x3xf32>
      %24 = arith.addf %23, %22 : f32
      memref.store %24, %alloca_12[%arg0, %c1] : memref<24x3xf32>
      %25 = arith.remsi %14, %c4_i32 : i32
      %26 = arith.addi %25, %c-1_i32 : i32
      %27 = arith.sitofp %26 : i32 to f32
      %28 = arith.mulf %27, %cst_9 : f32
      %29 = memref.load %alloca_12[%arg0, %c2] : memref<24x3xf32>
      %30 = arith.addf %29, %28 : f32
      memref.store %30, %alloca_12[%arg0, %c2] : memref<24x3xf32>
    }
    %alloca_20 = memref.alloca() : memref<1xf32>
    memref.store %cst_4, %alloca_20[%c0] : memref<1xf32>
    %alloca_21 = memref.alloca() : memref<1xf32>
    memref.store %cst_4, %alloca_21[%c0] : memref<1xf32>
    %alloca_22 = memref.alloca() : memref<1xf32>
    memref.store %cst_4, %alloca_22[%c0] : memref<1xf32>
    %alloca_23 = memref.alloca() {alignment = 64 : i64} : memref<24x3xf32>
    %alloca_24 = memref.alloca() {alignment = 64 : i64} : memref<24x3xf32>
    %alloca_25 = memref.alloca() {alignment = 64 : i64} : memref<8x3xf32>
    %alloca_26 = memref.alloca() {alignment = 64 : i64} : memref<8x3xf32>
    scf.for %arg0 = %c0 to %c6 step %c1 {
      scf.for %arg1 = %c0 to %c8 step %c1 {
        scf.for %arg2 = %c0 to %c3 step %c1 {
          memref.store %cst_4, %alloca_26[%arg1, %arg2] : memref<8x3xf32>
          memref.store %cst_2, %alloca_25[%arg1, %arg2] : memref<8x3xf32>
        }
      }
      scf.for %arg1 = %c0 to %c3 step %c1 {
        scf.for %arg2 = %c0 to %c24 step %c1 {
          memref.store %cst_4, %alloca_24[%arg2, %arg1] : memref<24x3xf32>
          memref.store %cst_2, %alloca_23[%arg2, %arg1] : memref<24x3xf32>
        }
      }
      scf.for %arg1 = %c0 to %c192 step %c1 {
        %20 = memref.load %alloca_11[%arg1] : memref<192xi32>
        %21 = memref.load %alloca_10[%arg1] : memref<192xi32>
        %22 = arith.index_cast %20 : i32 to index
        %23 = arith.index_cast %21 : i32 to index
        %24 = memref.load %alloca_12[%23, %c0] : memref<24x3xf32>
        %25 = memref.load %alloca_13[%22, %c0] : memref<8x3xf32>
        %26 = arith.subf %24, %25 : f32
        %27 = memref.load %alloca_12[%23, %c1] : memref<24x3xf32>
        %28 = memref.load %alloca_13[%22, %c1] : memref<8x3xf32>
        %29 = arith.subf %27, %28 : f32
        %30 = memref.load %alloca_12[%23, %c2] : memref<24x3xf32>
        %31 = memref.load %alloca_13[%22, %c2] : memref<8x3xf32>
        %32 = arith.subf %30, %31 : f32
        %33 = memref.load %2[%c0] : memref<1xf32>
        %34 = arith.addf %32, %33 : f32
        %35 = arith.divf %cst_3, %34 : f32
        memref.store %35, %alloca_20[%c0] : memref<1xf32>
        %36 = memref.load %3[%c0] : memref<1xf32>
        %37 = arith.mulf %36, %26 : f32
        %38 = memref.load %alloca_20[%c0] : memref<1xf32>
        %39 = arith.mulf %37, %38 : f32
        memref.store %39, %alloca_22[%c0] : memref<1xf32>
        %40 = memref.load %4[%c0] : memref<1xf32>
        %41 = arith.mulf %40, %29 : f32
        %42 = memref.load %alloca_20[%c0] : memref<1xf32>
        %43 = arith.mulf %41, %42 : f32
        memref.store %43, %alloca_21[%c0] : memref<1xf32>
        %44 = memref.load %alloca_12[%23, %c0] : memref<24x3xf32>
        %45 = memref.load %alloca_13[%22, %c0] : memref<8x3xf32>
        %46 = arith.subf %44, %45 : f32
        %47 = memref.load %alloca_12[%23, %c1] : memref<24x3xf32>
        %48 = memref.load %alloca_13[%22, %c1] : memref<8x3xf32>
        %49 = arith.subf %47, %48 : f32
        %50 = memref.load %alloca_22[%c0] : memref<1xf32>
        %51 = memref.load %alloca[%arg1, %c0] : memref<192x2xf32>
        %52 = arith.subf %50, %51 : f32
        %53 = memref.load %alloca_21[%c0] : memref<1xf32>
        %54 = memref.load %alloca[%arg1, %c1] : memref<192x2xf32>
        %55 = arith.subf %53, %54 : f32
        %56 = memref.load %3[%c0] : memref<1xf32>
        %57 = memref.load %alloca_20[%c0] : memref<1xf32>
        %58 = arith.mulf %56, %57 : f32
        %59 = memref.load %4[%c0] : memref<1xf32>
        %60 = arith.mulf %59, %57 : f32
        %61 = arith.negf %56 : f32
        %62 = arith.mulf %61, %46 : f32
        %63 = arith.mulf %62, %57 : f32
        %64 = arith.mulf %63, %57 : f32
        %65 = arith.negf %59 : f32
        %66 = arith.mulf %65, %49 : f32
        %67 = arith.mulf %66, %57 : f32
        %68 = arith.mulf %67, %57 : f32
        %69 = arith.negf %52 : f32
        %70 = arith.mulf %69, %58 : f32
        %71 = memref.load %alloca_26[%22, %c0] : memref<8x3xf32>
        %72 = arith.addf %71, %70 : f32
        memref.store %72, %alloca_26[%22, %c0] : memref<8x3xf32>
        %73 = arith.negf %55 : f32
        %74 = arith.mulf %73, %60 : f32
        %75 = memref.load %alloca_26[%22, %c1] : memref<8x3xf32>
        %76 = arith.addf %75, %74 : f32
        memref.store %76, %alloca_26[%22, %c1] : memref<8x3xf32>
        %77 = arith.mulf %52, %64 : f32
        %78 = arith.mulf %55, %68 : f32
        %79 = arith.addf %77, %78 : f32
        %80 = arith.negf %79 : f32
        %81 = memref.load %alloca_26[%22, %c2] : memref<8x3xf32>
        %82 = arith.addf %81, %80 : f32
        memref.store %82, %alloca_26[%22, %c2] : memref<8x3xf32>
        %83 = arith.mulf %58, %58 : f32
        %84 = memref.load %alloca_25[%22, %c0] : memref<8x3xf32>
        %85 = arith.addf %84, %83 : f32
        memref.store %85, %alloca_25[%22, %c0] : memref<8x3xf32>
        %86 = arith.mulf %60, %60 : f32
        %87 = memref.load %alloca_25[%22, %c1] : memref<8x3xf32>
        %88 = arith.addf %87, %86 : f32
        memref.store %88, %alloca_25[%22, %c1] : memref<8x3xf32>
        %89 = arith.mulf %64, %64 : f32
        %90 = arith.mulf %68, %68 : f32
        %91 = arith.addf %89, %90 : f32
        %92 = memref.load %alloca_25[%22, %c2] : memref<8x3xf32>
        %93 = arith.addf %92, %91 : f32
        memref.store %93, %alloca_25[%22, %c2] : memref<8x3xf32>
        %94 = arith.mulf %52, %58 : f32
        %95 = memref.load %alloca_24[%23, %c0] : memref<24x3xf32>
        %96 = arith.addf %95, %94 : f32
        memref.store %96, %alloca_24[%23, %c0] : memref<24x3xf32>
        %97 = arith.mulf %55, %60 : f32
        %98 = memref.load %alloca_24[%23, %c1] : memref<24x3xf32>
        %99 = arith.addf %98, %97 : f32
        memref.store %99, %alloca_24[%23, %c1] : memref<24x3xf32>
        %100 = memref.load %alloca_24[%23, %c2] : memref<24x3xf32>
        %101 = arith.addf %100, %79 : f32
        memref.store %101, %alloca_24[%23, %c2] : memref<24x3xf32>
        %102 = memref.load %alloca_23[%23, %c0] : memref<24x3xf32>
        %103 = arith.addf %102, %83 : f32
        memref.store %103, %alloca_23[%23, %c0] : memref<24x3xf32>
        %104 = memref.load %alloca_23[%23, %c1] : memref<24x3xf32>
        %105 = arith.addf %104, %86 : f32
        memref.store %105, %alloca_23[%23, %c1] : memref<24x3xf32>
        %106 = memref.load %alloca_23[%23, %c2] : memref<24x3xf32>
        %107 = arith.addf %106, %91 : f32
        memref.store %107, %alloca_23[%23, %c2] : memref<24x3xf32>
      }
      %intptr_30 = memref.extract_aligned_pointer_as_index %alloca_26 : memref<8x3xf32> -> index
      %14 = arith.index_castui %intptr_30 : index to i32
      %intptr_31 = memref.extract_aligned_pointer_as_index %alloca_25 : memref<8x3xf32> -> index
      %15 = arith.index_castui %intptr_31 : index to i32
      %intptr_32 = memref.extract_aligned_pointer_as_index %alloca_13 : memref<8x3xf32> -> index
      %16 = arith.index_castui %intptr_32 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2050, $2\0A.insn r 43, 3, 4, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %14, %15, %16 : (i32, i32, i32) -> ()
      %intptr_33 = memref.extract_aligned_pointer_as_index %alloca_24 : memref<24x3xf32> -> index
      %17 = arith.index_castui %intptr_33 : index to i32
      %intptr_34 = memref.extract_aligned_pointer_as_index %alloca_23 : memref<24x3xf32> -> index
      %18 = arith.index_castui %intptr_34 : index to i32
      %intptr_35 = memref.extract_aligned_pointer_as_index %alloca_12 : memref<24x3xf32> -> index
      %19 = arith.index_castui %intptr_35 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0A.insn r 43, 3, 3, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %17, %18, %19 : (i32, i32, i32) -> ()
    }
    %alloca_27 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_27[] : memref<i32>
    memref.store %c0_i32, %alloca_27[] : memref<i32>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      %14 = memref.load %alloca_13[%arg0, %c0] : memref<8x3xf32>
      %15 = arith.mulf %14, %cst : f32
      %16 = arith.fptosi %15 : f32 to i32
      %17 = memref.load %alloca_27[] : memref<i32>
      %18 = arith.addi %17, %16 : i32
      memref.store %18, %alloca_27[] : memref<i32>
      %19 = memref.load %alloca_13[%arg0, %c1] : memref<8x3xf32>
      %20 = arith.mulf %19, %cst_0 : f32
      %21 = arith.fptosi %20 : f32 to i32
      %22 = arith.addi %18, %21 : i32
      memref.store %22, %alloca_27[] : memref<i32>
      %23 = memref.load %alloca_13[%arg0, %c2] : memref<8x3xf32>
      %24 = arith.mulf %23, %cst_1 : f32
      %25 = arith.fptosi %24 : f32 to i32
      %26 = arith.addi %22, %25 : i32
      memref.store %26, %alloca_27[] : memref<i32>
    }
    %intptr_28 = memref.extract_aligned_pointer_as_index %alloca_12 : memref<24x3xf32> -> index
    %6 = arith.index_castui %intptr_28 : index to i32
    %intptr_29 = memref.extract_aligned_pointer_as_index %alloca_27 : memref<i32> -> index
    %7 = arith.index_castui %intptr_29 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,~{memory}" %6, %7 : (i32, i32) -> ()
    %8 = memref.load %alloca_27[] : memref<i32>
    %9 = arith.sitofp %8 : i32 to f32
    %10 = arith.cmpi slt, %8, %c0_i32 : i32
    %11 = scf.if %10 -> (f32) {
      %14 = arith.negf %9 : f32
      scf.yield %14 : f32
    } else {
      scf.yield %9 : f32
    }
    %12 = arith.cmpf ogt, %11, %cst_4 : f32
    %13 = arith.select %12, %8, %c1_i32 : i32
    call @pcl_validate_outputs(%alloca_13, %alloca_12, %alloca_11, %alloca_10, %alloca) : (memref<8x3xf32>, memref<24x3xf32>, memref<192xi32>, memref<192xi32>, memref<192x2xf32>) -> ()
    return %13 : i32
  }
  func.func private @pcl_validate_outputs(memref<8x3xf32>, memref<24x3xf32>, memref<192xi32>, memref<192xi32>, memref<192x2xf32>)
}

