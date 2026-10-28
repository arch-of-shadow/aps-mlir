module attributes {scaleisa.ci_instructions = 2 : i64, scaleisa.ci_sites = 2 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c12 = arith.constant 12 : index
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e-03 : f32
    %cst_2 = arith.constant 2.000000e+00 : f32
    %c11_i32 = arith.constant 11 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c9_i32 = arith.constant 9 : i32
    %cst_3 = arith.constant 0.00999999977 : f32
    %cst_4 = arith.constant 7.000000e-02 : f32
    %cst_5 = arith.constant 1.500000e-01 : f32
    %cst_6 = arith.constant 4.000000e-02 : f32
    %cst_7 = arith.constant 5.000000e-02 : f32
    %alloca = memref.alloca() {alignment = 64 : i64} : memref<12x12xf32>
    %alloca_8 = memref.alloca() {alignment = 64 : i64} : memref<12x12x3xf32>
    %alloca_9 = memref.alloca() {alignment = 64 : i64} : memref<12x12x3xf32>
    %alloca_10 = memref.alloca() : memref<12x12x3xf32>
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %4 = arith.index_cast %arg0 : index to i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.mulf %5, %cst_6 : f32
      %7 = arith.mulf %6, %cst_4 : f32
      scf.for %arg1 = %c0 to %c12 step %c4 {
        %8 = arith.addi %arg1, %c4 : index
        scf.for %arg2 = %arg1 to %8 step %c1 {
          %9 = arith.index_cast %arg2 : index to i32
          %10 = arith.sitofp %9 : i32 to f32
          %11 = arith.mulf %10, %cst_7 : f32
          memref.store %11, %alloca_10[%arg0, %arg2, %c0] : memref<12x12x3xf32>
          memref.store %6, %alloca_10[%arg0, %arg2, %c1] : memref<12x12x3xf32>
          %12 = arith.mulf %11, %cst_5 : f32
          %13 = arith.addf %12, %7 : f32
          %14 = arith.muli %9, %4 : i32
          %15 = arith.remsi %14, %c9_i32 : i32
          %16 = arith.sitofp %15 : i32 to f32
          %17 = arith.mulf %16, %cst_3 : f32
          %18 = arith.addf %13, %17 : f32
          memref.store %18, %alloca_10[%arg0, %arg2, %c2] : memref<12x12x3xf32>
        }
      }
    }
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %4 = arith.index_cast %arg0 : index to i32
      scf.for %arg1 = %c0 to %c12 step %c1 {
        %5 = arith.index_cast %arg1 : index to i32
        %6 = arith.addi %5, %c-1_i32 : i32
        %7 = arith.cmpi sge, %6, %c0_i32 : i32
        %8 = arith.cmpi sgt, %6, %c11_i32 : i32
        %9 = arith.andi %7, %8 : i1
        %10 = arith.xori %9, %true : i1
        %11 = arith.andi %10, %7 : i1
        %12 = arith.select %9, %c11_i32, %c0_i32 : i32
        %13 = arith.select %11, %6, %12 : i32
        %14 = arith.addi %5, %c1_i32 : i32
        %15 = arith.cmpi sge, %14, %c0_i32 : i32
        %16 = arith.cmpi sgt, %14, %c11_i32 : i32
        %17 = arith.andi %15, %16 : i1
        %18 = arith.xori %17, %true : i1
        %19 = arith.andi %18, %15 : i1
        %20 = arith.select %17, %c11_i32, %c0_i32 : i32
        %21 = arith.select %19, %14, %20 : i32
        %22 = arith.addi %4, %c-1_i32 : i32
        %23 = arith.cmpi sge, %22, %c0_i32 : i32
        %24 = arith.cmpi sgt, %22, %c11_i32 : i32
        %25 = arith.andi %23, %24 : i1
        %26 = arith.xori %25, %true : i1
        %27 = arith.andi %26, %23 : i1
        %28 = arith.select %25, %c11_i32, %c0_i32 : i32
        %29 = arith.select %27, %22, %28 : i32
        %30 = arith.addi %4, %c1_i32 : i32
        %31 = arith.cmpi sge, %30, %c0_i32 : i32
        %32 = arith.cmpi sgt, %30, %c11_i32 : i32
        %33 = arith.andi %31, %32 : i1
        %34 = arith.xori %33, %true : i1
        %35 = arith.andi %34, %31 : i1
        %36 = arith.select %33, %c11_i32, %c0_i32 : i32
        %37 = arith.select %35, %30, %36 : i32
        %38 = arith.index_cast %21 : i32 to index
        %39 = memref.load %alloca_10[%arg0, %38, %c0] : memref<12x12x3xf32>
        %40 = arith.index_cast %13 : i32 to index
        %41 = memref.load %alloca_10[%arg0, %40, %c0] : memref<12x12x3xf32>
        %42 = arith.subf %39, %41 : f32
        %43 = memref.load %alloca_10[%arg0, %38, %c1] : memref<12x12x3xf32>
        %44 = memref.load %alloca_10[%arg0, %40, %c1] : memref<12x12x3xf32>
        %45 = arith.subf %43, %44 : f32
        %46 = memref.load %alloca_10[%arg0, %38, %c2] : memref<12x12x3xf32>
        %47 = memref.load %alloca_10[%arg0, %40, %c2] : memref<12x12x3xf32>
        %48 = arith.subf %46, %47 : f32
        %49 = arith.index_cast %37 : i32 to index
        %50 = memref.load %alloca_10[%49, %arg1, %c0] : memref<12x12x3xf32>
        %51 = arith.index_cast %29 : i32 to index
        %52 = memref.load %alloca_10[%51, %arg1, %c0] : memref<12x12x3xf32>
        %53 = arith.subf %50, %52 : f32
        %54 = memref.load %alloca_10[%49, %arg1, %c1] : memref<12x12x3xf32>
        %55 = memref.load %alloca_10[%51, %arg1, %c1] : memref<12x12x3xf32>
        %56 = arith.subf %54, %55 : f32
        %57 = memref.load %alloca_10[%49, %arg1, %c2] : memref<12x12x3xf32>
        %58 = memref.load %alloca_10[%51, %arg1, %c2] : memref<12x12x3xf32>
        %59 = arith.subf %57, %58 : f32
        %60 = arith.mulf %45, %59 : f32
        %61 = arith.mulf %48, %56 : f32
        %62 = arith.subf %60, %61 : f32
        %63 = arith.mulf %48, %53 : f32
        %64 = arith.mulf %42, %59 : f32
        %65 = arith.subf %63, %64 : f32
        %66 = arith.mulf %42, %56 : f32
        %67 = arith.mulf %45, %53 : f32
        %68 = arith.subf %66, %67 : f32
        %69 = arith.cmpf olt, %62, %cst : f32
        %70 = scf.if %69 -> (f32) {
          %97 = arith.negf %62 : f32
          scf.yield %97 : f32
        } else {
          scf.yield %62 : f32
        }
        %71 = arith.cmpf olt, %65, %cst : f32
        %72 = scf.if %71 -> (f32) {
          %97 = arith.negf %65 : f32
          scf.yield %97 : f32
        } else {
          scf.yield %65 : f32
        }
        %73 = arith.addf %70, %72 : f32
        %74 = arith.cmpf olt, %68, %cst : f32
        %75 = scf.if %74 -> (f32) {
          %97 = arith.negf %68 : f32
          scf.yield %97 : f32
        } else {
          scf.yield %68 : f32
        }
        %76 = arith.addf %73, %75 : f32
        %77 = arith.addf %76, %cst_1 : f32
        %78 = arith.divf %cst_0, %77 : f32
        %79 = arith.mulf %62, %78 : f32
        memref.store %79, %alloca_9[%arg0, %arg1, %c0] : memref<12x12x3xf32>
        %80 = arith.mulf %65, %78 : f32
        memref.store %80, %alloca_9[%arg0, %arg1, %c1] : memref<12x12x3xf32>
        %81 = arith.mulf %68, %78 : f32
        memref.store %81, %alloca_9[%arg0, %arg1, %c2] : memref<12x12x3xf32>
        %82 = memref.load %alloca_10[%arg0, %38, %c2] : memref<12x12x3xf32>
        %83 = memref.load %alloca_10[%arg0, %arg1, %c2] : memref<12x12x3xf32>
        %84 = arith.mulf %83, %cst_2 : f32
        %85 = arith.subf %82, %84 : f32
        %86 = memref.load %alloca_10[%arg0, %40, %c2] : memref<12x12x3xf32>
        %87 = arith.addf %85, %86 : f32
        %88 = arith.cmpf olt, %87, %cst : f32
        %89 = scf.if %88 -> (f32) {
          %97 = arith.negf %87 : f32
          scf.yield %97 : f32
        } else {
          scf.yield %87 : f32
        }
        %90 = memref.load %alloca_10[%49, %arg1, %c2] : memref<12x12x3xf32>
        %91 = arith.subf %90, %84 : f32
        %92 = memref.load %alloca_10[%51, %arg1, %c2] : memref<12x12x3xf32>
        %93 = arith.addf %91, %92 : f32
        %94 = arith.cmpf olt, %93, %cst : f32
        %95 = scf.if %94 -> (f32) {
          %97 = arith.negf %93 : f32
          scf.yield %97 : f32
        } else {
          scf.yield %93 : f32
        }
        %96 = arith.addf %89, %95 : f32
        memref.store %96, %alloca[%arg0, %arg1] : memref<12x12xf32>
      }
    }
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %4 = arith.index_cast %arg0 : index to i32
      %intptr_14 = memref.extract_aligned_pointer_as_index %alloca_9 : memref<12x12x3xf32> -> index
      %5 = arith.index_castui %intptr_14 : index to i32
      %intptr_15 = memref.extract_aligned_pointer_as_index %alloca_8 : memref<12x12x3xf32> -> index
      %6 = arith.index_castui %intptr_15 : index to i32
      %7 = arith.index_castui %arg0 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0Acsrrw x0, 2050, $3\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,r,r,~{memory}" %4, %5, %6, %7 : (i32, i32, i32, i32) -> ()
    }
    %alloca_11 = memref.alloca() {alignment = 64 : i64} : memref<i32>
    memref.store %c0_i32, %alloca_11[] : memref<i32>
    memref.store %c0_i32, %alloca_11[] : memref<i32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_8 : memref<12x12x3xf32> -> index
    %0 = arith.index_castui %intptr : index to i32
    %intptr_12 = memref.extract_aligned_pointer_as_index %alloca : memref<12x12xf32> -> index
    %1 = arith.index_castui %intptr_12 : index to i32
    %intptr_13 = memref.extract_aligned_pointer_as_index %alloca_11 : memref<i32> -> index
    %2 = arith.index_castui %intptr_13 : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2051, $2\0A.insn r 43, 3, 1, x0, $0, $1\0Afence rw, rw", "r,r,r,~{memory}" %0, %1, %2 : (i32, i32, i32) -> ()
    %3 = memref.load %alloca_11[] : memref<i32>
    call @pcl_validate_outputs(%alloca_10, %alloca_9, %alloca_8, %alloca) : (memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12xf32>) -> ()
    return %3 : i32
  }
  func.func private @pcl_validate_outputs(memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12x3xf32>, memref<12x12xf32>)
}

