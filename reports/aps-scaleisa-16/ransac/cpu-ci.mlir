module attributes {scaleisa.ci_instructions = 2 : i64, scaleisa.ci_sites = 2 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 1.000000e+03 : f32
    %cst_0 = arith.constant 1.100000e+03 : f32
    %cst_1 = arith.constant 1.200000e+03 : f32
    %cst_2 = arith.constant 1.300000e+03 : f32
    %c0 = arith.constant 0 : index
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst_4 = arith.constant 1.000000e+30 : f32
    %c11_i32 = arith.constant 11 : i32
    %c32_i32 = arith.constant 32 : i32
    %c23_i32 = arith.constant 23 : i32
    %c29_i32 = arith.constant 29 : i32
    %cst_5 = arith.constant 2.500000e-01 : f32
    %c2_i32 = arith.constant 2 : i32
    %false = arith.constant false
    %true = arith.constant true
    %c31_i32 = arith.constant 31 : i32
    %c13_i32 = arith.constant 13 : i32
    %c17_i32 = arith.constant 17 : i32
    %c19_i32 = arith.constant 19 : i32
    %c7_i32 = arith.constant 7 : i32
    %alloca = memref.alloca() : memref<4xi32>
    %alloca_6 = memref.alloca() : memref<4xf32>
    %alloca_7 = memref.alloca() {alignment = 64 : i64} : memref<32x3xf32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_7 : memref<32x3xf32> -> index
    %0 = arith.index_castui %intptr : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 2, 1, x0, $0, x0\0Afence rw, rw", "r,~{memory}" %0 : (i32) -> ()
    %alloca_8 = memref.alloca() : memref<1xi32>
    memref.store %c0_i32, %alloca_8[%c0] : memref<1xi32>
    %alloca_9 = memref.alloca() : memref<1xi32>
    memref.store %c0_i32, %alloca_9[%c0] : memref<1xi32>
    %alloca_10 = memref.alloca() : memref<1xi32>
    %cast = memref.cast %alloca_10 : memref<1xi32> to memref<?xi32>
    memref.store %c0_i32, %alloca_10[%c0] : memref<1xi32>
    %alloca_11 = memref.alloca() : memref<1xi32>
    %cast_12 = memref.cast %alloca_11 : memref<1xi32> to memref<?xi32>
    memref.store %c0_i32, %alloca_11[%c0] : memref<1xi32>
    %alloca_13 = memref.alloca() : memref<f32>
    memref.store %cst_3, %alloca_13[] : memref<f32>
    %alloca_14 = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca_14[] : memref<i32>
    memref.store %c-1_i32, %alloca_14[] : memref<i32>
    memref.store %cst_4, %alloca_13[] : memref<f32>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      %17 = arith.index_cast %arg0 : index to i32
      %18 = arith.muli %17, %c7_i32 : i32
      %19 = arith.addi %18, %c11_i32 : i32
      %20 = arith.remsi %19, %c32_i32 : i32
      memref.store %20, %alloca_11[%c0] : memref<1xi32>
      %21 = arith.muli %17, %c13_i32 : i32
      %22 = arith.addi %21, %c19_i32 : i32
      %23 = arith.remsi %22, %c32_i32 : i32
      memref.store %23, %alloca_10[%c0] : memref<1xi32>
      %24 = arith.muli %17, %c17_i32 : i32
      %25 = arith.addi %24, %c23_i32 : i32
      %26 = arith.remsi %25, %c32_i32 : i32
      memref.store %26, %alloca_9[%c0] : memref<1xi32>
      %27 = arith.muli %17, %c29_i32 : i32
      %28 = arith.addi %27, %c31_i32 : i32
      %29 = arith.remsi %28, %c32_i32 : i32
      memref.store %29, %alloca_8[%c0] : memref<1xi32>
      %30 = memref.load %alloca_11[%c0] : memref<1xi32>
      %31 = arith.index_cast %30 : i32 to index
      %32 = memref.load %alloca_7[%31, %c0] : memref<32x3xf32>
      %33 = memref.load %alloca_10[%c0] : memref<1xi32>
      %34 = arith.index_cast %33 : i32 to index
      %35 = memref.load %alloca_7[%34, %c0] : memref<32x3xf32>
      %36 = arith.addf %32, %35 : f32
      %37 = memref.load %alloca_9[%c0] : memref<1xi32>
      %38 = arith.index_cast %37 : i32 to index
      %39 = memref.load %alloca_7[%38, %c0] : memref<32x3xf32>
      %40 = arith.addf %36, %39 : f32
      %41 = memref.load %alloca_8[%c0] : memref<1xi32>
      %42 = arith.index_cast %41 : i32 to index
      %43 = memref.load %alloca_7[%42, %c0] : memref<32x3xf32>
      %44 = arith.addf %40, %43 : f32
      %45 = arith.mulf %44, %cst_5 : f32
      %46 = memref.load %alloca_7[%31, %c1] : memref<32x3xf32>
      %47 = memref.load %alloca_7[%34, %c1] : memref<32x3xf32>
      %48 = arith.addf %46, %47 : f32
      %49 = memref.load %alloca_7[%38, %c1] : memref<32x3xf32>
      %50 = arith.addf %48, %49 : f32
      %51 = memref.load %alloca_7[%42, %c1] : memref<32x3xf32>
      %52 = arith.addf %50, %51 : f32
      %53 = arith.mulf %52, %cst_5 : f32
      %54 = memref.load %alloca_7[%31, %c2] : memref<32x3xf32>
      %55 = memref.load %alloca_7[%34, %c2] : memref<32x3xf32>
      %56 = arith.addf %54, %55 : f32
      %57 = memref.load %alloca_7[%38, %c2] : memref<32x3xf32>
      %58 = arith.addf %56, %57 : f32
      %59 = memref.load %alloca_7[%42, %c2] : memref<32x3xf32>
      %60 = arith.addf %58, %59 : f32
      %61 = arith.mulf %60, %cst_5 : f32
      %alloca_16 = memref.alloca() : memref<f32>
      memref.store %cst_3, %alloca_16[] : memref<f32>
      %alloca_17 = memref.alloca() {alignment = 64 : i64} : memref<i32>
      memref.store %c0_i32, %alloca_17[] : memref<i32>
      %alloca_18 = memref.alloca() {alignment = 64 : i64} : memref<f32>
      memref.store %cst_3, %alloca_18[] : memref<f32>
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %75 = arith.index_cast %arg1 : index to i32
        %76 = arith.cmpi eq, %75, %c0_i32 : i32
        %77 = scf.if %76 -> (memref<?xi32>) {
          scf.yield %cast_12 : memref<?xi32>
        } else {
          %96 = arith.cmpi eq, %75, %c1_i32 : i32
          %97 = scf.if %96 -> (memref<?xi32>) {
            scf.yield %cast : memref<?xi32>
          } else {
            %98 = arith.cmpi eq, %75, %c2_i32 : i32
            %99 = arith.select %98, %alloca_9, %alloca_8 : memref<1xi32>
            %cast_22 = memref.cast %99 : memref<1xi32> to memref<?xi32>
            scf.yield %cast_22 : memref<?xi32>
          }
          scf.yield %97 : memref<?xi32>
        }
        %78 = memref.load %77[%c0] : memref<?xi32>
        %79 = arith.index_cast %78 : i32 to index
        %80 = memref.load %alloca_7[%79, %c0] : memref<32x3xf32>
        %81 = arith.subf %80, %45 : f32
        %82 = memref.load %alloca_7[%79, %c1] : memref<32x3xf32>
        %83 = arith.subf %82, %53 : f32
        %84 = memref.load %alloca_7[%79, %c2] : memref<32x3xf32>
        %85 = arith.subf %84, %61 : f32
        %86 = arith.cmpf olt, %81, %cst_3 : f32
        %87 = scf.if %86 -> (f32) {
          %96 = arith.negf %81 : f32
          scf.yield %96 : f32
        } else {
          scf.yield %81 : f32
        }
        %88 = arith.cmpf olt, %83, %cst_3 : f32
        %89 = scf.if %88 -> (f32) {
          %96 = arith.negf %83 : f32
          scf.yield %96 : f32
        } else {
          scf.yield %83 : f32
        }
        %90 = arith.addf %87, %89 : f32
        %91 = arith.cmpf olt, %85, %cst_3 : f32
        %92 = scf.if %91 -> (f32) {
          %96 = arith.negf %85 : f32
          scf.yield %96 : f32
        } else {
          scf.yield %85 : f32
        }
        %93 = arith.addf %90, %92 : f32
        %94 = memref.load %alloca_16[] : memref<f32>
        %95 = arith.addf %94, %93 : f32
        memref.store %95, %alloca_16[] : memref<f32>
      }
      %62 = memref.load %alloca_16[] : memref<f32>
      %63 = arith.mulf %62, %cst_5 : f32
      memref.store %63, %alloca_16[] : memref<f32>
      %intptr_19 = memref.extract_aligned_pointer_as_index %alloca_7 : memref<32x3xf32> -> index
      %64 = arith.index_castui %intptr_19 : index to i32
      %65 = arith.bitcast %45 : f32 to i32
      %66 = arith.bitcast %53 : f32 to i32
      %67 = arith.bitcast %61 : f32 to i32
      %68 = arith.bitcast %63 : f32 to i32
      %intptr_20 = memref.extract_aligned_pointer_as_index %alloca_17 : memref<i32> -> index
      %69 = arith.index_castui %intptr_20 : index to i32
      %intptr_21 = memref.extract_aligned_pointer_as_index %alloca_18 : memref<f32> -> index
      %70 = arith.index_castui %intptr_21 : index to i32
      llvm.inline_asm has_side_effects "fence rw, rw\0Acsrrw x0, 2049, $2\0Acsrrw x0, 2050, $3\0Acsrrw x0, 2051, $4\0Acsrrw x0, 2052, $5\0Acsrrw x0, 2053, $6\0A.insn r 43, 3, 0, x0, $0, $1\0Afence rw, rw", "r,r,r,r,r,r,r,~{memory}" %64, %65, %66, %67, %68, %69, %70 : (i32, i32, i32, i32, i32, i32, i32) -> ()
      %71 = memref.load %alloca_17[] : memref<i32>
      memref.store %71, %alloca[%arg0] : memref<4xi32>
      %72 = memref.load %alloca_14[] : memref<i32>
      %73 = arith.cmpi sgt, %71, %72 : i32
      %74 = scf.if %73 -> (i1) {
        scf.yield %true : i1
      } else {
        %75 = arith.cmpi eq, %71, %72 : i32
        %76 = scf.if %75 -> (i1) {
          %77 = memref.load %alloca_18[] : memref<f32>
          %78 = memref.load %alloca_13[] : memref<f32>
          %79 = arith.cmpf olt, %77, %78 : f32
          scf.yield %79 : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %76 : i1
      }
      scf.if %74 {
        memref.store %71, %alloca_14[] : memref<i32>
        %75 = memref.load %alloca_18[] : memref<f32>
        memref.store %75, %alloca_13[] : memref<f32>
        memref.store %45, %alloca_6[%c0] : memref<4xf32>
        memref.store %53, %alloca_6[%c1] : memref<4xf32>
        memref.store %61, %alloca_6[%c2] : memref<4xf32>
        memref.store %63, %alloca_6[%c3] : memref<4xf32>
      }
    }
    %alloca_15 = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca_15[] : memref<i32>
    %1 = memref.load %alloca_6[%c0] : memref<4xf32>
    %2 = arith.mulf %1, %cst : f32
    %3 = arith.fptosi %2 : f32 to i32
    %4 = memref.load %alloca_6[%c1] : memref<4xf32>
    %5 = arith.mulf %4, %cst_0 : f32
    %6 = arith.fptosi %5 : f32 to i32
    %7 = arith.addi %3, %6 : i32
    %8 = memref.load %alloca_6[%c2] : memref<4xf32>
    %9 = arith.mulf %8, %cst_1 : f32
    %10 = arith.fptosi %9 : f32 to i32
    %11 = arith.addi %7, %10 : i32
    %12 = memref.load %alloca_6[%c3] : memref<4xf32>
    %13 = arith.mulf %12, %cst_2 : f32
    %14 = arith.fptosi %13 : f32 to i32
    %15 = arith.addi %11, %14 : i32
    memref.store %15, %alloca_15[] : memref<i32>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      %17 = memref.load %alloca[%arg0] : memref<4xi32>
      %18 = memref.load %alloca_15[] : memref<i32>
      %19 = arith.addi %18, %17 : i32
      memref.store %19, %alloca_15[] : memref<i32>
    }
    %16 = memref.load %alloca_15[] : memref<i32>
    call @pcl_validate_outputs(%alloca_7, %alloca_6, %alloca) : (memref<32x3xf32>, memref<4xf32>, memref<4xi32>) -> ()
    return %16 : i32
  }
  func.func private @pcl_validate_outputs(memref<32x3xf32>, memref<4xf32>, memref<4xi32>)
}

