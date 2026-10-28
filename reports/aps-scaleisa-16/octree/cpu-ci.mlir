module attributes {scaleisa.ci_instructions = 1 : i64, scaleisa.ci_sites = 1 : i64} {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c2 = arith.constant 2 : index
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %c16_i32 = arith.constant 16 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant 1.370000e+02 : f32
    %c137_i32 = arith.constant 137 : i32
    %c5_i32 = arith.constant 5 : i32
    %c19_i32 = arith.constant 19 : i32
    %cst_0 = arith.constant 1.310000e+02 : f32
    %c131_i32 = arith.constant 131 : i32
    %c11_i32 = arith.constant 11 : i32
    %c17_i32 = arith.constant 17 : i32
    %cst_1 = arith.constant 1.270000e+02 : f32
    %c127_i32 = arith.constant 127 : i32
    %c7_i32 = arith.constant 7 : i32
    %c13_i32 = arith.constant 13 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca[] : memref<i32>
    %alloca_2 = memref.alloca() : memref<128xi32>
    %alloca_3 = memref.alloca() {alignment = 64 : i64} : memref<4x4096xi32>
    %alloca_4 = memref.alloca() : memref<2048x3xf32>
    memref.store %c0_i32, %alloca[] : memref<i32>
    %intptr = memref.extract_aligned_pointer_as_index %alloca_3 : memref<4x4096xi32> -> index
    %0 = arith.index_castui %intptr : index to i32
    llvm.inline_asm has_side_effects "fence rw, rw\0A.insn r 43, 2, 0, x0, $0, x0\0Afence rw, rw", "r,~{memory}" %0 : (i32) -> ()
    scf.for %arg0 = %c0 to %c2048 step %c1 {
      %2 = arith.index_cast %arg0 : index to i32
      %3 = arith.muli %2, %c13_i32 : i32
      %4 = arith.addi %3, %c7_i32 : i32
      %5 = arith.remsi %4, %c127_i32 : i32
      %6 = arith.sitofp %5 : i32 to f32
      %7 = arith.divf %6, %cst_1 : f32
      memref.store %7, %alloca_4[%arg0, %c0] : memref<2048x3xf32>
      %8 = arith.muli %2, %c17_i32 : i32
      %9 = arith.addi %8, %c11_i32 : i32
      %10 = arith.remsi %9, %c131_i32 : i32
      %11 = arith.sitofp %10 : i32 to f32
      %12 = arith.divf %11, %cst_0 : f32
      memref.store %12, %alloca_4[%arg0, %c1] : memref<2048x3xf32>
      %13 = arith.muli %2, %c19_i32 : i32
      %14 = arith.addi %13, %c5_i32 : i32
      %15 = arith.remsi %14, %c137_i32 : i32
      %16 = arith.sitofp %15 : i32 to f32
      %17 = arith.divf %16, %cst : f32
      memref.store %17, %alloca_4[%arg0, %c2] : memref<2048x3xf32>
    }
    scf.for %arg0 = %c0 to %c4 step %c1 {
      %2 = arith.index_cast %arg0 : index to i32
      %3 = arith.shrsi %c16_i32, %2 : i32
      scf.for %arg1 = %c0 to %c2048 step %c1 {
        %4 = memref.load %alloca_4[%arg1, %c0] : memref<2048x3xf32>
        %5 = arith.sitofp %3 : i32 to f32
        %6 = arith.mulf %4, %5 : f32
        %7 = arith.fptosi %6 : f32 to i32
        %8 = arith.addi %3, %c-1_i32 : i32
        %9 = arith.cmpi sge, %7, %c0_i32 : i32
        %10 = arith.cmpi sgt, %7, %8 : i32
        %11 = arith.andi %9, %10 : i1
        %12 = arith.xori %11, %true : i1
        %13 = arith.andi %12, %9 : i1
        %14 = arith.select %11, %8, %c0_i32 : i32
        %15 = arith.select %13, %7, %14 : i32
        %16 = memref.load %alloca_4[%arg1, %c1] : memref<2048x3xf32>
        %17 = arith.mulf %16, %5 : f32
        %18 = arith.fptosi %17 : f32 to i32
        %19 = arith.cmpi sge, %18, %c0_i32 : i32
        %20 = arith.cmpi sgt, %18, %8 : i32
        %21 = arith.andi %19, %20 : i1
        %22 = arith.xori %21, %true : i1
        %23 = arith.andi %22, %19 : i1
        %24 = arith.select %21, %8, %c0_i32 : i32
        %25 = arith.select %23, %18, %24 : i32
        %26 = memref.load %alloca_4[%arg1, %c2] : memref<2048x3xf32>
        %27 = arith.mulf %26, %5 : f32
        %28 = arith.fptosi %27 : f32 to i32
        %29 = arith.cmpi sge, %28, %c0_i32 : i32
        %30 = arith.cmpi sgt, %28, %8 : i32
        %31 = arith.andi %29, %30 : i1
        %32 = arith.xori %31, %true : i1
        %33 = arith.andi %32, %29 : i1
        %34 = arith.select %31, %8, %c0_i32 : i32
        %35 = arith.select %33, %28, %34 : i32
        %36 = arith.muli %15, %c16_i32 : i32
        %37 = arith.addi %36, %25 : i32
        %38 = arith.muli %37, %c16_i32 : i32
        %39 = arith.addi %38, %35 : i32
        %40 = arith.index_cast %39 : i32 to index
        %41 = memref.load %alloca_3[%arg0, %40] : memref<4x4096xi32>
        %42 = arith.addi %41, %c1_i32 : i32
        memref.store %42, %alloca_3[%arg0, %40] : memref<4x4096xi32>
      }
    }
    %alloca_5 = memref.alloca() : memref<i32>
    memref.store %c0_i32, %alloca_5[] : memref<i32>
    scf.for %arg0 = %c0 to %c128 step %c1 {
      memref.store %c0_i32, %alloca_5[] : memref<i32>
      %2 = memref.load %alloca_4[%arg0, %c0] : memref<2048x3xf32>
      %3 = memref.load %alloca_4[%arg0, %c1] : memref<2048x3xf32>
      %4 = memref.load %alloca_4[%arg0, %c2] : memref<2048x3xf32>
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %6 = arith.index_cast %arg1 : index to i32
        %7 = arith.shrsi %c16_i32, %6 : i32
        %8 = arith.sitofp %7 : i32 to f32
        %9 = arith.mulf %2, %8 : f32
        %10 = arith.fptosi %9 : f32 to i32
        %11 = arith.addi %7, %c-1_i32 : i32
        %12 = arith.cmpi sge, %10, %c0_i32 : i32
        %13 = arith.cmpi sgt, %10, %11 : i32
        %14 = arith.andi %12, %13 : i1
        %15 = arith.xori %14, %true : i1
        %16 = arith.andi %15, %12 : i1
        %17 = arith.select %14, %11, %c0_i32 : i32
        %18 = arith.select %16, %10, %17 : i32
        %19 = arith.mulf %3, %8 : f32
        %20 = arith.fptosi %19 : f32 to i32
        %21 = arith.cmpi sge, %20, %c0_i32 : i32
        %22 = arith.cmpi sgt, %20, %11 : i32
        %23 = arith.andi %21, %22 : i1
        %24 = arith.xori %23, %true : i1
        %25 = arith.andi %24, %21 : i1
        %26 = arith.select %23, %11, %c0_i32 : i32
        %27 = arith.select %25, %20, %26 : i32
        %28 = arith.mulf %4, %8 : f32
        %29 = arith.fptosi %28 : f32 to i32
        %30 = arith.cmpi sge, %29, %c0_i32 : i32
        %31 = arith.cmpi sgt, %29, %11 : i32
        %32 = arith.andi %30, %31 : i1
        %33 = arith.xori %32, %true : i1
        %34 = arith.andi %33, %30 : i1
        %35 = arith.select %32, %11, %c0_i32 : i32
        %36 = arith.select %34, %29, %35 : i32
        scf.for %arg2 = %c0 to %c3 step %c1 {
          %37 = arith.index_cast %arg2 : index to i32
          %38 = arith.addi %37, %c-1_i32 : i32
          %39 = arith.addi %36, %38 : i32
          %40 = arith.cmpi sge, %39, %c0_i32 : i32
          %41 = arith.cmpi sgt, %39, %11 : i32
          %42 = arith.andi %40, %41 : i1
          %43 = arith.xori %42, %true : i1
          %44 = arith.andi %43, %40 : i1
          %45 = arith.select %42, %11, %c0_i32 : i32
          %46 = arith.select %44, %39, %45 : i32
          %47 = arith.addi %6, %c1_i32 : i32
          scf.for %arg3 = %c0 to %c3 step %c1 {
            %48 = arith.index_cast %arg3 : index to i32
            %49 = arith.addi %48, %c-1_i32 : i32
            scf.for %arg4 = %c0 to %c3 step %c1 {
              %50 = arith.index_cast %arg4 : index to i32
              %51 = arith.addi %50, %c-1_i32 : i32
              %52 = arith.addi %18, %51 : i32
              %53 = arith.cmpi sge, %52, %c0_i32 : i32
              %54 = arith.cmpi sgt, %52, %11 : i32
              %55 = arith.andi %53, %54 : i1
              %56 = arith.xori %55, %true : i1
              %57 = arith.andi %56, %53 : i1
              %58 = arith.select %55, %11, %c0_i32 : i32
              %59 = arith.select %57, %52, %58 : i32
              %60 = arith.addi %27, %49 : i32
              %61 = arith.cmpi sge, %60, %c0_i32 : i32
              %62 = arith.cmpi sgt, %60, %11 : i32
              %63 = arith.andi %61, %62 : i1
              %64 = arith.xori %63, %true : i1
              %65 = arith.andi %64, %61 : i1
              %66 = arith.select %63, %11, %c0_i32 : i32
              %67 = arith.select %65, %60, %66 : i32
              %68 = arith.muli %59, %c16_i32 : i32
              %69 = arith.addi %68, %67 : i32
              %70 = arith.muli %69, %c16_i32 : i32
              %71 = arith.addi %70, %46 : i32
              %72 = arith.index_cast %71 : i32 to index
              %73 = memref.load %alloca_3[%arg1, %72] : memref<4x4096xi32>
              %74 = arith.muli %73, %47 : i32
              %75 = memref.load %alloca_5[] : memref<i32>
              %76 = arith.addi %75, %74 : i32
              memref.store %76, %alloca_5[] : memref<i32>
            }
          }
        }
      }
      %5 = memref.load %alloca_5[] : memref<i32>
      memref.store %5, %alloca_2[%arg0] : memref<128xi32>
    }
    scf.for %arg0 = %c0 to %c128 step %c1 {
      %2 = arith.index_cast %arg0 : index to i32
      %3 = memref.load %alloca_2[%arg0] : memref<128xi32>
      %4 = arith.andi %2, %c7_i32 : i32
      %5 = arith.addi %4, %c1_i32 : i32
      %6 = arith.muli %3, %5 : i32
      %7 = memref.load %alloca[] : memref<i32>
      %8 = arith.addi %7, %6 : i32
      memref.store %8, %alloca[] : memref<i32>
    }
    %1 = memref.load %alloca[] : memref<i32>
    call @pcl_validate_outputs(%alloca_4, %alloca_3, %alloca_2) : (memref<2048x3xf32>, memref<4x4096xi32>, memref<128xi32>) -> ()
    return %1 : i32
  }
  func.func private @pcl_validate_outputs(memref<2048x3xf32>, memref<4x4096xi32>, memref<128xi32>)
}

