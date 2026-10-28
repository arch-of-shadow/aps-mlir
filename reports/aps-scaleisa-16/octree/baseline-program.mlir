module {
  func.func @pcl_selected_main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
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
    affine.store %c0_i32, %alloca[] : memref<i32>
    %alloca_2 = memref.alloca() : memref<128xi32>
    %alloca_3 = memref.alloca() : memref<4x4096xi32>
    %alloca_4 = memref.alloca() : memref<2048x3xf32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg0 = 0 to 4 {
      affine.for %arg1 = 0 to 4096 {
        affine.store %c0_i32, %alloca_3[%arg0, %arg1] : memref<4x4096xi32>
      }
    }
    affine.for %arg0 = 0 to 2048 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.muli %1, %c13_i32 : i32
      %3 = arith.addi %2, %c7_i32 : i32
      %4 = arith.remsi %3, %c127_i32 : i32
      %5 = arith.sitofp %4 : i32 to f32
      %6 = arith.divf %5, %cst_1 : f32
      affine.store %6, %alloca_4[%arg0, 0] : memref<2048x3xf32>
      %7 = arith.muli %1, %c17_i32 : i32
      %8 = arith.addi %7, %c11_i32 : i32
      %9 = arith.remsi %8, %c131_i32 : i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.divf %10, %cst_0 : f32
      affine.store %11, %alloca_4[%arg0, 1] : memref<2048x3xf32>
      %12 = arith.muli %1, %c19_i32 : i32
      %13 = arith.addi %12, %c5_i32 : i32
      %14 = arith.remsi %13, %c137_i32 : i32
      %15 = arith.sitofp %14 : i32 to f32
      %16 = arith.divf %15, %cst : f32
      affine.store %16, %alloca_4[%arg0, 2] : memref<2048x3xf32>
    }
    affine.for %arg0 = 0 to 4 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.shrsi %c16_i32, %1 : i32
      affine.for %arg1 = 0 to 2048 {
        %3 = affine.load %alloca_4[%arg1, 0] : memref<2048x3xf32>
        %4 = arith.sitofp %2 : i32 to f32
        %5 = arith.mulf %3, %4 : f32
        %6 = arith.fptosi %5 : f32 to i32
        %7 = arith.addi %2, %c-1_i32 : i32
        %8 = arith.cmpi sge, %6, %c0_i32 : i32
        %9 = arith.cmpi sgt, %6, %7 : i32
        %10 = arith.andi %8, %9 : i1
        %11 = arith.xori %10, %true : i1
        %12 = arith.andi %11, %8 : i1
        %13 = arith.select %10, %7, %c0_i32 : i32
        %14 = arith.select %12, %6, %13 : i32
        %15 = affine.load %alloca_4[%arg1, 1] : memref<2048x3xf32>
        %16 = arith.mulf %15, %4 : f32
        %17 = arith.fptosi %16 : f32 to i32
        %18 = arith.cmpi sge, %17, %c0_i32 : i32
        %19 = arith.cmpi sgt, %17, %7 : i32
        %20 = arith.andi %18, %19 : i1
        %21 = arith.xori %20, %true : i1
        %22 = arith.andi %21, %18 : i1
        %23 = arith.select %20, %7, %c0_i32 : i32
        %24 = arith.select %22, %17, %23 : i32
        %25 = affine.load %alloca_4[%arg1, 2] : memref<2048x3xf32>
        %26 = arith.mulf %25, %4 : f32
        %27 = arith.fptosi %26 : f32 to i32
        %28 = arith.cmpi sge, %27, %c0_i32 : i32
        %29 = arith.cmpi sgt, %27, %7 : i32
        %30 = arith.andi %28, %29 : i1
        %31 = arith.xori %30, %true : i1
        %32 = arith.andi %31, %28 : i1
        %33 = arith.select %30, %7, %c0_i32 : i32
        %34 = arith.select %32, %27, %33 : i32
        %35 = arith.muli %14, %c16_i32 : i32
        %36 = arith.addi %35, %24 : i32
        %37 = arith.muli %36, %c16_i32 : i32
        %38 = arith.addi %37, %34 : i32
        %39 = arith.index_cast %38 : i32 to index
        %40 = memref.load %alloca_3[%arg0, %39] : memref<4x4096xi32>
        %41 = arith.addi %40, %c1_i32 : i32
        memref.store %41, %alloca_3[%arg0, %39] : memref<4x4096xi32>
      }
    }
    %alloca_5 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_5[] : memref<i32>
    affine.for %arg0 = 0 to 128 {
      affine.store %c0_i32, %alloca_5[] : memref<i32>
      %1 = affine.load %alloca_4[%arg0, 0] : memref<2048x3xf32>
      %2 = affine.load %alloca_4[%arg0, 1] : memref<2048x3xf32>
      %3 = affine.load %alloca_4[%arg0, 2] : memref<2048x3xf32>
      affine.for %arg1 = 0 to 4 {
        %5 = arith.index_cast %arg1 : index to i32
        %6 = arith.shrsi %c16_i32, %5 : i32
        %7 = arith.sitofp %6 : i32 to f32
        %8 = arith.mulf %1, %7 : f32
        %9 = arith.fptosi %8 : f32 to i32
        %10 = arith.addi %6, %c-1_i32 : i32
        %11 = arith.cmpi sge, %9, %c0_i32 : i32
        %12 = arith.cmpi sgt, %9, %10 : i32
        %13 = arith.andi %11, %12 : i1
        %14 = arith.xori %13, %true : i1
        %15 = arith.andi %14, %11 : i1
        %16 = arith.select %13, %10, %c0_i32 : i32
        %17 = arith.select %15, %9, %16 : i32
        %18 = arith.mulf %2, %7 : f32
        %19 = arith.fptosi %18 : f32 to i32
        %20 = arith.cmpi sge, %19, %c0_i32 : i32
        %21 = arith.cmpi sgt, %19, %10 : i32
        %22 = arith.andi %20, %21 : i1
        %23 = arith.xori %22, %true : i1
        %24 = arith.andi %23, %20 : i1
        %25 = arith.select %22, %10, %c0_i32 : i32
        %26 = arith.select %24, %19, %25 : i32
        %27 = arith.mulf %3, %7 : f32
        %28 = arith.fptosi %27 : f32 to i32
        %29 = arith.cmpi sge, %28, %c0_i32 : i32
        %30 = arith.cmpi sgt, %28, %10 : i32
        %31 = arith.andi %29, %30 : i1
        %32 = arith.xori %31, %true : i1
        %33 = arith.andi %32, %29 : i1
        %34 = arith.select %31, %10, %c0_i32 : i32
        %35 = arith.select %33, %28, %34 : i32
        affine.for %arg2 = 0 to 3 {
          %36 = arith.index_cast %arg2 : index to i32
          %37 = arith.addi %36, %c-1_i32 : i32
          affine.for %arg3 = 0 to 3 {
            %38 = arith.index_cast %arg3 : index to i32
            %39 = arith.addi %38, %c-1_i32 : i32
            affine.for %arg4 = 0 to 3 {
              %40 = arith.index_cast %arg4 : index to i32
              %41 = arith.addi %40, %c-1_i32 : i32
              %42 = arith.addi %17, %41 : i32
              %43 = arith.cmpi sge, %42, %c0_i32 : i32
              %44 = arith.cmpi sgt, %42, %10 : i32
              %45 = arith.andi %43, %44 : i1
              %46 = arith.xori %45, %true : i1
              %47 = arith.andi %46, %43 : i1
              %48 = arith.select %45, %10, %c0_i32 : i32
              %49 = arith.select %47, %42, %48 : i32
              %50 = arith.addi %26, %39 : i32
              %51 = arith.cmpi sge, %50, %c0_i32 : i32
              %52 = arith.cmpi sgt, %50, %10 : i32
              %53 = arith.andi %51, %52 : i1
              %54 = arith.xori %53, %true : i1
              %55 = arith.andi %54, %51 : i1
              %56 = arith.select %53, %10, %c0_i32 : i32
              %57 = arith.select %55, %50, %56 : i32
              %58 = arith.addi %35, %37 : i32
              %59 = arith.cmpi sge, %58, %c0_i32 : i32
              %60 = arith.cmpi sgt, %58, %10 : i32
              %61 = arith.andi %59, %60 : i1
              %62 = arith.xori %61, %true : i1
              %63 = arith.andi %62, %59 : i1
              %64 = arith.select %61, %10, %c0_i32 : i32
              %65 = arith.select %63, %58, %64 : i32
              %66 = arith.muli %49, %c16_i32 : i32
              %67 = arith.addi %66, %57 : i32
              %68 = arith.muli %67, %c16_i32 : i32
              %69 = arith.addi %68, %65 : i32
              %70 = arith.index_cast %69 : i32 to index
              %71 = memref.load %alloca_3[%arg1, %70] : memref<4x4096xi32>
              %72 = arith.addi %5, %c1_i32 : i32
              %73 = arith.muli %71, %72 : i32
              %74 = affine.load %alloca_5[] : memref<i32>
              %75 = arith.addi %74, %73 : i32
              affine.store %75, %alloca_5[] : memref<i32>
            }
          }
        }
      }
      %4 = affine.load %alloca_5[] : memref<i32>
      affine.store %4, %alloca_2[%arg0] : memref<128xi32>
    }
    affine.for %arg0 = 0 to 128 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = affine.load %alloca_2[%arg0] : memref<128xi32>
      %3 = arith.andi %1, %c7_i32 : i32
      %4 = arith.addi %3, %c1_i32 : i32
      %5 = arith.muli %2, %4 : i32
      %6 = affine.load %alloca[] : memref<i32>
      %7 = arith.addi %6, %5 : i32
      affine.store %7, %alloca[] : memref<i32>
    }
    %0 = affine.load %alloca[] : memref<i32>
    call @pcl_validate_outputs(%alloca_4, %alloca_3, %alloca_2) : (memref<2048x3xf32>, memref<4x4096xi32>, memref<128xi32>) -> ()
    return %0 : i32
  }
  func.func private @pcl_validate_outputs(memref<2048x3xf32>, memref<4x4096xi32>, memref<128xi32>)
}

