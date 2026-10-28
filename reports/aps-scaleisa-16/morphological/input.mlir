module {
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %alloca = memref.alloca() : memref<12x12xf32>
    %alloca_0 = memref.alloca() : memref<12x12xf32>
    %alloca_1 = memref.alloca() : memref<12x12xf32>
    %alloca_2 = memref.alloca() : memref<12x12xf32>
    %cast = memref.cast %alloca_2 : memref<12x12xf32> to memref<?x12xf32>
    call @_ZL11init_heightPA12_f(%cast) : (memref<?x12xf32>) -> ()
    %cast_3 = memref.cast %alloca_1 : memref<12x12xf32> to memref<?x12xf32>
    %cast_4 = memref.cast %alloca_0 : memref<12x12xf32> to memref<?x12xf32>
    %cast_5 = memref.cast %alloca : memref<12x12xf32> to memref<?x12xf32>
    call @_ZL19open_close_pipelinePA12_fS0_S0_S0_(%cast, %cast_3, %cast_4, %cast_5) : (memref<?x12xf32>, memref<?x12xf32>, memref<?x12xf32>, memref<?x12xf32>) -> ()
    %0 = call @_ZL9summarizePA12_f(%cast_5) : (memref<?x12xf32>) -> i32
    return %0 : i32
  }
  func.func private @_ZL11init_heightPA12_f(%arg0: memref<?x12xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-31_i32 = arith.constant -31 : i32
    %c-53_i32 = arith.constant -53 : i32
    %c51_i32 = arith.constant 51 : i32
    %c13_i32 = arith.constant 13 : i32
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant 2.000000e-02 : f32
    %c17_i32 = arith.constant 17 : i32
    affine.for %arg1 = 0 to 12 {
      %0 = arith.index_cast %arg1 : index to i32
      affine.for %arg2 = 0 to 12 {
        %1 = arith.index_cast %arg2 : index to i32
        %2 = arith.addi %1, %c-31_i32 : i32
        %3 = arith.muli %2, %2 : i32
        %4 = arith.addi %0, %c-53_i32 : i32
        %5 = arith.muli %4, %4 : i32
        %6 = arith.addi %3, %5 : i32
        %7 = arith.remsi %6, %c17_i32 : i32
        %8 = arith.muli %1, %c7_i32 : i32
        %9 = arith.muli %0, %c13_i32 : i32
        %10 = arith.addi %8, %9 : i32
        %11 = arith.addi %10, %7 : i32
        %12 = arith.remsi %11, %c51_i32 : i32
        %13 = arith.sitofp %12 : i32 to f32
        %14 = arith.mulf %13, %cst : f32
        affine.store %14, %arg0[%arg1, %arg2] : memref<?x12xf32>
      }
    }
    return
  }
  func.func private @_ZL19open_close_pipelinePA12_fS0_S0_S0_(%arg0: memref<?x12xf32>, %arg1: memref<?x12xf32>, %arg2: memref<?x12xf32>, %arg3: memref<?x12xf32>) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    call @_ZL10morph_passPA12_fS0_i(%arg0, %arg1, %c1_i32) : (memref<?x12xf32>, memref<?x12xf32>, i32) -> ()
    call @_ZL10morph_passPA12_fS0_i(%arg1, %arg2, %c0_i32) : (memref<?x12xf32>, memref<?x12xf32>, i32) -> ()
    call @_ZL10morph_passPA12_fS0_i(%arg2, %arg1, %c0_i32) : (memref<?x12xf32>, memref<?x12xf32>, i32) -> ()
    call @_ZL10morph_passPA12_fS0_i(%arg1, %arg3, %c1_i32) : (memref<?x12xf32>, memref<?x12xf32>, i32) -> ()
    return
  }
  func.func private @_ZL9summarizePA12_f(%arg0: memref<?x12xf32>) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant 1.280000e+02 : f32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.store %c0_i32, %alloca[] : memref<i32>
    affine.for %arg1 = 0 to 12 {
      %1 = arith.index_cast %arg1 : index to i32
      affine.for %arg2 = 0 to 12 {
        %2 = arith.index_cast %arg2 : index to i32
        %3 = affine.load %arg0[%arg1, %arg2] : memref<?x12xf32>
        %4 = arith.mulf %3, %cst : f32
        %5 = arith.fptosi %4 : f32 to i32
        %6 = arith.addi %2, %1 : i32
        %7 = arith.andi %6, %c3_i32 : i32
        %8 = arith.addi %5, %7 : i32
        %9 = affine.load %alloca[] : memref<i32>
        %10 = arith.addi %9, %8 : i32
        affine.store %10, %alloca[] : memref<i32>
      }
    }
    %0 = affine.load %alloca[] : memref<i32>
    return %0 : i32
  }
  func.func private @_ZL10morph_passPA12_fS0_i(%arg0: memref<?x12xf32>, %arg1: memref<?x12xf32>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %c-2_i32 = arith.constant -2 : i32
    %c95_i32 = arith.constant 11 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %alloca = memref.alloca() : memref<f32>
    affine.store %cst, %alloca[] : memref<f32>
    affine.for %arg3 = 0 to 12 {
      %0 = arith.index_cast %arg3 : index to i32
      affine.for %arg4 = 0 to 12 {
        %1 = arith.index_cast %arg4 : index to i32
        %2 = affine.load %arg0[%arg3, %arg4] : memref<?x12xf32>
        affine.store %2, %alloca[] : memref<f32>
        affine.for %arg5 = 0 to 5 {
          %4 = arith.index_cast %arg5 : index to i32
          %5 = arith.addi %4, %c-2_i32 : i32
          affine.for %arg6 = 0 to 5 {
            %6 = arith.index_cast %arg6 : index to i32
            %7 = arith.addi %6, %c-2_i32 : i32
            %8 = arith.addi %0, %5 : i32
            %9 = func.call @_ZL7clamp_iiii(%8, %c0_i32, %c95_i32) : (i32, i32, i32) -> i32
            %10 = arith.addi %1, %7 : i32
            %11 = func.call @_ZL7clamp_iiii(%10, %c0_i32, %c95_i32) : (i32, i32, i32) -> i32
            %12 = arith.index_cast %9 : i32 to index
            %13 = arith.index_cast %11 : i32 to index
            %14 = memref.load %arg0[%12, %13] : memref<?x12xf32>
            %15 = arith.cmpi eq, %arg2, %c0_i32 : i32
            scf.if %15 {
              %16 = affine.load %alloca[] : memref<f32>
              %17 = arith.cmpf ogt, %14, %16 : f32
              scf.if %17 {
                affine.store %14, %alloca[] : memref<f32>
              }
            } else {
              %16 = affine.load %alloca[] : memref<f32>
              %17 = arith.cmpf olt, %14, %16 : f32
              scf.if %17 {
                affine.store %14, %alloca[] : memref<f32>
              }
            }
          }
        }
        %3 = affine.load %alloca[] : memref<f32>
        affine.store %3, %arg1[%arg3, %arg4] : memref<?x12xf32>
      }
    }
    return
  }
  func.func private @_ZL7clamp_iiii(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi slt, %arg0, %arg1 : i32
    %1 = arith.cmpi sge, %arg0, %arg1 : i32
    %2 = arith.select %0, %arg1, %c0_i32 : i32
    %3 = arith.cmpi sgt, %arg0, %arg2 : i32
    %4 = arith.andi %1, %3 : i1
    %5 = arith.xori %4, %true : i1
    %6 = arith.andi %5, %1 : i1
    %7 = arith.select %4, %arg2, %2 : i32
    %8 = arith.select %6, %arg0, %7 : i32
    return %8 : i32
  }
}

