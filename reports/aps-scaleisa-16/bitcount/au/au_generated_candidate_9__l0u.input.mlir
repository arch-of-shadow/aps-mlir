module {
  func.func @au_generated_candidate_9__l0u(%arg0: i32) -> i32 attributes {au.generated = true} {
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg1 = 0 to 32 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg1 : index to i32
      %2 = arith.shrui %arg0, %1 : i32
      %c1_i32 = arith.constant 1 : i32
      %3 = arith.andi %2, %c1_i32 : i32
      %4 = arith.addi %arg2, %3 : i32
      affine.yield %4 : i32
    } {unroll = 0 : i64}
    return %0 : i32
  }
}

