module {
  func.func @au_generated_candidate_38__l0u_l1u(%arg0: memref<24x3xf32>, %arg1: memref<24x3xf32>) attributes {au.generated = true} {
    affine.for %arg2 = 0 to 24 {
      affine.for %arg3 = 0 to 3 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<24x3xf32>
        affine.store %0, %arg1[%arg2, %arg3] : memref<24x3xf32>
      } {unroll = 0 : i64}
    } {unroll = 0 : i64}
    return
  }
}

