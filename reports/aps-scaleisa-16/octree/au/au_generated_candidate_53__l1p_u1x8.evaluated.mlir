module {
  func.func @au_generated_candidate_53__l1p_u1x8(%arg0: memref<4x4096xi32>) attributes {area_square_microns = 1245.9339 : f64, "array-partition" = true, au.generated = true, clock = 8.000000e+00 : f64, control_area_square_microns = 335.22336000000001 : f64, cycle = 2053 : i64, datapath_area_square_microns = 910.71054000000004 : f64, evaluation_status = "ok", fully_unrolled = false, hls.partition_bank_count_0 = 8 : i64, latency_cycles = 2053 : i64, partition_cyclic_array_0 = [true], partition_dim_array_0 = [1], partition_factor_array_0 = [8], resource = "/newdisk/aps-mlir/scaleisa/tmp/aps-scaleisa-build-aps-octree-asap7-8ns-a5000/target-resource.json", schedule = true} {
    %c7 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 7 : index
    %c6 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 6 : index
    %c5 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 5 : index
    %c3 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 3 : index
    %c2 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 2 : index
    %c8 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 8 : index
    %c4096 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4096 : index
    %c0_i32 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : i32
    %c0 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 0 : index
    %c4 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 4 : index
    %c1 = arith.constant {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} 1 : index
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c4096 step %c8 {
        memref.store %c0_i32, %arg0[%arg1, %arg2] {endtime = 1 : i64, hls.memory_bank = 0 : i64, hls.partition_factor = 8 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x4096xi32>
        %0 = arith.addi %arg2, %c1 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        memref.store %c0_i32, %arg0[%arg1, %0] {endtime = 1 : i64, hls.memory_bank = 1 : i64, hls.partition_factor = 8 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x4096xi32>
        %1 = arith.addi %arg2, %c2 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        memref.store %c0_i32, %arg0[%arg1, %1] {endtime = 1 : i64, hls.memory_bank = 2 : i64, hls.partition_factor = 8 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x4096xi32>
        %2 = arith.addi %arg2, %c3 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        memref.store %c0_i32, %arg0[%arg1, %2] {endtime = 1 : i64, hls.memory_bank = 3 : i64, hls.partition_factor = 8 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x4096xi32>
        %3 = arith.addi %arg2, %c4 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        memref.store %c0_i32, %arg0[%arg1, %3] {endtime = 1 : i64, hls.memory_bank = 4 : i64, hls.partition_factor = 8 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x4096xi32>
        %4 = arith.addi %arg2, %c5 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        memref.store %c0_i32, %arg0[%arg1, %4] {endtime = 1 : i64, hls.memory_bank = 5 : i64, hls.partition_factor = 8 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x4096xi32>
        %5 = arith.addi %arg2, %c6 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        memref.store %c0_i32, %arg0[%arg1, %5] {endtime = 1 : i64, hls.memory_bank = 6 : i64, hls.partition_factor = 8 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x4096xi32>
        %6 = arith.addi %arg2, %c7 {endtime = 0 : i64, ref_endtime = 0 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : index
        memref.store %c0_i32, %arg0[%arg1, %6] {endtime = 1 : i64, hls.memory_bank = 7 : i64, hls.partition_factor = 8 : i64, ref_endtime = 1 : i64, ref_starttime = 0 : i64, starttime = 0 : i64} : memref<4x4096xi32>
      } {II = 1 : i64, cycle = 513 : i64, endtime = 513 : i64, pipeline = 0 : i64, ref_endtime = 513 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 512 : i64, unroll = 8 : i64}
    } {cycle = 2053 : i64, endtime = 2053 : i64, ref_endtime = 2053 : i64, ref_starttime = 0 : i64, starttime = 0 : i64, "tripcount-max" = 4 : i64}
    return
  }
}

