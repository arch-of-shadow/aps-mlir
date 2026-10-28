#map = affine_map<() -> (0)>
#map1 = affine_map<() -> (4096)>
#map2 = affine_map<() -> (4)>
"builtin.module"() ({
  "memref.global"() <{sym_name = "au_generated_candidate_53__l1p_u1x8_buffer", type = memref<16384xi32>}> {partition_cyclic_array = [1], partition_dim_array = [0], partition_factor_array = [8], scaleisa.scratchpad, var_name = "au_generated_candidate_53__l1p_u1x8_buffer"} : () -> ()
  "func.func"() <{function_type = (i5, i5, i5) -> (), sym_name = "au_generated_candidate_53__l1p_u1x8"}> ({
  ^bb0(%arg0: i5, %arg1: i5, %arg2: i5):
    %0 = "aps.read_irf"(%arg0) : (i5) -> i32
    %1 = "memref.get_global"() <{name = @au_generated_candidate_53__l1p_u1x8_buffer}> : () -> memref<16384xi32>
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "arith.constant"() <{value = 16384 : i32}> : () -> i32
    %4 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map2}> ({
    ^bb0(%arg3: index):
      "affine.for"() <{lowerBoundMap = #map, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = #map1}> ({
      ^bb0(%arg4: index):
        %5 = "arith.index_cast"(%arg3) : (index) -> i32
        %6 = "arith.index_cast"(%arg4) : (index) -> i32
        %7 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %8 = "arith.constant"() <{value = 0 : i32}> : () -> i32
        %9 = "arith.constant"() <{value = 4096 : i32}> : () -> i32
        %10 = "arith.muli"(%5, %9) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %11 = "arith.addi"(%8, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %12 = "arith.addi"(%11, %6) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        %13 = "arith.addi"(%7, %12) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
        "aps.write_smem"(%4, %1, %13) : (i32, memref<16384xi32>, i32) -> ()
        "affine.yield"() : () -> ()
      }) {pipeline = 1 : i64, unroll = 8 : i64} : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "aps.copy"(%0, %1, %2, %3) <{direction = 1 : i32}> : (i32, memref<16384xi32>, i32, i32) -> ()
    "func.return"() : () -> ()
  }) {funct3 = 2 : i32, funct7 = 0 : i32, opcode = 43 : i32, scaleisa.hardware, scaleisa.operand_bindings = [{arg_index = 0 : i32, kind = "rs1", memory_mode = "scratchpad", memref_type = memref<4x4096xi32>, value_kind = "cpu_aligned_base"}], scaleisa.source_type = (memref<4x4096xi32>) -> ()} : () -> ()
}) : () -> ()

