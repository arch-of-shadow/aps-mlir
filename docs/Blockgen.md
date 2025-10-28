# Instruction on Block generation

Input: 

```
module {
  aps.memorymap {
    aps.mem_entry "mem_a" : banks([@mem_a_0, @mem_a_1, @mem_a_2, @mem_a_3]), base(0), size(64), count(4), cyclic(1)
    aps.mem_entry "mem_b" : banks([@mem_b_0, @mem_b_1, @mem_b_2, @mem_b_3]), base(64), size(64), count(4), cyclic(1)
    aps.mem_entry "mem_c" : banks([@mem_c_0, @mem_c_1, @mem_c_2, @mem_c_3]), base(128), size(64), count(4), cyclic(1)
    aps.mem_finish
  }
  tor.design @aps_isaxes {
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant {dump = "op_0"} 0 : i32
    %c16_i32 = arith.constant {dump = "op_1"} 16 : i32
    %c42_i32 = arith.constant {dump = "op_2"} 42 : i32
    %c1_i32 = arith.constant {dump = "op_5"} 1 : i32
    memref.global @mem_a_0 : memref<4xi32> = dense<[1, 5, 9, 13]> {dump = "op_6"}
    memref.global @mem_a_1 : memref<4xi32> = dense<[2, 6, 10, 14]> {dump = "op_7"}
    memref.global @mem_a_2 : memref<4xi32> = dense<[3, 7, 11, 15]> {dump = "op_8"}
    memref.global @mem_a_3 : memref<4xi32> = dense<[4, 8, 12, 16]> {dump = "op_9"}
    memref.global @mem_b_0 : memref<4xi32> = uninitialized {dump = "op_10"}
    memref.global @mem_b_1 : memref<4xi32> = uninitialized {dump = "op_11"}
    memref.global @mem_b_2 : memref<4xi32> = uninitialized {dump = "op_12"}
    memref.global @mem_b_3 : memref<4xi32> = uninitialized {dump = "op_13"}
    memref.global @mem_c_0 : memref<4xi32> = uninitialized {dump = "op_14"}
    memref.global @mem_c_1 : memref<4xi32> = uninitialized {dump = "op_15"}
    memref.global @mem_c_2 : memref<4xi32> = uninitialized {dump = "op_16"}
    memref.global @mem_c_3 : memref<4xi32> = uninitialized {dump = "op_17"}
    tor.func @flow_burst_add(%arg0: i5, %arg1: i5, %arg2: i5, ...) attributes {clock = 1.000000e+01 : f32, dump = "op_71", funct7 = 0 : i32, opcode = 43 : i32, resource = "examples/resource_ihp130.json", scheduled = true} {
      tor.timegraph (0 to 13){
        tor.succ 1 : [0 : i32] [{type = "static:1"}]
        tor.succ 2 : [1 : i32] [{type = "static:16"}]
        tor.succ 3 : [2 : i32] [{type = "static:1"}]
        tor.succ 4 : [3 : i32] [{type = "static:15"}]
        tor.succ 5 : [4 : i32] [{type = "static:1"}]
        tor.succ 6 : [5 : i32] [{type = "static"}]
        tor.succ 7 : [6 : i32] [{type = "static:1"}]
        tor.succ 8 : [7 : i32] [{type = "static:1"}]
        tor.succ 9 : [5 : i32] [{type = "static-for"}]
        tor.succ 10 : [9 : i32] [{type = "static:1"}]
        tor.succ 11 : [10 : i32] [{type = "static:1"}]
        tor.succ 12 : [11 : i32] [{type = "static:16"}]
        tor.succ 13 : [12 : i32] [{type = "static:1"}]
      }
      %0 = aps.readrf %arg0 {dump = "op_18", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %1 = aps.readrf %arg1 {dump = "op_19", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
      %2 = memref.get_global @mem_a_0 : memref<4xi32> {dump = "op_20", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %3 = memref.get_global @mem_a_1 : memref<4xi32> {dump = "op_21", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %4 = memref.get_global @mem_a_2 : memref<4xi32> {dump = "op_22", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %5 = memref.get_global @mem_a_3 : memref<4xi32> {dump = "op_23", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %6 = aps.itfc.burst_load_req %0, (%2, %3, %4, %5) [%c0_i32], %c16_i32 {endtime = 2 : i32, ref_endtime = 17 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i32, (memref<4xi32>, memref<4xi32>, memref<4xi32>, memref<4xi32>), i32, i32 -> none
      aps.itfc.burst_load_collect %6 {endtime = 3 : i32, ref_endtime = 18 : i32, ref_starttime = 17 : i32, starttime = 2 : i32} : none
      %7 = memref.get_global @mem_b_0 : memref<4xi32> {dump = "op_25", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %8 = memref.get_global @mem_b_1 : memref<4xi32> {dump = "op_26", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %9 = memref.get_global @mem_b_2 : memref<4xi32> {dump = "op_27", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %10 = memref.get_global @mem_b_3 : memref<4xi32> {dump = "op_28", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
      %11 = aps.itfc.burst_load_req %1, (%7, %8, %9, %10) [%c0_i32], %c16_i32 {endtime = 4 : i32, ref_endtime = 33 : i32, ref_starttime = 17 : i32, starttime = 2 : i32} : i32, (memref<4xi32>, memref<4xi32>, memref<4xi32>, memref<4xi32>), i32, i32 -> none
      aps.itfc.burst_load_collect %11 {endtime = 5 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 4 : i32} : none
      tor.for %arg3 = (%c0_i32 : i32) to (%c3_i32 : i32) step (%c1_i32 : i32)
      on (5 to 8){
        %17 = aps.memload %2[%arg3] {dump = "op_31", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 6 : i32} : memref<4xi32>, i32 -> i32
        %18 = aps.memload %7[%arg3] {dump = "op_33", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 6 : i32} : memref<4xi32>, i32 -> i32
        %19 = tor.addi %17 %18 on (7 to 8) {dump = "op_34", ref_endtime = 37 : i32, ref_starttime = 36 : i32} : (i32, i32) -> i32
        %20 = memref.get_global @mem_c_0 : memref<4xi32> {dump = "op_35", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, starttime = 6 : i32}
        %21 = memref.get_global @mem_c_1 : memref<4xi32> {dump = "op_36", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, starttime = 6 : i32}
        %22 = memref.get_global @mem_c_2 : memref<4xi32> {dump = "op_37", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, starttime = 6 : i32}
        %23 = memref.get_global @mem_c_3 : memref<4xi32> {dump = "op_38", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, starttime = 6 : i32}
        aps.memstore %19, %20[%arg3] {dump = "op_40", endtime = 8 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<4xi32>, i32
        %24 = aps.memload %3[%arg3] {dump = "op_42", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 6 : i32} : memref<4xi32>, i32 -> i32
        %25 = aps.memload %8[%arg3] {dump = "op_44", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 6 : i32} : memref<4xi32>, i32 -> i32
        %26 = tor.addi %24 %25 on (7 to 8) {dump = "op_45", ref_endtime = 37 : i32, ref_starttime = 36 : i32} : (i32, i32) -> i32
        aps.memstore %26, %21[%arg3] {dump = "op_47", endtime = 8 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<4xi32>, i32
        %27 = aps.memload %4[%arg3] {dump = "op_49", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 6 : i32} : memref<4xi32>, i32 -> i32
        %28 = aps.memload %9[%arg3] {dump = "op_51", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 6 : i32} : memref<4xi32>, i32 -> i32
        %29 = tor.addi %27 %28 on (7 to 8) {dump = "op_52", ref_endtime = 37 : i32, ref_starttime = 36 : i32} : (i32, i32) -> i32
        aps.memstore %29, %22[%arg3] {dump = "op_54", endtime = 8 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<4xi32>, i32
        %30 = aps.memload %5[%arg3] {dump = "op_56", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 6 : i32} : memref<4xi32>, i32 -> i32
        %31 = aps.memload %10[%arg3] {dump = "op_58", endtime = 7 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, slot = 0 : i32, starttime = 6 : i32} : memref<4xi32>, i32 -> i32
        %32 = tor.addi %30 %31 on (7 to 8) {dump = "op_59", ref_endtime = 37 : i32, ref_starttime = 36 : i32} : (i32, i32) -> i32
        aps.memstore %32, %23[%arg3] {dump = "op_61", endtime = 8 : i32, ref_endtime = 37 : i32, ref_starttime = 36 : i32, slot = 0 : i32, starttime = 7 : i32} : i32, memref<4xi32>, i32
      } {dump = "op_63", ref_endtime = 37 : i32, ref_starttime = 35 : i32, unroll = 4 : i32}
      %12 = memref.get_global @mem_c_0 : memref<4xi32> {dump = "op_64", endtime = 10 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, starttime = 9 : i32}
      %13 = memref.get_global @mem_c_1 : memref<4xi32> {dump = "op_65", endtime = 10 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, starttime = 9 : i32}
      %14 = memref.get_global @mem_c_2 : memref<4xi32> {dump = "op_66", endtime = 10 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, starttime = 9 : i32}
      %15 = memref.get_global @mem_c_3 : memref<4xi32> {dump = "op_67", endtime = 10 : i32, ref_endtime = 36 : i32, ref_starttime = 35 : i32, starttime = 9 : i32}
      %16 = aps.itfc.burst_store_req(%12, %13, %14, %15) [%c0_i32], %0, %c16_i32 {endtime = 12 : i32, ref_endtime = 53 : i32, ref_starttime = 37 : i32, starttime = 11 : i32} : (memref<4xi32>, memref<4xi32>, memref<4xi32>, memref<4xi32>), i32, i32, i32 -> none
      aps.itfc.burst_store_collect %16 {endtime = 13 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, starttime = 12 : i32} : none
      aps.writerf %arg2, %c42_i32 {dump = "op_69", endtime = 13 : i32, ref_endtime = 54 : i32, ref_starttime = 53 : i32, starttime = 12 : i32} : i5, i32
      tor.return {dump = "op_70"}
    }
  } {HoistConstCondIfOp = 1 : i32, dump = "op_72", schedule = true}
}
```

## First step: Analyze using block handler

You should first segment this with control flow, create op lists for each. In this example, you will find three blocks.

Block 1:
```
%0 = aps.readrf %arg0 {dump = "op_18", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
%1 = aps.readrf %arg1 {dump = "op_19", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32} : i5 -> i32
%2 = memref.get_global @mem_a_0 : memref<4xi32> {dump = "op_20", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
%3 = memref.get_global @mem_a_1 : memref<4xi32> {dump = "op_21", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
%4 = memref.get_global @mem_a_2 : memref<4xi32> {dump = "op_22", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
%5 = memref.get_global @mem_a_3 : memref<4xi32> {dump = "op_23", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
%6 = aps.itfc.burst_load_req %0, (%2, %3, %4, %5) [%c0_i32], %c16_i32 {endtime = 2 : i32, ref_endtime = 17 : i32, ref_starttime = 1 : i32, starttime = 1 : i32} : i32, (memref<4xi32>, memref<4xi32>, memref<4xi32>, memref<4xi32>), i32, i32 -> none
aps.itfc.burst_load_collect %6 {endtime = 3 : i32, ref_endtime = 18 : i32, ref_starttime = 17 : i32, starttime = 2 : i32} : none
%7 = memref.get_global @mem_b_0 : memref<4xi32> {dump = "op_25", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
%8 = memref.get_global @mem_b_1 : memref<4xi32> {dump = "op_26", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
%9 = memref.get_global @mem_b_2 : memref<4xi32> {dump = "op_27", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
%10 = memref.get_global @mem_b_3 : memref<4xi32> {dump = "op_28", endtime = 1 : i32, ref_endtime = 1 : i32, ref_starttime = 0 : i32, starttime = 0 : i32}
%11 = aps.itfc.burst_load_req %1, (%7, %8, %9, %10) [%c0_i32], %c16_i32 {endtime = 4 : i32, ref_endtime = 33 : i32, ref_starttime = 17 : i32, starttime = 2 : i32} : i32, (memref<4xi32>, memref<4xi32>, memref<4xi32>, memref<4xi32>), i32, i32 -> none
aps.itfc.burst_load_collect %11 {endtime = 5 : i32, ref_endtime = 34 : i32, ref_starttime = 33 : i32, starttime = 4 : i32} : none
```

Block 2:

```
tor.for %arg3 = (%c0_i32 : i32) to (%c3_i32 : i32) step (%c1_i32 : i32)
on (5 to 8){
    %17 = aps.memload %2[%arg3]
    %18 = aps.memload %7[%arg3]
    %19 = tor.addi %17 %18
    %20 = memref.get_global @mem_c_0 : memref<4xi32>
    %21 = memref.get_global @mem_c_1 : memref<4xi32>
    %22 = memref.get_global @mem_c_2 : memref<4xi32>
    %23 = memref.get_global @mem_c_3 : memref<4xi32>
    aps.memstore %19, %20[%arg3] 
    %24 = aps.memload %3[%arg3]
    %25 = aps.memload %8[%arg3] 
    %26 = tor.addi %24 %25
    aps.memstore %26, %21[%arg3]
    %27 = aps.memload %4[%arg3] 
    %28 = aps.memload %9[%arg3] 
    %29 = tor.addi %27 %28
    aps.memstore %29, %22[%arg3] 
    %30 = aps.memload %5[%arg3] 
    %31 = aps.memload %10[%arg3] 
    %32 = tor.addi %30 %31
    aps.memstore %32, %23[%arg3]
}
```

Block 3:

```
%12 = memref.get_global @mem_c_0 : memref<4xi32> 
%13 = memref.get_global @mem_c_1 : memref<4xi32>
%14 = memref.get_global @mem_c_2 : memref<4xi32>
%15 = memref.get_global @mem_c_3 : memref<4xi32>
%16 = aps.itfc.burst_store_req(%12, %13, %14, %15) [%c0_i32], %0, %c16_i32 
aps.itfc.burst_store_collect %16 
aps.writerf %arg2, %c42_i32 
tor.return
```

We first define the components of the block handler (both BB, no BB, and loop). We first define cross-block fifos, which are defined as {block i} to {block j} with mlir.Value -> FIFO Instance. This will consists of three (1) input fifos (2) output fifos, both passed as argument to this blockhandler, and (3) those we created inside block. Then, we have token fifo, which is 1-bit value passed across blocks, indicates out status of executing.

Name of all blocks and blocks should be iteratively, like 43_block0_block1, which is the second subblock of first block.

Then, you should analyze data dependency. You will be given input fifos and output fifos, a token_input fifo, and a token_output fifo. For each op in block, you should identify which op is defined at which block, and used in which block (can be multiple). Finally, you will get a map from op to define block and use blocks. 

With these informations, you can create FIFOs, `fifo_b{i}_b{j}_{op}` for cross block data communicate. Also, you are required to insert token fifos between each block, `token_fifo_b{i}_b{i+1}`. You need to pass those fifos (input fifos, output fifos, the fifos we just created) into those blocks, that is, pass as `output_fifo` if this block is block j, `input_fifo` if this is block i. You should also pass token fifo inside, if it's the first block, put the input token of the block passed in as input token, same for output token.

Now, for each block, if it's a basic block, hand over to BBHandler with corresponding ops, otherwise, hand it over to LoopHandler (op should be a scf.for only).

### TODO

(1) Subblock segmentation, using control flow

(2) Data dependency solving: create two mapping, crossBlockEdge: (op, def_block, use_block), inBlockEdge: (op, def_stage, use_stage), using input fifo, output fifo, and local analysis. Create fifo for both, called "fifo_{block_name_def}_{block_name_use}_postfix" (postfix is reserved for redundant), and "fifo_{stage_name_def}_{stage_name_use}_postfix"

(3) Rule generation: pass ops and structures to BBHandler or LoopHandler

## BBHandler

You will be given input fifos, output fifos, token input fifo, and token output fifo. You should use original cross_slot_fifo for internal data management, however.

Analyze dependency using old method, if it's first slot, your data should be acquired from input fifos. The first slot has another task, that is if this value in *block* input fifo, you should deq from that and enq into corresponding *cross_slot_fifo* from stage 0 to the stage in use (one fifo for each target stage). If the value we transformed is in *output_fifo*, you should enq that.

Remeber to add cross_slot_token_info, in first slot, deq from token_input_fifo, and enq to cross_slot_token_info_s0_s1, same for later, and at stage, enq to token_outpu_fifo.

### TODO: 

Please summarize

## LoopHandler

Normally, you will handle a scf.for, which have determined loop boundary, good for you. You should create two rules: entry rule and next rule. You will be given input fifos, output fifos, token input fifo, and token output fifo.

You need to initiate a loop carry fifo, containing loop variable (that i)'s current value, init value, end value, increase value. (could be four fifo actually). You need to enq that initial value in entry rule. You should then handle subblocks using BlockHandler, pass input_fifo (add loop variable with it, as it's also a input for loop body), output_fifo(directly). Create a new token_input_fifo, in entry rule you should deq from the loophandler's token input fifo, and enq to that, which is passed into subblocks as input fifo. Create a token_output_fifo also for the subblock, which triggers the next rule (deq in that!). Next rule should increase/decrease current loop variable and compare it with init value and end value to see if we still meet loop condition. If yes, enq token_input_fifo for subblock, if no, enq token_output_fifo of the *loophandler*. 

Noticing that input_fifos and output_fifos are passed into subblocks, so they are responsible for updating. But currently scf.for has no return value, so output_fifos will not be updated.

### TODO: 

Please summarize

## End

Iteratively do this, you will get a final design.