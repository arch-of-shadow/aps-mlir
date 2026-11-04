module {
  memref.global @bias : memref<32xi32> {var_name = "bias", cache_hint = "warm"}
  memref.global @output : memref<32xi32> {var_name = "output"}
  func.func @flow_elision(%rs1: i5, %rs2: i5, %rd: i5) attributes {funct7 = 30 : i32, opcode = 43 : i32} {
    %base = aps.read_irf %rs1 : i5 -> i32
    %dest = aps.read_irf %rs2 : i5 -> i32
    %buffer = memref.get_global @bias : memref<32xi32>
    %out = memref.get_global @output : memref<32xi32>
    %zero = arith.constant 0 : i32
    %count = arith.constant 32 : i32
    %one = arith.constant 1 : i32
    aps.copy %base, (%buffer)[%zero], %count : i32, (memref<32xi32>), i32, i32
    scf.for %i = %zero to %count step %one : i32 {
      %value = aps.read_smem %buffer[%i] : memref<32xi32>, i32 -> i32
      %seven = arith.constant 7 : i32
      %next = arith.addi %value, %seven : i32
      aps.write_smem %next, %out[%i] : i32, memref<32xi32>, i32
    } {pipeline = 1 : i32, II = 2 : i32}
    aps.copy (%out)[%zero], %dest, %count : (memref<32xi32>), i32, i32, i32
    aps.write_irf %rd, %zero : i5, i32
    return
  }
}
