// IEEE bits enter/leave integer RoCC registers; arithmetic is standalone ISAX IP.
module {
  func.func @flow_fp32_add(%rs1: i5, %rs2: i5, %rd: i5)
      attributes {opcode = 11 : i7, funct7 = 0 : i7} {
    %a_bits = aps.read_irf %rs1 : i5 -> i32
    %b_bits = aps.read_irf %rs2 : i5 -> i32
    %a = arith.bitcast %a_bits : i32 to f32
    %b = arith.bitcast %b_bits : i32 to f32
    %sum = arith.addf %a, %b : f32
    %result = arith.bitcast %sum : f32 to i32
    aps.write_irf %rd, %result : i5, i32
    return
  }
}
