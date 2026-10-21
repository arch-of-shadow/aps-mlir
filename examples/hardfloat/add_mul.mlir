// Two rounding steps; do NOT contract into FMA. Exercises cross-slot live values.
module {
  func.func @flow_fp32_add_mul(%rs1: i5, %rs2: i5, %rd: i5)
      attributes {opcode = 11 : i7, funct7 = 0 : i7} {
    %a_bits = aps.read_irf %rs1 : i5 -> i32
    %b_bits = aps.read_irf %rs2 : i5 -> i32
    %a = arith.bitcast %a_bits : i32 to f32
    %b = arith.bitcast %b_bits : i32 to f32
    %sum = arith.addf %a, %b : f32
    %product = arith.mulf %sum, %a : f32
    %diff = arith.subf %product, %b : f32
    %result = arith.bitcast %diff : f32 to i32
    aps.write_irf %rd, %result : i5, i32
    return
  }
}
