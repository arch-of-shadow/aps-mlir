// Legacy two-operand RNE interface backed by the native three-stage pipeline.
module HF32Mul #( parameter integer LATENCY=3, CAPACITY=5)(
 input wire clock,reset,issue_enable, output wire issue_ready,
 input wire [31:0] operand0,operand1,
 input wire collect_enable, output wire collect_ready,
 output wire [31:0] collect_data, output wire [4:0] collect_flags
);
 HFUnit #(.WIDTH(32),.OP(2),.LATENCY(LATENCY),.CAPACITY(CAPACITY)) impl(
 .clock(clock),.reset(reset),.issue_enable(issue_enable),.issue_ready(issue_ready),
 .operand0(operand0),.operand1(operand1),.operand2(32'b0),.rounding_mode(3'b0),
 .collect_enable(collect_enable),.collect_ready(collect_ready),.collect_data(collect_data),.collect_flags(collect_flags));
endmodule
