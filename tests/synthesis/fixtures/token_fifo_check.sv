// Formal reference for compiler control tokens: payload is unobservable.
`ifndef DEPTH
`define DEPTH 2
`endif
`ifndef TOKEN_DUT
`define TOKEN_DUT TokenFIFO2_I
`endif
module token_check(input clk, rst, enq_enable, deq_enable, enq_data);
  wire full_ready, full_, enq_ready, deq_ready, deq_;
  `TOKEN_DUT dut(.*);
  reg [1:0] count = 0;
  reg initialized = 0;
  wire pop = deq_enable && count != 0;
  wire ready = (count < `DEPTH) || (`DEPTH == 1 && pop);
  wire push = enq_enable && ready;
  always @(posedge clk) begin
    if (rst) begin
      count <= 0;
      initialized <= 1;
    end else begin
      case ({push, pop})
        2'b10: count <= count + 1;
        2'b01: count <= count - 1;
      endcase
    end
  end
  always @* if (initialized) begin
    assert(count <= `DEPTH);
    assert(full_ready);
    assert(full_ == (count == `DEPTH));
    assert(enq_ready == ready);
    assert(deq_ready == (count != 0));
    if (count != 0) assert(deq_ == 1);
  end
endmodule
