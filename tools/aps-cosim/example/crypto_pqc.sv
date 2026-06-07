`ifndef RANDOMIZE
  `ifdef RANDOMIZE_MEM_INIT
    `define RANDOMIZE
  `endif
`endif
`ifndef SYNTHESIS
  `ifndef ENABLE_INITIAL_MEM_
    `define ENABLE_INITIAL_MEM_
  `endif
`endif

`ifndef RANDOM
  `define RANDOM $random
`endif

`ifndef INIT_RANDOM
  `define INIT_RANDOM
`endif

`ifndef RANDOMIZE_DELAY
  `define RANDOMIZE_DELAY 0.002
`endif

`ifndef INIT_RANDOM_PROLOG_
  `ifdef RANDOMIZE
    `ifdef VERILATOR
      `define INIT_RANDOM_PROLOG_ `INIT_RANDOM
    `else
      `define INIT_RANDOM_PROLOG_ `INIT_RANDOM #`RANDOMIZE_DELAY begin end
    `endif
  `else
    `define INIT_RANDOM_PROLOG_
  `endif
`endif

module mymemory_2x16(
  input         R0_addr,
                R0_en,
                R0_clk,
  output [15:0] R0_data,
  input         W0_addr,
                W0_en,
                W0_clk,
  input  [15:0] W0_data
);

  reg [15:0] Memory[0:1];
  reg        _R0_en_d0;
  reg        _R0_addr_d0;
  always @(posedge R0_clk) begin
    _R0_en_d0 <= R0_en;
    _R0_addr_d0 <= R0_addr;
  end
  always @(posedge W0_clk) begin
    if (W0_en)
      Memory[W0_addr] <= W0_data;
  end
  `ifdef ENABLE_INITIAL_MEM_
    reg [31:0] _RANDOM_MEM;
    initial begin
      `INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_MEM_INIT
        for (logic [1:0] i = 2'h0; i < 2'h2; i += 2'h1) begin
          _RANDOM_MEM = `RANDOM;
          Memory[i[0]] = _RANDOM_MEM[15:0];
        end
      `endif
    end
  `endif
  assign R0_data = _R0_en_d0 ? Memory[_R0_addr_d0] : 16'bx;
endmodule

module Mem1r1w1c_AReg_w16_a1_d2(
  input         clock,
                reset,
                ren,
                raddr,
  output [15:0] rdata,
  input         wen,
                waddr,
  input  [15:0] wdata
);

  reg raddr_reg;
  always @(posedge clock) begin
    if (ren)
      raddr_reg <= raddr;
  end
  mymemory_2x16 mymemory_ext (
    .R0_addr (ren ? raddr : raddr_reg),
    .R0_en   (1'h1),
    .R0_clk  (clock),
    .R0_data (rdata),
    .W0_addr (waddr),
    .W0_en   (wen),
    .W0_clk  (clock),
    .W0_data (wdata)
  );
endmodule

module Wire_w1(
  input  write_enable,
  output write_ready,
  input  write_data,
  output read_data,
         read_ready
);

  assign write_ready = 1'h1;
  assign read_data = write_enable & write_data;
  assign read_ready = 1'h1;
endmodule

module Wire_w16(
  input         write_enable,
  output        write_ready,
  input  [15:0] write_data,
  output [15:0] read_data,
  output        read_ready
);

  assign write_ready = 1'h1;
  assign read_data = write_enable ? write_data : 16'h0;
  assign read_ready = 1'h1;
endmodule

module mymemory_8x16(
  input  [2:0]  R0_addr,
  input         R0_en,
                R0_clk,
  output [15:0] R0_data,
  input  [2:0]  W0_addr,
  input         W0_en,
                W0_clk,
  input  [15:0] W0_data
);

  reg [15:0] Memory[0:7];
  reg        _R0_en_d0;
  reg [2:0]  _R0_addr_d0;
  always @(posedge R0_clk) begin
    _R0_en_d0 <= R0_en;
    _R0_addr_d0 <= R0_addr;
  end
  always @(posedge W0_clk) begin
    if (W0_en)
      Memory[W0_addr] <= W0_data;
  end
  `ifdef ENABLE_INITIAL_MEM_
    reg [31:0] _RANDOM_MEM;
    initial begin
      `INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_MEM_INIT
        for (logic [3:0] i = 4'h0; i < 4'h8; i += 4'h1) begin
          _RANDOM_MEM = `RANDOM;
          Memory[i[2:0]] = _RANDOM_MEM[15:0];
        end
      `endif
    end
  `endif
  assign R0_data = _R0_en_d0 ? Memory[_R0_addr_d0] : 16'bx;
endmodule

module Mem1r1w1c_AReg_w16_a3_d8(
  input         clock,
                reset,
                ren,
  input  [2:0]  raddr,
  output [15:0] rdata,
  input         wen,
  input  [2:0]  waddr,
  input  [15:0] wdata
);

  reg [2:0] raddr_reg;
  always @(posedge clock) begin
    if (ren)
      raddr_reg <= raddr;
  end
  mymemory_8x16 mymemory_ext (
    .R0_addr (ren ? raddr : raddr_reg),
    .R0_en   (1'h1),
    .R0_clk  (clock),
    .R0_data (rdata),
    .W0_addr (waddr),
    .W0_en   (wen),
    .W0_clk  (clock),
    .W0_data (wdata)
  );
endmodule

module Wire_w3(
  input        write_enable,
  output       write_ready,
  input  [2:0] write_data,
  output [2:0] read_data,
  output       read_ready
);

  assign write_ready = 1'h1;
  assign read_data = write_enable ? write_data : 3'h0;
  assign read_ready = 1'h1;
endmodule

module mymemory_8x8(
  input  [2:0] R0_addr,
  input        R0_en,
               R0_clk,
  output [7:0] R0_data,
  input  [2:0] W0_addr,
  input        W0_en,
               W0_clk,
  input  [7:0] W0_data
);

  reg [7:0] Memory[0:7];
  reg       _R0_en_d0;
  reg [2:0] _R0_addr_d0;
  always @(posedge R0_clk) begin
    _R0_en_d0 <= R0_en;
    _R0_addr_d0 <= R0_addr;
  end
  always @(posedge W0_clk) begin
    if (W0_en)
      Memory[W0_addr] <= W0_data;
  end
  `ifdef ENABLE_INITIAL_MEM_
    reg [31:0] _RANDOM_MEM;
    initial begin
      `INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_MEM_INIT
        for (logic [3:0] i = 4'h0; i < 4'h8; i += 4'h1) begin
          _RANDOM_MEM = `RANDOM;
          Memory[i[2:0]] = _RANDOM_MEM[7:0];
        end
      `endif
    end
  `endif
  assign R0_data = _R0_en_d0 ? Memory[_R0_addr_d0] : 8'bx;
endmodule

module Mem1r1w1c_AReg_w8_a3_d8(
  input        clock,
               reset,
               ren,
  input  [2:0] raddr,
  output [7:0] rdata,
  input        wen,
  input  [2:0] waddr,
  input  [7:0] wdata
);

  reg [2:0] raddr_reg;
  always @(posedge clock) begin
    if (ren)
      raddr_reg <= raddr;
  end
  mymemory_8x8 mymemory_ext (
    .R0_addr (ren ? raddr : raddr_reg),
    .R0_en   (1'h1),
    .R0_clk  (clock),
    .R0_data (rdata),
    .W0_addr (waddr),
    .W0_en   (wen),
    .W0_clk  (clock),
    .W0_data (wdata)
  );
endmodule

module mymemory_4x16(
  input  [1:0]  R0_addr,
  input         R0_en,
                R0_clk,
  output [15:0] R0_data,
  input  [1:0]  W0_addr,
  input         W0_en,
                W0_clk,
  input  [15:0] W0_data
);

  reg [15:0] Memory[0:3];
  reg        _R0_en_d0;
  reg [1:0]  _R0_addr_d0;
  always @(posedge R0_clk) begin
    _R0_en_d0 <= R0_en;
    _R0_addr_d0 <= R0_addr;
  end
  always @(posedge W0_clk) begin
    if (W0_en)
      Memory[W0_addr] <= W0_data;
  end
  `ifdef ENABLE_INITIAL_MEM_
    reg [31:0] _RANDOM_MEM;
    initial begin
      `INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_MEM_INIT
        for (logic [2:0] i = 3'h0; i < 3'h4; i += 3'h1) begin
          _RANDOM_MEM = `RANDOM;
          Memory[i[1:0]] = _RANDOM_MEM[15:0];
        end
      `endif
    end
  `endif
  assign R0_data = _R0_en_d0 ? Memory[_R0_addr_d0] : 16'bx;
endmodule

module Mem1r1w1c_AReg_w16_a2_d4(
  input         clock,
                reset,
                ren,
  input  [1:0]  raddr,
  output [15:0] rdata,
  input         wen,
  input  [1:0]  waddr,
  input  [15:0] wdata
);

  reg [1:0] raddr_reg;
  always @(posedge clock) begin
    if (ren)
      raddr_reg <= raddr;
  end
  mymemory_4x16 mymemory_ext (
    .R0_addr (ren ? raddr : raddr_reg),
    .R0_en   (1'h1),
    .R0_clk  (clock),
    .R0_data (rdata),
    .W0_addr (waddr),
    .W0_en   (wen),
    .W0_clk  (clock),
    .W0_data (wdata)
  );
endmodule

module Wire_w2(
  input        write_enable,
  output       write_ready,
  input  [1:0] write_data,
  output [1:0] read_data,
  output       read_ready
);

  assign write_ready = 1'h1;
  assign read_data = write_enable ? write_data : 2'h0;
  assign read_ready = 1'h1;
endmodule

module mymemory_8x64(
  input  [2:0]  R0_addr,
  input         R0_en,
                R0_clk,
  output [63:0] R0_data,
  input  [2:0]  W0_addr,
  input         W0_en,
                W0_clk,
  input  [63:0] W0_data
);

  reg [63:0] Memory[0:7];
  reg        _R0_en_d0;
  reg [2:0]  _R0_addr_d0;
  always @(posedge R0_clk) begin
    _R0_en_d0 <= R0_en;
    _R0_addr_d0 <= R0_addr;
  end
  always @(posedge W0_clk) begin
    if (W0_en)
      Memory[W0_addr] <= W0_data;
  end
  `ifdef ENABLE_INITIAL_MEM_
    reg [63:0] _RANDOM_MEM;
    initial begin
      `INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_MEM_INIT
        for (logic [3:0] i = 4'h0; i < 4'h8; i += 4'h1) begin
          for (logic [6:0] j = 7'h0; j < 7'h40; j += 7'h20) begin
            _RANDOM_MEM[j[5:0] +: 32] = `RANDOM;
          end
          Memory[i[2:0]] = _RANDOM_MEM;
        end
      `endif
    end
  `endif
  assign R0_data = _R0_en_d0 ? Memory[_R0_addr_d0] : 64'bx;
endmodule

module Mem1r1w1c_AReg_w64_a3_d8(
  input         clock,
                reset,
                ren,
  input  [2:0]  raddr,
  output [63:0] rdata,
  input         wen,
  input  [2:0]  waddr,
  input  [63:0] wdata
);

  reg [2:0] raddr_reg;
  always @(posedge clock) begin
    if (ren)
      raddr_reg <= raddr;
  end
  mymemory_8x64 mymemory_ext (
    .R0_addr (ren ? raddr : raddr_reg),
    .R0_en   (1'h1),
    .R0_clk  (clock),
    .R0_data (rdata),
    .W0_addr (waddr),
    .W0_en   (wen),
    .W0_clk  (clock),
    .W0_data (wdata)
  );
endmodule

module Wire_w64(
  input         write_enable,
  output        write_ready,
  input  [63:0] write_data,
  output [63:0] read_data,
  output        read_ready
);

  assign write_ready = 1'h1;
  assign read_data = write_enable ? write_data : 64'h0;
  assign read_ready = 1'h1;
endmodule

module mymemory_2x32(
  input         R0_addr,
                R0_en,
                R0_clk,
  output [31:0] R0_data,
  input         W0_addr,
                W0_en,
                W0_clk,
  input  [31:0] W0_data
);

  reg [31:0] Memory[0:1];
  reg        _R0_en_d0;
  reg        _R0_addr_d0;
  always @(posedge R0_clk) begin
    _R0_en_d0 <= R0_en;
    _R0_addr_d0 <= R0_addr;
  end
  always @(posedge W0_clk) begin
    if (W0_en)
      Memory[W0_addr] <= W0_data;
  end
  `ifdef ENABLE_INITIAL_MEM_
    reg [31:0] _RANDOM_MEM;
    initial begin
      `INIT_RANDOM_PROLOG_
      `ifdef RANDOMIZE_MEM_INIT
        for (logic [1:0] i = 2'h0; i < 2'h2; i += 2'h1) begin
          _RANDOM_MEM = `RANDOM;
          Memory[i[0]] = _RANDOM_MEM;
        end
      `endif
    end
  `endif
  assign R0_data = _R0_en_d0 ? Memory[_R0_addr_d0] : 32'bx;
endmodule

module Mem1r1w1c_AReg_w32_a1_d2(
  input         clock,
                reset,
                ren,
                raddr,
  output [31:0] rdata,
  input         wen,
                waddr,
  input  [31:0] wdata
);

  reg raddr_reg;
  always @(posedge clock) begin
    if (ren)
      raddr_reg <= raddr;
  end
  mymemory_2x32 mymemory_ext (
    .R0_addr (ren ? raddr : raddr_reg),
    .R0_en   (1'h1),
    .R0_clk  (clock),
    .R0_data (rdata),
    .W0_addr (waddr),
    .W0_en   (wen),
    .W0_clk  (clock),
    .W0_data (wdata)
  );
endmodule

module Wire_w32(
  input         write_enable,
  output        write_ready,
  input  [31:0] write_data,
  output [31:0] read_data,
  output        read_ready
);

  assign write_ready = 1'h1;
  assign read_data = write_enable ? write_data : 32'h0;
  assign read_ready = 1'h1;
endmodule

module Reg_width32_init0(
  input         clock,
                reset,
                write_enable,
  input  [31:0] write_data,
  output        read_ready,
  output [31:0] read_data,
  output        write_ready
);

  reg [31:0] reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 32'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module Reg_width96_init0(
  input         clock,
                reset,
                write_enable,
  input  [95:0] write_data,
  output        read_ready,
  output [95:0] read_data,
  output        write_ready
);

  reg [95:0] reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 96'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module Reg_width1_init0(
  input  clock,
         reset,
         write_enable,
         write_data,
  output read_ready,
         read_data,
         write_ready
);

  reg reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 1'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module Reg_width37_init0(
  input         clock,
                reset,
                write_enable,
  input  [36:0] write_data,
  output        read_ready,
  output [36:0] read_data,
  output        write_ready
);

  reg [36:0] reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 37'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module Reg_width85_init0(
  input         clock,
                reset,
                write_enable,
  input  [84:0] write_data,
  output        read_ready,
  output [84:0] read_data,
  output        write_ready
);

  reg [84:0] reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 85'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module Wire_w85(
  input         write_enable,
  output        write_ready,
  input  [84:0] write_data,
  output [84:0] read_data,
  output        read_ready
);

  assign write_ready = 1'h1;
  assign read_data = write_enable ? write_data : 85'h0;
  assign read_ready = 1'h1;
endmodule

module Reg_width2_init0(
  input        clock,
               reset,
               write_enable,
  input  [1:0] write_data,
  output       read_ready,
  output [1:0] read_data,
  output       write_ready
);

  reg [1:0] reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 2'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module Reg_width8_init0(
  input        clock,
               reset,
               write_enable,
  input  [7:0] write_data,
  output       read_ready,
  output [7:0] read_data,
  output       write_ready
);

  reg [7:0] reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 8'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module Reg_width16_init0(
  input         clock,
                reset,
                write_enable,
  input  [15:0] write_data,
  output        read_ready,
  output [15:0] read_data,
  output        write_ready
);

  reg [15:0] reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 16'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module Reg_width5_init0(
  input        clock,
               reset,
               write_enable,
  input  [4:0] write_data,
  output       read_ready,
  output [4:0] read_data,
  output       write_ready
);

  reg [4:0] reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 5'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module Reg_width64_init0(
  input         clock,
                reset,
                write_enable,
  input  [63:0] write_data,
  output        read_ready,
  output [63:0] read_data,
  output        write_ready
);

  reg [63:0] reg_0;
  always @(posedge clock) begin
    if (reset)
      reg_0 <= 64'h0;
    else if (write_enable)
      reg_0 <= write_data;
  end
  assign read_ready = 1'h1;
  assign read_data = reg_0;
  assign write_ready = 1'h1;
endmodule

module WireDefault_w1_i0(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_acc_0(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input         bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input         bank_write_addr,
  input  [15:0] bank_write_data
);

  wire        _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire        _GEN_0 = ~burst_read_0_enable & bank_read_0_enable;
  wire [31:0] _GEN_1 = burst_write_addr / 32'h2;
  wire        _GEN_2 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a1_d2 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_0 | burst_read_0_enable),
    .raddr (_GEN_0 ? bank_read_0_addr : burst_read_0_enable & _GEN[0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_2 | _write_enable_wire_read_res0),
    .waddr
      (_GEN_2
         ? bank_write_addr
         : _write_enable_wire_read_res0 & _write_addr_wire_read_data),
    .wdata (_GEN_2 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i0 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w1 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (burst_write_enable & _GEN_1[0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 = {48'h0, _mem_bank_rdata};
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module memory_acc(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_0_enable,
  output        bank_read_0_0_ready,
  input         bank_read_0_0_addr,
  output        bank_read_1_0_ready,
  output [15:0] bank_read_1_0_res0,
  input         bank_write_0_enable,
  output        bank_write_0_ready,
  input         bank_write_0_addr,
  input  [15:0] bank_write_0_data
);

  wire _bank_wrap_0_burst_write_ready;
  wire _bank_wrap_0_bank_read_0_ready;
  wire _bank_wrap_0_bank_write_ready;
  wire _GEN = _bank_wrap_0_bank_read_0_ready & bank_read_0_0_enable;
  wire _GEN_0 = _bank_wrap_0_bank_write_ready & bank_write_0_enable;
  BankWrapper_acc_0 bank_wrap_0 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (burst_read_1_res0),
    .burst_write_enable  (_bank_wrap_0_burst_write_ready & burst_write_enable),
    .burst_write_ready   (_bank_wrap_0_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_GEN),
    .bank_read_0_ready   (_bank_wrap_0_bank_read_0_ready),
    .bank_read_0_addr    (_GEN & bank_read_0_0_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_0_res0),
    .bank_write_enable   (_GEN_0),
    .bank_write_ready    (_bank_wrap_0_bank_write_ready),
    .bank_write_addr     (_GEN_0 & bank_write_0_addr),
    .bank_write_data     (bank_write_0_data)
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_write_ready = _bank_wrap_0_burst_write_ready;
  assign bank_read_0_0_ready = _bank_wrap_0_bank_read_0_ready;
  assign bank_read_1_0_ready = 1'h1;
  assign bank_write_0_ready = _bank_wrap_0_bank_write_ready;
endmodule

module WireDefault_w1_i00(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_decompressed_weights_0(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [2:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [2:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [2:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h0 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h0 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h0 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a3_d8 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[2:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i00 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w3 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[2:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i01(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_decompressed_weights_1(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [2:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [2:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [2:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h1 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h1 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h1 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a3_d8 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[2:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i01 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w3 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[2:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i02(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_decompressed_weights_2(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [2:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [2:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [2:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h2 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h2 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h2 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a3_d8 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[2:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i02 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w3 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[2:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i03(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_decompressed_weights_3(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [2:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [2:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [2:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h3 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h3 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h3 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a3_d8 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[2:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i03 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w3 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[2:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module memory_decompressed_weights(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_0_enable,
  output        bank_read_0_0_ready,
  input  [2:0]  bank_read_0_0_addr,
  output        bank_read_1_0_ready,
  output [15:0] bank_read_1_0_res0,
  input         bank_write_0_enable,
  output        bank_write_0_ready,
  input  [2:0]  bank_write_0_addr,
  input  [15:0] bank_write_0_data,
  input         bank_read_0_1_enable,
  output        bank_read_0_1_ready,
  input  [2:0]  bank_read_0_1_addr,
  output        bank_read_1_1_ready,
  output [15:0] bank_read_1_1_res0,
  input         bank_write_1_enable,
  output        bank_write_1_ready,
  input  [2:0]  bank_write_1_addr,
  input  [15:0] bank_write_1_data,
  input         bank_read_0_2_enable,
  output        bank_read_0_2_ready,
  input  [2:0]  bank_read_0_2_addr,
  output        bank_read_1_2_ready,
  output [15:0] bank_read_1_2_res0,
  input         bank_write_2_enable,
  output        bank_write_2_ready,
  input  [2:0]  bank_write_2_addr,
  input  [15:0] bank_write_2_data,
  input         bank_read_0_3_enable,
  output        bank_read_0_3_ready,
  input  [2:0]  bank_read_0_3_addr,
  output        bank_read_1_3_ready,
  output [15:0] bank_read_1_3_res0,
  input         bank_write_3_enable,
  output        bank_write_3_ready,
  input  [2:0]  bank_write_3_addr,
  input  [15:0] bank_write_3_data
);

  wire [63:0] _bank_wrap_3_burst_read_1_res0;
  wire        _bank_wrap_3_burst_write_ready;
  wire        _bank_wrap_3_bank_read_0_ready;
  wire        _bank_wrap_3_bank_write_ready;
  wire [63:0] _bank_wrap_2_burst_read_1_res0;
  wire        _bank_wrap_2_burst_write_ready;
  wire        _bank_wrap_2_bank_read_0_ready;
  wire        _bank_wrap_2_bank_write_ready;
  wire [63:0] _bank_wrap_1_burst_read_1_res0;
  wire        _bank_wrap_1_burst_write_ready;
  wire        _bank_wrap_1_bank_read_0_ready;
  wire        _bank_wrap_1_bank_write_ready;
  wire [63:0] _bank_wrap_0_burst_read_1_res0;
  wire        _bank_wrap_0_burst_write_ready;
  wire        _bank_wrap_0_bank_read_0_ready;
  wire        _bank_wrap_0_bank_write_ready;
  wire        _GEN =
    _bank_wrap_0_burst_write_ready & _bank_wrap_1_burst_write_ready
    & _bank_wrap_2_burst_write_ready & _bank_wrap_3_burst_write_ready;
  wire        _GEN_0 = _GEN & burst_write_enable;
  BankWrapper_decompressed_weights_0 bank_wrap_0 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_0_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_0_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_0_bank_read_0_ready & bank_read_0_0_enable),
    .bank_read_0_ready   (_bank_wrap_0_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_0_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_0_res0),
    .bank_write_enable   (_bank_wrap_0_bank_write_ready & bank_write_0_enable),
    .bank_write_ready    (_bank_wrap_0_bank_write_ready),
    .bank_write_addr     (bank_write_0_addr),
    .bank_write_data     (bank_write_0_data)
  );
  BankWrapper_decompressed_weights_1 bank_wrap_1 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_1_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_1_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_1_bank_read_0_ready & bank_read_0_1_enable),
    .bank_read_0_ready   (_bank_wrap_1_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_1_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_1_res0),
    .bank_write_enable   (_bank_wrap_1_bank_write_ready & bank_write_1_enable),
    .bank_write_ready    (_bank_wrap_1_bank_write_ready),
    .bank_write_addr     (bank_write_1_addr),
    .bank_write_data     (bank_write_1_data)
  );
  BankWrapper_decompressed_weights_2 bank_wrap_2 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_2_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_2_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_2_bank_read_0_ready & bank_read_0_2_enable),
    .bank_read_0_ready   (_bank_wrap_2_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_2_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_2_res0),
    .bank_write_enable   (_bank_wrap_2_bank_write_ready & bank_write_2_enable),
    .bank_write_ready    (_bank_wrap_2_bank_write_ready),
    .bank_write_addr     (bank_write_2_addr),
    .bank_write_data     (bank_write_2_data)
  );
  BankWrapper_decompressed_weights_3 bank_wrap_3 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_3_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_3_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_3_bank_read_0_ready & bank_read_0_3_enable),
    .bank_read_0_ready   (_bank_wrap_3_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_3_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_3_res0),
    .bank_write_enable   (_bank_wrap_3_bank_write_ready & bank_write_3_enable),
    .bank_write_ready    (_bank_wrap_3_bank_write_ready),
    .bank_write_addr     (bank_write_3_addr),
    .bank_write_data     (bank_write_3_data)
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _bank_wrap_0_burst_read_1_res0 | _bank_wrap_1_burst_read_1_res0
    | _bank_wrap_2_burst_read_1_res0 | _bank_wrap_3_burst_read_1_res0;
  assign burst_write_ready = _GEN;
  assign bank_read_0_0_ready = _bank_wrap_0_bank_read_0_ready;
  assign bank_read_1_0_ready = 1'h1;
  assign bank_write_0_ready = _bank_wrap_0_bank_write_ready;
  assign bank_read_0_1_ready = _bank_wrap_1_bank_read_0_ready;
  assign bank_read_1_1_ready = 1'h1;
  assign bank_write_1_ready = _bank_wrap_1_bank_write_ready;
  assign bank_read_0_2_ready = _bank_wrap_2_bank_read_0_ready;
  assign bank_read_1_2_ready = 1'h1;
  assign bank_write_2_ready = _bank_wrap_2_bank_write_ready;
  assign bank_read_0_3_ready = _bank_wrap_3_bank_read_0_ready;
  assign bank_read_1_3_ready = 1'h1;
  assign bank_write_3_ready = _bank_wrap_3_bank_write_ready;
endmodule

module BankWrapper_dense_values_0(
  input        clk,
               rst,
               bank_read_0_enable,
  output       bank_read_0_ready,
  input  [2:0] bank_read_0_addr,
  output       bank_read_1_ready,
  output [7:0] bank_read_1_res0,
  input        bank_write_enable,
  output       bank_write_ready,
  input  [2:0] bank_write_addr,
  input  [7:0] bank_write_data
);

  Mem1r1w1c_AReg_w8_a3_d8 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (bank_read_0_enable),
    .raddr (bank_read_0_addr),
    .rdata (bank_read_1_res0),
    .wen   (bank_write_enable),
    .waddr (bank_write_addr),
    .wdata (bank_write_data)
  );
  assign bank_read_0_ready = 1'h1;
  assign bank_read_1_ready = 1'h1;
  assign bank_write_ready = 1'h1;
endmodule

module BankWrapper_dense_values_1(
  input        clk,
               rst,
               bank_read_0_enable,
  output       bank_read_0_ready,
  input  [2:0] bank_read_0_addr,
  output       bank_read_1_ready,
  output [7:0] bank_read_1_res0,
  input        bank_write_enable,
  output       bank_write_ready,
  input  [2:0] bank_write_addr,
  input  [7:0] bank_write_data
);

  Mem1r1w1c_AReg_w8_a3_d8 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (bank_read_0_enable),
    .raddr (bank_read_0_addr),
    .rdata (bank_read_1_res0),
    .wen   (bank_write_enable),
    .waddr (bank_write_addr),
    .wdata (bank_write_data)
  );
  assign bank_read_0_ready = 1'h1;
  assign bank_read_1_ready = 1'h1;
  assign bank_write_ready = 1'h1;
endmodule

module BankWrapper_dense_values_2(
  input        clk,
               rst,
               bank_read_0_enable,
  output       bank_read_0_ready,
  input  [2:0] bank_read_0_addr,
  output       bank_read_1_ready,
  output [7:0] bank_read_1_res0,
  input        bank_write_enable,
  output       bank_write_ready,
  input  [2:0] bank_write_addr,
  input  [7:0] bank_write_data
);

  Mem1r1w1c_AReg_w8_a3_d8 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (bank_read_0_enable),
    .raddr (bank_read_0_addr),
    .rdata (bank_read_1_res0),
    .wen   (bank_write_enable),
    .waddr (bank_write_addr),
    .wdata (bank_write_data)
  );
  assign bank_read_0_ready = 1'h1;
  assign bank_read_1_ready = 1'h1;
  assign bank_write_ready = 1'h1;
endmodule

module BankWrapper_dense_values_3(
  input        clk,
               rst,
               bank_read_0_enable,
  output       bank_read_0_ready,
  input  [2:0] bank_read_0_addr,
  output       bank_read_1_ready,
  output [7:0] bank_read_1_res0,
  input        bank_write_enable,
  output       bank_write_ready,
  input  [2:0] bank_write_addr,
  input  [7:0] bank_write_data
);

  Mem1r1w1c_AReg_w8_a3_d8 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (bank_read_0_enable),
    .raddr (bank_read_0_addr),
    .rdata (bank_read_1_res0),
    .wen   (bank_write_enable),
    .waddr (bank_write_addr),
    .wdata (bank_write_data)
  );
  assign bank_read_0_ready = 1'h1;
  assign bank_read_1_ready = 1'h1;
  assign bank_write_ready = 1'h1;
endmodule

module memory_dense_values(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_0_enable,
  output        bank_read_0_0_ready,
  input  [2:0]  bank_read_0_0_addr,
  output        bank_read_1_0_ready,
  output [7:0]  bank_read_1_0_res0,
  input         bank_write_0_enable,
  output        bank_write_0_ready,
  input  [2:0]  bank_write_0_addr,
  input  [7:0]  bank_write_0_data,
  input         bank_read_0_1_enable,
  output        bank_read_0_1_ready,
  input  [2:0]  bank_read_0_1_addr,
  output        bank_read_1_1_ready,
  output [7:0]  bank_read_1_1_res0,
  input         bank_write_1_enable,
  output        bank_write_1_ready,
  input  [2:0]  bank_write_1_addr,
  input  [7:0]  bank_write_1_data,
  input         bank_read_0_2_enable,
  output        bank_read_0_2_ready,
  input  [2:0]  bank_read_0_2_addr,
  output        bank_read_1_2_ready,
  output [7:0]  bank_read_1_2_res0,
  input         bank_write_2_enable,
  output        bank_write_2_ready,
  input  [2:0]  bank_write_2_addr,
  input  [7:0]  bank_write_2_data,
  input         bank_read_0_3_enable,
  output        bank_read_0_3_ready,
  input  [2:0]  bank_read_0_3_addr,
  output        bank_read_1_3_ready,
  output [7:0]  bank_read_1_3_res0,
  input         bank_write_3_enable,
  output        bank_write_3_ready,
  input  [2:0]  bank_write_3_addr,
  input  [7:0]  bank_write_3_data
);

  BankWrapper_dense_values_0 bank_wrap_0 (
    .clk                (clk),
    .rst                (rst),
    .bank_read_0_enable (bank_read_0_0_enable),
    .bank_read_0_ready  (),
    .bank_read_0_addr   (bank_read_0_0_addr),
    .bank_read_1_ready  (),
    .bank_read_1_res0   (bank_read_1_0_res0),
    .bank_write_enable  (bank_write_0_enable),
    .bank_write_ready   (),
    .bank_write_addr    (bank_write_0_addr),
    .bank_write_data    (bank_write_0_data)
  );
  BankWrapper_dense_values_1 bank_wrap_1 (
    .clk                (clk),
    .rst                (rst),
    .bank_read_0_enable (bank_read_0_1_enable),
    .bank_read_0_ready  (),
    .bank_read_0_addr   (bank_read_0_1_addr),
    .bank_read_1_ready  (),
    .bank_read_1_res0   (bank_read_1_1_res0),
    .bank_write_enable  (bank_write_1_enable),
    .bank_write_ready   (),
    .bank_write_addr    (bank_write_1_addr),
    .bank_write_data    (bank_write_1_data)
  );
  BankWrapper_dense_values_2 bank_wrap_2 (
    .clk                (clk),
    .rst                (rst),
    .bank_read_0_enable (bank_read_0_2_enable),
    .bank_read_0_ready  (),
    .bank_read_0_addr   (bank_read_0_2_addr),
    .bank_read_1_ready  (),
    .bank_read_1_res0   (bank_read_1_2_res0),
    .bank_write_enable  (bank_write_2_enable),
    .bank_write_ready   (),
    .bank_write_addr    (bank_write_2_addr),
    .bank_write_data    (bank_write_2_data)
  );
  BankWrapper_dense_values_3 bank_wrap_3 (
    .clk                (clk),
    .rst                (rst),
    .bank_read_0_enable (bank_read_0_3_enable),
    .bank_read_0_ready  (),
    .bank_read_0_addr   (bank_read_0_3_addr),
    .bank_read_1_ready  (),
    .bank_read_1_res0   (bank_read_1_3_res0),
    .bank_write_enable  (bank_write_3_enable),
    .bank_write_ready   (),
    .bank_write_addr    (bank_write_3_addr),
    .bank_write_data    (bank_write_3_data)
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 = 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_0_ready = 1'h1;
  assign bank_read_1_0_ready = 1'h1;
  assign bank_write_0_ready = 1'h1;
  assign bank_read_0_1_ready = 1'h1;
  assign bank_read_1_1_ready = 1'h1;
  assign bank_write_1_ready = 1'h1;
  assign bank_read_0_2_ready = 1'h1;
  assign bank_read_1_2_ready = 1'h1;
  assign bank_write_2_ready = 1'h1;
  assign bank_read_0_3_ready = 1'h1;
  assign bank_read_1_3_ready = 1'h1;
  assign bank_write_3_ready = 1'h1;
endmodule

module WireDefault_w1_i04(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_a_0(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h0 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h0 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h0 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i04 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i05(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_a_1(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h1 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h1 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h1 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i05 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i06(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_a_2(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h2 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h2 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h2 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i06 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i07(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_a_3(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h3 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h3 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h3 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i07 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module memory_matrix_a(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_0_enable,
  output        bank_read_0_0_ready,
  input  [1:0]  bank_read_0_0_addr,
  output        bank_read_1_0_ready,
  output [15:0] bank_read_1_0_res0,
  input         bank_write_0_enable,
  output        bank_write_0_ready,
  input  [1:0]  bank_write_0_addr,
  input  [15:0] bank_write_0_data,
  input         bank_read_0_1_enable,
  output        bank_read_0_1_ready,
  input  [1:0]  bank_read_0_1_addr,
  output        bank_read_1_1_ready,
  output [15:0] bank_read_1_1_res0,
  input         bank_write_1_enable,
  output        bank_write_1_ready,
  input  [1:0]  bank_write_1_addr,
  input  [15:0] bank_write_1_data,
  input         bank_read_0_2_enable,
  output        bank_read_0_2_ready,
  input  [1:0]  bank_read_0_2_addr,
  output        bank_read_1_2_ready,
  output [15:0] bank_read_1_2_res0,
  input         bank_write_2_enable,
  output        bank_write_2_ready,
  input  [1:0]  bank_write_2_addr,
  input  [15:0] bank_write_2_data,
  input         bank_read_0_3_enable,
  output        bank_read_0_3_ready,
  input  [1:0]  bank_read_0_3_addr,
  output        bank_read_1_3_ready,
  output [15:0] bank_read_1_3_res0,
  input         bank_write_3_enable,
  output        bank_write_3_ready,
  input  [1:0]  bank_write_3_addr,
  input  [15:0] bank_write_3_data
);

  wire [63:0] _bank_wrap_3_burst_read_1_res0;
  wire        _bank_wrap_3_burst_write_ready;
  wire        _bank_wrap_3_bank_read_0_ready;
  wire        _bank_wrap_3_bank_write_ready;
  wire [63:0] _bank_wrap_2_burst_read_1_res0;
  wire        _bank_wrap_2_burst_write_ready;
  wire        _bank_wrap_2_bank_read_0_ready;
  wire        _bank_wrap_2_bank_write_ready;
  wire [63:0] _bank_wrap_1_burst_read_1_res0;
  wire        _bank_wrap_1_burst_write_ready;
  wire        _bank_wrap_1_bank_read_0_ready;
  wire        _bank_wrap_1_bank_write_ready;
  wire [63:0] _bank_wrap_0_burst_read_1_res0;
  wire        _bank_wrap_0_burst_write_ready;
  wire        _bank_wrap_0_bank_read_0_ready;
  wire        _bank_wrap_0_bank_write_ready;
  wire        _GEN =
    _bank_wrap_0_burst_write_ready & _bank_wrap_1_burst_write_ready
    & _bank_wrap_2_burst_write_ready & _bank_wrap_3_burst_write_ready;
  wire        _GEN_0 = _GEN & burst_write_enable;
  BankWrapper_matrix_a_0 bank_wrap_0 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_0_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_0_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_0_bank_read_0_ready & bank_read_0_0_enable),
    .bank_read_0_ready   (_bank_wrap_0_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_0_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_0_res0),
    .bank_write_enable   (_bank_wrap_0_bank_write_ready & bank_write_0_enable),
    .bank_write_ready    (_bank_wrap_0_bank_write_ready),
    .bank_write_addr     (bank_write_0_addr),
    .bank_write_data     (bank_write_0_data)
  );
  BankWrapper_matrix_a_1 bank_wrap_1 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_1_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_1_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_1_bank_read_0_ready & bank_read_0_1_enable),
    .bank_read_0_ready   (_bank_wrap_1_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_1_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_1_res0),
    .bank_write_enable   (_bank_wrap_1_bank_write_ready & bank_write_1_enable),
    .bank_write_ready    (_bank_wrap_1_bank_write_ready),
    .bank_write_addr     (bank_write_1_addr),
    .bank_write_data     (bank_write_1_data)
  );
  BankWrapper_matrix_a_2 bank_wrap_2 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_2_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_2_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_2_bank_read_0_ready & bank_read_0_2_enable),
    .bank_read_0_ready   (_bank_wrap_2_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_2_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_2_res0),
    .bank_write_enable   (_bank_wrap_2_bank_write_ready & bank_write_2_enable),
    .bank_write_ready    (_bank_wrap_2_bank_write_ready),
    .bank_write_addr     (bank_write_2_addr),
    .bank_write_data     (bank_write_2_data)
  );
  BankWrapper_matrix_a_3 bank_wrap_3 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_3_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_3_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_3_bank_read_0_ready & bank_read_0_3_enable),
    .bank_read_0_ready   (_bank_wrap_3_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_3_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_3_res0),
    .bank_write_enable   (_bank_wrap_3_bank_write_ready & bank_write_3_enable),
    .bank_write_ready    (_bank_wrap_3_bank_write_ready),
    .bank_write_addr     (bank_write_3_addr),
    .bank_write_data     (bank_write_3_data)
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _bank_wrap_0_burst_read_1_res0 | _bank_wrap_1_burst_read_1_res0
    | _bank_wrap_2_burst_read_1_res0 | _bank_wrap_3_burst_read_1_res0;
  assign burst_write_ready = _GEN;
  assign bank_read_0_0_ready = _bank_wrap_0_bank_read_0_ready;
  assign bank_read_1_0_ready = 1'h1;
  assign bank_write_0_ready = _bank_wrap_0_bank_write_ready;
  assign bank_read_0_1_ready = _bank_wrap_1_bank_read_0_ready;
  assign bank_read_1_1_ready = 1'h1;
  assign bank_write_1_ready = _bank_wrap_1_bank_write_ready;
  assign bank_read_0_2_ready = _bank_wrap_2_bank_read_0_ready;
  assign bank_read_1_2_ready = 1'h1;
  assign bank_write_2_ready = _bank_wrap_2_bank_write_ready;
  assign bank_read_0_3_ready = _bank_wrap_3_bank_read_0_ready;
  assign bank_read_1_3_ready = 1'h1;
  assign bank_write_3_ready = _bank_wrap_3_bank_write_ready;
endmodule

module WireDefault_w1_i08(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_b_0(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h0 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h0 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h0 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i08 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i09(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_b_1(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h1 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h1 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h1 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i09 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i010(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_b_2(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h2 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h2 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h2 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i010 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i011(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_b_3(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h3 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h3 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h3 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i011 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module memory_matrix_b(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_0_enable,
  output        bank_read_0_0_ready,
  input  [1:0]  bank_read_0_0_addr,
  output        bank_read_1_0_ready,
  output [15:0] bank_read_1_0_res0,
  input         bank_write_0_enable,
  output        bank_write_0_ready,
  input  [1:0]  bank_write_0_addr,
  input  [15:0] bank_write_0_data,
  input         bank_read_0_1_enable,
  output        bank_read_0_1_ready,
  input  [1:0]  bank_read_0_1_addr,
  output        bank_read_1_1_ready,
  output [15:0] bank_read_1_1_res0,
  input         bank_write_1_enable,
  output        bank_write_1_ready,
  input  [1:0]  bank_write_1_addr,
  input  [15:0] bank_write_1_data,
  input         bank_read_0_2_enable,
  output        bank_read_0_2_ready,
  input  [1:0]  bank_read_0_2_addr,
  output        bank_read_1_2_ready,
  output [15:0] bank_read_1_2_res0,
  input         bank_write_2_enable,
  output        bank_write_2_ready,
  input  [1:0]  bank_write_2_addr,
  input  [15:0] bank_write_2_data,
  input         bank_read_0_3_enable,
  output        bank_read_0_3_ready,
  input  [1:0]  bank_read_0_3_addr,
  output        bank_read_1_3_ready,
  output [15:0] bank_read_1_3_res0,
  input         bank_write_3_enable,
  output        bank_write_3_ready,
  input  [1:0]  bank_write_3_addr,
  input  [15:0] bank_write_3_data
);

  wire [63:0] _bank_wrap_3_burst_read_1_res0;
  wire        _bank_wrap_3_burst_write_ready;
  wire        _bank_wrap_3_bank_read_0_ready;
  wire        _bank_wrap_3_bank_write_ready;
  wire [63:0] _bank_wrap_2_burst_read_1_res0;
  wire        _bank_wrap_2_burst_write_ready;
  wire        _bank_wrap_2_bank_read_0_ready;
  wire        _bank_wrap_2_bank_write_ready;
  wire [63:0] _bank_wrap_1_burst_read_1_res0;
  wire        _bank_wrap_1_burst_write_ready;
  wire        _bank_wrap_1_bank_read_0_ready;
  wire        _bank_wrap_1_bank_write_ready;
  wire [63:0] _bank_wrap_0_burst_read_1_res0;
  wire        _bank_wrap_0_burst_write_ready;
  wire        _bank_wrap_0_bank_read_0_ready;
  wire        _bank_wrap_0_bank_write_ready;
  wire        _GEN =
    _bank_wrap_0_burst_write_ready & _bank_wrap_1_burst_write_ready
    & _bank_wrap_2_burst_write_ready & _bank_wrap_3_burst_write_ready;
  wire        _GEN_0 = _GEN & burst_write_enable;
  BankWrapper_matrix_b_0 bank_wrap_0 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_0_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_0_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_0_bank_read_0_ready & bank_read_0_0_enable),
    .bank_read_0_ready   (_bank_wrap_0_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_0_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_0_res0),
    .bank_write_enable   (_bank_wrap_0_bank_write_ready & bank_write_0_enable),
    .bank_write_ready    (_bank_wrap_0_bank_write_ready),
    .bank_write_addr     (bank_write_0_addr),
    .bank_write_data     (bank_write_0_data)
  );
  BankWrapper_matrix_b_1 bank_wrap_1 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_1_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_1_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_1_bank_read_0_ready & bank_read_0_1_enable),
    .bank_read_0_ready   (_bank_wrap_1_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_1_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_1_res0),
    .bank_write_enable   (_bank_wrap_1_bank_write_ready & bank_write_1_enable),
    .bank_write_ready    (_bank_wrap_1_bank_write_ready),
    .bank_write_addr     (bank_write_1_addr),
    .bank_write_data     (bank_write_1_data)
  );
  BankWrapper_matrix_b_2 bank_wrap_2 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_2_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_2_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_2_bank_read_0_ready & bank_read_0_2_enable),
    .bank_read_0_ready   (_bank_wrap_2_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_2_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_2_res0),
    .bank_write_enable   (_bank_wrap_2_bank_write_ready & bank_write_2_enable),
    .bank_write_ready    (_bank_wrap_2_bank_write_ready),
    .bank_write_addr     (bank_write_2_addr),
    .bank_write_data     (bank_write_2_data)
  );
  BankWrapper_matrix_b_3 bank_wrap_3 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_3_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_3_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_3_bank_read_0_ready & bank_read_0_3_enable),
    .bank_read_0_ready   (_bank_wrap_3_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_3_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_3_res0),
    .bank_write_enable   (_bank_wrap_3_bank_write_ready & bank_write_3_enable),
    .bank_write_ready    (_bank_wrap_3_bank_write_ready),
    .bank_write_addr     (bank_write_3_addr),
    .bank_write_data     (bank_write_3_data)
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _bank_wrap_0_burst_read_1_res0 | _bank_wrap_1_burst_read_1_res0
    | _bank_wrap_2_burst_read_1_res0 | _bank_wrap_3_burst_read_1_res0;
  assign burst_write_ready = _GEN;
  assign bank_read_0_0_ready = _bank_wrap_0_bank_read_0_ready;
  assign bank_read_1_0_ready = 1'h1;
  assign bank_write_0_ready = _bank_wrap_0_bank_write_ready;
  assign bank_read_0_1_ready = _bank_wrap_1_bank_read_0_ready;
  assign bank_read_1_1_ready = 1'h1;
  assign bank_write_1_ready = _bank_wrap_1_bank_write_ready;
  assign bank_read_0_2_ready = _bank_wrap_2_bank_read_0_ready;
  assign bank_read_1_2_ready = 1'h1;
  assign bank_write_2_ready = _bank_wrap_2_bank_write_ready;
  assign bank_read_0_3_ready = _bank_wrap_3_bank_read_0_ready;
  assign bank_read_1_3_ready = 1'h1;
  assign bank_write_3_ready = _bank_wrap_3_bank_write_ready;
endmodule

module WireDefault_w1_i012(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_c_0(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h0 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h0 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h0 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i012 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i013(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_c_1(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h1 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h1 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h1 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i013 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i014(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_c_2(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h2 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h2 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h2 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i014 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module WireDefault_w1_i015(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_matrix_c_3(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [1:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [15:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [1:0]  bank_write_addr,
  input  [15:0] bank_write_data
);

  wire [1:0]  _write_addr_wire_read_data;
  wire [15:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [15:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h2;
  wire [33:0] _GEN_0 = ({1'h0, 33'h3 - {1'h0, _GEN % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_1 = ({1'h0, _GEN} + {1'h0, _GEN_0[31:0]}) / 33'h4;
  wire        _GEN_2 = ~burst_read_0_enable & bank_read_0_enable;
  wire [33:0] _GEN_3 =
    ({1'h0, 33'h3 - {1'h0, burst_read_1_addr / 32'h2 % 32'h4}} + 34'h4) % 34'h4;
  wire [31:0] _GEN_4 = burst_write_addr / 32'h2;
  wire [33:0] _GEN_5 = ({1'h0, 33'h3 - {1'h0, _GEN_4 % 32'h4}} + 34'h4) % 34'h4;
  wire [32:0] _GEN_6 = ({1'h0, _GEN_4} + {1'h0, _GEN_5[31:0]}) / 33'h4;
  wire        _GEN_7 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w16_a2_d4 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_2 | burst_read_0_enable),
    .raddr (_GEN_2 ? bank_read_0_addr : _GEN_1[1:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_7 | _write_enable_wire_read_res0),
    .waddr (_GEN_7 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_7 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i015 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable & _GEN_5[31:0] < 32'h4)
  );
  Wire_w16 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data
      (_GEN_5[31:0] == 32'h3
         ? burst_write_data[63:48]
         : _GEN_5[31:0] == 32'h2
             ? burst_write_data[47:32]
             : _GEN_5[31:0] == 32'h1 ? burst_write_data[31:16] : burst_write_data[15:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w2 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_6[1:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _GEN_3[31:0] < 32'h4
      ? (_GEN_3[31:0] == 32'h3
           ? {_mem_bank_rdata, 48'h0}
           : _GEN_3[31:0] == 32'h2
               ? {16'h0, _mem_bank_rdata, 32'h0}
               : _GEN_3[31:0] == 32'h1
                   ? {32'h0, _mem_bank_rdata, 16'h0}
                   : {48'h0, _mem_bank_rdata})
      : 64'h0;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module memory_matrix_c(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_0_enable,
  output        bank_read_0_0_ready,
  input  [1:0]  bank_read_0_0_addr,
  output        bank_read_1_0_ready,
  output [15:0] bank_read_1_0_res0,
  input         bank_write_0_enable,
  output        bank_write_0_ready,
  input  [1:0]  bank_write_0_addr,
  input  [15:0] bank_write_0_data,
  input         bank_read_0_1_enable,
  output        bank_read_0_1_ready,
  input  [1:0]  bank_read_0_1_addr,
  output        bank_read_1_1_ready,
  output [15:0] bank_read_1_1_res0,
  input         bank_write_1_enable,
  output        bank_write_1_ready,
  input  [1:0]  bank_write_1_addr,
  input  [15:0] bank_write_1_data,
  input         bank_read_0_2_enable,
  output        bank_read_0_2_ready,
  input  [1:0]  bank_read_0_2_addr,
  output        bank_read_1_2_ready,
  output [15:0] bank_read_1_2_res0,
  input         bank_write_2_enable,
  output        bank_write_2_ready,
  input  [1:0]  bank_write_2_addr,
  input  [15:0] bank_write_2_data,
  input         bank_read_0_3_enable,
  output        bank_read_0_3_ready,
  input  [1:0]  bank_read_0_3_addr,
  output        bank_read_1_3_ready,
  output [15:0] bank_read_1_3_res0,
  input         bank_write_3_enable,
  output        bank_write_3_ready,
  input  [1:0]  bank_write_3_addr,
  input  [15:0] bank_write_3_data
);

  wire [63:0] _bank_wrap_3_burst_read_1_res0;
  wire        _bank_wrap_3_burst_write_ready;
  wire        _bank_wrap_3_bank_read_0_ready;
  wire        _bank_wrap_3_bank_write_ready;
  wire [63:0] _bank_wrap_2_burst_read_1_res0;
  wire        _bank_wrap_2_burst_write_ready;
  wire        _bank_wrap_2_bank_read_0_ready;
  wire        _bank_wrap_2_bank_write_ready;
  wire [63:0] _bank_wrap_1_burst_read_1_res0;
  wire        _bank_wrap_1_burst_write_ready;
  wire        _bank_wrap_1_bank_read_0_ready;
  wire        _bank_wrap_1_bank_write_ready;
  wire [63:0] _bank_wrap_0_burst_read_1_res0;
  wire        _bank_wrap_0_burst_write_ready;
  wire        _bank_wrap_0_bank_read_0_ready;
  wire        _bank_wrap_0_bank_write_ready;
  wire        _GEN =
    _bank_wrap_0_burst_write_ready & _bank_wrap_1_burst_write_ready
    & _bank_wrap_2_burst_write_ready & _bank_wrap_3_burst_write_ready;
  wire        _GEN_0 = _GEN & burst_write_enable;
  BankWrapper_matrix_c_0 bank_wrap_0 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_0_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_0_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_0_bank_read_0_ready & bank_read_0_0_enable),
    .bank_read_0_ready   (_bank_wrap_0_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_0_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_0_res0),
    .bank_write_enable   (_bank_wrap_0_bank_write_ready & bank_write_0_enable),
    .bank_write_ready    (_bank_wrap_0_bank_write_ready),
    .bank_write_addr     (bank_write_0_addr),
    .bank_write_data     (bank_write_0_data)
  );
  BankWrapper_matrix_c_1 bank_wrap_1 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_1_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_1_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_1_bank_read_0_ready & bank_read_0_1_enable),
    .bank_read_0_ready   (_bank_wrap_1_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_1_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_1_res0),
    .bank_write_enable   (_bank_wrap_1_bank_write_ready & bank_write_1_enable),
    .bank_write_ready    (_bank_wrap_1_bank_write_ready),
    .bank_write_addr     (bank_write_1_addr),
    .bank_write_data     (bank_write_1_data)
  );
  BankWrapper_matrix_c_2 bank_wrap_2 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_2_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_2_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_2_bank_read_0_ready & bank_read_0_2_enable),
    .bank_read_0_ready   (_bank_wrap_2_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_2_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_2_res0),
    .bank_write_enable   (_bank_wrap_2_bank_write_ready & bank_write_2_enable),
    .bank_write_ready    (_bank_wrap_2_bank_write_ready),
    .bank_write_addr     (bank_write_2_addr),
    .bank_write_data     (bank_write_2_data)
  );
  BankWrapper_matrix_c_3 bank_wrap_3 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (_bank_wrap_3_burst_read_1_res0),
    .burst_write_enable  (_GEN_0),
    .burst_write_ready   (_bank_wrap_3_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_3_bank_read_0_ready & bank_read_0_3_enable),
    .bank_read_0_ready   (_bank_wrap_3_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_3_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_3_res0),
    .bank_write_enable   (_bank_wrap_3_bank_write_ready & bank_write_3_enable),
    .bank_write_ready    (_bank_wrap_3_bank_write_ready),
    .bank_write_addr     (bank_write_3_addr),
    .bank_write_data     (bank_write_3_data)
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 =
    _bank_wrap_0_burst_read_1_res0 | _bank_wrap_1_burst_read_1_res0
    | _bank_wrap_2_burst_read_1_res0 | _bank_wrap_3_burst_read_1_res0;
  assign burst_write_ready = _GEN;
  assign bank_read_0_0_ready = _bank_wrap_0_bank_read_0_ready;
  assign bank_read_1_0_ready = 1'h1;
  assign bank_write_0_ready = _bank_wrap_0_bank_write_ready;
  assign bank_read_0_1_ready = _bank_wrap_1_bank_read_0_ready;
  assign bank_read_1_1_ready = 1'h1;
  assign bank_write_1_ready = _bank_wrap_1_bank_write_ready;
  assign bank_read_0_2_ready = _bank_wrap_2_bank_read_0_ready;
  assign bank_read_1_2_ready = 1'h1;
  assign bank_write_2_ready = _bank_wrap_2_bank_write_ready;
  assign bank_read_0_3_ready = _bank_wrap_3_bank_read_0_ready;
  assign bank_read_1_3_ready = 1'h1;
  assign bank_write_3_ready = _bank_wrap_3_bank_write_ready;
endmodule

module WireDefault_w1_i016(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_values_0(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input  [2:0]  bank_read_0_addr,
  output        bank_read_1_ready,
  output [63:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input  [2:0]  bank_write_addr,
  input  [63:0] bank_write_data
);

  wire [2:0]  _write_addr_wire_read_data;
  wire [63:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [63:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h8;
  wire        _GEN_0 = ~burst_read_0_enable & bank_read_0_enable;
  wire [31:0] _GEN_1 = burst_write_addr / 32'h8;
  wire        _GEN_2 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w64_a3_d8 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_0 | burst_read_0_enable),
    .raddr (_GEN_0 ? bank_read_0_addr : _GEN[2:0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_2 | _write_enable_wire_read_res0),
    .waddr (_GEN_2 ? bank_write_addr : _write_addr_wire_read_data),
    .wdata (_GEN_2 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i016 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable)
  );
  Wire_w64 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (burst_write_data),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w3 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (_GEN_1[2:0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 = _mem_bank_rdata;
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module memory_values(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_0_enable,
  output        bank_read_0_0_ready,
  input  [2:0]  bank_read_0_0_addr,
  output        bank_read_1_0_ready,
  output [63:0] bank_read_1_0_res0,
  input         bank_write_0_enable,
  output        bank_write_0_ready,
  input  [2:0]  bank_write_0_addr,
  input  [63:0] bank_write_0_data
);

  wire _bank_wrap_0_burst_write_ready;
  wire _bank_wrap_0_bank_read_0_ready;
  wire _bank_wrap_0_bank_write_ready;
  BankWrapper_values_0 bank_wrap_0 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (burst_read_1_res0),
    .burst_write_enable  (_bank_wrap_0_burst_write_ready & burst_write_enable),
    .burst_write_ready   (_bank_wrap_0_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_bank_wrap_0_bank_read_0_ready & bank_read_0_0_enable),
    .bank_read_0_ready   (_bank_wrap_0_bank_read_0_ready),
    .bank_read_0_addr    (bank_read_0_0_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_0_res0),
    .bank_write_enable   (_bank_wrap_0_bank_write_ready & bank_write_0_enable),
    .bank_write_ready    (_bank_wrap_0_bank_write_ready),
    .bank_write_addr     (bank_write_0_addr),
    .bank_write_data     (bank_write_0_data)
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_write_ready = _bank_wrap_0_burst_write_ready;
  assign bank_read_0_0_ready = _bank_wrap_0_bank_read_0_ready;
  assign bank_read_1_0_ready = 1'h1;
  assign bank_write_0_ready = _bank_wrap_0_bank_write_ready;
endmodule

module WireDefault_w1_i017(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module BankWrapper_vidx_0(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_enable,
  output        bank_read_0_ready,
  input         bank_read_0_addr,
  output        bank_read_1_ready,
  output [31:0] bank_read_1_res0,
  input         bank_write_enable,
  output        bank_write_ready,
  input         bank_write_addr,
  input  [31:0] bank_write_data
);

  wire        _write_addr_wire_read_data;
  wire [31:0] _write_data_wire_read_data;
  wire        _write_enable_wire_read_res0;
  wire [31:0] _mem_bank_rdata;
  wire [31:0] _GEN = burst_read_0_addr / 32'h4;
  wire        _GEN_0 = ~burst_read_0_enable & bank_read_0_enable;
  wire [31:0] _GEN_1 = burst_write_addr / 32'h4;
  wire        _GEN_2 = ~_write_enable_wire_read_res0 & bank_write_enable;
  Mem1r1w1c_AReg_w32_a1_d2 mem_bank (
    .clock (clk),
    .reset (rst),
    .ren   (_GEN_0 | burst_read_0_enable),
    .raddr (_GEN_0 ? bank_read_0_addr : burst_read_0_enable & _GEN[0]),
    .rdata (_mem_bank_rdata),
    .wen   (_GEN_2 | _write_enable_wire_read_res0),
    .waddr
      (_GEN_2
         ? bank_write_addr
         : _write_enable_wire_read_res0 & _write_addr_wire_read_data),
    .wdata (_GEN_2 ? bank_write_data : _write_data_wire_read_data)
  );
  WireDefault_w1_i017 write_enable_wire (
    .read_ready   (),
    .read_res0    (_write_enable_wire_read_res0),
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_in_    (burst_write_enable)
  );
  Wire_w32 write_data_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (burst_write_data[31:0]),
    .read_data    (_write_data_wire_read_data),
    .read_ready   ()
  );
  Wire_w1 write_addr_wire (
    .write_enable (burst_write_enable),
    .write_ready  (),
    .write_data   (burst_write_enable & _GEN_1[0]),
    .read_data    (_write_addr_wire_read_data),
    .read_ready   ()
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_read_1_res0 = {32'h0, _mem_bank_rdata};
  assign burst_write_ready = 1'h1;
  assign bank_read_0_ready = ~burst_read_0_enable;
  assign bank_read_1_ready = 1'h1;
  assign bank_read_1_res0 = _mem_bank_rdata;
  assign bank_write_ready = ~_write_enable_wire_read_res0;
endmodule

module memory_vidx(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  input  [31:0] burst_read_1_addr,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         bank_read_0_0_enable,
  output        bank_read_0_0_ready,
  input         bank_read_0_0_addr,
  output        bank_read_1_0_ready,
  output [31:0] bank_read_1_0_res0,
  input         bank_write_0_enable,
  output        bank_write_0_ready,
  input         bank_write_0_addr,
  input  [31:0] bank_write_0_data
);

  wire _bank_wrap_0_burst_write_ready;
  wire _bank_wrap_0_bank_read_0_ready;
  wire _bank_wrap_0_bank_write_ready;
  wire _GEN = _bank_wrap_0_bank_read_0_ready & bank_read_0_0_enable;
  wire _GEN_0 = _bank_wrap_0_bank_write_ready & bank_write_0_enable;
  BankWrapper_vidx_0 bank_wrap_0 (
    .clk                 (clk),
    .rst                 (rst),
    .burst_read_0_enable (burst_read_0_enable),
    .burst_read_0_ready  (),
    .burst_read_0_addr   (burst_read_0_addr),
    .burst_read_1_enable (burst_read_1_enable),
    .burst_read_1_ready  (),
    .burst_read_1_addr   (burst_read_1_addr),
    .burst_read_1_res0   (burst_read_1_res0),
    .burst_write_enable  (_bank_wrap_0_burst_write_ready & burst_write_enable),
    .burst_write_ready   (_bank_wrap_0_burst_write_ready),
    .burst_write_addr    (burst_write_addr),
    .burst_write_data    (burst_write_data),
    .bank_read_0_enable  (_GEN),
    .bank_read_0_ready   (_bank_wrap_0_bank_read_0_ready),
    .bank_read_0_addr    (_GEN & bank_read_0_0_addr),
    .bank_read_1_ready   (),
    .bank_read_1_res0    (bank_read_1_0_res0),
    .bank_write_enable   (_GEN_0),
    .bank_write_ready    (_bank_wrap_0_bank_write_ready),
    .bank_write_addr     (_GEN_0 & bank_write_0_addr),
    .bank_write_data     (bank_write_0_data)
  );
  assign burst_read_0_ready = 1'h1;
  assign burst_read_1_ready = 1'h1;
  assign burst_write_ready = _bank_wrap_0_burst_write_ready;
  assign bank_read_0_0_ready = _bank_wrap_0_bank_read_0_ready;
  assign bank_read_1_0_ready = 1'h1;
  assign bank_write_0_ready = _bank_wrap_0_bank_write_ready;
endmodule

module ScratchpadMemoryPool(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         acc_read_0_enable,
  output        acc_read_0_ready,
  input         acc_read_0_addr,
  output        acc_read_1_ready,
  output [15:0] acc_read_1_res0,
  input         acc_write_enable,
  output        acc_write_ready,
  input         acc_write_addr,
  input  [15:0] acc_write_data,
  input         decompressed_weights_0_read_0_enable,
  output        decompressed_weights_0_read_0_ready,
  input  [2:0]  decompressed_weights_0_read_0_addr,
  output        decompressed_weights_0_read_1_ready,
  output [15:0] decompressed_weights_0_read_1_res0,
  input         decompressed_weights_0_write_enable,
  output        decompressed_weights_0_write_ready,
  input  [2:0]  decompressed_weights_0_write_addr,
  input  [15:0] decompressed_weights_0_write_data,
  input         decompressed_weights_1_read_0_enable,
  output        decompressed_weights_1_read_0_ready,
  input  [2:0]  decompressed_weights_1_read_0_addr,
  output        decompressed_weights_1_read_1_ready,
  output [15:0] decompressed_weights_1_read_1_res0,
  input         decompressed_weights_1_write_enable,
  output        decompressed_weights_1_write_ready,
  input  [2:0]  decompressed_weights_1_write_addr,
  input  [15:0] decompressed_weights_1_write_data,
  input         decompressed_weights_2_read_0_enable,
  output        decompressed_weights_2_read_0_ready,
  input  [2:0]  decompressed_weights_2_read_0_addr,
  output        decompressed_weights_2_read_1_ready,
  output [15:0] decompressed_weights_2_read_1_res0,
  input         decompressed_weights_2_write_enable,
  output        decompressed_weights_2_write_ready,
  input  [2:0]  decompressed_weights_2_write_addr,
  input  [15:0] decompressed_weights_2_write_data,
  input         decompressed_weights_3_read_0_enable,
  output        decompressed_weights_3_read_0_ready,
  input  [2:0]  decompressed_weights_3_read_0_addr,
  output        decompressed_weights_3_read_1_ready,
  output [15:0] decompressed_weights_3_read_1_res0,
  input         decompressed_weights_3_write_enable,
  output        decompressed_weights_3_write_ready,
  input  [2:0]  decompressed_weights_3_write_addr,
  input  [15:0] decompressed_weights_3_write_data,
  input         dense_values_0_read_0_enable,
  output        dense_values_0_read_0_ready,
  input  [2:0]  dense_values_0_read_0_addr,
  output        dense_values_0_read_1_ready,
  output [7:0]  dense_values_0_read_1_res0,
  input         dense_values_0_write_enable,
  output        dense_values_0_write_ready,
  input  [2:0]  dense_values_0_write_addr,
  input  [7:0]  dense_values_0_write_data,
  input         dense_values_1_read_0_enable,
  output        dense_values_1_read_0_ready,
  input  [2:0]  dense_values_1_read_0_addr,
  output        dense_values_1_read_1_ready,
  output [7:0]  dense_values_1_read_1_res0,
  input         dense_values_1_write_enable,
  output        dense_values_1_write_ready,
  input  [2:0]  dense_values_1_write_addr,
  input  [7:0]  dense_values_1_write_data,
  input         dense_values_2_read_0_enable,
  output        dense_values_2_read_0_ready,
  input  [2:0]  dense_values_2_read_0_addr,
  output        dense_values_2_read_1_ready,
  output [7:0]  dense_values_2_read_1_res0,
  input         dense_values_2_write_enable,
  output        dense_values_2_write_ready,
  input  [2:0]  dense_values_2_write_addr,
  input  [7:0]  dense_values_2_write_data,
  input         dense_values_3_read_0_enable,
  output        dense_values_3_read_0_ready,
  input  [2:0]  dense_values_3_read_0_addr,
  output        dense_values_3_read_1_ready,
  output [7:0]  dense_values_3_read_1_res0,
  input         dense_values_3_write_enable,
  output        dense_values_3_write_ready,
  input  [2:0]  dense_values_3_write_addr,
  input  [7:0]  dense_values_3_write_data,
  input         matrix_a_0_read_0_enable,
  output        matrix_a_0_read_0_ready,
  input  [1:0]  matrix_a_0_read_0_addr,
  output        matrix_a_0_read_1_ready,
  output [15:0] matrix_a_0_read_1_res0,
  input         matrix_a_0_write_enable,
  output        matrix_a_0_write_ready,
  input  [1:0]  matrix_a_0_write_addr,
  input  [15:0] matrix_a_0_write_data,
  input         matrix_a_1_read_0_enable,
  output        matrix_a_1_read_0_ready,
  input  [1:0]  matrix_a_1_read_0_addr,
  output        matrix_a_1_read_1_ready,
  output [15:0] matrix_a_1_read_1_res0,
  input         matrix_a_1_write_enable,
  output        matrix_a_1_write_ready,
  input  [1:0]  matrix_a_1_write_addr,
  input  [15:0] matrix_a_1_write_data,
  input         matrix_a_2_read_0_enable,
  output        matrix_a_2_read_0_ready,
  input  [1:0]  matrix_a_2_read_0_addr,
  output        matrix_a_2_read_1_ready,
  output [15:0] matrix_a_2_read_1_res0,
  input         matrix_a_2_write_enable,
  output        matrix_a_2_write_ready,
  input  [1:0]  matrix_a_2_write_addr,
  input  [15:0] matrix_a_2_write_data,
  input         matrix_a_3_read_0_enable,
  output        matrix_a_3_read_0_ready,
  input  [1:0]  matrix_a_3_read_0_addr,
  output        matrix_a_3_read_1_ready,
  output [15:0] matrix_a_3_read_1_res0,
  input         matrix_a_3_write_enable,
  output        matrix_a_3_write_ready,
  input  [1:0]  matrix_a_3_write_addr,
  input  [15:0] matrix_a_3_write_data,
  input         matrix_b_0_read_0_enable,
  output        matrix_b_0_read_0_ready,
  input  [1:0]  matrix_b_0_read_0_addr,
  output        matrix_b_0_read_1_ready,
  output [15:0] matrix_b_0_read_1_res0,
  input         matrix_b_0_write_enable,
  output        matrix_b_0_write_ready,
  input  [1:0]  matrix_b_0_write_addr,
  input  [15:0] matrix_b_0_write_data,
  input         matrix_b_1_read_0_enable,
  output        matrix_b_1_read_0_ready,
  input  [1:0]  matrix_b_1_read_0_addr,
  output        matrix_b_1_read_1_ready,
  output [15:0] matrix_b_1_read_1_res0,
  input         matrix_b_1_write_enable,
  output        matrix_b_1_write_ready,
  input  [1:0]  matrix_b_1_write_addr,
  input  [15:0] matrix_b_1_write_data,
  input         matrix_b_2_read_0_enable,
  output        matrix_b_2_read_0_ready,
  input  [1:0]  matrix_b_2_read_0_addr,
  output        matrix_b_2_read_1_ready,
  output [15:0] matrix_b_2_read_1_res0,
  input         matrix_b_2_write_enable,
  output        matrix_b_2_write_ready,
  input  [1:0]  matrix_b_2_write_addr,
  input  [15:0] matrix_b_2_write_data,
  input         matrix_b_3_read_0_enable,
  output        matrix_b_3_read_0_ready,
  input  [1:0]  matrix_b_3_read_0_addr,
  output        matrix_b_3_read_1_ready,
  output [15:0] matrix_b_3_read_1_res0,
  input         matrix_b_3_write_enable,
  output        matrix_b_3_write_ready,
  input  [1:0]  matrix_b_3_write_addr,
  input  [15:0] matrix_b_3_write_data,
  input         matrix_c_0_read_0_enable,
  output        matrix_c_0_read_0_ready,
  input  [1:0]  matrix_c_0_read_0_addr,
  output        matrix_c_0_read_1_ready,
  output [15:0] matrix_c_0_read_1_res0,
  input         matrix_c_0_write_enable,
  output        matrix_c_0_write_ready,
  input  [1:0]  matrix_c_0_write_addr,
  input  [15:0] matrix_c_0_write_data,
  input         matrix_c_1_read_0_enable,
  output        matrix_c_1_read_0_ready,
  input  [1:0]  matrix_c_1_read_0_addr,
  output        matrix_c_1_read_1_ready,
  output [15:0] matrix_c_1_read_1_res0,
  input         matrix_c_1_write_enable,
  output        matrix_c_1_write_ready,
  input  [1:0]  matrix_c_1_write_addr,
  input  [15:0] matrix_c_1_write_data,
  input         matrix_c_2_read_0_enable,
  output        matrix_c_2_read_0_ready,
  input  [1:0]  matrix_c_2_read_0_addr,
  output        matrix_c_2_read_1_ready,
  output [15:0] matrix_c_2_read_1_res0,
  input         matrix_c_2_write_enable,
  output        matrix_c_2_write_ready,
  input  [1:0]  matrix_c_2_write_addr,
  input  [15:0] matrix_c_2_write_data,
  input         matrix_c_3_read_0_enable,
  output        matrix_c_3_read_0_ready,
  input  [1:0]  matrix_c_3_read_0_addr,
  output        matrix_c_3_read_1_ready,
  output [15:0] matrix_c_3_read_1_res0,
  input         matrix_c_3_write_enable,
  output        matrix_c_3_write_ready,
  input  [1:0]  matrix_c_3_write_addr,
  input  [15:0] matrix_c_3_write_data,
  input         values_read_0_enable,
  output        values_read_0_ready,
  input  [2:0]  values_read_0_addr,
  output        values_read_1_ready,
  output [63:0] values_read_1_res0,
  input         values_write_enable,
  output        values_write_ready,
  input  [2:0]  values_write_addr,
  input  [63:0] values_write_data,
  input         vidx_read_0_enable,
  output        vidx_read_0_ready,
  input         vidx_read_0_addr,
  output        vidx_read_1_ready,
  output [31:0] vidx_read_1_res0,
  input         vidx_write_enable,
  output        vidx_write_ready,
  input         vidx_write_addr,
  input  [31:0] vidx_write_data
);

  wire [31:0] _burst_read_addr_reg_read_data;
  wire [63:0] _inst_vidx_burst_read_1_res0;
  wire        _inst_vidx_burst_write_ready;
  wire        _inst_vidx_bank_read_0_0_ready;
  wire        _inst_vidx_bank_write_0_ready;
  wire [63:0] _inst_values_burst_read_1_res0;
  wire        _inst_values_burst_write_ready;
  wire        _inst_values_bank_read_0_0_ready;
  wire        _inst_values_bank_write_0_ready;
  wire        _inst_matrix_c_burst_read_0_ready;
  wire        _inst_matrix_c_burst_read_1_ready;
  wire [63:0] _inst_matrix_c_burst_read_1_res0;
  wire        _inst_matrix_c_burst_write_ready;
  wire        _inst_matrix_c_bank_read_0_0_ready;
  wire        _inst_matrix_c_bank_write_0_ready;
  wire        _inst_matrix_c_bank_read_0_1_ready;
  wire        _inst_matrix_c_bank_write_1_ready;
  wire        _inst_matrix_c_bank_read_0_2_ready;
  wire        _inst_matrix_c_bank_write_2_ready;
  wire        _inst_matrix_c_bank_read_0_3_ready;
  wire        _inst_matrix_c_bank_write_3_ready;
  wire        _inst_matrix_b_burst_read_0_ready;
  wire        _inst_matrix_b_burst_read_1_ready;
  wire [63:0] _inst_matrix_b_burst_read_1_res0;
  wire        _inst_matrix_b_burst_write_ready;
  wire        _inst_matrix_b_bank_read_0_0_ready;
  wire        _inst_matrix_b_bank_write_0_ready;
  wire        _inst_matrix_b_bank_read_0_1_ready;
  wire        _inst_matrix_b_bank_write_1_ready;
  wire        _inst_matrix_b_bank_read_0_2_ready;
  wire        _inst_matrix_b_bank_write_2_ready;
  wire        _inst_matrix_b_bank_read_0_3_ready;
  wire        _inst_matrix_b_bank_write_3_ready;
  wire        _inst_matrix_a_burst_read_0_ready;
  wire        _inst_matrix_a_burst_read_1_ready;
  wire [63:0] _inst_matrix_a_burst_read_1_res0;
  wire        _inst_matrix_a_burst_write_ready;
  wire        _inst_matrix_a_bank_read_0_0_ready;
  wire        _inst_matrix_a_bank_write_0_ready;
  wire        _inst_matrix_a_bank_read_0_1_ready;
  wire        _inst_matrix_a_bank_write_1_ready;
  wire        _inst_matrix_a_bank_read_0_2_ready;
  wire        _inst_matrix_a_bank_write_2_ready;
  wire        _inst_matrix_a_bank_read_0_3_ready;
  wire        _inst_matrix_a_bank_write_3_ready;
  wire        _inst_decompressed_weights_burst_read_0_ready;
  wire        _inst_decompressed_weights_burst_read_1_ready;
  wire [63:0] _inst_decompressed_weights_burst_read_1_res0;
  wire        _inst_decompressed_weights_burst_write_ready;
  wire        _inst_decompressed_weights_bank_read_0_0_ready;
  wire        _inst_decompressed_weights_bank_write_0_ready;
  wire        _inst_decompressed_weights_bank_read_0_1_ready;
  wire        _inst_decompressed_weights_bank_write_1_ready;
  wire        _inst_decompressed_weights_bank_read_0_2_ready;
  wire        _inst_decompressed_weights_bank_write_2_ready;
  wire        _inst_decompressed_weights_bank_read_0_3_ready;
  wire        _inst_decompressed_weights_bank_write_3_ready;
  wire [63:0] _inst_acc_burst_read_1_res0;
  wire        _inst_acc_burst_write_ready;
  wire        _inst_acc_bank_read_0_0_ready;
  wire        _inst_acc_bank_write_0_ready;
  wire        _GEN =
    _inst_decompressed_weights_burst_read_0_ready & _inst_matrix_a_burst_read_0_ready
    & _inst_matrix_b_burst_read_0_ready & _inst_matrix_c_burst_read_0_ready;
  wire        _GEN_0 = _GEN & burst_read_0_enable;
  wire        _GEN_1 =
    _inst_decompressed_weights_burst_read_1_ready & _inst_matrix_a_burst_read_1_ready
    & _inst_matrix_b_burst_read_1_ready & _inst_matrix_c_burst_read_1_ready;
  wire        _GEN_2 = _GEN_1 & burst_read_1_enable;
  wire        _GEN_3 =
    _inst_acc_burst_write_ready & _inst_decompressed_weights_burst_write_ready
    & _inst_matrix_a_burst_write_ready & _inst_matrix_b_burst_write_ready
    & _inst_matrix_c_burst_write_ready & _inst_values_burst_write_ready
    & _inst_vidx_burst_write_ready;
  wire        _GEN_4 = _GEN_3 & burst_write_enable;
  wire        _GEN_5 = _inst_acc_bank_read_0_0_ready & acc_read_0_enable;
  wire        _GEN_6 = _inst_acc_bank_write_0_ready & acc_write_enable;
  wire        _GEN_7 = _inst_vidx_bank_read_0_0_ready & vidx_read_0_enable;
  wire        _GEN_8 = _inst_vidx_bank_write_0_ready & vidx_write_enable;
  memory_acc inst_acc (
    .clk                  (clk),
    .rst                  (rst),
    .burst_read_0_enable  (_GEN_0),
    .burst_read_0_ready   (),
    .burst_read_0_addr    (burst_read_0_addr),
    .burst_read_1_enable  (_GEN_2),
    .burst_read_1_ready   (),
    .burst_read_1_addr    (_burst_read_addr_reg_read_data),
    .burst_read_1_res0    (_inst_acc_burst_read_1_res0),
    .burst_write_enable   (_GEN_4 & burst_write_addr < 32'h2),
    .burst_write_ready    (_inst_acc_burst_write_ready),
    .burst_write_addr     (burst_write_addr),
    .burst_write_data     (burst_write_data),
    .bank_read_0_0_enable (_GEN_5),
    .bank_read_0_0_ready  (_inst_acc_bank_read_0_0_ready),
    .bank_read_0_0_addr   (_GEN_5 & acc_read_0_addr),
    .bank_read_1_0_ready  (),
    .bank_read_1_0_res0   (acc_read_1_res0),
    .bank_write_0_enable  (_GEN_6),
    .bank_write_0_ready   (_inst_acc_bank_write_0_ready),
    .bank_write_0_addr    (_GEN_6 & acc_write_addr),
    .bank_write_0_data    (acc_write_data)
  );
  memory_decompressed_weights inst_decompressed_weights (
    .clk                  (clk),
    .rst                  (rst),
    .burst_read_0_enable  (_GEN_0),
    .burst_read_0_ready   (_inst_decompressed_weights_burst_read_0_ready),
    .burst_read_0_addr    (burst_read_0_addr - 32'h2),
    .burst_read_1_enable  (_GEN_2),
    .burst_read_1_ready   (_inst_decompressed_weights_burst_read_1_ready),
    .burst_read_1_addr    (_burst_read_addr_reg_read_data - 32'h2),
    .burst_read_1_res0    (_inst_decompressed_weights_burst_read_1_res0),
    .burst_write_enable
      (_GEN_4 & (|(burst_write_addr[31:1])) & burst_write_addr < 32'h42),
    .burst_write_ready    (_inst_decompressed_weights_burst_write_ready),
    .burst_write_addr     (burst_write_addr - 32'h2),
    .burst_write_data     (burst_write_data),
    .bank_read_0_0_enable
      (_inst_decompressed_weights_bank_read_0_0_ready
       & decompressed_weights_0_read_0_enable),
    .bank_read_0_0_ready  (_inst_decompressed_weights_bank_read_0_0_ready),
    .bank_read_0_0_addr   (decompressed_weights_0_read_0_addr),
    .bank_read_1_0_ready  (),
    .bank_read_1_0_res0   (decompressed_weights_0_read_1_res0),
    .bank_write_0_enable
      (_inst_decompressed_weights_bank_write_0_ready
       & decompressed_weights_0_write_enable),
    .bank_write_0_ready   (_inst_decompressed_weights_bank_write_0_ready),
    .bank_write_0_addr    (decompressed_weights_0_write_addr),
    .bank_write_0_data    (decompressed_weights_0_write_data),
    .bank_read_0_1_enable
      (_inst_decompressed_weights_bank_read_0_1_ready
       & decompressed_weights_1_read_0_enable),
    .bank_read_0_1_ready  (_inst_decompressed_weights_bank_read_0_1_ready),
    .bank_read_0_1_addr   (decompressed_weights_1_read_0_addr),
    .bank_read_1_1_ready  (),
    .bank_read_1_1_res0   (decompressed_weights_1_read_1_res0),
    .bank_write_1_enable
      (_inst_decompressed_weights_bank_write_1_ready
       & decompressed_weights_1_write_enable),
    .bank_write_1_ready   (_inst_decompressed_weights_bank_write_1_ready),
    .bank_write_1_addr    (decompressed_weights_1_write_addr),
    .bank_write_1_data    (decompressed_weights_1_write_data),
    .bank_read_0_2_enable
      (_inst_decompressed_weights_bank_read_0_2_ready
       & decompressed_weights_2_read_0_enable),
    .bank_read_0_2_ready  (_inst_decompressed_weights_bank_read_0_2_ready),
    .bank_read_0_2_addr   (decompressed_weights_2_read_0_addr),
    .bank_read_1_2_ready  (),
    .bank_read_1_2_res0   (decompressed_weights_2_read_1_res0),
    .bank_write_2_enable
      (_inst_decompressed_weights_bank_write_2_ready
       & decompressed_weights_2_write_enable),
    .bank_write_2_ready   (_inst_decompressed_weights_bank_write_2_ready),
    .bank_write_2_addr    (decompressed_weights_2_write_addr),
    .bank_write_2_data    (decompressed_weights_2_write_data),
    .bank_read_0_3_enable
      (_inst_decompressed_weights_bank_read_0_3_ready
       & decompressed_weights_3_read_0_enable),
    .bank_read_0_3_ready  (_inst_decompressed_weights_bank_read_0_3_ready),
    .bank_read_0_3_addr   (decompressed_weights_3_read_0_addr),
    .bank_read_1_3_ready  (),
    .bank_read_1_3_res0   (decompressed_weights_3_read_1_res0),
    .bank_write_3_enable
      (_inst_decompressed_weights_bank_write_3_ready
       & decompressed_weights_3_write_enable),
    .bank_write_3_ready   (_inst_decompressed_weights_bank_write_3_ready),
    .bank_write_3_addr    (decompressed_weights_3_write_addr),
    .bank_write_3_data    (decompressed_weights_3_write_data)
  );
  memory_dense_values inst_dense_values (
    .clk                  (clk),
    .rst                  (rst),
    .burst_read_0_enable  (_GEN_0),
    .burst_read_0_ready   (),
    .burst_read_0_addr    (burst_read_0_addr - 32'h80),
    .burst_read_1_enable  (_GEN_2),
    .burst_read_1_ready   (),
    .burst_read_1_addr    (_burst_read_addr_reg_read_data - 32'h80),
    .burst_read_1_res0    (),
    .burst_write_enable
      (_GEN_4 & (|(burst_write_addr[31:7])) & burst_write_addr < 32'hA0),
    .burst_write_ready    (),
    .burst_write_addr     (burst_write_addr - 32'h80),
    .burst_write_data     (burst_write_data),
    .bank_read_0_0_enable (dense_values_0_read_0_enable),
    .bank_read_0_0_ready  (),
    .bank_read_0_0_addr   (dense_values_0_read_0_addr),
    .bank_read_1_0_ready  (),
    .bank_read_1_0_res0   (dense_values_0_read_1_res0),
    .bank_write_0_enable  (dense_values_0_write_enable),
    .bank_write_0_ready   (),
    .bank_write_0_addr    (dense_values_0_write_addr),
    .bank_write_0_data    (dense_values_0_write_data),
    .bank_read_0_1_enable (dense_values_1_read_0_enable),
    .bank_read_0_1_ready  (),
    .bank_read_0_1_addr   (dense_values_1_read_0_addr),
    .bank_read_1_1_ready  (),
    .bank_read_1_1_res0   (dense_values_1_read_1_res0),
    .bank_write_1_enable  (dense_values_1_write_enable),
    .bank_write_1_ready   (),
    .bank_write_1_addr    (dense_values_1_write_addr),
    .bank_write_1_data    (dense_values_1_write_data),
    .bank_read_0_2_enable (dense_values_2_read_0_enable),
    .bank_read_0_2_ready  (),
    .bank_read_0_2_addr   (dense_values_2_read_0_addr),
    .bank_read_1_2_ready  (),
    .bank_read_1_2_res0   (dense_values_2_read_1_res0),
    .bank_write_2_enable  (dense_values_2_write_enable),
    .bank_write_2_ready   (),
    .bank_write_2_addr    (dense_values_2_write_addr),
    .bank_write_2_data    (dense_values_2_write_data),
    .bank_read_0_3_enable (dense_values_3_read_0_enable),
    .bank_read_0_3_ready  (),
    .bank_read_0_3_addr   (dense_values_3_read_0_addr),
    .bank_read_1_3_ready  (),
    .bank_read_1_3_res0   (dense_values_3_read_1_res0),
    .bank_write_3_enable  (dense_values_3_write_enable),
    .bank_write_3_ready   (),
    .bank_write_3_addr    (dense_values_3_write_addr),
    .bank_write_3_data    (dense_values_3_write_data)
  );
  memory_matrix_a inst_matrix_a (
    .clk                  (clk),
    .rst                  (rst),
    .burst_read_0_enable  (_GEN_0),
    .burst_read_0_ready   (_inst_matrix_a_burst_read_0_ready),
    .burst_read_0_addr    (burst_read_0_addr - 32'hA0),
    .burst_read_1_enable  (_GEN_2),
    .burst_read_1_ready   (_inst_matrix_a_burst_read_1_ready),
    .burst_read_1_addr    (_burst_read_addr_reg_read_data - 32'hA0),
    .burst_read_1_res0    (_inst_matrix_a_burst_read_1_res0),
    .burst_write_enable
      (_GEN_4 & burst_write_addr > 32'h9F & burst_write_addr < 32'hC0),
    .burst_write_ready    (_inst_matrix_a_burst_write_ready),
    .burst_write_addr     (burst_write_addr - 32'hA0),
    .burst_write_data     (burst_write_data),
    .bank_read_0_0_enable (_inst_matrix_a_bank_read_0_0_ready & matrix_a_0_read_0_enable),
    .bank_read_0_0_ready  (_inst_matrix_a_bank_read_0_0_ready),
    .bank_read_0_0_addr   (matrix_a_0_read_0_addr),
    .bank_read_1_0_ready  (),
    .bank_read_1_0_res0   (matrix_a_0_read_1_res0),
    .bank_write_0_enable  (_inst_matrix_a_bank_write_0_ready & matrix_a_0_write_enable),
    .bank_write_0_ready   (_inst_matrix_a_bank_write_0_ready),
    .bank_write_0_addr    (matrix_a_0_write_addr),
    .bank_write_0_data    (matrix_a_0_write_data),
    .bank_read_0_1_enable (_inst_matrix_a_bank_read_0_1_ready & matrix_a_1_read_0_enable),
    .bank_read_0_1_ready  (_inst_matrix_a_bank_read_0_1_ready),
    .bank_read_0_1_addr   (matrix_a_1_read_0_addr),
    .bank_read_1_1_ready  (),
    .bank_read_1_1_res0   (matrix_a_1_read_1_res0),
    .bank_write_1_enable  (_inst_matrix_a_bank_write_1_ready & matrix_a_1_write_enable),
    .bank_write_1_ready   (_inst_matrix_a_bank_write_1_ready),
    .bank_write_1_addr    (matrix_a_1_write_addr),
    .bank_write_1_data    (matrix_a_1_write_data),
    .bank_read_0_2_enable (_inst_matrix_a_bank_read_0_2_ready & matrix_a_2_read_0_enable),
    .bank_read_0_2_ready  (_inst_matrix_a_bank_read_0_2_ready),
    .bank_read_0_2_addr   (matrix_a_2_read_0_addr),
    .bank_read_1_2_ready  (),
    .bank_read_1_2_res0   (matrix_a_2_read_1_res0),
    .bank_write_2_enable  (_inst_matrix_a_bank_write_2_ready & matrix_a_2_write_enable),
    .bank_write_2_ready   (_inst_matrix_a_bank_write_2_ready),
    .bank_write_2_addr    (matrix_a_2_write_addr),
    .bank_write_2_data    (matrix_a_2_write_data),
    .bank_read_0_3_enable (_inst_matrix_a_bank_read_0_3_ready & matrix_a_3_read_0_enable),
    .bank_read_0_3_ready  (_inst_matrix_a_bank_read_0_3_ready),
    .bank_read_0_3_addr   (matrix_a_3_read_0_addr),
    .bank_read_1_3_ready  (),
    .bank_read_1_3_res0   (matrix_a_3_read_1_res0),
    .bank_write_3_enable  (_inst_matrix_a_bank_write_3_ready & matrix_a_3_write_enable),
    .bank_write_3_ready   (_inst_matrix_a_bank_write_3_ready),
    .bank_write_3_addr    (matrix_a_3_write_addr),
    .bank_write_3_data    (matrix_a_3_write_data)
  );
  memory_matrix_b inst_matrix_b (
    .clk                  (clk),
    .rst                  (rst),
    .burst_read_0_enable  (_GEN_0),
    .burst_read_0_ready   (_inst_matrix_b_burst_read_0_ready),
    .burst_read_0_addr    (burst_read_0_addr - 32'hC0),
    .burst_read_1_enable  (_GEN_2),
    .burst_read_1_ready   (_inst_matrix_b_burst_read_1_ready),
    .burst_read_1_addr    (_burst_read_addr_reg_read_data - 32'hC0),
    .burst_read_1_res0    (_inst_matrix_b_burst_read_1_res0),
    .burst_write_enable
      (_GEN_4 & burst_write_addr > 32'hBF & burst_write_addr < 32'hE0),
    .burst_write_ready    (_inst_matrix_b_burst_write_ready),
    .burst_write_addr     (burst_write_addr - 32'hC0),
    .burst_write_data     (burst_write_data),
    .bank_read_0_0_enable (_inst_matrix_b_bank_read_0_0_ready & matrix_b_0_read_0_enable),
    .bank_read_0_0_ready  (_inst_matrix_b_bank_read_0_0_ready),
    .bank_read_0_0_addr   (matrix_b_0_read_0_addr),
    .bank_read_1_0_ready  (),
    .bank_read_1_0_res0   (matrix_b_0_read_1_res0),
    .bank_write_0_enable  (_inst_matrix_b_bank_write_0_ready & matrix_b_0_write_enable),
    .bank_write_0_ready   (_inst_matrix_b_bank_write_0_ready),
    .bank_write_0_addr    (matrix_b_0_write_addr),
    .bank_write_0_data    (matrix_b_0_write_data),
    .bank_read_0_1_enable (_inst_matrix_b_bank_read_0_1_ready & matrix_b_1_read_0_enable),
    .bank_read_0_1_ready  (_inst_matrix_b_bank_read_0_1_ready),
    .bank_read_0_1_addr   (matrix_b_1_read_0_addr),
    .bank_read_1_1_ready  (),
    .bank_read_1_1_res0   (matrix_b_1_read_1_res0),
    .bank_write_1_enable  (_inst_matrix_b_bank_write_1_ready & matrix_b_1_write_enable),
    .bank_write_1_ready   (_inst_matrix_b_bank_write_1_ready),
    .bank_write_1_addr    (matrix_b_1_write_addr),
    .bank_write_1_data    (matrix_b_1_write_data),
    .bank_read_0_2_enable (_inst_matrix_b_bank_read_0_2_ready & matrix_b_2_read_0_enable),
    .bank_read_0_2_ready  (_inst_matrix_b_bank_read_0_2_ready),
    .bank_read_0_2_addr   (matrix_b_2_read_0_addr),
    .bank_read_1_2_ready  (),
    .bank_read_1_2_res0   (matrix_b_2_read_1_res0),
    .bank_write_2_enable  (_inst_matrix_b_bank_write_2_ready & matrix_b_2_write_enable),
    .bank_write_2_ready   (_inst_matrix_b_bank_write_2_ready),
    .bank_write_2_addr    (matrix_b_2_write_addr),
    .bank_write_2_data    (matrix_b_2_write_data),
    .bank_read_0_3_enable (_inst_matrix_b_bank_read_0_3_ready & matrix_b_3_read_0_enable),
    .bank_read_0_3_ready  (_inst_matrix_b_bank_read_0_3_ready),
    .bank_read_0_3_addr   (matrix_b_3_read_0_addr),
    .bank_read_1_3_ready  (),
    .bank_read_1_3_res0   (matrix_b_3_read_1_res0),
    .bank_write_3_enable  (_inst_matrix_b_bank_write_3_ready & matrix_b_3_write_enable),
    .bank_write_3_ready   (_inst_matrix_b_bank_write_3_ready),
    .bank_write_3_addr    (matrix_b_3_write_addr),
    .bank_write_3_data    (matrix_b_3_write_data)
  );
  memory_matrix_c inst_matrix_c (
    .clk                  (clk),
    .rst                  (rst),
    .burst_read_0_enable  (_GEN_0),
    .burst_read_0_ready   (_inst_matrix_c_burst_read_0_ready),
    .burst_read_0_addr    (burst_read_0_addr - 32'hE0),
    .burst_read_1_enable  (_GEN_2),
    .burst_read_1_ready   (_inst_matrix_c_burst_read_1_ready),
    .burst_read_1_addr    (_burst_read_addr_reg_read_data - 32'hE0),
    .burst_read_1_res0    (_inst_matrix_c_burst_read_1_res0),
    .burst_write_enable
      (_GEN_4 & burst_write_addr > 32'hDF & burst_write_addr < 32'h100),
    .burst_write_ready    (_inst_matrix_c_burst_write_ready),
    .burst_write_addr     (burst_write_addr - 32'hE0),
    .burst_write_data     (burst_write_data),
    .bank_read_0_0_enable (_inst_matrix_c_bank_read_0_0_ready & matrix_c_0_read_0_enable),
    .bank_read_0_0_ready  (_inst_matrix_c_bank_read_0_0_ready),
    .bank_read_0_0_addr   (matrix_c_0_read_0_addr),
    .bank_read_1_0_ready  (),
    .bank_read_1_0_res0   (matrix_c_0_read_1_res0),
    .bank_write_0_enable  (_inst_matrix_c_bank_write_0_ready & matrix_c_0_write_enable),
    .bank_write_0_ready   (_inst_matrix_c_bank_write_0_ready),
    .bank_write_0_addr    (matrix_c_0_write_addr),
    .bank_write_0_data    (matrix_c_0_write_data),
    .bank_read_0_1_enable (_inst_matrix_c_bank_read_0_1_ready & matrix_c_1_read_0_enable),
    .bank_read_0_1_ready  (_inst_matrix_c_bank_read_0_1_ready),
    .bank_read_0_1_addr   (matrix_c_1_read_0_addr),
    .bank_read_1_1_ready  (),
    .bank_read_1_1_res0   (matrix_c_1_read_1_res0),
    .bank_write_1_enable  (_inst_matrix_c_bank_write_1_ready & matrix_c_1_write_enable),
    .bank_write_1_ready   (_inst_matrix_c_bank_write_1_ready),
    .bank_write_1_addr    (matrix_c_1_write_addr),
    .bank_write_1_data    (matrix_c_1_write_data),
    .bank_read_0_2_enable (_inst_matrix_c_bank_read_0_2_ready & matrix_c_2_read_0_enable),
    .bank_read_0_2_ready  (_inst_matrix_c_bank_read_0_2_ready),
    .bank_read_0_2_addr   (matrix_c_2_read_0_addr),
    .bank_read_1_2_ready  (),
    .bank_read_1_2_res0   (matrix_c_2_read_1_res0),
    .bank_write_2_enable  (_inst_matrix_c_bank_write_2_ready & matrix_c_2_write_enable),
    .bank_write_2_ready   (_inst_matrix_c_bank_write_2_ready),
    .bank_write_2_addr    (matrix_c_2_write_addr),
    .bank_write_2_data    (matrix_c_2_write_data),
    .bank_read_0_3_enable (_inst_matrix_c_bank_read_0_3_ready & matrix_c_3_read_0_enable),
    .bank_read_0_3_ready  (_inst_matrix_c_bank_read_0_3_ready),
    .bank_read_0_3_addr   (matrix_c_3_read_0_addr),
    .bank_read_1_3_ready  (),
    .bank_read_1_3_res0   (matrix_c_3_read_1_res0),
    .bank_write_3_enable  (_inst_matrix_c_bank_write_3_ready & matrix_c_3_write_enable),
    .bank_write_3_ready   (_inst_matrix_c_bank_write_3_ready),
    .bank_write_3_addr    (matrix_c_3_write_addr),
    .bank_write_3_data    (matrix_c_3_write_data)
  );
  memory_values inst_values (
    .clk                  (clk),
    .rst                  (rst),
    .burst_read_0_enable  (_GEN_0),
    .burst_read_0_ready   (),
    .burst_read_0_addr    (burst_read_0_addr - 32'h100),
    .burst_read_1_enable  (_GEN_2),
    .burst_read_1_ready   (),
    .burst_read_1_addr    (_burst_read_addr_reg_read_data - 32'h100),
    .burst_read_1_res0    (_inst_values_burst_read_1_res0),
    .burst_write_enable
      (_GEN_4 & (|(burst_write_addr[31:8])) & burst_write_addr < 32'h140),
    .burst_write_ready    (_inst_values_burst_write_ready),
    .burst_write_addr     (burst_write_addr - 32'h100),
    .burst_write_data     (burst_write_data),
    .bank_read_0_0_enable (_inst_values_bank_read_0_0_ready & values_read_0_enable),
    .bank_read_0_0_ready  (_inst_values_bank_read_0_0_ready),
    .bank_read_0_0_addr   (values_read_0_addr),
    .bank_read_1_0_ready  (),
    .bank_read_1_0_res0   (values_read_1_res0),
    .bank_write_0_enable  (_inst_values_bank_write_0_ready & values_write_enable),
    .bank_write_0_ready   (_inst_values_bank_write_0_ready),
    .bank_write_0_addr    (values_write_addr),
    .bank_write_0_data    (values_write_data)
  );
  memory_vidx inst_vidx (
    .clk                  (clk),
    .rst                  (rst),
    .burst_read_0_enable  (_GEN_0),
    .burst_read_0_ready   (),
    .burst_read_0_addr    (burst_read_0_addr - 32'h140),
    .burst_read_1_enable  (_GEN_2),
    .burst_read_1_ready   (),
    .burst_read_1_addr    (_burst_read_addr_reg_read_data - 32'h140),
    .burst_read_1_res0    (_inst_vidx_burst_read_1_res0),
    .burst_write_enable
      (_GEN_4 & burst_write_addr > 32'h13F & burst_write_addr < 32'h144),
    .burst_write_ready    (_inst_vidx_burst_write_ready),
    .burst_write_addr     (burst_write_addr - 32'h140),
    .burst_write_data     (burst_write_data),
    .bank_read_0_0_enable (_GEN_7),
    .bank_read_0_0_ready  (_inst_vidx_bank_read_0_0_ready),
    .bank_read_0_0_addr   (_GEN_7 & vidx_read_0_addr),
    .bank_read_1_0_ready  (),
    .bank_read_1_0_res0   (vidx_read_1_res0),
    .bank_write_0_enable  (_GEN_8),
    .bank_write_0_ready   (_inst_vidx_bank_write_0_ready),
    .bank_write_0_addr    (_GEN_8 & vidx_write_addr),
    .bank_write_0_data    (vidx_write_data)
  );
  Reg_width32_init0 burst_read_addr_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (burst_read_0_addr),
    .read_ready   (),
    .read_data    (_burst_read_addr_reg_read_data),
    .write_ready  ()
  );
  assign burst_read_0_ready = _GEN;
  assign burst_read_1_ready = _GEN_1;
  assign burst_read_1_res0 =
    (_burst_read_addr_reg_read_data < 32'h2 ? _inst_acc_burst_read_1_res0 : 64'h0)
    | ((|(_burst_read_addr_reg_read_data[31:1])) & _burst_read_addr_reg_read_data < 32'h42
         ? _inst_decompressed_weights_burst_read_1_res0
         : 64'h0)
    | (_burst_read_addr_reg_read_data > 32'h9F & _burst_read_addr_reg_read_data < 32'hC0
         ? _inst_matrix_a_burst_read_1_res0
         : 64'h0)
    | (_burst_read_addr_reg_read_data > 32'hBF & _burst_read_addr_reg_read_data < 32'hE0
         ? _inst_matrix_b_burst_read_1_res0
         : 64'h0)
    | (_burst_read_addr_reg_read_data > 32'hDF & _burst_read_addr_reg_read_data < 32'h100
         ? _inst_matrix_c_burst_read_1_res0
         : 64'h0)
    | ((|(_burst_read_addr_reg_read_data[31:8]))
       & _burst_read_addr_reg_read_data < 32'h140
         ? _inst_values_burst_read_1_res0
         : 64'h0)
    | (_burst_read_addr_reg_read_data > 32'h13F & _burst_read_addr_reg_read_data < 32'h144
         ? _inst_vidx_burst_read_1_res0
         : 64'h0);
  assign burst_write_ready = _GEN_3;
  assign acc_read_0_ready = _inst_acc_bank_read_0_0_ready;
  assign acc_read_1_ready = 1'h1;
  assign acc_write_ready = _inst_acc_bank_write_0_ready;
  assign decompressed_weights_0_read_0_ready =
    _inst_decompressed_weights_bank_read_0_0_ready;
  assign decompressed_weights_0_read_1_ready = 1'h1;
  assign decompressed_weights_0_write_ready =
    _inst_decompressed_weights_bank_write_0_ready;
  assign decompressed_weights_1_read_0_ready =
    _inst_decompressed_weights_bank_read_0_1_ready;
  assign decompressed_weights_1_read_1_ready = 1'h1;
  assign decompressed_weights_1_write_ready =
    _inst_decompressed_weights_bank_write_1_ready;
  assign decompressed_weights_2_read_0_ready =
    _inst_decompressed_weights_bank_read_0_2_ready;
  assign decompressed_weights_2_read_1_ready = 1'h1;
  assign decompressed_weights_2_write_ready =
    _inst_decompressed_weights_bank_write_2_ready;
  assign decompressed_weights_3_read_0_ready =
    _inst_decompressed_weights_bank_read_0_3_ready;
  assign decompressed_weights_3_read_1_ready = 1'h1;
  assign decompressed_weights_3_write_ready =
    _inst_decompressed_weights_bank_write_3_ready;
  assign dense_values_0_read_0_ready = 1'h1;
  assign dense_values_0_read_1_ready = 1'h1;
  assign dense_values_0_write_ready = 1'h1;
  assign dense_values_1_read_0_ready = 1'h1;
  assign dense_values_1_read_1_ready = 1'h1;
  assign dense_values_1_write_ready = 1'h1;
  assign dense_values_2_read_0_ready = 1'h1;
  assign dense_values_2_read_1_ready = 1'h1;
  assign dense_values_2_write_ready = 1'h1;
  assign dense_values_3_read_0_ready = 1'h1;
  assign dense_values_3_read_1_ready = 1'h1;
  assign dense_values_3_write_ready = 1'h1;
  assign matrix_a_0_read_0_ready = _inst_matrix_a_bank_read_0_0_ready;
  assign matrix_a_0_read_1_ready = 1'h1;
  assign matrix_a_0_write_ready = _inst_matrix_a_bank_write_0_ready;
  assign matrix_a_1_read_0_ready = _inst_matrix_a_bank_read_0_1_ready;
  assign matrix_a_1_read_1_ready = 1'h1;
  assign matrix_a_1_write_ready = _inst_matrix_a_bank_write_1_ready;
  assign matrix_a_2_read_0_ready = _inst_matrix_a_bank_read_0_2_ready;
  assign matrix_a_2_read_1_ready = 1'h1;
  assign matrix_a_2_write_ready = _inst_matrix_a_bank_write_2_ready;
  assign matrix_a_3_read_0_ready = _inst_matrix_a_bank_read_0_3_ready;
  assign matrix_a_3_read_1_ready = 1'h1;
  assign matrix_a_3_write_ready = _inst_matrix_a_bank_write_3_ready;
  assign matrix_b_0_read_0_ready = _inst_matrix_b_bank_read_0_0_ready;
  assign matrix_b_0_read_1_ready = 1'h1;
  assign matrix_b_0_write_ready = _inst_matrix_b_bank_write_0_ready;
  assign matrix_b_1_read_0_ready = _inst_matrix_b_bank_read_0_1_ready;
  assign matrix_b_1_read_1_ready = 1'h1;
  assign matrix_b_1_write_ready = _inst_matrix_b_bank_write_1_ready;
  assign matrix_b_2_read_0_ready = _inst_matrix_b_bank_read_0_2_ready;
  assign matrix_b_2_read_1_ready = 1'h1;
  assign matrix_b_2_write_ready = _inst_matrix_b_bank_write_2_ready;
  assign matrix_b_3_read_0_ready = _inst_matrix_b_bank_read_0_3_ready;
  assign matrix_b_3_read_1_ready = 1'h1;
  assign matrix_b_3_write_ready = _inst_matrix_b_bank_write_3_ready;
  assign matrix_c_0_read_0_ready = _inst_matrix_c_bank_read_0_0_ready;
  assign matrix_c_0_read_1_ready = 1'h1;
  assign matrix_c_0_write_ready = _inst_matrix_c_bank_write_0_ready;
  assign matrix_c_1_read_0_ready = _inst_matrix_c_bank_read_0_1_ready;
  assign matrix_c_1_read_1_ready = 1'h1;
  assign matrix_c_1_write_ready = _inst_matrix_c_bank_write_1_ready;
  assign matrix_c_2_read_0_ready = _inst_matrix_c_bank_read_0_2_ready;
  assign matrix_c_2_read_1_ready = 1'h1;
  assign matrix_c_2_write_ready = _inst_matrix_c_bank_write_2_ready;
  assign matrix_c_3_read_0_ready = _inst_matrix_c_bank_read_0_3_ready;
  assign matrix_c_3_read_1_ready = 1'h1;
  assign matrix_c_3_write_ready = _inst_matrix_c_bank_write_3_ready;
  assign values_read_0_ready = _inst_values_bank_read_0_0_ready;
  assign values_read_1_ready = 1'h1;
  assign values_write_ready = _inst_values_bank_write_0_ready;
  assign vidx_read_0_ready = _inst_vidx_bank_read_0_0_ready;
  assign vidx_read_1_ready = 1'h1;
  assign vidx_write_ready = _inst_vidx_bank_write_0_ready;
endmodule

module FIFO1_PUSH_w96(
  input         clk,
                rst,
  output        full_ready,
                full_res0,
  input         deq_enable,
  output        deq_ready,
  output [95:0] deq_res0,
  input         enq_enable,
  output        enq_ready,
  input  [95:0] enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width96_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w37(
  input         clk,
                rst,
  output        full_ready,
                full_res0,
  input         deq_enable,
  output        deq_ready,
  output [36:0] deq_res0,
  input         enq_enable,
  output        enq_ready,
  input  [36:0] enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width37_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module RoCCAdapter(
  input         clk,
                rst,
                cmd_from_bus_enable,
  output        cmd_from_bus_ready,
  input  [6:0]  cmd_from_bus_rocc_cmd_bus_funct,
  input  [4:0]  cmd_from_bus_rocc_cmd_bus_rs1,
                cmd_from_bus_rocc_cmd_bus_rs2,
                cmd_from_bus_rocc_cmd_bus_rd,
  input         cmd_from_bus_rocc_cmd_bus_xs1,
                cmd_from_bus_rocc_cmd_bus_xs2,
                cmd_from_bus_rocc_cmd_bus_xd,
  input  [6:0]  cmd_from_bus_rocc_cmd_bus_opcode,
  input  [31:0] cmd_from_bus_rocc_cmd_bus_rs1data,
                cmd_from_bus_rocc_cmd_bus_rs2data,
  input         resp_from_user_enable,
  output        resp_from_user_ready,
  input  [4:0]  resp_from_user_rocc_resp_user_rd,
  input  [31:0] resp_from_user_rocc_resp_user_rddata,
  input         cmd_to_user_0b05_enable,
  output        cmd_to_user_0b05_ready,
  output [6:0]  cmd_to_user_0b05_res0_funct,
  output [4:0]  cmd_to_user_0b05_res0_rs1,
                cmd_to_user_0b05_res0_rs2,
                cmd_to_user_0b05_res0_rd,
  output        cmd_to_user_0b05_res0_xs1,
                cmd_to_user_0b05_res0_xs2,
                cmd_to_user_0b05_res0_xd,
  output [6:0]  cmd_to_user_0b05_res0_opcode,
  output [31:0] cmd_to_user_0b05_res0_rs1data,
                cmd_to_user_0b05_res0_rs2data,
  input         cmd_to_user_2b38_enable,
  output        cmd_to_user_2b38_ready,
  output [6:0]  cmd_to_user_2b38_res0_funct,
  output [4:0]  cmd_to_user_2b38_res0_rs1,
                cmd_to_user_2b38_res0_rs2,
                cmd_to_user_2b38_res0_rd,
  output        cmd_to_user_2b38_res0_xs1,
                cmd_to_user_2b38_res0_xs2,
                cmd_to_user_2b38_res0_xd,
  output [6:0]  cmd_to_user_2b38_res0_opcode,
  output [31:0] cmd_to_user_2b38_res0_rs1data,
                cmd_to_user_2b38_res0_rs2data,
  output [4:0]  rocc_resp_rocc_resp_to_bus_result_rd,
  output [31:0] rocc_resp_rocc_resp_to_bus_result_rddata,
  input         rocc_resp_rocc_resp_to_bus_ready,
  output        rocc_resp_rocc_resp_to_bus_enable
);

  wire        _rocc_resp_fifo_deq_ready;
  wire [36:0] _rocc_resp_fifo_deq_res0;
  wire        _rocc_resp_fifo_enq_ready;
  wire        _rocc_cmd_fifo_2b38_deq_ready;
  wire [95:0] _rocc_cmd_fifo_2b38_deq_res0;
  wire        _rocc_cmd_fifo_2b38_enq_ready;
  wire        _rocc_cmd_fifo_0b05_deq_ready;
  wire [95:0] _rocc_cmd_fifo_0b05_deq_res0;
  wire        _rocc_cmd_fifo_0b05_enq_ready;
  wire        _GEN = _rocc_cmd_fifo_0b05_enq_ready & _rocc_cmd_fifo_2b38_enq_ready;
  wire        _GEN_0 = _GEN & cmd_from_bus_enable;
  wire [95:0] _GEN_1 =
    {cmd_from_bus_rocc_cmd_bus_rs2data,
     cmd_from_bus_rocc_cmd_bus_rs1data,
     cmd_from_bus_rocc_cmd_bus_opcode,
     cmd_from_bus_rocc_cmd_bus_xd,
     cmd_from_bus_rocc_cmd_bus_xs2,
     cmd_from_bus_rocc_cmd_bus_xs1,
     cmd_from_bus_rocc_cmd_bus_rd,
     cmd_from_bus_rocc_cmd_bus_rs2,
     cmd_from_bus_rocc_cmd_bus_rs1,
     cmd_from_bus_rocc_cmd_bus_funct};
  wire [13:0] _GEN_2 =
    {cmd_from_bus_rocc_cmd_bus_opcode, cmd_from_bus_rocc_cmd_bus_funct};
  wire        _GEN_3 = _rocc_resp_fifo_deq_ready & rocc_resp_rocc_resp_to_bus_ready;
  FIFO1_PUSH_w96 rocc_cmd_fifo_0b05 (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_rocc_cmd_fifo_0b05_deq_ready & cmd_to_user_0b05_enable),
    .deq_ready  (_rocc_cmd_fifo_0b05_deq_ready),
    .deq_res0   (_rocc_cmd_fifo_0b05_deq_res0),
    .enq_enable (_GEN_0 & _GEN_2 == 14'h585),
    .enq_ready  (_rocc_cmd_fifo_0b05_enq_ready),
    .enq_data   (_GEN_1)
  );
  FIFO1_PUSH_w96 rocc_cmd_fifo_2b38 (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_rocc_cmd_fifo_2b38_deq_ready & cmd_to_user_2b38_enable),
    .deq_ready  (_rocc_cmd_fifo_2b38_deq_ready),
    .deq_res0   (_rocc_cmd_fifo_2b38_deq_res0),
    .enq_enable (_GEN_0 & _GEN_2 == 14'h15B8),
    .enq_ready  (_rocc_cmd_fifo_2b38_enq_ready),
    .enq_data   (_GEN_1)
  );
  FIFO1_PUSH_w37 rocc_resp_fifo (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_3),
    .deq_ready  (_rocc_resp_fifo_deq_ready),
    .deq_res0   (_rocc_resp_fifo_deq_res0),
    .enq_enable (_rocc_resp_fifo_enq_ready & resp_from_user_enable),
    .enq_ready  (_rocc_resp_fifo_enq_ready),
    .enq_data   ({resp_from_user_rocc_resp_user_rddata, resp_from_user_rocc_resp_user_rd})
  );
  assign cmd_from_bus_ready = _GEN;
  assign resp_from_user_ready = _rocc_resp_fifo_enq_ready;
  assign cmd_to_user_0b05_ready = _rocc_cmd_fifo_0b05_deq_ready;
  assign cmd_to_user_0b05_res0_funct = _rocc_cmd_fifo_0b05_deq_res0[6:0];
  assign cmd_to_user_0b05_res0_rs1 = _rocc_cmd_fifo_0b05_deq_res0[11:7];
  assign cmd_to_user_0b05_res0_rs2 = _rocc_cmd_fifo_0b05_deq_res0[16:12];
  assign cmd_to_user_0b05_res0_rd = _rocc_cmd_fifo_0b05_deq_res0[21:17];
  assign cmd_to_user_0b05_res0_xs1 = _rocc_cmd_fifo_0b05_deq_res0[22];
  assign cmd_to_user_0b05_res0_xs2 = _rocc_cmd_fifo_0b05_deq_res0[23];
  assign cmd_to_user_0b05_res0_xd = _rocc_cmd_fifo_0b05_deq_res0[24];
  assign cmd_to_user_0b05_res0_opcode = _rocc_cmd_fifo_0b05_deq_res0[31:25];
  assign cmd_to_user_0b05_res0_rs1data = _rocc_cmd_fifo_0b05_deq_res0[63:32];
  assign cmd_to_user_0b05_res0_rs2data = _rocc_cmd_fifo_0b05_deq_res0[95:64];
  assign cmd_to_user_2b38_ready = _rocc_cmd_fifo_2b38_deq_ready;
  assign cmd_to_user_2b38_res0_funct = _rocc_cmd_fifo_2b38_deq_res0[6:0];
  assign cmd_to_user_2b38_res0_rs1 = _rocc_cmd_fifo_2b38_deq_res0[11:7];
  assign cmd_to_user_2b38_res0_rs2 = _rocc_cmd_fifo_2b38_deq_res0[16:12];
  assign cmd_to_user_2b38_res0_rd = _rocc_cmd_fifo_2b38_deq_res0[21:17];
  assign cmd_to_user_2b38_res0_xs1 = _rocc_cmd_fifo_2b38_deq_res0[22];
  assign cmd_to_user_2b38_res0_xs2 = _rocc_cmd_fifo_2b38_deq_res0[23];
  assign cmd_to_user_2b38_res0_xd = _rocc_cmd_fifo_2b38_deq_res0[24];
  assign cmd_to_user_2b38_res0_opcode = _rocc_cmd_fifo_2b38_deq_res0[31:25];
  assign cmd_to_user_2b38_res0_rs1data = _rocc_cmd_fifo_2b38_deq_res0[63:32];
  assign cmd_to_user_2b38_res0_rs2data = _rocc_cmd_fifo_2b38_deq_res0[95:64];
  assign rocc_resp_rocc_resp_to_bus_result_rd = _rocc_resp_fifo_deq_res0[4:0];
  assign rocc_resp_rocc_resp_to_bus_result_rddata = _rocc_resp_fifo_deq_res0[36:5];
  assign rocc_resp_rocc_resp_to_bus_enable = _GEN_3;
endmodule

module WireDefault_w1_i018(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w85(
  input         clk,
                rst,
  output        full_ready,
                full_res0,
  input         deq_enable,
  output        deq_ready,
  output [84:0] deq_res0,
  input         enq_enable,
  output        enq_ready,
  input  [84:0] enq_data
);

  wire [84:0] _enq_value_read_data;
  wire        _enqed_read_res0;
  wire        _deqed_read_res0;
  wire [1:0]  _state_read_data;
  wire [84:0] _reg1_read_data;
  wire [84:0] _reg0_read_data;
  wire        _GEN = (|_state_read_data) & deq_enable;
  wire        _GEN_0 = _state_read_data != 2'h2;
  wire        _GEN_1 = _GEN_0 & enq_enable;
  wire        _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire        _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire        _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire        _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width85_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data   (_GEN_5 ? _reg1_read_data : _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width85_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i018 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i018 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w85 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module MemoryTranslator(
  input         clk,
                rst,
                resp_from_bus_enable,
  output        resp_from_bus_ready,
  input  [31:0] resp_from_bus_hella_resp_data,
  input  [7:0]  resp_from_bus_hella_resp_tag,
  input  [4:0]  resp_from_bus_hella_resp_cmd,
  input  [1:0]  resp_from_bus_hella_resp_size,
  input         resp_from_bus_hella_resp_signed,
                cmd_from_user_enable,
  output        cmd_from_user_ready,
  input  [31:0] cmd_from_user_user_cmd_addr,
  input         cmd_from_user_user_cmd_cmd,
  input  [1:0]  cmd_from_user_user_cmd_size,
  input  [31:0] cmd_from_user_user_cmd_data,
  input  [3:0]  cmd_from_user_user_cmd_mask,
  input  [7:0]  cmd_from_user_user_cmd_tag,
  input         resp_to_user_enable,
  output        resp_to_user_ready,
  output [31:0] resp_to_user_res0_data,
  output [7:0]  resp_to_user_res0_tag,
  output [31:0] hella_cmd_hella_cmd_to_bus_cmd_addr,
  output [7:0]  hella_cmd_hella_cmd_to_bus_cmd_tag,
  output [4:0]  hella_cmd_hella_cmd_to_bus_cmd_cmd,
  output [1:0]  hella_cmd_hella_cmd_to_bus_cmd_size,
  output        hella_cmd_hella_cmd_to_bus_cmd_signed,
                hella_cmd_hella_cmd_to_bus_cmd_phys,
  output [31:0] hella_cmd_hella_cmd_to_bus_cmd_data,
  output [3:0]  hella_cmd_hella_cmd_to_bus_cmd_mask,
  input         hella_cmd_hella_cmd_to_bus_ready,
  output        hella_cmd_hella_cmd_to_bus_enable
);

  wire [31:0] _slot_recv_data_read_data;
  wire        _slot1_recv_wire_read_data;
  wire        _slot0_recv_wire_read_data;
  wire        _slot1_can_collect_wire_read_data;
  wire        _slot0_can_collect_wire_read_data;
  wire        _newer_slot_reg_read_data;
  wire        _slot1_rxd_reg_read_data;
  wire        _slot1_txd_reg_read_data;
  wire [7:0]  _slot1_tag_reg_read_data;
  wire [31:0] _slot1_data_reg_read_data;
  wire        _slot0_rxd_reg_read_data;
  wire        _slot0_txd_reg_read_data;
  wire [7:0]  _slot0_tag_reg_read_data;
  wire [31:0] _slot0_data_reg_read_data;
  wire        _hella_cmd_fifo_deq_ready;
  wire [84:0] _hella_cmd_fifo_deq_res0;
  wire        _hella_cmd_fifo_enq_ready;
  wire        _GEN = resp_from_bus_hella_resp_cmd == 5'h0;
  wire        _GEN_0 =
    (_slot0_can_collect_wire_read_data | _slot1_can_collect_wire_read_data)
    & ~_slot0_recv_wire_read_data & ~_slot1_recv_wire_read_data;
  wire        _GEN_1 = _GEN_0 & resp_to_user_enable;
  wire        _GEN_2 = _GEN_1 & _slot0_can_collect_wire_read_data;
  wire        _GEN_3 =
    _hella_cmd_fifo_deq_ready & hella_cmd_hella_cmd_to_bus_ready & ~_GEN_1;
  wire        _GEN_4 = _hella_cmd_fifo_deq_res0[44:40] == 5'h0;
  wire        _GEN_5 = _GEN_3 & _GEN_4;
  wire        _GEN_6 = _GEN_5 & _newer_slot_reg_read_data;
  wire        _GEN_7 = _GEN_5 & ~_newer_slot_reg_read_data;
  FIFO2_I_w85 hella_cmd_fifo (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_3),
    .deq_ready  (_hella_cmd_fifo_deq_ready),
    .deq_res0   (_hella_cmd_fifo_deq_res0),
    .enq_enable (_hella_cmd_fifo_enq_ready & cmd_from_user_enable),
    .enq_ready  (_hella_cmd_fifo_enq_ready),
    .enq_data
      ({cmd_from_user_user_cmd_data,
        cmd_from_user_user_cmd_mask,
        2'h0,
        cmd_from_user_user_cmd_size,
        4'h0,
        cmd_from_user_user_cmd_cmd,
        cmd_from_user_user_cmd_tag,
        cmd_from_user_user_cmd_addr})
  );
  Reg_width32_init0 slot0_data_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_slot0_recv_wire_read_data),
    .write_data   (_slot_recv_data_read_data),
    .read_ready   (),
    .read_data    (_slot0_data_reg_read_data),
    .write_ready  ()
  );
  Reg_width8_init0 slot0_tag_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_6),
    .write_data   (_hella_cmd_fifo_deq_res0[39:32]),
    .read_ready   (),
    .read_data    (_slot0_tag_reg_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 slot0_txd_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_3 & _GEN_4 & _newer_slot_reg_read_data | _GEN_1
       & _slot0_can_collect_wire_read_data),
    .write_data   (_GEN_6),
    .read_ready   (),
    .read_data    (_slot0_txd_reg_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 slot0_rxd_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_2 | _slot0_recv_wire_read_data),
    .write_data   (~_GEN_2 & _slot0_recv_wire_read_data),
    .read_ready   (),
    .read_data    (_slot0_rxd_reg_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 slot1_data_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_slot1_recv_wire_read_data),
    .write_data   (_slot_recv_data_read_data),
    .read_ready   (),
    .read_data    (_slot1_data_reg_read_data),
    .write_ready  ()
  );
  Reg_width8_init0 slot1_tag_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_7),
    .write_data   (_hella_cmd_fifo_deq_res0[39:32]),
    .read_ready   (),
    .read_data    (_slot1_tag_reg_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 slot1_txd_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5 & ~_newer_slot_reg_read_data | _GEN_1 & ~_slot0_can_collect_wire_read_data),
    .write_data   (_GEN_7),
    .read_ready   (),
    .read_data    (_slot1_txd_reg_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 slot1_rxd_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_1 & ~_slot0_can_collect_wire_read_data | _slot1_recv_wire_read_data),
    .write_data
      ((~_GEN_1 | _slot0_can_collect_wire_read_data) & _slot1_recv_wire_read_data),
    .read_ready   (),
    .read_data    (_slot1_rxd_reg_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 newer_slot_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_3 & _GEN_4),
    .write_data   (_GEN_7),
    .read_ready   (),
    .read_data    (_newer_slot_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 slot0_can_collect_wire (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data
      (_slot0_txd_reg_read_data & _slot0_rxd_reg_read_data
       & (~_slot1_txd_reg_read_data | _slot1_txd_reg_read_data
          & _newer_slot_reg_read_data)),
    .read_data    (_slot0_can_collect_wire_read_data),
    .read_ready   ()
  );
  Wire_w1 slot1_can_collect_wire (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data
      (_slot1_txd_reg_read_data & _slot1_rxd_reg_read_data
       & (~_slot0_txd_reg_read_data | _slot0_txd_reg_read_data
          & ~_newer_slot_reg_read_data)),
    .read_data    (_slot1_can_collect_wire_read_data),
    .read_ready   ()
  );
  Wire_w1 slot0_recv_wire (
    .write_enable (resp_from_bus_enable),
    .write_ready  (),
    .write_data
      (resp_from_bus_enable & _GEN & _slot0_txd_reg_read_data
       & _slot0_tag_reg_read_data == resp_from_bus_hella_resp_tag),
    .read_data    (_slot0_recv_wire_read_data),
    .read_ready   ()
  );
  Wire_w1 slot1_recv_wire (
    .write_enable (resp_from_bus_enable),
    .write_ready  (),
    .write_data
      (resp_from_bus_enable & _GEN & _slot1_txd_reg_read_data
       & _slot1_tag_reg_read_data == resp_from_bus_hella_resp_tag),
    .read_data    (_slot1_recv_wire_read_data),
    .read_ready   ()
  );
  Wire_w32 slot_recv_data (
    .write_enable (resp_from_bus_enable),
    .write_ready  (),
    .write_data   (resp_from_bus_hella_resp_data),
    .read_data    (_slot_recv_data_read_data),
    .read_ready   ()
  );
  assign resp_from_bus_ready = 1'h1;
  assign cmd_from_user_ready = _hella_cmd_fifo_enq_ready;
  assign resp_to_user_ready = _GEN_0;
  assign resp_to_user_res0_data =
    _slot0_can_collect_wire_read_data
      ? _slot0_data_reg_read_data
      : _slot1_data_reg_read_data;
  assign resp_to_user_res0_tag =
    _slot0_can_collect_wire_read_data
      ? _slot0_tag_reg_read_data
      : _slot1_tag_reg_read_data;
  assign hella_cmd_hella_cmd_to_bus_cmd_addr = _hella_cmd_fifo_deq_res0[31:0];
  assign hella_cmd_hella_cmd_to_bus_cmd_tag = _hella_cmd_fifo_deq_res0[39:32];
  assign hella_cmd_hella_cmd_to_bus_cmd_cmd = _hella_cmd_fifo_deq_res0[44:40];
  assign hella_cmd_hella_cmd_to_bus_cmd_size = _hella_cmd_fifo_deq_res0[46:45];
  assign hella_cmd_hella_cmd_to_bus_cmd_signed = _hella_cmd_fifo_deq_res0[47];
  assign hella_cmd_hella_cmd_to_bus_cmd_phys = _hella_cmd_fifo_deq_res0[48];
  assign hella_cmd_hella_cmd_to_bus_cmd_data = _hella_cmd_fifo_deq_res0[84:53];
  assign hella_cmd_hella_cmd_to_bus_cmd_mask = _hella_cmd_fifo_deq_res0[52:49];
  assign hella_cmd_hella_cmd_to_bus_enable = _GEN_3;
endmodule

module WireDefault_w1_i019(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w1(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i019 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i019 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module WireDefault_w1_i020(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w10(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i020 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i020 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module WireDefault_w1_i021(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w11(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i021 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i021 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module FIFO1_PUSH_w1(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w10(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w11(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w12(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module WireDefault_w1_i022(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w12(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i022 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i022 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module WireDefault_w1_i023(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w13(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i023 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i023 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module WireDefault_w1_i024(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w32(
  input         clk,
                rst,
  output        full_ready,
                full_res0,
  input         deq_enable,
  output        deq_ready,
  output [31:0] deq_res0,
  input         enq_enable,
  output        enq_ready,
  input  [31:0] enq_data
);

  wire [31:0] _enq_value_read_data;
  wire        _enqed_read_res0;
  wire        _deqed_read_res0;
  wire [1:0]  _state_read_data;
  wire [31:0] _reg1_read_data;
  wire [31:0] _reg0_read_data;
  wire        _GEN = (|_state_read_data) & deq_enable;
  wire        _GEN_0 = _state_read_data != 2'h2;
  wire        _GEN_1 = _GEN_0 & enq_enable;
  wire        _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire        _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire        _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire        _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width32_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data   (_GEN_5 ? _reg1_read_data : _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i024 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i024 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w32 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module FIFO1_PUSH_w13(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w14(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w15(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w16(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w17(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module WireDefault_w1_i025(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w14(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i025 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i025 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module WireDefault_w1_i026(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w15(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i026 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i026 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module WireDefault_w1_i027(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w16(
  input         clk,
                rst,
  output        full_ready,
                full_res0,
  input         deq_enable,
  output        deq_ready,
  output [15:0] deq_res0,
  input         enq_enable,
  output        enq_ready,
  input  [15:0] enq_data
);

  wire [15:0] _enq_value_read_data;
  wire        _enqed_read_res0;
  wire        _deqed_read_res0;
  wire [1:0]  _state_read_data;
  wire [15:0] _reg1_read_data;
  wire [15:0] _reg0_read_data;
  wire        _GEN = (|_state_read_data) & deq_enable;
  wire        _GEN_0 = _state_read_data != 2'h2;
  wire        _GEN_1 = _GEN_0 & enq_enable;
  wire        _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire        _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire        _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire        _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width16_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data   (_GEN_5 ? _reg1_read_data : _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i027 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i027 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w16 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module FIFO1_PUSH_w18(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w19(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w110(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w111(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w112(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module WireDefault_w1_i028(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w17(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i028 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i028 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module WireDefault_w1_i029(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w18(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i029 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i029 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module FIFO1_PUSH_w113(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w114(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w115(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w116(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module WireDefault_w1_i030(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w19(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i030 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i030 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module WireDefault_w1_i031(
  output read_ready,
         read_res0,
  input  write_enable,
  output write_ready,
  input  write_in_
);

  Wire_w1 inner (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (write_enable & write_in_),
    .read_data    (read_res0),
    .read_ready   ()
  );
  assign read_ready = 1'h1;
  assign write_ready = 1'h1;
endmodule

module FIFO2_I_w110(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire       _enq_value_read_data;
  wire       _enqed_read_res0;
  wire       _deqed_read_res0;
  wire [1:0] _state_read_data;
  wire       _reg1_read_data;
  wire       _reg0_read_data;
  wire       _GEN = (|_state_read_data) & deq_enable;
  wire       _GEN_0 = _state_read_data != 2'h2;
  wire       _GEN_1 = _GEN_0 & enq_enable;
  wire       _GEN_2 = ~(|_state_read_data) & _enqed_read_res0;
  wire       _GEN_3 = _state_read_data == 2'h1 & ~_GEN_2;
  wire       _GEN_4 = _GEN_3 & _enqed_read_res0;
  wire       _GEN_5 = _state_read_data == 2'h2 & _deqed_read_res0 & ~_GEN_2 & ~_GEN_3;
  Reg_width1_init0 reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5 | _GEN_4 | _GEN_2),
    .write_data
      (_GEN_5
         ? _reg1_read_data
         : _GEN_3
             ? (_enqed_read_res0 | _GEN_2) & _enq_value_read_data
             : _GEN_2 & _enq_value_read_data),
    .read_ready   (),
    .read_data    (_reg0_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_4 & ~_deqed_read_res0),
    .write_data   (_GEN_4 & ~_deqed_read_res0 & _reg0_read_data),
    .read_ready   (),
    .read_data    (_reg1_read_data),
    .write_ready  ()
  );
  Reg_width2_init0 state (
    .clock        (clk),
    .reset        (rst),
    .write_enable
      (_GEN_5
       | (_GEN_3
            ? (_enqed_read_res0 ? ~_deqed_read_res0 | _GEN_2 : _deqed_read_res0 | _GEN_2)
            : _GEN_2)),
    .write_data
      (_GEN_5 | ~_GEN_3
         ? 2'h1
         : _enqed_read_res0
             ? (_deqed_read_res0 ? 2'h1 : 2'h2)
             : {1'h0, ~_deqed_read_res0}),
    .read_ready   (),
    .read_data    (_state_read_data),
    .write_ready  ()
  );
  WireDefault_w1_i031 deqed (
    .read_ready   (),
    .read_res0    (_deqed_read_res0),
    .write_enable (_GEN),
    .write_ready  (),
    .write_in_    (_GEN)
  );
  WireDefault_w1_i031 enqed (
    .read_ready   (),
    .read_res0    (_enqed_read_res0),
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_in_    (_GEN_1)
  );
  Wire_w1 enq_value (
    .write_enable (_GEN_1),
    .write_ready  (),
    .write_data   (_GEN_1 & enq_data),
    .read_data    (_enq_value_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _state_read_data == 2'h2;
  assign deq_ready = |_state_read_data;
  assign deq_res0 = _reg0_read_data;
  assign enq_ready = _GEN_0;
endmodule

module FIFO1_PUSH_w117(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w118(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w119(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w120(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w121(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module FIFO1_PUSH_w122(
  input  clk,
         rst,
  output full_ready,
         full_res0,
  input  deq_enable,
  output deq_ready,
         deq_res0,
  input  enq_enable,
  output enq_ready,
  input  enq_data
);

  wire _enqed_read_data;
  wire _deqed_read_data;
  wire _full_reg_read_data;
  wire _GEN = ~_full_reg_read_data | _deqed_read_data;
  wire _GEN_0 = _GEN & enq_enable;
  Reg_width1_init0 reg_data (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   (_GEN_0 & enq_data),
    .read_ready   (),
    .read_data    (deq_res0),
    .write_ready  ()
  );
  Reg_width1_init0 full_reg (
    .clock        (clk),
    .reset        (rst),
    .write_enable (1'h1),
    .write_data   (_enqed_read_data | _full_reg_read_data & ~_deqed_read_data),
    .read_ready   (),
    .read_data    (_full_reg_read_data),
    .write_ready  ()
  );
  Wire_w1 deqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_full_reg_read_data & deq_enable),
    .read_data    (_deqed_read_data),
    .read_ready   ()
  );
  Wire_w1 enqed (
    .write_enable (1'h1),
    .write_ready  (),
    .write_data   (_GEN_0),
    .read_data    (_enqed_read_data),
    .read_ready   ()
  );
  assign full_ready = 1'h1;
  assign full_res0 = _full_reg_read_data;
  assign deq_ready = _full_reg_read_data;
  assign enq_ready = _GEN;
endmodule

module main(
  input         clk,
                rst,
                burst_read_0_enable,
  output        burst_read_0_ready,
  input  [31:0] burst_read_0_addr,
  input         burst_read_1_enable,
  output        burst_read_1_ready,
  output [63:0] burst_read_1_res0,
  input         burst_write_enable,
  output        burst_write_ready,
  input  [31:0] burst_write_addr,
  input  [63:0] burst_write_data,
  input         rocc_cmd_enable,
  output        rocc_cmd_ready,
  input  [6:0]  rocc_cmd_rocc_cmd_funct,
  input  [4:0]  rocc_cmd_rocc_cmd_rs1,
                rocc_cmd_rocc_cmd_rs2,
                rocc_cmd_rocc_cmd_rd,
  input         rocc_cmd_rocc_cmd_xs1,
                rocc_cmd_rocc_cmd_xs2,
                rocc_cmd_rocc_cmd_xd,
  input  [6:0]  rocc_cmd_rocc_cmd_opcode,
  input  [31:0] rocc_cmd_rocc_cmd_rs1data,
                rocc_cmd_rocc_cmd_rs2data,
  input         hella_resp_enable,
  output        hella_resp_ready,
  input  [31:0] hella_resp_hella_resp_data,
  input  [7:0]  hella_resp_hella_resp_tag,
  input  [4:0]  hella_resp_hella_resp_cmd,
  input  [1:0]  hella_resp_hella_resp_size,
  input         hella_resp_hella_resp_signed,
  output [31:0] dma_cpu_to_isax_ch0_cpu_addr,
                dma_cpu_to_isax_ch0_isax_addr,
  output [3:0]  dma_cpu_to_isax_ch0_length,
  output [7:0]  dma_cpu_to_isax_ch0_stride_x,
                dma_cpu_to_isax_ch0_stride_y,
  input         dma_cpu_to_isax_ch0_ready,
  output        dma_cpu_to_isax_ch0_enable,
  output [31:0] dma_isax_to_cpu_ch0_cpu_addr,
                dma_isax_to_cpu_ch0_isax_addr,
  output [3:0]  dma_isax_to_cpu_ch0_length,
  output [7:0]  dma_isax_to_cpu_ch0_stride_x,
                dma_isax_to_cpu_ch0_stride_y,
  input         dma_isax_to_cpu_ch0_ready,
  output        dma_isax_to_cpu_ch0_enable,
  input         dma_poll_for_idle_ch0_res0,
                dma_poll_for_idle_ch0_ready,
  output [31:0] dma_cpu_to_isax_ch1_cpu_addr,
                dma_cpu_to_isax_ch1_isax_addr,
  output [3:0]  dma_cpu_to_isax_ch1_length,
  output [7:0]  dma_cpu_to_isax_ch1_stride_x,
                dma_cpu_to_isax_ch1_stride_y,
  input         dma_cpu_to_isax_ch1_ready,
  output        dma_cpu_to_isax_ch1_enable,
  output [31:0] dma_isax_to_cpu_ch1_cpu_addr,
                dma_isax_to_cpu_ch1_isax_addr,
  output [3:0]  dma_isax_to_cpu_ch1_length,
  output [7:0]  dma_isax_to_cpu_ch1_stride_x,
                dma_isax_to_cpu_ch1_stride_y,
  input         dma_isax_to_cpu_ch1_ready,
  output        dma_isax_to_cpu_ch1_enable,
  input         dma_poll_for_idle_ch1_res0,
                dma_poll_for_idle_ch1_ready,
  output [4:0]  rocc_resp_rocc_resp_to_bus_result_rd,
  output [31:0] rocc_resp_rocc_resp_to_bus_result_rddata,
  input         rocc_resp_rocc_resp_to_bus_ready,
  output        rocc_resp_rocc_resp_to_bus_enable,
  output [31:0] hella_cmd_hella_cmd_to_bus_cmd_addr,
  output [7:0]  hella_cmd_hella_cmd_to_bus_cmd_tag,
  output [4:0]  hella_cmd_hella_cmd_to_bus_cmd_cmd,
  output [1:0]  hella_cmd_hella_cmd_to_bus_cmd_size,
  output        hella_cmd_hella_cmd_to_bus_cmd_signed,
                hella_cmd_hella_cmd_to_bus_cmd_phys,
  output [31:0] hella_cmd_hella_cmd_to_bus_cmd_data,
  output [3:0]  hella_cmd_hella_cmd_to_bus_cmd_mask,
  input         hella_cmd_hella_cmd_to_bus_ready,
  output        hella_cmd_hella_cmd_to_bus_enable
);

  wire [31:0] _op2b38_b2_reg0_read_data;
  wire        _op2b38_b2_s14tok_full_res0;
  wire        _op2b38_b2_s14tok_deq_ready;
  wire        _op2b38_b2_s14tok_enq_ready;
  wire        _op2b38_b2_s12tok_full_res0;
  wire        _op2b38_b2_s12tok_deq_ready;
  wire        _op2b38_b2_s12tok_enq_ready;
  wire [31:0] _op2b38_l1_b0_reg21_read_data;
  wire [15:0] _op2b38_l1_b0_reg20_read_data;
  wire [15:0] _op2b38_l1_b0_reg19_read_data;
  wire [15:0] _op2b38_l1_b0_reg18_read_data;
  wire [15:0] _op2b38_l1_b0_reg17_read_data;
  wire [15:0] _op2b38_l1_b0_reg16_read_data;
  wire [15:0] _op2b38_l1_b0_reg15_read_data;
  wire [15:0] _op2b38_l1_b0_reg14_read_data;
  wire [15:0] _op2b38_l1_b0_reg13_read_data;
  wire [15:0] _op2b38_l1_b0_reg12_read_data;
  wire [15:0] _op2b38_l1_b0_reg11_read_data;
  wire [15:0] _op2b38_l1_b0_reg10_read_data;
  wire [15:0] _op2b38_l1_b0_reg9_read_data;
  wire [15:0] _op2b38_l1_b0_reg8_read_data;
  wire [15:0] _op2b38_l1_b0_reg7_read_data;
  wire [15:0] _op2b38_l1_b0_reg6_read_data;
  wire [15:0] _op2b38_l1_b0_reg5_read_data;
  wire [15:0] _op2b38_l1_b0_reg4_read_data;
  wire [15:0] _op2b38_l1_b0_reg3_read_data;
  wire [15:0] _op2b38_l1_b0_reg2_read_data;
  wire [15:0] _op2b38_l1_b0_reg1_read_data;
  wire [15:0] _op2b38_l1_b0_reg0_read_data;
  wire        _op2b38_l1_b0_s9tok_full_res0;
  wire        _op2b38_l1_b0_s9tok_deq_ready;
  wire        _op2b38_l1_b0_s9tok_enq_ready;
  wire        _op2b38_l1_b0_s8tok_full_res0;
  wire        _op2b38_l1_b0_s8tok_deq_ready;
  wire        _op2b38_l1_b0_s8tok_enq_ready;
  wire        _op2b38_l1_b0_s7tok_full_res0;
  wire        _op2b38_l1_b0_s7tok_deq_ready;
  wire        _op2b38_l1_b0_s7tok_enq_ready;
  wire        _op2b38_l1_b0_s6tok_full_res0;
  wire        _op2b38_l1_b0_s6tok_deq_ready;
  wire        _op2b38_l1_b0_s6tok_enq_ready;
  wire [31:0] _op2b38_l1_iv_read_data;
  wire [95:0] _op2b38_l1_st_read_data;
  wire        _op2b38_l1_dntok_deq_ready;
  wire        _op2b38_l1_dntok_enq_ready;
  wire        _op2b38_l1_entok_deq_ready;
  wire        _op2b38_l1_entok_enq_ready;
  wire [31:0] _op2b38_b0_reg0_read_data;
  wire        _op2b38_b0_s3tok_full_res0;
  wire        _op2b38_b0_s3tok_deq_ready;
  wire        _op2b38_b0_s3tok_enq_ready;
  wire        _op2b38_b0_s2tok_full_res0;
  wire        _op2b38_b0_s2tok_deq_ready;
  wire        _op2b38_b0_s2tok_enq_ready;
  wire        _op2b38_b0_s1tok_full_res0;
  wire        _op2b38_b0_s1tok_deq_ready;
  wire        _op2b38_b0_s1tok_enq_ready;
  wire        _op2b38_b0_s0tok_full_res0;
  wire        _op2b38_b0_s0tok_deq_ready;
  wire        _op2b38_b0_s0tok_enq_ready;
  wire        _op2b38_b1b2tok_deq_ready;
  wire        _op2b38_b1b2tok_enq_ready;
  wire        _op2b38_b0b1tok_deq_ready;
  wire        _op2b38_b0b1tok_enq_ready;
  wire [31:0] _op2b38_b0v0_read_data;
  wire [4:0]  _reg_rd_2b38_read_data;
  wire [31:0] _op0b05_b3_reg0_read_data;
  wire        _op0b05_b3_s22tok_full_res0;
  wire        _op0b05_b3_s22tok_deq_ready;
  wire        _op0b05_b3_s22tok_enq_ready;
  wire        _op0b05_b3_s20tok_full_res0;
  wire        _op0b05_b3_s20tok_deq_ready;
  wire        _op0b05_b3_s20tok_enq_ready;
  wire [31:0] _op0b05_l2_b0_reg9_read_data;
  wire [31:0] _op0b05_l2_b0_reg8_read_data;
  wire [31:0] _op0b05_l2_b0_reg7_read_data;
  wire [31:0] _op0b05_l2_b0_reg6_read_data;
  wire [31:0] _op0b05_l2_b0_reg5_read_data;
  wire [31:0] _op0b05_l2_b0_reg4_read_data;
  wire [31:0] _op0b05_l2_b0_reg3_read_data;
  wire [31:0] _op0b05_l2_b0_reg2_read_data;
  wire [31:0] _op0b05_l2_b0_reg1_read_data;
  wire [31:0] _op0b05_l2_b0_reg0_read_data;
  wire        _op0b05_l2_b0_s17tok_full_res0;
  wire        _op0b05_l2_b0_s17tok_deq_ready;
  wire        _op0b05_l2_b0_s17tok_enq_ready;
  wire        _op0b05_l2_b0_s16tok_full_res0;
  wire        _op0b05_l2_b0_s16tok_deq_ready;
  wire        _op0b05_l2_b0_s16tok_enq_ready;
  wire        _op0b05_l2_b0_s15tok_full_res0;
  wire        _op0b05_l2_b0_s15tok_deq_ready;
  wire        _op0b05_l2_b0_s15tok_enq_ready;
  wire        _op0b05_l2_in0_deq_ready;
  wire [15:0] _op0b05_l2_in0_deq_res0;
  wire        _op0b05_l2_in0_enq_ready;
  wire [15:0] _op0b05_l2_isr0_read_data;
  wire [31:0] _op0b05_l2_iv_read_data;
  wire [95:0] _op0b05_l2_st_read_data;
  wire        _op0b05_l2_dntok_deq_ready;
  wire        _op0b05_l2_dntok_enq_ready;
  wire        _op0b05_l2_entok_deq_ready;
  wire        _op0b05_l2_entok_enq_ready;
  wire [31:0] _op0b05_l1_b0_reg21_read_data;
  wire [63:0] _op0b05_l1_b0_reg20_read_data;
  wire [31:0] _op0b05_l1_b0_reg19_read_data;
  wire [31:0] _op0b05_l1_b0_reg18_read_data;
  wire [63:0] _op0b05_l1_b0_reg17_read_data;
  wire [31:0] _op0b05_l1_b0_reg16_read_data;
  wire [31:0] _op0b05_l1_b0_reg15_read_data;
  wire [63:0] _op0b05_l1_b0_reg14_read_data;
  wire [31:0] _op0b05_l1_b0_reg13_read_data;
  wire        _op0b05_l1_b0_reg12_read_data;
  wire [31:0] _op0b05_l1_b0_reg11_read_data;
  wire        _op0b05_l1_b0_reg10_read_data;
  wire [31:0] _op0b05_l1_b0_reg9_read_data;
  wire [31:0] _op0b05_l1_b0_reg8_read_data;
  wire        _op0b05_l1_b0_reg7_read_data;
  wire [31:0] _op0b05_l1_b0_reg6_read_data;
  wire [31:0] _op0b05_l1_b0_reg5_read_data;
  wire [31:0] _op0b05_l1_b0_reg4_read_data;
  wire [63:0] _op0b05_l1_b0_reg3_read_data;
  wire        _op0b05_l1_b0_reg2_read_data;
  wire [31:0] _op0b05_l1_b0_reg1_read_data;
  wire [7:0]  _op0b05_l1_b0_reg0_read_data;
  wire        _op0b05_l1_b0_s11tok_full_res0;
  wire        _op0b05_l1_b0_s11tok_deq_ready;
  wire        _op0b05_l1_b0_s11tok_enq_ready;
  wire        _op0b05_l1_b0_s10tok_full_res0;
  wire        _op0b05_l1_b0_s10tok_deq_ready;
  wire        _op0b05_l1_b0_s10tok_enq_ready;
  wire        _op0b05_l1_b0_s9tok_full_res0;
  wire        _op0b05_l1_b0_s9tok_deq_ready;
  wire        _op0b05_l1_b0_s9tok_enq_ready;
  wire        _op0b05_l1_b0_s8tok_full_res0;
  wire        _op0b05_l1_b0_s8tok_deq_ready;
  wire        _op0b05_l1_b0_s8tok_enq_ready;
  wire        _op0b05_l1_b0_s7tok_full_res0;
  wire        _op0b05_l1_b0_s7tok_deq_ready;
  wire        _op0b05_l1_b0_s7tok_enq_ready;
  wire        _op0b05_l1_in0_deq_ready;
  wire [31:0] _op0b05_l1_in0_deq_res0;
  wire        _op0b05_l1_in0_enq_ready;
  wire [31:0] _op0b05_l1_isr0_read_data;
  wire [31:0] _op0b05_l1_iv_read_data;
  wire [95:0] _op0b05_l1_st_read_data;
  wire        _op0b05_l1_dntok_deq_ready;
  wire        _op0b05_l1_dntok_enq_ready;
  wire        _op0b05_l1_entok_deq_ready;
  wire        _op0b05_l1_entok_enq_ready;
  wire [31:0] _op0b05_b0_reg1_read_data;
  wire [31:0] _op0b05_b0_reg0_read_data;
  wire        _op0b05_b0_s3tok_full_res0;
  wire        _op0b05_b0_s3tok_deq_ready;
  wire        _op0b05_b0_s3tok_enq_ready;
  wire        _op0b05_b0_s2tok_full_res0;
  wire        _op0b05_b0_s2tok_deq_ready;
  wire        _op0b05_b0_s2tok_enq_ready;
  wire        _op0b05_b0_s1tok_full_res0;
  wire        _op0b05_b0_s1tok_deq_ready;
  wire        _op0b05_b0_s1tok_enq_ready;
  wire        _op0b05_b0_s0tok_full_res0;
  wire        _op0b05_b0_s0tok_deq_ready;
  wire        _op0b05_b0_s0tok_enq_ready;
  wire        _op0b05_b2b3tok_deq_ready;
  wire        _op0b05_b2b3tok_enq_ready;
  wire        _op0b05_b1b2tok_deq_ready;
  wire        _op0b05_b1b2tok_enq_ready;
  wire        _op0b05_b0b1tok_deq_ready;
  wire        _op0b05_b0b1tok_enq_ready;
  wire [15:0] _op0b05_b0v2_read_data;
  wire [31:0] _op0b05_b0v1_read_data;
  wire [31:0] _op0b05_b0v0_read_data;
  wire [4:0]  _reg_rd_0b05_read_data;
  wire        _hellacache_adapter_resp_from_bus_ready;
  wire        _hellacache_adapter_cmd_from_user_ready;
  wire        _hellacache_adapter_resp_to_user_ready;
  wire [31:0] _hellacache_adapter_resp_to_user_res0_data;
  wire        _rocc_adapter_cmd_from_bus_ready;
  wire        _rocc_adapter_resp_from_user_ready;
  wire        _rocc_adapter_cmd_to_user_0b05_ready;
  wire [4:0]  _rocc_adapter_cmd_to_user_0b05_res0_rd;
  wire [31:0] _rocc_adapter_cmd_to_user_0b05_res0_rs1data;
  wire [31:0] _rocc_adapter_cmd_to_user_0b05_res0_rs2data;
  wire        _rocc_adapter_cmd_to_user_2b38_ready;
  wire [4:0]  _rocc_adapter_cmd_to_user_2b38_res0_rd;
  wire [31:0] _rocc_adapter_cmd_to_user_2b38_res0_rs1data;
  wire [31:0] _rocc_adapter_cmd_to_user_2b38_res0_rs2data;
  wire        _scratchpad_pool_burst_read_0_ready;
  wire        _scratchpad_pool_burst_read_1_ready;
  wire        _scratchpad_pool_burst_write_ready;
  wire        _scratchpad_pool_decompressed_weights_0_write_ready;
  wire        _scratchpad_pool_decompressed_weights_1_write_ready;
  wire        _scratchpad_pool_decompressed_weights_2_write_ready;
  wire        _scratchpad_pool_decompressed_weights_3_write_ready;
  wire [7:0]  _scratchpad_pool_dense_values_0_read_1_res0;
  wire [7:0]  _scratchpad_pool_dense_values_1_read_1_res0;
  wire [7:0]  _scratchpad_pool_dense_values_2_read_1_res0;
  wire [7:0]  _scratchpad_pool_dense_values_3_read_1_res0;
  wire        _scratchpad_pool_matrix_a_0_read_0_ready;
  wire [15:0] _scratchpad_pool_matrix_a_0_read_1_res0;
  wire        _scratchpad_pool_matrix_a_1_read_0_ready;
  wire [15:0] _scratchpad_pool_matrix_a_1_read_1_res0;
  wire        _scratchpad_pool_matrix_a_2_read_0_ready;
  wire [15:0] _scratchpad_pool_matrix_a_2_read_1_res0;
  wire        _scratchpad_pool_matrix_a_3_read_0_ready;
  wire [15:0] _scratchpad_pool_matrix_a_3_read_1_res0;
  wire        _scratchpad_pool_matrix_b_0_read_0_ready;
  wire [15:0] _scratchpad_pool_matrix_b_0_read_1_res0;
  wire        _scratchpad_pool_matrix_b_1_read_0_ready;
  wire [15:0] _scratchpad_pool_matrix_b_1_read_1_res0;
  wire        _scratchpad_pool_matrix_b_2_read_0_ready;
  wire [15:0] _scratchpad_pool_matrix_b_2_read_1_res0;
  wire        _scratchpad_pool_matrix_b_3_read_0_ready;
  wire [15:0] _scratchpad_pool_matrix_b_3_read_1_res0;
  wire        _scratchpad_pool_matrix_c_0_write_ready;
  wire        _scratchpad_pool_matrix_c_1_write_ready;
  wire        _scratchpad_pool_matrix_c_2_write_ready;
  wire        _scratchpad_pool_matrix_c_3_write_ready;
  wire        _scratchpad_pool_values_read_0_ready;
  wire [63:0] _scratchpad_pool_values_read_1_res0;
  wire [31:0] _glbl_reg_vidx_read_data;
  wire        _GEN = _rocc_adapter_cmd_to_user_0b05_ready & _op0b05_b0_s0tok_enq_ready;
  wire        _GEN_0 =
    _op0b05_l1_b0_s10tok_full_res0 & _op0b05_l1_b0_s10tok_deq_ready
    & _scratchpad_pool_values_read_0_ready & _op0b05_l1_b0_s11tok_enq_ready & ~_GEN;
  wire [63:0] _GEN_1 =
    $signed($signed(_scratchpad_pool_values_read_1_res0)
            >>> _op0b05_l1_b0_reg17_read_data);
  wire        _GEN_2 =
    _op0b05_l1_b0_s9tok_full_res0 & _op0b05_l1_b0_s9tok_deq_ready
    & _scratchpad_pool_values_read_0_ready & _op0b05_l1_b0_s10tok_enq_ready & ~_GEN_0;
  wire [63:0] _GEN_3 =
    $signed($signed(_scratchpad_pool_values_read_1_res0)
            >>> _op0b05_l1_b0_reg14_read_data);
  wire [31:0] _GEN_4 = _op0b05_l1_b0_reg15_read_data + _op0b05_l1_b0_reg11_read_data;
  wire        _GEN_5 =
    _op0b05_l1_b0_s8tok_full_res0 & _op0b05_l1_b0_s8tok_deq_ready
    & _scratchpad_pool_values_read_0_ready & _op0b05_l1_b0_s9tok_enq_ready & ~_GEN_0
    & ~_GEN_2;
  wire [31:0] _GEN_6 = _op0b05_l1_b0_reg6_read_data + _op0b05_l1_b0_reg9_read_data;
  wire        _GEN_7 =
    _op0b05_l1_entok_deq_ready & _op0b05_l1_in0_deq_ready
    & _scratchpad_pool_values_read_0_ready & _op0b05_l1_b0_s7tok_enq_ready & ~_GEN_0
    & ~_GEN_2 & ~_GEN_5;
  wire [7:0]  _GEN_8 =
    _op0b05_l1_iv_read_data < 32'h2
      ? _op0b05_l1_in0_deq_res0[7:0]
      : _op0b05_l1_iv_read_data < 32'h4
          ? _op0b05_l1_in0_deq_res0[15:8]
          : _op0b05_l1_iv_read_data < 32'h6
              ? _op0b05_l1_in0_deq_res0[23:16]
              : _op0b05_l1_in0_deq_res0[31:24];
  wire [7:0]  _GEN_9 = _GEN_8 >> {5'h0, _op0b05_l1_iv_read_data[0], 2'h0};
  wire        _GEN_10 =
    _op0b05_b0_s2tok_full_res0 & _op0b05_b0_s2tok_deq_ready
    & _hellacache_adapter_resp_to_user_ready & _op0b05_b0_s3tok_enq_ready;
  wire        _GEN_11 =
    _op0b05_b0_s1tok_full_res0 & _op0b05_b0_s1tok_deq_ready
    & _hellacache_adapter_cmd_from_user_ready & _hellacache_adapter_resp_to_user_ready
    & _op0b05_b0_s2tok_enq_ready & ~_GEN_10;
  wire        _GEN_12 =
    _op0b05_b0_s0tok_full_res0 & _op0b05_b0_s0tok_deq_ready & dma_cpu_to_isax_ch0_ready
    & _hellacache_adapter_cmd_from_user_ready & _op0b05_b0_s1tok_enq_ready & ~_GEN_11;
  wire [31:0] _GEN_13 = _op0b05_b0_reg0_read_data + 32'h44;
  wire        _GEN_14 =
    _op0b05_b0_s3tok_full_res0 & _op0b05_b0_s3tok_deq_ready & dma_poll_for_idle_ch0_ready
    & _op0b05_b0b1tok_enq_ready;
  wire        _GEN_15 =
    _op0b05_l1_b0_s7tok_full_res0 & _op0b05_l1_b0_s7tok_deq_ready
    & _op0b05_l1_b0_s8tok_enq_ready;
  wire [63:0] _GEN_16 =
    $signed($signed(_scratchpad_pool_values_read_1_res0)
            >>> _op0b05_l1_b0_reg3_read_data);
  wire [31:0] _GEN_17 = _op0b05_l1_b0_reg4_read_data + _op0b05_l1_b0_reg5_read_data;
  wire [7:0]  _GEN_18 =
    _op0b05_l1_b0_reg0_read_data >> _op0b05_l1_b0_reg1_read_data[7:0] + 8'h1;
  wire [7:0]  _GEN_19 =
    _op0b05_l1_b0_reg0_read_data >> _op0b05_l1_b0_reg1_read_data[7:0] + 8'h2;
  wire [7:0]  _GEN_20 =
    _op0b05_l1_b0_reg0_read_data >> _op0b05_l1_b0_reg1_read_data[7:0] + 8'h3;
  wire        _GEN_21 =
    _op0b05_l1_b0_s11tok_full_res0 & _op0b05_l1_b0_s11tok_deq_ready
    & _op0b05_l1_dntok_enq_ready;
  wire [63:0] _GEN_22 =
    $signed($signed(_scratchpad_pool_values_read_1_res0)
            >>> _op0b05_l1_b0_reg20_read_data);
  wire        _GEN_23 =
    _op0b05_b0b1tok_deq_ready & _op0b05_l1_in0_enq_ready & _op0b05_l1_entok_enq_ready
    & _op0b05_b1b2tok_enq_ready;
  wire        _GEN_24 =
    _op0b05_l1_dntok_deq_ready & _op0b05_l1_in0_enq_ready & _op0b05_l1_entok_enq_ready
    & _op0b05_b1b2tok_enq_ready & ~_GEN_23;
  wire [32:0] _GEN_25 =
    {1'h0, _op0b05_l1_st_read_data[95:64]} + {1'h0, _op0b05_l1_st_read_data[31:0]};
  wire [32:0] _GEN_26 = {1'h0, _op0b05_l1_st_read_data[63:32]};
  wire        _GEN_27 = _GEN_24 & _GEN_25 <= _GEN_26;
  wire        _GEN_28 = _GEN_27 | _GEN_23;
  wire        _GEN_29 = _GEN_24 & _GEN_25 > _GEN_26;
  wire        _GEN_30 =
    _op0b05_l2_entok_deq_ready & _op0b05_l2_in0_deq_ready
    & _op0b05_l2_b0_s15tok_enq_ready;
  wire        _GEN_31 =
    _op0b05_l2_b0_s15tok_full_res0 & _op0b05_l2_b0_s15tok_deq_ready
    & _op0b05_l2_b0_s16tok_enq_ready;
  wire        _GEN_32 =
    _op0b05_l2_b0_s16tok_full_res0 & _op0b05_l2_b0_s16tok_deq_ready
    & _op0b05_l2_b0_s17tok_enq_ready;
  wire        _GEN_33 =
    _op0b05_l2_b0_s17tok_full_res0 & _op0b05_l2_b0_s17tok_deq_ready
    & _scratchpad_pool_decompressed_weights_0_write_ready
    & _scratchpad_pool_decompressed_weights_1_write_ready
    & _scratchpad_pool_decompressed_weights_2_write_ready
    & _scratchpad_pool_decompressed_weights_3_write_ready & _op0b05_l2_dntok_enq_ready;
  wire        _GEN_34 =
    _op0b05_b1b2tok_deq_ready & _op0b05_l2_in0_enq_ready & _op0b05_l2_entok_enq_ready
    & _op0b05_b2b3tok_enq_ready;
  wire        _GEN_35 =
    _op0b05_l2_dntok_deq_ready & _op0b05_l2_in0_enq_ready & _op0b05_l2_entok_enq_ready
    & _op0b05_b2b3tok_enq_ready & ~_GEN_34;
  wire [32:0] _GEN_36 =
    {1'h0, _op0b05_l2_st_read_data[95:64]} + {1'h0, _op0b05_l2_st_read_data[31:0]};
  wire [32:0] _GEN_37 = {1'h0, _op0b05_l2_st_read_data[63:32]};
  wire        _GEN_38 = _GEN_35 & _GEN_36 <= _GEN_37;
  wire        _GEN_39 = _GEN_38 | _GEN_34;
  wire        _GEN_40 = _GEN_35 & _GEN_36 > _GEN_37;
  wire        _GEN_41 = _op0b05_b2b3tok_deq_ready & _op0b05_b3_s20tok_enq_ready;
  wire        _GEN_42 =
    _op0b05_b3_s20tok_full_res0 & _op0b05_b3_s20tok_deq_ready & dma_isax_to_cpu_ch0_ready
    & _op0b05_b3_s22tok_enq_ready;
  wire        _GEN_43 =
    _op0b05_b3_s22tok_full_res0 & _op0b05_b3_s22tok_deq_ready
    & dma_poll_for_idle_ch0_ready & _rocc_adapter_resp_from_user_ready;
  wire        _GEN_44 =
    _op2b38_b2_s14tok_full_res0 & _op2b38_b2_s14tok_deq_ready
    & dma_poll_for_idle_ch0_ready & _rocc_adapter_resp_from_user_ready & ~_GEN_43;
  wire        _GEN_45 = _rocc_adapter_cmd_to_user_2b38_ready & _op2b38_b0_s0tok_enq_ready;
  wire        _GEN_46 =
    _op2b38_b0_s0tok_full_res0 & _op2b38_b0_s0tok_deq_ready & dma_cpu_to_isax_ch0_ready
    & _op2b38_b0_s1tok_enq_ready;
  wire        _GEN_47 =
    _op2b38_b0_s1tok_full_res0 & _op2b38_b0_s1tok_deq_ready & dma_cpu_to_isax_ch1_ready
    & _op2b38_b0_s2tok_enq_ready;
  wire        _GEN_48 =
    _op2b38_b0_s2tok_full_res0 & _op2b38_b0_s2tok_deq_ready & dma_poll_for_idle_ch0_ready
    & _op2b38_b0_s3tok_enq_ready;
  wire        _GEN_49 =
    _op2b38_b0_s3tok_full_res0 & _op2b38_b0_s3tok_deq_ready & dma_poll_for_idle_ch1_ready
    & _op2b38_b0b1tok_enq_ready;
  wire        _GEN_50 =
    _op2b38_l1_b0_s8tok_full_res0 & _op2b38_l1_b0_s8tok_deq_ready
    & _scratchpad_pool_matrix_a_0_read_0_ready & _scratchpad_pool_matrix_b_0_read_0_ready
    & _scratchpad_pool_matrix_a_1_read_0_ready & _scratchpad_pool_matrix_a_2_read_0_ready
    & _scratchpad_pool_matrix_a_3_read_0_ready & _scratchpad_pool_matrix_b_1_read_0_ready
    & _scratchpad_pool_matrix_b_2_read_0_ready & _scratchpad_pool_matrix_b_3_read_0_ready
    & _op2b38_l1_b0_s9tok_enq_ready;
  wire        _GEN_51 =
    _op2b38_l1_b0_s7tok_full_res0 & _op2b38_l1_b0_s7tok_deq_ready
    & _scratchpad_pool_matrix_b_0_read_0_ready & _scratchpad_pool_matrix_a_0_read_0_ready
    & _scratchpad_pool_matrix_a_1_read_0_ready & _scratchpad_pool_matrix_b_1_read_0_ready
    & _scratchpad_pool_matrix_a_2_read_0_ready & _scratchpad_pool_matrix_a_3_read_0_ready
    & _scratchpad_pool_matrix_b_2_read_0_ready & _scratchpad_pool_matrix_b_3_read_0_ready
    & _op2b38_l1_b0_s8tok_enq_ready & ~_GEN_50;
  wire        _GEN_52 =
    _op2b38_l1_b0_s6tok_full_res0 & _op2b38_l1_b0_s6tok_deq_ready
    & _scratchpad_pool_matrix_b_0_read_0_ready & _scratchpad_pool_matrix_b_1_read_0_ready
    & _scratchpad_pool_matrix_a_0_read_0_ready & _scratchpad_pool_matrix_a_1_read_0_ready
    & _scratchpad_pool_matrix_a_2_read_0_ready & _scratchpad_pool_matrix_b_2_read_0_ready
    & _scratchpad_pool_matrix_a_3_read_0_ready & _scratchpad_pool_matrix_b_3_read_0_ready
    & _op2b38_l1_b0_s7tok_enq_ready & ~_GEN_50 & ~_GEN_51;
  wire        _GEN_53 =
    _op2b38_l1_entok_deq_ready & _scratchpad_pool_matrix_b_0_read_0_ready
    & _scratchpad_pool_matrix_b_1_read_0_ready & _scratchpad_pool_matrix_b_2_read_0_ready
    & _scratchpad_pool_matrix_a_0_read_0_ready & _scratchpad_pool_matrix_a_1_read_0_ready
    & _scratchpad_pool_matrix_a_2_read_0_ready & _scratchpad_pool_matrix_a_3_read_0_ready
    & _scratchpad_pool_matrix_b_3_read_0_ready & _op2b38_l1_b0_s6tok_enq_ready & ~_GEN_50
    & ~_GEN_51 & ~_GEN_52;
  wire        _GEN_54 = _GEN_53 | _GEN_52 | _GEN_51 | _GEN_50;
  wire [1:0]  _GEN_55 = _GEN_53 ? 2'h3 : _GEN_52 ? 2'h2 : {1'h0, _GEN_51};
  wire [1:0]  _GEN_56 =
    _GEN_53 ? _op2b38_l1_iv_read_data[1:0] : _op2b38_l1_b0_reg21_read_data[1:0];
  wire        _GEN_57 =
    _op2b38_l1_b0_s9tok_full_res0 & _op2b38_l1_b0_s9tok_deq_ready
    & _scratchpad_pool_matrix_c_0_write_ready & _scratchpad_pool_matrix_c_1_write_ready
    & _scratchpad_pool_matrix_c_2_write_ready & _scratchpad_pool_matrix_c_3_write_ready
    & _op2b38_l1_dntok_enq_ready;
  wire        _GEN_58 =
    _op2b38_b0b1tok_deq_ready & _op2b38_l1_entok_enq_ready & _op2b38_b1b2tok_enq_ready;
  wire        _GEN_59 =
    _op2b38_l1_dntok_deq_ready & _op2b38_l1_entok_enq_ready & _op2b38_b1b2tok_enq_ready
    & ~_GEN_58;
  wire [32:0] _GEN_60 =
    {1'h0, _op2b38_l1_st_read_data[95:64]} + {1'h0, _op2b38_l1_st_read_data[31:0]};
  wire [32:0] _GEN_61 = {1'h0, _op2b38_l1_st_read_data[63:32]};
  wire        _GEN_62 = _GEN_59 & _GEN_60 <= _GEN_61;
  wire        _GEN_63 = _GEN_62 | _GEN_58;
  wire        _GEN_64 = _GEN_59 & _GEN_60 > _GEN_61;
  wire        _GEN_65 = _op2b38_b1b2tok_deq_ready & _op2b38_b2_s12tok_enq_ready;
  wire        _GEN_66 =
    _op2b38_b2_s12tok_full_res0 & _op2b38_b2_s12tok_deq_ready & dma_isax_to_cpu_ch0_ready
    & _op2b38_b2_s14tok_enq_ready;
  Reg_width32_init0 glbl_reg_vidx (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0 | _GEN),
    .write_data
      (_GEN_0 ? _op0b05_l1_b0_reg18_read_data + _op0b05_l1_b0_reg13_read_data : 32'h0),
    .read_ready   (),
    .read_data    (_glbl_reg_vidx_read_data),
    .write_ready  ()
  );
  ScratchpadMemoryPool scratchpad_pool (
    .clk                                  (clk),
    .rst                                  (rst),
    .burst_read_0_enable
      (_scratchpad_pool_burst_read_0_ready & burst_read_0_enable),
    .burst_read_0_ready                   (_scratchpad_pool_burst_read_0_ready),
    .burst_read_0_addr                    (burst_read_0_addr),
    .burst_read_1_enable
      (_scratchpad_pool_burst_read_1_ready & burst_read_1_enable),
    .burst_read_1_ready                   (_scratchpad_pool_burst_read_1_ready),
    .burst_read_1_res0                    (burst_read_1_res0),
    .burst_write_enable
      (_scratchpad_pool_burst_write_ready & burst_write_enable),
    .burst_write_ready                    (_scratchpad_pool_burst_write_ready),
    .burst_write_addr                     (burst_write_addr),
    .burst_write_data                     (burst_write_data),
    .acc_read_0_enable                    (1'h0),
    .acc_read_0_ready                     (),
    .acc_read_0_addr                      (1'h0),
    .acc_read_1_ready                     (),
    .acc_read_1_res0                      (),
    .acc_write_enable                     (1'h0),
    .acc_write_ready                      (),
    .acc_write_addr                       (1'h0),
    .acc_write_data                       (16'h0),
    .decompressed_weights_0_read_0_enable (1'h0),
    .decompressed_weights_0_read_0_ready  (),
    .decompressed_weights_0_read_0_addr   (3'h0),
    .decompressed_weights_0_read_1_ready  (),
    .decompressed_weights_0_read_1_res0   (),
    .decompressed_weights_0_write_enable  (_GEN_33),
    .decompressed_weights_0_write_ready
      (_scratchpad_pool_decompressed_weights_0_write_ready),
    .decompressed_weights_0_write_addr    (_op0b05_l2_b0_reg9_read_data[2:0]),
    .decompressed_weights_0_write_data    (_op0b05_l2_b0_reg5_read_data[23:8]),
    .decompressed_weights_1_read_0_enable (1'h0),
    .decompressed_weights_1_read_0_ready  (),
    .decompressed_weights_1_read_0_addr   (3'h0),
    .decompressed_weights_1_read_1_ready  (),
    .decompressed_weights_1_read_1_res0   (),
    .decompressed_weights_1_write_enable  (_GEN_33),
    .decompressed_weights_1_write_ready
      (_scratchpad_pool_decompressed_weights_1_write_ready),
    .decompressed_weights_1_write_addr    (_op0b05_l2_b0_reg9_read_data[2:0]),
    .decompressed_weights_1_write_data    (_op0b05_l2_b0_reg6_read_data[23:8]),
    .decompressed_weights_2_read_0_enable (1'h0),
    .decompressed_weights_2_read_0_ready  (),
    .decompressed_weights_2_read_0_addr   (3'h0),
    .decompressed_weights_2_read_1_ready  (),
    .decompressed_weights_2_read_1_res0   (),
    .decompressed_weights_2_write_enable  (_GEN_33),
    .decompressed_weights_2_write_ready
      (_scratchpad_pool_decompressed_weights_2_write_ready),
    .decompressed_weights_2_write_addr    (_op0b05_l2_b0_reg9_read_data[2:0]),
    .decompressed_weights_2_write_data    (_op0b05_l2_b0_reg7_read_data[23:8]),
    .decompressed_weights_3_read_0_enable (1'h0),
    .decompressed_weights_3_read_0_ready  (),
    .decompressed_weights_3_read_0_addr   (3'h0),
    .decompressed_weights_3_read_1_ready  (),
    .decompressed_weights_3_read_1_res0   (),
    .decompressed_weights_3_write_enable  (_GEN_33),
    .decompressed_weights_3_write_ready
      (_scratchpad_pool_decompressed_weights_3_write_ready),
    .decompressed_weights_3_write_addr    (_op0b05_l2_b0_reg9_read_data[2:0]),
    .decompressed_weights_3_write_data    (_op0b05_l2_b0_reg8_read_data[23:8]),
    .dense_values_0_read_0_enable         (_GEN_30),
    .dense_values_0_read_0_ready          (),
    .dense_values_0_read_0_addr           (_op0b05_l2_iv_read_data[2:0]),
    .dense_values_0_read_1_ready          (),
    .dense_values_0_read_1_res0           (_scratchpad_pool_dense_values_0_read_1_res0),
    .dense_values_0_write_enable          (_GEN_15),
    .dense_values_0_write_ready           (),
    .dense_values_0_write_addr            (_op0b05_l1_b0_reg21_read_data[2:0]),
    .dense_values_0_write_data
      (_op0b05_l1_b0_reg2_read_data ? _GEN_16[7:0] : 8'h0),
    .dense_values_1_read_0_enable         (_GEN_30),
    .dense_values_1_read_0_ready          (),
    .dense_values_1_read_0_addr           (_op0b05_l2_iv_read_data[2:0]),
    .dense_values_1_read_1_ready          (),
    .dense_values_1_read_1_res0           (_scratchpad_pool_dense_values_1_read_1_res0),
    .dense_values_1_write_enable          (_GEN_2),
    .dense_values_1_write_ready           (),
    .dense_values_1_write_addr            (_op0b05_l1_b0_reg21_read_data[2:0]),
    .dense_values_1_write_data
      (_op0b05_l1_b0_reg7_read_data ? _GEN_3[7:0] : 8'h0),
    .dense_values_2_read_0_enable         (_GEN_30),
    .dense_values_2_read_0_ready          (),
    .dense_values_2_read_0_addr           (_op0b05_l2_iv_read_data[2:0]),
    .dense_values_2_read_1_ready          (),
    .dense_values_2_read_1_res0           (_scratchpad_pool_dense_values_2_read_1_res0),
    .dense_values_2_write_enable          (_GEN_0),
    .dense_values_2_write_ready           (),
    .dense_values_2_write_addr            (_op0b05_l1_b0_reg21_read_data[2:0]),
    .dense_values_2_write_data
      (_op0b05_l1_b0_reg10_read_data ? _GEN_1[7:0] : 8'h0),
    .dense_values_3_read_0_enable         (_GEN_30),
    .dense_values_3_read_0_ready          (),
    .dense_values_3_read_0_addr           (_op0b05_l2_iv_read_data[2:0]),
    .dense_values_3_read_1_ready          (),
    .dense_values_3_read_1_res0           (_scratchpad_pool_dense_values_3_read_1_res0),
    .dense_values_3_write_enable          (_GEN_21),
    .dense_values_3_write_ready           (),
    .dense_values_3_write_addr            (_op0b05_l1_b0_reg21_read_data[2:0]),
    .dense_values_3_write_data
      (_op0b05_l1_b0_reg12_read_data ? _GEN_22[7:0] : 8'h0),
    .matrix_a_0_read_0_enable             (_GEN_54),
    .matrix_a_0_read_0_ready              (_scratchpad_pool_matrix_a_0_read_0_ready),
    .matrix_a_0_read_0_addr               (_GEN_56),
    .matrix_a_0_read_1_ready              (),
    .matrix_a_0_read_1_res0               (_scratchpad_pool_matrix_a_0_read_1_res0),
    .matrix_a_0_write_enable              (1'h0),
    .matrix_a_0_write_ready               (),
    .matrix_a_0_write_addr                (2'h0),
    .matrix_a_0_write_data                (16'h0),
    .matrix_a_1_read_0_enable             (_GEN_54),
    .matrix_a_1_read_0_ready              (_scratchpad_pool_matrix_a_1_read_0_ready),
    .matrix_a_1_read_0_addr               (_GEN_56),
    .matrix_a_1_read_1_ready              (),
    .matrix_a_1_read_1_res0               (_scratchpad_pool_matrix_a_1_read_1_res0),
    .matrix_a_1_write_enable              (1'h0),
    .matrix_a_1_write_ready               (),
    .matrix_a_1_write_addr                (2'h0),
    .matrix_a_1_write_data                (16'h0),
    .matrix_a_2_read_0_enable             (_GEN_54),
    .matrix_a_2_read_0_ready              (_scratchpad_pool_matrix_a_2_read_0_ready),
    .matrix_a_2_read_0_addr               (_GEN_56),
    .matrix_a_2_read_1_ready              (),
    .matrix_a_2_read_1_res0               (_scratchpad_pool_matrix_a_2_read_1_res0),
    .matrix_a_2_write_enable              (1'h0),
    .matrix_a_2_write_ready               (),
    .matrix_a_2_write_addr                (2'h0),
    .matrix_a_2_write_data                (16'h0),
    .matrix_a_3_read_0_enable             (_GEN_54),
    .matrix_a_3_read_0_ready              (_scratchpad_pool_matrix_a_3_read_0_ready),
    .matrix_a_3_read_0_addr               (_GEN_56),
    .matrix_a_3_read_1_ready              (),
    .matrix_a_3_read_1_res0               (_scratchpad_pool_matrix_a_3_read_1_res0),
    .matrix_a_3_write_enable              (1'h0),
    .matrix_a_3_write_ready               (),
    .matrix_a_3_write_addr                (2'h0),
    .matrix_a_3_write_data                (16'h0),
    .matrix_b_0_read_0_enable             (_GEN_54),
    .matrix_b_0_read_0_ready              (_scratchpad_pool_matrix_b_0_read_0_ready),
    .matrix_b_0_read_0_addr               (_GEN_55),
    .matrix_b_0_read_1_ready              (),
    .matrix_b_0_read_1_res0               (_scratchpad_pool_matrix_b_0_read_1_res0),
    .matrix_b_0_write_enable              (1'h0),
    .matrix_b_0_write_ready               (),
    .matrix_b_0_write_addr                (2'h0),
    .matrix_b_0_write_data                (16'h0),
    .matrix_b_1_read_0_enable             (_GEN_54),
    .matrix_b_1_read_0_ready              (_scratchpad_pool_matrix_b_1_read_0_ready),
    .matrix_b_1_read_0_addr               (_GEN_55),
    .matrix_b_1_read_1_ready              (),
    .matrix_b_1_read_1_res0               (_scratchpad_pool_matrix_b_1_read_1_res0),
    .matrix_b_1_write_enable              (1'h0),
    .matrix_b_1_write_ready               (),
    .matrix_b_1_write_addr                (2'h0),
    .matrix_b_1_write_data                (16'h0),
    .matrix_b_2_read_0_enable             (_GEN_54),
    .matrix_b_2_read_0_ready              (_scratchpad_pool_matrix_b_2_read_0_ready),
    .matrix_b_2_read_0_addr               (_GEN_55),
    .matrix_b_2_read_1_ready              (),
    .matrix_b_2_read_1_res0               (_scratchpad_pool_matrix_b_2_read_1_res0),
    .matrix_b_2_write_enable              (1'h0),
    .matrix_b_2_write_ready               (),
    .matrix_b_2_write_addr                (2'h0),
    .matrix_b_2_write_data                (16'h0),
    .matrix_b_3_read_0_enable             (_GEN_54),
    .matrix_b_3_read_0_ready              (_scratchpad_pool_matrix_b_3_read_0_ready),
    .matrix_b_3_read_0_addr               (_GEN_55),
    .matrix_b_3_read_1_ready              (),
    .matrix_b_3_read_1_res0               (_scratchpad_pool_matrix_b_3_read_1_res0),
    .matrix_b_3_write_enable              (1'h0),
    .matrix_b_3_write_ready               (),
    .matrix_b_3_write_addr                (2'h0),
    .matrix_b_3_write_data                (16'h0),
    .matrix_c_0_read_0_enable             (1'h0),
    .matrix_c_0_read_0_ready              (),
    .matrix_c_0_read_0_addr               (2'h0),
    .matrix_c_0_read_1_ready              (),
    .matrix_c_0_read_1_res0               (),
    .matrix_c_0_write_enable              (_GEN_57),
    .matrix_c_0_write_ready               (_scratchpad_pool_matrix_c_0_write_ready),
    .matrix_c_0_write_addr                (_op2b38_l1_b0_reg21_read_data[1:0]),
    .matrix_c_0_write_data
      (_scratchpad_pool_matrix_a_0_read_1_res0 & _scratchpad_pool_matrix_b_0_read_1_res0
       ^ _scratchpad_pool_matrix_a_1_read_1_res0 & _op2b38_l1_b0_reg14_read_data
       ^ _scratchpad_pool_matrix_a_2_read_1_res0 & _op2b38_l1_b0_reg7_read_data
       ^ _scratchpad_pool_matrix_a_3_read_1_res0 & _op2b38_l1_b0_reg0_read_data),
    .matrix_c_1_read_0_enable             (1'h0),
    .matrix_c_1_read_0_ready              (),
    .matrix_c_1_read_0_addr               (2'h0),
    .matrix_c_1_read_1_ready              (),
    .matrix_c_1_read_1_res0               (),
    .matrix_c_1_write_enable              (_GEN_57),
    .matrix_c_1_write_ready               (_scratchpad_pool_matrix_c_1_write_ready),
    .matrix_c_1_write_addr                (_op2b38_l1_b0_reg21_read_data[1:0]),
    .matrix_c_1_write_data
      (_op2b38_l1_b0_reg15_read_data & _scratchpad_pool_matrix_b_1_read_1_res0
       ^ _op2b38_l1_b0_reg16_read_data ^ _op2b38_l1_b0_reg17_read_data
       ^ _op2b38_l1_b0_reg18_read_data),
    .matrix_c_2_read_0_enable             (1'h0),
    .matrix_c_2_read_0_ready              (),
    .matrix_c_2_read_0_addr               (2'h0),
    .matrix_c_2_read_1_ready              (),
    .matrix_c_2_read_1_res0               (),
    .matrix_c_2_write_enable              (_GEN_57),
    .matrix_c_2_write_ready               (_scratchpad_pool_matrix_c_2_write_ready),
    .matrix_c_2_write_addr                (_op2b38_l1_b0_reg21_read_data[1:0]),
    .matrix_c_2_write_data
      (_op2b38_l1_b0_reg9_read_data & _scratchpad_pool_matrix_b_2_read_1_res0
       ^ _op2b38_l1_b0_reg19_read_data ^ _op2b38_l1_b0_reg11_read_data
       ^ _op2b38_l1_b0_reg12_read_data),
    .matrix_c_3_read_0_enable             (1'h0),
    .matrix_c_3_read_0_ready              (),
    .matrix_c_3_read_0_addr               (2'h0),
    .matrix_c_3_read_1_ready              (),
    .matrix_c_3_read_1_res0               (),
    .matrix_c_3_write_enable              (_GEN_57),
    .matrix_c_3_write_ready               (_scratchpad_pool_matrix_c_3_write_ready),
    .matrix_c_3_write_addr                (_op2b38_l1_b0_reg21_read_data[1:0]),
    .matrix_c_3_write_data
      (_op2b38_l1_b0_reg3_read_data & _scratchpad_pool_matrix_b_3_read_1_res0
       ^ _op2b38_l1_b0_reg20_read_data ^ _op2b38_l1_b0_reg13_read_data
       ^ _op2b38_l1_b0_reg6_read_data),
    .values_read_0_enable                 (_GEN_7 | _GEN_5 | _GEN_2 | _GEN_0),
    .values_read_0_ready                  (_scratchpad_pool_values_read_0_ready),
    .values_read_0_addr
      (_GEN_7
         ? _glbl_reg_vidx_read_data[5:3]
         : _GEN_5
             ? _op0b05_l1_b0_reg6_read_data[5:3]
             : _GEN_2
                 ? _op0b05_l1_b0_reg15_read_data[5:3]
                 : _op0b05_l1_b0_reg18_read_data[5:3]),
    .values_read_1_ready                  (),
    .values_read_1_res0                   (_scratchpad_pool_values_read_1_res0),
    .values_write_enable                  (1'h0),
    .values_write_ready                   (),
    .values_write_addr                    (3'h0),
    .values_write_data                    (64'h0),
    .vidx_read_0_enable                   (1'h0),
    .vidx_read_0_ready                    (),
    .vidx_read_0_addr                     (1'h0),
    .vidx_read_1_ready                    (),
    .vidx_read_1_res0                     (),
    .vidx_write_enable                    (1'h0),
    .vidx_write_ready                     (),
    .vidx_write_addr                      (1'h0),
    .vidx_write_data                      (32'h0)
  );
  RoCCAdapter rocc_adapter (
    .clk                                      (clk),
    .rst                                      (rst),
    .cmd_from_bus_enable
      (_rocc_adapter_cmd_from_bus_ready & rocc_cmd_enable),
    .cmd_from_bus_ready                       (_rocc_adapter_cmd_from_bus_ready),
    .cmd_from_bus_rocc_cmd_bus_funct          (rocc_cmd_rocc_cmd_funct),
    .cmd_from_bus_rocc_cmd_bus_rs1            (rocc_cmd_rocc_cmd_rs1),
    .cmd_from_bus_rocc_cmd_bus_rs2            (rocc_cmd_rocc_cmd_rs2),
    .cmd_from_bus_rocc_cmd_bus_rd             (rocc_cmd_rocc_cmd_rd),
    .cmd_from_bus_rocc_cmd_bus_xs1            (rocc_cmd_rocc_cmd_xs1),
    .cmd_from_bus_rocc_cmd_bus_xs2            (rocc_cmd_rocc_cmd_xs2),
    .cmd_from_bus_rocc_cmd_bus_xd             (rocc_cmd_rocc_cmd_xd),
    .cmd_from_bus_rocc_cmd_bus_opcode         (rocc_cmd_rocc_cmd_opcode),
    .cmd_from_bus_rocc_cmd_bus_rs1data        (rocc_cmd_rocc_cmd_rs1data),
    .cmd_from_bus_rocc_cmd_bus_rs2data        (rocc_cmd_rocc_cmd_rs2data),
    .resp_from_user_enable                    (_GEN_44 | _GEN_43),
    .resp_from_user_ready                     (_rocc_adapter_resp_from_user_ready),
    .resp_from_user_rocc_resp_user_rd
      (_GEN_44 ? _reg_rd_2b38_read_data : _reg_rd_0b05_read_data),
    .resp_from_user_rocc_resp_user_rddata     (32'h0),
    .cmd_to_user_0b05_enable                  (_GEN),
    .cmd_to_user_0b05_ready                   (_rocc_adapter_cmd_to_user_0b05_ready),
    .cmd_to_user_0b05_res0_funct              (),
    .cmd_to_user_0b05_res0_rs1                (),
    .cmd_to_user_0b05_res0_rs2                (),
    .cmd_to_user_0b05_res0_rd                 (_rocc_adapter_cmd_to_user_0b05_res0_rd),
    .cmd_to_user_0b05_res0_xs1                (),
    .cmd_to_user_0b05_res0_xs2                (),
    .cmd_to_user_0b05_res0_xd                 (),
    .cmd_to_user_0b05_res0_opcode             (),
    .cmd_to_user_0b05_res0_rs1data
      (_rocc_adapter_cmd_to_user_0b05_res0_rs1data),
    .cmd_to_user_0b05_res0_rs2data
      (_rocc_adapter_cmd_to_user_0b05_res0_rs2data),
    .cmd_to_user_2b38_enable                  (_GEN_45),
    .cmd_to_user_2b38_ready                   (_rocc_adapter_cmd_to_user_2b38_ready),
    .cmd_to_user_2b38_res0_funct              (),
    .cmd_to_user_2b38_res0_rs1                (),
    .cmd_to_user_2b38_res0_rs2                (),
    .cmd_to_user_2b38_res0_rd                 (_rocc_adapter_cmd_to_user_2b38_res0_rd),
    .cmd_to_user_2b38_res0_xs1                (),
    .cmd_to_user_2b38_res0_xs2                (),
    .cmd_to_user_2b38_res0_xd                 (),
    .cmd_to_user_2b38_res0_opcode             (),
    .cmd_to_user_2b38_res0_rs1data
      (_rocc_adapter_cmd_to_user_2b38_res0_rs1data),
    .cmd_to_user_2b38_res0_rs2data
      (_rocc_adapter_cmd_to_user_2b38_res0_rs2data),
    .rocc_resp_rocc_resp_to_bus_result_rd     (rocc_resp_rocc_resp_to_bus_result_rd),
    .rocc_resp_rocc_resp_to_bus_result_rddata (rocc_resp_rocc_resp_to_bus_result_rddata),
    .rocc_resp_rocc_resp_to_bus_ready         (rocc_resp_rocc_resp_to_bus_ready),
    .rocc_resp_rocc_resp_to_bus_enable        (rocc_resp_rocc_resp_to_bus_enable)
  );
  MemoryTranslator hellacache_adapter (
    .clk                                   (clk),
    .rst                                   (rst),
    .resp_from_bus_enable
      (_hellacache_adapter_resp_from_bus_ready & hella_resp_enable),
    .resp_from_bus_ready                   (_hellacache_adapter_resp_from_bus_ready),
    .resp_from_bus_hella_resp_data         (hella_resp_hella_resp_data),
    .resp_from_bus_hella_resp_tag          (hella_resp_hella_resp_tag),
    .resp_from_bus_hella_resp_cmd          (hella_resp_hella_resp_cmd),
    .resp_from_bus_hella_resp_size         (hella_resp_hella_resp_size),
    .resp_from_bus_hella_resp_signed       (hella_resp_hella_resp_signed),
    .cmd_from_user_enable                  (_GEN_12 | _GEN_11),
    .cmd_from_user_ready                   (_hellacache_adapter_cmd_from_user_ready),
    .cmd_from_user_user_cmd_addr
      (_GEN_12 ? _GEN_13 : _op0b05_b0_reg1_read_data),
    .cmd_from_user_user_cmd_cmd            (1'h0),
    .cmd_from_user_user_cmd_size           (2'h2),
    .cmd_from_user_user_cmd_data           (32'h0),
    .cmd_from_user_user_cmd_mask           (4'h0),
    .cmd_from_user_user_cmd_tag
      ({2'h0, _GEN_12 ? _GEN_13[5:0] : _op0b05_b0_reg1_read_data[5:0]}),
    .resp_to_user_enable                   (_GEN_11 | _GEN_10),
    .resp_to_user_ready                    (_hellacache_adapter_resp_to_user_ready),
    .resp_to_user_res0_data                (_hellacache_adapter_resp_to_user_res0_data),
    .resp_to_user_res0_tag                 (),
    .hella_cmd_hella_cmd_to_bus_cmd_addr   (hella_cmd_hella_cmd_to_bus_cmd_addr),
    .hella_cmd_hella_cmd_to_bus_cmd_tag    (hella_cmd_hella_cmd_to_bus_cmd_tag),
    .hella_cmd_hella_cmd_to_bus_cmd_cmd    (hella_cmd_hella_cmd_to_bus_cmd_cmd),
    .hella_cmd_hella_cmd_to_bus_cmd_size   (hella_cmd_hella_cmd_to_bus_cmd_size),
    .hella_cmd_hella_cmd_to_bus_cmd_signed (hella_cmd_hella_cmd_to_bus_cmd_signed),
    .hella_cmd_hella_cmd_to_bus_cmd_phys   (hella_cmd_hella_cmd_to_bus_cmd_phys),
    .hella_cmd_hella_cmd_to_bus_cmd_data   (hella_cmd_hella_cmd_to_bus_cmd_data),
    .hella_cmd_hella_cmd_to_bus_cmd_mask   (hella_cmd_hella_cmd_to_bus_cmd_mask),
    .hella_cmd_hella_cmd_to_bus_ready      (hella_cmd_hella_cmd_to_bus_ready),
    .hella_cmd_hella_cmd_to_bus_enable     (hella_cmd_hella_cmd_to_bus_enable)
  );
  Reg_width5_init0 reg_rd_0b05 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN),
    .write_data   (_rocc_adapter_cmd_to_user_0b05_res0_rd),
    .read_ready   (),
    .read_data    (_reg_rd_0b05_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_b0v0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN),
    .write_data   (_rocc_adapter_cmd_to_user_0b05_res0_rs2data),
    .read_ready   (),
    .read_data    (_op0b05_b0v0_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_b0v1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_10),
    .write_data   (_hellacache_adapter_resp_to_user_res0_data),
    .read_ready   (),
    .read_data    (_op0b05_b0v1_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op0b05_b0v2 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_11),
    .write_data   (_hellacache_adapter_resp_to_user_res0_data[15:0]),
    .read_ready   (),
    .read_data    (_op0b05_b0v2_read_data),
    .write_ready  ()
  );
  FIFO2_I_w1 op0b05_b0b1tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_23),
    .deq_ready  (_op0b05_b0b1tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_14),
    .enq_ready  (_op0b05_b0b1tok_enq_ready),
    .enq_data   (_GEN_14)
  );
  FIFO2_I_w10 op0b05_b1b2tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_34),
    .deq_ready  (_op0b05_b1b2tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_29),
    .enq_ready  (_op0b05_b1b2tok_enq_ready),
    .enq_data   (_GEN_29)
  );
  FIFO2_I_w11 op0b05_b2b3tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_41),
    .deq_ready  (_op0b05_b2b3tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_40),
    .enq_ready  (_op0b05_b2b3tok_enq_ready),
    .enq_data   (_GEN_40)
  );
  FIFO1_PUSH_w1 op0b05_b0_s0tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_b0_s0tok_full_res0),
    .deq_enable (_GEN_12),
    .deq_ready  (_op0b05_b0_s0tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN),
    .enq_ready  (_op0b05_b0_s0tok_enq_ready),
    .enq_data   (_GEN)
  );
  FIFO1_PUSH_w10 op0b05_b0_s1tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_b0_s1tok_full_res0),
    .deq_enable (_GEN_11),
    .deq_ready  (_op0b05_b0_s1tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_12),
    .enq_ready  (_op0b05_b0_s1tok_enq_ready),
    .enq_data   (_GEN_12)
  );
  FIFO1_PUSH_w11 op0b05_b0_s2tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_b0_s2tok_full_res0),
    .deq_enable (_GEN_10),
    .deq_ready  (_op0b05_b0_s2tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_11),
    .enq_ready  (_op0b05_b0_s2tok_enq_ready),
    .enq_data   (_GEN_11)
  );
  FIFO1_PUSH_w12 op0b05_b0_s3tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_b0_s3tok_full_res0),
    .deq_enable (_GEN_14),
    .deq_ready  (_op0b05_b0_s3tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_10),
    .enq_ready  (_op0b05_b0_s3tok_enq_ready),
    .enq_data   (_GEN_10)
  );
  Reg_width32_init0 op0b05_b0_reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN),
    .write_data   (_rocc_adapter_cmd_to_user_0b05_res0_rs1data),
    .read_ready   (),
    .read_data    (_op0b05_b0_reg0_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_b0_reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_12),
    .write_data   (_op0b05_b0_reg0_read_data + 32'h40),
    .read_ready   (),
    .read_data    (_op0b05_b0_reg1_read_data),
    .write_ready  ()
  );
  FIFO2_I_w12 op0b05_l1_entok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_7),
    .deq_ready  (_op0b05_l1_entok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_28),
    .enq_ready  (_op0b05_l1_entok_enq_ready),
    .enq_data   (_GEN_28)
  );
  FIFO2_I_w13 op0b05_l1_dntok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_24),
    .deq_ready  (_op0b05_l1_dntok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_21),
    .enq_ready  (_op0b05_l1_dntok_enq_ready),
    .enq_data   (_GEN_21)
  );
  Reg_width96_init0 op0b05_l1_st (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_28),
    .write_data
      (_GEN_27 ? {_GEN_25[31:0], _op0b05_l1_st_read_data[63:0]} : 96'h700000001),
    .read_ready   (),
    .read_data    (_op0b05_l1_st_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_iv (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_28),
    .write_data   (_GEN_27 ? _GEN_25[31:0] : 32'h0),
    .read_ready   (),
    .read_data    (_op0b05_l1_iv_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_isr0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_23),
    .write_data   (_op0b05_b0v1_read_data),
    .read_ready   (),
    .read_data    (_op0b05_l1_isr0_read_data),
    .write_ready  ()
  );
  FIFO2_I_w32 op0b05_l1_in0 (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_7),
    .deq_ready  (_op0b05_l1_in0_deq_ready),
    .deq_res0   (_op0b05_l1_in0_deq_res0),
    .enq_enable (_GEN_28),
    .enq_ready  (_op0b05_l1_in0_enq_ready),
    .enq_data   (_GEN_27 ? _op0b05_l1_isr0_read_data : _op0b05_b0v1_read_data)
  );
  FIFO1_PUSH_w13 op0b05_l1_b0_s7tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_l1_b0_s7tok_full_res0),
    .deq_enable (_GEN_15),
    .deq_ready  (_op0b05_l1_b0_s7tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_7),
    .enq_ready  (_op0b05_l1_b0_s7tok_enq_ready),
    .enq_data   (_GEN_7)
  );
  FIFO1_PUSH_w14 op0b05_l1_b0_s8tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_l1_b0_s8tok_full_res0),
    .deq_enable (_GEN_5),
    .deq_ready  (_op0b05_l1_b0_s8tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_15),
    .enq_ready  (_op0b05_l1_b0_s8tok_enq_ready),
    .enq_data   (_GEN_15)
  );
  FIFO1_PUSH_w15 op0b05_l1_b0_s9tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_l1_b0_s9tok_full_res0),
    .deq_enable (_GEN_2),
    .deq_ready  (_op0b05_l1_b0_s9tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_5),
    .enq_ready  (_op0b05_l1_b0_s9tok_enq_ready),
    .enq_data   (_GEN_5)
  );
  FIFO1_PUSH_w16 op0b05_l1_b0_s10tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_l1_b0_s10tok_full_res0),
    .deq_enable (_GEN_0),
    .deq_ready  (_op0b05_l1_b0_s10tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_2),
    .enq_ready  (_op0b05_l1_b0_s10tok_enq_ready),
    .enq_data   (_GEN_2)
  );
  FIFO1_PUSH_w17 op0b05_l1_b0_s11tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_l1_b0_s11tok_full_res0),
    .deq_enable (_GEN_21),
    .deq_ready  (_op0b05_l1_b0_s11tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_0),
    .enq_ready  (_op0b05_l1_b0_s11tok_enq_ready),
    .enq_data   (_GEN_0)
  );
  Reg_width8_init0 op0b05_l1_b0_reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_7),
    .write_data   (_GEN_8),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg0_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_7),
    .write_data   ({29'h0, _op0b05_l1_iv_read_data[0], 2'h0}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg1_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 op0b05_l1_b0_reg2 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_7),
    .write_data   (_GEN_7 & _GEN_9[0]),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg2_read_data),
    .write_ready  ()
  );
  Reg_width64_init0 op0b05_l1_b0_reg3 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_7),
    .write_data   ({58'h0, _glbl_reg_vidx_read_data[2:0], 3'h0}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg3_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg4 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_7),
    .write_data   (_glbl_reg_vidx_read_data),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg4_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg5 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_7),
    .write_data   ({31'h0, _GEN_9[0]}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg5_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg6 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_15),
    .write_data   (_GEN_17),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg6_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 op0b05_l1_b0_reg7 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_15),
    .write_data   (_GEN_15 & _GEN_18[0]),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg7_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg8 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_15),
    .write_data   ({29'h0, _GEN_17[2:0]}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg8_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg9 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_15),
    .write_data   ({31'h0, _GEN_18[0]}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg9_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 op0b05_l1_b0_reg10 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_15),
    .write_data   (_GEN_15 & _GEN_19[0]),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg10_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg11 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_15),
    .write_data   ({31'h0, _GEN_19[0]}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg11_read_data),
    .write_ready  ()
  );
  Reg_width1_init0 op0b05_l1_b0_reg12 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_15),
    .write_data   (_GEN_15 & _GEN_20[0]),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg12_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg13 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_15),
    .write_data   ({31'h0, _GEN_20[0]}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg13_read_data),
    .write_ready  ()
  );
  Reg_width64_init0 op0b05_l1_b0_reg14 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5),
    .write_data   ({32'h0, _op0b05_l1_b0_reg8_read_data[28:0], 3'h0}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg14_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg15 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5),
    .write_data   (_GEN_6),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg15_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg16 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_5),
    .write_data   ({29'h0, _GEN_6[2:0]}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg16_read_data),
    .write_ready  ()
  );
  Reg_width64_init0 op0b05_l1_b0_reg17 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_2),
    .write_data   ({32'h0, _op0b05_l1_b0_reg16_read_data[28:0], 3'h0}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg17_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg18 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_2),
    .write_data   (_GEN_4),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg18_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg19 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_2),
    .write_data   ({29'h0, _GEN_4[2:0]}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg19_read_data),
    .write_ready  ()
  );
  Reg_width64_init0 op0b05_l1_b0_reg20 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_0),
    .write_data   ({32'h0, _op0b05_l1_b0_reg19_read_data[28:0], 3'h0}),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg20_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l1_b0_reg21 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_7),
    .write_data   (_op0b05_l1_iv_read_data),
    .read_ready   (),
    .read_data    (_op0b05_l1_b0_reg21_read_data),
    .write_ready  ()
  );
  FIFO2_I_w14 op0b05_l2_entok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_30),
    .deq_ready  (_op0b05_l2_entok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_39),
    .enq_ready  (_op0b05_l2_entok_enq_ready),
    .enq_data   (_GEN_39)
  );
  FIFO2_I_w15 op0b05_l2_dntok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_35),
    .deq_ready  (_op0b05_l2_dntok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_33),
    .enq_ready  (_op0b05_l2_dntok_enq_ready),
    .enq_data   (_GEN_33)
  );
  Reg_width96_init0 op0b05_l2_st (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_39),
    .write_data
      (_GEN_38 ? {_GEN_36[31:0], _op0b05_l2_st_read_data[63:0]} : 96'h700000001),
    .read_ready   (),
    .read_data    (_op0b05_l2_st_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_iv (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_39),
    .write_data   (_GEN_38 ? _GEN_36[31:0] : 32'h0),
    .read_ready   (),
    .read_data    (_op0b05_l2_iv_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op0b05_l2_isr0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_34),
    .write_data   (_op0b05_b0v2_read_data),
    .read_ready   (),
    .read_data    (_op0b05_l2_isr0_read_data),
    .write_ready  ()
  );
  FIFO2_I_w16 op0b05_l2_in0 (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_30),
    .deq_ready  (_op0b05_l2_in0_deq_ready),
    .deq_res0   (_op0b05_l2_in0_deq_res0),
    .enq_enable (_GEN_39),
    .enq_ready  (_op0b05_l2_in0_enq_ready),
    .enq_data   (_GEN_38 ? _op0b05_l2_isr0_read_data : _op0b05_b0v2_read_data)
  );
  FIFO1_PUSH_w18 op0b05_l2_b0_s15tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_l2_b0_s15tok_full_res0),
    .deq_enable (_GEN_31),
    .deq_ready  (_op0b05_l2_b0_s15tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_30),
    .enq_ready  (_op0b05_l2_b0_s15tok_enq_ready),
    .enq_data   (_GEN_30)
  );
  FIFO1_PUSH_w19 op0b05_l2_b0_s16tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_l2_b0_s16tok_full_res0),
    .deq_enable (_GEN_32),
    .deq_ready  (_op0b05_l2_b0_s16tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_31),
    .enq_ready  (_op0b05_l2_b0_s16tok_enq_ready),
    .enq_data   (_GEN_31)
  );
  FIFO1_PUSH_w110 op0b05_l2_b0_s17tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_l2_b0_s17tok_full_res0),
    .deq_enable (_GEN_33),
    .deq_ready  (_op0b05_l2_b0_s17tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_32),
    .enq_ready  (_op0b05_l2_b0_s17tok_enq_ready),
    .enq_data   (_GEN_32)
  );
  Reg_width32_init0 op0b05_l2_b0_reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_30),
    .write_data   ({{16{_op0b05_l2_in0_deq_res0[15]}}, _op0b05_l2_in0_deq_res0}),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg0_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_b0_reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_31),
    .write_data
      ({{24{_scratchpad_pool_dense_values_0_read_1_res0[7]}},
        _scratchpad_pool_dense_values_0_read_1_res0}),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg1_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_b0_reg2 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_31),
    .write_data
      ({{24{_scratchpad_pool_dense_values_1_read_1_res0[7]}},
        _scratchpad_pool_dense_values_1_read_1_res0}),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg2_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_b0_reg3 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_31),
    .write_data
      ({{24{_scratchpad_pool_dense_values_2_read_1_res0[7]}},
        _scratchpad_pool_dense_values_2_read_1_res0}),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg3_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_b0_reg4 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_31),
    .write_data
      ({{24{_scratchpad_pool_dense_values_3_read_1_res0[7]}},
        _scratchpad_pool_dense_values_3_read_1_res0}),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg4_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_b0_reg5 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_32),
    .write_data   (_op0b05_l2_b0_reg1_read_data * _op0b05_l2_b0_reg0_read_data),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg5_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_b0_reg6 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_32),
    .write_data   (_op0b05_l2_b0_reg2_read_data * _op0b05_l2_b0_reg0_read_data),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg6_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_b0_reg7 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_32),
    .write_data   (_op0b05_l2_b0_reg3_read_data * _op0b05_l2_b0_reg0_read_data),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg7_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_b0_reg8 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_32),
    .write_data   (_op0b05_l2_b0_reg4_read_data * _op0b05_l2_b0_reg0_read_data),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg8_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op0b05_l2_b0_reg9 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_30),
    .write_data   (_op0b05_l2_iv_read_data),
    .read_ready   (),
    .read_data    (_op0b05_l2_b0_reg9_read_data),
    .write_ready  ()
  );
  FIFO1_PUSH_w111 op0b05_b3_s20tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_b3_s20tok_full_res0),
    .deq_enable (_GEN_42),
    .deq_ready  (_op0b05_b3_s20tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_41),
    .enq_ready  (_op0b05_b3_s20tok_enq_ready),
    .enq_data   (_GEN_41)
  );
  FIFO1_PUSH_w112 op0b05_b3_s22tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op0b05_b3_s22tok_full_res0),
    .deq_enable (_GEN_43),
    .deq_ready  (_op0b05_b3_s22tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_42),
    .enq_ready  (_op0b05_b3_s22tok_enq_ready),
    .enq_data   (_GEN_42)
  );
  Reg_width32_init0 op0b05_b3_reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_41),
    .write_data   (_op0b05_b0v0_read_data),
    .read_ready   (),
    .read_data    (_op0b05_b3_reg0_read_data),
    .write_ready  ()
  );
  Reg_width5_init0 reg_rd_2b38 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_45),
    .write_data   (_rocc_adapter_cmd_to_user_2b38_res0_rd),
    .read_ready   (),
    .read_data    (_reg_rd_2b38_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op2b38_b0v0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_45),
    .write_data   (_rocc_adapter_cmd_to_user_2b38_res0_rs2data),
    .read_ready   (),
    .read_data    (_op2b38_b0v0_read_data),
    .write_ready  ()
  );
  FIFO2_I_w17 op2b38_b0b1tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_58),
    .deq_ready  (_op2b38_b0b1tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_49),
    .enq_ready  (_op2b38_b0b1tok_enq_ready),
    .enq_data   (_GEN_49)
  );
  FIFO2_I_w18 op2b38_b1b2tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_65),
    .deq_ready  (_op2b38_b1b2tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_64),
    .enq_ready  (_op2b38_b1b2tok_enq_ready),
    .enq_data   (_GEN_64)
  );
  FIFO1_PUSH_w113 op2b38_b0_s0tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_b0_s0tok_full_res0),
    .deq_enable (_GEN_46),
    .deq_ready  (_op2b38_b0_s0tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_45),
    .enq_ready  (_op2b38_b0_s0tok_enq_ready),
    .enq_data   (_GEN_45)
  );
  FIFO1_PUSH_w114 op2b38_b0_s1tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_b0_s1tok_full_res0),
    .deq_enable (_GEN_47),
    .deq_ready  (_op2b38_b0_s1tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_46),
    .enq_ready  (_op2b38_b0_s1tok_enq_ready),
    .enq_data   (_GEN_46)
  );
  FIFO1_PUSH_w115 op2b38_b0_s2tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_b0_s2tok_full_res0),
    .deq_enable (_GEN_48),
    .deq_ready  (_op2b38_b0_s2tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_47),
    .enq_ready  (_op2b38_b0_s2tok_enq_ready),
    .enq_data   (_GEN_47)
  );
  FIFO1_PUSH_w116 op2b38_b0_s3tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_b0_s3tok_full_res0),
    .deq_enable (_GEN_49),
    .deq_ready  (_op2b38_b0_s3tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_48),
    .enq_ready  (_op2b38_b0_s3tok_enq_ready),
    .enq_data   (_GEN_48)
  );
  Reg_width32_init0 op2b38_b0_reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_45),
    .write_data   (_rocc_adapter_cmd_to_user_2b38_res0_rs2data),
    .read_ready   (),
    .read_data    (_op2b38_b0_reg0_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op2b38_b0_reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_45),
    .write_data   (_rocc_adapter_cmd_to_user_2b38_res0_rs1data),
    .read_ready   (),
    .read_data    (dma_cpu_to_isax_ch1_cpu_addr),
    .write_ready  ()
  );
  FIFO2_I_w19 op2b38_l1_entok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_53),
    .deq_ready  (_op2b38_l1_entok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_63),
    .enq_ready  (_op2b38_l1_entok_enq_ready),
    .enq_data   (_GEN_63)
  );
  FIFO2_I_w110 op2b38_l1_dntok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (),
    .deq_enable (_GEN_59),
    .deq_ready  (_op2b38_l1_dntok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_57),
    .enq_ready  (_op2b38_l1_dntok_enq_ready),
    .enq_data   (_GEN_57)
  );
  Reg_width96_init0 op2b38_l1_st (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_63),
    .write_data
      (_GEN_62 ? {_GEN_60[31:0], _op2b38_l1_st_read_data[63:0]} : 96'h300000001),
    .read_ready   (),
    .read_data    (_op2b38_l1_st_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op2b38_l1_iv (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_63),
    .write_data   (_GEN_62 ? _GEN_60[31:0] : 32'h0),
    .read_ready   (),
    .read_data    (_op2b38_l1_iv_read_data),
    .write_ready  ()
  );
  FIFO1_PUSH_w117 op2b38_l1_b0_s6tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_l1_b0_s6tok_full_res0),
    .deq_enable (_GEN_52),
    .deq_ready  (_op2b38_l1_b0_s6tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_53),
    .enq_ready  (_op2b38_l1_b0_s6tok_enq_ready),
    .enq_data   (_GEN_53)
  );
  FIFO1_PUSH_w118 op2b38_l1_b0_s7tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_l1_b0_s7tok_full_res0),
    .deq_enable (_GEN_51),
    .deq_ready  (_op2b38_l1_b0_s7tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_52),
    .enq_ready  (_op2b38_l1_b0_s7tok_enq_ready),
    .enq_data   (_GEN_52)
  );
  FIFO1_PUSH_w119 op2b38_l1_b0_s8tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_l1_b0_s8tok_full_res0),
    .deq_enable (_GEN_50),
    .deq_ready  (_op2b38_l1_b0_s8tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_51),
    .enq_ready  (_op2b38_l1_b0_s8tok_enq_ready),
    .enq_data   (_GEN_51)
  );
  FIFO1_PUSH_w120 op2b38_l1_b0_s9tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_l1_b0_s9tok_full_res0),
    .deq_enable (_GEN_57),
    .deq_ready  (_op2b38_l1_b0_s9tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_50),
    .enq_ready  (_op2b38_l1_b0_s9tok_enq_ready),
    .enq_data   (_GEN_50)
  );
  Reg_width16_init0 op2b38_l1_b0_reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_52),
    .write_data   (_scratchpad_pool_matrix_b_0_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg0_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg1 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_52),
    .write_data   (_scratchpad_pool_matrix_b_1_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg1_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg2 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_52),
    .write_data   (_scratchpad_pool_matrix_b_2_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg2_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg3 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_52),
    .write_data   (_scratchpad_pool_matrix_a_0_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg3_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg4 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_52),
    .write_data   (_scratchpad_pool_matrix_a_1_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg4_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg5 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_52),
    .write_data   (_scratchpad_pool_matrix_a_2_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg5_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg6 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_52),
    .write_data
      (_scratchpad_pool_matrix_a_3_read_1_res0 & _scratchpad_pool_matrix_b_3_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg6_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg7 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_51),
    .write_data   (_scratchpad_pool_matrix_b_0_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg7_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg8 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_51),
    .write_data   (_scratchpad_pool_matrix_b_1_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg8_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg9 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_51),
    .write_data   (_scratchpad_pool_matrix_a_0_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg9_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg10 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_51),
    .write_data   (_scratchpad_pool_matrix_a_1_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg10_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg11 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_51),
    .write_data
      (_scratchpad_pool_matrix_a_2_read_1_res0 & _scratchpad_pool_matrix_b_2_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg11_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg12 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_51),
    .write_data
      (_scratchpad_pool_matrix_a_3_read_1_res0 & _op2b38_l1_b0_reg2_read_data),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg12_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg13 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_51),
    .write_data
      (_op2b38_l1_b0_reg5_read_data & _scratchpad_pool_matrix_b_3_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg13_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg14 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_50),
    .write_data   (_scratchpad_pool_matrix_b_0_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg14_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg15 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_50),
    .write_data   (_scratchpad_pool_matrix_a_0_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg15_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg16 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_50),
    .write_data
      (_scratchpad_pool_matrix_a_1_read_1_res0 & _scratchpad_pool_matrix_b_1_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg16_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg17 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_50),
    .write_data
      (_scratchpad_pool_matrix_a_2_read_1_res0 & _op2b38_l1_b0_reg8_read_data),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg17_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg18 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_50),
    .write_data
      (_scratchpad_pool_matrix_a_3_read_1_res0 & _op2b38_l1_b0_reg1_read_data),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg18_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg19 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_50),
    .write_data
      (_op2b38_l1_b0_reg10_read_data & _scratchpad_pool_matrix_b_2_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg19_read_data),
    .write_ready  ()
  );
  Reg_width16_init0 op2b38_l1_b0_reg20 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_50),
    .write_data
      (_op2b38_l1_b0_reg4_read_data & _scratchpad_pool_matrix_b_3_read_1_res0),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg20_read_data),
    .write_ready  ()
  );
  Reg_width32_init0 op2b38_l1_b0_reg21 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_53),
    .write_data   (_op2b38_l1_iv_read_data),
    .read_ready   (),
    .read_data    (_op2b38_l1_b0_reg21_read_data),
    .write_ready  ()
  );
  FIFO1_PUSH_w121 op2b38_b2_s12tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_b2_s12tok_full_res0),
    .deq_enable (_GEN_66),
    .deq_ready  (_op2b38_b2_s12tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_65),
    .enq_ready  (_op2b38_b2_s12tok_enq_ready),
    .enq_data   (_GEN_65)
  );
  FIFO1_PUSH_w122 op2b38_b2_s14tok (
    .clk        (clk),
    .rst        (rst),
    .full_ready (),
    .full_res0  (_op2b38_b2_s14tok_full_res0),
    .deq_enable (_GEN_44),
    .deq_ready  (_op2b38_b2_s14tok_deq_ready),
    .deq_res0   (),
    .enq_enable (_GEN_66),
    .enq_ready  (_op2b38_b2_s14tok_enq_ready),
    .enq_data   (_GEN_66)
  );
  Reg_width32_init0 op2b38_b2_reg0 (
    .clock        (clk),
    .reset        (rst),
    .write_enable (_GEN_65),
    .write_data   (_op2b38_b0v0_read_data + 32'h20),
    .read_ready   (),
    .read_data    (_op2b38_b2_reg0_read_data),
    .write_ready  ()
  );
  assign burst_read_0_ready = _scratchpad_pool_burst_read_0_ready;
  assign burst_read_1_ready = _scratchpad_pool_burst_read_1_ready;
  assign burst_write_ready = _scratchpad_pool_burst_write_ready;
  assign rocc_cmd_ready = _rocc_adapter_cmd_from_bus_ready;
  assign hella_resp_ready = _hellacache_adapter_resp_from_bus_ready;
  assign dma_cpu_to_isax_ch0_cpu_addr =
    _GEN_46 ? _op2b38_b0_reg0_read_data : _op0b05_b0_reg0_read_data;
  assign dma_cpu_to_isax_ch0_isax_addr = _GEN_46 ? 32'hC0 : 32'h100;
  assign dma_cpu_to_isax_ch0_length = _GEN_46 ? 4'h5 : 4'h6;
  assign dma_cpu_to_isax_ch0_stride_x = 8'h0;
  assign dma_cpu_to_isax_ch0_stride_y = 8'h0;
  assign dma_cpu_to_isax_ch0_enable = _GEN_46 | _GEN_12;
  assign dma_isax_to_cpu_ch0_cpu_addr =
    _GEN_66 ? _op2b38_b2_reg0_read_data : _op0b05_b3_reg0_read_data;
  assign dma_isax_to_cpu_ch0_isax_addr = _GEN_66 ? 32'hE0 : 32'h2;
  assign dma_isax_to_cpu_ch0_length = {2'h1, ~_GEN_66, 1'h1};
  assign dma_isax_to_cpu_ch0_stride_x = 8'h0;
  assign dma_isax_to_cpu_ch0_stride_y = 8'h0;
  assign dma_isax_to_cpu_ch0_enable = _GEN_66 | _GEN_42;
  assign dma_cpu_to_isax_ch1_isax_addr = 32'hA0;
  assign dma_cpu_to_isax_ch1_length = 4'h5;
  assign dma_cpu_to_isax_ch1_stride_x = 8'h0;
  assign dma_cpu_to_isax_ch1_stride_y = 8'h0;
  assign dma_cpu_to_isax_ch1_enable = _GEN_47;
  assign dma_isax_to_cpu_ch1_cpu_addr = 32'h0;
  assign dma_isax_to_cpu_ch1_isax_addr = 32'h0;
  assign dma_isax_to_cpu_ch1_length = 4'h0;
  assign dma_isax_to_cpu_ch1_stride_x = 8'h0;
  assign dma_isax_to_cpu_ch1_stride_y = 8'h0;
  assign dma_isax_to_cpu_ch1_enable = 1'h0;
endmodule

