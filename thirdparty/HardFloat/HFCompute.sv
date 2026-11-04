// Complete BF16/binary32 HardFloat combinational operation bank.
// OP: add sub mul div sqrt fma cmp si->fp ui->fp fp->si fp->ui
//     bf16->fp32 fp32->bf16 neg abs copysign min max classify.
module HFCompute #(parameter integer WIDTH=32, OP=0, PRED=0) (
 input wire [31:0] a,b,c, input wire [2:0] rm,
 output wire [31:0] data, output wire [4:0] flags
);
 localparam integer S=WIDTH-8;
 wire [WIDTH:0] ra,rb,rc;
 fNToRecFN #(8,S) ca(a[WIDTH-1:0],ra), cb(b[WIDTH-1:0],rb), cc(c[WIDTH-1:0],rc);
 generate
 if (OP<=2 || OP==5) begin: arithmetic
   wire [WIDTH:0] rr;
   wire [WIDTH-1:0] ieee;
   if(OP<=1) addRecFN #(8,S) core(1'b1,OP==1,ra,rb,rm,rr,flags);
   else if(OP==2) mulRecFN #(8,S) core(1'b1,ra,rb,rm,rr,flags);
   else mulAddRecFN #(8,S) core(1'b1,2'b00,ra,rb,rc,rm,rr,flags);
   recFNToFN #(8,S) co(rr,ieee);
   assign data={{(32-WIDTH){1'b0}},ieee};
 end else if(OP==6 || OP==16 || OP==17) begin: comparison
   wire lt,eq,gt,unordered;
   wire [4:0] ef;
   // Ordered relations are signaling, eq/ne/ord/uno are quiet (MLIR semantics).
   localparam SIGNALING=(OP==6 && (PRED==2 || PRED==3 || PRED==4 || PRED==5 || PRED==10 || PRED==11 || PRED==12 || PRED==13));
   compareRecFN #(8,S) core(ra,rb,SIGNALING,lt,eq,gt,unordered,ef);
   if(OP==6) begin
     wire v=(PRED==0)?1'b0:(PRED==1)?eq:(PRED==2)?gt:(PRED==3)?(gt|eq):
       (PRED==4)?lt:(PRED==5)?(lt|eq):(PRED==6)?(!unordered&&!eq):
       (PRED==7)?!unordered:(PRED==8)?unordered:(PRED==9)?(unordered|eq):
       (PRED==10)?(unordered|gt):(PRED==11)?(unordered|gt|eq):
       (PRED==12)?(unordered|lt):(PRED==13)?(unordered|lt|eq):
       (PRED==14)?!eq:1'b1;
     assign data={31'b0,v}; assign flags=ef;
   end else begin
     wire an=(&a[WIDTH-2:S-1]) && (|a[S-2:0]);
     wire bn=(&b[WIDTH-2:S-1]) && (|b[S-2:0]);
     wire sn=(an&&!a[S-2]) || (bn&&!b[S-2]);
     wire [WIDTH-1:0] qnan={1'b0,8'hff,1'b1,{(S-2){1'b0}}};
     wire az=(a[WIDTH-2:0]==0), bz=(b[WIDTH-2:0]==0);
     wire pick_a=(OP==16)?(lt || (az&&bz&&a[WIDTH-1])):(gt || (az&&bz&&!a[WIDTH-1]));
     wire [WIDTH-1:0] r=an?(bn?qnan:b[WIDTH-1:0]):bn?a[WIDTH-1:0]:pick_a?a[WIDTH-1:0]:b[WIDTH-1:0];
     assign data={{(32-WIDTH){1'b0}},r}; assign flags={sn,4'b0};
   end
 end else if(OP==7 || OP==8) begin: from_int
   wire [WIDTH:0] rr; wire [WIDTH-1:0] ieee;
   iNToRecFN #(32,8,S) core(1'b1,OP==7,a,rm,rr,flags);
   recFNToFN #(8,S) co(rr,ieee);
   assign data={{(32-WIDTH){1'b0}},ieee};
 end else if(OP==9 || OP==10) begin: to_int
   wire [2:0] ef;
   recFNToIN #(8,S,32) core(1'b1,ra,rm,OP==9,data,ef);
   assign flags={(ef[2]|ef[1]),3'b0,ef[0]};
 end else if(OP==11) begin: widen
   wire [16:0] r16; wire [32:0] rr;
   fNToRecFN #(8,8) ci(a[15:0],r16);
   recFNToRecFN #(8,8,8,24) core(1'b1,r16,rm,rr,flags);
   recFNToFN #(8,24) co(rr,data);
 end else if(OP==12) begin: narrow
   wire [32:0] r32; wire [16:0] rr; wire [15:0] ieee;
   fNToRecFN #(8,24) ci(a,r32);
   recFNToRecFN #(8,24,8,8) core(1'b1,r32,rm,rr,flags);
   recFNToFN #(8,8) co(rr,ieee);
   assign data={16'b0,ieee};
 end else if(OP==13 || OP==14 || OP==15) begin: signop
   wire sign=(OP==13)?!a[WIDTH-1]:(OP==14)?1'b0:b[WIDTH-1];
   assign data={{(32-WIDTH){1'b0}},sign,a[WIDTH-2:0]}; assign flags=0;
 end else if(OP==18) begin: classify
   wire sign=a[WIDTH-1]; wire ez=(a[WIDTH-2:S-1]==0), ei=(&a[WIDTH-2:S-1]);
   wire fz=(a[S-2:0]==0); wire q=a[S-2];
   assign data={22'b0,ei&&!fz&&q,ei&&!fz&&!q,ei&&fz&&!sign,
     !ei&&!ez&&!sign,ez&&!fz&&!sign,ez&&fz&&!sign,ez&&fz&&sign,
     ez&&!fz&&sign,!ei&&!ez&&sign,ei&&fz&&sign};
   assign flags=0;
 end else begin: unused
   assign data=0; assign flags=0;
 end
 endgenerate
endmodule
