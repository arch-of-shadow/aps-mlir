// Explicit Rocket-style register boundaries. No latency-padding registers.
module HFNativePipeline #(parameter WIDTH=32, OP=0, PRED=0, LATENCY=3)(
 input wire clock, input wire [31:0] a,b,c, input wire [2:0] rm,
 output wire [36:0] payload
);
 localparam S=WIDTH-8, D=$clog2(S+1);
 generate if(OP<=2 || OP==5) begin: fma
   wire [WIDTH:0] ar,br,cr;
   // Add/sub: a*1 +/- b. Mul: a*b + signed zero (Rocket convention).
   wire [WIDTH-1:0] one={1'b0,8'd127,{(S-1){1'b0}}};
   wire [WIDTH-1:0] bv=OP<=1 ? one : b[WIDTH-1:0];
   wire [WIDTH-1:0] cv=OP<=1 ? b[WIDTH-1:0] : OP==2 ?
       {a[WIDTH-1]^b[WIDTH-1],{(WIDTH-1){1'b0}}} : c[WIDTH-1:0];
   fNToRecFN #(8,S) ca(a[WIDTH-1:0],ar), cb(bv,br), cc(cv,cr);
   reg [WIDTH:0] ra,rb,rc; reg [2:0] rm0,rm1,rm2;
   always @(posedge clock)begin ra<=ar; rb<=br; rc<=cr; rm0<=rm; end
   wire [S-1:0] ma,mb; wire [2*S-1:0] mc;
   wire [5:0] state; wire signed [9:0] exp;
   wire [D-1:0] align_dist; wire [S+1:0] high;
   mulAddRecFNToRaw_preMul #(8,S) pre(1'b1,OP==1 ? 2'b01 : 2'b00,
       ra,rb,rc,rm0,ma,mb,mc,state,exp,align_dist,high);
   reg [2*S:0] product; reg [5:0] state1; reg signed [9:0] exp1;
   reg [D-1:0] dist1; reg [S+1:0] high1;
   always @(posedge clock)begin
     product<=({{(S+1){1'b0}},ma} * mb)+{1'b0,mc};
     state1<=state; exp1<=exp; dist1<=align_dist; high1<=high; rm1<=rm0;
   end
   wire invalid,nan,inf,zero,sign; wire signed [9:0] exponent; wire [S+2:0] sig;
   mulAddRecFNToRaw_postMul #(8,S) post(state1,exp1,dist1,high1,product,rm1,
       invalid,nan,inf,zero,sign,exponent,sig);
   reg invalid2,nan2,inf2,zero2,sign2; reg signed [9:0] exp2; reg [S+2:0] sig2;
   always @(posedge clock)begin
     invalid2<=invalid; nan2<=nan; inf2<=inf; zero2<=zero; sign2<=sign;
     exp2<=exponent; sig2<=sig; rm2<=rm1;
   end
   wire [WIDTH:0] recoded; wire [WIDTH-1:0] ieee; wire [4:0] flags;
   roundRawFNToRecFN #(8,S,0) rounder(1'b1,invalid2,1'b0,nan2,inf2,zero2,
       sign2,exp2,sig2,rm2,recoded,flags);
   recFNToFN #(8,S) encode(recoded,ieee);
   assign payload={flags,{{(32-WIDTH){1'b0}},ieee}};
 end else begin: simple
   reg [31:0] av,bv,cv; reg [2:0] mode;
   always @(posedge clock)begin av<=a; bv<=b; cv<=c; mode<=rm; end
   wire [31:0] data; wire [4:0] flags;
   HFCompute #(.WIDTH(WIDTH),.OP(OP),.PRED(PRED)) core(av,bv,cv,mode,data,flags);
   if(LATENCY==2)begin: output_stage
     reg [36:0] result;
     always @(posedge clock)result<={flags,data};
     assign payload=result;
   end else assign payload={flags,data};
 end endgenerate
endmodule
