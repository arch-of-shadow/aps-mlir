// Native arithmetic pipelines plus lossless, flow-through response buffering.
module HFUnit #(parameter integer WIDTH=32, OP=0, PRED=0,
 LATENCY=`HF_NATIVE_LATENCY(WIDTH,OP), CAPACITY=`HF_NATIVE_CAPACITY(WIDTH,OP))(
 input wire clock,reset,issue_enable, output wire issue_ready,
 input wire [31:0] operand0,operand1,operand2, input wire [2:0] rounding_mode,
 input wire collect_enable, output wire collect_ready,
 output wire [31:0] collect_data, output wire [4:0] collect_flags
);
 localparam S=WIDTH-8;
 localparam NATIVE_LATENCY=`HF_NATIVE_LATENCY(WIDTH,OP);
 // Invalid contracts must fail both simulation and synthesis/elaboration.
 generate if(LATENCY!=NATIVE_LATENCY || CAPACITY<1 || (WIDTH!=16 && WIDTH!=32) || OP<0 || OP>18 || ((OP==3 || OP==4) && CAPACITY!=1))begin: bad_contract
   HF_INVALID_TIMING_CONTRACT invalid_contract();
 end endgenerate
 generate if(OP==3 || OP==4) begin: iterative
   wire [WIDTH:0] ar,br; reg [WIDTH:0] a,b; reg [2:0] mode;
   fNToRecFN #(8,S) ca(operand0[WIDTH-1:0],ar),cb(operand1[WIDTH-1:0],br);
   reg busy,request,held;
   wire core_ready,core_valid,sqrt_out; wire [WIDTH:0] recoded;
   wire [WIDTH-1:0] ieee; wire [4:0] flags;
   reg [36:0] saved;
   divSqrtRecFN_small #(8,S,0) core(!reset,clock,1'b1,core_ready,
       request,OP==4,a,b,mode,core_valid,sqrt_out,recoded,flags);
   recFNToFN #(8,S) encode(recoded,ieee);
   assign collect_ready=!reset && (held || (busy && core_valid));
   assign {collect_flags,collect_data}=held ? saved : {flags,{{(32-WIDTH){1'b0}},ieee}};
   wire consume=collect_enable && collect_ready;
   assign issue_ready=!reset && (!busy || consume) && core_ready;
   wire accept=issue_enable && issue_ready;
   always @(posedge clock)begin
     if(reset)begin busy<=0; request<=0; held<=0; end
     else begin
       request<=accept;
       if(consume)begin busy<=0; held<=0; end
       if(accept)begin busy<=1; a<=ar; b<=br; mode<=rounding_mode; end
       if(core_valid && busy && !held && !consume)begin
         saved<={flags,{{(32-WIDTH){1'b0}},ieee}}; held<=1;
       end
     end
   end
 end else begin: fixed
   wire [36:0] payload;
   HFNativePipeline #(.WIDTH(WIDTH),.OP(OP),.PRED(PRED),.LATENCY(LATENCY)) math_pipe(
       clock,operand0,operand1,operand2,rounding_mode,payload);
   localparam CW=$clog2(CAPACITY+1), PW=CAPACITY<2 ? 1 : $clog2(CAPACITY);
   reg [CW-1:0] outstanding,count;
   reg [PW-1:0] head,tail;
   reg [36:0] queue[0:CAPACITY-1];
   reg [LATENCY-1:0] valid;
   wire arrival=valid[LATENCY-1] && !reset;
   assign collect_ready=!reset && (count!=0 || arrival);
   assign {collect_flags,collect_data}=count!=0 ? queue[head] : payload;
   wire consume=collect_enable && collect_ready;
   assign issue_ready=!reset && (outstanding<CW'(CAPACITY) || consume);
   wire accept=issue_enable && issue_ready;
   wire enqueue=arrival && (count!=0 || !consume);
   wire dequeue=consume && count!=0;
   always @(posedge clock)begin
     if(reset)begin outstanding<=0; count<=0; head<=0; tail<=0; valid<=0; end
     else begin
       valid<=(valid<<1)|{{(LATENCY-1){1'b0}},accept};
       case({accept,consume})
         2'b10:outstanding<=outstanding+1'b1;
         2'b01:outstanding<=outstanding-1'b1;
         default:;
       endcase
       case({enqueue,dequeue})
         2'b10:count<=count+1'b1;
         2'b01:count<=count-1'b1;
         default:;
       endcase
       if(enqueue)begin queue[tail]<=payload; tail<=tail==PW'(CAPACITY-1) ? 0 : tail+1'b1; end
       if(dequeue)head<=head==PW'(CAPACITY-1) ? 0 : head+1'b1;
     end
   end
`ifndef SYNTHESIS
   always @(posedge clock)if(!reset)begin
     assert(outstanding<=CW'(CAPACITY) && count<=outstanding) else $fatal(1,"HFUnit credit invariant");
     assert(!(enqueue && count==CW'(CAPACITY) && !dequeue)) else $fatal(1,"HFUnit overflow");
   end
`endif
 end endgenerate
endmodule
