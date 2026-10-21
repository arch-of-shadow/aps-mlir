#!/usr/bin/env python3
"""Bind passive counters into the actual GEMM main module in Rocket RTL."""
import re
import sys
from pathlib import Path

rtl, output = map(Path, sys.argv[1:])
rtl_text = rtl.read_text()
units = re.findall(r'HardFloat_w16_o(\d+)_p0_l\d+_q\d+ (op0b00_l\d+(?:_l\d+)*_b0_fp\d+) \(', rtl_text)
assert len(units) == 34, 'Profile expects the current 16-multiply/18-add GEMM'
units.sort(key=lambda p: int(p[1].rsplit('fp', 1)[1]))
sv = '''// Simulation-only, passive monitor. No accelerator inputs are driven.
module BF16GemmProfile(
 input wire clk, rst, command, response,
 input wire [6:0] opcode,
 input wire iteration_start, iteration_end,
 input wire [33:0] issue, collect, ready
);
 integer tick=0, command_start=0, first_start=-1, previous_start=0;
 integer starts[0:63], head=0, tail=0, inflight=0, peak_iterations=0;
 integer kernels=0, kernel_cycles=0, iterations=0, busy_cycles=0;
 integer spans=0, min_span=1000000, max_span=0;
 integer gaps=0, gap_count=0, min_gap=1000000, max_gap=0;
 integer issues[0:33], collects[0:33], pending[0:33], peak[0:33];
 integer blocked[0:33], first_issue[0:33], first_collect[0:33];
 integer i, delta;
 reg active=0, previous_valid=0;
 initial begin
   for(i=0;i<34;i=i+1)begin
     issues[i]=0; collects[i]=0; pending[i]=0; peak[i]=0;
     blocked[i]=0; first_issue[i]=-1; first_collect[i]=-1;
   end
 end
 always @(posedge clk) if(!rst)begin
   tick=tick+1;
   if(command && opcode==7'h0b)begin
     active=1; command_start=tick; previous_valid=0;
   end
   if(inflight>0)busy_cycles=busy_cycles+1;
   if(iteration_start)begin
     delta=0;
     if(previous_valid)begin
       delta=tick-previous_start;
       gaps=gaps+delta; gap_count=gap_count+1;
       if(delta<min_gap)min_gap=delta;
       if(delta>max_gap)max_gap=delta;
     end
     previous_start=tick; previous_valid=1;
     if(inflight==64)$fatal(1,"GEMM profile timestamp queue overflow");
     starts[tail]=tick; tail=(tail+1)%64; inflight=inflight+1;
     if(inflight>peak_iterations)peak_iterations=inflight;
     if(first_start<0)first_start=tick;
     if(kernels==0)
       $display("GEMM_PROFILE admission iteration=%0d offset=%0d gap=%0d", iterations, tick-first_start, delta);
     iterations=iterations+1;
   end
   for(i=0;i<34;i=i+1)begin
     if(active && !ready[i])blocked[i]=blocked[i]+1;
     if(issue[i])begin
       issues[i]=issues[i]+1; pending[i]=pending[i]+1;
       if(pending[i]>peak[i])peak[i]=pending[i];
       if(first_issue[i]<0)first_issue[i]=tick-first_start;
     end
     if(collect[i])begin
       collects[i]=collects[i]+1; pending[i]=pending[i]-1;
       if(first_collect[i]<0)first_collect[i]=tick-first_start;
     end
   end
   if(iteration_end)begin
     if(inflight==0)$fatal(1,"GEMM profile completion without admission");
     delta=tick-starts[head]; head=(head+1)%64; inflight=inflight-1;
     spans=spans+delta;
     if(delta<min_span)min_span=delta;
     if(delta>max_span)max_span=delta;
   end
   if(response && active)begin
     delta=tick-command_start;
     $display("GEMM_PROFILE kernel=%0d cycles=%0d",kernels,delta);
     if(kernels==0)begin
       $display("GEMM_PROFILE first_kernel iterations=%0d span_min=%0d span_max=%0d gap_min=%0d gap_max=%0d peak_iterations=%0d outstanding_iterations=%0d",
         iterations,min_span,max_span,min_gap,max_gap,peak_iterations,inflight);
     end
     kernels=kernels+1; kernel_cycles=kernel_cycles+delta; active=0;
   end
 end
 final begin
   $display("GEMM_PROFILE summary kernels=%0d kernel_cycles=%0d iterations=%0d busy_cycles=%0d span_sum=%0d span_min=%0d span_max=%0d gap_sum=%0d gap_count=%0d gap_min=%0d gap_max=%0d peak_iterations=%0d outstanding_iterations=%0d",
     kernels,kernel_cycles,iterations,busy_cycles,spans,min_span,max_span,gaps,gap_count,min_gap,max_gap,peak_iterations,inflight);
   for(integer j=0;j<34;j=j+1)
     $display("GEMM_PROFILE fp=%0d issues=%0d collects=%0d peak_pending=%0d ready_low=%0d first_issue=%0d first_collect=%0d",
       j,issues[j],collects[j],peak[j],blocked[j],first_issue[j],first_collect[j]);
 end
endmodule
bind main BF16GemmProfile bf16_profile(
 .clk(clk), .rst(rst),
 .command(rocc_cmd_enable && rocc_cmd_ready),
 .response(rocc_resp_rocc_resp_to_bus_enable && rocc_resp_rocc_resp_to_bus_ready),
 .opcode(rocc_cmd_rocc_cmd_opcode),
'''
prefix = units[0][1].rsplit('_fp', 1)[0]
assert all(u.rsplit('_fp', 1)[0] == prefix for _, u in units)
tokens = re.findall(r'FIFO(?:1_PUSH|2_I)_\w+ (' + re.escape(prefix) + r'_s(\d+)tok) \(', rtl_text)
if tokens:
    # First-stage enqueue and last-stage dequeue bracket each iteration.
    # Timestamps are queued because several iterations can be in flight.
    tokens.sort(key=lambda p: int(p[1]))
    start, end = tokens[0][0], tokens[-1][0]
    sv += f' .iteration_start({start}.enq_enable && {start}.enq_ready),\n'
    sv += f' .iteration_end({end}.deq_enable && {end}.deq_ready),\n'
else:
    busy = prefix + '_fp_busy'
    assert re.search(r'\b' + busy + r' \(', rtl_text), 'Missing iteration monitor signals'
    sv += f' .iteration_start({busy}.write_enable && {busy}.write_data),\n'
    sv += f' .iteration_end({busy}.write_enable && !{busy}.write_data),\n'
for port, expr in [('issue', '{u}.issue_enable && {u}.issue_ready'),
                   ('collect', '{u}.collect_enable && {u}.collect_ready'),
                   ('ready', '{u}.issue_ready')]:
    sv += f' .{port}({{\n' + ',\n'.join('  '+expr.format(u=u) for _, u in reversed(units)) + '\n })' + (',\n' if port != 'ready' else '\n);\n')
if not output.exists() or output.read_text() != sv:
    output.write_text(sv)
