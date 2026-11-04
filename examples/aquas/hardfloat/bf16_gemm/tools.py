#!/usr/bin/env python3
"""Generate BF16 GEMM inputs, passive RTL instrumentation, and result summaries."""
import argparse
import json
from pathlib import Path
import random
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/hardfloat'))
from ieee_oracle import IEEE


def vectors(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0xBF168008)
    oracle = IEEE(16)

    def finite():
        return (rng.randrange(2) << 15) | (rng.randrange(123, 132) << 7) | rng.randrange(128)

    identity = [0x3F80 if i == j else 0 for i in range(8) for j in range(8)]
    ordinary = [finite() for _ in range(64)]
    cases = [([0] * 64, ordinary), (identity, ordinary), (ordinary, identity)]
    for pool in ([0x3F80, 0xBF80, 0x3F00, 0xBF00],
                 [0x3F81, 0x3F83, 0x3F01, 0xBF81],
                 [0x0001, 0x007F, 0x0080, 0x8080],
                 [0x7F7F, 0xFF7F, 0x4000, 0x3F00],
                 [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x7F81]):
        cases.append(([rng.choice(pool) for _ in range(64)],
                      [rng.choice(pool) for _ in range(64)]))
    for test in range(8):
        generate = finite if test < 4 else lambda: rng.getrandbits(16)
        cases.append(([generate() for _ in range(64)], [generate() for _ in range(64)]))

    records = []
    for a, b in cases:
        first, accumulated = [], []
        for i in range(8):
            for j in range(8):
                acc = 0
                for k in range(8):
                    product, _ = oracle.operation("mul", a[i * 8 + k], b[k * 8 + j])
                    acc, _ = oracle.operation("add", acc, product)
                result, _ = oracle.operation("add", acc, 0)
                first.append(result)
                result, _ = oracle.operation("add", acc, result)
                accumulated.append(result)
        matrices = ",\n".join("        {" + ", ".join(f"0x{bits:04x}" for bits in matrix) + "}"
                              for matrix in (a, b, first, accumulated))
        records.append("    {\n" + matrices + "\n    }")
    (output / "vectors.h").write_text(
        "/* Generated exact BF16 multiply/add RNE goldens, B in normal order. */\n"
        "#include <stdint.h>\n"
        "struct gemm_case { uint16_t a[64], b[64], first[64], accumulated[64]; };\n"
        "static const struct gemm_case cases[] = {\n" + ",\n".join(records) + "\n};\n")
    print(f"Generated {len(cases)} 8x8 inputs, two passes each (2048 outputs).")


def benchmark_vectors(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0xBF1680B)
    oracle = IEEE(16)
    fp32 = IEEE(32)
    fp32_expected = [0] * 512
    # Mixed signs and nontrivial mantissas; avoid exceptional-value fast paths.
    def value():
        return (rng.randrange(2) << 15) | (rng.randrange(124, 130) << 7) | rng.randrange(128)
    a, bt, expected = ([0] * 512 for _ in range(3))
    for tile in range(8):
        for row in range(8):
            for col in range(8):
                index = row * 64 + tile * 8 + col
                a[index], bt[index] = value(), value()
        for i in range(8):
            for j in range(8):
                acc = 0
                for k in range(8):
                    product, _ = oracle.operation("mul", a[i * 64 + tile * 8 + k],
                                                   bt[j * 64 + tile * 8 + k])
                    acc, _ = oracle.operation("add", acc, product)
                expected[i * 64 + tile * 8 + j], _ = oracle.operation("add", acc, 0)
                acc32 = 0
                for k in range(8):
                    acc32, _ = fp32.operation("fma", a[i * 64 + tile * 8 + k] << 16,
                                             bt[j * 64 + tile * 8 + k] << 16, acc32)
                fp32_expected[i * 64 + tile * 8 + j] = acc32
    text = "/* Deterministic seed 0xBF1680B, independent BF16 mul/add oracle. */\n"
    for name, values in (("bench_a", a), ("bench_bt", bt), ("bench_expected", expected)):
        text += f"static const uint16_t {name}[512] = {{\n"
        text += "\n".join("    " + ", ".join(f"0x{x:04x}" for x in values[i:i+8]) + ","
                          for i in range(0, 512, 8)) + "\n};\n"
    text += "static const uint32_t bench_fp32_expected[512] = {\n"
    text += "\n".join("    " + ", ".join(f"0x{x:08x}" for x in fp32_expected[i:i+8]) + ","
                      for i in range(0, 512, 8)) + "\n};\n"
    (output / "bench_vectors.h").write_text(text)
    print("Generated benchmark: 8 dense finite BF16 tiles / 512 outputs per trial")


PROFILE_RTL = '''// Simulation-only, passive monitor. No accelerator inputs are driven.
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


def profile(rtl, output):
    rtl_text = rtl.read_text()
    units = re.findall(r'HardFloat_w16_o(\d+)_p0_l\d+_q\d+ (op0b00_l\d+(?:_l\d+)*_b0_fp\d+) \(', rtl_text)
    assert len(units) == 34, 'Profile expects the current 16-multiply/18-add GEMM'
    units.sort(key=lambda p: int(p[1].rsplit('fp', 1)[1]))
    sv = PROFILE_RTL
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


def summarize(log, output):
    source = log.read_text()
    if "BF16 BENCH PASS trials=2 checked=2048" not in source:
        raise SystemExit("Benchmark did not complete correctness checks")
    software = {int(t): int(c) for t, c in re.findall(
        r"BF16_BENCH trial=(\d+) engine=fp32 tiles=8 elements=512 cycles=(\d+)", source)}
    hardware = {int(t): list(map(int, counts)) for t, *counts in re.findall(
        r"BF16_BENCH trial=(\d+) engine=isax tiles=8 elements=512 load=(\d+) compute=(\d+) store=(\d+) total=(\d+)", source)}
    if set(software) != {0, 1} or set(hardware) != {0, 1}:
        raise SystemExit("Missing/invalid benchmark samples")
    results = []
    for trial in range(2):
        load, compute, store, total = hardware[trial]
        assert total == load + compute + store and min(compute, total, software[trial]) > 0
        result = dict(trial=trial, tiles=8, software_cycles=software[trial],
                      isax_load_cycles=load, isax_compute_cycles=compute,
                      isax_store_cycles=store, isax_total_cycles=total,
                      compute_speedup=software[trial] / compute,
                      end_to_end_speedup=software[trial] / total)
        results.append(result)
        print(f"trial={trial}: native FP32={software[trial]} cycles, "
              f"ISAX compute={compute}, DMA+compute={total}; "
              f"speedup compute={result['compute_speedup']:.3f}x, "
              f"end-to-end={result['end_to_end_speedup']:.3f}x")
    sw = sum(r["software_cycles"] for r in results)
    summary = dict(baseline="native FP32 FMA; conversions excluded; separate FP32/BF16 goldens",
                   samples=results, aggregate_compute_speedup=sw / sum(r["isax_compute_cycles"] for r in results),
                   aggregate_end_to_end_speedup=sw / sum(r["isax_total_cycles"] for r in results))
    summary["isax_dma_fraction"] = sum(r["isax_load_cycles"] + r["isax_store_cycles"] for r in results) / sum(r["isax_total_cycles"] for r in results)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"aggregate: compute={summary['aggregate_compute_speedup']:.3f}x, "
          f"end-to-end={summary['aggregate_end_to_end_speedup']:.3f}x")


def summarize_profile(log, output):
    source = log.read_text()
    assert 'BF16 BENCH PASS trials=2 checked=2048' in source
    line = re.search(r'GEMM_PROFILE summary (.*)', source)
    assert line, 'Missing RTL monitor summary'
    summary = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line[1])}
    units = [{k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
             for line in re.findall(r'GEMM_PROFILE fp=.*', source)]
    assert summary['kernels'] == 16 and summary['iterations'] == 512
    assert summary.get('outstanding_iterations', 0) == 0
    assert len(units) == 34 and {u['fp'] for u in units} == set(range(34))
    assert all(u['issues'] == u['collects'] == 512 for u in units)
    summary['units'] = units
    summary['issue_slots_used_fraction'] = sum(u['issues'] for u in units) / (34 * summary['kernel_cycles'])
    summary['mean_iteration_start_gap'] = summary['gap_sum'] / summary['gap_count']
    summary['busy_fraction'] = summary['busy_cycles'] / summary['kernel_cycles']
    # Model supported by per-unit issue/collect offsets: one multiply, eight
    # dependent accumulation adds, and final C add; independent dots run in pairs.
    chain = [0, 1, 3, 5, 7, 9, 11, 13, 15, 16]
    by_id = {u['fp']: u for u in units}
    chain_cycles = sum(by_id[i]['first_collect'] - by_id[i]['first_issue'] for i in chain)
    summary['critical_chain_model_cycles_per_pair'] = chain_cycles
    # Overlapping chains cannot be added to obtain a fraction of elapsed time.
    if summary.get('peak_iterations', 1) == 1:
        summary['critical_chain_model_fraction'] = (chain_cycles * summary['iterations'] / summary['kernel_cycles'])
    output.write_text(json.dumps(summary, indent=2) + '\n')
    print(f"RTL: {summary['kernels']} tiles, {summary['iterations']} output pairs, "
          f"pair span {summary['span_min']}..{summary['span_max']} cycles, "
          f"start gap {summary['gap_min']}..{summary['gap_max']} cycles")
    print(f"FP issue-slot use {100*summary['issue_slots_used_fraction']:.3f}%, "
          f"max in-flight per unit {max(u['peak_pending'] for u in units)}, "
          f"credit-unavailable cycles {sum(u['ready_low'] for u in units)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    handlers = {'vectors': vectors, 'benchmark-vectors': benchmark_vectors,
                'profile': profile, 'summarize': summarize,
                'summarize-profile': summarize_profile}
    for name in handlers:
        command = commands.add_parser(name)
        if name not in {'vectors', 'benchmark-vectors'}:
            command.add_argument('source', type=Path)
        command.add_argument('output', type=Path)
    args = parser.parse_args()
    paths = ([args.source] if hasattr(args, 'source') else []) + [args.output]
    handlers[args.command](*paths)


if __name__ == '__main__':
    main()
