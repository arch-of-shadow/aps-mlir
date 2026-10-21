#!/usr/bin/env python3
"""Validate passive counters from the full Rocket/RoCC benchmark simulation."""
import json
from pathlib import Path
import re
import sys

source = Path(sys.argv[1]).read_text()
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
Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + '\n')
print(f"RTL: {summary['kernels']} tiles, {summary['iterations']} output pairs, "
      f"pair span {summary['span_min']}..{summary['span_max']} cycles, "
      f"start gap {summary['gap_min']}..{summary['gap_max']} cycles")
print(f"FP issue-slot use {100*summary['issue_slots_used_fraction']:.3f}%, "
      f"max in-flight per unit {max(u['peak_pending'] for u in units)}, "
      f"credit-unavailable cycles {sum(u['ready_low'] for u in units)}")
