#!/usr/bin/env python3
"""Ratios from actual Rocket mcycle measurements, never simulator wall time."""
import json
from pathlib import Path
import re
import sys


def main():
    source = Path(sys.argv[1]).read_text()
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
    Path(sys.argv[2]).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"aggregate: compute={summary['aggregate_compute_speedup']:.3f}x, "
          f"end-to-end={summary['aggregate_end_to_end_speedup']:.3f}x")


if __name__ == "__main__":
    main()
