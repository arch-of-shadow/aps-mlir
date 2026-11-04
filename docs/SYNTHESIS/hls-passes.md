# HLS transformations and scheduling

The [production pipeline](../../tools/aps-opt/main.cc) fixes transformation order.
Pass declarations are in [APS Passes.td](../../include/APS/Passes.td) and
[TOR Passes.td](../../include/TOR/Passes.td). Run individual passes with
`build/tools/aps-opt/aps-opt`; inspect `--help` for their current options.

## Loops and affine memory

`aps-raise-scf-to-affine` speculatively normalizes eligible loops locally. Loops
that cannot be raised retain their integer induction-variable types. Converted
loops carry `aps.original_iv_type`, allowing affine lowering to restore the
original width. This preserves narrow counters without globally converting all
loops to index type.

Affine memory inference recognizes supported index expressions. Unrolling and
array partitioning operate before affine loops return to SCF/TOR. Partition
metadata describes dimensions, factors, and cyclic/block mode. Dynamic bank
selection still needs legal per-bank accesses and scheduling dependencies.
`auto-burst-partition` adds cyclic partitioning for `aps.copy` globals without
explicit partition metadata, using a factor of `64 / element_bitwidth`.

Implementation: [APSRaiseSCFToAffine.cpp](../../lib/APS/APSRaiseSCFToAffine.cpp),
[RaiseMemRefToAffine.cpp](../../lib/APS/RaiseMemRefToAffine.cpp),
[LowerAffineFor.cpp](../../lib/TOR/LowerAffineFor.cpp),
[NewArrayPartition.cpp](../../lib/TOR/NewArrayPartition.cpp), and
[SCFToTOR.cpp](../../lib/TOR/SCFToTOR.cpp).
Regression: [affine wrapper tests](../../tests/synthesis/test_aps_raise_scf_to_affine_pass.py)
and [burst partition tests](../../tests/synthesis/test_auto_burst_partition.py).

## Memory binding and scheduling

Functional `aps.load`, `aps.store`, and `aps.copy` express behavior. The
`aps-functional-to-arch` pass validates interface models, selects interfaces using
shared startup/transfer/cache costs, and decomposes constant-size copies into
legal transactions. Unknown transfer lengths must be legalized before binding.

`aps-schedule-memory` orders architectural transactions and emits `after`
dependencies. TOR scheduling consumes those dependencies along with dataflow,
resource constraints, and operation timing. Read/write ordering must survive
request/collect splitting; read [memory interfaces](../ARCHITECTURE/aps-itfc.md)
for the platform model.

The scheduler uses a control/dataflow graph and system-of-difference-constraints
solver. Resources supply latency, initiation interval, availability, and delay
information. `--clock` constrains scheduling; it does not measure physical timing.
Scheduled operations carry `ref_starttime` and `ref_endtime`; loops also carry
pipeline/II information. A requested pipeline may be realized sequentially when
its structure requires fallback. Inspect `aps.pipeline_requested`,
`aps.pipeline_achieved_ii`, and `aps.pipeline_fallback` on scheduled IR where emitted.
The ScaleISA driver collects these outcomes in its report.

Implementation: [APSFunctionalToArch.cpp](../../lib/APS/APSFunctionalToArch.cpp),
[APSMemorySchedule.cpp](../../lib/APS/APSMemorySchedule.cpp),
[TORSchedulePass.cpp](../../lib/TOR/TORSchedulePass.cpp),
[CDFG.cpp](../../lib/Schedule/CDFG.cpp), and
[SDCSchedule.cpp](../../lib/Schedule/SDCSchedule.cpp).
Tests: [functional binding](../../tests/synthesis/test_functional_to_arch.py),
[memory ordering](../../tests/synthesis/test_memory_ordering.py), and
[TOR scheduling](../../tests/synthesis/test_tor_schedule_pass.py).

## Scratchpad elision

The driver automatically trials direct CPU reads for eligible staged read-only
buffers. Legality checks include access coverage, contiguous access patterns,
known loop bounds, pipeline shape, cache hints, and absence of conflicting writes.
A candidate's eligibility alone is insufficient: a real rescheduling comparison
must prove non-increasing invocation latency and every loop II. Unsupported cost
comparisons and failed/regressing trials retain staging.

Inspect [ScratchpadElision.cpp](../../lib/APS/ScratchpadElision.cpp), the trial
logic in [main.cc](../../tools/aps-opt/main.cc), and
[regressions](../../tests/synthesis/test_scratchpad_elision.py).
Debug builds expose `--debug-only=scratchpad-elision`; there is no separate public
elision-enable or report option.

## Storage planning before CMT2

`aps-optimize-storage` runs after temporal splitting and canonicalization:

- Rematerialize widening at later uses to transport narrower values.
- Compact eligible integer-square storage with exact reconstruction.
- Omit an unused scalar memory adapter while retaining external ports.
- Compact native register-operand command payloads when proven legal.
- Omit response-data storage only when every result is a literal 32-bit zero.
- Plan token payload elision while retaining queue occupancy/readiness behavior.

These transformations preserve firing cycles and effect ordering. Do not run CSE
again after rematerialization. Use [OptimizeStorage.cpp](../../lib/APS/OptimizeStorage.cpp)
and [storage tests](../../tests/synthesis/test_storage_optimization.py) for the
exact pattern restrictions. Scalar store completion and whole-instruction context
protection are described in [CMT2 lowering](aps-to-cmt2-transfer-model.md).
