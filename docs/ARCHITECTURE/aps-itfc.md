# Memory interfaces and DMA

APS separates functional memory behavior from platform-bound transactions and
scheduled issue/completion operations. Operation definitions are in
[APSOps.td](../../include/APS/APSOps.td).

| Level | Operations and responsibility |
|---|---|
| Functional | `aps.load`, `aps.store`, `aps.copy`: describe CPU-memory behavior |
| Architectural | `aps.load_by`, `aps.store_by`, `aps.copy_by`: select an `aps.memitfc` and legal transaction shape |
| Temporal | Issue/wait pairs: preserve timing dependencies and wait for completion |
| Hardware | CMT2 scalar-memory/DMA methods, scratchpad banks, and adapters |

`aps.read_smem` and `aps.write_smem` access accelerator scratchpads; scalar globals
and CSR-backed globals have separate register/interface paths.

## Platform model

[examples/interface.json](../../examples/interface.json) is a complete native
configuration. Each interface supplies:

| Field | Meaning |
|---|---|
| `width` | Bytes per beat |
| `min_beats`, `min_write_beats`, `max_beats` | Legal transaction beat bounds |
| `inflight` | Outstanding transaction capacity |
| `read_latency`, `write_completion` | Read/write completion timing model |
| `cache_line`, `cache_hint` | Cache-line bytes and `warm`/`cold` assumption |

Native `cpuitfc` uses one 4-byte beat and at most two outstanding transactions.
Native `busitfc` uses 8-byte beats, at most 16 beats per transaction, at least two
write beats, and two inflight channels. Beat bounds must be powers of two and the
cache-line size must be divisible by width. The exporter requires a bus interface.
Its `maxBurstBytes = width * max_beats`; its `nXacts = inflight`.

[APSFunctionalToArch.cpp](../../lib/APS/APSFunctionalToArch.cpp) validates and
selects modeled interfaces, jointly accounting for shared activation/transfer/cache
costs within a region. It decomposes constant-size copies into legal transactions.
Dynamic lengths require normalization first. If IR provides no interfaces, the
pass materializes backward-compatible defaults; explicit JSON is preferable for
reproducible integration.

[APSMemorySchedule.cpp](../../lib/APS/APSMemorySchedule.cpp) schedules
architectural transactions with timing windows and emits `after` dependencies.
Read ordering prefers warm hierarchy levels; writes prefer cold levels.
[LowerAPSMemToReqCollect.cpp](../../lib/APS/LowerAPSMemToReqCollect.cpp) splits
operations after TOR scheduling. Keep dependency and decomposition-group semantics
intact when extending these passes.

## Hardware and memory visibility

[MemoryGenerator.cpp](../../lib/APS/APSToCMT2/MemoryGenerator.cpp) constructs the
banked scratchpad pool from the memory map. Partitioning supplies the banks needed
for 64-bit burst access. Generated interfaces expose split burst reads and writes;
scalar HellaCache and DMA traffic use separate adapters.

The native two-channel DMA and trigger wrapper are in
[generators/aps resources](../../thirdparty/chipyard/generators/aps/src/main/resources/vsrc/),
with TileLink integration in
[TlDmaMultichannel.scala](../../thirdparty/chipyard/generators/aps/src/main/scala/TlDmaMultichannel.scala).
The generated operation's issue/wait protocol defines when data is usable and
when writeback is complete. DMA idle by itself is insufficient for instruction
completion.

Software must supply buffers aligned for the selected transaction size and obey
the platform's memory-visibility contract. The ScaleISA compiler aligns owned
buffered storage and emits ordering fences for memory CIs; external callers remain
responsible for their buffer allocation and interface alignment.

## Tests

[Functional binding](../../tests/test_mlir/test_functional_to_arch.py),
[memory model analysis](../../tests/test_mlir/test_memory_analysis.py),
[memory ordering](../../tests/test_mlir/test_memory_ordering.py), and
[configuration](../../tests/test_mlir/test_interface_config.py) cover binding,
models, decomposition, and capacity limits. Actual scalar/DMA completion is checked
in [storage co-simulation](../../tests/test_mlir/test_storage_cosim.py).
