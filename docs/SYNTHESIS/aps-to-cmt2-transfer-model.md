# Scheduled TOR to CMT2

`aps-to-cmt2` constructs transactional hardware from scheduled TOR and the APS
memory map. [APSToCMT2.cpp](../../lib/APS/APSToCMT2/APSToCMT2.cpp) creates the
memory pool, adapters, global storage, interfaces, and main module. It loads
external implementations through the CMT2 module-library manifest.

## Control and value ownership

| Component | Responsibility |
|---|---|
| [BlockHandler](../../lib/APS/APSToCMT2/BlockHandler.cpp) | Segment control flow, analyze cross-block values, wire boundary tokens and storage |
| [BBHandler](../../lib/APS/APSToCMT2/BBHandler.cpp) | Group scheduled operations into slots and generate slot rules |
| [LoopHandler](../../lib/APS/APSToCMT2/LoopHandler.cpp) | Loop admission, induction state, iteration issue/retirement, and exit |
| [IfHandler](../../lib/APS/APSToCMT2/IfHandler.cpp) | Conditional and while-region control/data transfer |
| [OpRuleGenerator](../../lib/APS/APSToCMT2/OpRuleGenerator.cpp) | Instruction context admission and function completion |

A block here is a generated control-flow segment; a slot groups operations by
scheduled start time within a regular segment. These differ from an MLIR block.
Tokens coordinate segment/slot execution. Constants and virtual memory references
can be rematerialized rather than transported as data.

Non-pipeline execution uses producer-owned cross-block registers and stage-local
registers, with tokens protecting their use. Parent live-ins can be captured once
into shared registers. A multicycle operation must not allow another execution
context to overwrite its pending state.

Pipeline execution transports live data through stage edges together with tokens,
including stages that do not otherwise consume that value. Pipeline slot edges
use depth-two registered-readiness queues. This bounds storage and applies
backpressure without a combinational readiness cycle. Admission respects the
scheduled II; progress state protects iteration ordering.

## Rules and firing

A `cmt2.rule` has guard and body regions. Explicit guards encode additional
predicates; method readiness from calls also constrains firing. An unconditional
guard is therefore not evidence that a dequeue or enqueue can execute while its
queue is unavailable. Body calls consume input tokens/data, perform operations,
update registers, and enqueue outputs as one transactional firing.

Slot precedence orders later stages before earlier stages where required.
Preserve the inferred call conflicts and declared precedence when modifying
queue behavior. [TokenFIFO.cpp](../../lib/APS/APSToCMT2/TokenFIFO.cpp) can omit
unused token payload registers while retaining the occupancy contract:
`FIFO1_PUSH` permits full dequeue/enqueue replacement; `FIFO2_I` uses registered
readiness. Queue depth is a property of the chosen implementation, not a general
`consumer.start - producer.end` formula.

## Loops and instruction contexts

Sequential loops keep induction/carry state and admit the next iteration after
body completion. Pipelined loops use separate entry, issue, and retire rules.
The current elastic loop backend requires a positive scheduled II and rejects
loop-carried results/iter_args. It also rejects nested control flow and the
listed scalar-memory/copy issue-wait and external-write operations in a pipeline
body; the legality checks in `LoopHandler` are authoritative. Frontend acceptance
of a loop does not imply support for its pipelined realization.

Functions containing memory, floating values, or nested regions receive a shared
execution-context busy bit. It serializes instruction invocations that use shared
state; internal loop iterations can still pipeline. Queueing a command and
admitting an execution context are separate events.

At function exit, a token toggles `instruction_completed_<id>_`, where
`id = (opcode << 8) + funct7` is rendered in decimal. Completion includes final
copy-out waits. The toggle supports instructions with no register result;
command acceptance or idle DMA alone does not establish completion.
See [ports and completion](../ARCHITECTURE/aps-port-convention.md).

## Memory and floating operations

Architectural accesses are split before CMT2 generation. The
[MemoryOpGenerator](../../lib/APS/APSToCMT2/MemoryOpGenerator.cpp) maps issues and
waits onto scalar or DMA methods. A scalar store waits for completion before its
dependents can proceed. Tags and adapter state associate outstanding responses
with their requests. Dependencies must remain attached to temporal operations.

[FloatOpGenerator](../../lib/APS/APSToCMT2/FloatOpGenerator.cpp) issues and collects
from dedicated operation instances; data becomes available at collect/end time.
The [floating-point guide](hardfloat-bf16-fp32-acceptance.md) defines timing,
credits, and formats.

## Inspect generated hardware

After producing `build/design.cmt2.mlir` with `aps-e2e --export cmt2`:

```sh
circt/build/bin/circt-opt build/design.cmt2.mlir -cmt2-print-call-info
circt/build/bin/circt-opt build/design.cmt2.mlir -cmt2-print-conflict-matrix
circt/build/bin/circt-opt build/design.cmt2.mlir -cmt2-print-scheduler
circt/build/bin/circt-opt build/design.cmt2.mlir -cmt2-inline-modules -o build/design.inlined.mlir
```

Use [CMT2 concepts](../../circt/docs/Cmt2/features/Concepts.md) for callable,
interface, and conflict semantics. Regression coverage includes
[FIFO ordering](../../tests/synthesis/test_cmt2_fifo_ordering.py),
[memory ordering](../../tests/synthesis/test_memory_ordering.py),
[storage co-simulation](../../tests/synthesis/test_storage_cosim.py), and
[Spike memory bridging](../../tests/synthesis/test_spike_memory_bridge.py).
