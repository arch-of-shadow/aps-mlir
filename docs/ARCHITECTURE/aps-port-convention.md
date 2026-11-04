# Generated ports, CSRs, and completion

The generated `main` module and Chipyard `MainBlackBoxIO` must expose the same
ports. The checked-in wrapper definition is
[ApsWrapper.scala](../../thirdparty/chipyard/generators/aps/src/main/scala/ApsWrapper.scala);
CMT2 interface generation lives in
[InterfaceGenerator.cpp](../../lib/APS/APSToCMT2/InterfaceGenerator.cpp).
Directions below are relative to the accelerator.

## RoCC and memory ports

| Group | Ports / behavior |
|---|---|
| Clock/reset | `clk`, `rst` |
| Command | Inputs `rocc_cmd_enable`, `rocc_cmd_rocc_cmd_*`; output `rocc_cmd_ready` |
| Response | Outputs `rocc_resp_rocc_resp_to_bus_enable`, `rocc_resp_rocc_resp_to_bus_result_rd`, `rocc_resp_rocc_resp_to_bus_result_rddata`; input `rocc_resp_rocc_resp_to_bus_ready` |
| Scalar memory | Outgoing `hella_cmd_hella_cmd_to_bus_*`, incoming `hella_resp_*` |
| DMA | `dma_cpu_to_isax_ch0_*`, `dma_isax_to_cpu_ch0_*`, corresponding channel 1 groups, and idle polling |
| Scratchpad burst | `burst_read_0_*`, `burst_read_1_*`, `burst_write_*` |

Command fields are `funct`, `rs1`, `rs2`, `rd`, `xs1`, `xs2`, `xd`, `opcode`,
`rs1data`, and `rs2data`. Selectors are five bits, data is 32 bits, and opcode/funct
are seven bits. A method transfers on `enable && ready`; output payload remains
valid under backpressure. Value results may have a trailing underscore, for example
`burst_read_1_`. Use the actual generated SV or wrapper definition for the complete
port list and widths.

## CSR declarations and configuration

CADL can declare an addressed register:

```cadl
#[csr_address(0x801)]
register cfg: u32;
```

Read it with `_csr[cfg]`. The frontend emits a CSR-backed global. Interface
configuration discovers IR globals carrying `csr_address`: each must have a unique
12-bit address/name and one i32 element. JSON `csrs` entries can override `mask`
and `init` for an existing IR CSR. An optional `id` must agree with IR; a JSON
entry cannot invent a missing CSR. Masks and initial values fit RV32.

The emitted YAML uses the IR symbol as the CSR name. Its ordered CSR list defines
wrapper lane order; software accesses the address. A name contributes the
`csr_<name>` prefix, so `cfg` produces:

| Port | Direction / meaning |
|---|---|
| `csr_cfg_value_` | Input, 32-bit committed CPU CSR value |
| `csr_cfg_value_ready` | Input, value-method readiness |
| `csr_cfg_set_cfg_sdata` | Output, 32-bit accelerator update |
| `csr_cfg_set_ready` | Input, update-method readiness |
| `csr_cfg_set_enable` | Output, update enable |

The wrapper drives value/set readiness true and CPU CSR stall false. CPU-side
`ren`, `wen`, and address decoding remain inside Rocket. Committed value and
accelerator set-data connect through these flattened ports.

A global CSR read is emitted where the corresponding APS operation executes;
the CSR is not automatically appended to the RoCC command FIFO. Keep configuration
stable for its required execution interval. If an instruction needs a snapshot,
capture it in its computation and preserve that value through later rules.
Software generation keeps CSR operand writes together with their custom instruction;
that does not introduce an additional hardware snapshot mechanism.

Implementation: [RegisterOpGenerator.cpp](../../lib/APS/APSToCMT2/RegisterOpGenerator.cpp),
[InterfaceConfig.h](../../tools/aps-opt/InterfaceConfig.h), and the simulator's
[CSR bridge](../../tools/aps-cosim/csr_runtime_bridge.cc).

## Instructions without register results

Each generated instruction exposes `instruction_completed_<id>_`, a one-bit toggle
updated at function exit. The decimal `id` is `(opcode << 8) + funct7`. Completion
includes final scalar-store/copy-out waits.

The wrapper discovers these ports from RTL. For `xd=0`, it tracks the completion
state at admission and waits for a change before releasing the pending command.
No fabricated register response is required. Keep completion connected when
changing blackbox construction, even if an instruction has no result data.

Tests cover [CSR matching](../../tests/test_megg_csr.py),
[configuration export](../../tests/test_mlir/test_interface_config.py), and
[scalar/DMA completion](../../tests/test_mlir/test_storage_cosim.py).
