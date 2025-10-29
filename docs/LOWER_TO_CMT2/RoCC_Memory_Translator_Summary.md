# RoCC and Memory Translator Design Summary

## RoCC Translator Architecture

### Core Responsibility
The RoCC translator bridges the **Rocket Chip Coprocessor Interface** with the accelerator's internal execution units. It dispatches commands based on opcodes and manages response flow.

### Key Rules and Constraints

#### 1. **Opcode-Based Command Dispatch**
```rust
// Rust Implementation: Command queue per opcode
for opcode in opcodes {
    let rocc_cmd_queue = instance!(stl::fifo1_push(&rocc_cmd_t));
    let cmd_to_user = named_method! {
        format!("cmd_to_user_{}", opcode);
        () -> (cmd) {
            ret!(rocc_cmd_queue.deq());
        }
    };
}
```

**CMT2 Translation Rules**:
- Each opcode gets its own FIFO queue (FIFO1Push_w64 for 64-bit RoCC commands)
- Commands are 64-bit bundles: `funct(7) + rs1(5) + rs2(5) + rd(5) + xs1(1) + xs2(1) + xd(1) + opcode(7) + rs1data(32) + rs2data(32)`
- Interface must provide `get_cmd()` value that returns command for specific opcode
- **Guard Rule**: Only provide command when FIFO is not empty

#### 2. **Interface Relationships**
```
TopLevel Module
├── RoCCAdapter (Consumer)
│   ├── InterfaceDecl: "RoCCCmdInterface"
│   └── InterfaceDecl: "RoCCRespInterface"
├── Execution Units (Provider)
│   └── Methods: read_gpr(), write_gpr(), etc.
└── Interface Bindings
    ├── "RoCCCmdBinding" -> Execution Units
    └── "RoCCRespBinding" -> RoCCAdapter
```

**Interface Pattern**:
```cpp
// Circuit Level
auto* roccCmdInterface = circuit.addInterface("RoCCCmdInterface");
roccCmdInterface->addValue("get_cmd", {}, {uint64Type});

// Consumer (RoCCAdapter)
auto* roccCmdDecl = roccAdapter->addInterfaceDecl("RoCCCmdInterface");
auto* cmd = roccCmdDecl->callValue("get_cmd", builder);

// Provider (Backend Execution)
auto* cmdToUser = roccAdapter->addMethod("cmd_to_user_" + opcode, {}, {cmdType});
cmdToUser->guard(fifoNotEmptyGuard);
cmdToUser->body(fifoDequeueBody);
```

#### 3. **Register File Interface**
```rust
// Rust Backend trait methods
fn request_instruction(&self, opcode: u32) -> Var;
fn read_gpr(&self, reg_id: u32) -> Var;
fn write_gpr(&self, reg_id: u32, var: &Var);
fn check_num_rs(&self) -> usize;
fn check_num_rd(&self) -> usize;
```

**Translation to CMT2**:
- RS1/RS2 become bypass registers with `read()`/`write(data)` methods
- RD becomes bypass register for response destination
- Backend implements methods that interface calls can invoke
- **Rule**: Register access must be synchronous with clock/reset

### Data Flow Pattern
```
RoCC Command → opcode decode → FIFO queue → cmd_to_user_N → Backend.read_gpr() →
Execution → Backend.write_gpr() → resp_from_user → Response FIFO → Rocket Chip
```

## Memory Translator Architecture

### Core Responsibility
The Memory translator bridges **HellaCache Interface** (TileLink protocol) with a simplified **User Memory Protocol** used by the accelerator. It handles buffered read/write operations with tag-based tracking.

### Key Rules and Constraints

#### 1. **Protocol Translation**
```rust
// HellaCache → User Memory Protocol Translation
let hella_cmd = HellaCacheCmdBundle {
    addr: user_cmd.addr,
    tag: user_cmd.tag,
    cmd: user_cmd.cmd.eq(MemoryOP::MEMREAD.lit()).mux(0.uint(5), 1.uint(5)),
    size: user_cmd.size,
    signed: false.uint(1),
    phys: false.uint(1),
    data: user_cmd.data,
    mask: user_cmd.mask
};
```

**Protocol Mapping Rules**:
- **User Memory Read** (`cmd = 0`) → **HellaCache Read** (`cmd = 0`)
- **User Memory Write** (`cmd = 1`) → **HellaCache Write** (`cmd = 1`)
- **Address**: Direct passthrough (32-bit)
- **Tag**: Direct passthrough (8-bit) - used for response matching
- **Data**: Direct passthrough (32-bit)
- **Size/Mask**: Direct passthrough

#### 2. **Buffered Response Tracking**
```rust
// Two-slot response buffer with tag-based matching
let slot_0 = instance!(stl::reg(&Type::UInt(32)));
let slot_0_tag = instance!(stl::reg(&Type::UInt(32)));
let slot_0_txd = instance!(stl::reg(&Type::UInt(1)));
let slot_0_rxd = instance!(stl::reg(&Type::UInt(1)));

// Response collection with ordering
let can_collect = slot0_ready & is_earlier;
```

**Buffering Rules**:
- **2-slot buffer** for pending memory responses
- **Tag-based matching**: Response matched by address tag
- **Ordering preservation**: Earlier responses must be collected first
- **Ready signals**: `txd` (transmit ready) and `rxd` (receive ready) per slot
- **Guard Rule**: Only collect when both ready and earliest conditions met

#### 3. **Interface Relationships**
```
TopLevel Module
├── MemoryAdapter (Translator)
│   ├── InterfaceDecl: "HellaCacheInterface" (consume)
│   ├── InterfaceDecl: "UserMemoryInterface" (provide)
│   └── Internal: Response buffer slots
├── HellaCache Slave (Provider)
│   └── Interface: "HellaCacheInterface"
└── Accelerator (Consumer)
    └── Interface: "UserMemoryInterface"
```

**Interface Pattern**:
```cpp
// Circuit Level
auto* hellaCacheInterface = circuit.addInterface("HellaCacheInterface");
hellaCacheInterface->addMethod("cmd_to_bus", {{"cmd", hellaCmdType}}, {});

auto* userMemInterface = circuit.addInterface("UserMemoryInterface");
userMemInterface->addMethod("cmd_from_user", {{"cmd", userCmdType}}, {});
userMemInterface->addValue("get_resp", {}, {userRespType});

// MemoryAdapter
auto* hellaCacheDecl = memAdapter->addInterfaceDecl("HellaCacheInterface");
auto* userMemDecl = memAdapter->addInterfaceDecl("UserMemoryInterface");

// Method implementations with proper guards
auto* respToUser = memAdapter->addMethod("resp_to_user", {}, {userRespType});
respToUser->guard(canCollectGuard);  // Only when slot ready + earliest
respToUser->body(responseCollectionBody);
```

### Data Flow Pattern
```
Accelerator → User Memory Cmd → MemoryAdapter.cmd_from_user() →
HellaCache Cmd Buffer → HellaCache.cmd_to_bus() → Memory System →
HellaCache Response → MemoryAdapter.resp_from_bus() → Response Buffer →
User Memory Response → Accelerator
```

## Critical Design Rules

### 1. **Interface Separation**
- **RoCC**: Command interface (value) + Response interface (method)
- **Memory**: Command interfaces (both directions) + Response value
- **Rule**: Never mix consumer/provider roles in same interface declaration

### 2. **FIFO Discipline**
- **RoCC**: Use `enq()`/`deq()` methods with `full()`/`empty()` guards
- **Memory**: Use internal registers, not FIFOs, for response buffering
- **Rule**: Always check FIFO status before operations

### 3. **Tag-Based Matching**
- **Memory Responses**: 8-bit tags for request-response pairing
- **Rule**: Tags must be preserved through entire transaction chain
- **Rule**: Response collection must respect ordering constraints

### 4. **Clock/Reset Synchronization**
- **All instances**: Must receive clock and reset signals
- **Rule**: Register state only updates on clock edge
- **Rule**: Reset must initialize all state elements

### 5. **Method Finalization**
- **All function-like ops**: Must call `finalize()` after guard/body definition
- **Rule**: Unfinalized methods cause FIRRTL generation failures

## Implementation Priorities

### High Priority (Fix Blocking Issues)
1. **FIFO callValue() bug** - Fix empty return vectors for regular Module instances
2. **Interface binding** - Complete the declaration → definition → binding chain
3. **Method finalization** - Ensure all methods call `finalize()`

### Medium Priority (Functional Completeness)
1. **Opcode dispatch logic** - Complete multi-opcode command routing
2. **Response buffering** - Implement 2-slot memory response buffer
3. **Tag matching** - Add address tag preservation and matching logic

### Low Priority (Optimization)
1. **Performance tuning** - Optimize FIFO depths and buffer sizes
2. **Error handling** - Add timeout and error recovery mechanisms
3. **Resource optimization** - Minimize register and FIFO usage

This design maintains the core functionality while properly leveraging CMT2's interface system for clean module separation and type-safe communication.