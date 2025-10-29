# CMT2 Rules, Relationships, and Interface Patterns for TOR to CMT2 Translation

## Overview

This document summarizes the key CMT2 rules, relationships, and interface patterns needed to redesign the MLIR implementation for TOR to CMT2 translation, based on the analysis of CMT2 documentation and comparison between the current C++ implementation and Rust reference implementation.

## Core CMT2 Architecture Patterns

### 1. Module Hierarchy and Instantiation

**Rule**: CMT2 uses strict hierarchical module instantiation with proper parent-child relationships.

**Pattern**:
```cpp
// Top-level circuit
Circuit circuit("TopModule", context);

// Module creation
auto* module = circuit.addModule("ModuleName");
Clock clk = module->addClockArgument("clk");
Reset rst = module->addResetArgument("rst");

// Instance creation
auto* instance = module->addInstance("instance_name", targetModule, {clk, rst});
```

**Key Requirements**:
- All instances must be created within a parent module
- Clock and reset must be explicitly passed to instances
- Module names must be unique within the circuit
- Instances cannot be shared across different modules

### 2. Interface-Based Communication (Formerly Friend Mechanism)

**Rule**: Use INTERFACE mechanism for module-to-module communication instead of direct FRIEND relationships.

**Pattern**:
```cpp
// Step 1: Define interface at circuit level
auto* interface = circuit.addInterface("InterfaceName");
interface->addMethod("methodName", argTypes, retTypes);
interface->addValue("valueName", {}, retTypes);

// Step 2: Declare interface dependency in consumer module
auto* consumer = circuit.addModule("ConsumerModule");
auto* interfaceDecl = consumer->defineInterface("localName", "InterfaceName");

// Step 3: Implement interface methods in provider module
auto* provider = circuit.addModule("ProviderModule");
auto* method = provider->addMethod("methodName", argTypes, retTypes);
method->guard(guardFunction);
method->body(bodyFunction);
method->finalize();

// Step 4: Bind interface to implementation at parent level
auto* parent = circuit.addModule("ParentModule");
auto* interfaceDef = parent->defineInterfaceDef("bindingName", "InterfaceName");
interfaceDef->bind("providerInstance", "providerMethod", "interfaceMethod");
interfaceDef->finalize();

// Step 5: Instantiate with interface binding
auto* providerInst = parent->addInstance("provider", provider, {clk, rst});
auto* consumerInst = parent->addInstance("consumer", consumer,
    {clk, rst}, {{"bindingName", "localName"}});
```

**Key Benefits**:
- Decouples interface definition from implementation
- Enables modular design and testing
- Supports interface method forwarding
- Type-safe method binding

### 3. Function-Like Operations (Methods, Rules, Values)

**Rule**: All executable logic must be encapsulated in function-like operations with proper guard and body regions.

**Pattern**:
```cpp
// Method with arguments and return values
auto* method = module->addMethod("methodName",
    {{"arg1", type1}, {"arg2", type2}}, {retType});
method->guard([](mlir::OpBuilder &b, auto args) {
    // Guard logic - when is this method ready?
    auto trueVal = UInt::constant(1, 1, b, loc);
    b.create<cmt2::ReturnOp>(loc, trueVal.getValue());
});
method->body([](mlir::OpBuilder &b, auto args) {
    // Body logic - what does this method do?
    Signal result = Signal(args[0], &b, loc) + Signal(args[1], &b, loc);
    b.create<cmt2::ReturnOp>(loc, result.getValue());
});
method->finalize();

// Value (combinational logic)
auto* value = module->addValue("valueName", {retType});
value->guard([](mlir::OpBuilder &b) {
    // Always ready
    auto trueVal = UInt::constant(1, 1, b, loc);
    b.create<cmt2::ReturnOp>(loc, trueVal.getValue());
});
value->body([](mlir::OpBuilder &b) {
    // Combinational computation
    auto result = b.create<firrtl::AddPrimOp>(loc, operand1, operand2);
    b.create<cmt2::ReturnOp>(loc, result.getResult());
});
value->finalize();

// Rule (stateful operation with guard)
auto* rule = module->addRule("ruleName");
rule->guard([](mlir::OpBuilder &b) {
    // Rule condition
    auto condition = b.create<firrtl::AndPrimOp>(loc, cond1, cond2);
    b.create<cmt2::ReturnOp>(loc, condition);
});
rule->body([](mlir::OpBuilder &b) {
    // Rule actions
    instance->callMethod("methodName", {arg1, arg2}, b);
    b.create<cmt2::ReturnOp>(loc, mlir::ValueRange{});
});
rule->finalize();
```

**Key Requirements**:
- Every function-like operation must have both guard and body regions
- Guard regions determine when the operation can execute
- Body regions contain the actual logic
- Must call `finalize()` after defining guard and body
- Methods can have arguments and return values
- Rules and values use different calling conventions

### 4. Signal Abstraction and Type System

**Rule**: Use Signal class for hardware operations instead of manual FIRRTL op creation.

**Pattern**:
```cpp
// Signal creation and arithmetic
Signal operandA(valueA, &builder, loc);
Signal operandB(valueB, &builder, loc);
Signal sum = operandA + operandB;  // firrtl.add
Signal result = sum.bits(31, 0);   // firrtl.bits

// Bundle creation and access
auto packet = BundleBuilder(&context)
    .addUInt("addr", 32)
    .addUInt("data", 64)
    .addVector("tags", firrtl::UIntType::get(&context, 8), 4)
    .build(builder, loc);

Bundle bundle = AsBundle(packet, &builder, loc);
Signal addr = bundle["addr"];
Signal data = bundle["data"];

// Vector operations
auto elemType = firrtl::UIntType::get(&context, 32);
FVector vector(elemType, 4, builder, loc);
Signal element = vector[0];

// Conditional execution
auto result = If(condition,
    [&](mlir::OpBuilder &b) -> Signal { return thenValue; },
    [&](mlir::OpBuilder &b) -> Signal { return elseValue; },
    builder, loc);
```

**Key Benefits**:
- Operator overloading makes hardware logic readable
- Automatic type inference and width management
- Error prevention through compile-time checks
- Support for complex data structures (bundles, vectors)

### 5. External Module Integration

**Rule**: External FIRRTL modules must be properly registered and configured with correct bindings.

**Pattern**:
```cpp
// Load module library
auto &library = ModuleLibrary::getInstance();
library.loadManifest("path/to/manifest.yaml");

// Create external module with parameters
llvm::StringMap<int64_t> params;
params["width"] = 32;
params["depth"] = 1024;
auto* extMod = circuit.addExternalModule("ModuleName", "FIRRTLModuleName", params);

// Configure external module bindings
extMod->bindClock("clk", "clock")
      .bindReset("rst", "reset")
      .bindValue("read", "read_ready", {"read_data"})
      .bindMethod("write", "write_enable", "write_ready", {"write_data"}, {})
      .addConflict("write", "write")
      .addConflictFree("read", "read");
```

**Key Requirements**:
- External modules must be registered in the module library
- Parameters must match the external module's expectations
- Clock/reset bindings must use correct port names
- Conflict relationships must match the external module's behavior

## Specific Translation Patterns

### 1. FIFO and Buffer Implementation

**Rust Pattern** (using stl::fifo1_push):
```rust
let rocc_cmd_queue = instance!(stl::fifo1_push(&rocc_cmd_t));
let cmd = output!(format!("rocc_cmd_user_{}", opcode), rocc_cmd_t.clone());

let cmd_to_user = named_method! {
  format!("cmd_to_user_{}", opcode);
  () -> (cmd) {
      ret!(rocc_cmd_queue.deq());
  }
};
```

**CMT2 Translation**:
```cpp
// Create FIFO1Push module
auto* fifo64Module = createFIFO1PushModule(circuit, 64);

// Create instance
auto* cmdQueue = module->addInstance("cmd_queue", fifo64Module, {clk, rst});

// Create method that uses FIFO
auto* cmdToUser = module->addMethod("cmd_to_user", {}, {cmdType});
cmdToUser->guard([](mlir::OpBuilder &b) {
    auto fullVal = cmdQueue->callValue("full", b);
    auto notFull = b.create<firrtl::NotPrimOp>(loc, fullVal[0]);
    b.create<cmt2::ReturnOp>(loc, notFull);
});
cmdToUser->body([&](mlir::OpBuilder &b) {
    auto data = cmdQueue->callValue("deq", b);
    b.create<cmt2::ReturnOp>(loc, data[0]);
});
cmdToUser->finalize();
```

### 2. Register and Wire Implementation

**Rust Pattern** (using stl::reg with bypass):
```rust
let rs1 = instance!("__backend_rs1".to_string(); mk_bypass_register(&Type::UInt(32)));

impl Backend for RoCCBackend {
  fn read_gpr(&self, reg_id: u32) -> Var {
    if reg_id == 1 {
      self.rs[0].read()
    } else {
      panic!("Invalid register id");
    }
  }
}
```

**CMT2 Translation**:
```cpp
// Create bypass register module
auto* bypassRegMod = generateBypassRegisterModule(circuit, 32);

// Create instance
auto* rs1 = mainModule->addInstance("rs1", bypassRegMod, {clk, rst});

// Read method
auto* readGpr = mainModule->addMethod("read_gpr",
    {{"reg_id", uint5Type}}, {uint32Type});
readGpr->body([&](mlir::OpBuilder &b, auto args) {
    auto regId = args[0];
    auto isRs1 = b.create<firrtl::EQPrimOp>(loc, regId,
        b.create<firrtl::ConstantOp>(loc, uint5Type, 1));

    auto rs1Val = rs1->callValue("read", b)[0];
    auto defaultVal = b.create<firrtl::ConstantOp>(loc, uint32Type, 0);

    auto result = b.create<firrtl::MuxPrimOp>(loc, isRs1, rs1Val, defaultVal);
    b.create<cmt2::ReturnOp>(loc, result);
});
readGpr->finalize();
```

### 3. Interface-Based Command Dispatch

**Rust Pattern** (using friend mechanism):
```rust
let rocc_master_virtual = itfc!("rocc".to_string(); mk_rocc_vmaster());
let rocc_master_friend = friend!(&rocc_master_virtual);
let roccitfc = instance!("roccitfc".to_string(); mk_rocc_adapter(&rocc_master_friend, opcode));
```

**CMT2 Translation**:
```cpp
// Define interfaces at circuit level
auto* roccCmdInterface = circuit.addInterface("RoCCCmdInterface");
roccCmdInterface->addValue("get_cmd", {}, {uint64Type});

// Create provider module
auto* roccMaster = circuit.addModule("RoCCMaster");
auto* roccMasterResp = roccMaster->addInterfaceDecl("RoCCRespInterface");

// Create consumer module with interface declaration
auto* roccAdapter = circuit.addModule("RoCCAdapter");
auto* roccCmdBus = roccAdapter->addInterfaceDef("RoCCCmdInterface");

// Bind interface at parent level
auto* mainModule = circuit.addModule("main");
auto* roccMasterInst = mainModule->addInstance("rocc_master", roccMaster, {clk, rst});
auto* roccAdapterInst = mainModule->addInstance("rocc_adapter", roccAdapter,
    {clk, rst}, {{"RoCCCmdBinding", "rocc_cmd_interface"}});
```

## Key Implementation Differences

### 1. Friend vs Interface Mechanism

**Rust (Friend)**: Direct module references with friend! macro
**CMT2 (Interface)**: Formal interface declarations and bindings

### 2. Module Instantiation

**Rust**: instance! macro with automatic parameter handling
**CMT2**: Explicit addInstance with clock/reset passing

### 3. Method Definition

**Rust**: method! macro with concise syntax
**CMT2**: addMethod with separate guard/body definition and finalize()

### 4. FIFO Operations

**Rust**: enq()/deq() methods on FIFO instances
**CMT2**: callMethod/callValue with proper full/empty checking

### 5. Scheduling and Precedence

**Rust**: schedule! macro for method ordering
**CMT2**: setPrecedence with explicit precedence pairs

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Create proper interface definitions for RoCC and memory interfaces
2. Implement module hierarchy with correct parent-child relationships
3. Set up external module registration and configuration

### Phase 2: Component Translation
1. Translate RoCC adapter to use interface mechanism
2. Translate memory adapter with proper FIFO integration
3. Implement scratchpad memory pool with burst access methods

### Phase 3: Logic Translation
1. Convert register files to use proper bypass register modules
2. Implement command dispatch logic with interface calls
3. Add memory operation translation with tag-based tracking

### Phase 4: Integration and Testing
1. Integrate all components with proper interface bindings
2. Add precedence relationships for correct scheduling
3. Verify correct FIRRTL generation and execution

## Critical Success Factors

1. **Interface Consistency**: All module-to-module communication must use interfaces
2. **Proper Hierarchical Structure**: Instances must be created in correct parent modules
3. **Method Finalization**: All function-like operations must call finalize()
4. **External Module Configuration**: Proper binding of external FIRRTL modules
5. **Type Safety**: Use Signal class and proper type conversions
6. **Clock/Reset Discipline**: Explicit clock/reset passing to all instances

This architecture provides a clean separation of concerns, proper modular design, and type-safe module communication that matches CMT2's design principles.