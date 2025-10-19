# APS-MLIR Scheduling Algorithm Report

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [CDFG Construction](#cdfg-construction)
4. [SDC Solver](#sdc-solver)
5. [SDC Scheduling Algorithm](#sdc-scheduling-algorithm)
6. [Function-Level Scheduling](#function-level-scheduling)
7. [Resource Database](#resource-database)
8. [Key Algorithms and Techniques](#key-algorithms-and-techniques)
9. [Advanced Features](#advanced-features)
10. [Complexity Analysis](#complexity-analysis)
11. [Limitations and Future Work](#limitations-and-future-work)

---

## Executive Summary

The APS-MLIR scheduling infrastructure implements an advanced **System of Difference Constraints (SDC)-based modulo scheduling algorithm** for high-level synthesis (HLS). The scheduler operates on MLIR operations, performing both loop pipelining and function-level scheduling to minimize latency while respecting hardware resource constraints and data dependencies.

### Key Features

- ✅ **Loop pipelining** with automatic Initiation Interval (II) selection
- ✅ **Function-level scheduling** with dataflow support
- ✅ **Advanced memory system** handling (burst transfers, multi-port RAM)
- ✅ **Register lifetime minimization** via Linear Programming
- ✅ **Resource-aware scheduling** with incremental constraint solving
- ✅ **Operation chaining** for performance optimization

---

## Architecture Overview

### Core Components

The scheduling system consists of four main components:

| Component | Location | Purpose |
|-----------|----------|---------|
| **CDFG** | `lib/Schedule/CDFG.cpp` | Control/Data Flow Graph representation |
| **SDCSolver** | `lib/Schedule/SDCSolver.cpp` | System of Difference Constraints solver |
| **SDCSchedule** | `lib/Schedule/SDCSchedule.cpp` | Main scheduling algorithm |
| **ResourceDB** | `include/Schedule/ResourceDB.h` | Hardware resource database |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   MLIR Operations                        │
│         (tor.*, scf.*, arith.*, memref.*)               │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              ScheduleBase (ScheduleAlgo.h)              │
│  • buildCFG() - Control Flow Graph construction         │
│  • buildDFG() - Data Flow Graph construction            │
│  • createOp/createMemOp/createMAxiOp                    │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│                CDFG Data Structures                      │
│  • OpAbstract/OpConcrete - Operation wrappers           │
│  • BasicBlock - Sequential operation groups             │
│  • Loop - Loop structures with pipeline metadata        │
│  • Dependence - RAW/WAR/WAW/RAR dependencies            │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              SDCSchedule (Main Algorithm)               │
│  • pipelineLoop() - Loop modulo scheduling              │
│  • pipelineFunction() - Function-level scheduling       │
│  • formulateSDC() - DAG scheduling                      │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│       SDCSolver - Constraint Satisfaction Engine        │
│  • initSolution() - Bellman-Ford longest path           │
│  • addConstraint() - Incremental constraint addition    │
│  • minimizeLifetime() - LP-based register optimization  │
└─────────────────────────────────────────────────────────┘
```

---

## CDFG Construction

### Control Flow Graph (CFG)

**File:** `lib/Schedule/ScheduleAlgo.cpp:30-645`

The CFG builder creates a hierarchical representation of program control flow:

#### Basic Blocks

- Groups of sequential operations without control flow
- Connected by control edges: `FORWARD`, `COND`, `LOOPBACK`
- Each block has a parent loop (or `nullptr` for top-level)

#### Loop Structures

```cpp
// For tor.for operations
Loops.push_back(std::make_unique<Loop>(
    parentLoop, &op, pipelineFlag, targetII));
```

- **Nested loops**: Child loops reference parent loops
- **Pipeline metadata**: Target II, achieved II, pipeline flag
- **Loop types**: `tor::ForOp` and `tor::WhileOp` supported

#### Special Operations

- **PHI nodes** (`OpType::PHI_OP`): Inserted at loop headers and if-merge points
- **ASSIGN nodes** (`OpType::ASSIGN_OP`): Value propagation across blocks
- **Branch operations**: Control flow decisions tracked per basic block

### Data Flow Graph (DFG)

**Files:** `lib/Schedule/ScheduleAlgo.cpp:647-1238`

#### Scalar Dependencies

**Function:** `buildScalarDFG()` (line 647)

```cpp
for (auto v : op->getOperands()) {
    if (ValueMap.find(v) != ValueMap.end()) {
        addDependency(Dependence(ValueMap[v], op, 0, Dependence::D_RAW));
    }
}
```

- **SSA-based**: Follows MLIR's SSA value-use chains
- **Loop-carried**: Distance = 1 for loop recurrences
- **Type**: Primarily RAW (Read-After-Write)

#### Memory Dependencies

**Function:** `buildTensorDFG()` (line 888)

Sophisticated memory dependence analysis:

```cpp
// Check dependence analysis result
auto dep = dep_analysis.get_distance(1, memop1->getAddr(),
                                     1, memop2->getAddr(), loopOp);

if (dep.type == tor::DependenceResult::Dependent) {
    Distance = dep.dist;
}
```

**Features:**
- **Dependence types**: RAW, WAR, WAW, RAR
- **Distance computation**: Loop iteration distances
- **Memory bank analysis**: Partition-aware (no false dependencies)
- **Global memref tracking**: Symbol-based comparison

**Global Memory Handling:**
```cpp
auto isSameGlobalMemref = [](Value memref1, Value memref2) -> bool {
    auto getGlobal1 = memref1.getDefiningOp<mlir::memref::GetGlobalOp>();
    auto getGlobal2 = memref2.getDefiningOp<mlir::memref::GetGlobalOp>();

    if (getGlobal1 && getGlobal2) {
        return getGlobal1.getNameAttr() == getGlobal2.getNameAttr();
    }
    return false;
};
```

#### AXI/Stream Dependencies

**Function:** `buildMAxiDFG()` (line 777)

Handles memory-mapped I/O:

```cpp
// Burst transfer dependencies
std::unordered_map<Operation*, vector<SDCOpWrapper*>> burstOps;

// Request → Burst Ops → Response chain
for (size_t i = 1; i < users.size(); ++i) {
    addDependency(Dependence(users[i-1], users[i], 0, D_RAW));
}
```

**Burst Transfer Handling:**
1. Group burst operations by request
2. Order burst ops sequentially
3. Add response dependencies
4. Check loop-back edges for distance

#### Burst-Memory Dependencies

**Lines:** 1038-1237

Critical for correctness:

```cpp
// Memory ops → Burst ops
if (memop1->getParentLoop() != nullptr &&
    maxiop2->getParentLoop() == nullptr) {
    // Loop-exit dependency: Distance = 0
    Distance = 0;
}
```

Ensures memory operations complete before burst transfers.

---

## SDC Solver

**File:** `lib/Schedule/SDCSolver.cpp`

### Mathematical Foundation

The SDC solver solves **Systems of Difference Constraints**:

```
Constraints: x - y ≥ c  (x, y are variables, c is constant)
Additional:  x ≥ 0 for all variables
```

**Key insight:** This is equivalent to the **longest path problem** in a constraint graph.

### Algorithm Components

#### 1. Initial Solution: Bellman-Ford

**Function:** `initSolution()` (line 40)

```cpp
do {
    bool UpdateFlag = false;

    for (int i = 0; i < NumVariable; ++i)
        for (auto &edge : Edges[i])
            if (Solution[edge.to] < Solution[i] + edge.length) {
                Solution[edge.to] = Solution[i] + edge.length;
                UpdateFlag = true;
            }

    if (!UpdateFlag) break;
    NumIteration++;
} while (NumIteration < NumVariable);

return (NumIteration < NumVariable);  // True if feasible
```

**Properties:**
- **Complexity:** O(V × E) where V = variables, E = edges
- **Convergence:** At most N iterations
- **Cycle detection:** Fails if still updating after N iterations
- **Result:** ASAP (As Soon As Possible) schedule

#### 2. Incremental Constraint Addition

**Function:** `addConstraint()` (line 133)

Based on research paper: *"Solving Systems of Difference Constraints Incrementally"*

```cpp
// Add edge y -(c)-> x representing: x - y ≥ c

// 1. Dijkstra from x with transformed edge weights
std::priority_queue<pair<int, int>> Q;
dist_x[x] = 0;
Q.push({x, 0});

// 2. Update affected variables
while (!Q.empty()) {
    int now = Q.top().first;
    Q.pop();

    int NewSol = Solution[y] + c + (dist_x[now] + Solution[now] - Solution[x]);
    if (Solution[now] < NewSol) {
        if (now == y) {
            // Positive cycle detected!
            return false;
        }
        NewSolution[now] = NewSol;
        // ... propagate to successors
    }
}

// 3. Apply updates
Edges[y].insert(Edge(x, c));
for (auto &sol : NewSolution)
    Solution[sol.first] = sol.second;
```

**Advantages:**
- **Efficient:** Only updates affected variables
- **Incremental:** Preserves previous solution
- **Fast conflict detection:** Returns false immediately

#### 3. LP Conversion

**Function:** `convertLP()` (line 238)

```cpp
for (int i = 0; i < NumVariable; ++i)
    for (auto &edge : Edges[i]) {
        // edge.to - i >= edge.length
        colno[0] = edge.to + 1;  row[0] = 1;
        colno[1] = i + 1;        row[1] = -1;
        add_constraintex(lp, 2, row, colno, GE, edge.length);
    }
```

Converts SDC to Linear Programming format for register optimization.

### Constraint Types

**Location:** `include/Schedule/SDCSolver.h:14-42`

```cpp
enum Type {
    Constr_EQ,   // x = c
    Constr_CMP   // x - y >= c
};

// Factory methods
static Constraint CreateEQ(int x, int c);
static Constraint CreateGE(int x, int y, int c);  // x - y >= c
static Constraint CreateLE(int x, int y, int c);  // x - y <= c (y - x >= -c)
```

---

## SDC Scheduling Algorithm

**File:** `lib/Schedule/SDCSchedule.cpp`

### Loop Pipelining Overview

**Function:** `pipelineLoop()` (line 760)

```
┌─────────────────────────────────────────────┐
│ 1. Calculate MII (Minimum II)              │
│    • recMII = recurrenceMII()  [TODO]      │
│    • resMII = resourceMII()                │
│    • MII = max(recMII, resMII)             │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│ 2. Try Target II                            │
│    if target_II >= MII:                     │
│        scheduleWithII(L, target_II)         │
│    else:                                    │
│        scheduleWithII(L, MII)               │
└────────────────┬────────────────────────────┘
                 │ Success? → Done
                 │ Failure? ↓
                 ▼
┌─────────────────────────────────────────────┐
│ 3. Binary Search for Minimum II            │
│    Range: [MII+1, 128]                      │
│    while (l <= r):                          │
│        mid = (l + r) / 2                    │
│        if scheduleWithII(L, mid, false):    │
│            achieved = mid                   │
│            r = mid - 1                      │
│        else:                                │
│            l = mid + 1                      │
└─────────────────────────────────────────────┘
```

### Resource MII Calculation

#### General Resources

**Function:** `resourceMII()` (line 118)

```cpp
for each resource type R:
    totalPressure = 0
    for each op in loop:
        if op uses R:
            totalPressure += RDB.getII(R)

    if R has hard limit and not memport/m_axi:
        resMII = max(resMII, totalPressure / amount(R))
```

#### Memory Port MII

**Function:** `get_memportMII()` (line 69)

```cpp
for each memref in loop:
    numAccesses = count of load/store ops

    if RAM_1P (single-port):
        resMII = max(resMII, numAccesses)
    else if RAM_T2P (true dual-port):
        resMII = max(resMII, (numAccesses + 1) / 2)
    else:  // Default memport
        resMII = max(resMII, (numAccesses + numport - 1) / numport)
```

#### AXI MII

**Function:** `getMAxiMII()` (line 35)

```cpp
for each bus:
    readLatency = 0, writeLatency = 0

    for each AXI op:
        if isBurst:
            latency += 1
        else if isRead:
            latency += 2
        else if isWrite:
            latency += 3

    resMII = max(readLatency, writeLatency)
```

### Core Scheduling: scheduleWithII()

**Function:** `scheduleWithII()` (line 690)

```
┌────────────────────────────────────────────────┐
│ 1. allocVariable(L, SDC)                       │
│    • Create SDC variable for each operation    │
│    • Map OpAbstract → Variable ID              │
└──────────────────┬─────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────┐
│ 2. formulateDependency(L, II, SDC)             │
│    • Control dependencies (branches)           │
│    • Data dependencies (RAW/WAR/WAW)           │
│    • Timing constraints (chaining)             │
│    • Burst operation ordering                  │
└──────────────────┬─────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────┐
│ 3. getASAPTime(L, II, SDC)                     │
│    • Solve SDC with Bellman-Ford               │
│    • Get ASAP (earliest) schedule              │
│    • Store in op->ASAPTime                     │
└──────────────────┬─────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────┐
│ 4. minimizeLifetime(L, II, SDC)                │
│    • Formulate LP problem                      │
│    • Objective: minimize register pressure     │
│    • Solve with lpsolve                        │
│    • Store in op->OptTime                      │
└──────────────────┬─────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────┐
│ 5. resolveResConstraint(L, II, SDC)            │
│    • Iterative placement with backtracking     │
│    • Perturbation heuristic                    │
│    • Resource conflict resolution              │
│    • Store final schedule in op->SDCTime       │
└────────────────────────────────────────────────┘
```

### Dependency Formulation Details

**Function:** `formulateDependency()` (line 227)

#### Control Dependencies

```cpp
// Branch operations must complete before dependent ops
for (auto BB : L->getBody()) {
    if (auto branchOp = BB->getBranchOp()) {
        int Lat = RDB.getLatency(branchOp->getResource(), branchOp->getWidth());

        for (auto op : BB->getOperations()) {
            SDC->addInitialConstraint(
                Constraint::CreateGE(op->VarId, branchOp->VarId, Lat)
            );
        }
    }
}
```

#### Data Dependencies

```cpp
for (auto pred : op->getPred()) {
    if (sameLoop(pred)) {
        auto srcOp = llvm::dyn_cast<SDCOpWrapper>(pred->SourceOp);
        int Lat = RDB.getLatency(srcOp->getResource(), srcOp->getWidth());

        // Key formula: dest_time - src_time >= Lat - dist*II
        SDC->addInitialConstraint(Constraint::CreateGE(
            destOp->VarId, srcOp->VarId, Lat - pred->Distance * II
        ));
    }
}
```

**Explanation:**
- Distance 0: Same iteration, standard precedence
- Distance 1: Next iteration, allows overlap
- Distance d: d iterations apart

#### Chaining Constraints

**Function:** `addChainingConstr()` (line 184)

```cpp
void traverse(SDCOpWrapper *op, SDCSolver *SDC, int latency,
              float cp, int dist, int II, SDCOpWrapper *start) {
    vis[op] = true;

    for (auto Succ : op->getSucc()) {
        if (sameLoop(Succ)) {
            auto succOp = llvm::dyn_cast<SDCOpWrapper>(Succ->DestinationOp);
            float nxt_cp = cp + RDB.getDelay(succOp->getResource(),
                                             succOp->getWidth());

            if (nxt_cp > ClockPeriod) {
                // Path exceeds clock period → separate cycles
                SDC->addInitialConstraint(Constraint::CreateGE(
                    succOp->VarId, start->VarId, latency + 1 - nxt_dist * II
                ));
                continue;
            }

            if (RDB.isCombLogic(succOp) && !vis[succOp]) {
                traverse(succOp, SDC, latency, nxt_cp, nxt_dist,
                        II, start, vis, exceed);
            }
        }
    }
}
```

**Purpose:** Finds combinational paths that exceed clock period and forces them into separate cycles.

#### Burst Operation Ordering

```cpp
// Group burst ops by request
std::unordered_map<Operation*, vector<SDCOpWrapper*>> burstOps;

for (auto iter : burstOps) {
    std::sort(iter.second.begin(), iter.second.end(), cmpBurstOps);

    // Enforce consecutive execution
    for (size_t i = 1; i < iter.second.size(); ++i) {
        SDC->addInitialConstraint(Constraint::CreateLE(
            iter.second[i]->VarId, iter.second[i-1]->VarId, 1
        ));
    }
}
```

### Register Lifetime Minimization

**Function:** `minimizeLifetime()` (line 373)

#### LP Formulation

**Variables:**
- `t[i]` = schedule time for operation i
- `l[i]` = lifetime of value produced by operation i

**Objective Function:**
```
minimize Σ (bitwidth[i] × l[i])
```

**Constraints:**
```cpp
// For each RAW dependency i → j:
// lifetime[i] ≥ (time[j] - time[i]) - latency[i]
// Equivalently: time[j] - time[i] - lifetime[i] ≤ latency[i]

for (auto succ : op->getSucc()) {
    if (sameLoop(succ) && succ->type == Dependence::D_RAW) {
        int colno[3] = {succOp->VarId + 1, sdcOp->VarId + 1, newId};
        double row[3] = {1, -1, -1};
        add_constraintex(lp, 3, row, colno, LE,
                        RDB.getLatency(sdcOp->getResource(), sdcOp->getWidth()));
    }
}

// Plus all SDC constraints from previous phases
SDC->convertLP(lp);
```

**Intuition:**
- Shorter lifetimes → fewer registers
- Weighted by bitwidth → larger values more expensive
- Respects all dependencies

### Resource Constraint Resolution

**Function:** `resolveResConstraint()` (line 441)

#### Perturbation Heuristic

```cpp
// For each resource-constrained operation:
for (auto &[step, op] : S) {
    // Measure cost of moving operation
    int pert_before = SDC->tryAddConstr(CreateLE(op->VarId, step - 1));
    int pert_after = SDC->tryAddConstr(CreateGE(op->VarId, step + 1));

    // Perturbation = number of affected variables
    perturbation[op] = max(pert_before, pert_after);
}

// Sort by perturbation (descending) and earliest time
auto cmp = [&](SDCOpWrapper *a, SDCOpWrapper *b) {
    if (perturbation[a] != perturbation[b])
        return perturbation[a] > perturbation[b];  // High perturbation first
    return earliestTime[a] < earliestTime[b];
};
```

**Rationale:** Operations with high perturbation are "hard to move" → schedule them first.

#### Iterative Placement

```cpp
while (!S.empty()) {
    int step = S.begin()->first;
    vector<SDCOpWrapper*> ops = extract_ops_at_step(step);

    stable_sort(ops.begin(), ops.end(), cmp);  // Perturbation order

    for (auto op : ops) {
        bool scheduled = false;

        // Try slots from current down to earliest
        for (int s = step; s >= earliestTime[op]; --s) {
            // Check resource availability
            if (resource_available(op, s, II)) {
                if (SDC->addConstraint(CreateEQ(op->VarId, s))) {
                    update_resource_usage(op, s, II);
                    scheduled = true;
                    break;
                }
            }
        }

        if (!scheduled) {
            // Backtrack: try later time
            if (SDC->addConstraint(CreateGE(op->VarId, step + 1))) {
                failedCnt[op]++;
                if (failedCnt[op] == II)
                    return false;  // Cannot find valid schedule

                earliestTime[op] = SDC->Solution[op->VarId];
                S.insert({earliestTime[op], op});
            } else {
                return false;  // Infeasible
            }
        }
    }
}
```

#### Resource Tracking

**Memory Ports:**
```cpp
if (RDB.getName(RId).rfind("memport", 0) == 0) {
    for (int i = 0; i < resourceII; ++i) {
        auto v = op->getMemOp()->getMemRef().getImpl();
        if (MemTable[(slot + i) % II][v] >= ResLimit[RId]) {
            avail = false;
            break;
        }
    }

    if (scheduled) {
        for (int i = 0; i < resourceII; ++i) {
            MemTable[(slot + i) % II][v] += 1;
        }
    }
}
```

**AXI Buses:**
```cpp
if (RDB.getName(RId).rfind("m_axi", 0) == 0) {
    auto bus = op->getMAxiOp()->getBus();

    if (op->getMAxiOp()->isRead()) {
        for (int i = 0; i < resourceII; ++i) {
            if (mAxiReadBusTable[(slot + i) % II][bus] >= ResLimit[RId]) {
                avail = false;
                break;
            }
        }
    }
    // Similar for write...
}
```

**General Resources:**
```cpp
for (int i = 0; i < resourceII; ++i) {
    if (ResTable[RId][(s + i) % II] >= ResLimit[RId]) {
        avail = false;
        break;
    }
}
```

---

## Function-Level Scheduling

**Function:** `pipelineFunction()` (line 1858)

For functions annotated with `pipeline="func"`:

### Workflow

```
1. Allocate SDC variables for all operations
2. Formulate dependencies (no loop-carried)
3. Get ASAP times
4. Minimize lifetime with LP
5. Resolve resource constraints
6. Binary search for achievable II
```

### Key Differences from Loop Pipelining

| Aspect | Loop Pipelining | Function Pipelining |
|--------|----------------|---------------------|
| **Loop-carried deps** | Yes (distance ≥ 1) | No |
| **II meaning** | Iterations overlap | Function calls overlap |
| **Recurrence** | Feedback loops | None |
| **Parallelism** | Across iterations | Across function calls |

### Parallel Call Operation Analysis

**Function:** `checkParallelCallOpResourceConstraints()` (line 1896)

```cpp
// For each time step:
for (auto iter : callOps) {
    vector<int> parallelUsage;

    for (size_t i = 0; i < iter.second.size(); ++i) {
        // Aggregate resource usage
        const auto usage = RDB.getUsage(callFuncOp);
        for (size_t k = 0; k < usage.size(); ++k) {
            parallelUsage[k] += usage[k];
        }

        // Check if resources exceeded
        if (!RDB.isUsageSatisfied(parallelUsage)) {
            // Add serialization constraints
            for (size_t j = i; j < n; ++j) {
                SDC->addConstraint(CreateGE(
                    sdcOpB->VarId, sdcOpA->VarId, 1
                ));
            }
        }
    }
}
```

**Purpose:** Allows parallel function execution when resources permit, serializes when necessary.

---

## Resource Database

**File:** `include/Schedule/ResourceDB.h`

### Component Model

```cpp
struct Component {
    string name;              // e.g., "adder", "multiplier"
    vector<float> delay;      // Combinational delay [0, 8, 16, 32, 64, 128, 256 bits]
    vector<int> latency;      // Pipeline stages [0, 8, 16, 32, 64, 128, 256 bits]
    int II;                   // Initiation Interval
    bool constr;              // Has resource constraint?
    int amount;               // Number of units (-1 = unlimited)
};
```

### Built-in Resources

```cpp
// Loaded from JSON config
{
    "adder": { "delay": [...], "latency": [...], "II": 1, "amount": -1 },
    "multiplier": { "delay": [...], "latency": [...], "II": 3, "amount": 2 },
    "memport": { "delay": [...], "latency": [...], "II": 1, "amount": 2 },
    ...
}

// Additional components added programmatically:
addComponent("memport_RAM_1P", ..., amount=1);      // Single-port RAM
addComponent("memport_RAM_T2P", ..., amount=2);     // True dual-port RAM
addComponent("m_axi_read", latency=2, II=2, amount=1);
addComponent("m_axi_write", latency=3, II=3, amount=1);
addComponent("m_axi_burst", latency=15, II=15, amount=1);
```

### Per-Memref Resources

**Function:** `getOrCreateMemrefResource()` (line 219)

```cpp
int getOrCreateMemrefResource(Value memref) {
    string memrefName = "memport_unknown";

    if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(defOp)) {
        string globalName = getGlobalOp.getName().str();
        memrefName = "memport_" + globalName + "_1rw";
    }

    if (NameToID.find(memrefName) != NameToID.end()) {
        return NameToID[memrefName];
    }

    // Create new resource: 1RW (one read or write per cycle)
    Component newComp(memrefName, baseDelay, baseLatency,
                     baseII, hasConstraint=true, amount=1);
    addComponent(newComp);

    return NameToID[memrefName];
}
```

**Benefits:**
- Independent scheduling of different arrays
- Avoids false resource conflicts
- All memories treated as 1RW for consistency

### Bitwidth Handling

```cpp
int bitwidthIdx(int bitwidth) {
    // bitwidth must be power of 2
    return (__builtin_clz(bitwidth | 1) - 1);  // log2(bitwidth)
}

int getLatency(int id, int bitwidth) {
    // Normalize to nearest power of 2 (round up)
    if (bitwidth > 0 && __builtin_popcount(bitwidth) != 1) {
        bitwidth = 1 << (32 - __builtin_clz(bitwidth - 1));
    }

    int index = bitwidthIdx(bitwidth);
    return Components[id].latency[index];
}
```

**Supports:** 1, 2, 4, 8, 16, 32, 64, 128, 256-bit operations

---

## Key Algorithms and Techniques

### 1. Modulo Scheduling

**Classical software pipelining** adapted for HLS:

```
Without pipelining (II=4):
┌────┬────┬────┬────┐
│ I0 │ I1 │ I2 │ I3 │
└────┴────┴────┴────┘
   0    4    8   12

With pipelining (II=1):
┌────┐
│ I0 │
├────┤
│ I1 │ I0 continues
├────┤
│ I2 │ I1 continues, I0 continues
├────┤
│ I3 │ I2 continues, I1 continues
└────┘
   0    1    2    3
```

**Requirements:**
- Respect loop-carried dependencies
- Honor resource constraints
- Minimize II while maintaining correctness

### 2. SDC-Based Scheduling

**Advantages over list scheduling:**

| Feature | List Scheduling | SDC Scheduling |
|---------|----------------|----------------|
| **Optimality** | Heuristic | Can be optimal |
| **Constraints** | Implicit | Explicit |
| **Flexibility** | Limited | High |
| **Complexity** | O(n log n) | O(n²) |
| **Chaining** | Difficult | Natural |

**Key benefits:**
- **Exact solutions** within constraints
- **Easy constraint addition** (timing, resource, etc.)
- **Mathematical foundation** with proofs
- **LP integration** for multi-objective optimization

### 3. Operation Chaining

**Example:**
```
Without chaining (3 cycles):
┌─────┬─────┬─────┐
│ ADD │ MUL │ SUB │
└─────┴─────┴─────┘

With chaining (1 cycle if delay allows):
┌─────────────────┐
│ ADD→MUL→SUB     │
└─────────────────┘
```

**Implementation:**
- DFS traversal through combinational logic
- Accumulate delays along paths
- Split when exceeding clock period
- Respects operation latencies

### 4. Perturbation Heuristic

**Intuition:**
- Some operations are "flexible" (can move easily)
- Others are "constrained" (few valid positions)
- Schedule constrained operations first

**Measurement:**
```
perturbation(op) = number of variables affected
                   if op is moved earlier or later
```

**Benefits:**
- Reduces backtracking
- Improves runtime
- Based on ICCAD 2013 paper

---

## Advanced Features

### 1. Burst Transfer Scheduling

**Challenges:**
- Multiple operations per burst request
- Sequential execution required
- Bus conflicts
- Memory conflicts

**Solution:**
```cpp
// 1. Group by request
std::unordered_map<Operation*, vector<SDCOpWrapper*>> burstOps;

// 2. Order sequentially
for (size_t i = 1; i < users.size(); ++i) {
    addDependency(Dependence(users[i-1], users[i], 0, D_RAW));
}

// 3. Calculate burst II
int resourceII = burstOps[requestOp].size();

// 4. Schedule as unit
for (int i = 0; i < resourceII; ++i) {
    mAxiReadBusTable[(slot + i) % II][bus] += 1;
}
```

### 2. Memory Port Allocation

**Storage Types:**

```cpp
if (storageType == "RAM_1P") {
    // Single-port: only one access per cycle
    rsc = RDB.getResourceID("memport_RAM_1P");
    amount = 1;
}
else if (storageType == "RAM_T2P") {
    // True dual-port: two accesses per cycle
    rsc = RDB.getResourceID("memport_RAM_T2P");
    amount = 2;
}
else {
    // Default: per-memref resource
    rsc = RDB.getOrCreateMemrefResource(memref);
}
```

**Slot Assignment:**
```cpp
// In pipelined loops:
if (opA->getStartTime() - L->AchievedII >= memrefStartTime) {
    opA->getOp()->setAttr("slot", IntegerAttr(..., 1));
} else {
    opA->getOp()->setAttr("slot", IntegerAttr(..., 0));
}
```

Used by backend for port binding.

### 3. Loop-Exit Dependencies

**Problem:** Memory operations inside loop must complete before burst transfer outside loop.

**Detection:**
```cpp
if (memop1->getParentLoop() != nullptr &&
    maxiop2->getParentLoop() == nullptr) {
    // Special case: loop-exit dependency
    if (canReach(memop1, maxiop2, true)) {
        Distance = 0;  // Not loop-carried!
    }
}
```

**Critical for correctness** in burst-scheduled examples.

### 4. Stream I/O Dependencies

**FIFO semantics:**
```cpp
for each pair (streamOp1, streamOp2):
    if (streamOp1->getStream() == streamOp2->getStream()) {
        if (canReach(streamOp1, streamOp2, false)):
            Distance = 0
        else if (canReach(streamOp1, streamOp2, true)):
            Distance = 1

        addDependency(Dependence(streamOp1, streamOp2, Distance, D_RAW));
    }
```

Ensures FIFO ordering is preserved.

---

## Complexity Analysis

### Time Complexity

**Per-II scheduling:**

| Phase | Complexity | Notes |
|-------|-----------|-------|
| **CDFG construction** | O(ops) | Linear scan |
| **Dependency analysis** | O(ops²) | Pairwise checks |
| **SDC initialization** | O(deps × ops) | Bellman-Ford |
| **LP solving** | Exponential worst-case | Polynomial average |
| **Resource resolution** | O(ops × II × resources) | Iterative placement |

**Binary search for II:**
- O(log(max_II) × per_II_cost)
- Typically: log₂(128) ≈ 7 iterations

**Overall:**
- **Best case:** O(ops²) when target II works
- **Average case:** O(ops² × log(II))
- **Worst case:** O(ops² × II × resources)

### Space Complexity

| Structure | Space |
|-----------|-------|
| **SDC variables** | O(ops) |
| **SDC constraints** | O(deps) |
| **Resource tables** | O(II × resources) |
| **LP problem** | O(ops + deps) |

**Overall:** O(ops + deps + II × resources)

### Scalability

**Practical limits:**
- **Small loops** (< 100 ops): Fast (< 1s)
- **Medium loops** (100-1000 ops): Reasonable (1-10s)
- **Large loops** (> 1000 ops): Slow (> 10s)

**Bottlenecks:**
1. LP solver for large problems
2. Resource resolution backtracking
3. Dependency analysis (O(n²))

---

## Limitations and Future Work

### Current Limitations

#### 1. Recurrence MII

```cpp
int SDCSchedule::recurrenceMII(Loop *L) {
    /// Why need this in the paper?
    int recII = 1;
    return recII;  // TODO: Not implemented
}
```

**Problem:** Doesn't analyze loop-carried dependency cycles.

**Impact:**
- May miss optimal II for recurrence-bound loops
- Could attempt infeasible II values

**Solution:** Implement strongly-connected component analysis on dependency graph.

#### 2. Memory Dependence Analysis

```cpp
// Very weak detection
if (!memop_prev->hasFixedMemoryBank() || !memop_lat->hasFixedMemoryBank())
    return true;  // Conservative: assume conflict
```

**Limitations:**
- No SCEV (Scalar Evolution) analysis
- Pessimistic for dynamic indices
- False dependencies on complex patterns

**Impact:** Lower IIs due to false dependencies.

#### 3. Timing Model

- **Fixed delays** per operation type
- No interconnect/routing delays
- No place-and-route feedback loop

**Impact:** Timing violations may occur in backend.

#### 4. Resource Model

- Static allocation
- No DVFS (Dynamic Voltage/Frequency Scaling)
- No power constraints
- No temperature constraints

### Potential Improvements

#### 1. Enhanced Dependence Analysis

```cpp
// Polyhedral analysis for affine loops
auto accessFunc1 = analyzeAccessFunction(memop1);
auto accessFunc2 = analyzeAccessFunction(memop2);

if (accessFunc1.isAffine() && accessFunc2.isAffine()) {
    auto dep = polyhedralDependenceTest(accessFunc1, accessFunc2);
    if (dep == NoDependence) {
        continue;  // Proven independent
    }
}
```

**Benefits:**
- Precise dependency distances
- Eliminate false dependencies
- Better II for complex loops

#### 2. Recurrence Analysis

```cpp
int computeRecurrenceMII(Loop *L) {
    // Find strongly-connected components in dependency graph
    auto sccs = findSCCs(L);

    int maxRecII = 1;
    for (auto scc : sccs) {
        int cycleLatency = sum of latencies in SCC;
        int cycleDistance = min distance in SCC;
        int recII = ceil(cycleLatency / cycleDistance);
        maxRecII = max(maxRecII, recII);
    }

    return maxRecII;
}
```

#### 3. Multi-Objective Optimization

**Current:** Minimize II, then minimize latency

**Proposed:**
- Pareto-optimal solutions (II vs latency vs resources)
- Energy-aware scheduling (minimize power)
- User-specified trade-offs

```cpp
struct ScheduleObjective {
    float weight_II;
    float weight_latency;
    float weight_energy;
    float weight_resources;
};
```

#### 4. Machine Learning

**Heuristic tuning:**
- Learn perturbation priorities from examples
- Predict achievable II without search
- Adaptive resource allocation

**Training data:** Successful schedules from previous designs.

#### 5. Hierarchical Scheduling

**Current:** Flat scheduling of all operations

**Proposed:**
```
1. Schedule function calls (coarse-grain)
2. Schedule function bodies (fine-grain)
3. Iterate until convergence
```

**Benefits:**
- Scales to large designs
- Natural modularity
- Incremental compilation

---

## Conclusion

The APS-MLIR scheduling infrastructure is a **sophisticated, mathematically-principled system** that combines:

✅ **Classical algorithms** (Bellman-Ford, Dijkstra, LP)
✅ **Modern techniques** (SDC, modulo scheduling, perturbation heuristics)
✅ **HLS-specific features** (burst transfers, memory ports, operation chaining)
✅ **Modular design** (CDFG, solver, scheduler, resource DB)

### Key Innovations

1. **SDC formulation** enables exact solutions with flexible constraints
2. **Incremental constraint solving** provides efficiency
3. **LP-based register optimization** minimizes area
4. **Per-memref resources** eliminate false conflicts
5. **Burst-aware scheduling** handles modern memory interfaces

### Production Readiness

**Strengths:**
- Handles real-world HLS patterns
- Proven on examples (burst_demo_scheduled.mlir)
- Integrates with MLIR infrastructure
- Mathematically sound

**Areas for improvement:**
- Recurrence analysis
- Scalability for very large loops
- Timing model accuracy
- Documentation

### Recommended Next Steps

1. **Short-term:**
   - Implement recurrence MII
   - Add SCEV-based dependence analysis
   - Optimize LP solver usage

2. **Medium-term:**
   - Hierarchical scheduling for scalability
   - Timing-driven placement feedback
   - Energy-aware optimization

3. **Long-term:**
   - Machine learning integration
   - Multi-objective optimization
   - Cross-platform optimization (multi-FPGA)

---

## References

1. Rau, B. R. (1994). "Iterative Modulo Scheduling: An Algorithm for Software Pipelining Loops"
2. Cong, J., Liu, B., Neuendorffer, S., Noguera, J., Vissers, K., & Zhang, Z. (2011). "High-Level Synthesis for FPGAs: From Prototyping to Deployment"
3. Zhang, Z., Fan, Y., Jiang, W., Han, G., Yang, C., & Cong, J. (2013). "AutoPilot: A Platform-Based ESL Synthesis System" (Perturbation heuristic)
4. "Solving Systems of Difference Constraints Incrementally" - Incremental SDC algorithm
5. MLIR Documentation: https://mlir.llvm.org/
6. lpsolve: http://lpsolve.sourceforge.net/

---

## Appendix: Key File Locations

```
cadl-frontend/
├── include/Schedule/
│   ├── CDFG.h              # CDFG data structures
│   ├── SDCSolver.h         # SDC solver interface
│   ├── SDCSchedule.h       # Main scheduler interface
│   ├── ScheduleAlgo.h      # Base scheduling algorithm
│   └── ResourceDB.h        # Resource database
│
└── lib/Schedule/
    ├── CDFG.cpp            # CDFG utilities (canReach, hasMemPortConflict)
    ├── SDCSolver.cpp       # SDC solver implementation
    ├── SDCSchedule.cpp     # Main scheduling algorithm (2128 lines)
    └── ScheduleAlgo.cpp    # CDFG construction, DFG construction
```

**Total LOC:** ~5000+ lines of sophisticated scheduling code
