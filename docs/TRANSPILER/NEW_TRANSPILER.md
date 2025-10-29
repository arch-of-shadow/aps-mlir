# Primary Goal

* Modify the current transpiler implementation to better support **instruction matching**, align more closely with **hardware-oriented semantics**, while still **preserving software semantics**.

# Current Problem

* In the current handling of `burst`, we simply remove the burst and lift the corresponding static **ScratchPad** array into the function’s input parameters. I believe this approach is problematic because it erases relationships that originally depended on **`rs`** and **`rd`**.
* My instruction matching uses a **tree-structured** approach; without the `rs/rd` dependency information, there is no way to determine the operands of pseudo-instructions.

# Details of CADL → C Conversion

Converting CADL to C semantics only needs to account for **three additional categories of operations**:

## 1) `register`

* CADL has explicit register read/write semantics. In software, when we use `asm volatile`, we can reference the **value at the register’s mapped address**. The compiler backend will take care of mapping this to the correct physical register automatically.

## 2) `_mem`

* In hardware, `_mem[rs1_value]` reads memory. In software, all arrays reside in **main memory**, and `rs1_value` corresponds to a **pointer**. We therefore access data via the pointer.

## 3) `burst` / `ScratchPad`

* A `burst` copies a contiguous region from `_mem` into the **ScratchPad**. The starting address is usually **`rs_value`**, which is the pointer in software semantics.
* Provided that each `rs` read is **properly aligned**, we can eliminate the `burst` and **replace all subsequent ScratchPad accesses** with **pointer-based loads from this `rs` pointer**, adding the appropriate **offset** when reading.

# Example

**CADL:**

```cadl
// V3DDIST.VS - Vector 3D Distance Squared (Vector-Scalar mode)
// Computes squared Euclidean distance from VL points to a single reference point
// Formula: dist² = (x-ref_x)² + (y-ref_y)² + (z-ref_z)²
// NOTE: Using fixed-point arithmetic with u32 types

#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static points_x: [u32; 16];

#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static points_y: [u32; 16];

#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static points_z: [u32; 16];

// Output distance array - complete partitioning for parallel access
#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static dist_out: [u32; 16];

// V3DDIST.VS - Vector-Scalar mode
// Computes distances from VL points to a single reference point
#[opcode(7'b0101011)]
#[funct7(7'b0101001)]
rtype v3ddist_vs(rs1: u5, rs2: u5, rd: u5) {
  let addr: u32 = _irf[rs1];      // Base address for point set
  let ref_addr: u32 = _irf[rs2];  // Address of reference point
  let vl: u32 = 16;               // Vector length

  // Burst read point coordinates from memory (SOA layout)
  points_x[0 +: ] = _burst_read[addr +: 16];
  points_y[0 +: ] = _burst_read[addr + 64 +: 16];   // 16 words × 4 bytes
  points_z[0 +: ] = _burst_read[addr + 128 +: 16];  // 2 × 16 words × 4 bytes

  // Read reference point (scalar broadcast)
  let ref_x: u32 = _mem[ref_addr];
  let ref_y: u32 = _mem[ref_addr + 4];
  let ref_z: u32 = _mem[ref_addr + 8];

  // Compute distances to reference point with partial unrolling
  [[unroll(4)]]
  with i: u32 = (0, i_) do {
    let x: u32 = points_x[i];
    let y: u32 = points_y[i];
    let z: u32 = points_z[i];

    // Compute differences from reference (unsigned subtraction)
    let dx: u32 = x - ref_x;
    let dy: u32 = y - ref_y;
    let dz: u32 = z - ref_z;

    // Compute squared distance
    let dist_sq: u32 = dx * dx + dy * dy + dz * dz;

    dist_out[i] = dist_sq;

    let i_: u32 = i + 1;
  } while (i_ < vl);

  // Burst write results back to memory
  let out_addr: u32 = _irf[rd];
  _burst_write[out_addr +: 16] = dist_out[0 +: ];

  _irf[rd] = 0;
}
```

**Current (Incorrect) C Function:**

```c
#include <stdint.h>
#include <string.h>

uint8_t v3ddist_vs(uint32_t *dist_out, uint32_t *points_x, uint32_t *points_y, uint32_t *points_z, uint32_t *_mem, uint32_t rd_value, uint32_t rs1_value, uint32_t rs2_value) {
    uint8_t rd_result = 0;
    uint32_t addr = rs1_value;
    uint32_t ref_addr = rs2_value;
    uint32_t vl = 16;
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    // burst_read eliminated (arrays directly accessible)
    uint32_t ref_x = _mem[ref_addr];
    uint32_t ref_y = _mem[(ref_addr + 4)];
    uint32_t ref_z = _mem[(ref_addr + 8)];
    uint32_t i;
    for (i = 0; i < vl; ++i) {
        uint32_t x = points_x[i];
        uint32_t y = points_y[i];
        uint32_t z = points_z[i];
        uint32_t dx = (x - ref_x);
        uint32_t dy = (y - ref_y);
        uint32_t dz = (z - ref_z);
        uint32_t dist_sq = (((dx * dx) + (dy * dy)) + (dz * dz));
        dist_out[i] = dist_sq;
        uint32_t i_ = (i + 1);
    }
    uint32_t out_addr = rd_value;
    // burst_write eliminated (arrays directly accessible)
    rd_result = 0;
    return rd_result;
}
```

**Correct C Function:**

```c
// right function for instruction use

uint8_t v3ddist_vv(uint32_t *rs1, uint32_t *rs2, uint32_t *rd) {
    // rs1: points1 base
    // rs2: points2 base
    // rd: output distances base
    uint32_t vl = 16;

    uint32_t i;
    for (i = 0; i < vl; ++i) {
        uint32_t x1 = rs1[i];
        uint32_t y1 = rs1[i + vl];
        uint32_t z1 = rs1[i + 2 * vl];
        uint32_t x2 = rs2[i];
        uint32_t y2 = rs2[i + vl];
        uint32_t z2 = rs2[i + 2 * vl];
        uint32_t dx = (x1 - x2);
        uint32_t dy = (y1 - y2);
        uint32_t dz = (z1 - z2);
        uint32_t dist_sq = (((dx * dx) + (dy * dy)) + (dz * dz));
        rd[i] = dist_sq;
    }
    return 0;
}
```

# Additional Notes

* I have added two more examples: `examples/pcl/v3ddist_vv.c` and `examples/pcl/vcovmat3d.c`.

---

**Request:**

* Please review the above and update the transpiler logic in `cadl_frontend/transpile_to_c.py` accordingly, and revise `docs/TRANSPILER/HIGH_LEVEL_C_TRANSPILER_GUIDE.md` to reflect these decisions.
