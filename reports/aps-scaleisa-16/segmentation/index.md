# Region growing (8×8, 2 iterations)

All MLIR files are byte-for-byte copies of the measured run artifacts. Hashes and source paths are in [manifest.json](manifest.json).

[Collected AU definitions](scaleisa-au.mlir) · [APS hardware](aps-hardware.mlir) · [APS planning](aps-planned.mlir) · [CPU CI lowering](cpu-ci.mlir) · [Selected program](selected-program.mlir)

| Selected AU / APS function | ScaleISA implementation input | Hector evaluated schedule |
|---|---|---|
| `au_generated_candidate_0__direct` | [input](au/au_generated_candidate_0__direct.input.mlir) | [evaluated](au/au_generated_candidate_0__direct.evaluated.mlir) |
| `au_generated_candidate_10__direct` | [input](au/au_generated_candidate_10__direct.input.mlir) | [evaluated](au/au_generated_candidate_10__direct.evaluated.mlir) |
| `au_generated_candidate_21__l0u_l1u` | [input](au/au_generated_candidate_21__l0u_l1u.input.mlir) | [evaluated](au/au_generated_candidate_21__l0u_l1u.evaluated.mlir) |
| `au_generated_candidate_35__l0u_l1u_l2u` | [input](au/au_generated_candidate_35__l0u_l1u_l2u.input.mlir) | [evaluated](au/au_generated_candidate_35__l0u_l1u_l2u.evaluated.mlir) |

[Scheduled APS/TOR and realized memory map](design.scheduled-tor.mlir)
