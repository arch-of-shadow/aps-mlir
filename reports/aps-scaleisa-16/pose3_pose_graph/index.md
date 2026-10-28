# Pose3 graph solver (18 poses, 30 edges)

All MLIR files are byte-for-byte copies of the measured run artifacts. Hashes and source paths are in [manifest.json](manifest.json).

[Collected AU definitions](scaleisa-au.mlir) · [APS hardware](aps-hardware.mlir) · [APS planning](aps-planned.mlir) · [CPU CI lowering](cpu-ci.mlir) · [Selected program](selected-program.mlir)

| Selected AU / APS function | ScaleISA implementation input | Hector evaluated schedule |
|---|---|---|
| `au_generated_candidate_32__direct` | [input](au/au_generated_candidate_32__direct.input.mlir) | [evaluated](au/au_generated_candidate_32__direct.evaluated.mlir) |
| `au_generated_candidate_40__l0u_l1u` | [input](au/au_generated_candidate_40__l0u_l1u.input.mlir) | [evaluated](au/au_generated_candidate_40__l0u_l1u.evaluated.mlir) |
| `au_generated_candidate_43__l0u_l1u` | [input](au/au_generated_candidate_43__l0u_l1u.input.mlir) | [evaluated](au/au_generated_candidate_43__l0u_l1u.evaluated.mlir) |
| `au_generated_candidate_5__l4p` | [input](au/au_generated_candidate_5__l4p.input.mlir) | [evaluated](au/au_generated_candidate_5__l4p.evaluated.mlir) |

[Scheduled APS/TOR and realized memory map](design.scheduled-tor.mlir)
