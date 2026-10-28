# Pose2 graph solver (24 poses, 40 edges)

All MLIR files are byte-for-byte copies of the measured run artifacts. Hashes and source paths are in [manifest.json](manifest.json).

[Collected AU definitions](scaleisa-au.mlir) · [APS hardware](aps-hardware.mlir) · [APS planning](aps-planned.mlir) · [CPU CI lowering](cpu-ci.mlir) · [Selected program](selected-program.mlir)

| Selected AU / APS function | ScaleISA implementation input | Hector evaluated schedule |
|---|---|---|
| `au_generated_candidate_21__direct` | [input](au/au_generated_candidate_21__direct.input.mlir) | [evaluated](au/au_generated_candidate_21__direct.evaluated.mlir) |
| `au_generated_candidate_31__l1p` | [input](au/au_generated_candidate_31__l1p.input.mlir) | [evaluated](au/au_generated_candidate_31__l1p.evaluated.mlir) |
| `au_generated_candidate_36__l0u_l1u` | [input](au/au_generated_candidate_36__l0u_l1u.input.mlir) | [evaluated](au/au_generated_candidate_36__l0u_l1u.evaluated.mlir) |
| `au_generated_candidate_38__l0u_l1u` | [input](au/au_generated_candidate_38__l0u_l1u.input.mlir) | [evaluated](au/au_generated_candidate_38__l0u_l1u.evaluated.mlir) |
| `au_generated_candidate_5__l0p_l1u` | [input](au/au_generated_candidate_5__l0p_l1u.input.mlir) | [evaluated](au/au_generated_candidate_5__l0p_l1u.evaluated.mlir) |

[Scheduled APS/TOR and realized memory map](design.scheduled-tor.mlir)
