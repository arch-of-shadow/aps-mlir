# Planar SLAM (16 poses, 20 landmarks)

All MLIR files are byte-for-byte copies of the measured run artifacts. Hashes and source paths are in [manifest.json](manifest.json).

[Collected AU definitions](scaleisa-au.mlir) · [APS hardware](aps-hardware.mlir) · [APS planning](aps-planned.mlir) · [CPU CI lowering](cpu-ci.mlir) · [Selected program](selected-program.mlir)

| Selected AU / APS function | ScaleISA implementation input | Hector evaluated schedule |
|---|---|---|
| `au_generated_candidate_12__direct` | [input](au/au_generated_candidate_12__direct.input.mlir) | [evaluated](au/au_generated_candidate_12__direct.evaluated.mlir) |
| `au_generated_candidate_21__direct` | [input](au/au_generated_candidate_21__direct.input.mlir) | [evaluated](au/au_generated_candidate_21__direct.evaluated.mlir) |
| `au_generated_candidate_34__l0u` | [input](au/au_generated_candidate_34__l0u.input.mlir) | [evaluated](au/au_generated_candidate_34__l0u.evaluated.mlir) |
| `au_generated_candidate_41__direct` | [input](au/au_generated_candidate_41__direct.input.mlir) | [evaluated](au/au_generated_candidate_41__direct.evaluated.mlir) |
| `au_generated_candidate_51__direct` | [input](au/au_generated_candidate_51__direct.input.mlir) | [evaluated](au/au_generated_candidate_51__direct.evaluated.mlir) |
| `au_generated_candidate_56__l0u` | [input](au/au_generated_candidate_56__l0u.input.mlir) | [evaluated](au/au_generated_candidate_56__l0u.evaluated.mlir) |

[Scheduled APS/TOR and realized memory map](design.scheduled-tor.mlir)
