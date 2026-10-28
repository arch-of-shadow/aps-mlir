# Stereo VO (32 landmarks, 10 iterations)

All MLIR files are byte-for-byte copies of the measured run artifacts. Hashes and source paths are in [manifest.json](manifest.json).

[Collected AU definitions](scaleisa-au.mlir) · [APS hardware](aps-hardware.mlir) · [APS planning](aps-planned.mlir) · [CPU CI lowering](cpu-ci.mlir) · [Selected program](selected-program.mlir)

| Selected AU / APS function | ScaleISA implementation input | Hector evaluated schedule |
|---|---|---|
| `au_generated_candidate_17__direct` | [input](au/au_generated_candidate_17__direct.input.mlir) | [evaluated](au/au_generated_candidate_17__direct.evaluated.mlir) |
| `au_generated_candidate_26__direct` | [input](au/au_generated_candidate_26__direct.input.mlir) | [evaluated](au/au_generated_candidate_26__direct.evaluated.mlir) |

[Scheduled APS/TOR and realized memory map](design.scheduled-tor.mlir)
