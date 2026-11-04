# APS + ScaleISA end-to-end examples

The 16 cases in [cases/](cases/) run discovery and selection, APS hardware
export, RV32 compilation, and Spike + Verilator validation. The shared runner also
checks a zero-CI software baseline against an independent reference.

## Setup and run

Initialize the repository's ScaleISA and Chipyard checkouts and build the tools:

```sh
pixi run setup-chipyard
pixi run build
```

Run one case or the complete suite from the repository root:

```sh
make -C examples/aps-scaleisa CASE=bitcount run
make -C examples/aps-scaleisa CASE=all run
```

Add `--chipyard` to measure accelerated and baseline programs on full Rocket RTL:

```sh
make -C examples/aps-scaleisa CASE=octree rocket
```

`make list` lists cases. `make CASE=all plan` prints exact commands without
executing them. `make CASES="octree iss" run` selects multiple cases. The Python
entry point remains available for direct invocation; `--list` and `--plan`
provide the same inspection modes.
`--output-dir` changes the default `build/aps-scaleisa-examples/` directory.
`--jobs` controls driver and Chipyard parallelism.

## Cases

| ID | Workload |
|---|---|
| `octree` | Octree search, 2048 points and 128 queries |
| `iss` | ISS keypoints, 16 x 16 |
| `normal` | Normal estimation, 12 x 12 |
| `morphological` | Morphological filter, 12 x 12 |
| `knn` | Top-8 nearest neighbors, 32 points and 4 queries |
| `icp` | ICP, 32 points and 4 iterations |
| `ransac` | Sphere RANSAC, 32 points and 4 hypotheses |
| `segmentation` | Region growing, 8 x 8 and 2 iterations |
| `poisson` | Poisson surface reconstruction, 6 cubed grid |
| `sfm_batch` | Batch structure from motion |
| `stereo_vo` | Stereo visual odometry |
| `planar_slam` | Planar SLAM |
| `pose2_slam` | Pose2 graph solver |
| `pose3_pose_graph` | Pose3 graph solver |
| `bitcount` | Bit counting, 64 deterministic inputs |
| `sq_dist` | Squared distance, 64 deterministic inputs |

All 16 cases use the same layout under [cases/](cases/): `source.cpp` contains
the workload, `main.cpp` independently validates its outputs, and `case.json`
supplies the evaluation settings. The cases derive from `scaleisa/benchmarks/`.

One shared [run.py](run.py) handles Cgeist translation, software baseline
compilation, Spike validation, hardware/software generation, and optional Rocket
measurement. The baseline and selected program use the same validation harness.
Startup assembly and the linker script are shared in [fixtures/](fixtures/).
All generated MLIR, RTL, binaries, logs, and summaries go under `build/` by default.
No prebuilt example products or previous discovery results are required.

Each case writes stage logs and driver artifacts under `<output>/<case>/native/`.
Rocket measurements use `<output>/<case>/chipyard/` and require a selected
accelerator. The suite summary records completed cases and stops at the first
failure. Output validation runs for both the software baseline and selected
program; Rocket timing excludes the independent reference checks.

Modeled speedup, measured CI latency, Rocket program cycles, and physical timing
are separate quantities. The runners validate their stated input sizes; they do
not measure complete upstream frameworks. See the
[driver guide](../../docs/COMPILER/aps-scaleisa-opt.md) for artifact contracts.
