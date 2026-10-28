# ScaleISA → APS: measured benchmark comparison

Measured speedup = warm software CPU cycles / warm accelerated CPU cycles on full Chipyard APSRocketConfig Verilator (`run-binary-debug LOADMEM=1`). ScaleISA speedup = its modeled software cycles / modeled selected-program cycles. Reference generation, validation and printing are outside the timed regions. These are cycle speedups; scheduling period is recorded separately and is not a post-layout frequency result.

Each case index lists every selected AU, with its exact implementation input and Hector schedule. The APS link contains all corresponding hardware functions. CPU CI lowering and file hashes are included in each case directory.

| Case | AUs / CI sites | HLS period (ns) | CPU cycles baseline → accelerated | Measured speedup | ScaleISA estimate | ScaleISA AUs | APS dialect |
|---|---:|---:|---:|---:|---:|---|---|
| Octree (2,048 points, 128 queries) | 1 / 1 | 8 | 1,855,586 → 1,624,970 | 1.142× | 1.103× | [AU MLIR](octree/scaleisa-au.mlir), [index](octree/index.md) | [APS](octree/aps-hardware.mlir) |
| ISS keypoints (16×16) | 3 / 3 | 8 | 840,039 → 225,267 | 3.729× | 2.749× | [AU MLIR](iss/scaleisa-au.mlir), [index](iss/index.md) | [APS](iss/aps-hardware.mlir) |
| Normal estimation (12×12) | 2 / 2 | 8 | 136,025 → 83,511 | 1.629× | 1.510× | [AU MLIR](normal/scaleisa-au.mlir), [index](normal/index.md) | [APS](normal/aps-hardware.mlir) |
| Morphological filter (12×12) | 3 / 5 | 8 | 525,669 → 154,694 | 3.398× | 5.635× | [AU MLIR](morphological/scaleisa-au.mlir), [index](morphological/index.md) | [APS](morphological/aps-hardware.mlir) |
| Brute-force top-8 KNN (32 points, 4 queries) | 4 / 4 | 8 | 52,198 → 26,624 | 1.961× | 1.972× | [AU MLIR](knn/scaleisa-au.mlir), [index](knn/index.md) | [APS](knn/aps-hardware.mlir) |
| ICP (32 points, 4 iterations) | 2 / 2 | 8 | 195,374 → 120,733 | 1.618× | 2.824× | [AU MLIR](icp/scaleisa-au.mlir), [index](icp/index.md) | [APS](icp/aps-hardware.mlir) |
| RANSAC sphere (32 points, 4 hypotheses) | 2 / 2 | 8 | 14,470 → 9,421 | 1.536× | 4.142× | [AU MLIR](ransac/scaleisa-au.mlir), [index](ransac/index.md) | [APS](ransac/aps-hardware.mlir) |
| Region growing (8×8, 2 iterations) | 4 / 4 | 8 | 85,499 → 46,326 | 1.846× | 3.643× | [AU MLIR](segmentation/scaleisa-au.mlir), [index](segmentation/index.md) | [APS](segmentation/aps-hardware.mlir) |
| Poisson surface (6³ grid, 24 samples, 2 iterations) | 3 / 3 | 8 | 18,389 → 17,632 | 1.043× | 1.922× | [AU MLIR](poisson/scaleisa-au.mlir), [index](poisson/index.md) | [APS](poisson/aps-hardware.mlir) |
| Batch SFM (8 cameras, 24 points, 6 iterations) | 6 / 6 | 8 | 282,909 → 265,326 | 1.066× | 1.040× | [AU MLIR](sfm_batch/scaleisa-au.mlir), [index](sfm_batch/index.md) | [APS](sfm_batch/aps-hardware.mlir) |
| Stereo VO (32 landmarks, 10 iterations) | 2 / 2 | 8 | 181,601 → 163,484 | 1.111× | 1.034× | [AU MLIR](stereo_vo/scaleisa-au.mlir), [index](stereo_vo/index.md) | [APS](stereo_vo/aps-hardware.mlir) |
| Planar SLAM (16 poses, 20 landmarks) | 6 / 6 | 8 | 86,494 → 74,392 | 1.163× | 1.058× | [AU MLIR](planar_slam/scaleisa-au.mlir), [index](planar_slam/index.md) | [APS](planar_slam/aps-hardware.mlir) |
| Pose2 graph solver (24 poses, 40 edges) | 5 / 6 | 8 | 102,343 → 55,311 | 1.850× | 1.464× | [AU MLIR](pose2_slam/scaleisa-au.mlir), [index](pose2_slam/index.md) | [APS](pose2_slam/aps-hardware.mlir) |
| Pose3 graph solver (18 poses, 30 edges) | 4 / 5 | 8 | 135,647 → 80,477 | 1.686× | 1.666× | [AU MLIR](pose3_pose_graph/scaleisa-au.mlir), [index](pose3_pose_graph/index.md) | [APS](pose3_pose_graph/aps-hardware.mlir) |
| Bitcount (64 inputs) | 1 / 1 | 10 | 11,899 → 1,167 | 10.196× | 6.397× | [AU MLIR](bitcount/scaleisa-au.mlir), [index](bitcount/index.md) | [APS](bitcount/aps-hardware.mlir) |
| Squared distance (64 inputs) | 1 / 1 | 10 | 2,401 → 1,397 | 1.719× | 2.219× | [AU MLIR](sq_dist/scaleisa-au.mlir), [index](sq_dist/index.md) | [APS](sq_dist/aps-hardware.mlir) |

Pose2 and Pose3 rows time their solvers and bounded input staging; graph construction runs outside the timed region. Bitcount and squared distance are kernel microbenchmarks. Other rows time the complete repository snapshot entry point. All selected CI implementations are included in the linked AU and APS modules.

Area budgets, selected area estimates, cold measurements and exact profiles are retained in [summary.json](summary.json).

[Executed numerical software-reference comparisons](result-comparisons.json).

[Numerical software-reference comparisons and wrong-result controls](correctness.md).
