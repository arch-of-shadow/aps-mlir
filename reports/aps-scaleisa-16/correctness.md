# Numerical result correctness

Each accelerated program was replayed on its full Chipyard Rocket RTL simulator. The original element-wise comparisons against independent C++ software references were instrumented to print executed comparison counts. Integers and float bit patterns must match exactly. These checks cover the measured deterministic inputs.

A separate negative control flips one bit of an actual result buffer immediately before those comparisons. The unmodified checker must report one mismatch and the simulated program must fail. This changes the computed data, not the checker or its expected software answer. Original benchmark MLIR and RTL are reused.

| Case | Compared words, two calls | Correct output | One-bit wrong output | Evidence |
|---|---:|---|---|---|
| octree | 45,312 | Exact match | Rejected | [correct](octree/correctness/correct-output.log), [corrupt](octree/correctness/corrupt-output.log), [harness](octree/correctness/corrupt-output.cpp) |
| iss | 2,560 | Exact match | Rejected | [correct](iss/correctness/correct-output.log), [corrupt](iss/correctness/corrupt-output.log), [harness](iss/correctness/corrupt-output.cpp) |
| normal | 2,880 | Exact match | Rejected | [correct](normal/correctness/correct-output.log), [corrupt](normal/correctness/corrupt-output.log), [harness](normal/correctness/corrupt-output.cpp) |
| morphological | 1,152 | Exact match | Rejected | [correct](morphological/correctness/correct-output.log), [corrupt](morphological/correctness/corrupt-output.log), [harness](morphological/correctness/corrupt-output.cpp) |
| knn | 344 | Exact match | Rejected | [correct](knn/correctness/correct-output.log), [corrupt](knn/correctness/corrupt-output.log), [harness](knn/correctness/corrupt-output.cpp) |
| icp | 398 | Exact match | Rejected | [correct](icp/correctness/correct-output.log), [corrupt](icp/correctness/corrupt-output.log), [harness](icp/correctness/corrupt-output.cpp) |
| ransac | 208 | Exact match | Rejected | [correct](ransac/correctness/correct-output.log), [corrupt](ransac/correctness/corrupt-output.log), [harness](ransac/correctness/corrupt-output.cpp) |
| segmentation | 384 | Exact match | Rejected | [correct](segmentation/correctness/correct-output.log), [corrupt](segmentation/correctness/corrupt-output.log), [harness](segmentation/correctness/corrupt-output.cpp) |
| poisson | 1,584 | Exact match | Rejected | [correct](poisson/correctness/correct-output.log), [corrupt](poisson/correctness/corrupt-output.log), [harness](poisson/correctness/corrupt-output.cpp) |
| sfm_batch | 1,728 | Exact match | Rejected | [correct](sfm_batch/correctness/correct-output.log), [corrupt](sfm_batch/correctness/corrupt-output.log), [harness](sfm_batch/correctness/corrupt-output.cpp) |
| stereo_vo | 582 | Exact match | Rejected | [correct](stereo_vo/correctness/correct-output.log), [corrupt](stereo_vo/correctness/corrupt-output.log), [harness](stereo_vo/correctness/corrupt-output.cpp) |
| planar_slam | 688 | Exact match | Rejected | [correct](planar_slam/correctness/correct-output.log), [corrupt](planar_slam/correctness/corrupt-output.log), [harness](planar_slam/correctness/corrupt-output.cpp) |
| pose2_slam | 784 | Exact match | Rejected | [correct](pose2_slam/correctness/correct-output.log), [corrupt](pose2_slam/correctness/corrupt-output.log), [harness](pose2_slam/correctness/corrupt-output.cpp) |
| pose3_pose_graph | 1,056 | Exact match | Rejected | [correct](pose3_pose_graph/correctness/correct-output.log), [corrupt](pose3_pose_graph/correctness/corrupt-output.log), [harness](pose3_pose_graph/correctness/corrupt-output.cpp) |
| bitcount | 128 | Exact match | Rejected | [correct](bitcount/correctness/correct-output.log), [corrupt](bitcount/correctness/corrupt-output.log), [harness](bitcount/correctness/corrupt-output.cpp) |
| sq_dist | 128 | Exact match | Rejected | [correct](sq_dist/correctness/correct-output.log), [corrupt](sq_dist/correctness/corrupt-output.log), [harness](sq_dist/correctness/corrupt-output.cpp) |
