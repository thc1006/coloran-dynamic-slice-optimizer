# Experiments

This document describes the experimental setup and how to reproduce the results.

## Original Experiment

The main experiment is defined in `experiments/run_experiment.py`. It uses the full ColO-RAN dataset and leverages an NVIDIA A100 GPU for acceleration.

### Key Parameters:
-   **Dataset**: ColO-RAN (`rome_static_medium` scenario)
-   **Combinations**: 588 (3 schedulers × 28 traffic profiles × 7 base stations)
-   **Models**:
    -   **Random Forest**: 300 estimators, max depth 16 (cuML).
    -   **Neural Network**: 3 hidden layers (256 -> 128 -> 64), Adam optimizer, mixed-precision.
-   **Optimization Algorithms**:
    -   **Genetic Algorithm**: 80 population, 15 generations.
    -   **Exhaustive Search**: For networks with a small number of resource blocks.

## Reproducing Results

To reproduce the original results, you can run the main experiment script:

