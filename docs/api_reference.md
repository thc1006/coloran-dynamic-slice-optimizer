# API Reference

This document provides a reference for the key classes and functions in the `coloran-dynamic-slice-optimizer` project.

## `src.data`

-   **`ColoRANDataLoader(base_path)`**: Handles downloading and loading the ColO-RAN dataset.
    -   `download_dataset()`: Clones the dataset from GitHub.
    -   `load_raw_data()`: Loads all CSV files into a pandas DataFrame.
-   **`MemoryOptimizedProcessor(slice_configs, batch_size)`**: Performs feature engineering and memory optimization.
    -   `process_data(slice_data)`: Processes the full dataset in batches.
    -   `optimize_datatypes(df)`: Downcasts data types to save memory.

## `src.models`

-   **`A100OptimizedTrainer(use_gpu, sample_size)`**: Manages the ML model training process.
    -   `prepare_data(df, features, target)`: Splits and scales the data.
    -   `train_random_forest(X_train, y_train)`: Trains a cuML or sklearn Random Forest.
    -   `train_neural_network(...)`: Trains a TensorFlow Neural Network.
    -   `evaluate_models(X_test, y_test)`: Evaluates model performance.
-   **`SliceEfficiencyPredictor(model_dir)`**: Loads pre-trained models for inference.
    -   `predict(feature_matrix, model_type)`: Returns efficiency predictions.

## `src.optimization`

-   **`SliceResourceAllocator(model_path, scaler_path, ...)`**: Orchestrates the optimization process.
    -   `simulate(steps, method)`: Runs a simulation using a specified optimization method.
-   **`GeneticOptimizer(...)`**: Implements the genetic algorithm for optimization.
    -   `run(state)`: Finds the best resource allocation for the given network state.
-   **`ExhaustiveOptimizer(...)`**: Implements the exhaustive search algorithm.
    -   `run(state)`: Finds the optimal resource allocation.

## `src.visualization`

-   **`create_comprehensive_visualization(...)`**: Generates and saves the main analysis plot.
