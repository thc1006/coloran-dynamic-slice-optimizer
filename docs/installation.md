# Installation Guide

This guide will walk you through setting up the `5G-Slice-Optimizer` project on your local machine.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **NVIDIA GPU**: A CUDA-enabled GPU is required for acceleration. An NVIDIA A100 with at least 16GB of VRAM is recommended for full dataset processing.
- **CUDA Toolkit**: Version 11.0 or newer.

## Installation Steps

1.  **Clone the Repository**

    ```
    git clone https://github.com/[your-username]/5G-Slice-Optimizer.git
    cd 5G-Slice-Optimizer
    ```

2.  **Create a Virtual Environment** (Recommended)

    ```
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**

    Install all the required Python packages using the `requirements.txt` file.

    ```
    pip install -r requirements.txt
    ```

    **Note**: The installation of `cuml-cu11` might take some time as it's a large, GPU-specific library.

4.  **Verify Installation**

    You can run the test suite to ensure that all components are set up correctly.

    ```
    pytest
    ```

You are now ready to use the project!
