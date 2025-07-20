# ColO-RAN Dynamic Slice Optimizer

> GPU-accelerated dynamic resource allocation optimization for 5G network slicing using machine learning and metaheuristic algorithms

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](#)

## Overview

This project implements a **GPU-accelerated optimization system** for dynamic resource allocation in 5G network slicing. Using the **ColO-RAN dataset** with over 35 million records, we achieve **99.97% prediction accuracy** and **5.7% average efficiency improvement** through advanced machine learning and metaheuristic optimization.

### Key Features

- 🚀 **GPU Acceleration**: cuML + TensorFlow with A100 optimization
- 🧠 **Hybrid ML Models**: Random Forest + Neural Networks
- 🎯 **Multi-Algorithm Optimization**: Exhaustive search + Genetic algorithms  
- 📊 **Memory Optimization**: 81.6% memory reduction with smart batching
- 📈 **Comprehensive Analysis**: 15 engineered features + 6-chart visualization
- ⚡ **Real-time Performance**: Sub-second allocation decisions

## 🏗 Architecture

For a detailed overview of the system architecture, data flow, and integration points, please refer to the [Architecture Documentation](docs/architecture.md).

## Project Structure

```
coloran-dynamic-slice-optimizer/
├── README.md                          # Complete project overview
├── CONTRIBUTING.md                    # Development guidelines
├── CHANGELOG.md                       # Version history
├── LICENSE                           # Open source license
├── pyproject.toml                    # Modern Python packaging
├── requirements.txt                  # Pinned production dependencies
├── requirements-dev.txt              # Development dependencies
├── docker-compose.yml                # Multi-service deployment
├── .github/
│   ├── workflows/                   # GitHub Actions workflows
│   └── ISSUE_TEMPLATE.md            # Issue reporting template
├── docs/                            # Project documentation
├── src/                             # Source code
│   └── coloran_optimizer/           # Main application package
├── tests/                           # Comprehensive test suite
├── experiments/                     # Experiment configurations, notebooks, and results
├── deploy/                          # Deployment files (Docker, Kubernetes, Terraform)
└── scripts/                         # Utility scripts
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA 11.0+ (for GPU acceleration)
- 16GB+ GPU memory (recommended: A100)

### Installation & Running with Docker Compose

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thc1006/coloran-dynamic-slice-optimizer.git
    cd coloran-dynamic-slice-optimizer
    ```
2.  **Build and run the services:**
    ```bash
    docker-compose up --build
    ```
    This will start the FastAPI application, Kafka, and Zookeeper.

### Basic Usage (within Docker)

Once the services are running, the FastAPI application will be accessible at `http://localhost:8000`.
You can interact with the API to send network data for optimization.

For example, to access the API documentation:
`http://localhost:8000/docs`

## Technical Details

### Feature Engineering (15 Features)
- **Resource Utilization**: `prb_utilization`, `sum_requested_prbs`, `sum_granted_prbs`
- **QoS Metrics**: `qos_score`, `throughput_efficiency`, `network_load`  
- **Network Context**: `bs_id`, `slice_id`, `num_ues`, `sched_policy_num`
- **Temporal Features**: `hour`, `minute`, `day_of_week`
- **Allocation Info**: `allocated_rbgs`, `exp_id`

### Model Architecture
- **Random Forest**: 300 estimators, max_depth=16, GPU-accelerated (cuML)
- **Neural Network**: 256→128→64→1, mixed precision training, early stopping
- **Optimization**: 17 RBGs across 3 slices (eMBB, URLLC, mMTC)

### Dataset Statistics
- **Total Records**: 35,512,393
- **Scheduling Policies**: 3 (Round Robin, Weighted Fair, Proportional Fair)
- **Training Configurations**: 28 scenarios  
- **Base Stations**: 7 locations
- **Memory Usage**: 40GB+ → 1.5GB (optimized)

## Experimental Results

Our system achieves significant improvements in network slice efficiency:

- **5.7% average efficiency improvement** with exhaustive search
- **100% positive improvement rate** across all scenarios
- **Sub-second optimization** for real-time applications
- **Scalable to 35M+ data points** with GPU acceleration

## 🛠 Development

### Running Tests
```bash
pytest tests/
```

### Training Custom Models
```bash
python experiments/run_experiment.py --config configs/custom.yaml
```

### Reproducing Results
```bash
jupyter notebook experiments/notebooks/full_experiment.ipynb
```

## Documentation

- [API Documentation](docs/api/)
- [Architecture Documentation](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Developer Guide](docs/development.md)
- [User Guide](docs/user-guide.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Roadmap

This project is being developed in phases:

- **Phase 1 (Foundation)**: GPU-accelerated ML pipeline, basic optimization algorithms, initial documentation, comprehensive test suite, security basics.
- **Phase 2 (Enhancement)**: Deep Reinforcement Learning integration, real-time streaming data processing, multi-dataset validation, advanced optimization algorithms.
- **Phase 3 (Production)**: Security and privacy mechanisms, horizontal scaling with Kubernetes, production monitoring and alerting.
- **Phase 4 (Research Extension)**: 6G network scenario extension, federated learning, edge computing integration, predictive traffic forecasting.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ColO-RAN Dataset**: [Colosseum Open RAN Dataset](https://github.com/wineslab/colosseum-oran-coloran-dataset)
- **NVIDIA**: A100 GPU computing support
- **cuML**: GPU-accelerated machine learning
- **TensorFlow**: Neural network framework

## Contact

- **Author**: Hsiu-Chi Tsai (thc1006)
- **Email**: hctsai@linux.com

## Citation

If you use this work in your research, please cite:

```
@software{coloran-dynamic-slice-optimizer,
  title={coloran-dynamic-slice-optimizer: GPU-Accelerated Dynamic Resource Allocation},
  author={Hsiu-Chi Tsai},
  year={2025},
  url={https://github.com/thc1006/coloran-dynamic-slice-optimizer}
}
```

---

⭐ **Star this repository** if you find it useful!