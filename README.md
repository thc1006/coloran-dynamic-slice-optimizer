# coloran-dynamic-slice-optimizer


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

## 🏗Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ColO-RAN      │ →  │ Feature          │ →  │ ML Models       │
│   Dataset       │    │ Engineering      │    │ (RF + NN)       │
│ (35M+ records)  │    │ (15 features)    │    │ (99.97% R²)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Visualization   │ ←  │ Results          │ ←  │ Optimization    │
│ & Reporting     │    │ Analysis         │    │ (Exhaustive+GA) │
│ (6 charts)      │    │                  │    │ (5.7% improve.) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 11.0+
- 16GB+ GPU memory (recommended: A100)

### Installation

```
git clone https://github.com/yourusername/coloran-dynamic-slice-optimizer.git
cd coloran-dynamic-slice-optimizer
pip install -r requirements.txt
```

### Basic Usage

```
from src.data import ColoRANDataLoader, MemoryOptimizedProcessor
from src.models import A100OptimizedTrainer
from src.optimization import SliceResourceAllocator

# 1. Load and process data
loader = ColoRANDataLoader("./data/")
processor = MemoryOptimizedProcessor(batch_size=75000)
data = processor.process_data_in_batches(loader.load_raw_data())

# 2. Train models
trainer = A100OptimizedTrainer()
models = trainer.train_models(data)

# 3. Optimize allocation
allocator = SliceResourceAllocator(total_rbgs=17)
results = allocator.optimize_allocation(models, method="genetic")

print(f"Efficiency improvement: {results['avg_improvement']:.4f}")
```

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

## 🛠Development

### Running Tests
```
pytest tests/ -v
```

### Training Custom Models
```
python experiments/run_experiment.py --config configs/custom.yaml
```

### Reproducing Results
```
jupyter notebook experiments/notebooks/full_experiment.ipynb
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Usage Tutorial](docs/usage.md)  
- [API Reference](docs/api_reference.md)
- [Experimental Setup](docs/experiments.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Roadmap

- [ ] 🤖 Deep Reinforcement Learning integration (DQN, A3C)
- [ ] 🌐 Multi-cell coordination optimization
- [ ] ⚡ Real-time QoS constraint handling
- [ ] 📈 6G network scenario extension
- [ ] 🔄 Predictive traffic forecasting
- [ ] 🛡️ Security and privacy mechanisms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ColO-RAN Dataset**: [Colosseum Open RAN Dataset](https://github.com/wineslab/colosseum-oran-coloran-dataset)
- **NVIDIA**: A100 GPU computing support
- **cuML**: GPU-accelerated machine learning
- **TensorFlow**: Neural network framework

## Contact

- **Author**: Hsiu-Chi Tsai(thc1006)
- **Email**: hctsai@linux.com

## Citation

If you use this work in your research, please cite:

```
@software{coloran-dynamic-slice-optimizer,
  title={coloran-dynamic-slice-optimizer: GPU-Accelerated Dynamic Resource Allocation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/coloran-dynamic-slice-optimizer}
}
```

---

⭐ **Star this repository** if you find it useful!

