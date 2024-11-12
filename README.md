# DRL_SDN
A Deep Reinforcement Learning Approach For Software-Defined Networking Routing Optimisation

# Network Packet Routing with Deep Q-Learning

This project implements an intelligent network packet routing system using Deep Q-Learning (DQL). The system optimizes server resource utilization by efficiently distributing network packets across multiple servers while considering CPU, memory, and bandwidth constraints.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Components](#components)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [File Structure](#file-structure)
8. [Model Architecture](#model-architecture)
9. [Output](#output)
10. [Metrics](#metrics)

## Project Overview

The system uses Deep Q-Learning to make intelligent packet routing decisions in real-time. It considers:
- Server resource capacities (CPU, memory, bandwidth)
- Current server loads
- Packet requirements
- Load balancing across servers
- Resource utilization optimization

## Components

- `packet_routing.py`: Main implementation file containing:
  - Server configuration and management
  - Packet processing logic
  - DQN implementation
  - Training pipeline
  - Metrics collection and visualization
  
- `test_dqn_routing.py`: Testing framework for evaluating trained models

## Requirements

```
contourpy==1.3.0
cycler==0.12.1
filelock==3.16.1
fonttools==4.54.1
fsspec==2024.9.0
Jinja2==3.1.4
joblib==1.4.2
kiwisolver==1.4.7
MarkupSafe==3.0.1
matplotlib==3.9.2
mpmath==1.3.0
networkx==3.4.1
numpy==1.24.3
packaging==24.1
pandas==2.2.3
pillow==10.4.0
pyparsing==3.2.0
python-dateutil==2.9.0.post0
pytz==2024.2
scikit-learn==1.5.2
scipy==1.14.1
seaborn==0.13.2
six==1.16.0
sympy==1.13.3
threadpoolctl==3.5.0
torch==2.2.2
torchaudio==2.2.2
torchvision==0.17.2
typing_extensions==4.12.2
tzdata==2024.2
```


## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DRL_SDN
```

2. Create and activate a virtual environment (optional but recommended):
```bash
virtualenv venv -p python3
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Training the model:
```bash
python packet_routing.py
```

2. Testing a trained model:
```bash
python test_dqn_routing.py
```

## Configuration

### Server Configurations

Server configurations are defined in both scripts. You can modify the `server_configs` list to match your infrastructure:

```python
server_configs = [
    {'cpu': 2.0, 'memory': 4, 'bandwidth': 1000},  # 2GHz CPU, 4GB RAM, 1Gbps
    {'cpu': 3.0, 'memory': 8, 'bandwidth': 2000},  # 3GHz CPU, 8GB RAM, 2Gbps
    {'cpu': 1.5, 'memory': 2, 'bandwidth': 500}    # 1.5GHz CPU, 2GB RAM, 500Mbps
]
```

### Training Parameters

Key parameters in `packet_routing.py`:
- `batch_size`: Size of training batches (default: 128)
- `gamma`: Discount factor (default: 0.99)
- `epsilon`: Initial exploration rate (default: 1.0)
- `epsilon_min`: Minimum exploration rate (default: 0.01)
- `epsilon_decay`: Exploration decay rate (default: 0.995)
- `learning_rate`: Learning rate (default: 0.001)

## File Structure

```
├── packet_routing.py          # Main implementation
├── test_dqn_routing.py       # Testing framework
├── requirements.txt          # Package dependencies
├── sampled_20000_data.csv   # Training data
├── network_traffic_subset.csv # Test data
└── training_results_*        # Generated training results
    ├── plots/               # Training visualization plots
    ├── metrics/             # Training metrics in JSON format
    └── models/              # Saved model checkpoints
```

## Model Architecture

The DQN model architecture consists of:
- Input layer: State size (varies based on number of servers)
- Hidden layer 1: 256 neurons with ReLU activation
- Hidden layer 2: 256 neurons with ReLU activation
- Output layer: Action size (number of servers)

## Output

### Training Results

The training process generates:
1. Training metrics plots:
   - Training rewards
   - Episode lengths
   - Acceptance rates
   - Server loads
   - Resource utilization

2. Model checkpoints:
   - Saved every 10 episodes
   - Final model saved as 'final_model.pth'

3. Metrics JSON files:
   - Episode-wise metrics
   - Final training summary

### Test Results

Testing generates:
1. Performance metrics:
   - Packet acceptance rate
   - Server utilization
   - Load balancing scores

2. Visualization plots:
   - Load balance over time
   - Server utilization
   - Resource usage comparison

## Metrics

The system tracks various metrics:

1. Performance Metrics:
   - Packet acceptance rate
   - Total processed packets
   - Rejection rate
   - Average episode reward

2. Resource Utilization:
   - CPU usage
   - Memory usage
   - Bandwidth utilization
   - Overall server load

3. Load Balancing:
   - Balance scores
   - Resource distribution
   - Server assignment distribution

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## License

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007