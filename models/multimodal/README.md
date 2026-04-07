# Multimodal CubeSat Collision Avoidance Framework

This directory contains the implementation of a multimodal predictive framework for satellite collision risk assessment in dense low Earth orbit (LEO) environments.

## Overview

The framework integrates three key modalities:
- **Trajectory Modeling**: Transformer-based temporal modeling of orbital dynamics
- **Graph Neural Networks**: Modeling of neighbor interactions and spatial relationships
- **Vision-Based Perception**: Object detection and tracking from onboard cameras

## Directory Structure

```
models/
├── multimodal/
│   ├── multimodal_predictor.py      # Main multimodal model
│   └── __init__.py
├── gnn/
│   ├── satellite_gnn.py            # GNN for neighbor interactions
│   └── __init__.py
├── vision/
│   ├── satellite_vision.py         # Vision model for detection
│   └── __init__.py
└── trajectory_transformer_model.py # Existing trajectory model

data/
├── multimodal/
│   └── process_multimodal_data.py   # Data processing utilities
└── synthetic/                      # Generated synthetic data

scripts/
└── train_multimodal.py             # Training script

evaluation/
└── evaluate_multimodal.py          # Evaluation script

configs/
└── multimodal_config.yaml          # Configuration file
```

## Key Components

### 1. MultimodalCollisionPredictor
The main model that fuses information from all three modalities:

- **Trajectory Input**: Temporal sequences of position/velocity (batch, time_steps, 6)
- **Graph Input**: Current satellite positions and velocities for GNN
- **Vision Input**: Camera images for object detection
- **Output**: Collision risk prediction (0-1)

### 2. SatelliteGNN
Graph neural network for modeling satellite neighbor relationships:

- **Nodes**: Individual satellites with position/velocity features
- **Edges**: Connections based on communication/proximity range
- **Edge Features**: Relative distance, velocity, and angular relationships

### 3. SatelliteVisionModel
Vision model for onboard perception:

- **Backbone**: Pre-trained ResNet for feature extraction
- **Detection Head**: Bounding box regression for satellite localization
- **Classification Head**: Satellite identification

## Usage

### Training

```bash
# Train the multimodal model
python scripts/train_multimodal.py --config configs/multimodal_config.yaml

# Or with custom data directory
python scripts/train_multimodal.py --data_dir data/your_data --checkpoint_dir results/models/custom
```

### Evaluation

```bash
# Evaluate trained model
python evaluation/evaluate_multimodal.py --checkpoint results/models/multimodal/best_multimodal_model.pth

# Compare with baselines
python evaluation/evaluate_multimodal.py --baseline_results results/baselines/trajectory.csv results/baselines/gnn.csv
```

### Data Processing

```bash
# Create synthetic multimodal dataset
python data/multimodal/process_multimodal_data.py
```

## Configuration

The framework is configured via `configs/multimodal_config.yaml`:

```yaml
# Model architecture
trajectory_config:
  input_dim: 6
  d_model: 64
  nhead: 4
  num_layers: 2

gnn_config:
  node_dim: 6
  hidden_dim: 64
  output_dim: 32
  gnn_type: "gcn"  # gcn, gat, or sage

vision_config:
  feature_dim: 512
  pretrained: true

# Training parameters
batch_size: 32
learning_rate: 0.001
num_epochs: 100
```

## Research Objectives Addressed

1. **Dynamic Satellite Behavior**: Transformer captures temporal orbital patterns
2. **Neighbor Interactions**: GNN models spatial relationships and congestion
3. **Perception Integration**: Vision provides robustness to noisy TLE data
4. **Risk Prediction**: Multimodal fusion enables accurate collision forecasting
5. **Safety-Critical Evaluation**: Metrics focus on collision detection and false alarms

## Key Features

- **Modular Design**: Each modality can be trained/evaluated independently
- **Flexible Fusion**: Attention-based and concatenation fusion strategies
- **Safety Metrics**: Collision detection rate, false alarm rate, precision@recall
- **Scalable**: Handles variable numbers of satellites and temporal sequences
- **Real-time Capable**: Efficient inference for onboard deployment

## Dependencies

- PyTorch & PyTorch Geometric
- torchvision
- numpy, pandas, scikit-learn
- matplotlib, seaborn
- tqdm, pyyaml

## Future Extensions

- Multi-view vision integration
- Temporal GNN for dynamic graph evolution
- Reinforcement learning for maneuver planning
- Real satellite imagery integration
- Hardware acceleration for edge deployment