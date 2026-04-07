# Multimodal Fusion Models

This directory contains various approaches for fusing multimodal inputs (trajectory, graph, vision) for collision risk prediction.

## Available Models

### 1. ConfidenceWeightedFusion
- **Purpose**: Fuses predictions from multiple models using confidence weighting
- **Key Features**:
  - Learns confidence scores for each modality
  - Adaptive weighting based on input reliability
  - Handles missing modalities gracefully
- **Use Case**: When you have pre-trained unimodal models to combine

### 2. FusionModel
- **Purpose**: End-to-end fusion of raw multimodal features
- **Key Features**:
  - Joint feature extraction and fusion
  - Cross-modal attention mechanisms
  - Unified risk prediction
- **Use Case**: Training from scratch with multimodal data

### 3. AttentionFusionModel
- **Purpose**: Advanced fusion using transformer-style attention
- **Key Features**:
  - Multi-head cross-modal attention
  - Learned importance weighting
  - Better handling of modality interactions
- **Use Case**: Complex multimodal relationships requiring attention

## Usage Examples

### Training a Fusion Model

```python
from models.fusion.fusion_model import FusionModel

# Initialize model
model = FusionModel(
    trajectory_dim=256,
    graph_dim=128,
    vision_dim=512,
    fusion_dim=256
)

# Training loop
for batch in dataloader:
    trajectory = batch['trajectory']
    graph_data = batch['graph']
    vision_features = batch['vision']
    labels = batch['risk']

    # Forward pass
    predictions = model(trajectory, graph_data, vision_features)

    # Compute loss
    loss = criterion(predictions, labels)
```

### Confidence-Weighted Fusion

```python
from models.fusion.confidence_fusion_model import ConfidenceWeightedFusion

# Initialize with pre-trained models
fusion_model = ConfidenceWeightedFusion(
    trajectory_model=trajectory_model,
    graph_model=graph_model,
    vision_model=vision_model
)

# Fit confidence weights
fusion_model.fit_confidence_weights(train_data)

# Predict with confidence weighting
risk_predictions = fusion_model.predict(batch)
```

## Model Comparison

| Model | Training | Inference | Fusion Strategy | Best Use Case |
|-------|----------|-----------|-----------------|---------------|
| ConfidenceWeightedFusion | Fast | Fast | Weighted averaging | Pre-trained models |
| FusionModel | Medium | Medium | Feature concatenation | End-to-end training |
| AttentionFusionModel | Slow | Medium | Cross-attention | Complex interactions |

## Performance Tips

1. **Data Normalization**: Ensure all modalities are properly normalized
2. **Missing Modalities**: Use confidence weighting for robust handling
3. **Training Stability**: Start with lower learning rates for attention models
4. **Regularization**: Use dropout and batch normalization for better generalization

## Extending the Models

To add a new fusion approach:

1. Create a new class inheriting from `nn.Module`
2. Implement `forward()` method taking trajectory, graph, and vision inputs
3. Add confidence estimation if applicable
4. Update this README and `__init__.py`

## References

- [Multimodal Learning Survey](https://arxiv.org/abs/1907.05589)
- [Attention Mechanisms](https://arxiv.org/abs/1706.03762)
- [Confidence Estimation](https://arxiv.org/abs/1805.08206)