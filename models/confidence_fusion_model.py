from models.fusion.confidence_fusion_model import ConfidenceWeightedFusion

# Backward compatibility alias; real implementation is in models/fusion
# This file is kept as a small wrapper to avoid breaking older imports.
class ConfidenceWeightedFusionWrapper(ConfidenceWeightedFusion):
    pass

# Maintain same symbol
ConfidenceWeightedFusion = ConfidenceWeightedFusionWrapper
