"""
Integrated End-to-End Pipeline for CubeSat Collision Avoidance

Combines all phases:
1. Vision Detection (YOLOv8)
2. Object Tracking (Kalman)
3. Trajectory Prediction (LSTM/Transformer)
4. Collision Risk Estimation (GNN)
5. Maneuver Planning (PPO/MADDPG)

This pipeline processes raw space imagery to autonomous collision avoidance maneuvers.
"""

import os
import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logger
from utils.data_loader import load_space_images
from utils.feature_engineering import build_state_vectors

from phases.phase2_vision_object_detection.models.yolov8_detector import YOLOv8Detector
from phases.phase3_object_tracking.models.kalman_tracker import KalmanTracker
from phases.phase3_trajectory_prediction.models.lstm_model import LSTMTrajectoryPredictor
from phases.phase4_collision_risk_estimation.models.gnn_model import GNNCollisionModel
from phases.phase6_maneuver_planning_rl.models.maddpg_agent import MADDPG
from phases.phase6_maneuver_planning_rl.environment.orbital_env import OrbitalCollisionEnv

logger = setup_logger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================
# INTEGRATED PIPELINE
# ================================================

class CubeSatCollisionAvoidancePipeline:
    """
    End-to-end collision avoidance pipeline
    """
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to main configuration
        """
        
        self.config = self._load_config(config_path)
        self.device = DEVICE
        
        logger.info("Initializing CubeSat Collision Avoidance Pipeline")
        
        # Initialize components
        self._init_models()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_models(self):
        """Initialize all pipeline components"""
        
        logger.info("Loading models...")
        
        model_dir = self.config.get('model_dir', 'results/saved_models')
        
        # Vision Detection
        try:
            self.detector = YOLOv8Detector(model_path=os.path.join(model_dir, 'yolov8n.pt'))
            logger.info("✓ YOLOv8 Detector loaded")
        except Exception as e:
            logger.warning(f"YOLOv8 Detector failed: {e}")
            self.detector = None
        
        # Object Tracking
        try:
            self.tracker = KalmanTracker(dt=1.0)
            logger.info("✓ Kalman Tracker initialized")
        except Exception as e:
            logger.warning(f"Kalman Tracker failed: {e}")
            self.tracker = None
        
        # Trajectory Prediction
        try:
            self.traj_predictor = LSTMTrajectoryPredictor().to(self.device)
            self.traj_predictor.load_state_dict(
                torch.load(os.path.join(model_dir, 'lstm_predictor.pth'), 
                          map_location=self.device)
            )
            self.traj_predictor.eval()
            logger.info("✓ LSTM Trajectory Predictor loaded")
        except Exception as e:
            logger.warning(f"LSTM Trajectory Predictor failed: {e}")
            self.traj_predictor = None
        
        # Collision Risk Estimation
        try:
            self.gnn_model = GNNCollisionModel().to(self.device)
            self.gnn_model.load_state_dict(
                torch.load(os.path.join(model_dir, 'gnn_collision_model.pth'),
                          map_location=self.device)
            )
            self.gnn_model.eval()
            logger.info("✓ GNN Collision Model loaded")
        except Exception as e:
            logger.warning(f"GNN Collision Model failed: {e}")
            self.gnn_model = None
        
        # Maneuver Planning (MADDPG)
        try:
            maddpg_config = self.config.get('maddpg_config', 'configs/maddpg_config.yaml')
            with open(maddpg_config, 'r') as f:
                maddpg_cfg = yaml.safe_load(f)
            
            self.maddpg = MADDPG(
                num_agents=maddpg_cfg['environment']['num_agents'],
                state_dim=maddpg_cfg['actor_network']['state_dim'],
                action_dim=maddpg_cfg['actor_network']['action_dim'],
                model_type=maddpg_cfg.get('model_type', 'standard'),
                device=self.device
            )
            
            self.maddpg.load(os.path.join(model_dir, 'maddpg_cubesat_final.pth'))
            logger.info("✓ MADDPG Maneuver Agent loaded")
        except Exception as e:
            logger.warning(f"MADDPG Agent failed: {e}")
            self.maddpg = None
    
    # ================================================
    # PIPELINE STAGES
    # ================================================
    
    def stage_1_object_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Stage 1: Detect objects in space imagery
        
        Args:
            image: Input image array
            
        Returns:
            List of detections {bbox, confidence, class}
        """
        
        if self.detector is None:
            logger.warning("Detector not available")
            return []
        
        detections = self.detector.detect(image)
        logger.debug(f"Detected {len(detections)} objects")
        
        return detections
    
    def stage_2_object_tracking(self, detections: List[Dict]) -> List[Dict]:
        """
        Stage 2: Track objects across frames
        
        Args:
            detections: List of detections
            
        Returns:
            List of tracked objects with IDs and states
        """
        
        if self.tracker is None:
            logger.warning("Tracker not available")
            return detections
        
        tracked_objects = self.tracker.update(detections)
        logger.debug(f"Tracking {len(tracked_objects)} objects")
        
        return tracked_objects
    
    def stage_3_trajectory_prediction(self, tracked_objects: List[Dict], 
                                     horizon: int = 10) -> List[Dict]:
        """
        Stage 3: Predict future trajectories
        
        Args:
            tracked_objects: Tracked object states
            horizon: Prediction horizon
            
        Returns:
            Objects with predicted trajectories
        """
        
        if self.traj_predictor is None:
            logger.warning("Trajectory predictor not available")
            return tracked_objects
        
        with torch.no_grad():
            for obj in tracked_objects:
                if 'state' in obj:
                    state_tensor = torch.FloatTensor(obj['state']).unsqueeze(0).to(self.device)
                    prediction = self.traj_predictor(state_tensor, horizon)
                    obj['predicted_trajectory'] = prediction.cpu().numpy()
        
        logger.debug(f"Predicted trajectories for {len(tracked_objects)} objects")
        
        return tracked_objects
    
    def stage_4_collision_risk_assessment(self, tracked_objects: List[Dict], 
                                         ego_state: np.ndarray) -> Dict:
        """
        Stage 4: Assess collision risks
        
        Args:
            tracked_objects: Tracked objects with predictions
            ego_state: CubeSat state [x, y, vx, vy]
            
        Returns:
            Risk assessment results
        """
        
        if self.gnn_model is None:
            logger.warning("GNN collision model not available")
            return {}
        
        # Build graph representation
        # (simplified - actual implementation would use Graph object)
        collision_risks = {}
        
        for i, obj in enumerate(tracked_objects):
            if 'predicted_trajectory' in obj:
                # Compute risk for this object
                risk = self._compute_collision_risk(ego_state, obj)
                collision_risks[f"object_{i}"] = risk
        
        logger.debug(f"Assessed collision risks: {len(collision_risks)} objects")
        
        return collision_risks
    
    def stage_5_maneuver_planning(self, ego_state: np.ndarray, 
                                 collision_risks: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Stage 5: Plan collision avoidance maneuvers
        
        Args:
            ego_state: CubeSat state
            collision_risks: Risk assessment for all objects
            
        Returns:
            Maneuver action and metadata
        """
        
        if self.maddpg is None:
            logger.warning("MADDPG not available")
            return np.array([0, 0]), {}
        
        # Get action from MADDPG
        with torch.no_grad():
            state_tensor = torch.FloatTensor(ego_state).unsqueeze(0).to(self.device)
            action = self.maddpg.actors[0](state_tensor).cpu().numpy()
            action = np.clip(action[0], -1, 1)  # Normalize to [-1, 1]
        
        logger.debug(f"Planned maneuver: {action}")
        
        return action, {'collision_risks': collision_risks}
    
    def _compute_collision_risk(self, ego_state: np.ndarray, target_obj: Dict) -> float:
        """Compute collision risk with single object"""
        
        if 'predicted_trajectory' not in target_obj:
            return 0.0
        
        # Simple risk: 1 / (min_distance + 1)
        pred_traj = target_obj['predicted_trajectory']
        ego_pos = ego_state[:2]
        
        min_distance = np.min([
            np.linalg.norm(ego_pos - pred_traj[t, :2])
            for t in range(len(pred_traj))
        ])
        
        return min(1.0, 20.0 / (min_distance + 1.0))
    
    # ================================================
    # END-TO-END PROCESSING
    # ================================================
    
    def process_frame(self, image: np.ndarray, ego_state: np.ndarray) -> Dict:
        """
        Process single frame through entire pipeline
        
        Args:
            image: Space image
            ego_state: Current CubeSat state
            
        Returns:
            Complete processing result with maneuver
        """
        
        result = {
            'timestamp': None,
            'detections': [],
            'tracked_objects': [],
            'collision_risks': {},
            'recommended_action': None,
            'maneuver_magnitude': 0.0
        }
        
        # Stage 1: Detection
        detections = self.stage_1_object_detection(image)
        result['detections'] = detections
        
        # Stage 2: Tracking
        tracked = self.stage_2_object_tracking(detections)
        result['tracked_objects'] = tracked
        
        # Stage 3: Trajectory Prediction
        predicted = self.stage_3_trajectory_prediction(tracked)
        
        # Stage 4: Collision Risk
        risks = self.stage_4_collision_risk_assessment(predicted, ego_state)
        result['collision_risks'] = risks
        
        # Stage 5: Maneuver Planning
        action, metadata = self.stage_5_maneuver_planning(ego_state, risks)
        result['recommended_action'] = action
        result['maneuver_magnitude'] = np.linalg.norm(action)
        
        return result
    
    def process_sequence(self, image_sequence: List[np.ndarray], 
                        state_sequence: List[np.ndarray]) -> List[Dict]:
        """
        Process sequence of frames
        
        Args:
            image_sequence: List of images
            state_sequence: List of CubeSat states
            
        Returns:
            List of results
        """
        
        results = []
        
        for img, state in zip(image_sequence, state_sequence):
            result = self.process_frame(img, state)
            results.append(result)
        
        return results


# ================================================
# INITIALIZATION HELPER
# ================================================

def initialize_pipeline() -> CubeSatCollisionAvoidancePipeline:
    """Quick initialization"""
    return CubeSatCollisionAvoidancePipeline()


# ================================================
# MAIN / DEMO
# ================================================

if __name__ == "__main__":
    
    logger.info("CubeSat Collision Avoidance Pipeline")
    logger.info("="*60)
    
    # Initialize
    pipeline = initialize_pipeline()
    
    # Demo with synthetic data
    logger.info("\nDemo: Processing synthetic scenario...")
    
    # Create dummy data
    dummy_image = np.random.rand(640, 480, 3)
    dummy_state = np.array([100, 50, 1.0, 0.5])  # [x, y, vx, vy]
    
    # Process
    result = pipeline.process_frame(dummy_image, dummy_state)
    
    logger.info("\nPipeline Output:")
    logger.info(f"  Detections: {len(result['detections'])}")
    logger.info(f"  Tracked Objects: {len(result['tracked_objects'])}")
    logger.info(f"  Collision Risks: {len(result['collision_risks'])}")
    logger.info(f"  Recommended Action: {result['recommended_action']}")
    logger.info(f"  Maneuver Magnitude: {result['maneuver_magnitude']:.4f}")
