"""
Test Suite for CubeSat Collision Avoidance Pipeline

Unit tests and integration tests for all components.
"""

import os
import sys
import unittest
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from phases.phase6_maneuver_planning_rl.environment.orbital_env import OrbitalCollisionEnv
from phases.phase6_maneuver_planning_rl.models.maddpg_agent import MADDPG, ReplayBuffer
from utils.rl_metrics import CollisionMetrics, FuelMetrics, SafetyMetrics


# ================================================
# ENVIRONMENT TESTS
# ================================================

class TestOrbitalEnvironment(unittest.TestCase):
    """Test orbital environment"""
    
    def setUp(self):
        self.env = OrbitalCollisionEnv(
            num_objects=5,
            max_steps=100,
            safe_distance=20.0,
            collision_distance=2.0
        )
    
    def tearDown(self):
        self.env.close()
    
    def test_environment_initialization(self):
        """Test environment initializes correctly"""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.max_steps, 100)
    
    def test_reset(self):
        """Test reset functionality"""
        state, info = self.env.reset()
        self.assertIsNotNone(state)
        self.assertTrue(isinstance(state, np.ndarray))
        self.assertTrue(isinstance(info, dict))
    
    def test_step(self):
        """Test environment step"""
        self.env.reset()
        
        # Random actions
        action = np.random.randn(3, 2)
        action = np.clip(action, -1, 1)
        
        state, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertIsNotNone(state)
        self.assertIsNotNone(reward)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsNotNone(info)
    
    def test_episode_termination(self):
        """Test episode termination"""
        self.env.reset()
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated) and steps < 1000:
            action = np.random.randn(3, 2)
            action = np.clip(action, -1, 1)
            _, _, terminated, truncated, _ = self.env.step(action)
            steps += 1
        
        self.assertGreater(steps, 0)
        self.assertTrue(terminated or truncated or steps >= 1000)


# ================================================
# MADDPG TESTS
# ================================================

class TestMADDPG(unittest.TestCase):
    """Test MADDPG agent"""
    
    def setUp(self):
        self.device = torch.device("cpu")
        self.maddpg = MADDPG(
            num_agents=3,
            state_dim=24,
            action_dim=2,
            device=self.device
        )
    
    def test_agent_initialization(self):
        """Test agent initializes"""
        self.assertIsNotNone(self.maddpg)
        self.assertEqual(len(self.maddpg.agents), 3)
    
    def test_actor_forward(self):
        """Test actor network forward pass"""
        state = torch.randn(1, 24)
        action = self.maddpg.agents[0].actor(state)
        
        self.assertEqual(action.shape, (1, 2))
        self.assertTrue(torch.all(action >= -1) and torch.all(action <= 1))
    
    def test_replay_buffer(self):
        """Test replay buffer"""
        buffer = ReplayBuffer(buffer_size=1000)
        
        state = np.random.randn(24)
        actions = np.random.randn(3, 2)
        rewards = np.array([1.0, 1.0, 1.0])
        next_state = np.random.randn(24)
        done = False
        
        buffer.push(state, actions, rewards, next_state, done)
        
        self.assertEqual(len(buffer), 1)
        
        # Sample batch
        batch = buffer.sample(batch_size=1)
        self.assertEqual(len(batch), 5)


# ================================================
# METRICS TESTS
# ================================================

class TestMetrics(unittest.TestCase):
    """Test metrics computation"""
    
    def test_collision_rate(self):
        """Test collision rate metric"""
        collisions = np.array([0, 0, 1, 1, 0])
        rate = CollisionMetrics.collision_rate(collisions)
        self.assertAlmostEqual(rate, 0.4)
    
    def test_success_rate(self):
        """Test success rate metric"""
        collisions = np.array([0, 0, 0, 1, 0])
        distances = np.array([25, 30, 20, 5, 22])
        
        success = CollisionMetrics.success_rate(collisions, distances, safe_threshold=20.0)
        self.assertGreaterEqual(success, 0)
        self.assertLessEqual(success, 1)
    
    def test_fuel_cost(self):
        """Test fuel cost metric"""
        actions = np.array([
            [0.1, 0.2],
            [0.0, 0.0],
            [0.3, 0.1]
        ])
        
        fuel = FuelMetrics.total_delta_v(actions)
        self.assertGreater(fuel, 0)
    
    def test_safety_margins(self):
        """Test safety margin statistics"""
        distances = np.array([25, 30, 20, 15, 22])
        margins = SafetyMetrics.safety_margin_distribution(distances, safe_distance=20.0)
        
        self.assertIn('mean_margin', margins)
        self.assertIn('std_margin', margins)
        self.assertGreater(margins['mean_margin'], 0)


# ================================================
# INTEGRATION TESTS
# ================================================

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_environment_maddpg_integration(self):
        """Test environment with MADDPG agent"""
        
        env = OrbitalCollisionEnv(num_objects=5, max_steps=50)
        maddpg = MADDPG(
            num_agents=3,
            state_dim=24,
            action_dim=2,
            device=torch.device("cpu")
        )
        
        state, info = env.reset()
        
        # Run one episode
        for step in range(10):
            actions = []
            with torch.no_grad():
                for agent_id in range(3):
                    agent_state = torch.FloatTensor(state).unsqueeze(0)
                    action = maddpg.agents[agent_id].actor(agent_state).numpy()
                    actions.append(np.clip(action[0], -1, 1))
            
            actions_array = np.array(actions)
            state, reward, terminated, truncated, info = env.step(actions_array)
            
            self.assertIsNotNone(state)
            self.assertTrue(isinstance(reward, (float, int, np.ndarray)))
            
            if terminated or truncated:
                break
        
        env.close()


# ================================================
# TEST RUNNER
# ================================================

def run_quick_tests():
    """Run quick test suite"""
    
    print("\n" + "="*60)
    print("Running CubeSat Pipeline Tests")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestOrbitalEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestMADDPG))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


# ================================================
# MAIN
# ================================================

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)
