# MADDPG Architecture for Multi-Agent Collision Avoidance

## Overview

This document describes the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) architecture used for coordinated satellite collision avoidance.

## Key Components

### 1. **Actor Networks**
- **Purpose**: Generate deterministic actions for each agent
- **Input**: Local state observation (24-dim: positions and velocities of nearby objects)
- **Output**: Continuous action (2-dim: velocity changes in x, y)
- **Architecture**: 3-layer MLP with 256 hidden units
- **Activation**: ReLU + Tanh output

### 2. **Critic Networks**
- **Purpose**: Q-value estimation for centralized learning
- **Input**: All agents' states + all agents' actions (concatenated)
- **Output**: Scalar Q-value
- **Architecture**: 3-layer MLP with 256 hidden units
- **Key Feature**: Centralized training, decentralized execution

### 3. **Target Networks**
- **Purpose**: Stabilize learning via temporal difference updates
- **Update Rule**: Soft updates with coefficient τ = 0.01
- **Formula**: θ_target = (1-τ) * θ_target + τ * θ_online

### 4. **Replay Buffer**
- **Capacity**: 100,000 transitions
- **Sampling**: Batch size 128
- **Storage**: [state, actions, rewards, next_state, done]

## Training Algorithm

### MADDPG Training Loop

```
for episode in 1 to max_episodes:
    
    # Reset environment
    state = env.reset()
    
    for step in 1 to max_steps:
        
        # 1. Collect experience
        for each agent:
            action = actor(state) + exploration_noise
        
        # Execute all actions in environment
        next_state, rewards, done, info = env.step(actions)
        
        # Store transition
        replay_buffer.push(state, actions, rewards, next_state, done)
        
        # 2. Update networks (if warmup period over)
        if episode > warmup_episodes and len(buffer) > batch_size:
            
            # Sample batch
            batch = replay_buffer.sample(batch_size)
            
            # Update each agent's critic
            for each agent i:
                # Compute Q-target
                Q_target = r_i + γ * critic_target(s', all_actors_target(s'))
                
                # Minimize critic loss
                L_critic = MSE(critic(s, actions), Q_target)
                ∇ critic
            
            # Update each agent's actor
            for each agent i:
                # Maximize expected reward
                L_actor = -mean(critic(s, [a_1, ..., actor(s_i), ..., a_n]))
                ∇ actor
            
            # Soft update target networks
            θ_target = (1-τ) * θ_target + τ * θ_online
        
        state = next_state
        if done: break
```

## Reward Design

### Objectives
1. **Maintain Safety**: Positive reward for maintaining safe distance
2. **Avoid Collisions**: Large negative reward for collisions
3. **Minimize Fuel**: Penalty for excessive maneuvers
4. **Encouragement**: Small reward for early episode completion

### Reward Function
```
r = w_safety * safety_bonus
    - w_collision * collision_penalty
    - w_fuel * fuel_cost
    + w_completion * completion_bonus
```

**Weights:**
- w_safety = 1.0
- w_collision = 10.0
- w_fuel = 0.1
- w_completion = 0.5

## Configuration Parameters

See `configs/maddpg_config.yaml` for all parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_agents | 3 | Number of CubeSats |
| state_dim | 24 | Input state dimension |
| action_dim | 2 | Output action dimension |
| hidden_dim | 256 | MLP hidden layer size |
| learning_rate | 0.001 | Optimizer learning rate |
| gamma | 0.99 | Discount factor |
| tau | 0.01 | Soft update coefficient |
| buffer_size | 100000 | Replay buffer capacity |
| batch_size | 128 | Training batch size |

## Performance Metrics

### Primary Metrics
- **Collision Rate**: % of episodes with collisions
- **Success Rate**: % of safe episodes (no collision, maintained distance)
- **Fuel Efficiency**: Total Δv per episode

### Secondary Metrics
- **Close Call Rate**: % episodes with min_distance < 20m but no collision
- **Critical Encounters**: % episodes with min_distance < 5m
- **Episode Length**: Average steps to completion

## Coordination Strategy

### Multi-Agent Dynamics
- **Independent Exploration**: Each actor explores independently
- **Centralized Critic**: Single critic sees all states/actions
- **Decentralized Execution**: Each agent acts on local observations
- **Implicit Coordination**: Critic provides global guidance

### Key Benefits
1. Scalability: Can handle variable number of agents
2. Stability: Centralized critic prevents policy divergence
3. Efficiency: Parallel execution on satellites
4. Robustness: Individual agent failure doesn't collapse system

## Computational Considerations

### CubeSat Deployment
- **Model Size**: ~256KB (actors only)
- **Inference Time**: <5ms per decision
- **Memory Usage**: ~50MB
- **Update Frequency**: 1Hz (space orbital dynamics)

### Training Infrastructure
- **GPU Acceleration**: Yes (NVIDIA GPU recommended)
- **Training Time**: ~4 hours for 5000 episodes
- **Data Usage**: ~50GB (replay buffer iterations)

## Potential Improvements

1. **Communication-Based Actions**: Allow agents to explicitly exchange information
2. **Attention Mechanisms**: Focus on nearest threats
3. **Curriculum Learning**: Easy to hard scenario progression
4. **Transfer Learning**: Pre-train on simulation, fine-tune on real data
5. **Hierarchical Policy**: High-level planner + low-level controller

## References

- Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", ICML 2017
- Project configuration: `configs/maddpg_config.yaml`
- Training script: `phases/phase6_maneuver_planning_rl/training/train_maddpg_agent.py`
- Evaluation: `phases/phase6_maneuver_planning_rl/evaluation/evaluate_maddpg.py`
