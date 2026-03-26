## System Architecture

### General Pipeline Architecture

The following diagram illustrates the complete end-to-end system pipeline, covering all major modules from data ingestion to autonomous maneuver execution.

<p align="center">
  <img src="Architectural_Diagram/general_architecture.png" alt="General Architecture Diagram" width="900"/>
</p>

This architecture includes:

- Data ingestion from TLE and debris sources  
- Orbital propagation using SGP4  
- Object detection and tracking  
- Temporal trajectory prediction  
- Collision risk estimation  
- Reinforcement learning-based maneuver planning  

---

### Conceptual Model (Core Contribution)

The conceptual diagram highlights the core research contribution of this project, focusing on temporal modeling and intelligent decision-making.

<p align="center">
  <img src="Architectural_Diagram/concept_diagram.png" alt="Concept Diagram" width="900"/>
</p>

This model emphasizes:

- Transformer-based temporal trajectory modeling  
- Comparison with static baseline approaches  
- Feature engineering for relative motion  
- Risk prediction pipeline  
- Decision-making via reinforcement learning  