# AeroNet Lite: Autonomous Drone Delivery Simulation
**Final Project Report**

## Abstract
This project presents AeroNet Lite, an autonomous drone delivery simulation system designed to operate over a city grid. The system incorporates Constraint Satisfaction Problems (CSP) for layout validation, Genetic Algorithms (GA) for optimal fleet selection, and A* Search for dynamic route planning. Furthermore, Machine Learning pipelines are integrated to forecast delivery demand and detect drone anomalies in real-time. The simulation demonstrates the integration of these AI techniques in a cohesive 20-step scenario, highlighting their effectiveness in managing urban aerial logistics.

## Introduction & Problem Statement
Urban drone delivery networks face complex operational challenges: ensuring safe city layouts, managing limited budgets for drone fleets, navigating around dynamic obstacles, and dealing with unpredictable hardware failures and fluctuating demands. AeroNet Lite addresses these issues by combining classical AI planning algorithms with modern machine learning approaches.

## System Architecture
The system consists of five main modules:
1. **Layout Validator**: Verifies zoning rules.
2. **Fleet Selector**: Optimizes the number of light and heavy drones.
3. **A* Planner**: Computes the shortest obstacle-free path for deliveries.
4. **Delivery Simulator**: The core engine that runs the scenario step-by-step.
5. **ML Pipeline**: Handles predictive demand and anomaly classification.

## Module 1: CSP Layout Validation
The city grid must comply with specific rules for safety and efficiency. We define 4 key rules:
- **R1**: Industrial cells cannot be adjacent to Schools or Hospitals.
- **R2**: Residential cells must be within 3 Manhattan distance of a Drone Hub.
- **R3**: Hubs must have a Charging Pad within 2 distance.
- **R4**: Hospitals require Medical Pickups within 1 distance.

The algorithm checks each cell against these constraints.
*Results*: Intentional violations were placed in the simulation to demonstrate the validator's effectiveness in identifying layout flaws and suggesting fixes.

## Module 2: Fleet Selection GA
Given a budget of $15,000, the Genetic Algorithm searches for the optimal fleet composition.
- **Chromosome**: `[light_count, heavy_count]`
- **Fitness Function**: `0.75 * coverage - 0.25 * budget_usage` (penalty if over budget)
*Results*: The GA rapidly converges (typically within 20 generations) to a mix that maximizes coverage without exceeding the budget constraint.

## Module 3: A* Path Planning
Delivery paths are computed using the A* algorithm.
- **State Space**: The 10x10 city grid.
- **Heuristic**: Manhattan distance (admissible since diagonal moves are disallowed).
*Results*: The algorithm efficiently routes drones around designated No-Fly Zones. When disruptions occur (e.g., dynamic No-Fly Zone activation), drones dynamically recalculate paths mid-flight.

## Module 4: Disruption Handling
In the 20-step simulation, the system successfully handles:
1. **Dynamic Rerouting**: Activating a No-Fly Zone forces drones to find alternative paths.
2. **Hardware Anomalies**: A simulated battery failure triggers an immediate "Return to Base" protocol, minimizing the risk of a drone crash.

## Module 5: ML Pipeline
**Demand Forecasting**: Uses a synthetic dataset based on the Bike Sharing Demand format. We compared Linear Regression and Random Forest Regressor models.
- Random Forest consistently outperforms Linear Regression due to non-linear feature interactions (season, weather, temperature).

**Anomaly Detection**: Uses synthetic UAV telemetry data. We compared Decision Tree and Random Forest classifiers.
- Random Forest achieves near-perfect accuracy on the synthetic data, correctly identifying Battery Anomalies, Route Anomalies, and Sensor Spikes.

## Simulation Results
The 20-step execution successfully initializes the grid, validates layout, deploys the optimized fleet, and handles deliveries. When disruptions (No-Fly zones and Battery Anomalies) are introduced, the system successfully reroutes and aborts missions as intended.

## Conclusion & Future Work
AeroNet Lite effectively demonstrates the power of combining traditional AI (CSP, GA, A*) with ML. Future work could involve expanding to a continuous 3D coordinate space, implementing multi-agent pathfinding (MAPF) to prevent drone collisions, and using real-world GIS data.

## References
1. Russell, S. J., & Norvig, P. (2010). *Artificial Intelligence: A Modern Approach*.
2. Fanaei, M., et al. (2021). ALFA: A Dataset for UAV Fault and Anomaly Detection.
3. Fanaei, Kaggle Bike Sharing Demand Dataset.
4. Kaggle US City Population Densities Dataset.
5. Streamlit Documentation (2024).
6. Scikit-learn Developers (2024). *Scikit-learn: Machine Learning in Python*.
7. Matplotlib Developers (2024). *Matplotlib: Visualization with Python*.
8. Holland, J. H. (1992). *Adaptation in Natural and Artificial Systems*.
