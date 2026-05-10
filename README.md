# AeroNet Lite

```text
    _    ___  ____   ___  _   _ _____ _____   _     ___ _____ _____ 
   / \  / _ \|  _ \ / _ \| \ | | ____|_   _| | |   |_ _|_   _| ____|
  / _ \| | | | |_) | | | |  \| |  _|   | |   | |    | |  | | |  _|  
 / ___ \ |_| |  _ <| |_| | |\  | |___  | |   | |___ | |  | | | |___ 
/_/   \_\___/|_| \_\\___/|_| \_|_____| |_|   |_____|___| |_| |_____|
                                                                    
         _.-=-._
       /'       `\
      |   (o)(o)  |
      |     __    |
       \   /  \  /
        `-.__.-'
          |  |
          |  |
          |  |
        --====--
```

## Overview & Objectives

**AeroNet Lite** is an Autonomous Drone Delivery Simulation system. It simulates a drone delivery network over a 10×10 city grid and incorporates five main AI modules to handle everything from layout validation to machine learning-based demand forecasting and anomaly detection.

The objectives of this project are:
- Demonstrate CSP (Constraint Satisfaction Problem) for grid layout validation.
- Implement a Genetic Algorithm (GA) for optimal drone fleet selection based on a budget.
- Plan dynamic drone delivery paths using the A* search algorithm.
- Simulate an end-to-end 20-step process including real-time disruption handling.
- Employ Machine Learning models (Regression for demand forecasting, Classification for anomaly detection) with a Streamlit interface.

## Architecture Diagram

```text
+-------------------+      +-------------------+      +-------------------+
|  Layout Validator | ---> |  Fleet Selector   | ---> |  A* Path Planner  |
|      (CSP)        |      |       (GA)        |      |      (A*)         |
+-------------------+      +-------------------+      +-------------------+
                                                           |
                                                           v
+-------------------+      +-------------------+      +-------------------+
|    ML Pipeline    | <--- |   Streamlit UI    | <--- |Delivery Simulator |
| (Reg. & Class.)   |      |  (app.py) & Vis.  |      |  (20-step logic)  |
+-------------------+      +-------------------+      +-------------------+
```

## Module Descriptions

- **Layout Validator**: Employs Constraint Satisfaction Problem (CSP) techniques to ensure the city grid layout adheres to specific zoning constraints (e.g., Industrial zones cannot be adjacent to Schools). It provides pass/fail rules and suggestions for fixes.
- **Fleet Selector**: Uses a Genetic Algorithm (GA) to select an optimal mix of Light and Heavy drones under a specified budget. It maximizes coverage while keeping costs within the budget limit.
- **A* Planner**: Computes the shortest delivery route for drones on the 10x10 grid using the A* search algorithm with Manhattan distance heuristics, avoiding designated no-fly zones.
- **Delivery Simulator**: A step-by-step engine that runs the drone simulation over 20 steps, handling deliveries, activating real-time no-fly zones, managing drone battery anomalies, and rerouting affected drones.
- **ML Pipeline**: Trains a Random Forest/Decision Tree model to detect anomalies in drone telemetry, and a Linear/Random Forest regression model for forecasting delivery demand based on weather and season features.

## Dataset Links & Sources

| Purpose | Dataset | URL |
|---|---|---|
| Population Density | US City Population Densities | https://www.kaggle.com/datasets/mmcgurr/us-city-population-densities |
| Demand Forecasting | Bike Sharing Demand | https://www.kaggle.com/c/bike-sharing-demand |
| Delivery Demand (alt) | Amazon Delivery Dataset | https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset |
| Drone Telemetry | ALFA UAV Fault & Anomaly Detection | https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963 |
| Drone Telemetry (alt) | Supplemental Drone Telemetry | https://www.kaggle.com/datasets/samsudeenashad/supplemental-drone-telemetry-data-and-operations-log |

*(Note: In this specific project, synthetic data is generated to simulate these datasets for simplicity and demonstration purposes.)*

## Installation & Setup

1. Clone or download this repository.
2. Install Python 3.10+.
3. Create a virtual environment (optional but recommended).
4. Run the following command to install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Command Line Interface (CLI)
To run the full backend simulation pipeline and train models, execute:
```bash
python src/main.py
```
This will print out the validation, GA results, the 20-step event log, and ML metrics. It will also generate figures in `report/figures/`.

### Streamlit Dashboard (UI)
To launch the interactive holographic dashboard:
```bash
streamlit run app.py
```
This will open the frontend on your browser.

## 20-Step Simulation Walkthrough

1. **Steps 1–3**: Initialize grid, validate layout, and select optimal drone fleet via GA.
2. **Steps 4–6**: Generate new deliveries and calculate initial A* routes for each drone.
3. **Steps 7–10**: Drones move along paths (1 cell/step).
4. **Step 11**: Disruption! A temporary no-fly zone is activated at a specific coordinate.
5. **Steps 12–14**: System re-evaluates and dynamically reroutes affected drones using A*.
6. **Steps 15–17**: Demand forecasting model is run to anticipate future delivery surges.
7. **Step 18**: A drone battery anomaly is injected (sudden massive drop).
8. **Step 19**: The Anomaly Detection ML model catches the issue and forces the drone to abort and return to the hub.
9. **Step 20**: The simulation concludes and produces a final status summary.

## ML Results Summary

| Model Type | Algorithm | Metrics |
|---|---|---|
| Demand Forecast (Regression) | Linear Regression | MAE: TBD, RMSE: TBD |
| Demand Forecast (Regression) | Random Forest | MAE: TBD, RMSE: TBD |
| Anomaly Detection (Classification) | Decision Tree | Accuracy: TBD |
| Anomaly Detection (Classification) | Random Forest | Accuracy: TBD |

## File Structure

```text
aeronet_lite/
├── data/
│   ├── raw/
│   │   └── bike_sharing_sample.csv       
│   └── processed/
│       └── anomaly_data.csv              
├── src/
│   ├── grid_model.py
│   ├── layout_validator.py
│   ├── fleet_selector.py
│   ├── astar_planner.py
│   ├── delivery_simulator.py
│   ├── ml_pipeline.py
│   ├── visualization.py
│   └── main.py
├── notebooks/
│   ├── demand_forecasting.ipynb
│   └── anomaly_classifier.ipynb
├── report/
│   ├── figures/                          
│   └── final_report.md
├── app.py                                
├── requirements.txt
└── README.md
```

## Constraints & Assumptions
- The grid is exactly 10x10.
- Movements are purely orthogonal (up, down, left, right).
- Fleet budget is initially set to $15,000.
- Machine learning models operate on simplified synthetic data generated at runtime if the CSV files don't exist.

## Team & Academic Context
Created as a comprehensive master project integrating Constraint Satisfaction, Genetic Algorithms, Path Planning (A*), Simulation, Machine Learning, and interactive Visualization.
