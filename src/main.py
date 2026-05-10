import sys
import os

# Ensure src is in path so we can import modules properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from grid_model import create_city_grid
from layout_validator import validate_layout
from fleet_selector import run_ga_fleet_selection
from delivery_simulator import Delivery, run_simulation
from ml_pipeline import generate_bike_sharing_data, generate_anomaly_data, train_demand_model, train_anomaly_model
from visualization import plot_zone_map, plot_demand_heatmap, plot_fitness_history, plot_forecast_results, plot_confusion_matrix

def main():
    print("="*50)
    print("AERONET LITE // AUTONOMOUS DRONE DELIVERY SYSTEM")
    print("="*50)

    # 1. Build Grid
    print("\n[1] Building 10x10 City Grid...")
    grid = create_city_grid()
    plot_zone_map(grid)
    plot_demand_heatmap(grid)
    print("Grid built and initial maps generated.")

    # 2. Validate Layout
    print("\n[2] Validating Layout via CSP...")
    validation = validate_layout(grid)
    print(f"Passed rules: {validation['passed']}")
    print(f"Failed rules: {validation['failed']}")
    if validation['violations']:
        print("Violations:")
        for v in validation['violations']:
            print(f" - {v}")
    if validation['suggestions']:
        print("Suggestions:")
        for s in validation['suggestions']:
            print(f" - {s}")

    # 3. Fleet Selection GA
    print("\n[3] Running Genetic Algorithm for Fleet Selection...")
    fleet_res = run_ga_fleet_selection(grid, budget=15000)
    print(f"Optimal Fleet: {fleet_res['light']} Light Drones, {fleet_res['heavy']} Heavy Drones")
    print(f"Cost: ${fleet_res['cost']}, Coverage: {fleet_res['coverage']:.2f}%")
    plot_fitness_history(fleet_res['fitness_history'])

    # 4. Generate Deliveries
    print("\n[4] Generating Initial Deliveries...")
    deliveries = [
        Delivery(id=1, pickup=(0,0), dropoff=(2,2), weight=1.5, status="pending"),
        Delivery(id=2, pickup=(5,5), dropoff=(1,8), weight=4.0, status="pending"),
        Delivery(id=3, pickup=(9,9), dropoff=(7,1), weight=2.0, status="pending"),
        Delivery(id=4, pickup=(0,0), dropoff=(9,0), weight=1.0, status="pending"),
        Delivery(id=5, pickup=(5,5), dropoff=(4,4), weight=3.5, status="pending"),
        Delivery(id=6, pickup=(9,9), dropoff=(6,8), weight=2.5, status="pending"),
        Delivery(id=7, pickup=(0,0), dropoff=(3,9), weight=1.8, status="pending"),
        Delivery(id=8, pickup=(5,5), dropoff=(8,3), weight=4.5, status="pending"),
    ]
    print(f"Generated {len(deliveries)} pending deliveries.")

    # 5. Run Simulation
    print("\n[5] Running 20-Step Simulation Scenario...")
    sim_res = run_simulation(grid, fleet_res, deliveries)
    for event in sim_res.event_log:
        print(f"Step {event['step']:02d} [{event['level'].upper()}]: {event['message']}")

    # 6. Train ML Models
    print("\n[6] Initializing ML Pipeline...")
    if not os.path.exists("data/raw/bike_sharing_sample.csv"):
        generate_bike_sharing_data()
    if not os.path.exists("data/processed/anomaly_data.csv"):
        generate_anomaly_data()

    import pandas as pd
    demand_df = pd.read_csv("data/raw/bike_sharing_sample.csv")
    anomaly_df = pd.read_csv("data/processed/anomaly_data.csv")

    print("\nTraining Demand Forecasting Models...")
    demand_res = train_demand_model(demand_df)
    print(f"Linear Regression -> MAE: {demand_res.mae_lr:.2f}, RMSE: {demand_res.rmse_lr:.2f}")
    print(f"Random Forest     -> MAE: {demand_res.mae_rf:.2f}, RMSE: {demand_res.rmse_rf:.2f}")
    print(f"Best Model: {demand_res.best_name}")

    print("\nTraining Anomaly Detection Models...")
    anomaly_res = train_anomaly_model(anomaly_df)
    print(f"Decision Tree -> Accuracy: {anomaly_res.acc_dt*100:.2f}%")
    print(f"Random Forest -> Accuracy: {anomaly_res.acc_rf*100:.2f}%")
    print(f"Best Model: {anomaly_res.best_name}")

    # Generate ML plots
    from sklearn.model_selection import train_test_split
    X = demand_df[["season", "holiday", "workingday", "weather", "temp", "humidity", "windspeed"]]
    y = demand_df["count"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = demand_res.best_model.predict(X_test)
    plot_forecast_results(y_test.values, y_pred, demand_res.best_name)

    plot_confusion_matrix(anomaly_res.cm_rf, ["Normal", "BattAnom", "RteAnom", "Spike"])

    # Final Summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Deliveries Completed: {sim_res.completed}")
    print(f"Deliveries Delayed:   {sim_res.delayed}")
    print(f"Deliveries Failed:    {sim_res.failed}")
    print(f"No-Fly Activations:   {len(sim_res.no_fly_activations)}")
    print("All artifacts and figures saved to report/figures/")
    print("="*50)

if __name__ == "__main__":
    main()
