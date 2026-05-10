import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
import os

@dataclass
class DemandForecastResult:
    model_lr: Any
    model_rf: Any
    mae_lr: float
    rmse_lr: float
    mae_rf: float
    rmse_rf: float
    best_model: Any
    best_name: str

@dataclass
class AnomalyDetectionResult:
    model_dt: Any
    model_rf: Any
    acc_dt: float
    acc_rf: float
    cm_dt: np.ndarray
    cm_rf: np.ndarray
    best_model: Any
    best_name: str
    feature_importances: np.ndarray

def generate_bike_sharing_data(n=1000) -> pd.DataFrame:
    """Generates synthetic bike sharing data."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="H")
    season = np.random.randint(1, 5, n)
    holiday = np.random.choice([0, 1], p=[0.95, 0.05], size=n)
    workingday = np.random.choice([0, 1], p=[0.3, 0.7], size=n)
    weather = np.random.randint(1, 4, n)
    temp = np.random.uniform(0.1, 1.0, n)
    atemp = temp + np.random.uniform(-0.05, 0.05, n)
    humidity = np.random.uniform(0.2, 0.9, n)
    windspeed = np.random.uniform(0.0, 0.5, n)
    
    # Base count formula
    noise = np.random.normal(0, 10, n)
    count = 50 + 200 * temp + 30 * season - 20 * weather + noise
    count = np.maximum(0, count).astype(int)
    
    casual = (count * 0.2).astype(int)
    registered = count - casual
    
    df = pd.DataFrame({
        "datetime": dates,
        "season": season,
        "holiday": holiday,
        "workingday": workingday,
        "weather": weather,
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
        "casual": casual,
        "registered": registered,
        "count": count
    })
    
    # Save if path exists
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/bike_sharing_sample.csv", index=False)
    return df

def generate_anomaly_data(n=800) -> pd.DataFrame:
    """Generates synthetic anomaly data."""
    np.random.seed(42)
    
    # 0=Normal, 1=BatteryAnomaly, 2=RouteAnomaly, 3=SensorSpike
    labels = np.random.choice([0, 1, 2, 3], p=[0.7, 0.1, 0.1, 0.1], size=n)
    
    battery_drop = np.zeros(n)
    speed = np.zeros(n)
    route_dev = np.zeros(n)
    alt_change = np.zeros(n)
    speed_change = np.zeros(n)
    
    for i in range(n):
        if labels[i] == 0:
            battery_drop[i] = np.random.normal(5, 3)
            speed[i] = np.random.normal(15, 4)
            route_dev[i] = np.random.normal(2, 1.5)
            alt_change[i] = np.random.normal(1, 1)
            speed_change[i] = np.random.normal(2, 1)
        elif labels[i] == 1:
            battery_drop[i] = np.random.normal(12, 4)
            speed[i] = np.random.normal(14, 4)
            route_dev[i] = np.random.normal(3, 2)
            alt_change[i] = np.random.normal(1, 1.5)
            speed_change[i] = np.random.normal(2, 2)
        elif labels[i] == 2:
            battery_drop[i] = np.random.normal(6, 3)
            speed[i] = np.random.normal(15, 4)
            route_dev[i] = np.random.normal(8, 3)
            alt_change[i] = np.random.normal(2, 1)
            speed_change[i] = np.random.normal(3, 2)
        elif labels[i] == 3:
            battery_drop[i] = np.random.normal(5, 3)
            speed[i] = np.random.normal(15, 5)
            route_dev[i] = np.random.normal(2, 2)
            alt_change[i] = np.random.normal(8, 4)
            speed_change[i] = np.random.normal(10, 5)
            
    df = pd.DataFrame({
        "battery_drop": battery_drop,
        "speed": speed,
        "route_deviation": route_dev,
        "altitude_change": alt_change,
        "speed_change": speed_change,
        "label": labels
    })
    
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/anomaly_data.csv", index=False)
    return df

def train_demand_model(df: pd.DataFrame) -> DemandForecastResult:
    features = ["season", "holiday", "workingday", "weather", "temp", "humidity", "windspeed"]
    X = df[features]
    y = df["count"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    best_model = rf if rmse_rf < rmse_lr else lr
    best_name = "RandomForest" if rmse_rf < rmse_lr else "LinearRegression"
    
    return DemandForecastResult(lr, rf, mae_lr, rmse_lr, mae_rf, rmse_rf, best_model, best_name)

def train_anomaly_model(df: pd.DataFrame) -> AnomalyDetectionResult:
    features = ["battery_drop", "speed", "route_deviation", "altitude_change", "speed_change"]
    X = df[features]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    best_model = rf if acc_rf >= acc_dt else dt
    best_name = "RandomForest" if acc_rf >= acc_dt else "DecisionTree"
    
    return AnomalyDetectionResult(dt, rf, acc_dt, acc_rf, cm_dt, cm_rf, best_model, best_name, rf.feature_importances_)

def predict_demand(model: Any, features: Dict[str, float]) -> float:
    df = pd.DataFrame([features])
    return float(model.predict(df)[0])

def detect_anomaly(model: Any, telemetry: Dict[str, float]) -> int:
    df = pd.DataFrame([telemetry])
    return int(model.predict(df)[0])
