# pyrefly: ignore [missing-import]
import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure src in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.grid_model import create_city_grid, GRID_SIZE, ZONE_COLORS
from src.layout_validator import validate_layout
from src.fleet_selector import run_ga_fleet_selection
from src.delivery_simulator import run_simulation, Delivery
from src.ml_pipeline import train_demand_model, train_anomaly_model, generate_bike_sharing_data, generate_anomaly_data, predict_demand, detect_anomaly

st.set_page_config(page_title="AeroNet Lite", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300;1,400;1,500&family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap');

:root {
    --bg-color: #0A0E1A;
    --cyan: #00F5FF;
    --green: #39FF14;
    --magenta: #FF006E;
    --text: #E0E0E0;
}

body {
    background-color: var(--bg-color);
    color: var(--text);
    font-family: 'DM Mono', monospace;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Mono', monospace;
}

.stApp {
    background: linear-gradient(135deg, #0A0E1A 0%, #111526 100%);
}

.title-glow {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    color: transparent;
    background-clip: text;
    -webkit-background-clip: text;
    background-image: linear-gradient(90deg, var(--cyan), var(--magenta));
    position: relative;
    padding-bottom: 20px;
}

.title-glow::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 20%;
    right: 20%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    box-shadow: 0 0 10px var(--cyan);
}

.card {
    background: rgba(10, 14, 26, 0.6);
    border: 1px solid rgba(0, 245, 255, 0.2);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 15px rgba(0, 245, 255, 0.1);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.4);
    border-color: rgba(0, 245, 255, 0.6);
}

.status-pass { color: var(--green); text-shadow: 0 0 5px var(--green); font-weight: bold;}
.status-fail { color: var(--magenta); text-shadow: 0 0 5px var(--magenta); font-weight: bold;}

.grid-table {
    width: 100%;
    border-collapse: collapse;
}

.grid-cell {
    width: 30px;
    height: 30px;
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
    vertical-align: middle;
    font-size: 0.7em;
    font-weight: bold;
}

.event-log-box {
    height: 300px;
    overflow-y: auto;
    background: #05070D;
    border: 1px solid var(--cyan);
    padding: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.85em;
    border-radius: 5px;
}

.event-info { color: #3498DB; }
.event-success { color: var(--green); }
.event-alert { color: var(--magenta); font-weight: bold; }
.event-warning { color: #F1C40F; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.5); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title-glow'>AERONET LITE // AUTONOMOUS DRONE DELIVERY SYSTEM_</div>", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'grid' not in st.session_state:
    st.session_state.grid = create_city_grid()
if 'budget' not in st.session_state:
    st.session_state.budget = 15000
if 'sim_result' not in st.session_state:
    st.session_state.sim_result = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:var(--cyan);'>Control Panel</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⚙️ Simulation Parameters")
    st.session_state.budget = st.slider("Fleet Budget ($)", 5000, 20000, 15000, 500)
    
    if st.button("🔄 Reset Simulation"):
        st.session_state.grid = create_city_grid()
        st.session_state.sim_result = None
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 📁 Datasets")
    st.markdown("- [Population Densities](https://www.kaggle.com/datasets/mmcgurr/us-city-population-densities)")
    st.markdown("- [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)")
    st.markdown("- [UAV Anomaly Data](https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963)")

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ CITY GRID COMMAND", 
    "✅ CSP LAYOUT VALIDATOR", 
    "🚁 FLEET COMMAND", 
    "📡 LIVE SIMULATION", 
    "🤖 ML INTELLIGENCE", 
    "📊 ANALYTICS REPORT"
])

def render_grid_html(grid):
    html = "<table class='grid-table'>"
    for r in range(GRID_SIZE):
        html += "<tr>"
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            bg = ZONE_COLORS.get(cell.zone, "#FFFFFF")
            content = cell.zone[:3].upper()
            if cell.is_hub: content += "<br><b>[H]</b>"
            elif cell.is_charging: content += "<br>[C]"
            elif cell.zone == "Hospital": content += "<br>[+]"
            elif cell.is_medical_pickup: content += "<br>[M]"
            
            if cell.no_fly: 
                bg = "#FF006E"
                content = "<b>NFZ</b>"
                
            html += f"<td class='grid-cell' style='background-color: {bg}; color: black;'>{content}</td>"
        html += "</tr>"
    html += "</table>"
    return html

# TAB 1: CITY GRID COMMAND
with tab1:
    st.markdown("### 🗺️ Interactive City Grid")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(render_grid_html(st.session_state.grid), unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Zone Legend")
        for z, color in ZONE_COLORS.items():
            st.markdown(f"<div style='display:flex;align-items:center;margin-bottom:5px;'><div style='width:15px;height:15px;background-color:{color};margin-right:10px;'></div>{z}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Toggle No-Fly Zone")
        nfz_r = st.number_input("Row", 0, 9, 0)
        nfz_c = st.number_input("Col", 0, 9, 0)
        if st.button("Toggle NFZ"):
            st.session_state.grid[nfz_r][nfz_c].no_fly = not st.session_state.grid[nfz_r][nfz_c].no_fly
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# TAB 2: CSP LAYOUT VALIDATOR
with tab2:
    st.markdown("<h3 style='color:var(--cyan);'>LAYOUT VALIDATION REPORT</h3>", unsafe_allow_html=True)
    val_res = validate_layout(st.session_state.grid)
    
    score = len(val_res['passed'])
    st.progress(score / 4.0)
    st.write(f"**Score: {score}/4 Rules Passed**")
    
    rules = {
        "R1": "Industrial cells cannot be directly adjacent to Schools or Hospitals.",
        "R2": "Every Residential cell must be within 3 Manhattan distance of a Drone Hub.",
        "R3": "Every Drone Hub must have a Charging Pad within 2 Manhattan distance.",
        "R4": "At least one Hospital must have a Medical Pickup within 1 Manhattan distance."
    }
    
    for r_id, desc in rules.items():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        status = "PASS ✅" if r_id in val_res['passed'] else "FAIL ❌"
        status_class = "status-pass" if "PASS" in status else "status-fail"
        
        st.markdown(f"<h4>{r_id}: <span class='{status_class}'>{status}</span></h4>", unsafe_allow_html=True)
        st.write(desc)
        
        if "FAIL" in status:
            with st.expander("View Violations"):
                for v in val_res['violations']:
                    if r_id in v or (r_id == "R1" and "Industrial" in v) or (r_id == "R2" and "Residential" in v) or (r_id == "R3" and "Hub" in v) or (r_id == "R4" and "Hospital" in v):
                        st.markdown(f"<span style='color:var(--magenta);'>- {v}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    if val_res['suggestions']:
        st.markdown("#### Suggestions to Fix")
        for s in val_res['suggestions']:
            st.info(s)

# TAB 3: FLEET COMMAND
with tab3:
    st.markdown("### 🚁 Fleet Optimization (Genetic Algorithm)")
    
    with st.spinner("Running GA..."):
        fleet_res = run_ga_fleet_selection(st.session_state.grid, st.session_state.budget)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Light Drones", fleet_res['light'], delta="Cost: $1000/ea", delta_color="off")
        st.metric("Heavy Drones", fleet_res['heavy'], delta="Cost: $1800/ea", delta_color="off")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Total Fleet Cost", f"${fleet_res['cost']}", delta=f"Budget: ${st.session_state.budget}", delta_color="normal" if fleet_res['cost'] <= st.session_state.budget else "inverse")
        st.metric("Estimated Coverage", f"{fleet_res['coverage']:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("#### GA Fitness History")
    fig = px.line(x=range(len(fleet_res['fitness_history'])), y=fleet_res['fitness_history'], labels={'x': 'Generation', 'y': 'Fitness'})
    fig.update_layout(plot_bgcolor='#0A0E1A', paper_bgcolor='#0A0E1A', font_color='#E0E0E0')
    fig.update_traces(line_color='#00F5FF')
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: LIVE SIMULATION
with tab4:
    st.markdown("### 📡 20-Step Autonomous Delivery Simulation")
    
    if st.button("🚀 LAUNCH SIMULATION", type="primary"):
        fleet_res = run_ga_fleet_selection(st.session_state.grid, st.session_state.budget)
        deliveries = [
            Delivery(id=1, pickup=(0,0), dropoff=(2,2), weight=1.5, status="pending"),
            Delivery(id=2, pickup=(5,5), dropoff=(1,8), weight=4.0, status="pending"),
            Delivery(id=3, pickup=(9,9), dropoff=(7,1), weight=2.0, status="pending"),
            Delivery(id=4, pickup=(0,0), dropoff=(9,0), weight=1.0, status="pending"),
            Delivery(id=5, pickup=(5,5), dropoff=(4,4), weight=3.5, status="pending"),
            Delivery(id=6, pickup=(9,9), dropoff=(6,8), weight=2.5, status="pending"),
        ]
        
        sim_res = run_simulation(st.session_state.grid, fleet_res, deliveries)
        st.session_state.sim_result = sim_res
        
        log_container = st.empty()
        progress_bar = st.progress(0)
        
        # Animate the log
        log_html = "<div class='event-log-box'>"
        for i, event in enumerate(sim_res.event_log):
            c_class = "event-info"
            if event['level'] == 'success': c_class = 'event-success'
            elif event['level'] == 'alert': c_class = 'event-alert'
            elif event['level'] == 'warning': c_class = 'event-warning'
            
            log_html = f"<div class='{c_class}'>[{event['step']:02d}] {event['message']}</div>" + log_html
            
            log_container.markdown(log_html + "</div>", unsafe_allow_html=True)
            progress_bar.progress((i + 1) / len(sim_res.event_log))
            time.sleep(0.1) # Simulate real-time
            
        st.success("Simulation Complete!")

    if st.session_state.sim_result:
        st.markdown("#### Drone Status Board")
        d_cols = st.columns(4)
        for i, drone in enumerate(st.session_state.sim_result.drone_states):
            with d_cols[i % 4]:
                st.markdown(f"""
                <div class='card' style='font-size:0.8em;'>
                    <b>ID:</b> {drone.id} ({drone.type})<br>
                    <b>Pos:</b> {drone.position}<br>
                    <b>Bat:</b> {drone.battery:.1f}%<br>
                    <b>Stat:</b> {drone.status}
                </div>
                """, unsafe_allow_html=True)

# TAB 5: ML INTELLIGENCE
with tab5:
    st.markdown("### 🤖 ML Intelligence")
    ml_t1, ml_t2 = st.tabs(["Demand Forecasting (Regression)", "Anomaly Detection (Classification)"])
    
    # Load / Train ML
    if not os.path.exists("data/raw/bike_sharing_sample.csv"):
        generate_bike_sharing_data()
    if not os.path.exists("data/processed/anomaly_data.csv"):
        generate_anomaly_data()
        
    df_demand = pd.read_csv("data/raw/bike_sharing_sample.csv")
    df_anomaly = pd.read_csv("data/processed/anomaly_data.csv")
    
    with st.spinner("Training Models..."):
        demand_res = train_demand_model(df_demand)
        anomaly_res = train_anomaly_model(df_anomaly)
        
    with ml_t1:
        st.markdown("#### Demand Model Comparison")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Linear Regression RMSE", f"{demand_res.rmse_lr:.2f}")
        m_col2.metric("Random Forest RMSE", f"{demand_res.rmse_rf:.2f}")
        
        st.markdown("#### Interactive Predictor")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        temp = p_col1.slider("Temp", 0.0, 1.0, 0.5)
        hum = p_col2.slider("Humidity", 0.0, 1.0, 0.5)
        wind = p_col3.slider("Windspeed", 0.0, 1.0, 0.2)
        season = p_col4.selectbox("Season", [1,2,3,4])
        
        input_data = {"season": season, "holiday": 0, "workingday": 1, "weather": 1, "temp": temp, "humidity": hum, "windspeed": wind}
        pred = predict_demand(demand_res.best_model, input_data)
        st.success(f"Predicted Demand Count: **{pred:.0f}**")
        
    with ml_t2:
        st.markdown("#### Anomaly Model Comparison")
        a_col1, a_col2 = st.columns(2)
        a_col1.metric("Decision Tree Accuracy", f"{anomaly_res.acc_dt*100:.2f}%")
        a_col2.metric("Random Forest Accuracy", f"{anomaly_res.acc_rf*100:.2f}%")
        
        st.markdown("#### Confusion Matrix (Random Forest)")
        labels = ["Normal", "BattAnom", "RteAnom", "Spike"]
        fig_cm = px.imshow(anomaly_res.cm_rf, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues")
        fig_cm.update_layout(plot_bgcolor='#0A0E1A', paper_bgcolor='#0A0E1A', font_color='#E0E0E0')
        st.plotly_chart(fig_cm)
        
        st.markdown("#### Live Anomaly Checker")
        b_drop = st.slider("Battery Drop", 0.0, 100.0, 5.0)
        spd = st.slider("Speed", 0.0, 30.0, 15.0)
        r_dev = st.slider("Route Deviation", 0.0, 50.0, 2.0)
        a_chg = st.slider("Altitude Change", 0.0, 50.0, 1.0)
        s_chg = st.slider("Speed Change", 0.0, 50.0, 2.0)
        
        a_input = {"battery_drop": b_drop, "speed": spd, "route_deviation": r_dev, "altitude_change": a_chg, "speed_change": s_chg}
        a_pred = detect_anomaly(anomaly_res.best_model, a_input)
        st.info(f"Classification: **{labels[a_pred]}**")

# TAB 6: ANALYTICS REPORT
with tab6:
    st.markdown("### 📊 Analytics & Reporting")
    
    if st.session_state.sim_result:
        res = st.session_state.sim_result
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(names=['Completed', 'Delayed', 'Failed'], values=[res.completed, res.delayed, res.failed], hole=0.4)
            fig_pie.update_layout(plot_bgcolor='#0A0E1A', paper_bgcolor='#0A0E1A', font_color='#E0E0E0')
            st.plotly_chart(fig_pie)
            
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Simulation Summary")
            st.write(f"- **Total Deliveries Processed**: {len(res.delivery_states)}")
            st.write(f"- **Fleet Deployed**: {len(res.drone_states)} Drones")
            st.write(f"- **No-Fly Disruptions Triggered**: {len(res.no_fly_activations)}")
            st.write(f"- **Completed**: {res.completed}")
            st.write(f"- **Delayed**: {res.delayed}")
            st.write(f"- **Failed/Aborted**: {res.failed}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        report_text = f"""
        # AeroNet Lite - Final Report
        
        ## Simulation Results
        - Total Deliveries: {len(res.delivery_states)}
        - Completed: {res.completed}
        - Delayed: {res.delayed}
        - Failed: {res.failed}
        - No-Fly Disruptions: {len(res.no_fly_activations)}
        
        ## ML Results
        - Demand Best Model: {demand_res.best_name} (RMSE: {demand_res.rmse_rf:.2f})
        - Anomaly Best Model: {anomaly_res.best_name} (Acc: {anomaly_res.acc_rf*100:.2f}%)
        """
        st.download_button("Download Full Report", report_text, file_name="final_report.md")
    else:
        st.info("Run the simulation in Tab 4 to see analytics.")
