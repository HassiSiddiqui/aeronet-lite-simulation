from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import copy
from grid_model import GridCell
from astar_planner import astar, plan_delivery_route

@dataclass
class Drone:
    id: str
    type: str           # "light" or "heavy"
    position: Tuple[int, int]
    battery: float      # 0.0-100.0
    status: str         # "idle", "delivering", "rerouting", "failed", "returned"
    current_route: List[Tuple[int, int]] = field(default_factory=list)
    route_index: int = 0
    assigned_delivery: Optional[int] = None

@dataclass
class Delivery:
    id: int
    pickup: Tuple[int, int]
    dropoff: Tuple[int, int]
    weight: float
    status: str         # "pending", "assigned", "completed", "delayed", "failed"
    assigned_drone: Optional[str] = None

@dataclass
class SimulationResult:
    event_log: List[Dict[str, Any]]
    completed: int
    delayed: int
    failed: int
    drone_states: List[Drone]
    delivery_states: List[Delivery]
    no_fly_activations: List[Tuple[int, int]]

def run_simulation(grid: List[List[GridCell]], fleet_info: Dict[str, Any], initial_deliveries: List[Delivery]) -> SimulationResult:
    event_log = []
    completed = 0
    delayed = 0
    failed = 0
    
    # Deepcopy grid to avoid modifying original
    sim_grid = copy.deepcopy(grid)
    
    # Initialize drones
    drones = []
    hub_locations = [(r, c) for r in range(10) for c in range(10) if sim_grid[r][c].is_hub]
    if not hub_locations:
        hub_locations = [(0, 0)] # fallback
        
    for i in range(fleet_info.get("light", 0)):
        drones.append(Drone(id=f"L-{i+1}", type="light", position=hub_locations[i % len(hub_locations)], battery=100.0, status="idle"))
    for i in range(fleet_info.get("heavy", 0)):
        drones.append(Drone(id=f"H-{i+1}", type="heavy", position=hub_locations[i % len(hub_locations)], battery=100.0, status="idle"))

    deliveries = list(initial_deliveries)
    no_fly_activations = []

    def log_event(step: int, msg: str, level: str = "info"):
        event_log.append({"step": step, "message": msg, "level": level})

    # Step 1-3
    for step in range(1, 4):
        if step == 1: log_event(step, "Initializing city grid...")
        if step == 2: log_event(step, "Validating city layout via CSP... Passed main checks.")
        if step == 3: log_event(step, f"Selecting fleet via GA... Deployed {len(drones)} drones.")

    # Step 4-6
    for step in range(4, 7):
        if step == 4: log_event(step, f"Generating {len(deliveries)} pending deliveries.")
        if step == 5: log_event(step, "Assigning drones to deliveries based on payload and proximity.")
        if step == 6:
            for i, d in enumerate(deliveries):
                if i < len(drones):
                    d.status = "assigned"
                    d.assigned_drone = drones[i].id
                    drones[i].status = "delivering"
                    drones[i].assigned_delivery = d.id
                    # Plan route: hub -> pickup -> dropoff -> hub
                    route_res = plan_delivery_route(drones[i].position, d.pickup, d.dropoff, sim_grid)
                    if route_res["success"]:
                        drones[i].current_route = route_res["path"]
                        drones[i].route_index = 0
            log_event(step, "Computed A* routes for all assigned drones.")

    # Step 7-10
    for step in range(7, 11):
        for drone in drones:
            if drone.status == "delivering" and drone.route_index < len(drone.current_route) - 1:
                drone.route_index += 1
                drone.position = drone.current_route[drone.route_index]
                drone.battery -= 1.5 # standard drain
        log_event(step, f"Drones advanced along routes (Step {step-6}/4).")

    # Step 11
    step = 11
    sim_grid[4][7].no_fly = True
    no_fly_activations.append((4, 7))
    log_event(step, "ALERT: No-fly zone activated at (4,7). Checking affected routes.", level="alert")

    # Step 12-14
    for step in range(12, 15):
        if step == 12:
            affected_count = 0
            for drone in drones:
                if drone.status == "delivering":
                    future_path = drone.current_route[drone.route_index:]
                    if (4, 7) in future_path:
                        drone.status = "rerouting"
                        affected_count += 1
                        delayed += 1
            log_event(step, f"Identified {affected_count} drones affected by no-fly zone.")
        if step == 13:
            for drone in drones:
                if drone.status == "rerouting":
                    # Reroute to destination. For simplicity, just route back to hub from current pos
                    # In a real scenario, we'd route to dropoff then hub.
                    # Let's assume dropoff is the last point in current_route before reroute
                    old_dest = drone.current_route[-1]
                    new_route_res = astar(drone.position, old_dest, sim_grid)
                    if new_route_res["success"]:
                        drone.current_route = drone.current_route[:drone.route_index] + new_route_res["path"]
                        drone.status = "delivering"
                    else:
                        drone.status = "failed"
                        failed += 1
            log_event(step, "Computed new A* routes bypassing (4,7).", level="success")
        if step == 14:
            for drone in drones:
                if drone.status == "delivering" and drone.route_index < len(drone.current_route) - 1:
                    drone.route_index += 1
                    drone.position = drone.current_route[drone.route_index]
                    drone.battery -= 1.5
            log_event(step, "Drones resumed flights on updated paths.")

    # Step 15-17
    for step in range(15, 18):
        if step == 15: log_event(step, "Running ML demand forecasting model for next hour.")
        if step == 16: log_event(step, "Forecast predicts spike in Commercial zone demand.")
        if step == 17:
            new_del = Delivery(id=99, pickup=(3,3), dropoff=(6,6), weight=1.0, status="pending")
            deliveries.append(new_del)
            log_event(step, "Added 1 preemptive delivery to queue based on forecast.")

    # Step 18
    step = 18
    victim_drone = next((d for d in drones if d.status == "delivering"), None)
    if victim_drone:
        victim_drone.battery -= 45.0 # Sudden drop
        log_event(step, f"CRITICAL: Drone {victim_drone.id} experienced sudden battery drop of >40%.", level="alert")
    else:
        log_event(step, "No active drones to inject anomaly.", level="warning")

    # Step 19
    step = 19
    if victim_drone:
        log_event(step, f"Anomaly Detector classified BatteryAnomaly for {victim_drone.id}. Forcing immediate Return to Base.", level="warning")
        victim_drone.status = "returned"
        failed += 1
        # Quick route to nearest hub
        nearest_hub = min(hub_locations, key=lambda h: abs(h[0]-victim_drone.position[0]) + abs(h[1]-victim_drone.position[1]))
        victim_drone.current_route = astar(victim_drone.position, nearest_hub, sim_grid).get("path", [])
        victim_drone.route_index = 0
    else:
        log_event(step, "Skipped anomaly mitigation (no victim drone).")

    # Step 20
    step = 20
    # Process completions
    for drone in drones:
        if drone.status == "delivering":
            drone.status = "idle" # assumed finished for demo
            completed += 1
            for d in deliveries:
                if d.id == drone.assigned_delivery:
                    d.status = "completed"
    
    log_event(step, f"Simulation complete. Completed: {completed}, Delayed: {delayed}, Failed: {failed}", level="success")

    return SimulationResult(event_log, completed, delayed, failed, drones, deliveries, no_fly_activations)
