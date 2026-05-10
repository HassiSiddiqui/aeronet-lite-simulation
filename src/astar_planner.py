import heapq
from typing import List, Tuple, Dict, Any
from grid_model import GRID_SIZE, GridCell

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_valid_neighbors(pos: Tuple[int, int], grid: List[List[GridCell]]) -> List[Tuple[int, int]]:
    r, c = pos
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            if not grid[nr][nc].no_fly:
                neighbors.append((nr, nc))
    return neighbors

def astar(start: Tuple[int, int], goal: Tuple[int, int], grid: List[List[GridCell]]) -> Dict[str, Any]:
    """A* search on the 10x10 grid."""
    if not (0 <= start[0] < GRID_SIZE and 0 <= start[1] < GRID_SIZE):
        return {"path": [], "cost": 0.0, "success": False, "message": "Invalid start"}
    if not (0 <= goal[0] < GRID_SIZE and 0 <= goal[1] < GRID_SIZE):
        return {"path": [], "cost": 0.0, "success": False, "message": "Invalid goal"}
    
    if grid[goal[0]][goal[1]].no_fly:
        return {"path": [], "cost": 0.0, "success": False, "message": "Goal is in no-fly zone"}
    if grid[start[0]][start[1]].no_fly:
        return {"path": [], "cost": 0.0, "success": False, "message": "Start is in no-fly zone"}

    open_set = []
    heapq.heappush(open_set, (0.0, start))
    
    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return {"path": path, "cost": g_score[goal], "success": True, "message": "Path found"}
            
        for neighbor in get_valid_neighbors(current, grid):
            # Move cost: 1.0 normal, 0.8 for Commercial
            nr, nc = neighbor
            move_cost = 0.8 if grid[nr][nc].zone == "Commercial" else 1.0
            
            tentative_g_score = g_score[current] + move_cost
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                
    return {"path": [], "cost": 0.0, "success": False, "message": "No path found"}

def plan_delivery_route(hub: Tuple[int, int], pickup: Tuple[int, int], dropoff: Tuple[int, int], grid: List[List[GridCell]]) -> Dict[str, Any]:
    """hub -> pickup -> dropoff -> hub"""
    leg1 = astar(hub, pickup, grid)
    leg2 = astar(pickup, dropoff, grid)
    leg3 = astar(dropoff, hub, grid)
    
    if not (leg1["success"] and leg2["success"] and leg3["success"]):
        return {"path": [], "cost": 0.0, "success": False, "message": "Could not plan full route"}
        
    # Combine paths, removing duplicate waypoints at junctions
    full_path = leg1["path"][:-1] + leg2["path"][:-1] + leg3["path"]
    total_cost = leg1["cost"] + leg2["cost"] + leg3["cost"]
    
    return {"path": full_path, "cost": total_cost, "success": True, "message": "Full route planned"}
