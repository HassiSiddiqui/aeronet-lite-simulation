from typing import List, Tuple, Dict, Any
from grid_model import GRID_SIZE, GridCell

def get_neighbors(row: int, col: int, grid_size: int = GRID_SIZE) -> List[Tuple[int, int]]:
    """Returns valid adjacent neighbors (up, down, left, right)."""
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < grid_size and 0 <= c < grid_size:
            neighbors.append((r, c))
    return neighbors

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Calculates Manhattan distance between two coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def check_r1_industrial_safety(grid: List[List[GridCell]]) -> List[str]:
    """R1: Industrial cells cannot be directly adjacent to Schools or Hospitals."""
    violations = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c].zone == "Industrial":
                neighbors = get_neighbors(r, c)
                for nr, nc in neighbors:
                    neighbor_zone = grid[nr][nc].zone
                    if neighbor_zone in ["School", "Hospital"]:
                        violations.append(f"Industrial at ({r},{c}) is adjacent to {neighbor_zone} at ({nr},{nc})")
    return violations

def check_r2_residential_coverage(grid: List[List[GridCell]]) -> List[str]:
    """R2: Every Residential cell must be within 3 Manhattan distance of a Drone Hub."""
    violations = []
    hubs = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c].is_hub]
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c].zone == "Residential":
                distances = [manhattan((r, c), hub) for hub in hubs]
                if not distances or min(distances) > 3:
                    violations.append(f"Residential at ({r},{c}) is out of hub range (min dist: {min(distances) if distances else 'N/A'})")
    return violations

def check_r3_hub_charging(grid: List[List[GridCell]]) -> List[str]:
    """R3: Every Drone Hub must have a Charging Pad within 2 Manhattan distance."""
    violations = []
    pads = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c].is_charging]
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c].is_hub:
                distances = [manhattan((r, c), pad) for pad in pads]
                if not distances or min(distances) > 2:
                    violations.append(f"Hub at ({r},{c}) lacks nearby charging pad (min dist: {min(distances) if distances else 'N/A'})")
    return violations

def check_r4_medical_access(grid: List[List[GridCell]]) -> List[str]:
    """R4: At least one Hospital must have a Medical Pickup within 1 Manhattan distance."""
    hospitals = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c].zone == "Hospital"]
    pickups = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c].is_medical_pickup]
    
    if not hospitals:
        return ["No hospitals found in the grid."]
    if not pickups:
        return ["No medical pickups found in the grid."]
        
    for hr, hc in hospitals:
        for pr, pc in pickups:
            if manhattan((hr, hc), (pr, pc)) <= 1:
                return [] # Requirement met

    return ["No hospital has a medical pickup within distance 1."]

def validate_layout(grid: List[List[GridCell]]) -> Dict[str, Any]:
    """
    Validates the layout against all CSP rules.
    Returns {"passed": [...], "failed": [...], "violations": [...], "suggestions": [...]}
    """
    passed = []
    failed = []
    all_violations = []
    suggestions = []

    # Rule 1
    v1 = check_r1_industrial_safety(grid)
    if v1:
        failed.append("R1")
        all_violations.extend(v1)
        suggestions.append("Relocate Industrial zones away from Schools/Hospitals.")
    else:
        passed.append("R1")

    # Rule 2
    v2 = check_r2_residential_coverage(grid)
    if v2:
        failed.append("R2")
        all_violations.extend(v2)
        suggestions.append("Add a new Drone Hub closer to isolated Residential areas.")
    else:
        passed.append("R2")

    # Rule 3
    v3 = check_r3_hub_charging(grid)
    if v3:
        failed.append("R3")
        all_violations.extend(v3)
        suggestions.append("Place Charging Pads within 2 cells of every Hub.")
    else:
        passed.append("R3")

    # Rule 4
    v4 = check_r4_medical_access(grid)
    if v4:
        failed.append("R4")
        all_violations.extend(v4)
        suggestions.append("Add a Medical Pickup directly adjacent to at least one Hospital.")
    else:
        passed.append("R4")

    return {
        "passed": passed,
        "failed": failed,
        "violations": all_violations,
        "suggestions": suggestions
    }
