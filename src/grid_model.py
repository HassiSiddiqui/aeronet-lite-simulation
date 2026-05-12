from dataclasses import dataclass
from typing import List, Dict

GRID_SIZE = 10

ZONE_COLORS: Dict[str, str] = {
    "Residential": "#2ECC71",  # Green
    "Commercial": "#3498DB",   # Blue
    "Industrial": "#E74C3C",   # Red
    "Hospital": "#9B59B6",     # Purple
    "School": "#F1C40F",       # Yellow
    "Open Field": "#95A5A6"    # Gray
}

@dataclass
class GridCell:
    """
    Represents a single cell in the city grid.
    """
    row: int
    col: int
    zone: str
    density: float
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0

def create_city_grid() -> List[List[GridCell]]:
    """
    Creates a 10x10 city grid with predefined zones, hubs, charging pads,
    hospitals, and medical pickups.
    Designed so that some CSP constraints pass and at least one fails.
    """
    grid = []
    for r in range(GRID_SIZE):
        row_cells = []
        for c in range(GRID_SIZE):
            # Default zone is Open Field
            zone = "Open Field"
            density = 0.1
            
            # Populate with some generic zones
            if r < 3 and c < 3:
                zone = "Residential"
                density = 0.6
            elif r > 6 and c > 6:
                zone = "Residential"
                density = 0.8
            elif 3 <= r <= 6 and 3 <= c <= 6:
                zone = "Commercial"
                density = 0.9

            row_cells.append(GridCell(row=r, col=c, zone=zone, density=density))
        grid.append(row_cells)

    # Place specific hubs
    grid[0][0].is_hub = True
    grid[5][5].is_hub = True
    grid[9][9].is_hub = True

    # Place specific charging pads
    grid[0][1].is_charging = True
    grid[5][6].is_charging = True
    grid[9][8].is_charging = True

    # Place hospital and medical pickup
    grid[2][7].zone = "Hospital"
    grid[2][7].density = 0.8
    grid[2][8].is_medical_pickup = True
    grid[2][8].zone = "Open Field"

    # Intentional Constraint Violations for Demo
    
    # R1: Industrial cell cannot be adjacent to School/Hospital.
    # We will include one violation: Industrial at (2,6) next to Hospital at (2,7)
    grid[2][6].zone = "Industrial"
    grid[2][6].density = 0.3
    
    # We'll also put a safe Industrial cell
    grid[8][0].zone = "Industrial"
    grid[8][0].density = 0.4
    
    # Put a School somewhere safe
    grid[0][8].zone = "School"
    grid[0][8].density = 0.7

    # R2: Every Residential cell must be within 3 Manhattan distance of a Drone Hub.
    # Violation: Residential cell at (0, 9) is distance 9 from (0,0) and (9,9) and 9 from (5,5).
    grid[0][9].zone = "Residential"
    grid[0][9].density = 0.5
    
    # Populate some random demand
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c].zone == "Residential":
                grid[r][c].demand = 5.0
            elif grid[r][c].zone == "Commercial":
                grid[r][c].demand = 8.0
                
    return grid
