import random
from typing import List, Dict, Any
from grid_model import GRID_SIZE, GridCell

# Drone constants
LIGHT_COST = 1000
LIGHT_PAYLOAD = 2
LIGHT_RANGE = 12

HEAVY_COST = 1800
HEAVY_PAYLOAD = 5
HEAVY_RANGE = 20

def compute_coverage(light: int, heavy: int, grid: List[List[GridCell]]) -> float:
    """
    Simulates coverage based on drone counts.
    A simplified coverage model: Each light drone covers ~5 cells of demand, 
    heavy covers ~10 cells, capped at total demand cells.
    """
    total_demand_cells = sum(1 for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c].demand > 0)
    if total_demand_cells == 0:
        return 1.0

    capacity = (light * 5) + (heavy * 10)
    coverage = min(capacity / total_demand_cells, 1.0)
    return coverage

def fitness(chromosome: List[int], grid: List[List[GridCell]], budget: float = 15000) -> float:
    """
    Fitness: 0.75 * coverage_pct - 0.25 * budget_used_pct
    Penalty if budget exceeded.
    """
    light, heavy = chromosome
    cost = light * LIGHT_COST + heavy * HEAVY_COST
    
    if cost > budget:
        return -1.0  # Invalid solution

    coverage_pct = compute_coverage(light, heavy, grid)
    budget_used_pct = cost / budget
    
    return 0.75 * coverage_pct - 0.25 * budget_used_pct

def generate_population(pop_size: int) -> List[List[int]]:
    return [[random.randint(0, 10), random.randint(0, 10)] for _ in range(pop_size)]

def tournament_selection(population: List[List[int]], fitnesses: List[float], k: int = 3) -> List[int]:
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]

def crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """Single-point crossover for length 2 chromosome means we just swap one gene."""
    return [parent1[0], parent2[1]]

def mutate(chromosome: List[int]) -> List[int]:
    """Random +/-1 to one gene."""
    mutated = chromosome.copy()
    idx = random.randint(0, 1)
    change = random.choice([-1, 1])
    mutated[idx] = max(0, min(10, mutated[idx] + change))
    return mutated

def run_ga_fleet_selection(grid: List[List[GridCell]], budget: float = 15000) -> Dict[str, Any]:
    """
    Runs Genetic Algorithm to find the optimal fleet.
    Returns best fleet + fitness history.
    """
    pop_size = 50
    generations = 100
    
    population = generate_population(pop_size)
    fitness_history = []
    
    best_overall = None
    best_fitness = -float('inf')
    
    for gen in range(generations):
        fitnesses = [fitness(ind, grid, budget) for ind in population]
        
        # Track best
        max_fit = max(fitnesses)
        best_idx = fitnesses.index(max_fit)
        
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_overall = population[best_idx]
            
        fitness_history.append(max_fit)
        
        new_population = []
        # Elitism
        new_population.append(best_overall)
        
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            
            child = crossover(p1, p2)
            if random.random() < 0.2: # 20% mutation rate
                child = mutate(child)
            new_population.append(child)
            
        population = new_population

    final_cost = best_overall[0] * LIGHT_COST + best_overall[1] * HEAVY_COST
    final_coverage = compute_coverage(best_overall[0], best_overall[1], grid)
    
    return {
        "light": best_overall[0],
        "heavy": best_overall[1],
        "cost": final_cost,
        "coverage": final_coverage * 100,
        "fitness_history": fitness_history
    }

def brute_force_fleet(grid: List[List[GridCell]], budget: float = 15000) -> Dict[str, Any]:
    """Backup option using brute force."""
    best_fleet = [0, 0]
    best_fitness = -float('inf')
    
    for l in range(11):
        for h in range(11):
            fit = fitness([l, h], grid, budget)
            if fit > best_fitness:
                best_fitness = fit
                best_fleet = [l, h]
                
    final_cost = best_fleet[0] * LIGHT_COST + best_fleet[1] * HEAVY_COST
    final_coverage = compute_coverage(best_fleet[0], best_fleet[1], grid)
    
    return {
        "light": best_fleet[0],
        "heavy": best_fleet[1],
        "cost": final_cost,
        "coverage": final_coverage * 100,
        "fitness_history": []
    }
