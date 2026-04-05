"""
CODE APPENDIX — Delivery Route Optimization (TSP)
Optimization Methods Programming Assignment
Authors: Chamod (ILP) & Hirantha (GA)
"""

# ---- SHARED: Dataset and Distance Matrix ----

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json

# Kaggle dataset: "Travelling Salesman Problem: Visit 49 UK Cities"
# https://www.kaggle.com/datasets/patricklford/travelling-salesman-problem
# Selecting 15 cities for computational feasibility

df = pd.read_csv('data/UK_Cities.csv')

selected_cities = [
    'London', 'Birmingham', 'Leeds', 'Glasgow', 'Sheffield',
    'Manchester', 'Liverpool', 'Edinburgh', 'Bristol', 'Cardiff',
    'Leicester', 'Nottingham', 'Newcastle upon Tyne', 'Southampton', 'Plymouth'
]
df = df[df['City'].isin(selected_cities)].reset_index(drop=True)

city_names = df['City'].tolist()
coords = df[['Latitude', 'Longitude']].values
n = len(city_names)

# Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            distance_matrix[i][j] = haversine(
                coords[i][0], coords[i][1], coords[j][0], coords[j][1]
            )


# ---- PART 1: ILP EXACT METHOD (Chamod) ----

from pulp import *

def solve_tsp_ilp(distance_matrix, city_names):
    """Solve TSP with ILP + MTZ subtour elimination."""
    n = len(city_names)
    prob = LpProblem("TSP_ILP", LpMinimize)

    x = LpVariable.dicts("x",
        ((i, j) for i in range(n) for j in range(n) if i != j),
        cat='Binary')
    u = LpVariable.dicts("u",
        (i for i in range(1, n)),
        lowBound=1, upBound=n-1, cat='Integer')

    prob += lpSum(distance_matrix[i][j] * x[(i, j)]
                  for i in range(n) for j in range(n) if i != j)

    for j in range(n):
        prob += lpSum(x[(i, j)] for i in range(n) if i != j) == 1
    for i in range(n):
        prob += lpSum(x[(i, j)] for j in range(n) if i != j) == 1

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1

    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=0))
    solve_time = time.time() - start_time

    tour = [0]
    current = 0
    for _ in range(n - 1):
        for j in range(n):
            if j != current and value(x[(current, j)]) == 1:
                tour.append(j)
                current = j
                break
    tour.append(0)

    return {
        'tour': tour,
        'tour_names': [city_names[i] for i in tour],
        'total_distance': value(prob.objective),
        'solve_time': solve_time,
        'status': LpStatus[prob.status]
    }


print("=" * 60)
print("PART 1: ILP EXACT METHOD (Chamod)")
print("=" * 60)
ilp_result = solve_tsp_ilp(distance_matrix, city_names)
print(f"Status: {ilp_result['status']}")
print(f"Optimal Tour: {' -> '.join(ilp_result['tour_names'])}")
print(f"Total Distance: {ilp_result['total_distance']:.2f} km")
print(f"Solve Time: {ilp_result['solve_time']:.4f} seconds")


# ---- PART 2: GENETIC ALGORITHM (Hirantha) ----

from deap import base, creator, tools, algorithms
import random

def evaluate_tsp(individual, dist_matrix):
    route = [0] + [x + 1 for x in individual] + [0]
    total = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    return (total,)

def solve_tsp_ga(dist_matrix, city_names, pop_size=200, n_gen=500,
                 cx_prob=0.8, mut_prob=0.2, seed=42):
    """Solve TSP with Genetic Algorithm using DEAP."""
    random.seed(seed)
    np.random.seed(seed)
    n = len(city_names)

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(n - 1), n - 1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_tsp, dist_matrix=dist_matrix)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

    pop = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    hof = tools.HallOfFame(1)

    start_time = time.time()
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob,
                                        ngen=n_gen, stats=stats, halloffame=hof,
                                        verbose=False)
    solve_time = time.time() - start_time

    best = hof[0]
    best_route = [0] + [x + 1 for x in best] + [0]

    return {
        'tour': best_route,
        'tour_names': [city_names[i] for i in best_route],
        'total_distance': best.fitness.values[0],
        'solve_time': solve_time,
        'logbook': logbook
    }


print("\n" + "=" * 60)
print("PART 2: GENETIC ALGORITHM (Hirantha)")
print("=" * 60)
ga_result = solve_tsp_ga(distance_matrix, city_names)
print(f"Best Tour: {' -> '.join(ga_result['tour_names'])}")
print(f"Total Distance: {ga_result['total_distance']:.2f} km")
print(f"Solve Time: {ga_result['solve_time']:.4f} seconds")


# ---- PART 3: COMPARISON ----

print("\n" + "=" * 60)
print("PART 3: COMPARISON")
print("=" * 60)

gap = ((ga_result['total_distance'] - ilp_result['total_distance'])
       / ilp_result['total_distance'] * 100)

print(f"\n{'Metric':<25} {'ILP (Exact)':<20} {'GA (Metaheuristic)':<20}")
print("-" * 65)
print(f"{'Distance (km)':<25} {ilp_result['total_distance']:<20.2f} {ga_result['total_distance']:<20.2f}")
print(f"{'Solve Time (s)':<25} {ilp_result['solve_time']:<20.4f} {ga_result['solve_time']:<20.4f}")
print(f"{'Optimality':<25} {'Guaranteed':<20} {'Near-optimal':<20}")
print(f"{'Optimality Gap':<25} {'0%':<20} {f'{gap:.2f}%':<20}")
print(f"{'Scalability':<25} {'Exponential':<20} {'Polynomial':<20}")
