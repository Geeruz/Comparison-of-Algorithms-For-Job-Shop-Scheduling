# Install required libraries
!pip install qiskit qiskit-aer jobshop numpy matplotlib

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from jobshop import load_instance
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.optimizers import COBYLA
import time

# Tabu Search Implementation
def tabu_search(problem, max_iter=100, tabu_size=10):
    def initial_solution():
        # Generate a random feasible schedule
        schedule = []
        for job in problem.jobs:
            shuffled_job = job.copy()
            np.random.shuffle(shuffled_job)
            schedule.append(shuffled_job)
        return schedule

    def get_neighbors(schedule):
        # Generate neighboring schedules by swapping two operations in a job
        neighbors = []
        for i in range(len(schedule)):
            for j in range(len(schedule[i]) - 1):
                neighbor = [job.copy() for job in schedule]
                neighbor[i][j], neighbor[i][j + 1] = neighbor[i][j + 1], neighbor[i][j]
                neighbors.append(neighbor)
        return neighbors

    def evaluate(schedule):
        # Calculate the makespan (total completion time) of the schedule
        machine_times = [0] * problem.num_machines
        for job in schedule:
            for operation in job:
                machine, duration = operation
                machine_times[machine] += duration
        return max(machine_times)

    current_solution = initial_solution()
    best_solution = current_solution
    tabu_list = []

    for iteration in range(max_iter):
        neighbors = get_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_cost = float('inf')

        for neighbor in neighbors:
            if neighbor not in tabu_list:
                cost = evaluate(neighbor)
                if cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost

        if best_neighbor is None:
            break

        current_solution = best_neighbor
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        if best_neighbor_cost < evaluate(best_solution):
            best_solution = best_neighbor

    return best_solution, evaluate(best_solution)

# QAOA Implementation
def run_qaoa(cost_hamiltonian):
    optimizer = COBYLA()
    qaoa = QAOA(optimizer=optimizer, quantum_instance=Aer.get_backend('statevector_simulator'))
    result = qaoa.compute_minimum_eigenvalue(cost_hamiltonian)
    return result.eigenvalue.real

# Load JSSP Instance using jobshop library
instance = load_instance("taillard", "tai20_15.txt")  # Example instance
print("Loaded JSSP instance with", len(instance.jobs), "jobs and", instance.num_machines, "machines.")

# Benchmark Tabu Search
start_time = time.time()
best_schedule, best_makespan = tabu_search(instance)
tabu_time = time.time() - start_time
print("Tabu Search - Best Makespan:", best_makespan, "Time:", tabu_time)

# Benchmark QAOA
# Define a simple cost Hamiltonian for QAOA (example)
cost_hamiltonian = PauliSumOp.from_list([("ZZ", 1), ("IZ", -0.5), ("ZI", -0.5)])

start_time = time.time()
qaoa_makespan = run_qaoa(cost_hamiltonian)
qaoa_time = time.time() - start_time
print("QAOA - Best Makespan:", qaoa_makespan, "Time:", qaoa_time)

# Compare Results
algorithms = ["Tabu Search", "QAOA"]
makespans = [best_makespan, qaoa_makespan]
times = [tabu_time, qaoa_time]

# Plot Makespan Comparison
plt.bar(algorithms, makespans)
plt.title("Makespan Comparison")
plt.ylabel("Makespan")
plt.show()

# Plot Computation Time Comparison
plt.bar(algorithms, times)
plt.title("Computation Time Comparison")
plt.ylabel("Time (seconds)")
plt.show()