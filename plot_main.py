import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import time
import json
from pathlib import Path
from quantum_annealing_jss import QuantumAnnealingJSS
from genetic_algorithm_jss import GeneticAlgorithmJSS
from tabu_search_jss import TabuSearchJSS

class BenchmarkDataLoader:
    @staticmethod
    def load_small_dataset():
        """Load a small dataset for testing."""
        return [
            [(0, 2), (1, 3)],  # Job 1: Operation 1 on Machine 0 for 2 units, Operation 2 on Machine 1 for 3 units
            [(1, 2), (0, 1)]   # Job 2: Operation 1 on Machine 1 for 2 units, Operation 2 on Machine 0 for 1 unit
        ]

def create_problem_instance(benchmark_data):
    # Convert benchmark data to the problem instance format required by the algorithms
    problem_instance = []
    for job_id, operations in enumerate(benchmark_data):
        job_operations = []
        for op_idx, (machine, duration) in enumerate(operations):
            job_operations.append((machine, duration))
        problem_instance.append(job_operations)
    return problem_instance

def main(config):
    """Main function to run the comparison analysis."""
    # Load small dataset
    benchmark_data = BenchmarkDataLoader.load_small_dataset()

    # Create problem instance
    problem_instance = create_problem_instance(benchmark_data)

    # Solve using Quantum Annealing
    qa_solver = QuantumAnnealingJSS(problem_instance)
    qa_solution = qa_solver.solve(num_reads=config['qa_reads'])
    print("Quantum Annealing Solution:", qa_solution)

    # Solve using Genetic Algorithm
    ga_solver = GeneticAlgorithmJSS(problem_instance)
    ga_solution = ga_solver.solve(generations=config['ga_generations'], population_size=config['ga_population'])
    print("Genetic Algorithm Solution:", ga_solution)

    # Solve using Tabu Search
    ts_solver = TabuSearchJSS(problem_instance)
    ts_solution = ts_solver.solve(max_iterations=config['tabu_iterations'], tabu_tenure=config['tabu_tenure'])
    print("Tabu Search Solution:", ts_solution)

    # Generate report
    report = {
        'qa_solution': qa_solution,
        'ga_solution': ga_solution,
        'ts_solution': ts_solution
    }
    
    # Ensure the output directory exists
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Plot results
    plot_results(report, output_dir)

    return report

def plot_results(report, output_dir):
    # Extract data for plotting
    algorithms = ['Quantum Annealing', 'Genetic Algorithm', 'Tabu Search']
    makespans = [report['qa_solution']['makespan'], report['ga_solution']['makespan'], report['ts_solution']['makespan']]
    computation_times = [report['qa_solution']['computation_time'], report['ga_solution']['computation_time'], report['ts_solution']['computation_time']]
    qa_convergence = report['qa_solution']['convergence']
    ga_convergence = report['ga_solution']['convergence']
    ts_convergence = report['ts_solution']['convergence']

    # Plot makespan comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=algorithms, y=makespans)
    plt.title('Makespan Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Makespan')
    plt.savefig(output_dir / 'makespan_comparison.png')
    plt.close()

    # Plot computation time comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x=algorithms, y=computation_times)
    plt.title('Computation Time Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Computation Time (s)')
    plt.savefig(output_dir / 'computation_time_comparison.png')
    plt.close()

    # Plot convergence over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(qa_convergence, label='Quantum Annealing')
    plt.plot(ga_convergence, label='Genetic Algorithm')
    plt.plot(ts_convergence, label='Tabu Search')
    plt.title('Convergence Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Makespan')
    plt.legend()
    plt.savefig(output_dir / 'convergence_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Example configuration
    example_config = {
        'benchmark': 'small_dataset',
        'num_runs': 10,
        'algorithms': ['qa', 'ga', 'tabu'],
        'output_dir': 'results/test',
        'qa_reads': 100,
        'ga_generations': 10,
        'ga_population': 10,
        'tabu_iterations': 10,
        'tabu_tenure': 5
    }
    
    report = main(example_config)
    #print("\nComparison Report:")
    #print(json.dumps(report, indent=2))