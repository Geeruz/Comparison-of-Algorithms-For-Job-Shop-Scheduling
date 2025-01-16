# comparison_visualization.py
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
    def load_ft06():
        """Load Fisher and Thompson's 6x6 benchmark."""
        return [
            [(2,1), (0,3), (1,6), (3,7), (5,3), (4,6)],
            [(1,8), (2,5), (4,10), (5,10), (0,10), (3,4)],
            [(2,5), (3,4), (5,8), (0,9), (1,1), (4,7)],
            [(1,5), (0,5), (2,5), (3,3), (4,8), (5,9)],
            [(2,9), (1,3), (4,5), (5,4), (0,3), (3,1)],
            [(1,3), (3,3), (5,9), (0,10), (4,4), (2,1)]
        ]
    
    @staticmethod
    def load_ft10():
        """Load Fisher and Thompson's 10x10 benchmark."""
        # Add FT10 benchmark data here
        pass
    
    @staticmethod
    def load_la01():
        """Load Lawrence's LA01 benchmark."""
        # Add LA01 benchmark data here
        pass
    
    @classmethod
    def load_benchmark(cls, name):
        """Load benchmark by name."""
        loaders = {
            'ft06': cls.load_ft06,
            'ft10': cls.load_ft10,
            'la01': cls.load_la01
        }
        return loaders[name]()

class AlgorithmRunner:
    def __init__(self, jobs_data, config):
        self.jobs_data = jobs_data
        self.config = config
        self.solvers = {
            'qa': self._run_quantum_annealing,
            'ga': self._run_genetic_algorithm,
            'tabu': self._run_tabu_search
        }
    
    def _run_quantum_annealing(self):
        solver = QuantumAnnealingJSS(self.jobs_data)
        start_memory = psutil.Process().memory_info().rss
        result = solver.solve(num_reads=self.config.get('qa_reads', 1000))
        end_memory = psutil.Process().memory_info().rss
        result['memory'] = end_memory - start_memory
        return result
    
    def _run_genetic_algorithm(self):
        solver = GeneticAlgorithmJSS(self.jobs_data, 
                                   population_size=self.config.get('ga_population', 100))
        start_memory = psutil.Process().memory_info().rss
        result = solver.solve(generations=self.config.get('ga_generations', 100))
        end_memory = psutil.Process().memory_info().rss
        result['memory'] = end_memory - start_memory
        return result
    
    def _run_tabu_search(self):
        solver = TabuSearchJSS(self.jobs_data)
        start_memory = psutil.Process().memory_info().rss
        result = solver.solve(max_iterations=self.config.get('tabu_iterations', 1000),
                            tabu_tenure=self.config.get('tabu_tenure', 10))
        end_memory = psutil.Process().memory_info().rss
        result['memory'] = end_memory - start_memory
        return result
    
    def run_algorithm(self, algo_name):
        """Run specified algorithm and return results."""
        return self.solvers[algo_name]()

class ResultsVisualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_plot(self, plot, name):
        """Save plot to output directory."""
        plot.savefig(self.output_dir / f'{name}.png')
        plot.close()
    
    def plot_makespan_comparison(self, results):
        plt.figure(figsize=(10, 6))
        data = [results[algo]['makespans'] for algo in results]
        plt.boxplot(data, labels=[algo.upper() for algo in results])
        plt.title('Makespan Comparison')
        plt.ylabel('Makespan')
        plt.grid(True)
        return plt
    
    def plot_computation_time(self, results):
        plt.figure(figsize=(10, 6))
        data = [results[algo]['times'] for algo in results]
        plt.boxplot(data, labels=[algo.upper() for algo in results])
        plt.title('Computation Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')
        plt.grid(True)
        return plt
    
    def plot_convergence_comparison(self, results):
        plt.figure(figsize=(12, 6))
        
        for algo, data in results.items():
            if 'convergence' in data:
                convergence = np.mean(data['convergence'], axis=0)
                plt.plot(convergence, label=algo.upper(), alpha=0.8)
        
        if 'qa' in results:
            qa_avg = np.mean(results['qa']['makespans'])
            plt.axhline(y=qa_avg, color='r', linestyle='--', 
                       label='Quantum Annealing', alpha=0.8)
        
        plt.title('Convergence Rate Comparison')
        plt.xlabel('Iteration')
        plt.ylabel('Makespan')
        plt.legend()
        plt.grid(True)
        return plt
    
    def plot_solution_distribution(self, results):
        plt.figure(figsize=(12, 6))
        data = {algo.upper(): data['makespans'] 
                for algo, data in results.items()}
        df = pd.DataFrame(data)
        sns.kdeplot(data=df, fill=True)
        plt.title('Solution Quality Distribution')
        plt.xlabel('Makespan')
        plt.ylabel('Density')
        plt.grid(True)
        return plt
    
    def plot_resource_utilization(self, results):
        plt.figure(figsize=(10, 6))
        memory_usage = {algo.upper(): np.mean(data['memory']) / 1024 / 1024 
                       for algo, data in results.items()}
        plt.bar(memory_usage.keys(), memory_usage.values())
        plt.title('Average Memory Usage')
        plt.ylabel('Memory (MB)')
        plt.xticks(rotation=45)
        plt.grid(True)
        return plt

class ComparisonAnalysis:
    def __init__(self, jobs_data, config):
        self.jobs_data = jobs_data
        self.config = config
        self.algorithm_runner = AlgorithmRunner(jobs_data, config)
        self.results = {}
    
    def run_comparisons(self):
        """Run all specified algorithms multiple times."""
        for algo in self.config['algorithms']:
            self.results[algo] = {
                'makespans': [],
                'times': [],
                'convergence': [],
                'memory': []
            }
            
            for _ in range(self.config['num_runs']):
                result = self.algorithm_runner.run_algorithm(algo)
                self.results[algo]['makespans'].append(result['makespan'])
                self.results[algo]['times'].append(result['computation_time'])
                self.results[algo]['memory'].append(result['memory'])
                if 'convergence' in result:
                    self.results[algo]['convergence'].append(result['convergence'])
    
    def generate_report(self):
        """Generate comprehensive comparison report."""
        report = {
            'Makespan Statistics': {},
            'Computation Time Statistics': {},
            'Memory Usage (MB)': {}
        }
        
        for algo, data in self.results.items():
            report['Makespan Statistics'][algo] = {
                'mean': np.mean(data['makespans']),
                'std': np.std(data['makespans']),
                'min': np.min(data['makespans']),
                'max': np.max(data['makespans'])
            }
            report['Computation Time Statistics'][algo] = np.mean(data['times'])
            report['Memory Usage (MB)'][algo] = np.mean(data['memory']) / 1024 / 1024
        
        return report

def main(config):
    """Main function to run the comparison analysis."""
    # Load benchmark data
    jobs_data = BenchmarkDataLoader.load_benchmark(config['benchmark'])
    
    # Initialize analysis
    analysis = ComparisonAnalysis(jobs_data, config)
    
    # Run comparisons
    analysis.run_comparisons()
    
    # Generate visualizations
    visualizer = ResultsVisualizer(config['output_dir'])
    
    # Create and save plots
    plots = {
        'makespan': visualizer.plot_makespan_comparison(analysis.results),
        'computation_time': visualizer.plot_computation_time(analysis.results),
        'convergence': visualizer.plot_convergence_comparison(analysis.results),
        'solution_distribution': visualizer.plot_solution_distribution(analysis.results),
        'resource_utilization': visualizer.plot_resource_utilization(analysis.results)
    }
    
    # Save plots
    for name, plot in plots.items():
        visualizer.save_plot(plot, name)
    
    # Generate and save report
    report = analysis.generate_report()
    with open(Path(config['output_dir']) / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report, analysis.results

if __name__ == "__main__":
    # Example configuration
    example_config = {
        'benchmark': 'ft06',
        'num_runs': 10,
        'algorithms': ['qa', 'ga', 'tabu'],
        'output_dir': 'results/test',
        'qa_reads': 1000,
        'ga_generations': 100,
        'ga_population': 100,
        'tabu_iterations': 1000,
        'tabu_tenure': 10
    }
    
    report, results = main(example_config)
    print("\nComparison Report:")
    print(json.dumps(report, indent=2))