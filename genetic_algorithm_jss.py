# genetic_algorithm_jss.py
import random
import numpy as np
import time

class GeneticAlgorithmJSS:
    def __init__(self, jobs_data, population_size=100):
        """
        Initialize Genetic Algorithm solver for JSS
        jobs_data: List of jobs, where each job is a list of (machine_id, processing_time) tuples
        """
        self.jobs_data = jobs_data
        self.num_jobs = len(jobs_data)
        self.num_machines = max(machine for job in jobs_data 
                              for machine, _ in job) + 1
        self.population_size = population_size
        
    def _create_individual(self):
        """Create a random individual (schedule)."""
        # Create operation sequence for each job
        operations = []
        for job_id in range(self.num_jobs):
            for op_idx in range(len(self.jobs_data[job_id])):
                operations.append((job_id, op_idx))
        
        # Randomly shuffle operations while maintaining precedence constraints
        random.shuffle(operations)
        return operations
    
    def _decode_schedule(self, chromosome):
        """Decode chromosome into actual schedule with start times."""
        machine_timeframes = [0] * self.num_machines
        job_timeframes = [0] * self.num_jobs
        schedule = {}
        
        for job_id, op_idx in chromosome:
            machine, proc_time = self.jobs_data[job_id][op_idx]
            
            # Earliest possible start time considering both machine and job constraints
            start_time = max(machine_timeframes[machine], job_timeframes[job_id])
            
            # Add operation to schedule
            if job_id not in schedule:
                schedule[job_id] = []
            schedule[job_id].append((op_idx, start_time))
            
            # Update timeframes
            machine_timeframes[machine] = start_time + proc_time
            job_timeframes[job_id] = start_time + proc_time
            
        return schedule, max(machine_timeframes)
    
    def _crossover(self, parent1, parent2):
        """Perform precedence preserving crossover."""
        crossover_point = random.randint(0, len(parent1) - 1)
        child = parent1[:crossover_point]
        
        # Add remaining operations from parent2 while preserving precedence
        remaining_ops = [op for op in parent2 if op not in child]
        child.extend(remaining_ops)
        
        return child
    
    def _mutate(self, individual, mutation_rate=0.1):
        """Perform swap mutation while preserving precedence constraints."""
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual
    
    def solve(self, generations=100):
        """Solve JSS using genetic algorithm."""
        start_time = time.time()
        
        # Initialize population
        population = [self._create_individual() for _ in range(self.population_size)]
        best_makespan = float('inf')
        best_schedule = None
        makespan_history = []
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                schedule, makespan = self._decode_schedule(individual)
                fitness_scores.append(1.0 / makespan)
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = schedule
            
            makespan_history.append(best_makespan)
            
            # Selection
            selected = []
            for _ in range(self.population_size):
                tournament = random.sample(list(enumerate(fitness_scores)), 3)
                winner_idx = max(tournament, key=lambda x: x[1])[0]
                selected.append(population[winner_idx])
            
            # Crossover and Mutation
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, self.population_size - 1)]
                
                child1 = self._crossover(parent1, parent2)
                child2 = self._crossover(parent2, parent1)
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        computation_time = time.time() - start_time
        
        return {
            'schedule': best_schedule,
            'makespan': best_makespan,
            'computation_time': computation_time,
            'convergence': makespan_history,
            'generations': generations
        }