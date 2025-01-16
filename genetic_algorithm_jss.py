import random
import time

class GeneticAlgorithmJSS:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance
        self.num_jobs = len(problem_instance)
        self.num_machines = max(machine for job in problem_instance for machine, _ in job) + 1

    def solve(self, generations=100, population_size=50):
        self.generations = generations
        self.population_size = population_size
        start_time = time.time()
        population = self._initialize_population()
        best_schedule = None
        best_makespan = float('inf')
        makespan_history = []
        for generation in range(generations):
            fitness_scores = [self._evaluate(individual) for individual in population]
            best_idx = fitness_scores.index(min(fitness_scores))
            best_makespan = fitness_scores[best_idx]
            best_schedule = population[best_idx]
            makespan_history.append(best_makespan)
            selected = self._select_parents(population, fitness_scores)
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, population_size - 1)]
                child1 = self._crossover(parent1, parent2)
                child2 = self._crossover(parent2, parent1)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            population = new_population[:population_size]
        computation_time = time.time() - start_time
        return {
            'schedule': best_schedule,
            'makespan': best_makespan,
            'computation_time': computation_time,
            'convergence': makespan_history,
            'generations': generations
        }
    
    def _initialize_population(self):
        population = [self._create_individual() for _ in range(self.population_size)]
        return population
    
    def _create_individual(self):
        operations = []
        for job_id in range(self.num_jobs):
            for op_idx in range(len(self.problem_instance[job_id])):
                operations.append((job_id, op_idx))
        random.shuffle(operations)
        return operations
    
    def _evaluate(self, schedule):
        machine_timeframes = [0] * self.num_machines
        job_end_times = [0] * self.num_jobs
        for job_id, op_idx in schedule:
            machine, duration = self.problem_instance[job_id][op_idx]
            start_time = max(machine_timeframes[machine], job_end_times[job_id])
            end_time = start_time + duration
            machine_timeframes[machine] = end_time
            job_end_times[job_id] = end_time
        return max(machine_timeframes)
    
    def _select_parents(self, population, fitness_scores):
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(list(enumerate(fitness_scores)), 3)
            winner_idx = min(tournament, key=lambda x: x[1])[0]
            selected.append(population[winner_idx])
        return selected
    
    def _crossover(self, parent1, parent2):
        crossover_point = random.randint(0, len(parent1) - 1)
        child = parent1[:crossover_point]
        remaining_ops = [op for op in parent2 if op not in child]
        child.extend(remaining_ops)
        return child
    
    def _mutate(self, individual, mutation_rate=0.1):
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual