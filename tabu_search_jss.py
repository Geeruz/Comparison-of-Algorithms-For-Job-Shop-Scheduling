import random
import time
from collections import defaultdict

class TabuSearchJSS:
    def __init__(self, jobs_data):
        self.jobs_data = jobs_data
        self.num_jobs = len(jobs_data)
        self.num_machines = max(machine for job in jobs_data for machine, _ in job) + 1
    
    def _create_initial_solution(self):
        operations = []
        for job_id in range(self.num_jobs):
            for op_idx in range(len(self.jobs_data[job_id])):
                operations.append((job_id, op_idx))
        random.shuffle(operations)
        return operations
    
    def _evaluate(self, schedule):
        machine_timeframes = [0] * self.num_machines
        job_end_times = [0] * self.num_jobs
        for job_id, op_idx in schedule:
            machine, duration = self.jobs_data[job_id][op_idx]
            start_time = max(machine_timeframes[machine], job_end_times[job_id])
            end_time = start_time + duration
            machine_timeframes[machine] = end_time
            job_end_times[job_id] = end_time
        return max(machine_timeframes)
    
    def _get_neighbors(self, schedule):
        neighbors = []
        for i in range(len(schedule)):
            for j in range(i + 1, len(schedule)):
                neighbor = schedule[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors
    
    def _decode_schedule(self, sequence):
        machine_timeframes = [0] * self.num_machines
        job_timeframes = [0] * self.num_jobs
        schedule = {}
        for job_id, op_idx in sequence:
            machine, proc_time = self.jobs_data[job_id][op_idx]
            start_time = max(machine_timeframes[machine], job_timeframes[job_id])
            if job_id not in schedule:
                schedule[job_id] = []
            schedule[job_id].append((op_idx, start_time))
            machine_timeframes[machine] = start_time + proc_time
            job_timeframes[job_id] = start_time + proc_time
        return schedule, max(machine_timeframes)
    
    def solve(self, max_iterations=100, tabu_tenure=10):
        start_time = time.time()
        current_solution = self._create_initial_solution()
        current_makespan = self._evaluate(current_solution)
        best_solution = current_solution
        best_makespan = current_makespan
        tabu_list = defaultdict(int)
        makespan_history = []
        for iteration in range(max_iterations):
            neighbors = self._get_neighbors(current_solution)
            neighbors = sorted(neighbors, key=self._evaluate)
            for neighbor in neighbors:
                if tuple(neighbor) not in tabu_list or self._evaluate(neighbor) < best_makespan:
                    current_solution = neighbor
                    current_makespan = self._evaluate(current_solution)
                    break
            if current_makespan < best_makespan:
                best_solution = current_solution
                best_makespan = current_makespan
            makespan_history.append(best_makespan)
            tabu_list[tuple(current_solution)] = iteration + tabu_tenure
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
        computation_time = time.time() - start_time
        best_schedule, best_makespan = self._decode_schedule(best_solution)
        return {
            'schedule': best_schedule,
            'makespan': best_makespan,
            'computation_time': computation_time,
            'iterations': max_iterations,
            'convergence': makespan_history
        }