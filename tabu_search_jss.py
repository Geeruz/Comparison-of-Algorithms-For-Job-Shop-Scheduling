# tabu_search_jss.py
import random
import time
from collections import defaultdict

class TabuSearchJSS:
    def __init__(self, jobs_data):
        """
        Initialize TABU Search solver for JSS
        jobs_data: List of jobs, where each job is a list of (machine_id, processing_time) tuples
        """
        self.jobs_data = jobs_data
        self.num_jobs = len(jobs_data)
        self.num_machines = max(machine for job in jobs_data 
                              for machine, _ in job) + 1
    
    def _create_initial_solution(self):
        """Create initial random solution."""
        operations = []
        for job_id in range(self.num_jobs):
            for op_idx in range(len(self.jobs_data[job_id])):
                operations.append((job_id, op_idx))
        random.shuffle(operations)
        return operations
    
    def _get_neighbors(self, solution):
        """Get neighboring solutions by swapping operations."""
        neighbors = []
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors
    
    def _decode_schedule(self, sequence):
        """Decode operation sequence into schedule with start times."""
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
    
    def solve(self, max_iterations=1000, tabu_tenure=10):
        """Solve JSS using TABU search."""
        start_time = time.time()
        
        # Initialize
        current_solution = self._create_initial_solution()
        current_schedule, current_makespan = self._decode_schedule(current_solution)
        
        best_solution = current_solution.copy()
        best_schedule = current_schedule
        best_makespan = current_makespan
        
        # Initialize tabu list and makespan history
        tabu_list = defaultdict(int)
        makespan_history = [current_makespan]
        
        iteration = 0
        while iteration < max_iterations:
            # Get neighbors
            neighbors = self._get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_makespan = float('inf')
            
            # Evaluate neighbors
            for neighbor in neighbors:
                neighbor_tuple = tuple(map(tuple, neighbor))
                if tabu_list[neighbor_tuple] <= iteration:
                    _, makespan = self._decode_schedule(neighbor)
                    if makespan < best_neighbor_makespan:
                        best_neighbor = neighbor
                        best_neighbor_makespan = makespan
            
            if best_neighbor is None:
                break
            
            # Update current solution
            current_solution = best_neighbor
            current_schedule, current_makespan = self._decode_schedule(current_solution)
            
            # Update tabu list
            tabu_list[tuple(map(tuple, current_solution))] = iteration + tabu_tenure
            
            # Update best solution
            if current_makespan < best_makespan:
                best_solution = current_solution.copy()
                best_schedule = current_schedule
                best_makespan = current_makespan
            
            makespan_history.append(best_makespan)
            iteration += 1
        
        computation_time = time.time() - start_time
        
        return {
            'schedule': best_schedule,
            'makespan': best_makespan,
            'computation_time': computation_time,
            'convergence': makespan_history,
            'iterations': iteration
        }