import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
import time

class QuantumAnnealingJSS:
    def __init__(self, jobs_data):
        self.jobs_data = jobs_data
        self.num_jobs = len(jobs_data)
        self.num_machines = max(machine for job in jobs_data for machine, _ in job) + 1
        self.max_time = sum(proc_time for job in jobs_data for _, proc_time in job)
        
    def create_qubo(self):
        Q = {}
        for job_id in range(self.num_jobs):
            for op_idx, (machine, proc_time) in enumerate(self.jobs_data[job_id]):
                for t in range(self.max_time - proc_time + 1):
                    key = (job_id, op_idx, t)
                    Q[(key, key)] = -1
                    for t2 in range(t + 1, self.max_time - proc_time + 1):
                        key2 = (job_id, op_idx, t2)
                        Q[(key, key2)] = 2
        for machine in range(self.num_machines):
            for t in range(self.max_time):
                machine_ops = []
                for job_id in range(self.num_jobs):
                    for op_idx, (m, proc_time) in enumerate(self.jobs_data[job_id]):
                        if m == machine:
                            for t_start in range(max(0, t - proc_time + 1), t + 1):
                                machine_ops.append((job_id, op_idx, t_start))
                for i in range(len(machine_ops)):
                    for j in range(i + 1, len(machine_ops)):
                        Q[(machine_ops[i], machine_ops[j])] = 2
        return Q
    
    def solve(self, num_reads=100):
        start_time = time.time()
        Q = self.create_qubo()
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample_qubo(Q, num_reads=num_reads)
        best_solution = response.first.sample
        schedule = self._convert_to_schedule(best_solution)
        computation_time = time.time() - start_time
        makespan = self._calculate_makespan(schedule)
        convergence = [makespan]  # Placeholder for convergence data
        return {
            'schedule': schedule,
            'makespan': makespan,
            'computation_time': computation_time,
            'energy': response.first.energy,
            'num_reads': num_reads,
            'convergence': convergence
        }
    
    def _convert_to_schedule(self, solution):
        schedule = {}
        for (job_id, op_idx, time_slot), value in solution.items():
            if value == 1:
                if job_id not in schedule:
                    schedule[job_id] = []
                schedule[job_id].append((op_idx, time_slot))
        return schedule
    
    def _calculate_makespan(self, schedule):
        makespan = 0
        for job_id, operations in schedule.items():
            for op_idx, start_time in operations:
                end_time = start_time + self.jobs_data[job_id][op_idx][1]
                makespan = max(makespan, end_time)
        return makespan