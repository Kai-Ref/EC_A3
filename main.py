import numpy as np
import src.gsemo as gsemo
import ioh
import multiprocessing
from tqdm import tqdm

def multi_objective_fitness(x: np.ndarray, problem: ioh.ProblemClass):
    f1 = problem(x)
    f2 = -np.sum(x) # prefer less 1s
    return np.array([f1, f2])

# Single-run worker: execute exactly one (problem_number, repeat_idx) in its own process
def run_one(task):
    problem_number, repeat_idx = task
    print(f"Running GSEMO on problem {problem_number} | repeat {repeat_idx}")
    problem = ioh.get_problem(problem_number, problem_class=ioh.ProblemClass.GRAPH)
    optimizer = gsemo.GSEMO(problem)
    logger = ioh.logger.Analyzer(
        root='./data',
        folder_name='GSEMO_runs',
        algorithm_name='GSEMO',
        algorithm_info=f'GSEMO on bi-objective problems {problem_number} (rep={repeat_idx})'
    )
    problem.attach_logger(logger)
    population, fitness = optimizer(g=lambda x: multi_objective_fitness(x, problem))
    print(f"Final population size for problem {problem_number} (rep {repeat_idx}): {len(population)}")
    return None

if __name__ == "__main__":
    problem_numbers = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203, 2300, 2301, 2302, 2303]
    seperate_runs = 30
    tasks = [(pn, r) for pn in problem_numbers for r in range(seperate_runs)]
    with multiprocessing.Pool() as pool:
        for _ in tqdm(pool.imap_unordered(run_one, tasks), total=len(tasks), desc="All runs"):
            pass
