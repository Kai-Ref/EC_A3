import numpy as np
import ioh
from ioh import logger
import random
from tqdm import tqdm
import plotly.express as px
import os
import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


# ===== Utility functions =====
def evaluate_solution_single_obj(problem, solution, B):
    """
    Evaluate solution for single-objective formulation.
    Returns -inf if constraint violated, otherwise returns objective value.
    """
    f_val = problem(solution)
    cost = np.sum(solution)

    if cost > B:
        return -np.inf, cost
    return f_val, cost


def bit_flip_mutation_vec(solution, p_mut):
    """Vectorized mutation."""
    mutant = solution.copy()
    mask = np.random.rand(len(solution)) < p_mut
    mutant[mask] = 1 - mutant[mask]
    return mutant


def tournament_selection(population, fitness, tournament_size=3):
    """Tournament selection."""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmax(fitness[indices])]
    return best_idx


def diversity_hamming(population):
    """Calculate average pairwise Hamming distance."""
    n = len(population)
    if n <= 1:
        return 0.0

    total_dist = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += np.sum(population[i] != population[j])
            count += 1
    return total_dist / count if count > 0 else 0.0


def repair_solution(solution, B):
    """
    Repair solution by randomly removing bits until constraint is satisfied.
    """
    while np.sum(solution) > B:
        ones_indices = np.where(solution == 1)[0]
        if len(ones_indices) == 0:
            break
        remove_idx = np.random.choice(ones_indices)
        solution[remove_idx] = 0
    return solution


def greedy_repair(problem, solution, B):
    """
    Greedy repair: Remove elements with lowest marginal contribution.
    """
    while np.sum(solution) > B:
        ones_indices = np.where(solution == 1)[0]
        if len(ones_indices) == 0:
            break

        # Evaluate marginal loss for each element
        current_val = problem(solution)
        min_loss = np.inf
        worst_idx = ones_indices[0]

        for idx in ones_indices:
            temp = solution.copy()
            temp[idx] = 0
            new_val = problem(temp)
            loss = current_val - new_val
            if loss < min_loss:
                min_loss = loss
                worst_idx = idx

        solution[worst_idx] = 0

    return solution


# ===== Single-Objective EA with Diversity Mechanisms =====
def single_objective_ea(
        problem_id=2100,
        pop_size=20,
        budget=10000,
        p_mut=None,
        diversity_mechanism='fitness_sharing',  # 'fitness_sharing', 'crowding', 'niching', 'age'
        tournament_size=3,
        repair_strategy='greedy',  # 'random', 'greedy', 'none'
        elitism_ratio=0.1,
        save_path="results_single.pkl"
):
    """
    Single-objective EA with multiple diversity mechanisms.

    Diversity mechanisms:
    - 'fitness_sharing': Share fitness based on Hamming distance
    - 'crowding': Deterministic crowding replacement
    - 'niching': Replace most similar individual
    - 'age': Age-based replacement with fitness tournament
    """
    problem = ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)
    dim = problem.meta_data.n_variables
    p_mut = 1.0 / dim if p_mut is None else p_mut

    # Determine budget B (uniform constraint)
    # For submodular problems, reasonable constraint is around dim/10 to dim/2
    B = dim // 10  # Can be adjusted based on problem

    # --- Initialization ---
    population = np.random.randint(0, 2, size=(pop_size, dim))

    # Repair initial population if needed
    if repair_strategy != 'none':
        for i in range(pop_size):
            if np.sum(population[i]) > B:
                if repair_strategy == 'greedy':
                    population[i] = greedy_repair(problem, population[i], B)
                else:
                    population[i] = repair_solution(population[i], B)

    # Evaluate initial population
    fitness = np.array([evaluate_solution_single_obj(problem, ind, B)[0] for ind in population])
    evals = pop_size

    # Track ages for age-based diversity
    ages = np.zeros(pop_size)

    best_f_vals = [np.max(fitness[np.isfinite(fitness)]) if np.any(np.isfinite(fitness)) else 0]
    diversity_history = [diversity_hamming(population)]

    # Main evolution loop
    while evals < budget:
        # --- Parent Selection ---
        if diversity_mechanism == 'fitness_sharing':
            # Calculate shared fitness
            shared_fitness = calculate_shared_fitness(population, fitness, dim)
            parent_idx = tournament_selection(population, shared_fitness, tournament_size)
        else:
            parent_idx = tournament_selection(population, fitness, tournament_size)

        parent = population[parent_idx]

        # --- Variation ---
        child = bit_flip_mutation_vec(parent, p_mut)

        # Repair child if constraint violated
        if np.sum(child) > B and repair_strategy != 'none':
            if repair_strategy == 'greedy':
                child = greedy_repair(problem, child, B)
            else:
                child = repair_solution(child, B)

        child_fitness, _ = evaluate_solution_single_obj(problem, child, B)
        evals += 1

        # Track best fitness
        if np.isfinite(child_fitness):
            best_f_vals.append(max(best_f_vals[-1], child_fitness))
        else:
            best_f_vals.append(best_f_vals[-1])

        # --- Survivor Selection ---
        if diversity_mechanism == 'crowding':
            # Deterministic crowding
            replace_idx = find_most_similar(population, child)
            if child_fitness > fitness[replace_idx]:
                population[replace_idx] = child
                fitness[replace_idx] = child_fitness
                ages[replace_idx] = 0

        elif diversity_mechanism == 'niching':
            # Replace most similar individual if child is better
            replace_idx = find_most_similar(population, child)
            if child_fitness > fitness[replace_idx]:
                population[replace_idx] = child
                fitness[replace_idx] = child_fitness
                ages[replace_idx] = 0

        elif diversity_mechanism == 'age':
            # Age-based replacement with elitism
            n_elite = max(1, int(pop_size * elitism_ratio))
            elite_indices = np.argsort(fitness)[-n_elite:]

            # Find oldest non-elite individual
            non_elite_mask = np.ones(pop_size, dtype=bool)
            non_elite_mask[elite_indices] = False
            non_elite_ages = ages.copy()
            non_elite_ages[~non_elite_mask] = -1

            replace_idx = np.argmax(non_elite_ages)

            # Replace if child is better or if replacing old individual
            if child_fitness > fitness[replace_idx] or ages[replace_idx] > pop_size:
                population[replace_idx] = child
                fitness[replace_idx] = child_fitness
                ages[replace_idx] = 0

        else:  # fitness_sharing or default
            # Replace worst individual
            worst_idx = np.argmin(fitness)
            if child_fitness > fitness[worst_idx]:
                population[worst_idx] = child
                fitness[worst_idx] = child_fitness

        # Update ages
        ages += 1

        # Track diversity
        if evals % 100 == 0:
            diversity_history.append(diversity_hamming(population))

    # Find best solution
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx]
    best_fitness = fitness[best_idx]

    # Save results
    results = {
        "best_fitness": best_fitness,
        "best_solution": best_solution.tolist(),
        "best_f_vals": best_f_vals,
        "diversity_history": diversity_history,
        "final_population": population.tolist(),
        "final_fitness": fitness.tolist()
    }

    # Create directory if path contains a directory component
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    return best_fitness, best_solution, best_f_vals, diversity_history


def calculate_shared_fitness(population, fitness, dim, sigma_share=None):
    """
    Calculate shared fitness based on Hamming distance.
    """
    if sigma_share is None:
        sigma_share = dim / 4  # Default sharing radius

    n = len(population)
    shared_fitness = np.zeros(n)

    for i in range(n):
        niche_count = 0
        for j in range(n):
            dist = np.sum(population[i] != population[j])
            if dist < sigma_share:
                niche_count += 1 - (dist / sigma_share)
        shared_fitness[i] = fitness[i] / max(niche_count, 1)

    return shared_fitness


def find_most_similar(population, individual):
    """
    Find the most similar individual in population (smallest Hamming distance).
    """
    distances = np.array([np.sum(ind != individual) for ind in population])
    return np.argmin(distances)


# ===== Experiment Runner with Multiprocessing =====
def run_single_experiment(args):
    """
    Wrapper function for running a single experiment.
    Used for parallel execution.
    """
    problem_id, pop_size, mechanism, run, budget, base_dir = args
    save_path = f"{base_dir}/problem_{problem_id}/pop_{pop_size}/{mechanism}/run_{run}.pkl"

    try:
        best_fit, best_sol, best_vals, div_hist = single_objective_ea(
            problem_id=problem_id,
            pop_size=pop_size,
            budget=budget,
            diversity_mechanism=mechanism,
            repair_strategy='greedy',
            save_path=save_path
        )
        return (problem_id, pop_size, mechanism, run, best_fit, best_vals, div_hist, None)
    except Exception as e:
        return (problem_id, pop_size, mechanism, run, None, None, None, str(e))


def run_experiments(
        problems=[2100, 2101, 2102, 2103],
        pop_sizes=[10, 20, 50],
        diversity_mechanisms=['fitness_sharing', 'crowding', 'niching', 'age'],
        num_runs=30,
        budget=10000,
        base_dir="results_exercise3",
        n_jobs=None
):
    """
    Run comprehensive experiments for Exercise 3 with multiprocessing.

    Args:
        n_jobs: Number of parallel workers. If None, uses cpu_count() - 1
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # Leave one core free

    print(f"Running experiments with {n_jobs} parallel workers...")

    # Initialize results structure
    all_results = {}
    for problem_id in problems:
        all_results[problem_id] = {}
        for pop_size in pop_sizes:
            all_results[problem_id][pop_size] = {}
            for mechanism in diversity_mechanisms:
                all_results[problem_id][pop_size][mechanism] = {
                    'best_fitness': [],
                    'best_f_vals': [],
                    'diversity': []
                }

    # Create list of all experiment configurations
    experiment_configs = []
    for problem_id in problems:
        for pop_size in pop_sizes:
            for mechanism in diversity_mechanisms:
                for run in range(num_runs):
                    experiment_configs.append(
                        (problem_id, pop_size, mechanism, run, budget, base_dir)
                    )

    total_runs = len(experiment_configs)
    print(f"Total experiments to run: {total_runs}")

    # Run experiments in parallel
    completed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all jobs
        futures = {executor.submit(run_single_experiment, config): config
                   for config in experiment_configs}

        # Process completed jobs with progress bar
        with tqdm(total=total_runs, desc="Running Experiments", ncols=100) as pbar:
            for future in as_completed(futures):
                result = future.result()
                problem_id, pop_size, mechanism, run, best_fit, best_vals, div_hist, error = result

                if error is None:
                    # Store successful results
                    all_results[problem_id][pop_size][mechanism]['best_fitness'].append(best_fit)
                    all_results[problem_id][pop_size][mechanism]['best_f_vals'].append(best_vals)
                    all_results[problem_id][pop_size][mechanism]['diversity'].append(div_hist)
                else:
                    # Track errors
                    errors.append({
                        'problem': problem_id,
                        'pop_size': pop_size,
                        'mechanism': mechanism,
                        'run': run,
                        'error': error
                    })
                    print(f"\nError in problem {problem_id}, pop {pop_size}, {mechanism}, run {run}: {error}")

                completed += 1
                pbar.update(1)

    print(f"\nCompleted {completed} experiments")
    if errors:
        print(f"Encountered {len(errors)} errors")

    # Save aggregated results
    os.makedirs(base_dir, exist_ok=True)
    with open(f"{base_dir}/all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    if errors:
        with open(f"{base_dir}/errors.pkl", "wb") as f:
            pickle.dump(errors, f)

    return all_results


# ===== Visualization =====
def plot_comparison(all_results, problems, pop_sizes, mechanisms):
    """
    Create comparison plots for different configurations.
    """
    # Plot 1: Best fitness comparison
    fig1 = make_subplots(
        rows=len(problems), cols=len(pop_sizes),
        subplot_titles=[f"Problem {pid}, Pop={ps}"
                        for pid in problems for ps in pop_sizes]
    )

    for i, pid in enumerate(problems):
        for j, ps in enumerate(pop_sizes):
            for mechanism in mechanisms:
                data = all_results[pid][ps][mechanism]['best_f_vals']
                if not data:
                    continue

                # Calculate mean progress
                max_len = max(len(r) for r in data)
                padded = np.array([r + [r[-1]] * (max_len - len(r)) for r in data])
                mean_progress = np.mean(padded, axis=0)

                fig1.add_trace(
                    go.Scatter(x=np.arange(len(mean_progress)), y=mean_progress,
                               name=mechanism, mode='lines'),
                    row=i + 1, col=j + 1
                )

    fig1.update_layout(height=300 * len(problems), width=400 * len(pop_sizes),
                       title_text="Progress Comparison Across Configurations")
    fig1.show()

    fig1.write_image("comparison_plot.png", scale=10)

    # Plot 2: Final fitness box plots
    fig2 = go.Figure()

    for ps in pop_sizes:
        for mechanism in mechanisms:
            final_fitness = []
            for pid in problems:
                final_fitness.extend(all_results[pid][ps][mechanism]['best_fitness'])

            fig2.add_trace(go.Box(
                y=final_fitness,
                name=f"{mechanism} (pop={ps})",
                boxmean='sd'
            ))

    fig2.update_layout(title="Final Fitness Distribution",
                       yaxis_title="Fitness",
                       template="plotly_white")
    fig2.show()

    fig2.write_image("final_fitness_boxplot.png", scale=10)


# ===== Main Execution =====
if __name__ == "__main__":
    # Quick test on single problem
    # print("Running single test...")
    # best_fit, best_sol, best_vals, div_hist = single_objective_ea(
    #     problem_id=2100,
    #     pop_size=20,
    #     budget=10000,
    #     diversity_mechanism='fitness_sharing',
    #     save_path="test_single.pkl"
    # )
    # print(f"Best fitness: {best_fit}")
    # print(f"Final diversity: {div_hist[-1]:.2f}")

    # # Run full experiments with multiprocessing
    # print("\n" + "=" * 60)
    # print("Starting full experiments with parallel processing...")
    # print("=" * 60 + "\n")
    #
    # all_results = run_experiments(
    #     problems=[2100, 2101, 2102, 2103],
    #     pop_sizes=[10, 20, 50],
    #     diversity_mechanisms=['fitness_sharing', 'crowding', 'niching', 'age'],
    #     num_runs=30,
    #     budget=10000,
    #     n_jobs=None  # Uses cpu_count() - 1, or set to specific number like 4
    # )

    all_results = pickle.load(open("../../../results/exercise3/single_objective/all_results.pkl", "rb"))

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(all_results, [2100, 2101, 2102, 2103], [10, 20, 50],
                    ['fitness_sharing', 'crowding', 'niching', 'age'])