import numpy as np
import argparse
import ioh
from ioh import logger
import random
from tqdm import tqdm
import plotly.express as px
import os
import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ===== Utility functions =====
def evaluate_solution(problem, solution):
    f_val = problem(solution)
    cost = np.sum(solution)
    return f_val, cost

def dominates(a, b):
    return (a[0] >= b[0] and a[1] <= b[1]) and (a[0] > b[0] or a[1] < b[1])

def non_dominated_sort_indices(objs):
    """Return indices of Pareto front only (first front)."""
    objs = np.array(objs)
    n = len(objs)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(objs[j], objs[i]):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]  # indices of first front

def crowding_distance(front_objs):
    """Compute crowding distance for a given front (array of shape [n, num_obj])."""
    n_points = len(front_objs)
    if n_points == 0:
        return np.array([])
    distances = np.zeros(n_points)
    num_obj = front_objs.shape[1]
    for m in range(num_obj):
        values = front_objs[:, m]
        sorted_idx = np.argsort(values)
        distances[sorted_idx[0]] = distances[sorted_idx[-1]] = np.inf
        denom = values[sorted_idx[-1]] - values[sorted_idx[0]]
        if denom == 0:
            continue
        for i in range(1, n_points-1):
            distances[sorted_idx[i]] += (values[sorted_idx[i+1]] - values[sorted_idx[i-1]]) / denom
    return distances

def bit_flip_mutation_vec(solution, p_mut):
    """Vectorized mutation."""
    mutant = solution.copy()
    mask = np.random.rand(len(solution)) < p_mut
    mutant[mask] = 1 - mutant[mask]
    return mutant


# ===== Multi-Objective EA =====
# ===== Multi-Objective EA =====
def multi_objective_ea(problem_id=2100, pop_size=20, budget=10000, p_mut=None, save_path="results.pkl"):
    problem = ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)
    dim = problem.meta_data.n_variables
    p_mut = 1.0 / dim if p_mut is None else p_mut

    # --- Initialization ---
    population = np.random.randint(0, 2, size=(pop_size, dim))
    objs = np.array([evaluate_solution(problem, ind) for ind in population])
    evals = pop_size

    best_f_vals = [np.max(objs[:, 0])]

    while evals < budget:
        # Select parent from Pareto front
        pareto_idx = non_dominated_sort_indices(objs)
        parent_idx = np.random.choice(pareto_idx)
        parent = population[parent_idx]

        # Variation
        child = bit_flip_mutation_vec(parent, p_mut)
        f_val, cost = evaluate_solution(problem, child)
        child_obj = np.array([f_val, cost])
        evals += 1

        best_f_vals.append(max(best_f_vals[-1], f_val))

        # Add child
        population = np.vstack([population, child])
        objs = np.vstack([objs, child_obj])

        # --- Replacement ---
        # Compute first front
        pareto_idx = non_dominated_sort_indices(objs)
        front_objs = objs[pareto_idx]
        distances = crowding_distance(front_objs)
        # Select top pop_size individuals
        if len(pareto_idx) <= pop_size:
            selected_idx = pareto_idx
        else:
            sorted_idx = np.argsort(-distances)[:pop_size]
            selected_idx = pareto_idx[sorted_idx]

        population = population[selected_idx]
        objs = objs[selected_idx]

    # Final Pareto front
    pareto_idx = non_dominated_sort_indices(objs)
    pareto_front = [tuple(objs[i]) for i in pareto_idx]

    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"pareto": pareto_front, "best_f_vals": best_f_vals}, f)

    return pareto_front, population, best_f_vals

# ===== Results =====
def load_results(problem_id, save_path="results.pkl"):
    import pickle
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    return data["pareto"], data["best_f_vals"]

def plot_progress(best_f_vals, problem_id):
    fig = px.line(y=best_f_vals, labels={"y": "Best Objective Value", "x": "Evaluations"},
                  title=f"Progress Plot - Problem {problem_id}")
    fig.update_layout(template="plotly_white")
    fig.show()

def plot_pareto_front_plotly(pareto_data, problem_id):
    f_vals, costs = zip(*pareto_data)
    fig = px.scatter(x=costs, y=f_vals,
                     labels={"x": "Cost", "y": "Objective Value"},
                     title=f"Pareto Front - Problem {problem_id}")
    fig.update_traces(marker=dict(size=8, color='blue', opacity=0.7))
    fig.update_layout(template="plotly_white")
    fig.show()

def plot_progress_small_multiple(all_best_f_vals, problems, cols=4, img_path=""):
    """
    all_best_f_vals: dict {problem_id: list of best_f_vals per run}
    problems: list of problem IDs
    """
    rows = (len(problems) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Problem {pid}" for pid in problems])

    for i, pid in enumerate(problems):
        runs = all_best_f_vals[pid]
        if not runs:
            continue

        # Pad runs to same length
        max_len = max(len(r) for r in runs)
        padded = np.array([r + [r[-1]]*(max_len - len(r)) for r in runs])

        best_progress = np.max(padded, axis=0)
        worst_progress = np.min(padded, axis=0)
        mean_progress = np.mean(padded, axis=0)
        x_vals = np.arange(max_len)

        row = i // cols + 1
        col = i % cols + 1

        # Mean line
        fig.add_trace(
            go.Scatter(x=x_vals, y=mean_progress, mode='lines', name=f"{pid}"),
            row=row, col=col
        )

        # Shaded area between best and worst
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_vals, x_vals[::-1]]),
                y=np.concatenate([worst_progress, best_progress[::-1]]),
                fill='tonexty',
                fillcolor='rgba(0,100,200,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Evaluations", row=row, col=col)
        fig.update_yaxes(title_text="Objective Value", row=row, col=col)

    fig.update_layout(height=300*rows, width=350*cols,
                      title_text="Progress Plots (Best–Worst Across Runs)", title_x=0.5,
                      showlegend=False)
    
    os.makedirs(img_path, exist_ok=True)
    # save_plot_as_pdf(fig, f"{img_path}/progress_plot.pdf")
    # fig.write_html(f"{img_path}/progress_plot.html")
    fig.write_image(f"{img_path}/progress_plot.png")
    # fig.show()

def plot_pareto_small_multiple(all_pareto, problems, cols=4, img_path=""):
    """
    all_pareto: dict {problem_id: list of list of (f_val, cost) per run}
    """
    rows = (len(problems) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Problem {pid}" for pid in problems])

    for i, pid in enumerate(problems):
        row = i // cols + 1
        col = i % cols + 1
        runs = all_pareto[pid]
        if not runs:
            continue
        for run in runs:
            if not run:
                continue
            f_vals, costs = zip(*run)
            fig.add_trace(
                go.Scatter(x=costs, y=f_vals, mode='markers',
                           marker=dict(size=6, opacity=0.5),
                           showlegend=False),
                row=row, col=col
            )

        fig.update_xaxes(title_text="Cost", row=row, col=col)
        fig.update_yaxes(title_text="Objective Value", row=row, col=col)

    fig.update_layout(height=300*rows, width=350*cols,
                      title_text="Fixed-Budget Pareto Fronts Across Runs", title_x=0.5)
    
    os.makedirs(img_path, exist_ok=True)
    # save_plot_as_pdf(fig, f"{img_path}/pareto.pdf")
    # fig.write_html(f"{img_path}/pareto.html")
    fig.write_image(f"{img_path}/pareto.png")
    # fig.show()



# ===== Main =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run multi-objective EA experiments.")
    parser.add_argument("--pop_size", type=int, default=10, help="Population size for the evolutionary algorithm (e.g., 10, 20, 50)")
    parser.add_argument("--rerun", action="store_true", help="Re-run the EA even if saved results exist")
    parser.add_argument("--save_path", type=str, default="", help="Base path for saving results and images")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of runs per problem instance")
    parser.add_argument("--budget", type=int, default=100000, help="Evaluation budget for the evolutionary algorithm")
    parser.add_argument("--instances", type=int, default=30, help="Number of instances to evaluate")
    parser.add_argument("--problems", type=int, nargs='+', default=[2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203], 
                        help="List of problem IDs to solve (e.g., --problems 2100 2101 2102)")
    # In the argument parser section
    parser.add_argument("--run_id", type=int, default=None, help="Specific run ID to execute (0 to num_runs-1)")

    # In the main section, replace the loop:
    args = parser.parse_args()

    # --- Assign arguments ---
    pop_size = args.pop_size
    rerun = args.rerun
    save_path_ = args.save_path
    num_runs_per_instance = args.num_runs
    budget = args.budget
    instances = args.instances
    problems = args.problems
    run_id = args.run_id

    # -- set other parameters --
    all_best_f_vals = {}
    all_pareto = {}

    # If run_id is specified, only do that one run
    if run_id is not None:
        # Single run mode (for parallel jobs)
        if run_id < 0 or run_id >= num_runs_per_instance:
            print(f"Error: run_id {run_id} out of range [0, {num_runs_per_instance-1}]")
            exit(1)
        
        print(f"Running single run_id: {run_id}")
        
        for problem_id in problems:
            save_path = f"{save_path_}results/{pop_size}/problem_{problem_id}_{run_id}.pkl"
            
            # Check if results exist and skip if not rerunning
            if not rerun and os.path.exists(save_path):
                print(f"Results already exist for problem {problem_id}, run {run_id}. Skipping...")
                continue
            
            try:
                print(f"Running problem {problem_id}, run {run_id}...")
                pareto, pop, best_f_vals = multi_objective_ea(problem_id, pop_size, budget, save_path=save_path)
                print(f"Completed problem {problem_id}, run {run_id}")
            except Exception as e:
                print(f"Error running problem {problem_id}, run {run_id}: {e}")
                continue

    else:
        # Original mode: run all
        total_iterations = len(problems) * num_runs_per_instance
        pbar = tqdm(total=total_iterations, desc="All Runs", ncols=100)

        for problem_id in problems:
            all_best_f_vals[problem_id] = []
            all_pareto[problem_id] = []
            
            for run in range(num_runs_per_instance):
                save_path = f"{save_path_}results/{pop_size}/problem_{problem_id}_{run}.pkl"
                
                # Check if results exist and skip if not rerunning
                if not rerun and os.path.exists(save_path):
                    try:
                        pareto, best_f_vals = load_results(problem_id, save_path=save_path)
                    except Exception as e:
                        print(f"\nWarning: Could not load {save_path}: {e}. Re-running...")
                        try:
                            pareto, pop, best_f_vals = multi_objective_ea(problem_id, pop_size, budget, save_path=save_path)
                        except Exception as e:
                            print(f"\nError running problem {problem_id}, run {run}: {e}. Skipping...")
                            pbar.update(1)
                            continue
                else:
                    # Run the EA
                    try:
                        pareto, pop, best_f_vals = multi_objective_ea(problem_id, pop_size, budget, save_path=save_path)
                    except Exception as e:
                        print(f"\nError running problem {problem_id}, run {run}: {e}. Skipping to next problem...")
                        pbar.update(num_runs_per_instance - run)
                        break

                all_pareto[problem_id].append(pareto)
                all_best_f_vals[problem_id].append(best_f_vals)
                pbar.update(1)

        pbar.close()
    # args = parser.parse_args()
    
    # # --- Assign arguments ---
    # pop_size = args.pop_size
    # rerun = args.rerun
    # save_path_ = args.save_path
    # num_runs_per_instance = args.num_runs
    # budget = args.budget
    # instances = args.instances
    # problems = args.problems

    # # -- set other parameters, that dont change across experiments --
    # all_best_f_vals = {}
    # all_pareto = {}
    # # num_runs_per_instance = 30
    # # problems = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203]
    # # budget = 100000
    # # instances = 30 #TODO
    # if type(problems) == int:
    #     problems = list(problems)

    # total_iterations = len(problems) * num_runs_per_instance
    # pbar = tqdm(total=total_iterations, desc="All Runs", ncols=100)

    # for problem_id in problems:
    #     all_best_f_vals[problem_id] = []
    #     all_pareto[problem_id] = []
        
    #     for run in range(num_runs_per_instance):
    #         save_path = f"{save_path_}results/{pop_size}/problem_{problem_id}_{run}.pkl"
            
    #         # Check if results exist and skip if not rerunning
    #         if not rerun and os.path.exists(save_path):
    #             try:
    #                 pareto, best_f_vals = load_results(problem_id, save_path=save_path)
    #             except Exception as e:
    #                 print(f"\nWarning: Could not load {save_path}: {e}. Re-running...")
    #                 try:
    #                     pareto, pop, best_f_vals = multi_objective_ea(problem_id, pop_size, budget, save_path=save_path)
    #                 except Exception as e:
    #                     print(f"\nError running problem {problem_id}, run {run}: {e}. Skipping...")
    #                     pbar.update(1)
    #                     continue
    #         else:
    #             # Run the EA
    #             try:
    #                 pareto, pop, best_f_vals = multi_objective_ea(problem_id, pop_size, budget, save_path=save_path)
    #             except Exception as e:
    #                 print(f"\nError running problem {problem_id}, run {run}: {e}. Skipping to next problem...")
    #                 # Skip remaining runs for this problem if instance doesn't exist
    #                 pbar.update(num_runs_per_instance - run)
    #                 break

    #         all_pareto[problem_id].append(pareto)
    #         all_best_f_vals[problem_id].append(best_f_vals)
    #         pbar.update(1)

    # pbar.close()

    # Plot all in small multiples
    img_path = f"{save_path_}/results/img/{pop_size}/"
    plot_progress_small_multiple(all_best_f_vals, problems, img_path=img_path)
    plot_pareto_small_multiple(all_pareto, problems, img_path=img_path)
