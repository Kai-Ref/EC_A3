from ioh import get_problem, ProblemClass, logger
import sys
import argparse
import numpy as np

rng = None

def tournament_select(pop, fitness, k=3):
    idxs = rng.integers(0, len(pop), size=k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitness[i] > fitness[best]:
            best = i
    return best

def uniform_crossover(p1, p2):
    # 0/1 int arrays
    mask = rng.random(p1.size) < 0.5
    return np.where(mask, p1, p2).astype(np.int64)

def bitflip_mutation(x, p):
    flips = rng.random(x.size) < p
    return np.bitwise_xor(x, flips).astype(np.int64)

def ga_uniform(func,
               budget=10000,
               mu=20,              # parents per brief
               lambd=20,           # offspring per gen
               pc=1.0,             # crossover prob
               pm=None,            # default 1/n
               tour_k=3,
               elitism=True,
               runs=10):
    n = func.meta_data.n_variables
    if pm is None:
        pm = 1.0 / n

    optimum = func.optimum.y
    for r in range(runs):
        # init parents
        print(f"GA - func: {func.meta_data.name} dim: {func.meta_data.n_variables} run: {r+1}")
        P = rng.integers(0, 2, size=(mu, n), dtype=np.int64)
        F = np.empty(mu, dtype=float)
        evals = 0
        for i in range(mu):
            F[i] = func(P[i]); evals += 1
        best_f = F.max()
        if best_f >= optimum:
            func.reset(); continue

        # gens til budget
        while evals < budget:
            # produce offspring
            O = np.empty((lambd, n), dtype=np.int64)
            for j in range(lambd):
                p1 = P[tournament_select(P, F, k=tour_k)]
                p2 = P[tournament_select(P, F, k=tour_k)]
                child = uniform_crossover(p1, p2) if rng.random() < pc else p1.copy()
                child = bitflip_mutation(child, pm)
                O[j] = child

            # evaluate offspring
            FO = np.empty(lambd, dtype=float)
            for j in range(lambd):
                if evals >= budget:
                    break
                FO[j] = func(O[j]); evals += 1

            # elitism survivor selection 
            if elitism:
                P_all = np.vstack([P, O])
                F_all = np.concatenate([F, FO])
                idx = np.argsort(-F_all)[:mu]
                P, F = P_all[idx], F_all[idx]
            else:
                # preseve best parent + best offspring
                bp = np.argmax(F)
                top_off = np.argsort(-FO)[:mu-1]
                P = np.vstack([P[bp], O[top_off]])
                F = np.concatenate([[F[bp]], FO[top_off]])

            if F.max() > best_f:
                best_f = F.max()
            if best_f >= optimum:
                break

        func.reset()



def RLS_EA(func, budget = None):
    if budget is None:
        budget = int(10000)

    optimum = func.optimum.y
    for r in range(30):
        print(f"RLS - func: {func.meta_data.name} dim: {func.meta_data.n_variables} run: {r+1}")
        x_opt = rng.integers(low=0, high=2, size = func.meta_data.n_variables)
        x = x_opt
        f_opt = func(x_opt)
        for i in range(budget):
            #choose a random bit to flip
            flip = rng.integers(func.meta_data.n_variables)
            #flip that bits
            if x[flip] == 1:
                x[flip] = 0
            else:
                x[flip] = 1
            
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break

            
        func.reset()
    return f_opt, x_opt  

def one_one_EA(func, budget = None):
    if budget is None:
        budget = int(10000)

    optimum = func.optimum.y
    for r in range(30):
        print(f"(1+1) - func: {func.meta_data.name} dim: {func.meta_data.n_variables} run: {r+1}")
        x_opt = rng.integers(low=0, high=2, size = func.meta_data.n_variables)
        x = x_opt
        f_opt = func(x_opt)
        for i in range(budget):
            #create a random array and see if any of their values have probability 1/n
            flips = rng.random(func.meta_data.n_variables) <= (1/func.meta_data.n_variables)
            #flip those bits
            x = np.bitwise_xor(x, flips)
            f = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break

            
        func.reset()
    return f_opt, x_opt  


parser = argparse.ArgumentParser(prog="PBO Exercise 3 Script", description="Tests RLS, (1+1), and GA on MaxCoverage, MaxInfluence, PackWhileTravel")
parser.add_argument('function')
parser.add_argument('seed')
parser.add_argument('budget')


problems = [get_problem(fid=2100, problem_class = ProblemClass.GRAPH),
get_problem(fid=2101, problem_class = ProblemClass.GRAPH),
get_problem(fid=2102, problem_class = ProblemClass.GRAPH),
get_problem(fid=2103, problem_class = ProblemClass.GRAPH),
get_problem(fid=2200, problem_class = ProblemClass.GRAPH),
get_problem(fid=2201, problem_class = ProblemClass.GRAPH),
get_problem(fid=2202, problem_class = ProblemClass.GRAPH),
get_problem(fid=2203, problem_class = ProblemClass.GRAPH),
get_problem(fid=2300, problem_class = ProblemClass.GRAPH),
get_problem(fid=2301, problem_class = ProblemClass.GRAPH),
get_problem(fid=2302, problem_class = ProblemClass.GRAPH)
]

if __name__ == "__main__":
    #args should be "function, seed, budget" in that order
    args = parser.parse_args()
    
    run_seed = int(args.seed)
    run_budget = int(args.budget)

    rng = np.random.default_rng(seed=run_seed)

    if args.function == "one" :
        algo_name = "(1+1)-EA"
        algo_info = "Assignment 3: Exercise 1 - (1+1) EA Algorithm"
    elif args.function == "rls":
        algo_name = "RLS-EA"
        algo_info = "Assignment 3: Exercise 1 - RLS EA Algorithm"
    elif args.function == "ga":
        algo_name = "GA"
        algo_info = "Assignment 3: Exercise 1 - GA Algorithm"

    log = logger.Analyzer(root="data", 
        folder_name="run", 
        algorithm_name=algo_name, 
        algorithm_info=algo_info)

    for f in problems:
        f.attach_logger(log)
        if args.function == "one" :
            one_one_EA(f, budget=run_budget)
        elif args.function == "rls":
            RLS_EA(f, budget=run_budget)
        elif args.function == "ga":
            ga_uniform(f, budget=run_budget, mu=20, lambd=20, pc=1.0, pm=None, tour_k=3, elitism=True, runs=30)

    del log