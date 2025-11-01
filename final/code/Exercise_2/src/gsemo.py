import numpy as np
import ioh
from tqdm import tqdm
from matplotlib import pyplot as plt
class GSEMO():
    def __init__(self, problem: ioh.ProblemClass):
        self.problem = problem
        # self.population = np.random.randint(2, size=problem.meta_data.n_variables)[np.newaxis, :]
        self.population = np.zeros((1, problem.meta_data.n_variables), dtype=int)  # start with all-zero solution
        print(self.population.shape)
        self.fitness = []
        
    def reset(self):
        self.population = np.random.randint(2, size=self.problem.meta_data.n_variables)[np.newaxis, :]
        self.fitness = []

    def bit_flip(self, x: np.ndarray):
        index = np.where(np.random.rand(np.size(x)) < 1/np.size(x))
        x[index] = 1 - x[index]
        return x
        
    def is_strictly_dominated(self, g_prime):
        for fit in self.fitness:
            if all(fit >= g_prime) and any(fit > g_prime):
                return True
        return False
    
    def dominating(self, g_prime):
        dominated_indices = []
        for i, fit in enumerate(self.fitness):
            if all(g_prime >= fit):
                dominated_indices.append(i)   
        return dominated_indices
                     
    def __call__(self, budget: int = 10000, g: callable = None):
        self.fitness.append(g(self.population[0]))
        for i in range(budget):
            random_index = np.random.randint(np.shape(self.population)[0])
            x = self.population[random_index].copy()
            x = self.bit_flip(x)
            g_prime = g(x)
            if not self.is_strictly_dominated(g_prime):
                # plt.scatter(g_prime[0], g_prime[1]) if i % 100 == 0 else None
                dominated_indices = self.dominating(g_prime)
                # print(g_prime, self.fitness)
                self.population = np.delete(self.population, dominated_indices, axis=0)
                self.fitness = [self.fitness[i] for i in range(len(self.fitness)) if i not in dominated_indices]
                self.population = np.vstack((self.population, x))
                self.fitness.append(g_prime)
        # plt.show()
        return self.population, np.array(self.fitness)