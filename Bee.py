import numpy as np


class Bee:
    def __init__(self, problem=None, solution=None):
        self.problem = problem
        self.fitness = 0
        self.cost = 0
        self.trial = 0
        if solution is None:
            self.solution = np.random.random(self.problem.dimension) > 0.5
        else:
            self.solution = solution
        self.evaluate()

    def evaluate(self):
        self.solution, self.cost = self.problem.objective_function(self.solution)
        self.calculate_fitness()

    def initial(self):
        self.trial = 0
        self.solution = np.random.random(self.problem.dimension) > 0.5
        self.evaluate()

    def get_better(self, candidate):
        if candidate.cost > self.cost:
            candidate.trial = 0
            return candidate
        else:
            self.trial += 1
            return self

    def __str__(self):
        return f'Trial:{self.trial}, Cost:{self.cost}'

    def calculate_fitness(self):
        self.fitness = 1 + self.cost
