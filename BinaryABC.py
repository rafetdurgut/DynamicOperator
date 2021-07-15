import copy
from Bee import Bee
import numpy as np
from itertools import accumulate


class BinaryABC:
    def __init__(self, problem, operator_pool, operator_selector, pop_size, maxFE, limit, log_file=None):
        '''
        Constructor
        :param problem: The problem
        :param operator: Operator used by Bee's
        :param pop_size: Population size of bee's
        :param maxFE: Maximum number of evaluation size for objective function
        :param limit: Maximum number of trial for the bee.
        :param log_file: if none, nothing will be logged. Provide file name in order to write this file
        '''
        self.pop_size = pop_size
        self.problem = problem
        self.maxFE = maxFE
        self.operator_pool = operator_pool
        self.operator_selector = operator_selector
        self.max_iteration =int(maxFE / (pop_size*2))
        self.limit = limit # bee trail limit
        self.colony = [Bee(problem) for _ in range(self.pop_size)]
        # keep best Bee
        self.global_best = Bee(problem)
        self.probabilities = np.zeros(self.pop_size)
        self.convergence = list()
        self.iteration = 0
        self.log_file = log_file
        self.FE = 0

    def employed_bee(self):
        self.calculate_cdf()
        for i in range(self.pop_size):
            candidate = copy.deepcopy(self.colony[i])
            neighbor = self.neighbor_selection()
            while self.colony[i] == neighbor:
                neighbor = self.neighbor_selection()
            op_no = self.operator_selector.operator_selection(candidate)
            self.FE += self.operator_pool[op_no].costFE()
            candidate = self.operator_pool[op_no].get_candidate(candidate, neighbor)
            #print(f'  current cost: {self.colony[i].cost}--> candidate cost:{candidate.cost} -- gbest cost:{self.global_best.cost}')
            if self.log_file is not None:
                self.fout.write(
                    f'{self.iteration};{self.colony[i].solution.astype(int)};{self.operator_pool[op_no]};{self.colony[i].cost};{candidate.cost};')
                self.fout.write(f'{candidate.solution.astype(int)};{candidate.cost}\n')
            self.operator_selector.add_reward(op_no, candidate, self.colony[i])
            self.colony[i] = self.colony[i].get_better(candidate)

    def onlooker_bee(self):
        self.calculate_cdf()
        max_fitness = max([bee.fitness for bee in self.colony])
        t = 0
        i = 0
        while t < self.pop_size:
            if np.random.random() < self.colony[i].fitness / max_fitness:
                t += 1
                candidate = copy.deepcopy(self.colony[i])
                neighbor = self.neighbor_selection()
                while self.colony[i] == neighbor:
                    neighbor = self.neighbor_selection()
                op_no = self.operator_selector.operator_selection(candidate)
                self.FE += self.operator_pool[op_no].costFE()
                candidate = self.operator_pool[op_no].get_candidate(candidate, neighbor)
                
                if self.log_file is not None:
                    self.fout.write(
                        f'{self.iteration};{self.colony[i].solution.astype(int)};{self.operator_pool[op_no]};{self.colony[i].cost};{candidate.cost};')
                    self.fout.write(f'{candidate.solution.astype(int)};{candidate.cost}\n')
                self.operator_selector.add_reward(op_no, candidate, self.colony[i])
                self.colony[i] = self.colony[i].get_better(candidate)
            i += 1
            i = i % self.pop_size

    def scout_bee(self):
        t_index=-1
        for ind, b in enumerate(self.colony):
            if b.trial >= self.limit:
                t_index = ind
                break
        if t_index >= 0:
            self.colony[t_index].initial()

    def memorize(self):
        best_solution = max(self.colony, key=lambda b: b.cost)
        if best_solution.cost > self.global_best.cost:
            self.global_best = copy.deepcopy(best_solution)
            self.convergence.append((self.iteration, best_solution.cost))
            # print(self.global_best.cost)


    def calculate_cdf(self):
        #CDF
        sum_fitness = sum([x.fitness for x in self.colony])
        self.probabilities = list(accumulate([x.fitness / sum_fitness for x in self.colony]))

    def neighbor_selection(self):
        r = np.random.random()
        n: int = next(index for index, i in enumerate(self.probabilities) if i > r)
        return self.colony[n]

    def stop_condition(self):
        # if one of the solution reach maxFE, returns true, else false
        return self.iteration >= self.max_iteration

    def derive_func_eval_count(self):
        return self.iteration * self.operator.costFE()

    def run(self):
        self.fout = None
        if self.log_file is not None:
            np.set_printoptions(linewidth=5000000)  # numpy printing automatically adds line break. Prevent this.
            self.fout = open(self.log_file, 'w')
            self.fout.write('iteration;solution;operator;pcost;candi_cost;nsolution;ncost\n')
        self.iteration = 0
        while not self.stop_condition():
            self.employed_bee()
            self.onlooker_bee()
            self.memorize()
            self.scout_bee()
            self.iteration += 1
            self.operator_selector.next_iteration()
            if self.global_best.cost >= self.problem.best:
                return
        if self.fout is not None:
            self.fout.close()

