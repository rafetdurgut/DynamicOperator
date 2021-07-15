import numpy as np


class abstractOperatorSelection:
    def __init__(self, operator_size, reward_type="insta", W=5, alpha=0.1, beta=0.5, Pmin=0.1):
        self.rewards = [[0] for _ in range(operator_size)]
        self.credits = [[0] for _ in range(operator_size)]

        self.success_counter = [[0] for _ in range(operator_size)]
        self.total_succ_counters = np.zeros((operator_size))
        self.usage_counter = [[0] for _ in range(operator_size)]
        self.probabilities = np.zeros((operator_size))
        self.reward = np.zeros((operator_size))
        self.type = 'iteration'
        self.iteration = 0
        self.operator_size = operator_size
        self.reward_type = reward_type
        self.W = W
        self.Pmin = Pmin
        self.Pmax = 0.8
        self.alpha = alpha
        self.beta = beta

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        if isinstance(self, ClusterRL):
            self.create_clusters()

    def get_reward(self, new_fitness, old_fitness):
        r = (self.algorithm.problem.dimension / self.algorithm.global_best.cost) * float((new_fitness - old_fitness))
        if r < 0:
            r = 0
        return r

    def next_iteration(self):
        self.update_credits()
        self.iteration += 1
        for i in range(self.operator_size):
            self.rewards[i].append(0)
            self.usage_counter[i].append(0)
            self.success_counter[i].append(0)

    def add_reward(self, op_no, candidate, current):
        self.usage_counter[op_no][self.iteration] += 1
        reward = self.get_reward(candidate.cost, current.cost)
        if reward > 0:
            self.success_counter[op_no][self.iteration] += 1
            self.total_succ_counters[op_no] += 1
            if self.type == 'iteration':
                self.rewards[op_no][self.iteration] += reward

    def apply_rewards(self):
        for i in range(self.operator_size):
            if self.reward_type == "insta":
                self.reward[i] = self.rewards[i][self.iteration]
            elif self.reward_type == "average":
                start_pos = max(0, len(self.rewards[i]) - self.W)
                reward = np.average(self.rewards[i][start_pos:len(self.rewards[i])])
                self.reward[i] = reward
            elif self.reward_type == "extreme":
                start_pos = max(0, len(self.rewards[i]) - self.W)
                reward = np.max(self.rewards[i][start_pos:len(self.rewards[i])])
                self.reward[i] = reward

    def update_credits(self):
        self.apply_rewards()
        for i in range(self.operator_size):
            credit = (1 - self.alpha) * self.credits[i][self.iteration] + self.alpha * self.reward[i]
            self.credits[i].append(credit)

    def operator_selection(self, candidate=None):
        raise Exception("Should not call Abstract Class!")

    def roulette_wheel(self, ):
        sumProbs = sum(self.probabilities)
        probs = [item / sumProbs for item in self.probabilities]
        op = np.random.choice(len(probs), p=probs)
        return op


class ProbabilityMatching(abstractOperatorSelection):
    def operator_selection(self, candidate):
        credits = [a[-1] for a in self.credits]
        if np.all(self.total_succ_counters) == 0 or np.sum(credits) == 0:
            for i in range(self.operator_size):
                self.probabilities[i] = 1 / self.operator_size
            return self.roulette_wheel()

        for i in range(self.operator_size):
            self.probabilities[i] = self.Pmin + (1 - self.operator_size * self.Pmin) * (
                    self.credits[i][self.iteration] / np.sum(credits))
        return self.roulette_wheel()

    def __conf__(self):
        return ['PM', self.operator_size,  self.reward_type, self.Pmin, self.W, self.alpha]

class AdaptivePursuit(abstractOperatorSelection):
    def operator_selection(self, candidate):
        credits = [a[-1] for a in self.credits]
        if np.all(self.total_succ_counters) == 0:
            for i in range(self.operator_size):
                self.probabilities[i] = 1 / self.operator_size
            return self.roulette_wheel()

        best_op = np.argmax(credits)
        for i in range(self.operator_size):
            if i == best_op:
                self.probabilities[i] = self.probabilities[i] + self.beta * (
                        self.Pmax - self.probabilities[i])
            else:
                self.probabilities[i] = self.probabilities[i] + self.beta * (
                        self.Pmin - self.probabilities[i])
        return self.roulette_wheel()

    def __conf__(self):
        return ['AP', self.operator_size,  self.reward_type, self.Pmin, self.W, self.alpha]

import math


class UpperConfidenceBound(abstractOperatorSelection):
    def __init__(self, operator_size, reward_type, W=5, alpha=0.1, beta=0.1, Pmin=0.1, C=2):
        super(UpperConfidenceBound, self).__init__(operator_size, reward_type, W, alpha, beta, Pmin)
        self.C = C
    def operator_selection(self, candidate):
        credits = [a[-1] for a in self.credits]
        if np.all(self.total_succ_counters) == 0:
            for i in range(self.operator_size):
                self.probabilities[i] = 1 / self.operator_size
            return self.roulette_wheel()
        # Burayi optimize et.
        values = [
            val + self.C * (math.sqrt(2 * math.log(np.sum(self.usage_counter)) / np.sum(self.usage_counter[ind][:])))
            for
            ind, val in enumerate(credits)]
        best_op = np.argmax(values)
        for i in range(self.operator_size):
            if i == best_op:
                self.probabilities[i] = 1 - (self.operator_size - 1) * self.Pmin
            else:
                self.probabilities[i] = self.Pmin
        return self.roulette_wheel()
    def __conf__(self):
        return ['UCB', self.operator_size,  self.reward_type, self.Pmin, self.W, self.alpha]

class RandomSelection(abstractOperatorSelection):
    def operator_selection(self, candidate):
        return np.random.randint(0,self.operator_size)

    def __conf__(self):
        return ['RND', self.operator_size,  self.reward_type, self.Pmin, self.W, self.alpha]
class ClusterRL(abstractOperatorSelection):
    def __init__(self, operator_size, reward_type, W, alpha,  Pmin, gama = 0):
        super(ClusterRL, self).__init__(operator_size, reward_type, W, alpha=alpha, beta=0.1, Pmin=Pmin)
        self.operator_size = operator_size
        self.alpha = alpha
        self.type = 'function'
        self.gama = gama
        self.timer = np.zeros((self.operator_size),dtype=int)

    def create_clusters(self):
        self.clusters = np.zeros((self.operator_size, self.algorithm.problem.dimension))

    def get_reward_bytype(self,op_no):
        if self.timer[op_no] == 0:
            return self.rewards[op_no][self.iteration]
        if self.reward_type == "insta":
            return self.rewards[op_no][self.iteration]
        elif self.reward_type == "average":
            start_pos = max(0, self.iteration - self.W)
            reward = np.sum(self.rewards[op_no][start_pos:self.iteration])/(self.iteration-start_pos)
            return  reward
        elif self.reward_type == "extreme":
            start_pos = max(0, len(self.rewards[op_no]) - self.W)
            reward = np.max(self.rewards[op_no][start_pos:self.iteration])
            return reward

    def add_reward(self, op_no, candidate, current):
        super(ClusterRL, self).add_reward(op_no, candidate, current)
        reward = self.get_reward(candidate.cost, current.cost)
        if reward > 0:
            self.update_cluster(op_no, candidate)
            r = reward

            r = r + self.gama * self.hamming_distance(self.clusters[op_no], candidate.solution)
            self.rewards[op_no][self.iteration] += r
            #self.rewards[op_no].append(reward)
            self.timer[op_no] += 1
            #self.iter_rewards[op_no][self.iteration] += reward
            #self.iter_credits[op_no][self.iteration] += credit
            #self.credits[op_no].append(credit)
            #self.iter_rewards[op_no][self.iteration] += r + self.gama * self.hamming_distance(self.clusters[op_no], candidate.solution)

    def update_cluster(self, op, candidate):
        if self.total_succ_counters[op] == 0:
            for i in range(self.algorithm.problem.dimension):
                self.clusters[op][i] = candidate.solution[i]
            return
        for i in range(self.algorithm.problem.dimension):
            self.clusters[op][i] = self.clusters[op][i] / (self.timer[op] + 1) + candidate.solution[i] / (
                        self.timer[op] + 1)

    def hamming_distance(self, x, y):
        return np.count_nonzero(y != (x>0.5))

    def operator_selection(self, candidate):
        if np.all(self.total_succ_counters) == 0:
            for i in range(self.operator_size):
                self.probabilities[i] = 1 / self.operator_size
            return self.roulette_wheel()
        values = [-1 * self.credits[ind][self.iteration] + self.gama * self.hamming_distance(
            self.clusters[ind], candidate.solution) for ind in range(self.operator_size)]
        best_op = np.argmin(values)
        for i in range(self.operator_size):
            if i == best_op:
                self.probabilities[i] = (1 - (self.operator_size - 1) * self.Pmin)
            else:
                self.probabilities[i] = self.Pmin
        op = self.roulette_wheel()
        return op

    def __conf__(self):
        return ['CLRL', self.operator_size,  self.reward_type, self.Pmin, self.W, self.alpha, self.gama]

