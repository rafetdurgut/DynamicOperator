
import numpy as np


class abstractOperatorSelection:
    def __init__(self, pool_size, operator_size, reward_type="insta", W=5, alpha=0.1, beta=0.5, Pmin=0.1):
        self.rewards = [[0] for _ in range(pool_size)]
        self.credits = [[0] for _ in range(pool_size)]
        self.w_credits = [[0] for _ in range(pool_size)]

        self.active_operators =np.random.permutation(range(pool_size))[0:pool_size] 
        self.success_counter = [[0] for _ in range(pool_size)]
        self.total_succ_counters = np.zeros((pool_size))
        self.active_list = []
        self.usage_counter = [[0] for _ in range(pool_size)]
        self.probabilities = np.zeros((pool_size))
        self.reward = np.zeros((pool_size))
        self.type = 'iteration'
        self.iteration = 0
        self.operator_size = operator_size
        self.pool_size = pool_size
        self.reward_type = reward_type
        self.W = W
        self.Pmin = Pmin
        self.Pmax = 1 - (self.operator_size - 1) * Pmin
        self.alpha = alpha
        self.beta = beta

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def get_reward(self, new_fitness, old_fitness):
        r =   float((new_fitness - old_fitness))
        if r <= 0:
            r = 0
        return ( r  * (self.algorithm.problem.dimension / self.algorithm.global_best.cost))
    
    def add_operator(self, remove_list):
        
        if  len(self.active_operators) > self.operator_size and len(self.active_operators)-len(remove_list) < self.operator_size:
            self.active_operators = np.delete(self.active_operators,remove_list)
            
            while len(self.active_operators) < self.operator_size:
                possible_operator = [ind for ind in range(self.pool_size) if ind not in self.active_operators]
                self.active_operators = np.append(self.active_operators, np.random.choice(possible_operator) )
            return
        if len(self.active_operators) > self.operator_size:
            self.active_operators = np.delete(self.active_operators,remove_list)
            return
        for i in range(len(remove_list)):
            possible_operator = [ind for ind in range(self.pool_size) if ind not in self.active_operators]
            if len(possible_operator)>0:
                self.active_operators[i] = np.random.choice(possible_operator)
        
        
    def determine_operators(self):
        if self.iteration <= self.W:
            return
        remove_list = []
        for i in range(len(self.active_operators)):
            if np.all([v==0 for v in self.success_counter[self.active_operators[i]][-self.W:]]):
                if np.all([v>0 for v in self.usage_counter[self.active_operators[i]][-self.W:]]):                   
                    remove_list.append(i)
        if remove_list:
            self.add_operator(remove_list)
                    
    def next_iteration(self):
        self.active_list.append(self.active_operators)
        self.update_credits()
        self.iteration += 1
        self.determine_operators()
        for i in range(len(self.active_operators)):
            self.rewards[self.active_operators[i]].append(0)
            
        for i in range(self.pool_size):
            self.usage_counter[i].append(0)
            self.success_counter[i].append(0)

    def add_reward(self, op_no, candidate, current):
        self.usage_counter[op_no][self.iteration] += 1
        reward = self.get_reward(candidate.cost, current.cost)
        if reward > 0:
            self.success_counter[op_no][self.iteration] += 1
            self.total_succ_counters[op_no] += 1
            if self.type == 'iteration':
                self.rewards[op_no][-1] += reward

    def apply_rewards(self):
        for i in range(len(self.active_operators)):
            if self.reward_type == "insta":
                self.reward[self.active_operators[i]] = self.rewards[self.active_operators[i]][self.iteration]/(self.usage_counter[self.active_operators[i]][self.iteration]+1)
            elif self.reward_type == "average":
                start_pos = max(0, len(self.rewards[self.active_operators[i]]) - self.W)
                reward = np.average(self.rewards[self.active_operators[i]][start_pos:len(self.rewards[self.active_operators[i]])])/(self.usage_counter[self.active_operators[i]][self.iteration]+1)
                self.reward[self.active_operators[i]] = reward
            elif self.reward_type == "extreme":
                start_pos = max(0, len(self.rewards[self.active_operators[i]]) - self.W)
                reward = np.max(self.rewards[self.active_operators[i]][start_pos:len(self.rewards[self.active_operators[i]])])/(self.usage_counter[self.active_operators[i]][self.iteration]+1)
                self.reward[self.active_operators[i]] = reward

    def update_credits(self):
        self.apply_rewards()
        for i in range(self.pool_size):            
            if i not in self.active_operators:
                credit = (1 - self.alpha) * self.credits[i][self.iteration]
            else:
                credit = (1 - self.alpha) * self.credits[i][self.iteration] + self.alpha * self.reward[i]
            
            self.credits[i].append(credit)

                

    def operator_selection(self, candidate=None):
        raise Exception("Should not call Abstract Class!")

    def roulette_wheel(self):
        sumProbs = sum(a for ind,a in enumerate(self.probabilities) if ind in self.active_operators)
        probs = [a / sumProbs for ind,a in enumerate(self.probabilities) if ind in self.active_operators]
        op = np.random.choice(len(probs), p=probs)
        return self.active_operators[op]


class D_AdaptivePursuit(abstractOperatorSelection):
    def operator_selection(self, candidate):
        if self.iteration == 0 or np.all(self.total_succ_counters[self.active_operators]) == 0:
            for i in range(len(self.active_operators)):
                self.probabilities[self.active_operators] = 1 / self.operator_size
            return self.roulette_wheel()
        
        
        credits = [a[-1] for ind,a in enumerate(self.credits) if ind in self.active_operators]
        best_op = self.active_operators[np.argmax(credits)]
        for i in range(len(self.active_operators)):
            if self.active_operators[i] == best_op:
                self.probabilities[self.active_operators[i]] = self.probabilities[self.active_operators[i]] + self.beta * (
                        self.Pmax - self.probabilities[self.active_operators[i]])
            else:
                self.probabilities[self.active_operators[i]] = self.probabilities[self.active_operators[i]] + self.beta * (
                        self.Pmin - self.probabilities[self.active_operators[i]])
        return self.roulette_wheel()

    def __conf__(self):
        return ['AP', self.operator_size,  self.reward_type, self.Pmin, self.W, self.alpha]

