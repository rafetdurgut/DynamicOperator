# -*- coding: utf-8 -*-
"""
Created on Wed May 12 01:28:57 2021

@author: user
"""

from itertools import product
from Problem import *
from Operators import *
from BinaryABC import BinaryABC
from DynamicAOS import *
from AOS import *
from Utilities import Log
import sys
c=dict()
c["Method"] = sys.argv[1]
c["W"] = int(sys.argv[2])
c["Pmin"] = float(sys.argv[3])
c["Alpha"] = float(sys.argv[4])
# c["K"] = int(sys.argv[5])
#
# c["Method"] = 'average'
# c["W"] = 25
# c["Pmin"] = 0.05
# c["Alpha"] = 0.9
# c["K"] = 3

problem = MKP("Benchmarks_GK",3)
#problem = OneMax(2000)
runtime = 30
K = 1
operator_pool = [binABC(), ibinABC(0.3, 0.1), disABC(0.9, 0.1), BABC(),  twoOptABC(), GBABC(), nABC()]
for run in range(runtime):
    operator_selectors = [
        D_AdaptivePursuit(len(operator_pool),K, reward_type=c["Method"], W=c["W"], alpha=c["Alpha"],
                        Pmin=c["Pmin"])
    ]
    for operator_selector in operator_selectors:

        abc = BinaryABC(problem, operator_pool, operator_selector, pop_size=20, maxFE=100000,
                        limit=100)
        for operator in operator_pool:
            operator.set_algorithm(abc)
        operator_selector.set_algorithm(abc)
        abc.run()
        convergence_logs = Log(abc.convergence, 'results', 'cg', abc.operator_selector.__conf__())
        # if abc.operator_selector.type == 'iteration':
        #     credit_logs = Log(abc.operator_selector.credits, 'results', 'credits', abc.operator_selector.__conf__())
        #     reward_logs = Log(abc.operator_selector.rewards, 'results', 'rewards', abc.operator_selector.__conf__())
        #
        # else:
        #     credit_logs = Log(abc.operator_selector.iter_credits, 'results', 'credits', abc.operator_selector.__conf__())
        #     reward_logs = Log(abc.operator_selector.iter_rewards, 'results', 'rewards', abc.operator_selector.__conf__())
        #
        # usage_logs = Log(abc.operator_selector.usage_counter, 'results', 'usage', abc.operator_selector.__conf__())
        # success_logs = Log(abc.operator_selector.success_counter, 'results', 'success', abc.operator_selector.__conf__())
#        active_logs = Log(abc.operator_selector.active_list, 'results', 'actives', abc.operator_selector.__conf__())
        