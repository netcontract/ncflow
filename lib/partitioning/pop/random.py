from .abstract_pop_splitter import AbstractPOPSplitter
import random
import numpy as np
from .entity_splitting import split_entities

# assign commodities to subproblems at random
class RandomSplitter(AbstractPOPSplitter):
    def __init__(self, num_subproblems, split_fraction=0.1):
        super().__init__(num_subproblems)
        self.split_fraction = split_fraction

    def split(self, problem):
        sub_problems = [problem.copy() for _ in range(self._num_subproblems)]
        # zero-out the traffic matrices; they will be populated at random using commodity list
        for sp in sub_problems:
            for u in sp.G.nodes:
                for v in sp.G.nodes:
                    sp.traffic_matrix.tm[u,v] = 0 

        entity_list = [[k, u, v, d] for (k, (u, v, d)) in problem.commodity_list]

        split_entity_list = split_entities(entity_list, self.split_fraction)
        for [_, source, target, demand] in split_entity_list:
           
            # sample sp to assign commodity to
            sp_assignment = random.randint(0, self._num_subproblems - 1)
            sub_problems[sp_assignment].traffic_matrix.tm[source, target] += demand
        
        for sub_problem in sub_problems:
            for u, v in sub_problems[-1].G.edges:
                sub_problem.G[u][v]["capacity"] = (
                    sub_problem.G[u][v]["capacity"] / self._num_subproblems
                )
        return sub_problems
