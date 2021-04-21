from .abstract_pop_splitter import AbstractPOPSplitter
import random
# assign commodities to subproblems at random
class RandomSplitter(AbstractPOPSplitter):
    def __init__(self, num_subproblems):
        super().__init__(num_subproblems)

    def split(self, problem):
        sub_problems = [problem.copy() for _ in range(self._num_subproblems)]
        # num_coms = len(problem.commodity_list)

        for _, (source, target, _) in problem.commodity_list:
            sp_assignment = random.randint(0, self._num_subproblems-1)
            for sp in range(self._num_subproblems):
                if sp == sp_assignment:
                    continue
                sub_problems[sp].traffic_matrix.tm[source, target] = 0
        for sub_problem in sub_problems:
            for u, v in sub_problems[-1].G.edges:
                sub_problem.G[u][v]['capacity'] = sub_problem.G[u][v]['capacity'] / \
                    self._num_subproblems
        return sub_problems
