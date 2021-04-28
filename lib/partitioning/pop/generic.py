from .abstract_pop_splitter import AbstractPOPSplitter
from .utils import create_edges_onehot_dict, split_generic


class GenericSplitter(AbstractPOPSplitter):
    def __init__(self, num_subproblems, pf, method="means", verbose=False):
        super().__init__(num_subproblems)
        self._pf = pf
        self.verbose = verbose
        self.method = method

    def split(self, problem):
        input_dict = create_edges_onehot_dict(problem, self._pf, self._num_subproblems)

        # create subproblems, zero out commodities in traffic matrix that aren't assigned to each
        sub_problems = [problem.copy() for _ in range(self._num_subproblems)]
        if self._num_subproblems == 1:
            return sub_problems

        # zero-out the traffic matrices; they will be populated later using entity_assignments_lists
        for sp in sub_problems:
            for u in sp.G.nodes:
                for v in sp.G.nodes:
                    sp.traffic_matrix.tm[u,v] = 0

        entity_assignments_lists = split_generic(
            input_dict, self._num_subproblems, verbose=self.verbose, method=self.method
        )

        for i in range(self._num_subproblems):

            # populate TM for commodities assigned to subproblem i
            for _, source, target, demand in entity_assignments_lists[i]:
                sub_problems[i].traffic_matrix.tm[source, target] += demand

            # split the capacity of each link
            for u, v in sub_problems[i].G.edges:
                sub_problems[i].G[u][v]["capacity"] = (
                    sub_problems[i].G[u][v]["capacity"] / self._num_subproblems
                )

        return sub_problems
