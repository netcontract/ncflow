
from collections import defaultdict
from .abstract_pop_splitter import AbstractPOPSplitter
from ...graph_utils import path_to_edge_list


class SmartSplitter(AbstractPOPSplitter):
    # paths_dict: key: (source, target), value: array of paths,
    #             where a path is a list of sequential nodes
    #             use lib.graph_utils.path_to_edge_list to get edges.
    def __init__(self, num_subproblems, paths_dict):
        super().__init__(num_subproblems)
        self._paths_dict = paths_dict

    def split(self, problem):
        com_list = problem.commodity_list

        # create dictionary of all edges used by each commodity
        com_path_edges_dict = defaultdict(list)
        for k, (source, target, _) in com_list:
            paths_array = self._paths_dict[(source, target)]
            for path in paths_array:
                com_path_edges_dict[(k, source, target)
                                    ] += list(path_to_edge_list(path))

        # for each edge, split all commodities using that edge across subproblems
        subproblem_com_indices = defaultdict(list)
        current_subproblem = 0
        for (u, v) in problem.G.edges:
            coms_on_edge = [x for x in com_path_edges_dict.keys() if (
                u, v) in com_path_edges_dict[x]]

            # split commodities that share path across all subproblems
            for (k, source, target) in coms_on_edge:
                subproblem_com_indices[current_subproblem] += [
                    (k, source, target)]
                current_subproblem = (
                    current_subproblem + 1) % self._num_subproblems
                # remove commodity from cosideration when processing later edges
                del com_path_edges_dict[(k, source, target)]

        # create subproblems, zero out commodities in traffic matrix that aren't assigned to each
        sub_problems = []
        for i in range(self._num_subproblems):

            sub_problems.append(problem.copy())

            # zero out all commodities not assigned to subproblem i
            for k in subproblem_com_indices.keys():
                if k == i:
                    continue
                zero_out_list = subproblem_com_indices[k]
                for _, source, target in zero_out_list:
                    sub_problems[-1].traffic_matrix.tm[source, target] = 0

            # split the capacity of each link
            for u, v in sub_problems[-1].G.edges:
                sub_problems[-1].G[u][v]['capacity'] = sub_problems[-1].G[u][v]['capacity'] / \
                    self._num_subproblems

        return sub_problems
