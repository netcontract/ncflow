from ..lp_solver import LpSolver
from ..partitioning.pop import SmartSplitter, BaselineSplitter, GenericSplitter, RandomSplitter
from ..graph_utils import path_to_edge_list
from ..path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles
from ..config import TOPOLOGIES_DIR
from .abstract_formulation import Objective
from .path_formulation import PathFormulation
from gurobipy import GRB, Model, quicksum
from collections import defaultdict
import numpy as np
import math
import random
import re
import os
import time
import pickle

PATHS_DIR = os.path.join(TOPOLOGIES_DIR, 'paths', 'path-form')


class POP(PathFormulation):
    @classmethod
    def new_max_flow(cls, num_subproblems, split_method, num_paths=4, edge_disjoint=True, dist_metric='inv-cap', out=None):
        return cls(objective=Objective.MAX_FLOW,
                   num_subproblems=num_subproblems,
                   split_method=split_method,
                   num_paths=num_paths,
                   edge_disjoint=edge_disjoint,
                   dist_metric=dist_metric,
                   DEBUG=True,
                   VERBOSE=False,
                   out=out)

    @classmethod
    def new_min_max_link_util(cls, num_subproblems, split_method, num_paths=4, edge_disjoint=True, dist_metric='inv-cap', out=None):
        return cls(objective=Objective.MIN_MAX_LINK_UTIL,
                   num_subproblems=num_subproblems,
                   split_method=split_method,
                   num_paths=num_paths,
                   edge_disjoint=edge_disjoint,
                   dist_metric=dist_metric,
                   DEBUG=True,
                   VERBOSE=False,
                   out=out)

    @classmethod
    def new_max_concurrent_flow(cls, num_subproblems, split_method, num_paths=4, edge_disjoint=True, dist_metric='inv-cap', out=None):
        return cls(objective=Objective.MAX_CONCURRENT_FLOW,
                   num_subproblems=num_subproblems,
                   split_method=split_method,
                   num_paths=num_paths,
                   edge_disjoint=edge_disjoint,
                   dist_metric=dist_metric,
                   DEBUG=True,
                   VERBOSE=False,
                   out=out)

    def __init__(self, *, objective, num_subproblems, split_method, num_paths, edge_disjoint, dist_metric, DEBUG, VERBOSE, out=None):
        super().__init__(objective=objective, num_paths=num_paths, edge_disjoint=edge_disjoint,
                         dist_metric=dist_metric, DEBUG=DEBUG, VERBOSE=VERBOSE, out=out)
        self._num_subproblems = num_subproblems
        self._split_method = split_method

    def split_problems(self, problem):
        splitter = None
        if self._split_method == 'skewed':
            splitter = BaselineSplitter(self._num_subproblems)
        elif self._split_method == 'random':
            splitter = RandomSplitter(self._num_subproblems)
        elif self._split_method in ['tailored', 'means', 'covs']:
            pf_original = PathFormulation.get_pf_for_obj(
                self._objective, self._num_paths)
            if self._split_method == 'tailored':
                paths_dict = pf_original.compute_paths(problem)
                splitter = SmartSplitter(self._num_subproblems, paths_dict)
            else:
                splitter = GenericSplitter(
                    self._num_subproblems, pf_original, self._split_method)
        else:
            raise Exception(
                'Invalid split_method {}'.format(self._split_method))

        return splitter.split(problem)

    ###############################
    # Override superclass methods #
    ###############################

    def solve(self, problem):
        self._problem = problem
        self._subproblem_list = self.split_problems(problem)
        self._pfs = [PathFormulation.get_pf_for_obj(
            self._objective, self._num_paths) for subproblem in self._subproblem_list]
        for subproblem, pf in zip(self._subproblem_list, self._pfs):
            pf.solve(subproblem)

    def extract_sol_as_dict(self):
        sol_dict_def = defaultdict(list)
        for var in self.model.getVars():
            if var.varName.startswith('f[') and var.x != 0.0:
                match = re.match(r'f\[(\d+)\]', var.varName)
                p = int(match.group(1))
                sol_dict_def[self.commodity_list[
                    self._path_to_commod[p]]] += [
                        (edge, var.x)
                        for edge in path_to_edge_list(self._all_paths[p])
                ]

        # Set zero-flow commodities to be empty lists
        sol_dict = {}
        sol_dict_def = dict(sol_dict_def)
        for commod_key in self.problem.commodity_list:
            if commod_key in sol_dict_def:
                sol_dict[commod_key] = sol_dict_def[commod_key]
            else:
                sol_dict[commod_key] = []

        return sol_dict

    def extract_sol_as_mat(self):
        edge_idx = self.problem.edge_idx
        sol_mat = np.zeros((len(edge_idx), len(self._path_to_commod)),
                           dtype=np.float32)
        for var in self.model.getVars():
            if var.varName.startswith('f[') and var.x != 0.0:
                match = re.match(r'f\[(\d+)\]', var.varName)
                p = int(match.group(1))
                k = self._path_to_commod[p]
                for edge in path_to_edge_list(self._all_paths[p]):
                    sol_mat[edge_idx[edge], k] += var.x

        return sol_mat

    @property
    def runtime(self):
        return min([pf._solver.model.Runtime for pf in self._pfs])

    @property
    def obj_val(self):
        if self._objective == Objective.MAX_FLOW:
            return sum([pf._solver.model.objVal for pf in self._pfs])
        elif self._objective == Objective.MAX_CONCURRENT_FLOW:
            return min([pf._solver.model.objVal for pf in self._pfs])
