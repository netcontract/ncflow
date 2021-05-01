#! /usr/bin/env python

import traceback
import pickle
import os
from itertools import product
import argparse

import sys
sys.path.append("..")
sys.path.append("../..")

from benchmark_consts import PATH_FORM_HYPERPARAMS
from lib.algorithms import POP
from lib.problem import Problem
from lib.graph_utils import check_feasibility

def run_pop(args):
    topo_fname = args.topo_fname
    tm_fname = args.tm_fname
    num_subproblems = args.num_subproblems
    split_method = args.split_method

    problem = Problem.from_file(topo_fname, tm_fname)
    print(problem.name, tm_fname)

    num_paths, edge_disjoint, dist_metric = PATH_FORM_HYPERPARAMS

    pop = POP.new_max_flow(
        num_subproblems,
        split_method,
        num_paths=num_paths,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
    )
    pop.solve(problem)
    sol_dict = pop.sol_dict
    check_feasibility(problem, [sol_dict])
    with open("pop-sol-dict.pkl", "wb") as w:
        pickle.dump(sol_dict, w)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--topo_fname", type=str, required=True)
    parser.add_argument("--tm_fname", type=str, required=True)
    parser.add_argument(
        "--num_subproblems", type=int, choices=[1, 2, 4, 8, 16, 32, 64], required=True
    )
    parser.add_argument(
        "--split_method", type=str, choices=["random", "tailored", "means"], required=True
    )
    args = parser.parse_args()
    run_pop(args)
