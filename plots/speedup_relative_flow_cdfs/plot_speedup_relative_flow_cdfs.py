#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.append("..")
from plot_utils import (
    save_figure,
    change_poisson_in_df,
    filter_by_hyperparams,
    LABEL_NAMES_DICT,
    COLOR_NAMES_DICT,
    LINE_STYLES_DICT,
)

PF_PARAMS = 'num_paths == 4 and edge_disjoint == True and dist_metric == "inv-cap"'


def get_ratio_df(other_df, baseline_df, suffix):
    join_df = baseline_df.join(
        other_df, how="inner", lsuffix="_baseline", rsuffix=suffix
    ).reset_index()
    results = []
    for _, row in join_df.iterrows():
        flow_ratio = row["total_flow{}".format(suffix)] / row["total_flow_baseline"]
        speedup_ratio = row["runtime_baseline"] / row["runtime{}".format(suffix)]
        results.append(
            [
                row["problem"],
                row["tm_model"],
                row["traffic_seed"],
                row["scale_factor"],
                flow_ratio,
                speedup_ratio,
            ]
        )

    df = pd.DataFrame(
        columns=[
            "problem",
            "tm_model",
            "traffic_seed",
            "scale_factor",
            "flow_ratio",
            "speedup_ratio",
        ],
        data=results,
    )
    return df


def join_with_fib_entries(df, fib_entries_df, index_cols):
    return (
        df.set_index(index_cols)
        .join(fib_entries_df)
        .reset_index()
        .set_index(["traffic_seed", "problem", "tm_model"])
    )


def plot_cdfs(
    vals_list,
    labels,
    line_style_keys,
    fname,
    *,
    ax=None,
    title=None,
    x_log=False,
    x_label=None,
    figsize=(6, 12),
    bbta=(0, 0, 1, 1),
    ncol=2,
    xlim=None,
    xticklabels=None,
    add_ylabel=True,
    arrow_coords=None,
    show_legend=True,
    save=True
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for vals, label, line_style_key in zip(vals_list, labels, line_style_keys):
        vals = sorted([x for x in vals if not np.isnan(x)])
        ax.plot(
            vals,
            np.arange(len(vals)) / len(vals),
            label=LABEL_NAMES_DICT[label] if label in LABEL_NAMES_DICT else label,
            linestyle=LINE_STYLES_DICT[line_style_key],
            color=COLOR_NAMES_DICT[line_style_key],
        )
    if add_ylabel:
        ax.set_ylabel("Fraction of Cases")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([0.0, 0.25, 0.50, 0.75, 1.0])
    if x_label:
        ax.set_xlabel(x_label)
    if x_log:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(xlim)
    if title:
        ax.set_title(title)
    if xticklabels:
        if isinstance(xticklabels, tuple):
            xticks, xlabels = xticklabels[0], xticklabels[-1]
        else:
            xticks, xlabels = xticklabels, xticklabels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
    extra_artists = []
    if show_legend:
        legend = ax.legend(
            ncol=ncol, loc="upper center", bbox_to_anchor=bbta, frameon=False
        )
        extra_artists.append(legend)

    if arrow_coords:
        bbox_props = {
            "boxstyle": "rarrow,pad=0.45",
            "fc": "white",
            "ec": "black",
            "lw": 2,
        }
        t = ax.text(
            arrow_coords[0],
            arrow_coords[1],
            "Better",
            ha="center",
            va="center",
            color="black",
            bbox=bbox_props,
        )
        extra_artists.append(t)
    if save:
        save_figure(fname, extra_artists=extra_artists)
    # plt.show()


def sort_and_set_index(df, drop=False):
    return change_poisson_in_df(
        df.reset_index(drop=drop).sort_values(
            by=["problem", "tm_model", "scale_factor", "traffic_seed"]
        )
    ).set_index(["problem", "tm_model", "traffic_seed", "scale_factor"])


def per_iter_to_nc_df(per_iter_fname):
    per_iter_df = filter_by_hyperparams(per_iter_fname).drop(
        columns=[
            "num_nodes",
            "num_edges",
            "num_commodities",
            "partition_runtime",
            "size_of_largest_partition",
            "iteration",
            "r1_runtime",
            "r2_runtime",
            "recon_runtime",
            "r3_runtime",
            "kirchoffs_runtime",
        ]
    )
    nc_iterative_df = per_iter_df.groupby(
        [
            "problem",
            "tm_model",
            "traffic_seed",
            "scale_factor",
            "total_demand",
            "algo",
            "clustering_algo",
            "num_partitions",
            "num_paths",
            "edge_disjoint",
            "dist_metric",
        ]
    ).sum()

    return sort_and_set_index(nc_iterative_df)


def get_ratio_dataframes(curr_dir, query_str=None):
    # Path Formulation DF
    path_form_df = (
        pd.read_csv(curr_dir + "path-form.csv")
        .drop(columns=["num_nodes", "num_edges", "num_commodities"])
        .query(PF_PARAMS)
    )
    path_form_df = sort_and_set_index(path_form_df, drop=True)
    if query_str is not None:
        path_form_df = path_form_df.query(query_str)

    # NC Iterative DF
    nc_iterative_df = per_iter_to_nc_df(curr_dir + "per-iteration.csv")
    if query_str is not None:
        nc_iterative_df = nc_iterative_df.query(query_str)

    # POP DF
    pop_df = pd.read_csv(curr_dir + "pop.csv")
    pop_df = sort_and_set_index(pop_df, drop=True)
    if query_str is not None:
        pop_df = pop_df.query(query_str)

    pop_tailored_64_df = pop_df.query(
        'split_method == "tailored" and num_subproblems == 64'
    )
    pop_tailored_16_df = pop_df.query(
        'split_method == "tailored" and num_subproblems == 16'
    )
    pop_tailored_4_df = pop_df.query(
        'split_method == "tailored" and num_subproblems == 4'
    )

    pop_random_64_df = pop_df.query(
        'split_method == "random" and num_subproblems == 64'
    )
    pop_random_16_df = pop_df.query(
        'split_method == "random" and num_subproblems == 16'
    )
    pop_random_4_df = pop_df.query('split_method == "random" and num_subproblems == 4')

    pop_means_64_df = pop_df.query('split_method == "means" and num_subproblems == 64')
    pop_means_16_df = pop_df.query('split_method == "means" and num_subproblems == 16')
    pop_means_4_df = pop_df.query('split_method == "means" and num_subproblems == 4')

    # Ratio DFs
    nc_ratio_df = get_ratio_df(nc_iterative_df, path_form_df, "_nc")
    return [
        get_ratio_df(df, path_form_df, "_pop")
        for df in [
            pop_tailored_64_df,
            pop_tailored_16_df,
            pop_tailored_4_df,
            pop_random_64_df,
            pop_random_16_df,
            pop_random_4_df,
            pop_means_64_df,
            pop_means_16_df,
            pop_means_4_df,
        ]
    ] + [nc_ratio_df]


def plot_speedup_relative_flow_cdfs(
    curr_dir,
    title="",
    query_str='problem not in ["Uninett2010.graphml", "Ion.graphml", "Interoute.graphml"]',
):
    ratio_dfs = get_ratio_dataframes(curr_dir, query_str)
    # pop_tailored_64_df = ratio_dfs[0]
    pop_tailored_16_df = ratio_dfs[1]
    # pop_tailored_4_df = ratio_dfs[2]
    # pop_random_64_df = ratio_dfs[3]
    pop_random_16_df = ratio_dfs[4]
    # pop_random_4_df = ratio_dfs[5]
    # pop_means_64_df = ratio_dfs[6]
    # pop_means_16_df = ratio_dfs[7]
    # pop_means_4_df = ratio_dfs[8]
    nc_ratio_df = ratio_dfs[9]

    # Print stats
    print(
        "Random 16 Flow ratio vs PF4:\nmin: {},\nmedian: {},\nmean: {},\nmax: {}".format(
            np.min(pop_random_16_df["flow_ratio"]),
            np.median(pop_random_16_df["flow_ratio"]),
            np.mean(pop_random_16_df["flow_ratio"]),
            np.max(pop_random_16_df["flow_ratio"]),
        )
    )
    print()
    if len(pop_tailored_16_df) > 0:
        print(
            "Tailored 16 Flow ratio vs PF4:\nmin: {},\nmedian: {},\nmean: {},\nmax: {}".format(
                np.min(pop_tailored_16_df["flow_ratio"]),
                np.median(pop_tailored_16_df["flow_ratio"]),
                np.mean(pop_tailored_16_df["flow_ratio"]),
                np.max(pop_tailored_16_df["flow_ratio"]),
            )
        )
        print()

    print(
        "Random 16 Speedup ratio vs PF4:\nmin: {},\nmedian: {},\nmean: {},\nmax: {}".format(
            np.min(pop_random_16_df["speedup_ratio"]),
            np.median(pop_random_16_df["speedup_ratio"]),
            np.mean(pop_random_16_df["speedup_ratio"]),
            np.max(pop_random_16_df["speedup_ratio"]),
        )
    )
    print()
    if len(pop_tailored_16_df) > 0:
        print(
            "Tailored 16 Speedup ratio vs PF4:\nmin: {},\nmedian: {},\nmean: {},\nmax: {}".format(
                np.min(pop_tailored_16_df["speedup_ratio"]),
                np.median(pop_tailored_16_df["speedup_ratio"]),
                np.mean(pop_tailored_16_df["speedup_ratio"]),
                np.max(pop_tailored_16_df["speedup_ratio"]),
            )
        )
        print()

    # Plot CDFs
    plot_cdfs(
        [
            nc_ratio_df["speedup_ratio"],
            pop_random_16_df["speedup_ratio"],
            pop_tailored_16_df["speedup_ratio"],
        ],
        ["NCFlow", "Random, 16", "Tailored, 16"],
        ["nc", "smore", "fe"],
        "speedup-cdf-{}".format(title),
        x_log=True,
        x_label=r"Speedup, relative to PF4 (log scale)",
        bbta=(0, 0, 1, 1.4),
        figsize=(9, 4.5),
        ncol=4,
        title=title,
    )

    plot_cdfs(
        [
            nc_ratio_df["flow_ratio"],
            pop_random_16_df["flow_ratio"],
            pop_tailored_16_df["flow_ratio"],
        ],
        ["NCFlow", "Random, 16", "Tailored, 16"],
        ["nc", "smore", "fe"],
        "total-flow-cdf-{}".format(title),
        x_log=False,
        x_label=r"Total Flow, relative to PF4",
        bbta=(0, 0, 1, 1.4),
        figsize=(9, 4.5),
        ncol=4,
        title=title,
    )


if __name__ == "__main__":
    plot_speedup_relative_flow_cdfs("./")
