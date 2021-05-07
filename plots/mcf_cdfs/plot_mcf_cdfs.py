#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.append("..")
from plot_utils import (
    save_figure,
    print_stats,
    sort_and_set_index,
    change_poisson_in_df,
    filter_by_hyperparams,
    LABEL_NAMES_DICT,
    COLOR_NAMES_DICT,
    LINE_STYLES_DICT,
)

PF_PARAMS = 'num_paths == 4 and edge_disjoint == True and dist_metric == "inv-cap"'


def get_ratio_df(other_df, baseline_df, target_col, suffix):
    join_df = baseline_df.join(
        other_df, how="inner", lsuffix="_baseline", rsuffix=suffix
    ).reset_index()
    results = []
    for _, row in join_df.iterrows():
        target_col_ratio = (
            row[target_col + suffix] / row["{}_baseline".format(target_col)]
        )
        speedup_ratio = row["runtime_baseline"] / row["runtime{}".format(suffix)]
        results.append(
            [
                row["problem"],
                row["tm_model"],
                row["traffic_seed"],
                row["scale_factor"],
                target_col_ratio,
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

    print(df)

    return df


def plot_cdfs(
    vals_list,
    labels,
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

    for vals, label in zip(vals_list, labels):
        vals = sorted([x for x in vals if not np.isnan(x)])
        ax.plot(
            vals,
            np.arange(len(vals)) / len(vals),
            label=LABEL_NAMES_DICT[label] if label in LABEL_NAMES_DICT else label,
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
        ax.set_title(title, y=1.25)
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


def get_ratio_dataframes(path_form_csv, pop_csv, query_str=None):
    # Path Formulation DF
    path_form_df = (
        pd.read_csv(path_form_csv)
        .drop(columns=["num_nodes", "num_edges", "num_commodities"])
        .query(PF_PARAMS)
    )
    path_form_df = sort_and_set_index(path_form_df, drop=True)
    if query_str is not None:
        path_form_df = path_form_df.query(query_str)

    # POP DF
    pop_df = pd.read_csv(pop_csv)
    pop_df = sort_and_set_index(pop_df, drop=True)
    if query_str is not None:
        pop_df = pop_df.query(query_str)

    def get_pop_dfs(pop_parent_df, suffix):
        pop_random_16_df_kdl = pop_parent_df.query(
            'split_method == "random" and num_subproblems == 16 and problem == "Kdl.graphml"'
        )
        pop_random_16_df_non_kdl = pop_parent_df.query(
            'split_method == "random" and num_subproblems == 16 and problem != "Kdl.graphml"'
        )

        return [
            get_ratio_df(df, path_form_df, "obj_val", suffix)
            for df in [
                pop_random_16_df_kdl,
                pop_random_16_df_non_kdl,
            ]
        ]

    return get_pop_dfs(pop_df, "_pop")


def plot_mcf_cdfs(
    curr_dir,
    title="",
    query_str='problem not in ["Uninett2010.graphml", "Ion.graphml", "Interoute.graphml"]',
):
    ratio_dfs = get_ratio_dataframes(curr_dir, query_str)

    pop_random_32_df = ratio_dfs[0]
    pop_random_16_df = ratio_dfs[1]
    pop_random_4_df = ratio_dfs[2]

    pop_means_32_df = ratio_dfs[4]
    pop_means_16_df = ratio_dfs[4]
    pop_means_4_df = ratio_dfs[5]

    # print_stats(pop_random_32_df, "Random, 32", ["flow_ratio", "speedup_ratio"])
    # print_stats(pop_means_32_df, "Power-of-two, 32", ["flow_ratio", "speedup_ratio"])

    # print_stats(pop_random_16_df, "Random, 16", ["flow_ratio", "speedup_ratio"])
    # print_stats(pop_means_16_df, "Power-of-two, 16", ["flow_ratio", "speedup_ratio"])

    # print_stats(pop_random_4_df, "Random, 4", ["flow_ratio", "speedup_ratio"])
    # print_stats(pop_means_4_df, "Power-of-two, 4", ["flow_ratio", "speedup_ratio"])

    # Plot CDFs
    plot_cdfs(
        [
            pop_random_32_df["speedup_ratio"],
            pop_means_32_df["speedup_ratio"],
            pop_random_16_df["speedup_ratio"],
            pop_means_16_df["speedup_ratio"],
            pop_random_4_df["speedup_ratio"],
            pop_means_4_df["speedup_ratio"],
        ],
        [
            "Random, 32",
            "Power-of-two, 32",
            "Random, 16",
            "Power-of-two, 16",
            "Random, 4",
            "Power-of-two, 4",
        ],
        "speedup-cdf-mcf-{}".format(title),
        x_log=True,
        x_label=r"Speedup, relative to PF4 (log scale)",
        bbta=(0, 0, 1, 1.3),
        figsize=(9, 5.5),
        ncol=3,
        title=title,
    )

    plot_cdfs(
        [
            pop_random_32_df["flow_ratio"],
            pop_means_32_df["flow_ratio"],
            pop_random_16_df["flow_ratio"],
            pop_means_16_df["flow_ratio"],
            pop_random_4_df["flow_ratio"],
            pop_means_4_df["flow_ratio"],
        ],
        [
            "Random, 32",
            "Power-of-two, 32",
            "Random, 16",
            "Power-of-two, 16",
            "Random, 4",
            "Power-of-two, 4",
        ],
        "min-frac-flow-cdf-mcf-{}".format(title),
        x_log=False,
        x_label=r"Min Frac. Flow, relative to PF4",
        bbta=(0, 0, 1, 1.3),
        figsize=(9, 5.5),
        ncol=3,
        title=title,
    )


if __name__ == "__main__":
    plot_mcf_cdfs("./")
