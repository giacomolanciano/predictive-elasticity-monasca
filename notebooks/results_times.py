# ---
# jupyter:
#   jupytext:
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pred-ops-os
#     language: python
#     name: pred-ops-os
# ---

# %%
import sys
from datetime import datetime, timedelta

import holoviews as hv
import numpy as np
import pandas as pd
from constants import (
    ARIMA_RUNS,
    DATA_ROOT,
    LIN_RUNS,
    MLP_RUNS,
    RNN_RUNS,
    STATIC_RUNS,
)
from holoviews.operation.datashader import datashade, dynspread, rasterize

hv.extension("bokeh")
pd.options.plotting.backend = "holoviews"

# %%
static_runs_lim = (60, None)
static_runs_exclude = set([61, 62])
lin_runs_lim = (9, None)
lin_runs_exclude = set([10, 12, 14])
mlp_runs_lim = (7, None)
mlp_runs_exclude = set([8])
rnn_runs_lim = (26, None)
rnn_runs_exclude = set([27, 28, 31, 32, 33])
arima_runs_lim = (4, None)
arima_runs_exclude = set([])

run_time_limit = None

# %% [markdown]
# ## Client-side response times plots

# %%
color_cycle = hv.Cycle(
    [
        "#d62728",
        "#e5ae38",
        "#6d904f",
        "#8b8b8b",
        "#17becf",
        "#9467bd",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#1f77b4",
    ]
)
opts = [hv.opts.Curve(tools=["hover"])]
opts_scatter = hv.opts.Scatter(size=5, marker="o", tools=["hover"])

# %%
descr_stats_table = pd.DataFrame()
descr_stats_table_peak_1 = pd.DataFrame()
descr_stats_table_peak_2 = pd.DataFrame()
descr_stats_table_peak_real = pd.DataFrame()
times_fig_list = []
times_label_list = []
for times_file in sorted(DATA_ROOT.glob("*-times.csv")):
    ## filter-out excluded result files
    i = int(times_file.name.split("-")[-2])
    if "-as-rnn-" in times_file.name:
        if i in rnn_runs_exclude:
            continue
        if rnn_runs_lim[0] is not None and i < rnn_runs_lim[0]:
            continue
        if rnn_runs_lim[1] is not None and i > rnn_runs_lim[1]:
            continue
        if rnn_runs_lim[0] is None and rnn_runs_lim[1] is None:
            continue
        label = "RNN"
        mapping = RNN_RUNS
    elif "-as-mlp-" in times_file.name:
        if i in mlp_runs_exclude:
            continue
        if mlp_runs_lim[0] is not None and i < mlp_runs_lim[0]:
            continue
        if mlp_runs_lim[1] is not None and i > mlp_runs_lim[1]:
            continue
        if mlp_runs_lim[0] is None and mlp_runs_lim[1] is None:
            continue
        label = "MLP"
        mapping = MLP_RUNS
    elif "-as-lin-" in times_file.name:
        if i in lin_runs_exclude:
            continue
        if lin_runs_lim[0] is not None and i < lin_runs_lim[0]:
            continue
        if lin_runs_lim[1] is not None and i > lin_runs_lim[1]:
            continue
        if lin_runs_lim[0] is None and lin_runs_lim[1] is None:
            continue
        label = "LR"
        mapping = LIN_RUNS
    elif "-as-aim-" in times_file.name:
        if i in arima_runs_exclude:
            continue
        if arima_runs_lim[0] is not None and i < arima_runs_lim[0]:
            continue
        if arima_runs_lim[1] is not None and i > arima_runs_lim[1]:
            continue
        if arima_runs_lim[0] is None and arima_runs_lim[1] is None:
            continue
        label = "ARIMA"
        mapping = ARIMA_RUNS
    elif "-as-stc-" in times_file.name:
        if i in static_runs_exclude:
            continue
        if static_runs_lim[0] is not None and i < static_runs_lim[0]:
            continue
        if static_runs_lim[1] is not None and i > static_runs_lim[1]:
            continue
        if static_runs_lim[0] is None and static_runs_lim[1] is None:
            continue
        label = "Static"
        mapping = STATIC_RUNS

    traces = []

    print(times_file)
    df = pd.read_csv(times_file, header=None, names=["timestamp", "delay"])

    # drop rows containing 0 beacuse:
    # - timestamp == 0 means the request was never sent
    # - delay == 0 means the response was never received
    len_before = len(df)
    df.drop(df[(df["timestamp"] == 0) | (df["delay"] == 0)].index, inplace=True)
    len_after = len(df)
    print(f"dropped {len_before - len_after}/{len_before} rows.")

    # set datetimeindex for easy resampling
    df["index"] = pd.to_datetime(df["timestamp"], unit="us")
    df.set_index("index", inplace=True)

    # convert microsec to millisec
    df = df / 1000

    # convert timestamp to min
    df["timestamp"] = df["timestamp"] / 1000 / 60

    if run_time_limit:
        df = df[df["timestamp"] < run_time_limit]

    # estract run parameters
    run_params = {
        "run_type": label,
        "input_size": mapping[i].get("input_size"),
        "vm_delay_min": mapping[i].get("vm_delay_min"),
        "anomaly": mapping[i].get("anomaly"),
        "load_profile": mapping[i].get("load_profile"),
        "raw_data": f"{times_file.name}",
    }

    ## compute percentiles
    # overall stats
    descr_stats_table = pd.concat(
        [
            descr_stats_table,
            pd.DataFrame(
                [
                    {
                        **run_params,
                        "avg (ms)": df["delay"].mean(),
                        "p50 (ms)": df["delay"].quantile(0.5),
                        "p90 (ms)": df["delay"].quantile(0.9),
                        "p95 (ms)": df["delay"].quantile(0.95),
                        "p99 (ms)": df["delay"].quantile(0.99),
                        "p99.5 (ms)": df["delay"].quantile(0.995),
                        "p99.9 (ms)": df["delay"].quantile(0.999),
                        # "max (ms)": df["delay"].max(),
                    }
                ],
            ),
        ]
    )

    # peak real data
    df_peak_real = df[df["timestamp"].between(45, 90)]
    descr_stats_table_peak_real = pd.concat(
        [
            descr_stats_table_peak_real,
            pd.DataFrame(
                [
                    {
                        **run_params,
                        "avg (ms)": df_peak_real["delay"].mean(),
                        "p50 (ms)": df_peak_real["delay"].quantile(0.5),
                        "p90 (ms)": df_peak_real["delay"].quantile(0.9),
                        "p95 (ms)": df_peak_real["delay"].quantile(0.95),
                        "p99 (ms)": df_peak_real["delay"].quantile(0.99),
                        "p99.5 (ms)": df_peak_real["delay"].quantile(0.995),
                        "p99.9 (ms)": df_peak_real["delay"].quantile(0.999),
                        # "max (ms)": df_peak_real["delay"].max(),
                    }
                ],
            ),
        ]
    )

    # 1st peak stats
    df_peak_1 = df[df["timestamp"].between(0, 120)]
    descr_stats_table_peak_1 = pd.concat(
        [
            descr_stats_table_peak_1,
            pd.DataFrame(
                [
                    {
                        **run_params,
                        "avg (ms)": df_peak_1["delay"].mean(),
                        "p50 (ms)": df_peak_1["delay"].quantile(0.5),
                        "p90 (ms)": df_peak_1["delay"].quantile(0.9),
                        "p95 (ms)": df_peak_1["delay"].quantile(0.95),
                        "p99 (ms)": df_peak_1["delay"].quantile(0.99),
                        "p99.5 (ms)": df_peak_1["delay"].quantile(0.995),
                        "p99.9 (ms)": df_peak_1["delay"].quantile(0.999),
                        # "max (ms)": df_peak_1["delay"].max(),
                    }
                ],
            ),
        ]
    )

    # 2nd peak stats
    df_peak_2 = df[df["timestamp"].between(121, 220)]
    descr_stats_table_peak_2 = pd.concat(
        [
            descr_stats_table_peak_2,
            pd.DataFrame(
                [
                    {
                        **run_params,
                        "avg (ms)": df_peak_2["delay"].mean(),
                        "p50 (ms)": df_peak_2["delay"].quantile(0.5),
                        "p90 (ms)": df_peak_2["delay"].quantile(0.9),
                        "p95 (ms)": df_peak_2["delay"].quantile(0.95),
                        "p99 (ms)": df_peak_2["delay"].quantile(0.99),
                        "p99.5 (ms)": df_peak_2["delay"].quantile(0.995),
                        "p99.9 (ms)": df_peak_2["delay"].quantile(0.999),
                        # "max (ms)": df_peak_2["delay"].max(),
                    }
                ],
            ),
        ]
    )

    ## filter out outliers before plotting
    df = df[df["delay"] > 0]
    df = df[df["delay"] <= df["delay"].quantile(0.999)]

    # rolling stats
    quantiles = [
        0.5,
        0.9,
        # 0.95,
        0.99,
        # 0.995,
        # 0.999,
    ]
    traces = []
    for q in quantiles:
        df_res = df["delay"].resample("5min", closed="right", label="right").quantile(q)
        df_res.index = [x * 5 for x in range(1, df_res.index.size + 1)]

        q_label = f"5-mins p{q*100:g}"
        traces.append(
            hv.Scatter(
                (df_res.index, df_res.values),
                label=q_label,
            )
            .opts(color=color_cycle)
            .opts(opts_scatter)
        )
        traces.append(
            hv.Curve(
                (df_res.index, df_res.values),
                label=q_label,
            ).opts(color=color_cycle)
        )

    ## build scatter plot
    # traces.append(
    #     rasterize(
    #         hv.Scatter(
    #             (df["timestamp"].values, df["delay"].values),
    #         ),
    #         pixel_ratio=2,
    #     ).opts(
    #         cmap="Blues",
    #         # cmap="bkr",
    #         cnorm="eq_hist",
    #         rescale_discrete_levels=True,
    #     )
    # )
    times_fig = (
        hv.Overlay(traces)
        .opts(
            width=950,
            height=550,
            show_grid=True,
            # title=f"{times_file.name}",
            xlabel="time [min]",
            ylabel="delay [ms]",
            legend_position="top_right",
            fontsize={
                "title": 13,
                "legend": 16,
                "labels": 20,
                "xticks": 20,
                "yticks": 20,
            },
            logy=True,
        )
        .opts(opts)
    )
    # times_fig = times_fig.collate() # NOTE: decomment when including rasterized scatterplot
    times_fig_list.append(times_fig)
    times_label_list.append(times_file.stem)

descr_stats_table.set_index(["run_type", "input_size"], inplace=True)
descr_stats_table.sort_index(inplace=True)

descr_stats_table_peak_real.set_index(["run_type", "input_size"], inplace=True)
descr_stats_table_peak_real.sort_index(inplace=True)

descr_stats_table_peak_1.set_index(["run_type", "input_size"], inplace=True)
descr_stats_table_peak_1.sort_index(inplace=True)

descr_stats_table_peak_2.set_index(["run_type", "input_size"], inplace=True)
descr_stats_table_peak_2.sort_index(inplace=True)

# times_layout = datashade(hv.Layout(times_fig_list).cols(1).opts(shared_axes=False))
times_layout = hv.Layout(times_fig_list).cols(1).opts(shared_axes=False)
times_layout

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Client-side response times tables

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Overall

# %% tags=[]
descr_stats_table.round(2)

# %%
printable_table = descr_stats_table[
    descr_stats_table.filter(regex=f"\(ms\)$", axis=1).columns.to_list()
].sort_index()
printable_table.index = [
    f"{run_type} ({input_size:02.0f})".replace(" (nan)", "")
    for run_type, input_size in printable_table.index.to_flat_index()
]
printable_table.columns = [c.replace(" (ms)", "") for c in printable_table.columns]

col_fmt = "r" + "r" * printable_table.columns.size
print(
    printable_table.style.format(precision=2)
    .format_index(escape="latex", axis=1)
    .format_index(escape="latex", axis=0)
    .hide(axis=0, names=True)
    .to_latex(column_format=col_fmt, hrules=True, sparse_index=False)
)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Real data peak

# %% tags=[]
# focus on real data peak
descr_stats_table_peak_real.round(2)

# %%
printable_table = descr_stats_table_peak_real[
    descr_stats_table_peak_real.filter(regex=f"\(ms\)$", axis=1).columns.to_list()
].sort_index()
printable_table.index = [
    f"{run_type} ({input_size:02.0f})".replace(" (nan)", "")
    for run_type, input_size in printable_table.index.to_flat_index()
]
printable_table.columns = [c.replace(" (ms)", "") for c in printable_table.columns]

col_fmt = "r" + "r" * printable_table.columns.size
print(
    printable_table.style.format(precision=2)
    .format_index(escape="latex", axis=1)
    .format_index(escape="latex", axis=0)
    .hide(axis=0, names=True)
    .to_latex(column_format=col_fmt, hrules=True, sparse_index=False)
)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Synthetic data 1st peak

# %% tags=[]
# render client-side delays stats table (focus on 1st peak)
descr_stats_table_peak_1.round(2)

# %%
printable_table = descr_stats_table_peak_1[
    descr_stats_table_peak_1.filter(regex=f"\(ms\)$", axis=1).columns.to_list()
].sort_index()
printable_table.index = [
    f"{run_type} ({input_size:02.0f})".replace(" (nan)", "")
    for run_type, input_size in printable_table.index.to_flat_index()
]
printable_table.columns = [c.replace(" (ms)", "") for c in printable_table.columns]
# printable_table

col_fmt = "r" + "r" * printable_table.columns.size
print(
    printable_table.style.format(precision=2)
    .format_index(escape="latex", axis=1)
    .format_index(escape="latex", axis=0)
    .hide(axis=0, names=True)
    .to_latex(column_format=col_fmt, hrules=True, sparse_index=False)
)

# %% [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Synthetic data 2nd peak

# %% tags=[]
# render client-side delays stats table (focus on 2nd peak)
descr_stats_table_peak_2.round(2)

# %%
printable_table = descr_stats_table_peak_2[
    descr_stats_table_peak_2.filter(regex=f"\(ms\)$", axis=1).columns.to_list()
].sort_index()
printable_table.index = [
    f"{run_type} ({input_size:02.0f})".replace(" (nan)", "")
    for run_type, input_size in printable_table.index.to_flat_index()
]
printable_table.columns = [c.replace(" (ms)", "") for c in printable_table.columns]

col_fmt = "r" + "r" * printable_table.columns.size
print(
    printable_table.style.format(precision=2)
    .format_index(escape="latex", axis=1)
    .format_index(escape="latex", axis=0)
    .hide(axis=0, names=True)
    .to_latex(column_format=col_fmt, hrules=True, sparse_index=False)
)

# %%
