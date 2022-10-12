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
import json
from datetime import timedelta
from itertools import zip_longest

import holoviews as hv
import pandas as pd
from constants import (
    ARIMA_RUNS,
    DATA_ROOT,
    DATETIME_FORMAT,
    LIN_RUNS,
    MLP_RUNS,
    RNN_RUNS,
    STATIC_RUNS,
)
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

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

# %% tags=[]
# load .json export files into DFs
df_list = list()

for real_file, pred_file in zip_longest(
    sorted(DATA_ROOT.glob("*-real.json")), sorted(DATA_ROOT.glob("*-pred.json"))
):
    if real_file:
        real_file = real_file.resolve()
    if pred_file:
        pred_file = pred_file.resolve()
    i = int(real_file.name.split("-")[-2])

    if "-as-rnn-" in real_file.name:
        if i in rnn_runs_exclude:
            continue
        if rnn_runs_lim[0] is not None and i < rnn_runs_lim[0]:
            continue
        if rnn_runs_lim[1] is not None and i > rnn_runs_lim[1]:
            continue
        if rnn_runs_lim[0] is None and rnn_runs_lim[1] is None:
            continue
    elif "-as-mlp-" in real_file.name:
        if i in mlp_runs_exclude:
            continue
        if mlp_runs_lim[0] is not None and i < mlp_runs_lim[0]:
            continue
        if mlp_runs_lim[1] is not None and i > mlp_runs_lim[1]:
            continue
        if mlp_runs_lim[0] is None and mlp_runs_lim[1] is None:
            continue
    elif "-as-lin-" in real_file.name:
        if i in lin_runs_exclude:
            continue
        if lin_runs_lim[0] is not None and i < lin_runs_lim[0]:
            continue
        if lin_runs_lim[1] is not None and i > lin_runs_lim[1]:
            continue
        if lin_runs_lim[0] is None and lin_runs_lim[1] is None:
            continue
    elif "-as-aim-" in real_file.name:
        if i in arima_runs_exclude:
            continue
        if arima_runs_lim[0] is not None and i < arima_runs_lim[0]:
            continue
        if arima_runs_lim[1] is not None and i > arima_runs_lim[1]:
            continue
        if arima_runs_lim[0] is None and arima_runs_lim[1] is None:
            continue
    elif "-as-stc-" in real_file.name:
        if i in static_runs_exclude:
            continue
        if static_runs_lim[0] is not None and i < static_runs_lim[0]:
            continue
        if static_runs_lim[1] is not None and i > static_runs_lim[1]:
            continue
        if static_runs_lim[0] is None and static_runs_lim[1] is None:
            continue

    print(f"reading from {real_file} and {pred_file} ...")

    with open(real_file, "r+") as fp:
        real_json_body = json.load(fp)

    real_metric = real_json_body[0]["name"]

    real_df = pd.DataFrame(
        columns=["timestamp", "resource_id", "hostname", real_metric]
    )
    for item in real_json_body:
        resource_id = item["dimensions"]["resource_id"]
        hostname = item["dimensions"]["hostname"]
        measurement_list = item["measurements"]

        real_df = pd.concat(
            [
                real_df,
                pd.DataFrame(
                    [
                        pd.Series(
                            [m[0], resource_id, hostname, m[1]], index=real_df.columns
                        )
                        for m in measurement_list
                    ]
                ),
            ]
        )
    real_df = real_df.astype(
        {
            "resource_id": "string",
            "hostname": "string",
            real_metric: "float64",
        }
    )

    # cast index to DateTimeIndex
    real_df.set_index(["timestamp"], inplace=True)
    real_df.index = pd.to_datetime(real_df.index, format=DATETIME_FORMAT)

    pred_df = None
    if pred_file is not None:
        with open(pred_file, "r+") as fp:
            pred_json_body = json.load(fp)

        pred_metric = pred_json_body[0]["name"]

        pred_df = pd.DataFrame(columns=["timestamp", pred_metric])
        for item in pred_json_body:
            measurement_list = item["measurements"]
            pred_df = pd.concat(
                [
                    pred_df,
                    pd.DataFrame(
                        [
                            pd.Series([m[0], m[1]], index=pred_df.columns)
                            for m in measurement_list
                        ]
                    ),
                ]
            )
        pred_df = pred_df.astype(
            {
                pred_metric: "float64",
            }
        )

        # cast index to DateTimeIndex
        pred_df.set_index(["timestamp"], inplace=True)
        pred_df.index = pd.to_datetime(pred_df.index, format=DATETIME_FORMAT)

    label = real_file.name.split("-real.json")[0]
    df_list.append((label, real_df, pred_df))

# %% tags=[]
fig_list = []
label_list = []
color_cycle = hv.Cycle(
    [
        "#30a2da",
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
opts_scatter_cross = hv.opts.Scatter(size=10, marker="x", tools=["hover"])
opts_scatter_diam = hv.opts.Scatter(size=12, marker="d", tools=["hover"])

for label, real_df, pred_df in df_list:
    traces = []
    index = int(label[-2:])

    if "-rnn-" in label:
        mapping = RNN_RUNS
    elif "-mlp-" in label:
        mapping = MLP_RUNS
    elif "-lin-" in label:
        mapping = LIN_RUNS
    elif "-aim-" in label:
        mapping = ARIMA_RUNS
    elif "-stc-" in label:
        mapping = STATIC_RUNS

    ### data manipulation ###
    table = pd.pivot_table(
        real_df,
        values="cpu.utilization_perc",
        index=["timestamp"],
        columns=["hostname"],
    )
    table = table.resample("1min").mean()

    # reorder columns by VMs start time
    table = table[
        table.apply(pd.Series.first_valid_index).sort_values().index.to_list()
    ]

    orig_cols = table.columns.copy()
    orig_cols_num = len(orig_cols)

    # compute spatial statistics
    table["count"] = (~table.isnull()).iloc[:, 0:orig_cols_num].sum(axis=1)
    table["sum"] = table.iloc[:, 0:orig_cols_num].sum(axis=1)
    table["mean"] = table["sum"] / table["count"]
    table["std"] = (
        ((table.iloc[:, 0:orig_cols_num].subtract(table["mean"], axis=0)) ** 2).sum(
            axis=1
        )
        / table["count"]
    ) ** 0.5

    if pred_df is not None:
        # insert prediction data to align timestamps
        table = table.join(pred_df, how="outer")

        # interpolate missing predictions
        table[pred_metric] = table[pred_metric].interpolate()

        # compute raw prediction values (as initially outputted by the model)
        table["raw_pred"] = table[pred_metric] * table["count"]

    table.reset_index(inplace=True)

    # insert distwalk trace data to align timestamps
    load_file_basename = mapping[index]["load_profile"]
    load_file = (DATA_ROOT / load_file_basename).resolve()
    print(f"reading from {load_file} ...")
    load_df = pd.read_csv(load_file, header=None, names=["distwalk"])
    table = table.join(load_df / 10, how="outer")

    # truncate data & remove NaN-only cols
    if run_time_limit:
        table = table.iloc[:run_time_limit, :].dropna(axis=1, how="all")

    # save to .csv
    csv_dump_file = DATA_ROOT / f"{label}.csv"
    if csv_dump_file.exists():
        print(f"{csv_dump_file} exists, skipping...")
    else:
        print(f"Saving to {csv_dump_file} ...")
        table.to_csv(csv_dump_file, index=False)

    ### plot customization ###
    # plot scale-out threshold
    traces.append(hv.HLine(80).opts(color="black", line_dash="dashed"))

    # plot distwalk trace
    distwalk_trace_label = "distwalk"
    traces.append(
        hv.Scatter(
            (table.index, table["distwalk"].values),
            label=distwalk_trace_label,
        )
        .opts(color=color_cycle)
        .opts(opts_scatter)
    )
    traces.append(
        hv.Curve(
            (table.index, table["distwalk"].values),
            label=distwalk_trace_label,
        ).opts(color=color_cycle)
    )

    # plot metrics observed by VMs
    instance_idx = 0
    for group_label in orig_cols:
        if group_label in table.columns:
            load_trace_label = f"VM {instance_idx}"
            traces.append(
                hv.Scatter(
                    (table.index, table[group_label].values),
                    label=load_trace_label,
                    kdims=[],
                )
                .opts(color=color_cycle)
                .opts(opts_scatter)
            )
            traces.append(
                hv.Curve(
                    (table.index, table[group_label].values),
                    label=load_trace_label,
                ).opts(color=color_cycle)
            )
            instance_idx += 1

    # plot predictor output
    if pred_df is not None:
        prediction_trace_label = "prediction"
        traces.append(
            hv.Scatter(
                (table.index, table[pred_metric].values),
                label=prediction_trace_label,
            )
            .opts(color="#d62728")
            .opts(opts_scatter)
        )
        traces.append(
            hv.Curve(
                (table.index, table[pred_metric].values),
                label=prediction_trace_label,
            ).opts(color="#d62728")
        )

        # compute prediction errors
        raw_pred_df = pd.concat(
            [table[["raw_pred"]].shift(15), table[["sum"]]], axis=1
        ).dropna(axis=0, how="any")

        mape = mean_absolute_percentage_error(
            raw_pred_df["sum"], raw_pred_df["raw_pred"]
        )
        mae = mean_absolute_error(
            raw_pred_df["sum"], raw_pred_df["raw_pred"]
        )
        mse = mean_squared_error(
            raw_pred_df["sum"], raw_pred_df["raw_pred"]
        )
        print(f"MAPE: {mape:.2f} | MAE: {mae:.2f} | MSE: {mse:.2f}")

    title = f"{label} - load: {load_file_basename}"
    input_size = mapping[index].get("input_size")
    vm_delay_min = mapping[index].get("vm_delay_min")
    if input_size:
        title += f" | input size: {input_size}"
    if vm_delay_min:
        title += f" | vm delay: {vm_delay_min}"

    fig = (
        hv.Overlay(traces)
        .opts(
            width=950,
            height=550,
            show_grid=True,
            xlabel="time [min]",
            ylabel="CPU usage [%]",
            legend_position="top_right",
            legend_opts={"background_fill_alpha": 0.5},
            fontsize={
                "title": 13,
                "legend": 16,
                "labels": 20,
                "xticks": 20,
                "yticks": 20,
            },
            padding=0.05,
        )
        .opts(opts)
    )
    fig_list.append(fig)
    label_list.append(label)

layout = hv.Layout(fig_list).cols(1).opts(shared_axes=False)
layout

# %%
