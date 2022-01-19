#!/bin/bash

set -e
set -o pipefail

predictor_list=("lin" "aim" "mlp" "rnn")

log_file=$(realpath "$1")
log_file_prefix=${log_file%.*}

grep "\"lookback_period_seconds\":\|\"time_aggregation_period_seconds\":\|Loaded model\|overhead \[sec\]:" "$log_file" \
    | cut -d'|' -f5 \
    | sed -e "s/^[[:space:]]*//g; s/[[:space:]]*$//g" \
    | sed -z 's/\nProcessing/ Processing/g' > "${log_file_prefix}-times.log"

for predictor in "${predictor_list[@]}"; do
    echo "input,total (sec),processing (sec)" > "${log_file_prefix}-times-${predictor}.csv"
done

cat "${log_file_prefix}-times.log" | while read -r line; do
    if [[ $line == *"lookback_period_seconds"* ]]; then
        lookback_period=$(echo "$line" | cut -d':' -f2 | sed -e "s/^[[:space:]]*//g; s/[[:space:]]*$//g; s/,$//")
        predictor_type=

    elif [[ $line == *"time_aggregation_period_seconds"* ]]; then
        time_aggregation=$(echo "$line" | cut -d':' -f2 | sed -e "s/^[[:space:]]*//g; s/[[:space:]]*$//g; s/,$//")

    elif [[ $line == *"Loaded model of type"* ]]; then
        if [[ $line == *"RNN"* ]]; then
            predictor_type="rnn"
        elif [[ $line == *"MLP"* ]]; then
            predictor_type="mlp"
        elif [[ $line == *"ARIMA"* ]]; then
            predictor_type="aim"
        fi

    elif [[ $line == *"overhead"* ]]; then
        times=$(echo "$line" | sed -e "s/Total overhead \[sec\]://; s/ Processing overhead \[sec\]: /,/g; s/[[:space:]]//g")
        input_size=$((lookback_period / time_aggregation))
        echo "$input_size,$times" >> "${log_file_prefix}-times-${predictor_type:-"lin"}.csv"
    fi
done
