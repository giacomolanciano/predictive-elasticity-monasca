"""Costant values"""

from pathlib import Path

DATA_ROOT = Path("../data/")
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

STATIC_RUNS = {
    60: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-08T15:21:26Z",
        "end_real": "2021-10-08T19:09:15Z",
        "vm_delay_min": 10,
    },
}

RNN_RUNS = {
    26: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-01T13:24:39Z",
        "end_real": "2021-10-01T17:06:00Z",
        "model": "rnn-20_sum_2021-07-22.pt",
        "scaler": "rnn_scaler.joblib",
        "input_size": 20,
        "vm_delay_min": 10,
    },
    29: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-08T06:41:20Z",
        "end_real": "2021-10-08T10:28:01Z",
        "model": "rnn-05_sum_2021-09-26.pt",
        "scaler": "rnn_scaler.joblib",
        "input_size": 5,
        "vm_delay_min": 10,
    },
    30: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-09T07:28:51Z",
        "end_real": "2021-10-09T11:10:39Z",
        "model": "rnn-10_sum_2021-09-26.pt",
        "scaler": "rnn_scaler.joblib",
        "input_size": 10,
        "vm_delay_min": 10,
    },
}

LIN_RUNS = {
    9: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-01T08:47:22Z",
        "end_real": "2021-10-01T12:29:24Z",
        "scaler": "scaler.joblib",
        "input_size": 20,
        "vm_delay_min": 10,
    },
    11: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-06T14:51:47Z",
        "end_real": "2021-10-06T18:33:01Z",
        "scaler": "scaler.joblib",
        "input_size": 10,
        "vm_delay_min": 10,
    },
    13: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-07T16:43:18Z",
        "end_real": "2021-10-07T20:25:04Z",
        "scaler": "scaler.joblib",
        "input_size": 5,
        "vm_delay_min": 10,
    },
}

MLP_RUNS = {
    7: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-05T10:49:54Z",
        "end_real": "2021-10-05T14:31:26Z",
        "model": "mlp-20_sum_2021-07-20.pt",
        "scaler": "scaler.joblib",
        "input_size": 20,
        "vm_delay_min": 10,
    },
    9: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-06T22:34:35Z",
        "end_real": "2021-10-07T02:15:12Z",
        "model": "mlp-10_sum_2021-09-24.pt",
        "scaler": "scaler.joblib",
        "input_size": 10,
        "vm_delay_min": 10,
    },
    10: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-10-07T21:02:31Z",
        "end_real": "2021-10-08T00:53:02Z",
        "model": "mlp-05_sum_2021-09-24.pt",
        "scaler": "scaler.joblib",
        "input_size": 5,
        "vm_delay_min": 10,
    },
}

ARIMA_RUNS = {
    4: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-12-20T12:53:51Z",
        "end_real": "2021-12-20T16:35:22Z",
        "model": "arima-20-1-0_sum_2021-09-15.sm",
        "scaler": "scaler.joblib",
        "input_size": 20,
        "vm_delay_min": 10,
    },
    5: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-12-20T17:13:48Z",
        "end_real": "2021-12-20T20:55:16Z",
        "model": "arima-10-1-0_sum_2021-12-20.sm",
        "scaler": "scaler.joblib",
        "input_size": 10,
        "vm_delay_min": 10,
    },
    6: {
        "load_profile": "test_behavior_03_distwalk-6t_+10.dat",
        "start_real": "2021-12-21T09:28:35Z",
        "end_real": "2021-12-21T13:09:56Z",
        "model": "arima-05-1-0_sum_2021-12-20.sm",
        "scaler": "scaler.joblib",
        "input_size": 5,
        "vm_delay_min": 10,
    },
}
