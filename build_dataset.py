from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.signal import find_peaks
from scipy.stats import iqr


# =========================
# Helper functions
# =========================

def sliding_window(signal, fs, window_sec, step_sec):
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    for start in range(0, len(signal) - win + 1, step):
        yield signal[start:start + win]


def compute_hr(ecg, fs):
    peaks, _ = find_peaks(ecg, distance=fs * 0.6)
    if len(peaks) < 2:
        return np.nan
    rr_intervals = np.diff(peaks) / fs
    hr = 60.0 / np.mean(rr_intervals)
    return hr


'''def compute_hrv(ecg, fs):
    peaks, _ = find_peaks(ecg, distance=fs * 0.6)
    if len(peaks) < 2:
        return np.nan
    rr = np.diff(peaks) / fs
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))
    return rmssd * 1000  # ms'''


def compute_hrv_features(ecg, fs):
    ecg = np.asarray(ecg).flatten()  # 防止1D错误🔥

    peaks, _ = find_peaks(ecg, distance=fs * 0.6)

    if len(peaks) < 3:
        return np.nan, np.nan

    rr = np.diff(peaks) / fs * 1000  # ms

    # RMSSD
    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))

    # SDNN
    sdnn = np.std(rr)

    return rmssd, sdnn


def compute_resp_rate(resp, fs):
    peaks, _ = find_peaks(resp, distance=fs * 1.0)
    if len(peaks) < 2:
        return np.nan
    intervals = np.diff(peaks) / fs
    return 60.0 / np.mean(intervals)


# =========================
# Main builder
# =========================

def build_feature_table(settings) -> pd.DataFrame:
    eps = np.finfo(float).eps

    data_dir: Path = settings.data_dir
    window_sec = settings.window_seconds
    step_sec = settings.step_seconds

    rows = []

    subjects = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    for subj_path in subjects:
        subj_id = subj_path.name
        file_path = subj_path / f"{subj_id}.pkl"

        if not file_path.exists():
            continue

        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

            print("顶层 keys：", data.keys())

            print("\nsignal 层：", data["signal"].keys())

            print("\nchest 传感器：", data["signal"]["chest"].keys())

            #print(data.keys())

        try:
            chest = data["signal"]["chest"]
            #ecg = chest["ECG"]
            #resp = chest["Resp"]
            ecg = chest["ECG"].squeeze()
            resp = chest["Resp"].squeeze()

            '''eda = chest["EDA"].squeeze()#'''

            labels = data["label"]
        except Exception:
            continue

        fs = 700  # WESAD chest sampling rate
        for ecg_win, resp_win, label_win in zip(
            sliding_window(ecg, fs, window_sec, step_sec),
            sliding_window(resp, fs, window_sec, step_sec),
            sliding_window(labels, fs, window_sec, step_sec),
        ):


            # majority label
            label = int(pd.Series(label_win).mode()[0])

            # only keep baseline(1) and stress(2)
            if label not in [1, 2]:
                continue

            '''hr = compute_hr(ecg_win, fs)
            hrv = compute_hrv(ecg_win, fs)
            rr = compute_resp_rate(resp_win, fs)'''

            hr = compute_hr(ecg_win, fs)
            rmssd, sdnn = compute_hrv_features(ecg_win, fs)
            resp_rate = compute_resp_rate(resp_win, fs)

            '''eda_mean = np.mean(eda_win)#
            eda_std = np.std(eda_win)#'''

            '''if np.isnan(hr) or np.isnan(hrv) or np.isnan(rr):
                continue'''

            if np.isnan(hr) or np.isnan(rmssd) or np.isnan(sdnn) or np.isnan(resp_rate):
                continue

            '''if (
                    np.isnan(hr)
                    or np.isnan(rmssd)
                    or np.isnan(sdnn)
                    or np.isnan(resp_rate)
                    or np.isnan(eda_mean)#
                    or np.isnan(eda_std)#
            ):
                continue'''

            '''rows.append({
                "subject": subj_id,
                "label": label,
                "hr_mean_bpm": hr,
                "rmssd_ms": hrv,
                "resp_rate_mean_bpm": rr,
            })'''

            rows.append({
                "subject": subj_id,
                "label": label,

                "hr_mean_bpm": hr,
                "rmssd_ms": rmssd,
                "sdnn_ms": sdnn,  # 👈 新加🔥
                "resp_rate_mean_bpm": resp_rate,

                # =====================
                # 比值特征（核心🔥）
                # =====================
                "hr_div_rmssd": hr / (rmssd + eps),
                "hr_div_sdnn": hr / (sdnn + eps),
                "rmssd_div_sdnn": rmssd / (sdnn + eps),
                "hr_div_resp": hr / (resp_rate + eps),

            })

            '''rows.append({
                "subject": subj_id,
                "label": label,
                "hr_mean_bpm": hr,
                "rmssd_ms": rmssd,
                "sdnn_ms": sdnn,
                "resp_rate_mean_bpm": resp_rate,
                "eda_mean": eda_mean,  #
                "eda_std": eda_std,  #
            })'''

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # label mapping
    df["label_name"] = df["label"].map({1: "baseline", 2: "stress"})

    return df
