import autorootcwd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, medfilt, stft
from scipy.linalg import eigh
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import kurtosis, skew
import random

def compute_high_frequency_feature(signal_1d, fs=100, hf_band=(10, 20), nperseg=64, noverlap=32):
    f, t, Zxx = stft(signal_1d, fs=fs, nperseg=nperseg, noverlap=noverlap)
    hf_mask = (f >= hf_band[0]) & (f <= hf_band[1])
    hf_energy = np.sum(np.abs(Zxx[hf_mask, :]) ** 2, axis=0)
    return t, hf_energy, f[hf_mask]

def interpolate_to_match(signal, target_length):
    if len(signal) == target_length:
        return signal
    x_original = np.linspace(0, 1, len(signal))
    x_target = np.linspace(0, 1, target_length)
    return np.interp(x_target, x_original, signal)

def convert_csi_to_complex(file_path, SUBCARRIER_NUM=52):
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M.%S")
    I = df[[f"I{i}" for i in range(SUBCARRIER_NUM)]].values
    Q = df[[f"Q{i}" for i in range(SUBCARRIER_NUM)]].values
    csi = I + 1j * Q
    ts = df["timestamp"]
    return csi, ts

def bandpass_filter(data, lowcut=5.0, highcut=20.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def hampel_filter_magnitude(csi_signal, window=15, n_sigma=3):
    mag = np.abs(csi_signal)
    phase = np.angle(csi_signal)
    median = medfilt(mag, kernel_size=window)
    dev = np.abs(mag - median)
    mad = np.median(dev)
    outliers = dev > n_sigma * mad
    mag[outliers] = median[outliers]
    return mag * np.exp(1j * phase)

def ema_detrending_complex(csi_signal, span=10):
    real_ema = pd.Series(csi_signal.real).ewm(span=span).mean().values
    imag_ema = pd.Series(csi_signal.imag).ewm(span=span).mean().values
    ema = real_ema + 1j * imag_ema
    return csi_signal - ema

def auto_detect_noisy_channels(csi, threshold=3.0):
    mags = np.abs(csi)
    variances = np.var(mags, axis=0)
    z_scores = (variances - np.mean(variances)) / (np.std(variances) + 1e-8)
    return np.where(np.abs(z_scores) > threshold)[0]

def preprocess_complex_pipeline(CSI_PATH, k=52, indices_to_remove=None):
    csi, ts = convert_csi_to_complex(CSI_PATH)
    if isinstance(indices_to_remove, list):
        csi = np.delete(csi, indices_to_remove, axis=1)
    elif indices_to_remove == 'auto':
        indices_to_remove = auto_detect_noisy_channels(csi)
        csi = np.delete(csi, indices_to_remove, axis=1)
    N, C = csi.shape
    csi_out = np.zeros_like(csi, dtype=np.complex64)
    for c in range(C):
        sig = csi[:, c]
        sig_bp = bandpass_filter(sig.real) + 1j * bandpass_filter(sig.imag)
        sig_bp = ema_detrending_complex(sig_bp)
        sig_hampel = hampel_filter_magnitude(sig_bp)
        sig_detrended = sig_hampel
        #sig_detrended = ema_detrending_complex(sig_hampel)
        csi_out[:, c] = sig_detrended
    return csi_out, ts

def calibrate_beamforming_weights(
    activity_paths, no_activity_paths, k=52, top_csv=10, indices_to_remove=None
):
    X_activity_all, X_no_activity_all = [], []

    selected_indices = random.sample(range(min(len(activity_paths), len(no_activity_paths))), top_csv)

    for i in selected_indices:
        act_path = activity_paths[i]
        noact_path = no_activity_paths[i]
        X_act, _ = preprocess_complex_pipeline(
            act_path, k=k, indices_to_remove=indices_to_remove
        )
        X_noact, _ = preprocess_complex_pipeline(
            noact_path, k=k, indices_to_remove=indices_to_remove
        )
        X_activity_all.append(X_act)
        X_no_activity_all.append(X_noact)

    X_activity = np.vstack(X_activity_all)
    X_noise = np.vstack(X_no_activity_all)

    X_activity -= X_activity.mean(axis=0, keepdims=True)
    X_noise -= X_noise.mean(axis=0, keepdims=True)

    R_s = (X_activity.conj().T @ X_activity) / X_activity.shape[0]
    R_n = (X_noise.conj().T @ X_noise) / X_noise.shape[0] + 1e-6 * np.eye(X_noise.shape[1])

    eigvals, eigvecs = eigh(R_s, R_n)
    return eigvecs[:, ::-1]  # beamforming matrix
def evaluate_activity_vs_noactivity_features_side_by_side(
    ACTIVITY_CSI_PATHS, NO_ACTIVITY_CSI_PATHS, W, threshold=5.0,
    top_k=5, max_pairs=5, window_size=64,
    aggregation=False, channel_vis=False, indices_to_remove=None
):
    from scipy.signal import stft

    selected_feats = ["Mean", "Std Dev", "Energy"]
    for idx, (path_act, path_noact) in enumerate(zip(ACTIVITY_CSI_PATHS, NO_ACTIVITY_CSI_PATHS)):
        if idx >= max_pairs:
            break
        pair_data = {}
        for label, path in [("ACTIVITY", path_act), ("NO-ACTIVITY", path_noact)]:
            print(f"Processing [{label}]: {path}")
            try:
                activity_sig, ts = preprocess_complex_pipeline(
                    CSI_PATH=path,
                    k=W.shape[0],
                    indices_to_remove=indices_to_remove
                )
            except Exception as e:
                print(f"Failed preprocessing {path}: {e}")
                continue

            projected = activity_sig @ W[:, :top_k]
            signal_1d = np.abs(np.mean(projected, axis=1))
            magnitude = np.abs(activity_sig)
            ts_plot = ts[:len(signal_1d)]

            if channel_vis:
                pair_data[label] = {"Channels": (ts_plot, np.abs(projected))}
                continue

            features = {k: [] for k in [
                "Mean", "Std Dev", "Energy", "MAD", "Inter-Variance", "Kurtosis", "Skewness", "High-Freq Power"
            ]}
            averaged_signal = signal_1d.copy()
            inter_var = np.var(np.mean(magnitude[:, :top_k], axis=1))

            for t in range(window_size, len(signal_1d)):
                window = signal_1d[t - window_size:t]
                fft_vals = np.fft.rfft(window)
                freqs = np.fft.rfftfreq(window_size, d=1.0)  # assume fs = 1.0 Hz

                high_freq_mask = freqs >= 0.25
                hf_power_val = np.mean(np.abs(fft_vals[high_freq_mask]) ** 2) if np.any(high_freq_mask) else 0.0

                f = {
                    "Mean": np.mean(window),
                    "Std Dev": np.std(window),
                    "Energy": np.sum(window ** 2),
                    "MAD": np.median(np.abs(window - np.median(window))),
                    "High-Freq Power": hf_power_val
                }
                for k in f:
                    features[k].append(f[k])

            ts_window = ts_plot[window_size:]
            pair_data[label] = {
                "Averaged Signal": (ts_plot, averaged_signal),
                "Windowed Features": (ts_window, features)
            }

        feature_names = [
            "Averaged Signal", "Mean", "Std Dev", "Energy", "MAD",
            "High-Freq Power"
        ]

        if channel_vis:
            fig, axs = plt.subplots(top_k, 2, figsize=(12, 1.5 * top_k))
            for ch in range(top_k):
                max_val = max([
                    np.max(pair_data[l]["Channels"][1][:, ch])
                    for l in ["ACTIVITY", "NO-ACTIVITY"] if l in pair_data
                ])
                for col, label in enumerate(["ACTIVITY", "NO-ACTIVITY"]):
                    if label in pair_data:
                        ts_, sig = pair_data[label]["Channels"]
                        axs[ch, col].plot(ts_, sig[:, ch])
                    axs[ch, col].set_ylim(0, max_val * 1.05)
                    axs[ch, col].set_title(f"{label} - Top-{ch+1}")
                    axs[ch, col].set_xlabel("Time")
                    axs[ch, col].set_ylabel("Magnitude")
                    axs[ch, col].xaxis.set_major_formatter(mdates.DateFormatter('%M:%S.%f'))
            fig.suptitle(f"Top-{top_k} Beamformed CSI Channels - Pair {idx}")

        else:
            fig, axs = plt.subplots(len(feature_names), 2, figsize=(12, 1.5 * len(feature_names)))
            for row, fname in enumerate(feature_names):
                max_val = 0
                for l in ["ACTIVITY", "NO-ACTIVITY"]:
                    if l in pair_data:
                        if fname == "Averaged Signal":
                            sig = pair_data[l][fname][1]
                        else:
                            sig = pair_data[l]["Windowed Features"][1][fname]
                        if len(sig) > 0 and not np.all(np.isnan(sig)):
                            max_val = max(max_val, np.nanmax(sig))

                for col, label in enumerate(["ACTIVITY", "NO-ACTIVITY"]):
                    ax = axs[row, col]
                    if label in pair_data:
                        if fname == "Averaged Signal":
                            ts_, sig = pair_data[label][fname]
                        else:
                            ts_ = pair_data[label]["Windowed Features"][0]
                            sig = pair_data[label]["Windowed Features"][1][fname]
                        ax.plot(ts_, sig)
                    ax.set_ylim(0, max_val * 1.05 if max_val > 0 else 1)
                    ax.set_title(f"{label} - {fname}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Value")
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S.%f'))
            fig.suptitle(f"Sliding Features Comparison (Expanded) - Pair {idx}")

        plt.tight_layout()
        plt.draw()
        plt.pause(3.0)
        plt.close()
