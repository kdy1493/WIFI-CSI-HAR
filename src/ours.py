import autorootcwd
from scipy.signal import butter, filtfilt, savgol_filter, medfilt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.decomposition import PCA 

def convert_csi_to_amplitude(file_path, SUBCARRIER_NUM=52):  # src/utils.py
    """
    Convert CSI data to amplitude.
    :param file_path: Path to the CSV file containing CSI data.
    :param SUBCARRIER_NUM: Number of subcarriers (default = 52)

    Example usage:
    NO_ACTIVITY_CSI_PATH = r"data\Raw_CSI_To_CSV_NoActivity\merged_csi_data_noactivity.csv"
    amp, ts = convert_csi_to_amplitude(NO_ACTIVITY_CSI_PATH, SUBCARRIER_NUM) # amp: signal amplitude, ts: timestamp
    :return: Tuple of amplitude and timestamp arrays.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M.%S")

    I = df[[f"I{i}" for i in range(SUBCARRIER_NUM)]].values
    Q = df[[f"Q{i}" for i in range(SUBCARRIER_NUM)]].values
    amp = np.sqrt(I**2 + Q**2)
    ts = ts = df["timestamp"]

    return amp, ts

def butter_lowpass_filter(raw_data, cutoff=15.0, fs=100, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, raw_data)

def hampel_filter(filtered_data, window=9, n_sigma=3):
    """Hampel filter로 이상치 제거"""
    median = medfilt(filtered_data, kernel_size=window)
    dev    = np.abs(filtered_data - median)
    mad    = np.median(dev)
    out    = dev > n_sigma * mad
    filtered_data[out] = median[out]
    return filtered_data

def ema_detrending(filtered_hampel):
    ema_window = 5
    ema = pd.Series(filtered_hampel).ewm(span=ema_window).mean()
    detrended = filtered_hampel - ema
    return detrended


def plot_csi_amplitude(amp, time_stamp, title="None", FRAME_NUM=500):  # in src/utils.py
    N = min(FRAME_NUM, len(amp))
    tick_spacing = 10
    ts = time_stamp[:N]

    plt.figure(figsize=(12, 6))
    
    # Handle different input shapes
    if len(amp.shape) == 2 and amp.shape[1] > 1:  # (N, c) where c > 1
        for i in range(amp.shape[1]):
            plt.plot(ts, amp[:N, i], alpha=0.6)
    else:  # (N, 1) or (N,)
        plt.plot(ts, amp[:N], alpha=0.6)       
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Amplitude")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%M:%S.%f"))
    plt.xticks(ts[::tick_spacing], rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

def preprocess_pipeline(CSI_PATH):
    amp , ts = convert_csi_to_amplitude(CSI_PATH, SUBCARRIER_NUM=52)
    num_channel = amp.shape[1]
    detrended_signal = amp * 0
    for channel in range(num_channel):
        sig = amp[:,channel]
        filtered_signal = butter_lowpass_filter(sig)
        hampel_signal = hampel_filter(filtered_signal)
        
        #detrended_signal[:,channel] = hampel_signal # deleted detrending
        detrended_signal[:,channel] = ema_detrending(hampel_signal)
    return detrended_signal, amp, ts


def compression_pipeline(sig):
    pca = PCA(n_components=5)
    pca.fit(sig)
    compressed_signal = pca.fit_transform(sig)
    reconstructed_signal = pca.inverse_transform(compressed_signal)
    mse_score = np.mean((sig - reconstructed_signal)**2)
    return compressed_signal, mse_score

'''
필요한 함수들 
<Reading>
1. convert_csi_to_amplitude: input (CSV), output (amp, ts)

<Processing>
1. 52개 subcarrier 신호를 추출
--> 각자 신호에 대하여 다음을 수행
(1) Filtering (Butterworth) raw --> filtered_data
(2) Hampel Filtering (Hample) filtered_data --> filtered_hampel
(3) EMA detrending filtered_hampel --> detrended
'''
