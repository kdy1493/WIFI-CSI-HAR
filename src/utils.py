import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates


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

def plot_csi_amplitude(amp, time_stamp, title="None", FRAME_NUM=500,
                       amp2=None,amp3=None ):  # in src/utils.py
    """
    Loads CSI data, calculates amplitude, and plots it.
    param amp : amplitude
    param time_stamp : timestamp
    param title : title of the plot
    param FRAME_NUM : number of frames to plot
    param amp2 : amplitude 2
    param amp3 : amplitude 3
    
    Example usage:
    NO_ACTIVITY_CSI_PATH = r"data\Raw_CSI_To_CSV_NoActivity\merged_csi_data_noactivity.csv"
    amp, ts = convert_csi_to_amplitude(NO_ACTIVITY_CSI_PATH, SUBCARRIER_NUM) # amp: signal amplitude, ts: timestamp
    plot_csi_amplitude(amp, ts, title="No Activity")
    """
    N = min(FRAME_NUM, len(amp))
    tick_spacing = 10
    ts = time_stamp[:N]

    plt.figure(figsize=(12, 6))

    if amp.ndim == 2 : 
        SUBCARRIER_NUM = amp.shape[1]
        for i in range(SUBCARRIER_NUM):
            plt.plot(ts, amp[:N, i], alpha=0.6)
            plt.xticks(ts[::tick_spacing], rotation=45)
    else : 
        valid_len = len(amp)
        plt.plot(time_stamp[1:1+valid_len], amp, alpha=0.6)
        plt.xticks( rotation=45)

    if amp2 is not None : 
        valid_len = len(amp2)
        plt.plot(time_stamp[1:1+valid_len], amp2,label="Activity Flag", linestyle='-', alpha=0.6)
    if amp3 is not None : 
        plt.axhline(amp3, color='red', linestyle='--', label=f"Threshold = {amp3:.4f}")

    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Amplitude")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%M:%S.%f"))
    plt.tight_layout()
    plt.show()
    plt.close()
    return

def plot_csi_amplitude_from_file( file_path, title="None", FRAME_NUM=500, SUBCARRIER_NUM=52):
    """Loads CSI data, calculates amplitude, and plots it.

    Example:
    ACTIVITY_CSI_PATH = r"data\Raw_CSI_To_CSV_DoorOpen\merged_csi_data_dooropen.csv"
    plot_csi_amplitude(NO_ACTIVITY_CSI_PATH, title='No Activity') # plots signal amplitude of 52 channels
    """
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M.%S")

        I = df[[f"I{i}" for i in range(SUBCARRIER_NUM)]].values
        Q = df[[f"Q{i}" for i in range(SUBCARRIER_NUM)]].values
        amp = np.sqrt(I**2 + Q**2)

        N = min(FRAME_NUM, len(df))
        tick_spacing = 10
        ts = df["timestamp"][:N]

        plt.figure(figsize=(12, 6))
        for i in range(SUBCARRIER_NUM):
            plt.plot(ts, amp[:N, i], alpha=0.6)
        plt.title(title)
        plt.xlabel("Timestamp")
        plt.ylabel("Amplitude")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%M:%S.%f"))
        plt.xticks(ts[::tick_spacing], rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

if __name__ == "__main__":
    # Example usage
    NO_ACTIVITY_CSI_PATH = (
        r"data\raw\raw_noActivity_csi\merged_csi_data_noactivity.csv"
    )
    ACTIVITY_CSI_PATH = r"data\raw\raw_activity_csi\merged_csi_data_dooropen.csv"

    # Convert and plot CSI data
    amp, ts = convert_csi_to_amplitude(NO_ACTIVITY_CSI_PATH)
    plot_csi_amplitude(amp, ts, title="No Activity")
    amp, ts = convert_csi_to_amplitude(ACTIVITY_CSI_PATH)
    plot_csi_amplitude(amp, ts, title="Door Open")
    plot_csi_amplitude_from_file(NO_ACTIVITY_CSI_PATH, title="No Activity")
    plot_csi_amplitude_from_file(ACTIVITY_CSI_PATH, title="Door Open")
