import os
import shutil
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from collections import deque


CSV_DIR = r"data\processed"
# CSV_DIR = r"data\Raw_CSI_To_CSV_NoActivity"


TOPICS = ["L0382/ESP/1","L0382/ESP/2","L0382/ESP/3","L0382/ESP/4","L0382/ESP/5","L0382/ESP/6","L0382/ESP/7","L0382/ESP/8"]
#TOPICS = ["L0382/ESP/8"]

BUFFER_SIZE = 4096  # Number of samples to keep and plot
# 이미 읽은 파일 저장
already_read_files = set()

# Circular buffers for each topic
activity_detection_buffer = {topic: deque(maxlen=BUFFER_SIZE) for topic in TOPICS}
activity_flag_buffer = {topic: deque(maxlen=BUFFER_SIZE) for topic in TOPICS}
th_buffer = {topic: deque(maxlen=BUFFER_SIZE) for topic in TOPICS}
timestamp_buffer = {topic: deque(maxlen=BUFFER_SIZE) for topic in TOPICS}

def find_all_csv(topic):
    files = sorted(glob.glob(f"{CSV_DIR}/{topic.replace('/','_')}_*.csv"), key=os.path.getctime)
    return files

def update_plot(frame):
    plt.clf()
    for idx, topic in enumerate(TOPICS):
        csv_files = find_all_csv(topic)
        # Read and delete each CSV, appending to the circular buffer
        for csv_file in csv_files:
            if csv_file in already_read_files:
                continue  # 이미 읽은 파일은 건너뜀
            try:
                df = pd.read_csv(csv_file)
                times = pd.to_datetime(df['timestamp'])
                activity_detection = df['activity_detection']
                activity_flag = df['activity_flag']
                Th = df['Th'] if 'Th' in df.columns else [0]*len(df)  # fallback if missing
                # Append to circular buffer
                timestamp_buffer[topic].extend(times)
                activity_detection_buffer[topic].extend(activity_detection)
                activity_flag_buffer[topic].extend(activity_flag)
                th_buffer[topic].extend(Th)
                # Delete the file after reading
                already_read_files.add(csv_file)
                # os.remove(csv_file)
            except Exception as e:
                print(f"Error processing or deleting {csv_file}: {e}")

        # Plot only if we have data
        if len(timestamp_buffer[topic]) > 0:
            times_to_plot = list(timestamp_buffer[topic])
            activity_detection_to_plot = list(activity_detection_buffer[topic])
            activity_flag_to_plot = list(activity_flag_buffer[topic])
            th_to_plot = list(th_buffer[topic])

            # Ensure all arrays are the same length
            min_len = min(len(times_to_plot), len(activity_detection_to_plot), len(activity_flag_to_plot), len(th_to_plot))
            times_to_plot = times_to_plot[-min_len:]
            activity_detection_to_plot = activity_detection_to_plot[-min_len:]
            activity_flag_to_plot = activity_flag_to_plot[-min_len:]
            th_to_plot = th_to_plot[-min_len:]

            
            cols = 2
            rows = math.ceil(len(TOPICS) / 2)
            
            plt.subplot(rows, cols, idx + 1)
            plt.plot(times_to_plot, activity_detection_to_plot, label='ActivityDetection')
            plt.step(times_to_plot, activity_flag_to_plot, label='ActivityFlag', color='r', where='mid')
            plt.plot(times_to_plot, th_to_plot, label='Th', color='g', linestyle='--')  # Plot Th as dashed green line
            plt.title(topic)
            plt.xlabel("Time")
            plt.ylabel("Activity")
            plt.legend()
            #plt.ylim(-0.2, 5.0)
            plt.legend(loc='upper left')
            plt.tight_layout(h_pad=2.0)  # Increase h_pad as needed


def main():
    fig = plt.figure(figsize=(10, 6))
    ani = FuncAnimation(fig, update_plot, interval=1000)
    plt.show()

if __name__ == "__main__":
    main()