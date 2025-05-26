import os
import glob
import csv
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from collections import deque
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
import math

# --- Constants (CSI_To_CSV.py에서 가져옴) ---
# MQTT 관련 상수는 시각화 스크립트에서는 직접 사용하지 않으므로 제외 가능
# BROKER_ADDRESS = "61.252.57.136"
# BROKER_PORT = 4991
# TOPICS = ["L0382/ESP/8"] # 시각화할 토픽 지정 (Raw_CSI_To_CSV.py와 일치 필요)
TOPICS = ["L0382/ESP/8"] # 이 부분은 Raw_CSI_To_CSV.py와 동기화

# 입력 CSV 디렉토리 (Raw_CSI_To_CSV.py의 출력 디렉토리)
# INPUT_CSV_DIR = r"data\Raw_CSI_To_CSV_NoActivity"
INPUT_CSV_DIR = r"data\Raw_CSI_To_CSV_Activity"

# 캘리브레이션 파일이 저장된 디렉토리 (CSI_To_CSV.py와 동일)
CALIB_DIR = r"data\CSI_Calibration"
os.makedirs(CALIB_DIR, exist_ok=True) # CALIB_DIR이 없을 경우 생성

SUBCARRIERS = 52
WIN_SIZE = 64
# ADD_BUFFER_SIZE, CSV_WRITE_FRAME_COUNT는 Raw 데이터 처리 방식에 따라 조정 필요
# 여기서는 CSI_To_CSV.py의 처리 단위를 따르기 위해 유사한 버퍼 크기 개념을 도입
# PROCESSING_BUFFER_SIZE 만큼 데이터가 모여야 CSI_To_CSV.py의 한 사이클 처리가 가능
PROCESSING_BUFFER_SIZE = 128 + 128 + 64 # 임의로 CSI_To_CSV.py와 유사하게 설정 (조정 필요)

# 시각화용 버퍼 크기
PLOT_BUFFER_SIZE = 4096

# --- Global Variables (CSI_To_CSV.py에서 가져오거나 유사하게 관리) ---
# processing_buffer: raw CSI 데이터를 I/Q 쌍이 아닌, CSI_To_CSV.py의 z_normalized 형태으로 변환하여 저장
processing_buffer_viz = {topic: deque(maxlen=PROCESSING_BUFFER_SIZE) for topic in TOPICS}
packet_timestamps_viz = {topic: deque(maxlen=PROCESSING_BUFFER_SIZE) for topic in TOPICS}

# 전처리 결과 저장용 버퍼 (시각화 대상)
timestamp_plot_buffer = {topic: deque(maxlen=PLOT_BUFFER_SIZE) for topic in TOPICS}
activity_detection_plot_buffer = {topic: deque(maxlen=PLOT_BUFFER_SIZE) for topic in TOPICS}
activity_flag_plot_buffer = {topic: deque(maxlen=PLOT_BUFFER_SIZE) for topic in TOPICS}
th_plot_buffer = {topic: deque(maxlen=PLOT_BUFFER_SIZE) for topic in TOPICS}

# CSI_To_CSV.py의 상태 유지 변수들 (캘리브레이션 값, EWMA 평균, 이전 샘플 등)
mu_bg_dict_viz = {}
sigma_bg_dict_viz = {}
ewma_avg_dict_viz = {topic: 0.0 for topic in TOPICS}
mean_buffer_viz = {topic: deque(maxlen=100) for topic in TOPICS} # CSI_To_CSV의 mean_buffer와 동일 목적
prev_samples_viz = {topic: np.zeros(WIN_SIZE) for topic in TOPICS} # CSI_To_CSV의 prev_samples와 동일 목적

already_read_files_viz = set()

# --- Helper Functions (CSI_To_CSV.py에서 가져옴) ---
def robust_hampel_filter(column, window_size=5, n_sigma=3):
    '''이상치 제거 필터'''
    median = medfilt(column, kernel_size=window_size)
    deviation = np.abs(column - median)
    mad = np.median(deviation)
    threshold = n_sigma * mad
    outliers = deviation > threshold
    column[outliers] = median[outliers]
    return column

def parse_custom_timestamp(ts_str):
    '''timestamp 파싱 : ESP timestamp string -> datetime'''
    # Raw_CSI_To_CSV.py는 datetime 객체를 문자열로 저장하므로, 이를 다시 파싱
    # 예: "2023-10-27 10:00:00.123456"
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # 밀리초가 없는 경우도 고려 (포맷이 다를 수 있음)
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")


def load_calibration_params(topic):
    '''캘리브레이션 파라미터 로드'''
    global mu_bg_dict_viz, sigma_bg_dict_viz
    calib_file = os.path.join(CALIB_DIR, f"{topic.replace('/', '_')}_bg_params.csv")
    if os.path.exists(calib_file):
        with open(calib_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        mu_bg_dict_viz[topic] = np.array([float(x) for x in rows[0]])
        sigma_bg_dict_viz[topic] = np.array([float(x) for x in rows[1]])
        print(f"[{topic}] Loaded calibration parameters from: {calib_file}")
        return True
    else:
        print(f"[{topic}] Calibration file not found: {calib_file}. Please run CSI_To_CSV.py first to generate calibration.")
        # 캘리브레이션 파일이 없으면, 일단 기본값으로 설정하거나 처리를 중단할 수 있음
        # 여기서는 Z-score 정규화가 제대로 안될 수 있음을 인지하고 진행
        # 또는, 이 경우 CSI_To_CSV.py의 calibration_data 없이 mu, sigma를 0과 1로 설정하는 등의 fallback 필요
        # 지금은 로드 실패 시 False 반환
        indices_to_remove = list(range(21, 32)) # CSI_To_CSV.py와 동일
        num_reduced_subcarriers = SUBCARRIERS - len(indices_to_remove)
        mu_bg_dict_viz[topic] = np.zeros(num_reduced_subcarriers)
        sigma_bg_dict_viz[topic] = np.ones(num_reduced_subcarriers) # 0으로 나누는 것을 방지하기 위해 1로 설정
        print(f"[{topic}] Using default calibration parameters (mu=0, sigma=1) due to missing file.")
        return False # 실제로는 False를 반환하거나, 프로그램이 사용자에게 알리고 종료하는 것이 나을 수 있음


# --- Main Processing Logic (CSI_To_CSV.py의 process_csi_data 함수 로직을 여기에 재구성) ---
def process_single_csi_packet_for_viz(topic, timestamp_dt, raw_iq_values):
    '''하나의 raw CSI 패킷(I/Q 값들)을 CSI_To_CSV.py 방식으로 전처리'''
    global processing_buffer_viz, packet_timestamps_viz, ewma_avg_dict_viz, mean_buffer_viz, prev_samples_viz
    global timestamp_plot_buffer, activity_detection_plot_buffer, activity_flag_plot_buffer, th_plot_buffer

    if not (topic in mu_bg_dict_viz and topic in sigma_bg_dict_viz):
        print(f"[{topic}] Calibration parameters not loaded. Skipping processing.")
        return

    # 1. CSI Amplitude 계산 및 서브캐리어 축소 (CSI_To_CSV.py 로직과 동일하게)
    if len(raw_iq_values) < SUBCARRIERS * 2:
        print(f"[{topic}] Insufficient I/Q data for packet. Expected {SUBCARRIERS*2}, got {len(raw_iq_values)}")
        return
    
    csi_complex = [raw_iq_values[i] + 1j * raw_iq_values[i+1] for i in range(0, SUBCARRIERS*2, 2)]
    csi_amplitude = np.array([np.abs(x) for x in csi_complex])[:SUBCARRIERS]
    indices_to_remove = list(range(21, 32)) # 0MHz 주변 및 DC 채널 제거 (CSI_To_CSV.py와 동일)
    csi_amplitude_reduced = np.delete(csi_amplitude, indices_to_remove)

    # 2. Z-score 정규화 (CSI_To_CSV.py 로직과 동일하게)
    z_normalized = (csi_amplitude_reduced - mu_bg_dict_viz[topic]) / sigma_bg_dict_viz[topic]
    
    processing_buffer_viz[topic].append(z_normalized.copy())
    packet_timestamps_viz[topic].append(timestamp_dt)

    # 3. 버퍼가 충분히 찼을 때 전처리 및 특징 추출 (CSI_To_CSV.py 로직과 동일하게)
    # CSI_To_CSV.py는 PROCESSING_BUFFER_SIZE 만큼 차면 CSV_WRITE_FRAME_COUNT 만큼 처리하고 popleft
    # 여기서는 시각화를 위해 데이터를 계속 누적하고, 버퍼가 특정 크기 이상일 때마다 최신 CSV_WRITE_FRAME_COUNT 만큼을 처리
    # 또는, 슬라이딩 윈도우 방식으로 처리할 수 있음.
    # 여기서는 CSI_To_CSV.py와 유사하게, 특정 개수(예: 128개)의 새로운 결과가 나올 때마다 plot 버퍼에 추가.

    # 처리할 최소 데이터 덩어리 크기 (CSI_To_CSV.py의 CSV_WRITE_FRAME_COUNT와 유사)
    # 이 값은 CSI_To_CSV.py의 컨볼루션 등 윈도우 기반 처리를 고려하여 설정
    # 여기서는 processing_buffer_viz가 PROCESSING_BUFFER_SIZE 만큼 찼을 때,
    # 가장 오래된 CSV_WRITE_FRAME_COUNT 만큼을 처리한다고 가정
    # (실제 CSI_To_CSV.py는 popleft 하므로, 여기서는 deque의 특성을 활용하여 슬라이딩 윈도우처럼 동작)

    # CSI_To_CSV.py는 PROCESSING_BUFFER_SIZE가 되면 CSV_WRITE_FRAME_COUNT 만큼의 결과를 생성
    # 여기서는 매 패킷마다 시도하되, 실제 연산은 데이터가 충분할 때 수행
    if len(processing_buffer_viz[topic]) >= PROCESSING_BUFFER_SIZE: # 실제로는 이 조건이 항상 참이 되도록 외부에서 호출 조절 필요 X, 내부에서 윈도우만큼 데이터 슬라이스
        
        # processing_buffer_viz에서 실제 처리할 윈도우 크기의 데이터를 가져옴
        # CSI_To_CSV.py는 전체 PROCESSING_BUFFER_SIZE를 사용해 연산 후 앞부분 CSV_WRITE_FRAME_COUNT 만큼 결과를 냄.
        proc_array = np.array(list(processing_buffer_viz[topic])) # 현재까지 쌓인 전체 버퍼
        current_timestamps_array = list(packet_timestamps_viz[topic])

        # CSI_To_CSV.py와 동일한 전처리 수행
        amp_filtered = np.apply_along_axis(robust_hampel_filter, 0, proc_array)
        mean_per_packet = np.mean(amp_filtered, axis=1, keepdims=True)
        detrended_packet = amp_filtered - mean_per_packet
        
        mean_current = np.mean(amp_filtered, axis=0)
        if len(mean_buffer_viz[topic]) > 0:
            hist_array = np.array(mean_buffer_viz[topic])
            mean_historical = np.mean(hist_array, axis=0)
        else:
            mean_historical = mean_current
        mean_buffer_viz[topic].append(mean_current) # CSI_To_CSV에서는 mean_current를 mean_buffer에 바로 넣지 않음. 여기서는 일단 넣어둠. (확인필요)
                                                 # CSI_To_CSV.py에서는 combined_mean 계산 후 mean_buffer를 업데이트 안함.
                                                 # 이 부분은 장기적인 평균 변화를 어떻게 반영할지에 따라 달라짐. 여기서는 일단 현재 평균을 mean_buffer_viz에 추가.
        
        combined_mean = (mean_current + mean_historical) / 2
        detrended = detrended_packet - combined_mean
        
        SCAbsEuclidSumFeatured = np.std(detrended, axis=1)
        if len(SCAbsEuclidSumFeatured) < 2: # np.diff는 최소 2개 요소 필요
             # 데이터가 아직 충분하지 않아 diff 계산 불가
            return

        FeaturedDerivative = np.diff(SCAbsEuclidSumFeatured)
        # bufferLength = len(FeaturedDerivative) # 이 변수는 CSI_To_CSV.py에서 avgSigVal 계산에 사용
        FeaturedDerivativeAbs = np.abs(FeaturedDerivative)

        padded_signal = np.concatenate([prev_samples_viz[topic], FeaturedDerivativeAbs])
        window = np.ones(WIN_SIZE)
        convolved = np.convolve(padded_signal, window, mode='valid')
        prev_samples_viz[topic] = FeaturedDerivativeAbs[-WIN_SIZE:]
        
        # convolved 결과에서 현재 프레임에 해당하는 부분만 추출
        # CSI_To_CSV.py: FeaturedDerivativeAbsSum = convolved[-len(FeaturedDerivativeAbs):]
        # 이 방식은 convolved 길이가 FeaturedDerivativeAbs보다 길 때만 유효.
        # convolve mode='valid'는 (len(padded_signal) - WIN_SIZE + 1) 길이의 결과를 반환.
        # len(padded_signal) = WIN_SIZE + len(FeaturedDerivativeAbs)
        # 따라서 convolved 길이는 len(FeaturedDerivativeAbs) + 1 이 됨.
        # 즉, convolved[-len(FeaturedDerivativeAbs):]는 마지막 len(FeaturedDerivativeAbs)개의 요소를 가져옴.
        # 이는 FeaturedDerivativeAbs와 길이가 같지 않을 수 있음. (convolved가 1 더 김)
        # 여기서는 FeaturedDerivativeAbs와 같은 길이의 결과를 얻기 위해 조정 필요.
        # 또는, CSI_To_CSV.py의 의도는 가장 최근의 convolved 값들을 사용하는 것일 수 있음.
        # np.convolve 결과의 의미를 정확히 파악하고 슬라이싱해야 함.
        # mode='valid'의 결과는 입력신호에 윈도우가 완전히 겹치는 부분에 대해서만 나옴.
        # padded_signal = [prev_WIN_SIZE] + [current_N] -> convolved 길이는 N+1
        # FeaturedDerivativeAbsSum은 현재 처리 중인 FeaturedDerivativeAbs에 대한 이동평균값이어야 함.
        # convolved는 padded_signal 전체에 대한 이동평균.
        # convolved의 마지막 len(FeaturedDerivativeAbs)개의 값이 현재 FeaturedDerivativeAbs에 대한 이동평균에 해당.
        
        # FeaturedDerivativeAbsSum = convolved # 만약 convolved가 이미 현재 프레임에 대한 결과라면
        # CSI_To_CSV.py의 convolved[-len(FeaturedDerivativeAbs):] 는 convolved의 끝에서부터 FeaturedDerivativeAbs 길이만큼.
        # convolved 길이가 len(FeaturedDerivativeAbs)+1 이므로, convolved[1:] 이 FeaturedDerivativeAbs에 해당.
        if len(convolved) > len(FeaturedDerivativeAbs):
            FeaturedDerivativeAbsSum = convolved[1:] # 첫번째 값은 이전 윈도우의 영향을 많이 받으므로 제외 가능성
        elif len(convolved) == len(FeaturedDerivativeAbs): # 이런 경우는 잘 없음
             FeaturedDerivativeAbsSum = convolved
        else: # convolved 길이가 더 짧으면 문제 (발생하면 안됨)
            print(f"[{topic}] Convolution result shorter than expected. Skipping feature extraction.")
            return


        avgSigVal = np.mean(FeaturedDerivativeAbsSum) if len(FeaturedDerivativeAbsSum) > 0 else 0
        alpha = 0.01
        if ewma_avg_dict_viz[topic] == 0.0:
            ewma_avg_dict_viz[topic] = avgSigVal
        else:
            ewma_avg_dict_viz[topic] = alpha * avgSigVal + (1 - alpha) * ewma_avg_dict_viz[topic]
        
        Th_viz = 2.5 * ewma_avg_dict_viz[topic]
        ActivityDetected_viz = (FeaturedDerivativeAbsSum > Th_viz).astype(float)

        # 결과를 plot 버퍼에 추가 (가장 최근 결과만)
        # FeaturedDerivativeAbsSum, ActivityDetected_viz는 FeaturedDerivative에 대한 결과이므로,
        # 타임스탬프는 SCAbsEuclidSumFeatured의 두번째 값부터 대응.
        # SCAbsEuclidSumFeatured는 proc_array (길이 PROCESSING_BUFFER_SIZE)에 대한 결과.
        # FeaturedDerivative는 SCAbsEuclidSumFeatured의 diff이므로 길이가 1 짧음.
        # 따라서, 타임스탬프도 그에 맞게 슬라이싱.
        
        # current_timestamps_array는 processing_buffer_viz에 대응.
        # SCAbsEuclidSumFeatured는 processing_buffer_viz의 각 행(시간)에 대한 값.
        # FeaturedDerivativeAbsSum, ActivityDetected_viz는 SCAbsEuclidSumFeatured의 시간축에서 diff이므로,
        # 타임스탬프는 current_timestamps_array의 두번째 값부터 시작하는 것과 대응.
        
        # 여기서는 가장 최근의 한 프레임 결과만 시각화 버퍼에 추가한다고 가정.
        # (실제로는 CSI_To_CSV처럼 CSV_WRITE_FRAME_COUNT 만큼 묶어서 처리 후, 그 결과들을 순차적으로 추가)
        # 지금은 FeaturedDerivativeAbsSum의 마지막 값, ActivityDetected_viz의 마지막 값, Th_viz (스칼라)를
        # current_timestamps_array의 마지막 값 (또는 그 이전 값)과 매칭하여 추가.
        
        # FeaturedDerivativeAbsSum 등은 길이가 processing_buffer_viz 길이 -1 임.
        # 가장 최근 값은 이 배열들의 마지막 요소.
        # 이 마지막 요소에 해당하는 시간은 current_timestamps_array의 마지막 요소임.
        if len(FeaturedDerivativeAbsSum) > 0:
            latest_timestamp = current_timestamps_array[-1]
            latest_activity_detection = FeaturedDerivativeAbsSum[-1]
            latest_activity_flag = ActivityDetected_viz[-1]
            
            timestamp_plot_buffer[topic].append(latest_timestamp)
            activity_detection_plot_buffer[topic].append(latest_activity_detection)
            activity_flag_plot_buffer[topic].append(latest_activity_flag)
            th_plot_buffer[topic].append(Th_viz) # Th는 모든 프레임에 동일하게 적용될 수 있음

        # CSI_To_CSV.py처럼 처리한 만큼 processing_buffer_viz에서 popleft (선택사항)
        # 여기서는 maxlen을 사용하므로 자동으로 오래된 데이터가 제거됨.
        # 다만, CSI_To_CSV.py의 popleft는 처리 단위를 명확히 하기 위함.
        # 시각화에서는 모든 데이터를 계속 그려주므로, popleft를 안해도 maxlen에 의해 관리됨.
        # CSI_To_CSV.py는 CSV_WRITE_FRAME_COUNT 만큼 popleft.
        # 여기서는 매번 한 프레임씩 결과가 나온다고 가정하고 있으므로, popleft(1)을 할 수도 있지만,
        # deque(maxlen=...)이 자동으로 관리해주므로 추가적인 popleft는 불필요.

# --- Matplotlib Animation Functions ---
def find_new_csv_files(topic):
    global INPUT_CSV_DIR, already_read_files_viz
    search_path = os.path.join(INPUT_CSV_DIR, f"{topic.replace('/', '_')}_*.csv")
    all_files = sorted(glob.glob(search_path), key=os.path.getctime)
    new_files = [f for f in all_files if f not in already_read_files_viz]
    return new_files

def read_and_process_new_files(topic):
    new_files = find_new_csv_files(topic)
    for csv_file in new_files:
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                already_read_files_viz.add(csv_file)
                continue

            # CSV의 각 행을 순회하며 패킷 처리
            for index, row in df.iterrows():
                timestamp_str = row['timestamp']
                # 나머지 컬럼은 I0, Q0, I1, Q1, ... 형태여야 함.
                iq_data_cols = [col for col in df.columns if col.startswith('I') or col.startswith('Q')]
                raw_iq_values = row[iq_data_cols].astype(int).tolist()
                
                timestamp_dt = parse_custom_timestamp(timestamp_str) # 문자열 타임스탬프를 datetime 객체로
                
                process_single_csi_packet_for_viz(topic, timestamp_dt, raw_iq_values)
            
            already_read_files_viz.add(csv_file)
            print(f"[{topic}] Processed file: {csv_file}")
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
            already_read_files_viz.add(csv_file) # 오류 발생 시 다시 읽지 않도록 추가

def update_plot_viz(frame):
    plt.clf()
    for idx, topic in enumerate(TOPICS):
        read_and_process_new_files(topic) # 새 파일 읽고 처리

        if len(timestamp_plot_buffer[topic]) > 0:
            times = list(timestamp_plot_buffer[topic])
            detections = list(activity_detection_plot_buffer[topic])
            flags = list(activity_flag_plot_buffer[topic])
            thresholds = list(th_plot_buffer[topic])

            # 그래프 레이아웃 설정
            cols = 1 
            rows = math.ceil(len(TOPICS) / cols)
            
            plt.subplot(rows, cols, idx + 1)
            plt.plot(times, detections, label='Activity Detection Feature')
            plt.step(times, flags, label='Activity Flag', color='r', where='mid')
            plt.plot(times, thresholds, label='Threshold (Th)', color='g', linestyle='--')
            
            plt.title(f"Processed CSI - Topic: {topic}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend(loc='upper left')
            plt.grid(True)
            plt.tight_layout(h_pad=2.0)

def main_visualizer():
    # 캘리브레이션 파라미터 로드 시도
    for topic in TOPICS:
        load_calibration_params(topic)
        # 로드 실패 시 사용자에게 알리고 종료하거나, 기본값으로 계속 진행할 수 있음.
        # 여기서는 load_calibration_params 내부에서 기본값을 사용하도록 이미 처리.

    fig = plt.figure(figsize=(12, min(8, 4 * len(TOPICS)))) # 토픽 개수에 따라 세로 크기 조절
    ani = FuncAnimation(fig, update_plot_viz, interval=1000, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    # 이 스크립트가 src/에 있고, INPUT_CSV_DIR과 CALIB_DIR이 프로젝트 루트에 있다면
    # 상대 경로로 접근 시 ../ 를 사용해야 할 수 있음.
    # 예: INPUT_CSV_DIR = r"../Raw_CSI_To_CSV_NoActivity"
    #     CALIB_DIR = r"../CSI_Calibration"
    # 여기서는 CSI_Plot_Main.py에서 import될 때를 가정하고,
    # 프로젝트 루트가 sys.path에 잡혀있으므로, 루트 기준 상대경로로 설정.
    # 스크립트 직접 실행 시에는 위 경로 조정 필요.
    
    # 임시로 직접 실행을 위한 경로 조정 (테스트용)
    # current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(current_file_dir) # src 폴더의 부모 = 프로젝트 루트
    # INPUT_CSV_DIR = os.path.join(project_root, "Raw_CSI_To_CSV_NoActivity")
    # CALIB_DIR = os.path.join(project_root, "CSI_Calibration")
    
    main_visualizer() 