import re   # 정규표현식 모듈
import os
import csv
import time
import threading
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
from collections import deque
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter

BROKER_ADDRESS = "61.252.57.136"
BROKER_PORT = 4991
TOPICS = ["L0382/ESP/1","L0382/ESP/2","L0382/ESP/3","L0382/ESP/4","L0382/ESP/5","L0382/ESP/6","L0382/ESP/7", "L0382/ESP/8"]
#TOPICS = ["L0382/ESP/8"]

#### Calibration Parameters
FORCE_NEW_CALIBRATION = False  # Set to False to use existing calibration
CALIBRATION_SAMPLES = 10*60*100 # 10분 동안 정지상태 데이터 수집


SUBCARRIERS = 52
WIN_SIZE = 64
ADD_BUFFER_SIZE = 128           # 실시간 버퍼에 계속 쌓이는 데이터 
CSV_WRITE_FRAME_COUNT = 128     # 한번에 csv 파일에 쓰는 프레임 수

PROCESSING_BUFFER_SIZE = ADD_BUFFER_SIZE + CSV_WRITE_FRAME_COUNT + WIN_SIZE     # 실시간 분석을 위한 전체 버퍼크기 : 320 프레임
CSV_DIR = r"data\processed"
CALIB_DIR = r"data\calibration"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(CALIB_DIR, exist_ok=True)

client = None
processing_buffer = {topic: deque(maxlen=PROCESSING_BUFFER_SIZE) for topic in TOPICS}   # 양쪽에서 뺄 수 있는 큐 구조의 임시 버퍼 ( 길이 초과 시 : 오래된 데이터 삭제 )
packet_timestamps = {topic: deque(maxlen=PROCESSING_BUFFER_SIZE) for topic in TOPICS}   
subcarrier_means = {topic: np.zeros(SUBCARRIERS) for topic in TOPICS}
ewma_avg_dict = {topic: 0.0 for topic in TOPICS}
csv_buffer = {topic: [] for topic in TOPICS}
mean_buffer = {topic: deque(maxlen=100) for topic in TOPICS}
prev_samples = {topic: np.zeros(WIN_SIZE) for topic in TOPICS}

# --- Calibration parameters ---
mu_bg_dict = {}
sigma_bg_dict = {}

def robust_hampel_filter(column, window_size=5, n_sigma=3):
    '''이상치 제거 필터'''
    median = medfilt(column, kernel_size=window_size)       # cloumn 데이터에서 중앙값 찾기
    deviation = np.abs(column - median)                     # 중앙값과 원래 데이터 값의 차이 계산
    mad = np.median(deviation)                              # 그 차이의 중앙값 계산
    threshold = n_sigma * mad                               # threshold 계산( 이상치로 판단할 기준값 )
    outliers = deviation > threshold                        # threshold을 넘는 이상치 확인
    column[outliers] = median[outliers]                     # 이상치는 중앙값으로 대체
    return column

def normalize(x):
    '''정규화'''
    min_val = np.min(x)
    max_val = np.max(x)
    return (x - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(x)

def parse_custom_timestamp(ts):
    '''timestamp 파싱 : ESP timestamp -> datetime'''
    year = 2000 + int(ts[0:2])
    month = int(ts[2:4])
    day = int(ts[4:6])
    hour = int(ts[6:8])
    minute = int(ts[8:10])
    second = int(ts[10:12])
    millisecond = int(ts[12:15])
    microsecond = millisecond * 1000
    return datetime(year, month, day, hour, minute, second, microsecond)

def write_csv_for_topic(topic, feature_rows):
    '''CSV 파일 저장'''
    if feature_rows:
        filename = f"{CSV_DIR}/{topic.replace('/','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" 
        with open(filename, 'w', newline='') as f:                                                        
            writer = csv.writer(f)                                                                      
            writer.writerow(['timestamp', 'activity_detection', 'activity_flag','Th'])                
            writer.writerows(feature_rows)                                                             
        #print(f"Wrote {len(feature_rows)} rows to {filename}")

def calibrate_background(topic, calibration_data, force_new=False):
    '''배경 데이터 보정
    평균, 표준편차 계산'''
    calib_file = os.path.join(CALIB_DIR, f"{topic.replace('/','_')}_bg_params.csv")
    if not force_new and os.path.exists(calib_file):        # calibration data가 있는 경우
        with open(calib_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        mu_bg = np.array([float(x) for x in rows[0]])           # csv 파일 첫번째 행 : background data 평균
        sigma_bg = np.array([float(x) for x in rows[1]])        # csv 파일 두번째 행 : background data 표준편차
        print(f"[{topic}] Loaded: {calib_file}")
    else:                                                   # calibration data가 없는 경우
        mu_bg = np.mean(calibration_data, axis=0)               # background data 평균
        sigma_bg = np.std(calibration_data, axis=0)             # background data 표준편차
        sigma_bg[sigma_bg == 0] = 1                             # 표준편차가 0인 경우 1로 처리
        with open(calib_file, 'w', newline='') as f:   
            writer = csv.writer(f)
            writer.writerow(mu_bg)
            writer.writerow(sigma_bg)
        print(f"[{topic}] Saved: {calib_file}")
    return mu_bg, sigma_bg

def run_parallel_calibration(FORCE_NEW_CALIBRATION, CALIBRATION_SAMPLES):
    '''병렬 캘리브레이션'''
    global mu_bg_dict, sigma_bg_dict

    if FORCE_NEW_CALIBRATION:                                                                               # 새로 calibration 필요한 경우
        calibration_buffers = {topic: [] for topic in TOPICS}                                               #   topic( 센서 ) 마다 임시 버퍼 생성
        calibration_done = {topic: False for topic in TOPICS}                                               #   topic( 센서 ) 마다 calibration 완료 여부 : 초기 false 

        print("Collecting No Movement Data...")                                                             # 정지 데이터 수집 시작
        while not all(calibration_done.values()):                                                           # true 까지 실행
            for topic in TOPICS:
                if not calibration_done[topic]:                                                             #   calibration 완료 안된 경우 
                    if len(processing_buffer[topic]) > 0:                                                   #       processing buffer에 데이터가 하나라도 있으면
                        calibration_buffers[topic].append(processing_buffer[topic][-1])                     #           calibration buffer에 processing buffer의 마지막 데이터 추가
                    if len(calibration_buffers[topic]) >= CALIBRATION_SAMPLES:                              #       센서별 목표 샘플 수 (CALIBRATION_SAMPLES ) 도달 시 
                        calibration_done[topic] = True                                                      #           calibration true 표시
                        print(f"Collected {len(calibration_buffers[topic])} samples for topic {topic}.")    #           센서별 수집한 sample 수 출력

            time.sleep(0.01)  # 대기 시간 추가 ( 빠른 반복 방지 )

        for topic in TOPICS:                                                                                 
            calibration_data = np.array(calibration_buffers[topic])                                         
            mu_bg, sigma_bg = calibrate_background(topic, calibration_data, force_new=True)                 #  평균, 표준편차 계산
            mu_bg_dict[topic] = mu_bg                                                                       #  전역 변수에 저장
            sigma_bg_dict[topic] = sigma_bg

        print("Calibration complete for all topics. Now running real-time processing...")

    else:                                                                                                  #    calibration 필요 없는 경우
        for topic in TOPICS:
            dummy = np.zeros((1, SUBCARRIERS - len(range(21, 32))))
            mu_bg, sigma_bg = calibrate_background(topic, dummy, force_new=False)                          #    calibration 데이터 로드
            mu_bg_dict[topic] = mu_bg                                                                      #    전역 변수에 저장
            sigma_bg_dict[topic] = sigma_bg
            print(f"Loaded calibration for {topic}...")

        print("Calibration loaded. Starting real-time processing...")

def process_csi_data(topic, payload):
    global processing_buffer, packet_timestamps, subcarrier_means, ewma_avg_dict, csv_buffer, mu_bg_dict, sigma_bg_dict, prev_samples

    try:
        # Extract time=250424141012116 from payload
        match = re.search(r'time=(\d{15})', payload)    
        if match:
            ts_str = match.group(1)
            packet_time = parse_custom_timestamp(ts_str)
            packet_timestamps[topic].append(packet_time)
        else:
            packet_time = datetime.now()
            packet_timestamps[topic].append(packet_time)

        # Parse CSI data--------------------------------    # ex) time=250424141012116 CSI values: 32 21    -1 14   7 0  ...
        csi_data_str = payload.split("CSI values: ")[1].strip()    # CSI 데이터 파싱 : ex) 32 21 -1 14 7 0 ...
        csi_values = list(map(int, csi_data_str.split()))          # 문자열을 정수 리스트로 변환 : ex) [32, 21, -1, 14, 7, 0, ...]
        if len(csi_values) < SUBCARRIERS * 2:                      # 데이터 길이가 부족한 경우 종료
            return
        csi_complex = [csi_values[i] + 1j * csi_values[i + 1] for i in range(0, len(csi_values), 2)]  # 복소수 형태로 변환: ex) [32 + 21j, -1 + 14j, 7 + 0j, ...]
        csi_amplitude = np.array([np.abs(x) for x in csi_complex])[:SUBCARRIERS]                      # 진폭 : 복소수 크기 계산(서브캐리어 52 개) : ex) [37.416573867739416, 14.035668847618199, 7.0, ...]
        indices_to_remove = list(range(21, 32))                                                       # 21~32 인덱스 : ex) [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        csi_amplitude_reduced = np.delete(csi_amplitude, indices_to_remove)                           # 위의 11개 채널( 0 MHz 포함 좌우 5개 ) 을 삭제함 -> 중앙의 서브캐리어는 잡음이 많음 

        # calibration_done = (topic in mu_bg_dict) and (topic in sigma_bg_dict)
        # print(f"Calibration done for {topic}: {calibration_done}")


        # --- Z-score normalization using background calibration ---
        if topic in mu_bg_dict and topic in sigma_bg_dict:                                           # 정지 상태에서 수집한 평균과 표준편차 있으면
            z_normalized = (csi_amplitude_reduced - mu_bg_dict[topic]) / sigma_bg_dict[topic]        # Z-score 정규화
        else:
            z_normalized = csi_amplitude_reduced                                                     # 없으면 원본 csi 데이터 사용

        processing_buffer[topic].append(z_normalized.copy())                                         # 추후 분석위해 버퍼에 저장

        # Only process if buffer is full
        if len(processing_buffer[topic]) == PROCESSING_BUFFER_SIZE:                                  # 버퍼가 채워졌으면
            proc_array = np.array(processing_buffer[topic])                                          # 버퍼에 저장된 데이터를 배열로 변환
            timestamps_array = list(packet_timestamps[topic])                                        # 패킷 타임스탬프 배열로 변환

            amp_filtered = np.apply_along_axis(robust_hampel_filter, 0, proc_array)                  # 서브캐리어 별 Hampel 필터로 이상치 제거 (주변 중앙값 기준, 이상치-> 중앙값 교체)
            mean_per_packet = np.mean(amp_filtered, axis=1, keepdims=True)                           # 패킷별(시간별) 전체체 서브캐리어의 평균값 계산  ( 프레임 정규화 목적 )
            detrended_packet = amp_filtered - mean_per_packet                                        # 패킷에서 평균값 제거 -> 평균값 주변에 있는 데이터가 0에 가까워짐 : 서브캐리어간 상대변화에 집중 (무슨 변화가 있는 지 파악 가능

            mean_current = np.mean(amp_filtered, axis=0)                                             # 서브캐리어별 전체 시간 평균값 계산 ( 기준 패턴 생성 목적적)
            if len(mean_buffer[topic]) > 0:                                                          # 과거의 평균값이 버퍼에 있으면                   
                hist_array = np.array(mean_buffer[topic])                                           
                mean_historical = np.mean(hist_array, axis=0)                                           # 그것들의 평균값 계산
            else:                                                                                    # 과거의 평균값이 없으면
                mean_historical = mean_current                                                          # 현재 패턴을 사용
            combined_mean = (mean_current + mean_historical) / 2                                     # 현재 기준 패턴과 과거 기준 패턴의 평균값 계산

            detrended = detrended_packet - combined_mean                                             # 기준선 제거 -> 패턴 정규화
            print(f"detrended : {detrended}")
            SCAbsEuclidSumFeatured = np.std(detrended, axis=1)                                       # 패킷별 서브캐리어 진폭(변화정도) 표준편차 계산 -> 움직임이 많을수록 std가 커짐
            FeaturedDerivative = np.diff(SCAbsEuclidSumFeatured)
            bufferLength = len(FeaturedDerivative)
            FeaturedDerivativeAbs = np.abs(FeaturedDerivative)                                       # 표준편차의 변화량(미분값) 절대값 계산

            # Overlap-save convolution                                                               # 긴 데이터에 슬라이딩 윈도우 형태로 convolution 빠르게 처리 ( 과거 데이터를 현재 데이터에 붙이고 전체 conv -> 원하는 길이만 자름 )
            padded_signal = np.concatenate([prev_samples[topic], FeaturedDerivativeAbs])             # 과거 데이터 WIN_SIZE 만큼 + 현재 패킷의 변화량
            window = np.ones(WIN_SIZE)                                                               # 크기 WIN_SIZE, 모든 원소 1 배열 ( 이동 평균필터 )
            convolved = np.convolve(padded_signal, window, mode='valid')                             # 이동 평균필터 적용 -> 부드럽게 변화하는 곡선 얻음
            prev_samples[topic] = FeaturedDerivativeAbs[-WIN_SIZE:]                                  # 현재 패킷의 변화량 저장 -> 과거 데이터로 저장
            
            FeaturedDerivativeAbsSum = convolved[-len(FeaturedDerivativeAbs):]                      # 현재값만 추출 -> 현재 패킷에서 감지된 변화량 

            # min_val = np.min(FeaturedDerivativeAbsSum)
            # max_val = np.max(FeaturedDerivativeAbsSum)
            # if max_val - min_val == 0:
            #     FeaturedDerivativeAbsSum = np.zeros_like(FeaturedDerivativeAbsSum)
            # else:
            #     FeaturedDerivativeAbsSum = (FeaturedDerivativeAbsSum - min_val) / (max_val - min_val)
            
            # 지수 가중 이동 평균
            avgSigVal = np.mean(FeaturedDerivativeAbsSum) if bufferLength > 0 else 0            # 처리된 변화량의 평균 계산 : 현재시점의 전반적인 움직임 세기 요약
            alpha = 0.01                                                                        # 과거 정보 99% 반영 -> 아주 천천히 변화하는 기준선 만듦
            if ewma_avg_dict[topic] == 0.0:
                ewma_avg_dict[topic] = avgSigVal
            else:
                ewma_avg_dict[topic] = alpha * avgSigVal + (1 - alpha) * ewma_avg_dict[topic]   # 지수 가중 이동 평균 계산 : 최근값과 이전 평균을 섞음
            
            # 움직임 감지 ( Threshold )
            Th = 2.5 * ewma_avg_dict[topic]                                     # 임계치 계산 : 평균값의 2.5배
            ActivityDetected = (FeaturedDerivativeAbsSum > Th).astype(float)    # 변화량이 임계치 이상인 경우 1, 아니면 0   

            feature_rows = []
            ts_list = timestamps_array[:CSV_WRITE_FRAME_COUNT]
            feat_sum_list = FeaturedDerivativeAbsSum[:CSV_WRITE_FRAME_COUNT]
            activity_list = ActivityDetected[:CSV_WRITE_FRAME_COUNT]

            for i in range(CSV_WRITE_FRAME_COUNT):
                feature_rows.append([
                    ts_list[i].strftime("%Y-%m-%d %H:%M:%S.%f"),
                    feat_sum_list[i],
                    activity_list[i],
                    Th
                ])
            write_csv_for_topic(topic, feature_rows)

            frames_to_remove = CSV_WRITE_FRAME_COUNT
            for _ in range(frames_to_remove):
                processing_buffer[topic].popleft()
                packet_timestamps[topic].popleft()

    except Exception as e:
        print(f"Error processing CSI data for topic {topic}: {e}")
    print(f"[{topic}] Packet received at {datetime.now()}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        for topic in TOPICS:
            client.subscribe(topic)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    process_csi_data(msg.topic, msg.payload.decode())

def mqtt_thread_func():
    global client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    client.loop_forever()

def main():
    # Start MQTT thread first so buffers fill for all topics
    mqtt_thread_obj = threading.Thread(target=mqtt_thread_func, daemon=True)
    mqtt_thread_obj.start()

    # Run parallel calibration (all topics at once)
    run_parallel_calibration(FORCE_NEW_CALIBRATION, CALIBRATION_SAMPLES)

    # Keep main thread alive for real-time processing
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()


''' 
[목적 요약]
이 코드는 Wi-Fi 기반 CSI(Channel State Information) 데이터를 MQTT로 실시간 수신하고,
이를 분석하여 사람이 움직였는지를 감지하고, 그 결과를 CSV 파일로 저장하는 시스템이다.

[전체 흐름 요약]
1. MQTT 브로커와 연결하여 특정 topic(ESP 디바이스)으로부터 CSI 데이터를 받는다.
2. CSI 데이터는 복소수로 구성되어 있으며, 진폭(amplitude)을 추출한 후 전처리를 수행한다.
3. 수신된 CSI 데이터를 일정량 모으면, 평균·표준편차 기준선과 비교하여 움직임 여부를 판단한다.
4. 움직임이 감지된 시점의 정보(timestamp, amplitude, flag)를 CSV 파일로 저장한다.
5. 처음 실행 시에는 움직임 없는 상태의 데이터를 일정량 수집하여 calibration을 생성한다.

[전처리 과정]
1. 복소수 변환 & 진폭 추출  
2. 중앙 서브캐리어 제거 (노이즈 많은 DC 근처)  
3. Z-score 정규화 (기준선 기반)  
4. 이상치 제거 (Hampel Filter)  
5. 중심화 (detrending)  
6. 변화량 요약  
   - 각 프레임에서 서브캐리어 진폭의 표준편차를 구해 변화량으로 사용  
   - 변화량의 프레임 간 미분 → 얼마나 급격히 변화했는지 계산  
   - 절댓값 + 이동 평균 적용 (Overlap-Save 방식)
7. 지수 이동 평균(EWMA) 기반 임계값 설정  
최종 출력  
   → 각 시간 프레임마다 [timestamp, 변화량, 움직임 여부(0/1), 임계값]을 CSV에 기록
'''