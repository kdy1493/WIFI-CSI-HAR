import re
import os
import csv
import time
import threading
import queue
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
from collections import deque

# MQTT 연결 정보
BROKER_ADDRESS = "61.252.57.136"
BROKER_PORT = 4991
TOPICS = ["L0382/ESP/8"]

# 저장 디렉토리 및 설정
#CSV_DIR = r"Raw_CSI_To_CSV_NoActivity"
CSV_DIR = r"data\Raw_CSI_To_CSV_DoorOpen"
SUBCARRIERS = 52
CSV_WRITE_FRAME_COUNT = 128
os.makedirs(CSV_DIR, exist_ok=True)

# 데이터 버퍼 (raw I/Q pairs)
processing_buffer = {topic: deque(maxlen=CSV_WRITE_FRAME_COUNT) for topic in TOPICS}
packet_timestamps = {topic: deque(maxlen=CSV_WRITE_FRAME_COUNT) for topic in TOPICS}

# CSV 쓰기를 위한 큐
write_queue = queue.Queue()

def parse_custom_timestamp(ts):
    year = 2000 + int(ts[0:2])
    month = int(ts[2:4])
    day = int(ts[4:6])
    hour = int(ts[6:8])
    minute = int(ts[8:10])
    second = int(ts[10:12])
    millisecond = int(ts[12:15])
    microsecond = millisecond * 1000
    return datetime(year, month, day, hour, minute, second, microsecond)

# 백그라운드에서 CSV 파일을 쓰는 쓰레드
def writer_thread():
    while True:
        task = write_queue.get()
        if task is None:
            break
        topic, rows = task
        filename = f"{CSV_DIR}/{topic.replace('/','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # 변경된 헤더 생성
            header = ['timestamp'] + [x for i in range(SUBCARRIERS) for x in (f'I{i}', f'Q{i}')]
            writer.writerow(header)
            writer.writerows(rows)
        write_queue.task_done()

# CSI 데이터 수신 후 처리
def process_csi_data(topic, payload):
    try:
        match = re.search(r'time=(\d{15})', payload)
        if match:
            ts_str = match.group(1)
            packet_time = parse_custom_timestamp(ts_str)
        else:
            packet_time = datetime.now()

        # CSI raw I/Q 값 파싱
        csi_values = list(map(int, payload.split("CSI values: ")[1].split()))
        if len(csi_values) < SUBCARRIERS*2:
            return
        raw_iq = csi_values[:SUBCARRIERS*2]

        processing_buffer[topic].append(raw_iq)
        packet_timestamps[topic].append(packet_time.strftime("%Y-%m-%d %H:%M:%S.%f"))

        if len(processing_buffer[topic]) == CSV_WRITE_FRAME_COUNT:
            rows = []
            for i in range(CSV_WRITE_FRAME_COUNT):
                rows.append([packet_timestamps[topic][i]] + processing_buffer[topic][i])
            # 큐에 쓰기 작업 등록
            write_queue.put((topic, rows))
            processing_buffer[topic].clear()
            packet_timestamps[topic].clear()

    except Exception as e:
        print(f"[{topic}] Error: {e}")

# MQTT 연결 콜백
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        for topic in TOPICS:
            client.subscribe(topic)
    else:
        print(f"Failed to connect, return code {rc}")

# MQTT 메시지 콜백
def on_message(client, userdata, msg):
    process_csi_data(msg.topic, msg.payload.decode())

# MQTT 수신 쓰레드 함수
def mqtt_thread_func():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    client.loop_forever()

# 메인 함수
def main():
    # CSV writer 쓰레드 시작
    t_writer = threading.Thread(target=writer_thread, daemon=True)
    t_writer.start()

    # MQTT 수신 쓰레드 시작
    t_mqtt = threading.Thread(target=mqtt_thread_func, daemon=True)
    t_mqtt.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        write_queue.put(None)
        t_writer.join()

if __name__ == "__main__":
    main()