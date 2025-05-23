import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일들이 있는 디렉토리
csv_dir = "Raw_CSI_To_CSV_NoActivity"

# 모든 CSV 파일 찾기
csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

# 빈 리스트 생성
dfs = []

print("\n=== 각 파일별 데이터 분석 ===")
# 각 CSV 파일 읽어서 리스트에 추가
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)
    
    # 파일명 추출
    filename = os.path.basename(file)
    
    # 기본 통계 계산
    print(f"\n파일: {filename}")
    print(f"데이터 수: {len(df)} 행")
    print(f"컬럼: {', '.join(df.columns)}")
    
    # ActivityDetection 통계
    if 'ActivityDetection' in df.columns:
        print(f"ActivityDetection - 평균: {df['ActivityDetection'].mean():.4f}, 최대: {df['ActivityDetection'].max():.4f}, 최소: {df['ActivityDetection'].min():.4f}")
    
    # ActivityFlag 통계
    if 'ActivityFlag' in df.columns:
        print(f"ActivityFlag - 1의 개수: {df['ActivityFlag'].sum()}, 0의 개수: {len(df) - df['ActivityFlag'].sum()}")
    
    # Th 통계
    if 'Th' in df.columns:
        print(f"Th - 평균: {df['Th'].mean():.4f}, 최대: {df['Th'].max():.4f}, 최소: {df['Th'].min():.4f}")

# 모든 데이터프레임 합치기
merged_df = pd.concat(dfs, ignore_index=True)

# 시각화를 위한 데이터 준비
plt.figure(figsize=(15, 10))

# 1. I/Q 데이터 시각화 (첫 번째 서브캐리어)
plt.subplot(2, 1, 1)
plt.plot(merged_df['I0'], label='I0')
plt.plot(merged_df['Q0'], label='Q0')
plt.title('첫 번째 서브캐리어의 I/Q 데이터')
plt.legend()
plt.grid(True)

# 2. CSI 진폭 시각화 (여러 서브캐리어)
plt.subplot(2, 1, 2)
for i in range(0, 52, 10):  # 10개 서브캐리어마다 하나씩 선택
    amplitude = np.sqrt(merged_df[f'I{i}']**2 + merged_df[f'Q{i}']**2)
    plt.plot(amplitude, label=f'Subcarrier {i}')
plt.title('CSI 진폭 변화')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('csi_analysis.png')
plt.close()

# 결과를 새로운 CSV 파일로 저장
output_file = "merged_csi_data_noactivity.csv"
merged_df.to_csv(output_file, index=False)

print("\n=== 전체 데이터 통계 ===")
print(f"총 데이터 수: {len(merged_df)} 행")
if 'ActivityDetection' in merged_df.columns:
    print(f"전체 ActivityDetection - 평균: {merged_df['ActivityDetection'].mean():.4f}, 최대: {merged_df['ActivityDetection'].max():.4f}, 최소: {merged_df['ActivityDetection'].min():.4f}")
if 'ActivityFlag' in merged_df.columns:
    print(f"전체 ActivityFlag - 1의 개수: {merged_df['ActivityFlag'].sum()}, 0의 개수: {len(merged_df) - merged_df['ActivityFlag'].sum()}")
if 'Th' in merged_df.columns:
    print(f"전체 Th - 평균: {merged_df['Th'].mean():.4f}, 최대: {merged_df['Th'].max():.4f}, 최소: {merged_df['Th'].min():.4f}")

print(f"모든 CSV 파일이 {output_file}로 합쳐졌습니다.")
print(f"총 {len(merged_df)} 행의 데이터가 저장되었습니다.")
print(f"\n시각화 결과가 'csi_analysis.png' 파일로 저장되었습니다.") 