import autorootcwd
import glob
from src.ours2 import (
    calibrate_beamforming_weights,
    evaluate_activity_vs_noactivity_features_side_by_side
)

# Set parameters
TOP_K = 5            # 몇개의 compressed channel 사용할지
WINDOW_SIZE = 32     # WINDOW 사이즈
TOP_CSV = 15         # 몇개를 Calibration에 쓸지
MAX_PAIRS = 50       # 좌우 비교할 파일 쌍 개수
K = 52               # Subcarrier 채널 수
INDICES_TO_REMOVE = None
#INDICES_TO_REMOVE = list(range(21, 32))  # 수동 채널 제거

# CSI Paths
ACTIVITY_CSI_PATHS = glob.glob("data/raw/raw_activity_csi/L0382*.csv")
NO_ACTIVITY_CSI_PATHS = glob.glob("data/raw/raw_noActivity_csi/L0382*.csv")

# Step 1: Beamforming Calibration.
# activity on 신호와 activity off 신호를 이용하여, 
# 어떤 W를 취해야 가장 Activity가 Maximize 될지 결정

W = calibrate_beamforming_weights(
    activity_paths=ACTIVITY_CSI_PATHS,
    no_activity_paths=NO_ACTIVITY_CSI_PATHS,
    k=K,
    top_csv=TOP_CSV,
    indices_to_remove=INDICES_TO_REMOVE
)
print("✅ Beamforming calibration completed.")

# ------------------------------------------- # 


# 원신호 채널만 보기 (no windowing)
evaluate_activity_vs_noactivity_features_side_by_side(
    ACTIVITY_CSI_PATHS=ACTIVITY_CSI_PATHS,
    NO_ACTIVITY_CSI_PATHS=NO_ACTIVITY_CSI_PATHS,
    W=W,
    top_k=TOP_K,
    max_pairs=MAX_PAIRS,
    window_size=WINDOW_SIZE,
    indices_to_remove=INDICES_TO_REMOVE,
    channel_vis=True
)

# 신호의 주요 Feature들을 보기 (moving window)
evaluate_activity_vs_noactivity_features_side_by_side(
    ACTIVITY_CSI_PATHS=ACTIVITY_CSI_PATHS,
    NO_ACTIVITY_CSI_PATHS=NO_ACTIVITY_CSI_PATHS,
    W=W,
    top_k=TOP_K,
    max_pairs=MAX_PAIRS,
    window_size=WINDOW_SIZE,
    indices_to_remove=INDICES_TO_REMOVE,
    channel_vis=False
)


