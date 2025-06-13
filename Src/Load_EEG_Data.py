import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_eeg_data(subject_id, run_id, freq, win_size, step_size):
    
    DERIVATIVES_PATH = '../Dataset/EEG/derivatives'
    TASK_NAME = 'task-MIvsRest'
    
    if freq == 'alpha':
        L_FREQ, H_FREQ = 8.0, 13.0
    elif freq == 'beta':
        L_FREQ, H_FREQ = 13.0, 30.0
    else:
        # 지원하지 않는 주파수 대역에 대한 처리
        print(f"오류: 지원하지 않는 주파수 대역 '{freq}' 입니다. 'alpha' 또는 'beta'를 사용하세요.")
        return None
        
    WINDOW_LENGTH_SEC = win_size 
    STEP_SIZE_SEC = step_size     
    
    fif_file_name = f"{subject_id}_{TASK_NAME}_{run_id}_preprocessed_raw.fif"
    fif_file_path = os.path.join(DERIVATIVES_PATH, fif_file_name)
    
    if not os.path.exists(fif_file_path):
        print(f"오류: 파일을 찾을 수 없습니다 -> {fif_file_path}")
        return None
    
    try:
        print(f"--- {subject_id} ({freq} 대역) 처리 시작 ---")
        raw = mne.io.read_raw_fif(fif_file_path, preload=True, verbose='WARNING')
        current_sfreq = raw.info['sfreq']
        raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design='firwin',
                   skip_by_annotation='edge', phase='zero-double', verbose='WARNING')
    except Exception as e:
        print(f"데이터 로드 또는 필터링 중 오류 발생: {e}")
        return None
    
    eeg_data = raw.get_data()
    n_channels, n_total_samples = eeg_data.shape
    
    window_n_samples = int(WINDOW_LENGTH_SEC * current_sfreq)
    step_n_samples = int(STEP_SIZE_SEC * current_sfreq)
    distance_matrices = []
    
    for start_sample in range(0, n_total_samples - window_n_samples + 1, step_n_samples):
        end_sample = start_sample + window_n_samples
        window_data = eeg_data[:, start_sample:end_sample]
        
        if window_data.shape[1] == 0:
            continue

        correlation_matrix = np.corrcoef(window_data)
        abs_correlation_matrix = np.abs(correlation_matrix)
        distance_matrix = 1 - abs_correlation_matrix
        distance_matrices.append(distance_matrix)
        
    print(f"--- {subject_id}: 거리 행렬 생성 완료 (크기: {len(distance_matrices), distance_matrices[0].shape}) ---")

    return distance_matrices

def plot_distance_matrix(distance_matrix, channel_names=None, subject_id=None, window_index=None):
    """
    주어진 거리 행렬(distance matrix)을 히트맵으로 시각화합니다.

    Args:
        distance_matrix (np.ndarray): 시각화할 (n, n) 크기의 거리 행렬.
        channel_names (list, optional): 히트맵 축에 표시될 채널 이름 리스트. 
                                        None이거나 길이가 맞지 않으면 기본값이 사용됩니다.
        subject_id (str, optional): 그래프 제목에 표시될 피험자 ID.
        window_index (int, optional): 그래프 제목에 표시될 윈도우 인덱스.
    """
    # 채널 이름이 제공되지 않았거나, 행렬 크기와 맞지 않으면 기본값 생성
    if channel_names is None or len(channel_names) != distance_matrix.shape[0]:
        channel_names = [f'Ch {i+1}' for i in range(distance_matrix.shape[0])]

    # 그래프 제목 설정
    title = "Distance Matrix"
    if subject_id:
        title += f" - Subject: {subject_id}"
    if window_index is not None:
        title += f", Window: {window_index}"

    # 히트맵 그리기
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, 
                annot=False, 
                cmap='viridis_r',  # _r을 붙이면 색상 반전 (낮은 값->진하게)
                xticklabels=channel_names, 
                yticklabels=channel_names,
                cbar_kws={'label': 'Distance (1 - |Correlation|)'})
    
    plt.title(title)
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()  # 레이블이 잘리지 않도록 조정
    plt.show()
    
def sanity_check_distance_matrix(matrix, matrix_name="Distance Matrix", expected_dim=15):
    """
    거리 행렬에 대한 sanity check를 수행하고 결과를 출력합니다.

    Args:
        matrix (np.ndarray): 확인할 거리 행렬.
        matrix_name (str): 출력 시 사용할 행렬의 이름.
        expected_dim (int): 예상되는 행렬의 차원 (예: 채널 수).
    """
    print(f"\n--- {matrix_name} Sanity Check ---")

    if not isinstance(matrix, np.ndarray):
        print("  오류: 입력된 데이터가 NumPy 배열이 아닙니다.")
        return

    # 1. 형태 및 크기 확인
    print(f"  Shape                : {matrix.shape}")
    print(f"  Number of entries    : {matrix.size}")
    
    is_square = matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    print(f"  Is square matrix     : {is_square}")

    if not is_square:
        print("    경고: 정방 행렬이 아닙니다! 추가 검사를 수행할 수 없습니다.")
        return
    elif matrix.shape[0] != expected_dim:
        print(f"    경고: 예상된 차원({expected_dim}x{expected_dim})과 다릅니다. 현재 차원: {matrix.shape[0]}x{matrix.shape[0]}")
    else:
        print(f"  Expected entries     : {expected_dim} * {expected_dim} = {expected_dim * expected_dim}")

    # 2. 값의 범위 확인 (d = 1 - |corr| 이므로 0과 1 사이)
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    print(f"  Min value            : {min_val:.4f}")
    print(f"  Max value            : {max_val:.4f}")
    if not (np.allclose(min_val, 0, atol=1e-9) or min_val > 0) or not (np.allclose(max_val, 1, atol=1e-9) or max_val < 1) :
        # 부동소수점 오차를 고려하여 0과 1을 약간 벗어나는 것도 체크 (단, 1-|corr| 이므로 보통은 정확히 0~1 범위)
        if not (0 <= min_val <= 1 and 0 <= max_val <= 1):
            print("    경고: 값의 일반적인 범위(0~1)를 벗어났습니다. d = 1 - |corr| 정의를 확인하세요.")


    # 3. 유효하지 않은 값 (NaN, Inf) 확인
    has_nans = np.isnan(matrix).any()
    has_infs = np.isinf(matrix).any()
    print(f"  Any NaNs             : {has_nans}")
    if has_nans:
        print(f"    경고: NaN 값이 {np.isnan(matrix).sum()}개 포함되어 있습니다!")
    print(f"  Any Infs             : {has_infs}")
    if has_infs:
        print(f"    경고: Inf 값이 {np.isinf(matrix).sum()}개 포함되어 있습니다!")

    # 4. 대각선 요소 확인 (0에 매우 가까운지)
    diagonal_elements = np.diag(matrix)
    is_diag_zero = np.allclose(diagonal_elements, 0, atol=1e-9) # atol은 허용 오차
    print(f"  Diagonal elements ~0 : {is_diag_zero}")
    if not is_diag_zero:
        non_zero_diags = diagonal_elements[~np.isclose(diagonal_elements, 0, atol=1e-9)]
        print(f"    경고: 일부 대각선 요소가 0이 아닙니다 (예: {non_zero_diags[:3]}...).")

    # 5. 대칭성 확인 (D_ij == D_ji)
    is_symmetric = np.allclose(matrix, matrix.T, atol=1e-9)
    print(f"  Is symmetric         : {is_symmetric}")
    if not is_symmetric:
        print("    경고: 행렬이 대칭적이지 않습니다!")
        # diff = np.abs(matrix - matrix.T)
        # print(f"      Max difference from transpose: {np.max(diff):.4e}")

    # 6. 모든 값이 음수가 아닌지 확인 (거리는 일반적으로 음수가 아님)
    all_non_negative = np.all(matrix >= -1e-9) # 부동소수점 오차 감안
    print(f"  All non-negative     : {all_non_negative}")
    if not all_non_negative:
        print("    경고: 음수 값을 포함하고 있습니다!")

# --- 함수 사용 예시 ---
if __name__ == '__main__':
    # 예시: sub-01, run-1의 alpha 대역 데이터에 대한 거리 행렬 계산
    matrices = load_eeg_data(
        subject_id = 'sub-01',
        run_id = 'run-1',
        freq = 'alpha',
        win_size = 30.0,  
        step_size = 2  
    )

    if matrices is not None:
        print("\n함수가 반환한 최종 거리 행렬의 크기:", matrices.shape)