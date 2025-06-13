import os
import json
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import matplotlib.patches as mpatches 
import mne 
import pandas as pd
import seaborn as sns 

def plot_selected_vines(scores, title="Persistence Vines", num_vines_to_plot=15):
    plt.figure(figsize=(20, 5))
    
    # 평균 지속성이 높은 순으로 정렬 (선택 사항)
    # scores_sorted = sorted(scores, key=lambda v: np.mean([p[1] for p in v]) if v else 0, reverse=True)
    
    # 또는 길이가 긴 순으로 정렬
    scores_sorted_by_length = sorted(scores, key=lambda v: len(v) if v else 0, reverse=True)

    
    count = 0
    for i, vine in enumerate(scores_sorted_by_length):
        if not vine:
            continue
        
        times, values = zip(*vine)
        plt.plot(times, values, linestyle='-', label=f'vine_{i}') # 원래 인덱스 기반 레이블
        plt.plot(times, values, linestyle='-') # 너무 많으면 레이블 생략
        count += 1
        if count >= num_vines_to_plot:
            break
            
    plt.xlabel('Time Index (Window)')
    plt.ylabel('Value (death - birth)')
    plt.title(f'{title}')
    plt.grid(True)
    if count < 20 : plt.legend() 
    plt.tight_layout()
    plt.show()

def plot_vines_with_event_blocks( 
    subject_event_blocks_data,
    vines_data_time_value,
    sfreq,
    window_len_sec,
    step_len_sec,
    event_colors,
    plot_title="Persistence Vines with Event Blocks",
    num_vines_to_plot=4,
    y_label="Value (Persistence)" 
    ):
    
    if not subject_event_blocks_data:
        print(f"피험자에 대한 이벤트 블록 데이터가 없습니다.")
        return

    total_duration_sec = subject_event_blocks_data[-1]['end_sec']
    n_total_samples = int(total_duration_sec * sfreq)
    window_n_samples = int(window_len_sec * sfreq)
    step_n_samples = int(step_len_sec * sfreq)
    
    num_windows = 0
    if n_total_samples >= window_n_samples:
        num_windows = (n_total_samples - window_n_samples) // step_n_samples + 1
    else:
        print(f"데이터 길이({total_duration_sec:.2f}s)가 윈도우 길이({window_len_sec}s)보다 짧아 플롯을 그릴 수 없습니다.")
        return

    window_event_colors = []
    for i in range(num_windows):
        window_start_sec = i * step_len_sec
        window_center_sec = window_start_sec + (window_len_sec / 2.0)
        current_window_event_color = event_colors['NO_EVENT_DATA']
        for block in subject_event_blocks_data:
            if block['start_sec'] <= window_center_sec < block['end_sec']:
                current_window_event_color = event_colors.get(block['name'], event_colors['UNKNOWN_EVENT'])
                break
        window_event_colors.append(current_window_event_color)

    fig, ax = plt.subplots(figsize=(22, 6))

    current_block_start_idx = 0
    for i in range(1, num_windows):
        if window_event_colors[i] != window_event_colors[current_block_start_idx] or i == num_windows - 1:
            end_idx = i if window_event_colors[i] != window_event_colors[current_block_start_idx] else num_windows
            color_to_use = window_event_colors[current_block_start_idx]
            if color_to_use != event_colors['NO_EVENT_DATA']:
                 ax.axvspan(current_block_start_idx, end_idx, 
                           color=color_to_use, alpha=0.3, ymin=0, ymax=1, zorder=1)
            current_block_start_idx = i
    if current_block_start_idx < num_windows and window_event_colors and \
       window_event_colors[current_block_start_idx] != event_colors['NO_EVENT_DATA']:
         ax.axvspan(current_block_start_idx, num_windows, 
                   color=window_event_colors[current_block_start_idx], alpha=0.3, ymin=0, ymax=1, zorder=1)

    vines_sorted_by_length = sorted(vines_data_time_value, key=lambda v: len(v) if v else 0, reverse=True)
    vine_plot_colors = plt.cm.get_cmap('tab10') # 색상맵 미리 정의
    
    count = 0
    plotted_vine_handles = []
    plotted_vine_labels = []

    for i, vine_path_tv in enumerate(vines_sorted_by_length): # tv: time-value
        if not vine_path_tv:
            continue
        
        # vine_path_tv는 이미 [(시간_인덱스, 값), ...] 형태
        time_indices = [p[0] for p in vine_path_tv]
        actual_values = [p[1] for p in vine_path_tv] # Y축에 사용될 값 (이미 persistence)
        
        valid_indices = [j for j, t_idx in enumerate(time_indices) if t_idx < num_windows]
        if not valid_indices:
            continue
            
        time_indices_filtered = [time_indices[j] for j in valid_indices]
        values_filtered = [actual_values[j] for j in valid_indices]

        if not time_indices_filtered:
            continue

        # num_vines_to_plot이 0이거나 양수일때만 색상 인덱싱 하도록 수정
        color_index = count % vine_plot_colors.N if num_vines_to_plot > 0 else count % vine_plot_colors.N
        line, = ax.plot(time_indices_filtered, values_filtered, linestyle='-', color=vine_plot_colors(color_index), zorder=2)
        
        if count < num_vines_to_plot :
            plotted_vine_handles.append(line)
            plotted_vine_labels.append(f'Vine {count+1}')
            
        count += 1
        if num_vines_to_plot > 0 and count >= num_vines_to_plot: # num_vines_to_plot이 0이면 모두 그림
             break
         
    ax.set_xlim(0, num_windows)
    ax.set_xlabel('Time Index (Window)')
    ax.set_ylabel(y_label)
    ax.set_title(f'{plot_title}')
    ax.grid(True, linestyle=':')

    event_legend_patches = []
    cleaned_event_color_map_for_legend = {
        name: color for name, color in event_colors.items()
        if name in ['Rest', 'Motor Imagery']
    }
    for event_label, color_val in cleaned_event_color_map_for_legend.items():
         event_legend_patches.append(mpatches.Patch(color=color_val, label=event_label, alpha=0.5))
    
    if event_legend_patches:
        leg1 = ax.legend(handles=event_legend_patches, loc='upper right', title='Task Blocks')
        ax.add_artist(leg1)

    if plotted_vine_handles:
         ax.legend(handles=plotted_vine_handles, labels=plotted_vine_labels, loc='upper left', bbox_to_anchor=(0.01, 0.9), title="Vines") # 범례 위치 조정


    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.show()
    
def compute_bottleneck_distance_series(diagram_series, epsilon=0.0):
    """
    GUDHI를 사용하여 연속된 지속성 다이어그램들 간의 Bottleneck 거리를 계산합니다.
    """
    bottleneck_distances = []
    # if not gudhi_available or len(diagram_series) < 2:
    #     if gudhi_available: print("비교할 다이어그램 쌍이 부족합니다.")
    #     return bottleneck_distances

    for i in range(len(diagram_series) - 1):
        diag1 = diagram_series[i]
        diag2 = diagram_series[i+1]
        
        # GUDHI는 (N,2) 형태의 float64 NumPy 배열을 기대함
        # birth < death 조건은 GUDHI 내부에서 처리되거나, 특정 조건에서는 문제 없을 수 있으나,
        # 일반적으로 유효한 persistence pair (death > birth)만 전달하는 것이 안전.
        # diagrams_H0/H1 생성 시 이미 np.array로 변환되었고, birth, death 값이 채워짐.
        # GUDHI의 bottleneck_distance는 빈 다이어그램도 처리 가능.
        valid_diag1 = diag1[diag1[:, 1] > diag1[:, 0]] if diag1.size > 0 else np.empty((0,2))
        valid_diag2 = diag2[diag2[:, 1] > diag2[:, 0]] if diag2.size > 0 else np.empty((0,2))
        
        dist = gd.bottleneck_distance(valid_diag1, valid_diag2, epsilon)
        bottleneck_distances.append(dist)
    return bottleneck_distances

def plot_bottleneck_distance_with_event_blocks(
    distances,
    homology_dim_str,
    subject_id,
    subject_event_blocks_data, # 해당 피험자의 이벤트 블록 리스트 [{start_sec:.., end_sec:.., name:..}, ...]
    sfreq_plot, window_len_sec_plot, step_len_sec_plot, # _plot 접미사로 내부 변수와 구분
    num_total_windows, # 전체 윈도우 개수 (distances 길이 + 1)
    event_colors,
    task_name_str # 그래프 제목용
):
    if not distances:
        print(f"{homology_dim_str}에 대한 Bottleneck 거리가 없어 플롯할 수 없습니다.")
        return

    plt.figure(figsize=(22, 5)) # figsize 조정
    ax = plt.gca()

    time_indices = np.arange(len(distances)) # 0부터 num_total_windows-2 까지
    ax.plot(time_indices, distances, marker='.', linestyle='-', label=f'Bottleneck Distance {homology_dim_str}')

    ax.set_xlabel('Time (Window Transition Index t to t+1)')
    ax.set_ylabel('Bottleneck Distance')
    ax.set_title(f'{subject_id}: {homology_dim_str} Bottleneck Distance')
    ax.grid(True, linestyle=':')
    
    # X축 범위: 0부터 (num_total_windows - 2) 까지의 점들이 있으므로, num_total_windows -1까지 표시
    ax.set_xlim(-1, num_total_windows - 1) 

    # 이벤트 블록 오버레이 (window_event_colors 생성 로직 필요)
    window_event_colors = []
    if num_total_windows > 0 and subject_event_blocks_data:
        for i in range(num_total_windows): # 0부터 num_total_windows-1까지
            window_start_sec = i * step_len_sec_plot
            window_center_sec = window_start_sec + (window_len_sec_plot / 2.0)
            current_window_event_color = event_colors['NO_EVENT_DATA']
            for block in subject_event_blocks_data:
                if block['start_sec'] <= window_center_sec < block['end_sec']:
                    current_window_event_color = event_colors.get(block['name'], event_colors['UNKNOWN_EVENT'])
                    break
            window_event_colors.append(current_window_event_color)

        # 이벤트 블록 그리기
        current_block_start_idx = 0
        for i in range(1, num_total_windows):
            if window_event_colors[i] != window_event_colors[current_block_start_idx] or i == num_total_windows - 1:
                end_idx = i if window_event_colors[i] != window_event_colors[current_block_start_idx] else num_total_windows
                color_to_use = window_event_colors[current_block_start_idx]
                if color_to_use != event_colors['NO_EVENT_DATA']:
                     ax.axvspan(current_block_start_idx -0.5, end_idx -0.5, # X축 인덱스에 맞게 조정
                               color=color_to_use, alpha=0.2, ymin=0, ymax=1, zorder=0)
                current_block_start_idx = i
        if current_block_start_idx < num_total_windows and window_event_colors and \
           window_event_colors[current_block_start_idx] != event_colors['NO_EVENT_DATA']:
             ax.axvspan(current_block_start_idx -0.5, num_total_windows -0.5,
                       color=window_event_colors[current_block_start_idx], alpha=0.2, zorder=0)
    
    # 범례
    handles, labels = ax.get_legend_handles_labels() # Bottleneck distance 범례
    block_legend_patches = []
    cleaned_event_color_map_for_legend = {
        name: color for name, color in event_colors.items()
        if name in ['Rest', 'Motor Imagery']
    }
    for event_label, color_val in cleaned_event_color_map_for_legend.items():
         block_legend_patches.append(mpatches.Patch(color=color_val, label=event_label, alpha=0.3))
    
    if block_legend_patches:
        ax.legend(handles=handles + block_legend_patches, labels=labels + [p.get_label() for p in block_legend_patches], loc='upper right')
    else:
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()