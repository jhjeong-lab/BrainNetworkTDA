import os
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def compute_persistence_diagram(distance_tensor):
    
    diagrams_H0 = []
    diagrams_H1 = []
    num_windows = len(distance_tensor)
    
    for i, mat in enumerate(distance_tensor): # 바로 리스트를 사용
        if (i + 1) % 20 == 0 or i == num_windows - 1:
            print(f"  Processing matrix for window {i+1}/{num_windows}")
            
        rips_complex = gd.RipsComplex(distance_matrix=mat, max_edge_length=1.01)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        diag = simplex_tree.persistence()
        
        current_H0 = []
        current_H1 = []
        for dim, (birth, death) in diag:
            if death == float('inf') or death > 1.0:
                death = 1.0
            if birth == float('inf') or birth > 1.0:
                birth = 1.0
            
            if death > birth + 1e-9: # 유효한 페어만
                if dim == 0:
                    current_H0.append([birth, death])
                elif dim == 1:
                    current_H1.append([birth, death])

        diagrams_H0.append(np.array(current_H0) if current_H0 else np.empty((0,2)))
        diagrams_H1.append(np.array(current_H1) if current_H1 else np.empty((0,2)))

    if diagrams_H0:
        print(f"  H0 Diagrams: {len(diagrams_H0)}. (First H0 shape: {diagrams_H0[0].shape if diagrams_H0[0].size > 0 else 'Empty'})")
    if diagrams_H1:
        print(f"  H1 Diagrams: {len(diagrams_H1)}. (First H1 shape: {diagrams_H1[0].shape if diagrams_H1[0].size > 0 else 'Empty'})")
        
    return diagrams_H0, diagrams_H1
        
def plot_persistence_diagrams(dgm0, dgm1, subject_id=None, window_index=None):
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- H0 다이어그램 그리기 ---
    ax_h0 = axes[0]
    
    # H0 제목 설정
    title_h0 = 'H0 Persistence Diagram'

    if dgm0 is not None and dgm0.size > 0:
        max_val_h0 = np.max(dgm0)
        ax_h0.scatter(dgm0[:, 0], dgm0[:, 1], color='blue', s=20, label='H0 Pairs')
        ax_h0.plot([0, max_val_h0], [0, max_val_h0], 'k--', label='Diagonal')
        plot_limit = max_val_h0 * 1.05
    else:
        ax_h0.text(0.5, 0.5, 'No H0 features', ha='center', va='center', transform=ax_h0.transAxes)
        ax_h0.plot([0, 1], [0, 1], 'k--', label='Diagonal') # 기본 대각선
        plot_limit = 1.05
        
    ax_h0.set_title(title_h0)
    ax_h0.set_xlabel('Birth')
    ax_h0.set_ylabel('Death')
    ax_h0.set_xlim(-0.01, plot_limit)
    ax_h0.set_ylim(-0.01, plot_limit)
    ax_h0.grid(True)
    ax_h0.legend(loc='lower right')
    ax_h0.set_aspect('equal', adjustable='box')

    # --- H1 다이어그램 그리기 ---
    ax_h1 = axes[1]

    # H1 제목 설정
    title_h1 = 'H1 Persistence Diagram'

    if dgm1 is not None and dgm1.size > 0:
        max_val_h1 = np.max(dgm1)
        ax_h1.scatter(dgm1[:, 0], dgm1[:, 1], color='green', s=20, label='H1 Pairs')
        ax_h1.plot([0, max_val_h1], [0, max_val_h1], 'k--', label='Diagonal')
        plot_limit_h1 = max_val_h1 * 1.05
    else:
        ax_h1.text(0.5, 0.5, 'No H1 features', ha='center', va='center', transform=ax_h1.transAxes)
        ax_h1.plot([0, 1], [0, 1], 'k--', label='Diagonal')
        plot_limit_h1 = 1.05

    ax_h1.set_title(title_h1)
    ax_h1.set_xlabel('Birth')
    ax_h1.set_ylabel('Death')
    ax_h1.set_xlim(-0.01, plot_limit_h1)
    ax_h1.set_ylim(-0.01, plot_limit_h1)
    ax_h1.grid(True)
    ax_h1.legend(loc='lower right')
    ax_h1.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
    
def build_vines(diagrams, match_threshold=0.1):
    """
    연속된 영속성 다이어그램으로부터 바인(vine)을 구축합니다.
    헝가리안 알고리즘을 사용하여 시간 단계별로 점들을 매칭합니다.

    Args:
        diagrams (list): 영속성 다이어그램(Numpy 배열)의 리스트.
                         각 다이어그램의 shape는 (n_points, 2)여야 합니다.
        match_threshold (float): 두 점을 동일한 바인의 일부로 간주하기 위한 최대 거리.

    Returns:
        list: 구축된 바인들의 리스트. 각 바인은 [(시간, 탄생, 소멸), ...] 형태의 튜플 리스트입니다.
    """
    if not diagrams:
        return []

    completed_vines = []
    active_vines = {}  # {vine_id: {'path': [...], 'last_point': ndarray}}
    next_vine_id = 0

    # t=0: 첫 번째 다이어그램의 모든 점을 새로운 활성 바인으로 초기화
    initial_diagram = diagrams[0]
    # 무한대 값을 가진 점은 분석에서 제외 (또는 적절한 값으로 대체 가능)
    initial_diagram = initial_diagram[np.all(np.isfinite(initial_diagram), axis=1)]
    
    for i in range(len(initial_diagram)):
        birth, death = initial_diagram[i]
        active_vines[next_vine_id] = {
            'path': [(0, birth, death)],
            'last_point': initial_diagram[i]
        }
        next_vine_id += 1
    
    # t > 0: 시간 흐름에 따라 바인 추적
    for t in range(len(diagrams) - 1):
        # 현재 활성 바인의 마지막 점들 (D1)
        prev_points = np.array([data['last_point'] for data in active_vines.values()]) if active_vines else np.empty((0, 2))
        prev_vine_ids = list(active_vines.keys())

        # 다음 시간 단계의 점들 (D2)
        next_points = diagrams[t + 1]
        next_points = next_points[np.all(np.isfinite(next_points), axis=1)]

        # 매칭된 점과 바인을 저장할 변수
        d2_matched_indices = set()
        current_step_matches = {} # {vine_id: new_point_info}

        # D1과 D2에 점이 모두 있을 경우에만 매칭 수행
        if prev_points.size > 0 and next_points.size > 0:
            # 비용 행렬 계산 (L-infinity norm)
            cost_matrix = np.max(np.abs(prev_points[:, np.newaxis, :] - next_points[np.newaxis, :, :]), axis=2)
            
            # 헝가리안 알고리즘으로 최적 매칭 찾기
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                cost = cost_matrix[r, c]
                if cost < match_threshold:
                    vine_id = prev_vine_ids[r]
                    matched_point = next_points[c]
                    
                    current_step_matches[vine_id] = matched_point
                    d2_matched_indices.add(c)

        # 활성 바인 업데이트
        next_active_vines = {}
        
        # 1. 매칭된 바인: 경로를 연장하고 활성 상태 유지
        for vine_id, new_point in current_step_matches.items():
            active_vines[vine_id]['path'].append((t + 1, new_point[0], new_point[1]))
            active_vines[vine_id]['last_point'] = new_point
            next_active_vines[vine_id] = active_vines[vine_id]

        # 2. 매칭되지 않은 이전 바인: 추적 종료하고 완료 목록으로 이동
        for vine_id in prev_vine_ids:
            if vine_id not in current_step_matches:
                completed_vines.append(active_vines[vine_id]['path'])

        # 3. 매칭되지 않은 새로운 점: 새로운 활성 바인으로 시작
        for i in range(len(next_points)):
            if i not in d2_matched_indices:
                new_point = next_points[i]
                next_active_vines[next_vine_id] = {
                    'path': [(t + 1, new_point[0], new_point[1])],
                    'last_point': new_point
                }
                next_vine_id += 1
        
        active_vines = next_active_vines

    # 루프 종료 후 남은 활성 바인들을 모두 완료 목록에 추가
    for vine_id, data in active_vines.items():
        completed_vines.append(data['path'])
        
    return completed_vines

def vines_to_time_value(vines):
    new_vines = []
    for vine in vines:
        if vine:
            tv_list = []
            for (t, birth, death) in vine:
                value = death - birth
                # GUDHI에서 death가 birth보다 작게 나오는 경우가 간혹 있을 수 있으므로 (수치적 문제), 방어 코드
                if value < 0: value = 0 
                tv_list.append((t, value))
            new_vines.append(tv_list)
        else:
            new_vines.append([])
    return new_vines