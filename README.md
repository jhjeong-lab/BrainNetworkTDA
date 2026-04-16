# BrainNetworkTDA

동적 뇌 네트워크 분석(Dynamic Brain Netowrk Analysis)을 위한 위상수학적 데이터 분석(TDA) 방법론 구현 코드입니다.
두 가지 TDA 접근법인 **Persistence Vineyard**와 **Dynamic TDA**를 각각 EEG, fMRI 데이터에 적용합니다.

Seoul National University, Department of Mathematical Science — Master's Thesis

---

## 개요

정적인 표현에 집중했던 기존 뇌 네트워크 분석의 한계를 극복하기 위해, 두 가지 동적 TDA 방법론을 구현·비교합니다.

| | Persistence Vineyard | Dynamic TDA |
|---|---|---|
| **데이터** | EEG (Motor Imagery vs Rest) | fMRI (HCP Resting-state) |
| **목표** | 위상적 특징의 시간적 추적 | 피험자 그룹 간 네트워크 동역학 비교 |
| **핵심 알고리즘** | Hungarian algorithm (매칭) | Dynamic Embedding + Permutation test |
| **결과** | Vineyard 시각화, Bottleneck Distance | 그룹 간 유의한 위상적 차이 (p=0.04) |

---

## 방법론

### 1. Persistence Vineyard

연속된 영속성 다이어그램에서 개별 위상적 특징의 궤적을 시간 축으로 추적합니다.

- **데이터**: OpenNeuro Motor Imagery vs Rest EEG (15채널)
- **전처리**: 50Hz notch filter → Butterworth bandpass (0.5–45 Hz) → ICA artifact removal
- **슬라이딩 윈도우**: 30초 window, 2초 step → 15×15 거리 행렬 (d = 1 − |corr|)
- **분석**: H0·H1 영속성 다이어그램 계산 → Hungarian algorithm으로 연속 다이어그램 매칭 → Vine 구성
- **주파수 대역**: Alpha (8–13 Hz), Beta (13–30 Hz)
- **시각화**: Vineyard plot, Bottleneck Distance 시계열 (이벤트 블록 오버레이)

> Cohen-Steiner et al. (2006) 알고리즘 대신 Hungarian algorithm 기반 휴리스틱 매칭을 구현

### 2. Dynamic TDA

Dynamic Embedding으로 다변량 시계열의 동적 위상 구조를 포착하고, 그룹 간 차이를 통계적으로 검정합니다.

- **데이터**: HCP FIX-Denoised resting-state fMRI (10 male / 10 female)
- **파셀레이션**: AAL 아틀라스 (116 ROI), z-score 표준화, detrending
- **슬라이딩 윈도우**: 60 TR window, 1 TR step → 6670차원 상관 벡터 (116×116 행렬 상삼각)
- **Dynamic Embedding**: 상관 벡터 시계열 → 60×60 유클리드 거리 텐서
- **위상적 거리**: Birth-death decomposition (Maximum Spanning Tree 기반) → 2-Wasserstein distance → 피험자 쌍별 20×20 거리 행렬
- **통계 검정**: 비율 통계량 λ(z) + Permutation test (10,000 permutations)
- **결과**: λ = 1.1610, p = 0.0401 → 남녀 그룹 간 유의한 위상적 차이 확인

> [PH-STAT (Moo K. Chung, 2025)](https://github.com/appliedtopology/ph-stat) 기반 구현

---

## 프로젝트 구조

```
BrainNetworkTDA/
├── Src/
│   │   ── [Persistence Vineyard] ──────────────────────────
│   ├── Load_EEG_Data.py        # EEG 로드, 주파수 필터링, 거리 행렬 생성
│   ├── Build_Vineyard.py       # Persistent Homology (GUDHI) + Vineyard 구축
│   ├── Plot_Vineyard.py        # Vineyard · Bottleneck Distance 시각화
│   │
│   │   ── [Dynamic TDA] ────────────────────────────────────
│   ├── Load_MRI_Data.py        # HCP fMRI 파셀레이션, 슬라이딩 윈도우 상관 행렬
│   ├── Embedding.m             # Dynamic Embedding → 거리 텐서 생성 (MATLAB)
│   ├── Comparing.m             # 피험자 간 위상적 거리 계산 + 통계 검정 (MATLAB)
│   ├── DynamiceTest.m          # 실험 테스트 스크립트 (MATLAB)
│   │
│   └── AAL_Info.py             # AAL 116개 뇌 영역 기능 정보
│
└── Analysis/                   # 분석 노트북
    ├── Preprocess_EEG.ipynb
    ├── Preprocess_fMRI.ipynb
    ├── EEG_Event.ipynb
    ├── EEG_Vineyard.ipynb
    ├── Embedding.mlx
    ├── Comparing.mlx
    └── DynamiceTest.mlx
```

---

## 설치

```bash
# Python
pip install numpy pandas matplotlib seaborn plotly nilearn scikit-learn scipy gudhi mne ripser

# MATLAB (Dynamic TDA)
# PH-STAT 패키지 필요: https://github.com/appliedtopology/ph-stat
```

---

## 데이터 경로

```
Dataset/
├── {subject_id}/MNINonLinear/Results/rfMRI_{run}/   # HCP fMRI
└── EEG/derivatives/                                  # EEG (BIDS 포맷)
```

---

## 참고 문헌

- Yoo et al. (2016). Topological persistence vineyard for dynamic functional brain connectivity. *Journal of Neuroscience Methods*
- Chung et al. (2022). Dynamic topological data analysis of functional human brain networks. *arXiv:2210.09092*
- Cohen-Steiner et al. (2006). Vines and vineyards by updating persistence in linear time. *SoCG*
