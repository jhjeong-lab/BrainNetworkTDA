# BrainNetworkTDA

뇌 기능적 연결성(Functional Connectivity)의 동적 변화를 위상수학적 데이터 분석(Topological Data Analysis, TDA)으로 분석하는 석사 학위 논문 구현 코드입니다.

fMRI와 EEG 데이터에 지속적 호몰로지(Persistent Homology)를 적용하고, **Vineyard** 기법으로 위상적 특징의 시간적 추적을 수행합니다.

---

## 프로젝트 구조

```
BrainNetworkTDA/
├── Src/                        # 핵심 분석 모듈
│   ├── Load_MRI_Data.py        # HCP fMRI 데이터 로드 및 슬라이딩 윈도우 상관 행렬 생성
│   ├── Load_EEG_Data.py        # EEG 데이터 로드 및 주파수 필터링, 거리 행렬 생성
│   ├── Build_Vineyard.py       # Persistent Homology 계산 및 Vineyard 구축 (Hungarian algorithm)
│   ├── Plot_Vineyard.py        # Vineyard 및 Bottleneck Distance 시각화
│   ├── AAL_Info.py             # AAL 아틀라스 116개 뇌 영역 기능 정보
│   ├── Embedding.m             # 슬라이딩 윈도우 기반 포인트 클라우드 임베딩 (MATLAB)
│   ├── Comparing.m             # 피험자 간 위상적 특징 비교 (MATLAB, PH-STAT 사용)
│   └── DynamiceTest.m          # 동적 뇌 네트워크 분석 테스트 (MATLAB)
│
└── Analysis/                   # 분석 노트북 및 실습 파일
    ├── Preprocess_fMRI.ipynb   # fMRI 전처리 파이프라인
    ├── Preprocess_EEG.ipynb    # EEG 전처리 파이프라인
    ├── EEG_Event.ipynb         # EEG 이벤트 기반 분석 (Motor Imagery vs Rest)
    ├── EEG_Vineyard.ipynb      # EEG 데이터에 Vineyard 적용
    ├── Embedding.mlx           # MATLAB Live Script: 임베딩 분석
    ├── Comparing.mlx           # MATLAB Live Script: 비교 분석
    └── DynamiceTest.mlx        # MATLAB Live Script: 동적 테스트
```

---

## 분석 파이프라인

```
원시 데이터 (fMRI / EEG)
        ↓
  슬라이딩 윈도우
        ↓
  상관 행렬 → 거리 행렬
        ↓
  Vietoris-Rips 복체
        ↓
  지속적 호몰로지 (H0, H1)
        ↓
  Vineyard 구축 (헝가리안 알고리즘)
        ↓
  시간-지속성 시계열 추출 / Bottleneck Distance
```

---

## 데이터

### fMRI
- **HCP (Human Connectome Project)** resting-state fMRI
- 파셀레이션: AAL 아틀라스 (116 ROI)
- TR: 0.72s, 슬라이딩 윈도우 상관 행렬 생성
- 데이터 경로: `Dataset/{subject_id}/MNINonLinear/Results/rfMRI_{run_label}/`

### EEG
- **Motor Imagery vs Rest** 과제 (BIDS 포맷)
- 주파수 대역: Alpha (8–13 Hz), Beta (13–30 Hz)
- 채널 간 상관 기반 거리 행렬 생성
- 데이터 경로: `Dataset/EEG/derivatives/`

---

## 설치

```bash
pip install numpy pandas matplotlib seaborn plotly nilearn scikit-learn scipy gudhi mne ripser
```

MATLAB 분석의 경우 [PH-STAT](https://github.com/appliedtopology/ph-stat) 패키지가 필요합니다.

---

## 주요 기능

- **동적 기능적 연결성**: 슬라이딩 윈도우로 시간에 따른 뇌 네트워크 변화 포착
- **Persistent Homology**: GUDHI 라이브러리로 H0 (연결 성분), H1 (루프) 특징 추출
- **Vineyard**: 연속된 영속성 다이어그램의 특징점을 헝가리안 알고리즘으로 시간 추적
- **Bottleneck Distance**: 인접 시간 단계 간 위상적 변화량 정량화
- **이벤트 오버레이 시각화**: 과제 블록(Rest / Motor Imagery)과 위상적 특징 변화 비교
