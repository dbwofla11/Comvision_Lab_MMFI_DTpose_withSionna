# Comvision_Lab_MMFI_DTpose_withSionna

MMFI 데이터셋, DT-Pose 구조, NVIDIA Sionna RT 환경을 활용한
WiFi CSI 기반 3D Human Pose Estimation 연구 프로젝트.

---

# 프로젝트 개요

본 프로젝트는 WiFi CSI(Channel State Information)를 이용하여
카메라 없이 사람의 3D 자세를 추정하는 RF 기반 Human Pose Estimation 연구.

MMFI 멀티모달 데이터셋과 DT-Pose 아키텍처를 기반으로 모델을 구축하고,
Sionna RT 시뮬레이션 환경 및 실제 주거 환경 데이터를 활용하여
실험실 환경과 실제 환경 사이의 Domain Gap 문제를 분석함.

주요 목표:

* MMFI CSI 데이터 분석 및 전처리
* DT-Pose 기반 베이스라인 모델 구현
* Sionna RT 기반 무선 채널 시뮬레이션
* 실제 환경에서의 Domain Gap 검증
* Fine-Tuning 기반 환경 적응 성능 향상

---

# MMFI 데이터 처리 파이프라인

## 1. MMFI 데이터 분석 및 가공

* 데이터 규격 및 차원 검증
* 시간표(Timestamp) 연동 및 행동별 정렬
* CSI 파동 시각화
* 정답지(3D Keypoint) 시각화
* PyTorch Dataset 형태로 최종 통합

---

## 2. MMFI 기반 사전 학습 모델 구축

### Step 1 — 입력 특징 벡터 구성

* CSI Feature Vector 설계
* 신호 정규화 및 전처리

### Step 2 — 베이스라인 모델 설계

* PyTorch 기반 DT-Pose 스타일 구조 구현
* CSI → 3D Pose 매핑 구조 설계

### Step 3 — 학습 설정

* Loss Function 구성
* Optimizer 및 Scheduler 설정

### Step 4 — 학습 모니터링

* Checkpoint 저장
* Learning Curve 기록
* Validation Pipeline 구축

---

# 연구 진행 단계

## Phase 1 — MMFI Base Validation

MMFI 환경 내부에서 사전학습 모델이 정상적으로 동작하는지 검증함.

### 검증 내용

* MMFI 데이터셋 Train/Test 분리
* MPJPE(Mean Per Joint Position Error) 측정
* 3D Skeleton 복원 성능 확인
* Pose 시각화 검증

---

## Phase 2 — Domain Gap 검증

MMFI 환경에서 학습된 모델을 실제 주거 환경 데이터에 직접 적용하여
성능 저하 문제를 분석함.

### 실험 환경

* Sionna RT 기반 실내 시뮬레이션
* ESP32-S3 CSI 실측 데이터
* 실제 주거 환경 기반 Multipath 환경

### 예상 문제

* 벽 재질 차이
* 가구 반사
* 다중 경로(Multipath) 간섭
* 실제 RF 노이즈

등으로 인해 성능이 급격히 저하될 것으로 예상됨.

### 목표

실험실 데이터만으로 학습된 모델은
실제 환경에서 일반화 성능이 부족하다는 점을 검증함.

---

## Phase 3 — Fine-Tuning 기반 환경 적응

실제 환경 CSI 데이터를 소량 추가 학습하여
모델이 새로운 RF 환경에 적응하도록 만듬.

### 적용 기법

* Transfer Learning
* Domain Adaptation
* Fine-Tuning

### 검증 내용

* Loss 감소 추이 분석
* Learning Curve 비교
* 성능 회복 여부 검증

---

## Phase 4 — 최종 성능 비교 및 시각화

### 정량적 평가

다음 성능을 비교합니다:

* 사전학습 모델
* Fine-Tuning 적용 모델

평가 지표:

* MPJPE
* Pose Reconstruction Accuracy
* Skeleton Stability

### 정성적 평가

3D Skeleton 시각화를 통해:

* Ground Truth
* Pretrained Only
* Fine-Tuned(Ours)

결과를 비교 분석합니다.

---

# 기술 스택

* Python
* PyTorch
* TensorFlow
* NVIDIA Sionna RT
* WiFi CSI Signal Processing
* 3D Pose Visualization

---

# 최종 목표

본 연구의 목표는
실험실 기반 데이터셋과 실제 주거 환경 사이의 Domain Gap을 줄여
실제 환경에서도 안정적으로 동작하는 RF 기반 3D Human Pose Estimation 모델을 구축하는 것입니다.
