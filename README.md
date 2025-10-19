# Autonomous Driving AI Chanllenge
 **자율주행 AI 챌린지 대회**를 위해 진행된 검출 모델 학습 파이프라인
 
 Ubuntu 20.04, RTX 3070 GPU 1개 - 최종 점수 **49.78 (Rank 5)**

<br />

## 주요 특징

- **모델 아키텍처**
  - [PV-RCNN++](https://arxiv.org/abs/2102.00463)

- **데이터 전처리**
  - augmentation
    - EDA 시 64ch, 128ch 혼합 확인, 이후 test dataset은 64ch만 있는 것을 확인하고 downsampling 추가
    - Flip, Scale, rotation, translation 등

- **학습 기법**
  - `OpenPCDet`을 활용한 **3D Detection Training**
  - `Tensorboard`을 활용한 실시간 학습 그래프 시각화

- **검증 및 평가**
  - `Waymo L2/mAP` 기반 성능 평가
  - 2024 dataset의 64ch만 추출하여 val.py 진행 (검증)


<br />


## 학습 파이프라인

```text
[Dataset] 
  ↓
[Augmentation]
  ↓
[CustomDataset]
  ↓
[PV-RCNN++ 모델 학습]
  ↓
[최적 모델 저장]
  ↓
[Validation: Waymo L2/mAP] 
```


<br />


## 대회 참여 전 공부
- **논문 공부**
  - [PointNet(CVPR 2017) Review](https://cafe.daum.net/SmartRobot/RoVa/2256)
  - [VoxelNet(CVPR 2017) Review](https://cafe.daum.net/SmartRobot/RoVa/2261)
  - [SECOND(2018) Review](https://cafe.daum.net/SmartRobot/RoVa/2273)
  - [Center-based 3D Object Detection and Tracking](https://kind-slip-86b.notion.site/Center-based-3D-Object-Detection-and-Tracking-2808a2c2bfdb80d2a308c5ea2a45c8f4?source=copy_link)
  - [3D 후보 모델 논문 증강 확인 및 제안](https://cafe.daum.net/SmartRobot/RoVa/2348)
 

- **OpenPCDet 프레임워크 공부**
  - [모델 설정 파일 조사](https://cafe.daum.net/SmartRobot/RoVa/2275)
  - [Voxel-RCNN 훈련 및 추론](https://cafe.daum.net/SmartRobot/RoVa/2282)
  - [Pretrained model / backbone freeze (pv-rcnn)](https://cafe.daum.net/SmartRobot/RoVa/2284)
  - [.pkl 정보 활용 및 분석](https://cafe.daum.net/SmartRobot/RoVa/2286)
  - [데이터 증강 전후 결과 확인 및 추가 (pv-rcnn)](https://cafe.daum.net/SmartRobot/RoVa/2290)
  - [Custom Dataset](https://cafe.daum.net/SmartRobot/RoVa/2304)
  - [3D detection 평가 지표](https://cafe.daum.net/SmartRobot/RoVa/2318)
  - [3D 후보 모델 선정](https://cafe.daum.net/SmartRobot/RoVa/2334)
  - [DSVT-Pillar .yaml pix & Training](https://cafe.daum.net/SmartRobot/RoVa/2344)


 
<br />

---

## 작년 데이터셋을 이용한 모의 테스트 구성
- 2024 자율주행 AI 챌린지 dataset에서 64ch 데이터셋만 추출하여 검증 진행
- [작년 데이터셋을 이용한 모의 테스트 + CenterPoint-Pillar](https://cafe.daum.net/SmartRobot/RoVa/2361)

---

<br />


## MODEL TRAIN (9.22-9.28)
- **PV-RCNN++ (test1, test2, test3)**
  - test1: downsampling 증강 적용, 클래스 별 가중치 적용, LR + GT-Sampling 조정
  - test2: test1 + 클래스 별 가중치, NMS 임계값, MIN_RADIUS, GRID_SIZE 조정 + translation 증강 추가
  - test3: test1 + LOCAL_AGGREGATION_TYPE, PointCloudRange, VoxelSize 조정
- score: X (리더보드 제출 X)  
- [MODEL TRAIN (9.22-9.28)](https://cafe.daum.net/SmartRobot/RoVa/2372)

<br />


---

<br />


## MODEL TRAIN (9.29-10.6)
- **PV-RCNN++ (test3, test4)**
  - test3: test1 + LOCAL_AGGREGATION_TYPE, PointCloudRange, VoxelSize 조정
  - test4: test3 + translation 증강 추가 + heatmap 조정 + point head 조정 + voxel size 조정
- [MODEL TRAIN (9.29-10.6)](https://cafe.daum.net/SmartRobot/RoVa/2375)

<br />


---

<br />



## MODEL TRAIN (9.22-9.28)
- **PV-RCNN++ (test1, test2, test3)**
  - test1: downsampling 증강 적용, 클래스 별 가중치 적용, LR + GT-Sampling 조정
  - test2: test1 + 클래스 별 가중치, NMS 임계값, MIN_RADIUS, GRID_SIZE 조정 + translation 증강 추가
  - test3: test1 + LOCAL_AGGREGATION_TYPE, PointCloudRange, VoxelSize 조정  
- [MODEL TRAIN (9.22-9.28)](https://cafe.daum.net/SmartRobot/RoVa/2372)

<br />


---

<br />



## MODEL TRAIN (9.22-9.28)
- **PV-RCNN++ (test1, test2, test3)**
  - test1: downsampling 증강 적용, 클래스 별 가중치 적용, LR + GT-Sampling 조정
  - test2: test1 + 클래스 별 가중치, NMS 임계값, MIN_RADIUS, GRID_SIZE 조정 + translation 증강 추가
  - test3: test1 + LOCAL_AGGREGATION_TYPE, PointCloudRange, VoxelSize 조정  
- [MODEL TRAIN (9.22-9.28)](https://cafe.daum.net/SmartRobot/RoVa/2372)
