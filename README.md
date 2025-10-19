# Autonomous Driving AI Chanllenge
 **자율주행 AI 챌린지 대회**를 위해 진행된 검출 모델 학습 파이프라인
 
 Ubuntu 20.04, RTX 3070 GPU 1개 - 최종 점수 **49.78 (Rank 5)**

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


## 대회 참여 전 공부
- **논문 공부**
  - [PointNet(CVPR 2017) Review](https://cafe.daum.net/SmartRobot/RoVa/2256)
  - [VoxelNet(CVPR 2017) Review](https://cafe.daum.net/SmartRobot/RoVa/2261)
  - [SECOND(2018) Review](https://cafe.daum.net/SmartRobot/RoVa/2273)
  - [Center-based 3D Object Detection and Tracking](https://kind-slip-86b.notion.site/Center-based-3D-Object-Detection-and-Tracking-2808a2c2bfdb80d2a308c5ea2a45c8f4?source=copy_link)

- **OpenPCDet 프레임워크 공부**
  - 

---

## rock_classification_multi_gpu_v1.py
- **score**: 0.73568  
- Jetson 기반 Multi-Node 학습, Jetson AGX Orin 4대  
- resnet101 훈련  
- [Jetson 기반 Multi-Node 학습](https://cafe.daum.net/SmartRobot/RoVa/2206)

---

## rock_classification_multi_gpu_v2.py
- **score**: 0.76311  
- class weight, augmentation 추가  
- resnet50 훈련  
- [class weight, augmentation](https://cafe.daum.net/SmartRobot/RoVa/2216)

---

## rock_classification_multi_gpu_v3.py
- **score**: 0.78917  
- Two-Stage Fine-Tuning 적용, TTA 적용  
- resnet50 훈련  
- [Two-Stage Fine-Tuning](https://cafe.daum.net/SmartRobot/RoVa/2222)

---

## rock_classification_multi_gpu_v4.py
- **score**: 0.8195  
- Two-Stage Fine-Tuning 적용  
- resnet101 훈련  
- [Two-Stage Fine-Tuning resnet101](https://cafe.daum.net/SmartRobot/RoVa/2227)

---

## rock_classification_multi_gpu_v5.py
- **score**: 0.84178  
- TTA 삭제, Window RTX4070 super 환경에서 학습, Two-Stage Fine-Tuning 유지  
- resnet101 훈련  

