# Crack Segmentation 프로젝트

## 📖 Overview

- Duration : 2025.07 ~ 2025.08
- Crack 이미지를 통해 Crack Segmentation Task를 수행하는 모델 개발
- 다양한 유형의 두께, 형태의 Crack 존재
- 모델 [SCSegamba](https://github.com/Karl1109/SCSegamba) 고정, input size 512x512 고정

## 📄 Metrics
- IoU
<img width="331" height="152" alt="image" src="https://github.com/user-attachments/assets/0fcf714a-9112-47c5-86c7-c935394ef602" />


## 🧪 실험 내용

- Crack 마스크 데이터 전처리
- 데이터 증강 실험
- 모델 구조 변경 실험
- Fine-tuning 실험
- 후처리 실험 (Morphology, BPR, TTA)
- 배경 제거 실험

## 🎯 실험 결과

- 초기 성능 max mIoU 기준 0.639
- 최종 결과 max mIoU 기준 0.704 (+0.065)
