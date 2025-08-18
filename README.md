# 다중 클래스 균열 분할 프로젝트 (SCSegamba 기반)

이 프로젝트는 **SCSegamba** 모델을 기반으로 한 딥러닝 다중 클래스 균열 분할(Multi-Class Crack Segmentation) 솔루션입니다. Python으로 작성되었으며, MMSegmentation/MMClassification (`mmcls`) 프레임워크의 일부를 활용하는 것으로 보입니다.

## 주요 특징

*   SCSegamba 모델을 활용한 고성능 균열 분할
*   다중 클래스 지원 (예: 두께, 형태 등 다양한 유형의 균열)
*   MMClassification 프레임워크 기반의 유연한 실험 및 개발 환경

## 프로젝트 구조

-   `main.py`: 모델 학습 및 평가를 위한 메인 실행 스크립트입니다.
-   `engine.py`: 학습 및 추론의 핵심 로직을 담고 있는 엔진 파일입니다.
-   `models/`: **SCSegamba**를 포함한 다양한 모델 아키텍처가 정의되어 있습니다.
-   `datasets/`: 데이터셋 로딩 및 전처리를 위한 코드를 포함합니다.
    -   `crack_dataset.py`: 균열 데이터셋을 위한 커스텀 데이터 로더입니다.
-   `mmcls/`: MMSegmentation/MMClassification 프레임워크 관련 코드입니다.
-   `checkpoints/`: 학습된 모델의 가중치(weight)가 저장되는 디렉토리입니다.
-   `results/`: 모델의 추론 결과 및 평가 결과가 저장되는 디렉토리입니다.
-   `util/`: 로깅, 기타 유틸리티 함수 등 보조적인 기능들을 포함합니다.

## 사용 방법

1.  **(환경 설정)** 필요한 라이브러리를 설치합니다.
2.  **(데이터 준비)** `datasets` 디렉토리 구조에 맞게 학습 및 테스트 데이터를 준비합니다.
3.  **(학습 실행)** `main.py` 스크립트를 실행하여 모델 학습을 시작합니다.
    ```bash
    python main.py --[your_arguments]
    ```
4.  **(평가)** `eval_compute.py` 또는 관련 스크립트를 사용하여 학습된 모델의 성능을 평가합니다.

---
*이 README 파일은 Gemini에 의해 생성되었습니다.*