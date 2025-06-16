# Production System TeamProject (WIP)
2025년 1학기 생산시스템 구축실무 과목의 팀 프로젝트에 대한 레포지토리입니다.

이 레포지토리는 Deploy용 레포지토리이며, 개발용 레포지토리는 [여기](https://github.com/Nekerworld/Production_System_TeamProject)를 클릭해주세요

## ⚙️ 개발 환경 및 실행 환경 (Environment)

이 프로젝트는 열풍건조 공정 데이터를 기반으로 한 딥러닝 품질 예측 모델을 구현하며, 다음과 같은 소프트웨어 및 하드웨어 환경에서 수행되었습니다.
### [데이터 링크 (Kamp AI)](https://www.kamp-ai.kr/aidataDetail?page=1&DATASET_SEQ=23)

## 📊 데이터 소개 (Dataset Overview)

| 항목     | 내용                                                    |
| ------ | ----------------------------------------------------- |
| 데이터셋 명 | 장비이상 조기탐지 AI 데이터셋                                     |
| 적용 공정  | 열풍건조(Baking·Heat‐Dry)                                 |
| 제공 기관  | KAIST (수행: ㈜에이비에이치, ㈜임픽스)                             |
| 목적     | 설비 예지보전 및 이상 탐지                                       |
| 등록·수정일 | 2021-12-27 등록, 2025-04-02 최종 수정                       |
| 데이터 크기 | 307 083 개의 데이터 포인트, 실제 데이터의 개수는 CSV가 51,084, WAV가 183개, 총 1.8 GB                                 |

이 데이터에는 CSV, WAV 파일 형태의 데이터가 있습니다.

아래는 최종발표자료의 일부입니다.

![image](https://github.com/user-attachments/assets/2d49e857-427e-4ecb-9956-f132c9a356bd)
![image](https://github.com/user-attachments/assets/022c1680-2e64-4f60-be73-5e0e16a0ee0f)
![image](https://github.com/user-attachments/assets/b37d5bec-c5f0-4dd3-80c0-d8f6d0f9d856)
![image](https://github.com/user-attachments/assets/f6bb1cd8-bbbd-4a44-b9b8-2e6f7fed1607)

Process 컬럼은 어떠한 공정에서 측정된 데이터인지 나타내는 컬럼으로, 예를 들자면 1번은 배추 건조 공정, 2번은 웨이퍼 건조 공정.... 과 같은 식입니다.

---

### 수집 방법과 형태

* 공정 제어용 PLC와 센서로부터 온도, pH, 전압, 시간 등 실시간 데이터를 수신하여 CSV 파일로 저장한다.
* 열풍건조 공정 팬에 부착된 사운드 센서로 음성 신호를 수집해 WAV 파일로 저장한다.
* 최종 데이터셋은 데이터 CSV 33종, Error Lot List, WAV 183종으로 구성된다.

### 주요 특징

* 센서·사운드 데이터의 이중 입력 구조로 설비 상태를 다각도로 파악한다.
* 48개 이상의 공정 변수를 포함해 모델 학습에 충분한 피처를 제공한다.
* 가이드북에 LSTM-AE, Decision Tree 적용 절차가 명시돼 있어 재현이 용이하다.

---

### 📌 사용된 Python 환경

* **Python 버전**: `3.7.16`
* **가상환경 추천**: `venv` 또는 `conda` 환경 사용 권장
* [Tensorflow GPU 환경 구축 가이드](https://youtu.be/M4urbN0fPyM?list=FLymXUrZPMX6J6TBv0ytj4jA)

---

### 📦 주요 패키지 및 라이브러리

| 라이브러리                   | 버전             | 용도                   |
| ----------------------- | -------------- | -------------------- |
| `tensorflow`            | 2.6.0          | DNN 모델 구축 및 학습       |
| `keras-preprocessing`   | 1.1.2          | 딥러닝 전처리 유틸           |
| `numpy`                 | 1.21.5         | 수치 연산                |
| `pandas`                | 1.3.5          | 데이터프레임 처리            |
| `scikit-learn`          | 1.0.2          | 전처리, 모델 평가 등         |
| `scipy`                 | 1.7.3          | 수학/통계 기반 함수          |
| `matplotlib`, `seaborn` | 3.5.3 / 0.12.2 | 시각화                  |
| `imbalanced-learn`      | 0.8.1          | 클래스 불균형 처리 (SMOTE 등) |
| `xgboost`               | 1.6.2          | 비교용 머신러닝 모델          |

추가적으로 `tensorboard`, `protobuf`, `h5py`, `joblib`, `graphviz` 등도 함께 사용되었습니다.

---

### 📎 실행 방법

이 레포지토리를 클론한 후, 해당 폴더에서 bash/CMD를 열고 아래의 명령어를 입력해주세요.

```bash
pip install -r requirements.txt
python main.py
```
