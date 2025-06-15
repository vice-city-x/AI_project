# AI_project

## Motivation

- 최근 AI 기술의 발전으로 사용자 감정 상태를 인식하고 반응하는 감정 분석 기술의 중요성이 증가하고 있습니다.
- 기존의 CNN 기반 감정 분류 모델은 명확한 감정 클래스를 분류하는 데에는 효과적이지만, 감정의 복잡성과 문맥적 의미를 반영하는 데에는 한계가 존재합니다.
- 이를 보완하기 위해 본 프로젝트에서는 VLM(Vision-Language Model)을 활용하여 이미지로부터 감정 상태를 설명하는 자연어 문장을 생성하고, 이를 임베딩하여 이미지 feature와 결합하는 멀티모달 감정 인식 구조를 제안합니다.




![image](https://github.com/user-attachments/assets/97247f12-f48b-4a0c-8982-222eaf73b7af)
<sub>[이미지 출처 링크](https://www.ki-it.com/_common/do.php?a=full&b=22&bidx=3223&aidx=35957)</sub>

## Dataset

### 1) 데이터 이름
### RAF-DB (Real-world Affective Faces Database)

| 구분   | 개수     |
|--------|----------|
| Train  | 9,816장  |
| Test   | 1,533장  |
| Valid  | 982장    |

![image](https://github.com/user-attachments/assets/901ce171-e9c6-45e2-b7a5-e8c6f3947ea9)
<sub>[이미지 출처 링크](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/data)</sub>

### 2) 데이터 탐색

![image](https://github.com/user-attachments/assets/00bd4646-b2bf-4871-b9b3-8d28a6a7b75d)
: Fear Data 적음 → Augmentation 활용 (transform = 회전, 크롭, 반전 등)

## Augmentation – transform 30 epochs
Model : Resnet18, batch size : 64, epochs : 30, learning rate : 0.001, patience=5  

### Transform 성능 비교

| Transform | Test Accuracy | Early Stopping Epoch |
|-----------|----------------|-----------------------|
| Original  | 0.7349         | Early stopping(8)                     |
| **Flip**      | **0.7462**         | Early stopping(10)                    |
| Rotate    | 0.7273         | Early stopping(12)                    |
| Crop      | 0.5773         | Early stopping(8)                     |


### CNN Baseline 성능 비교

| CNN         | epochs | 성능 (Test acc) | 비고               |
|-------------|--------|------------------|--------------------|
| VGG16       | 30     | 0.6680           | Early_stopping(16) |
| Resnet18    | 30     | 0.7462           | Early_stopping(12) |
| DenseNet121 | 30     | **0.7697**       | Early_stopping(17) |

### VLM (BLIP_v2)

#### VLM이란?
- VLM은 "Vision-Language Model"의 약자로, 이미지와 텍스트를 동시에 이해할 수 있는 인공지능 모델.
- 단순히 정의하면, 영상과 텍스트를 결합하여 이해하고 연관 지을 수 있는 모델.
![image](https://github.com/user-attachments/assets/50206850-d902-4190-b298-6e0ee134eb19)
Bordes, Florian, et al. "An introduction to vision-language modeling." arXiv preprint arXiv:2405.17247 (2024).

### VLM (BLIP_v2)
**prompt = "Question: What emotion is the person showing in the image? Answer:"**
![image](https://github.com/user-attachments/assets/6f8ca179-e9c8-4bac-9b3d-17360b5fa209)

### VLM (CLIP)

- **CLIP 텍스트 인코더만을 "문장 임베더"로 활용**
  - CNN에서 뽑은 이미지 feature와 합치기 위해, 문장도 벡터로 변경
  - CLIP의 텍스트 인코더는 문장을 **고차원 의미 벡터로 표현**하는 데 탁월함
![image](https://github.com/user-attachments/assets/b890d9d0-2f37-4190-8a6f-0dd398c0e7f2)

- 본 실험에서는 **ViT-L/14 모델의 텍스트 인코더**를 사용하였으며, 따라서 생성된 **텍스트 feature의 차원은 768**

### 왜 BLIP + CLIP을 같이 썼을까?

#### BLIP의 역할: 텍스트 생성기 (Caption Generator)
- BLIP2는 이미지를 입력받아 **"텍스트"** 를 생성하는 데 강함.
- 이 문장은 표정에 대한 언어적 설명이므로, 감정 인식에 아주 유용.

#### CLIP의 역할: 의미기반 임베더 (Text Encoder)
- CLIP은 문장을 숫자 벡터로 **임베딩**하는 데 특화.
- 자연어 감정 설명을 수치로 바꾸는 데 CLIP이 사용.

### Feature Extraction & Fusion

#### CNN – Image feature 추출
- 이미지를 **DenseNet-121 모델**에 통과시켜 feature를 추출 (차원: 1024)

#### CLIP – Text feature 추출
- BLIP에서 생성된 문장을 **CLIP의 Text Encoder**에 입력하여 feature 추출 (차원: 768)

#### 두 feature를 결합하여 하나의 멀티모달 감정 feature 생성
- 두 feature를 **concatenate 후 L2 정규화**
- 최종 feature 차원: **1792**

### MLP training & results – MLP + Attention

| MLP                         | Test acc | CNN 단일 모델 대비 향상률 |
|----------------------------|----------|----------------------------|
| MLP + Attention            | **0.8226** | **7.3%**                   |
| MLP + Self-Attention       | 0.8147   | 5.8%                       |
| MLP + Cross-Attention      | 0.8089   | 5.3%                       |

### MLP training & results – Best MLP
![image](https://github.com/user-attachments/assets/f981446a-01e6-4312-be6a-8caf1ac31333)
- 기존 MLP는 CNN와 CLIP feature을 동등하게 처리하여 중요한 정보를 구별하지 못함.
- Attention을 쓰면 중요한 feature에는 더 많은 가중치를
덜 중요한 feature에는 적은 가중치를 부여할 수 있어서 정보 손실 없이 효율적인 표현 학습이 가능

### MLP training & results – Best MLP (F1 Score 비교)

| Class         | CNN 단일모델 | CNN + VLM 멀티모달 | 향상률 |
|---------------|--------------|---------------------|--------|
| 0 (surprise)  | 0.7548       | 0.7976              | 5%     |
| 1 (fear)      | 0.4324       | **0.5846**          | **35%**|
| 2 (disgust)   | 0.5176       | 0.5521              | 6%     |
| 3 (happy)     | 0.9012       | 0.9184              | 1.9%   |
| 4 (sad)       | 0.7326       | 0.7930              | 8%     |
| 5 (angry)     | 0.6418       | 0.7284              | 13%    |
| 6 (neutral)   | 0.6864       | 0.8000              | 16%    |

### MLP training & results – 단일모델 vs 멀티모달 성능 비교

| 모델        | Test acc | CNN 단일 모델 대비 향상률 |
|-------------|----------|----------------------------|
| CNN         | 0.7697   | -                          |
| CNN + VLM   | **0.8226** | **7.3% 상승**              |
- CNN이 시각 정보만 반영한 것과 달리, VLM은 시각 + 언어 정보 간의 상호 보완적 의미를 활용하여 감정 분류 성능을 강화함.
- 표정이 모호하거나 혼합적인 경우, 텍스트 기반 맥락이 분류를 돕는 역할을 했을 가능성이 큼.
- 정확도 **7.3% 상승**은 실질적인 성능 향상.
- 단일모델 대비, 멀티모달 모델은 **정보를 더 풍부하고 정교하게 표현**할 수 있음을 증명함.

### Problems & Improvements – 프롬프트 품질의 한계
- **BLIP2는 행동 기반 학습 모델**로, **정적인 표정/감정 설명에 약함**
<img width="893" alt="image" src="https://github.com/user-attachments/assets/8dc9b59b-4458-4097-aa65-d31aba125f13" />
- 일부 이미지에서 부정확하거나 일관되지 않은 프롬프트 생성
- 감정 분류 성능에 부정적 영향을 줌
- 따라서 표정/감정 이미지 전용 VLM 탐색이 필요함

### Problems & Improvements - 데이터 불균형 문제
RAF-DB의 클래스 불균형 존재
<img width="613" alt="image" src="https://github.com/user-attachments/assets/9795a267-f536-4a22-a6fb-e1cd4810986f" />
- Upsampling, Downsampling 미적용
- 소수 클래스의 학습 불안정 가능성
- 추후 재학습 시 보완 필요

### Problems & Improvements - Augmentation 실험 일관성 부족
얼굴을 잘라내는 crop 기법은 감정 분석에서 중요한 요소로, 얼굴 표정에서 감정의 핵심 특징을 효율적으로 추출하고 배경 노이즈를 제거하는 데 효과적
<img width="890" alt="image" src="https://github.com/user-attachments/assets/9a69f2e5-79e5-4959-a720-2647e7e7eccd" />
- crop과 rotate는 효과가 낮아 제외되었으며, flip만 사용. 
- crop 비율 조정 등 세분화된 실험과 재현성 확보가 필요

### Problems & Improvements - Overfitting 방지 전략 미적용
<img width="553" alt="image" src="https://github.com/user-attachments/assets/85f593f1-1c04-4444-8eaa-fe314ffa232a" />
<img width="501" alt="image" src="https://github.com/user-attachments/assets/8a03f323-ab1a-41be-8451-fe7d0219842f" />

- 일부 모델에 Early Stopping, LR Scheduler 적용하여 일반화 성능 개선
- 그러나 일부는 여전히 과적합 발생
- 다양한 과적합 방지 기법 도입 필요






