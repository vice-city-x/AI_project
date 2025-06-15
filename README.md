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



