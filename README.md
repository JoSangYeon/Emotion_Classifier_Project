# Emotion Classifier Project

## Introduction
+ AI HUB에서 제공하는 한국말 대화 뭉치에서 문장에 대한 감정을 분류하는 모델 설계
  + https://aihub.or.kr/aidata/7978
+ KoBERT와 RoBERTa의 성능을 비교
+ Pooling 전략 중 : CLS Tokenr과 Mean Pooling 전략의 성능을 비교
+ Metric Learning의 대표적 Loss Function인 Contrastive Loss와 Triplet Loss의 성능을 비교

## Data
Class_tag = ['불안', '슬픔', '기쁨']
### Train
|    | sentence                                                                          | label   |
|---:|:----------------------------------------------------------------------------------:|:--------|
|  0 | 오늘은 누구랑 같이 밥을 먹어야 할까?                                              | 불안    |
|  1 | 난 남들 다 받는 과외를 못 받아서 공부를 못하는 것 같아.                           | 슬픔    |
|  2 | 오늘 친구가 몸이 너무 안 좋아서 친구 잘못으로 내가 대신 혼났어.                   | 슬픔    |
|  3 | 우리 주위에는 불치병으로 인하여 어려운 분들이 많은 것 같아.                       | 슬픔    |
|  4 | 아이들이 사춘기가 되니 예전처럼 나에게 먼저 말을 걸지 않고 자꾸 나를 피하려고 해. | 불안    |
|  5 | 딸이 결혼을 안 하려고 해서 집사람이 화가 났어.                                    | 슬픔    |
|  6 | 학원 친구 중에 나보다 뛰어난 애가 눈에 밟혀서 화가 나.                            | 슬픔    |
|  7 | 아빠가 보너스를 받고 엄마와 내게 선물을 해주었어!                                 | 기쁨    |
| .. | ... | ...| 

### TEST
|    | sentence                                                                                             | label   |
|---:|:----------------------------------------------------------------------------------------------------:|:--------|
|  0 | 수험생인 아들과 기분 전환 겸 집 앞 산을 등산하다가 아들이 발목을 접질려서 너무 마음 아파.            | 불안    |
|  1 | 내 장례식만큼은 내가 원하는 대로 해 달라고 말했는데 자식들은 들은 체도 안 해. 속상해.                | 슬픔    |
|  2 | 친구가 지난해에 이혼을 했어. 양육비를 월 오십만 원씩 주기로 하고 아이는 부인이 키우기로 했다는 거야. | 불안    |
|  3 | 학교에 늦어서 엄마 출근길에 나 좀 데려다 달라고 했는데 엄마가 화가 났을까?                           | 슬픔   |
|  4 | 가족들이 집으로 빨리 돌아오라고 했는데 늦게 가서 걱정만 시켰어.                                      | 슬픔    |
|  5 | 건강에 회의적이야.                                                                                   | 불안    |
|  6 | 은퇴하고 아픈 곳이 없지만 돈이 없어.                                                                 | 불안    |
|  7 | 어제 우리 아이 지능테스트를 받고 왔는데 결과가 좋아. 너무 신이 나.                                   |	기쁨   |
| .. | ... | ...| 

## Model
### basis BERT
+ https://huggingface.co/klue
+ CLS Token
    + Using [CLS] Token
    + Model 1 : 
      + Basic KoBERT
      + Basic KoRoBERTa
    + Model 3 :
      + KoBERT, KoRoBERTa
      + Apply Contrastive Loss
    + Model 5 :
      + KoBERT, KoRoBERTa
      + Apply Triplet Loss
+ Mean Pooling
    + Using Mean Pooling Strategy
    + Model 2
      + Basic KoBERT, KoRoBERTa
    + Model 4
      + KoBERT, KoRoBERTa
      + Apply Contrastive Loss
    + Model 6
      + KoBERT, KoRoBERTa
      + Apply Triplet Loss

## Result

### Training History
학습 그래프
#### BERT
![Model 1_6(BERT) Train History](https://user-images.githubusercontent.com/28241676/170709315-7450abbf-bab7-4e00-a537-2827b6da3b19.png)
#### RoBERTa
![Model 1_6(RoBERTa) Train History](https://user-images.githubusercontent.com/28241676/170709318-494e1348-1125-4720-9344-cd22d8003a1b.png)

### Performance Table(Test)
![Performance Table](https://user-images.githubusercontent.com/28241676/170706907-a08b6b9b-9203-4be6-a109-90887db9b088.png)

### Embedding Clustering
+ 과연, Metric Learning 방법론은 강력할까?
+ (T-SNE를 통해 256 -> 2로 차원축소함)
  
![Model's Embedding Clustering](https://user-images.githubusercontent.com/28241676/170704149-e5661acb-e9da-45ff-b202-e7a367758ade.png)

## Conclusion
Contrsative와 Triplet의 규제 비율을 조금 더 높게(학습 시 0.1로 설정함) 설정했다면 어떻게 됬을 지 궁금함


### Training
+ 전체적으로 Loss는 오르는 방향
+ Training 데이터에 대해서는 Acc가 상승
+ Valid, Test 데이터에 대해서는 Acc 상승 X
    + 학습 데이터가 조금 더 많았으면 하는 아쉬움이 남음

### Performance
+ 위 그림에서 확인
+ 전체적으로 BERT의 성능이 더 좋다. RoBERTa의 성능이 조금 아쉽다.
    + 어째서 이론적으로 더 개선된 RoBERTa가 성능이 안좋은지는 의문 

### Clustering
+ Pooling Strategy는 Mean Pooling보다 CLS Token이 더 좋음(정확도 면에서)
+ Contrastive, Triplet Loss의 성능을 확인
    + Clustering 부분에서는 확실히 성능향상이 시각적으로 확인됨
    + Loss, Acc는 CLS Token은 성능 차이가 보임
    + 그러나 Mean Pooling에서는 크게 들어나지 않음
+ 기쁨의 감정은 거의 정확하게 분리된 공간에 위치하도록 학습됨
+ 슬픔과 불안의 감정을 헷갈려 하는 경향이 있음
+ BERT(CLS)의 Contrastive Loss(Model_3_BERT)는 거의 3개의 공간으로 분리해 낸것을 확인할 수 있음
+ RoBERTa(Mean)의 Triplet Loss(Model_6_RoBerta)는 거의 3개의 공간으로 분리해 낸것을 확인할 수 있음

## Issues
+ 2022.03.26
  + 뭣 때문인지(Dataset 설정으로 예상) 몰라도 정확도와 Loss가 떨어지지 않았음
  + Dataset을 다시 정의하고, learning.py에서 학습 데이터를 불러오는 방식을 다르게 바꾸니까 해결됨
