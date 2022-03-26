# Emotion Classifier Project

## Introduction
+ AI HUB에서 제공하는 한국말 대화 뭉치에서 문장에 대한 감정을 분류하는 모델 설계
  + https://aihub.or.kr/aidata/7978
+ BERT 계열의 여러 파생모델을 사용해보고 성능비교와 성능향상을 위한 Method 제시
    + KoBERT, KoElectra, RoBERTa,,, etc.
+ 

## Data
### Train
|    | sentence                                                                          | label   |
|---:|:----------------------------------------------------------------------------------:|:--------|
|  0 | 오늘은 누구랑 같이 밥을 먹어야 할까?                                              | 불안    |
|  1 | 난 남들 다 받는 과외를 못 받아서 공부를 못하는 것 같아.                           | 슬픔    |
|  2 | 오늘 친구가 몸이 너무 안 좋아서 친구 잘못으로 내가 대신 혼났어.                   | 상처    |
|  3 | 우리 주위에는 불치병으로 인하여 어려운 분들이 많은 것 같아.                       | 상처    |
|  4 | 아이들이 사춘기가 되니 예전처럼 나에게 먼저 말을 걸지 않고 자꾸 나를 피하려고 해. | 당황    |
|  5 | 딸이 결혼을 안 하려고 해서 집사람이 화가 났어.                                    | 분노    |
|  6 | 학원 친구 중에 나보다 뛰어난 애가 눈에 밟혀서 화가 나.                            | 상처    |
| .. | ... | ...| 

### TEST
|    | sentence                                                                                             | label   |
|---:|:----------------------------------------------------------------------------------------------------:|:--------|
|  0 | 수험생인 아들과 기분 전환 겸 집 앞 산을 등산하다가 아들이 발목을 접질려서 너무 마음 아파.            | 당황    |
|  1 | 내 장례식만큼은 내가 원하는 대로 해 달라고 말했는데 자식들은 들은 체도 안 해. 속상해.                | 상처    |
|  2 | 친구가 지난해에 이혼을 했어. 양육비를 월 오십만 원씩 주기로 하고 아이는 부인이 키우기로 했다는 거야. | 당황    |
|  3 | 학교에 늦어서 엄마 출근길에 나 좀 데려다 달라고 했는데 엄마가 화가 났을까?                           | 분노    |
|  4 | 가족들이 집으로 빨리 돌아오라고 했는데 늦게 가서 걱정만 시켰어.                                      | 슬픔    |
|  5 | 건강에 회의적이야.                                                                                   | 불안    |
|  6 | 은퇴하고 아픈 곳이 없지만 돈이 없어.                                                                 | 당황    |
| .. | ... | ...| 

## Model
### BERT based
+ https://huggingface.co/kykim/bert-kor-base
+ Model 1 : 
  + Basic KoBERT
  +

### Ko-ALBERT
+ https://huggingface.co/kykim/albert-kor-base
+ 크고 무거운 BERT의 문제점을 보완한 모델
+ Model ? :
  + Basic Ko-ALBERT
  
### KoElectra
+ https://huggingface.co/kykim/electra-kor-base
+ 설명 추가

### 그 밖에 추가할거
+ https://github.com/KLUE-benchmark/KLUE

## Result

## Conclustion

## Issues
+ 2022.03.26
  + 뭣 때문인지(Dataset 설정으로 예상) 몰라도 정확도와 Loss가 떨어지지 않았음
  + Dataset을 다시 정의하고, learning.py에서 학습 데이터를 불러오는 방식을 다르게 바꾸니까 해결됨

## 참고
https://huggingface.co/docs/transformers/master/en/model_doc/bert#bert