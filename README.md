# Emotion Classifier Project

## Introduction
+ AI HUB에서 제공하는 한국말 대화 뭉치에서 문장에 대한 감정을 분류하는 모델 설계
  + https://aihub.or.kr/aidata/7978
+ BERT 계열의 여러 파생모델을 사용해보고 성능비교와 성능향상을 위한 Method 제시
    + KoBERT, KoElectra, RoBERTa,,, etc.
+ 

## Data
### BERT based
+ https://github.com/monologg/KoBERT-Transformer
  + ```pip3 install kobert-transformers```
+ Model 1 : 
  + Basic KoBERT
  +

### DistilKoBERT
+ https://github.com/monologg/KoBERT-Transformers
  + ```pip3 install kobert-transformers```
+ Huggingface가 자체적으로 공개한 모델
+ 크고 무거운 BERT의 문제점을 보완한 모델
  + BERT에 비해 크기가 40%가량 줄고,
  + 60%가량 연산속도가 빠르다.
  + 성능 또한 97%의 성능을 유지함
+ Model ? :
  + Basic DisitlKoBERT

## Model

## Result

## Conclustion

## 참고
https://huggingface.co/docs/transformers/master/en/model_doc/bert#bert