# YOLO_NAS 커스텀 데이터셋 활용
## 목표
> Yolo-Nas를 이용한 감자와 고구마 식별 모델
## 1차 시도
### 과정
> 우선 커스텀 데이터 셋을 만들기 위해 roboflow 사이트를 사용하였다.
> 감자와 고구마의 사진을 각 30장씩 준비하였고 라벨링을 진행하였다.
> 이후 라벨링된 사진들로 커스텀 데이터셋을 생성하였다.
> colab을 통해 batch 10, epoch 15, adam 알고리즘을 사용한 S 모델을 학습시켰다.
### 겪은 문제
> 그 결과 감자는 정상적으로 인식하였지만 고구마는 인식하지 못하는 문제가 발생하였다.
> 해당 결과는 데이터 수의 문제라고 판단하였으나 감자와 고구마의 사진의 경우 대부분 겹쳐있거나 쌓여 있어 라벨링을 하기 어려워 사진을 구하는 데에 어려움이 있었다.
## 2차 시도
### 과정
> 때문에 데이터셋을 만들 때에 시계 방향과 반시계 방향으로 각 90도씩 회전한 사진을 포함하여 데이터의 수를 3배로 하였다.
> 또한 epoch의 수를 70으로 변경하고 L 모델로 학습하였다.
### 겪은 문제
> 학습에 10분 정도 걸렸던 S 모델과 달리 L 모델은 1시간 50분 가량 소모되었다.
> 그리고 고구마를 아예 인식하지 못한 이전 모델과는 다르게 고구마를 인식하였지만 감자 또한 고구마로 잘못 인식하는 경우가 발생하였다.
> 또한 colab에서 제공하는 GPU 사용량을 모두 소모하여 더 이상의 사용이 제한되었다.
## 3차 시도
### 과정
> 다른 구글 계정을 통해 colab을 이용하였고 GPU 사용량과 학습 시간을 줄이기 위해 epoch를 50으로 줄이고 num_worker를 2에서 5로 증가시켰다.
> 그 결과 모델의 학습 시간은 1시간 20분 가량으로 줄어들었다.
> 인식의 경우 잘못 인식하는 경우는 확실히 줄었다.
### 겪은 문제
> 잘못된 인식의 발생은 줄었지만 같은 종류의 작물이 겹쳐있는 경우 하나로 인식하는 경우가 발생하였다.
![스크린샷 2024-03-30 234842](https://github.com/woowal/YOLO_NAS_Custom_Data/assets/61446702/edb892f0-7c10-4ed1-b48e-1052d364cee2)
### 모델 pth 파일
> https://drive.google.com/file/d/174Cz_tF1uUbirboQFRPmwsfzVQys4_hG/view?usp=sharing
#Evaluate
### PR Curve
![myplot](https://github.com/woowal/YOLO_NAS_Custom_Data/assets/61446702/ddd7088c-a6a3-4a47-b02e-de77dc01beb3)
> 과정: https://velog.io/@jijilee4/Yolo-Nas-%EB%AA%A8%EB%8D%B8%EC%9D%98-PR-Curve
