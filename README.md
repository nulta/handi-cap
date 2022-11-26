# handi-cap

```
handi-cap
  + learn.py
  + webcam.py
  + webcam_do.py
  + 1
    + (PNG data for gesture type 1)
  + 2
    + (PNG data for gesture type 2)
  + 3
    + (PNG data for gesture type 3)
```

## learn.py
모델을 학습시킬 때 사용합니다. Early-stopping이 구현되어 있으며, validation set을 자동으로 나누어 각 epoch마다 validation accuracy를 출력해줍니다. 학습시킬 PNG 파일들이 필요합니다.

## webcam.py
학습용 데이터를 수집할 때 사용합니다. [c]를 눌러 사진을 캡처합니다.

## webcam_do.py
모델을 이용한 HCI 인터페이스의 구현입니다. 3가지 동작을 인식합니다. 학습된 모델이 필요합니다.
