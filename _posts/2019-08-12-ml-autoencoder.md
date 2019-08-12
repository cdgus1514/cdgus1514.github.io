---
title: "Machine Learning Unsupervised Learning"
excerrpt: "머신러닝 공부3"

categories:
  - 딥러닝
tags:
  - 딥러닝
  - 머신러닝
last_modified_at: 2019-08-12T08:06:00-05:00
---
<br>

# 머신러닝 비지도학습 오토인코더
- 데이터가 어떻게 구성되었는지를 알아내는 문제 범주
- 입력값에 대한 목표치가 주어지지 않는다
- Y값 없이 X값을 찾는 방식
<br>
<br>

## **비지도 학습 방식**
#### 1. 군집(통계)
#### 2. 차원축소(시각화)
##### 2-1. 주성분 분석(PCA)
##### 2-2. autoencode
<br>
<br>

## __차원축소 - 주성분 분석(PCA)__
###### 특성에 대한 중요도 반영
<br>
### __[Usage]__ <br>
```python
tree = RandomForestClassifier(max_depth=7, random_state=0)
tree.fit(x_train, y_train)
print("훈련 세트 정확도 >> {:.3f}".format(tree.score(x_train, y_train)))
print("테스트 세트 정확도 >> {:.3f}".format(tree.score(x_test, y_test)))

# 데이터 컬럼 별 중요도 출력
print("특성 중요도\n", tree.feature_importances_)


# 중요도 시각화
import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("특성 중요도")
    plt.ylabel("특성")
    plt.ylim(-1, n_features)


plot_feature_importances_cancer(tree)
plt.show()
```
<br>
![]({{ site.url }}{{ site.baseurl }}/assets/images/ml_importance_features.png)
<br>
<br>

## __autoencode__
###### 데이터의 입력값만 주어진 상태로 학습하는 방법
###### autoencode (encoder > Representatin > decoder)
![]({{ site.url }}{{ site.baseurl }}/assets/images/ml_autoencoder.png)
<br>
### __[Usage]__
```python
## 모델구성
from keras.layers import Input, Dense, Dropout
from keras.models import Model

# 인코딩 데이터 크기 설정
encoding_dim = 32
drop = 0.5

# 입력 플레이스홀더
input_img = Input(shape=(784,))             # Input 784

# 인코딩된 표현 (히든레이어 input=32)
encoded = Dense(encoding_dim, activation="relu")(input_img)
hidden = Dense(128, activation="relu")(encoded)
Dropout(drop)(hidden)
hidden = Dense(128, activation="relu")(hidden)
hidden = Dense(128, activation="relu")(hidden)
Dropout(drop)(hidden)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
hidden = Dense(64, activation="relu")(hidden)
Dropout(drop)(hidden)
hidden = Dense(32, activation="relu")(hidden)
hidden = Dense(32, activation="relu")(hidden)

# 입력의 손실이 있는 재구성
decoded = Dense(784, activation="sigmoid")(encoded)
# 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded)     # 784 >> 32 >> 784
# 인코딩된 입력의 표현으로 매핑
encoder = Model(input_img, encoded)         # 784 >> 32

# 인코딩된 입력을 입력 (히든레이어의 ~를 인풋으로 사용)
encoded_input = Input(shape=(encoding_dim,))
# 오토 인코더 모델의 마지막 레이어
decoded_layer = autoencoder.layers[-1]
# 디코더 모델 생성
decoder = Model(encoded_input, decoded_layer(encoded_input))        # Output 32 >> 784

autoencoder.summary()

autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)
```