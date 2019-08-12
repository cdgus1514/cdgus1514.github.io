---
title: "Machine Learning HyperParameter"
excerrpt: "머신러닝 공부3"

categories:
  - 딥러닝
tags:
  - 딥러닝
  - 머신러닝
last_modified_at: 2019-08-07T08:06:00-05:00
---
<br>

# 머신러닝 비지도학습 오토인코더
-  Y값 없이 X값을 찾는 방식
<br>
<br>

## ** 비지도 학습 방식 **
#### 1. 군집(통계)
#### 2. 차원축소(시각화)
#####  2-1. 주성분 분석(PCA)
#####  2-2. autoencode

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
![]({{ site.url }}{{ site.baseurl }}/assets/images/ml_importance_features.png)
