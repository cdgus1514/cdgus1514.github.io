---
title: "Machine Learning HyperParameter"
excerrpt: "머신러닝 공부2"

categories:
  - 딥러닝
tags:
  - 딥러닝
  - 머신러닝
last_modified_at: 2019-08-07T08:06:00-05:00
---

<br>

# 머신러닝 하이퍼파라미터 튜닝

- all_estimators
- GridSearchCV
- RandomizedSearchCV
  <br>
  <br>

## **all_estimators** <br>

#### 각각의 카테고리별 모델들을 불러와 데이터를 학습하고 최적의 결과를 낼 수 있는 모델을 찾을 수있다.

#### 1. 분류모델

#### 2. 회귀모델

#### 3. 클러스터링 모델

#### 4. 차원축소 모델

<br>
![]({{ site.url }}{{ site.baseurl }}/assets/images/ml_sklearn_algorithm_map.png)
###### *출처 : scikit-learn.org*
<br>
### __[Usage]__ <br>
```python
from sklearn.utils.testing import all_estimators

## classifier 알고리즘 모두 추출

allAlgorithms = all_estimators(type_filter="classifier")

for(name, algorithm) in allAlgorithms: #각 알고리즘 객체 생성
clf = algorithm()

    #학습 후 평가
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(name,"의 정답률 >> ", accuracy_score(y_test, y_pred))

````
<br>
<br>

## __GridSearchCV__
###### 생성한 모델과 파라미터를 **순차적** 으로 적용하면서 해당 모델에 최적의 매개변수를 찾아주는 기능
<br>
### __[Usage]__ <br>
```python
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# GridSearchCV 매개변수 >> list, dic 모두 사용 가능
parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"]},
    {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]

## 서치
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = GridSearchCV(SVC(), parameters, cv=kfold_cv)
clf.fit(x_train, y_train)
print("최적의 매개변수 >> ", clf.best_estimator_)
````

<br>
<br>

## **RandomizedSearchCV**

###### 생성한 모델과 파라미터를 **램덤** 으로 적용하면서 해당 모델에 최적의 매개변수를 찾아주는 기능

<br>
### __[Usage]__ <br>
```python
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV

# RandomizedSearchCV 매개변수 >> list형식만 가능

parameters = {
"C":[1,10,100,1000],
"kernel":["linear", "rbf", "sigmoid"],
"gamma":[0.001, 0.0001],
}

## 서치

kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(SVC(), parameters, cv=kfold_cv)

# clf = GridSearchCV(SVC(), parameters, cv=kfold_cv)

clf.fit(x*train, y_train)
print("최적의 매개변수 >> ", clf.best_estimator*)

```

```
