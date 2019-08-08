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
<br>

__#머신러닝 하이퍼파라미터 튜닝__
=====
- all_estimators
- GridSearchCV
- RandomizedSearchCV

<br>
<br>
<br>

**all_estimators**  
각각의 카테고리별 모델들을 불러와 데이터를 학습하고 최적의 결과를 낼 수 있는 모델을 찾을 수있다.
1. 분류모델
2. 회귀모델
3. 클러스터링 모델
4. 차원축소 모델

<br>
<br>
<br>
<br>

__##[Usage]__
-----

**from** sklearn.utils.testing **import** all_estimators  
allAlgorithms = all_estimators(type_filter="모델타입")

__from__ sklearn.metrics __import__ accuracy_score  
for (name, algorithm) in allAlgorithms:

  clf = algorithm()   *# 각각의 알고리듬 객체 생성*

  *##학습 후 평가*  
  clf.fit(x_train, y_train)
  y_pred = clf.predict(x_test)
  print(name,"알고리듬 정답률 >> ", accuracy_score(y_test, y_pred))
