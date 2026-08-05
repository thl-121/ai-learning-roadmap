# Day 06 分类模型

## 今天学了什么

- 分类问题
- 鸢尾花数据集
- 逻辑回归 Logistic Regression
- 准确率 Accuracy
- 混淆矩阵 Confusion Matrix
- 分类报告 Classification Report
- predict 预测类别
- predict_proba 预测概率
- 使用散点图观察不同类别的数据分布

## 回归和分类的区别

回归用于预测连续数值，比如房价、成绩、温度。

分类用于预测类别，比如花的种类、邮件是否垃圾、图片是猫还是狗。

## 重要代码

```python
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)