# Day 05 Scikit-learn 线性回归

## 今天学了什么

- 机器学习基本流程
- 特征 X
- 标签 y
- 训练集
- 测试集
- 线性回归
- model.fit 训练模型
- model.predict 预测
- MAE、MSE、R2 模型评估
- 使用 Matplotlib 可视化预测结果

## 重要代码

```python
X = df[["area", "rooms", "distance", "age"]]
y = df["price"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)