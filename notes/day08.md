# Day 08 Scikit-learn 随机森林模型

## 今天学了什么
- 了解了集成学习（Ensemble Learning）的思想。
- 使用 RandomForestClassifier 训练了随机森林。
- 理解了 n_estimators 参数（树的数量）的作用。
- 提取并可视化了 Feature Importance（特征重要性）。

## 我理解的随机森林
随机森林就是建很多棵不同的决策树，让它们一起投票。它可以防止单棵决策树过拟合的问题，通常准确率更高。

## 练习观察结果
- 修改树的数量为 10 后，准确率的变化是：变小了一点
- 模型认为最重要的特征是：worst area