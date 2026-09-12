# Day 22：模型调参与保存

## 一、今天学习的内容

今天学习了：

- 模型超参数
- 网格搜索
- GridSearchCV
- 调参前后的模型比较
- 使用 joblib 保存模型

## 二、什么是超参数

超参数是在模型训练之前人为设置的参数。

随机森林中的常见超参数包括：

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

## 三、什么是网格搜索

网格搜索会自动尝试多组参数组合。

每组参数都会进行交叉验证，然后根据评分选择表现较好的参数。

本次实验使用 F1-score 作为主要评分指标。

## 四、调参前结果

请根据程序输出填写：

- Accuracy：
- Precision：
- Recall：
- F1：

## 五、调参后结果

请根据程序输出填写：

- Accuracy：
- Precision：
- Recall：
- F1：

## 六、最佳参数

请复制程序输出的最佳参数：

```text
在这里填写最佳参数