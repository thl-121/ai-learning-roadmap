# 学生成绩分析与挂科风险预测系统

## 项目简介

本项目使用学生成绩、学习时间和缺勤次数等数据，训练机器学习模型预测学生是否存在挂科风险。

## 使用技术

- Python
- Pandas
- Scikit-learn
- Matplotlib
- RandomForestClassifier

## 项目流程

1. 读取学生成绩数据
2. 计算总分和平均分
3. 构建特征和标签
4. 划分训练集和测试集
5. 训练随机森林分类模型
6. 评估模型准确率
7. 预测新学生挂科风险
8. 分析特征重要性

## 特征说明

- math：数学成绩
- english：英语成绩
- python：Python 成绩
- study_hours：每天学习小时数
- absences：缺勤次数
- total：总分
- average：平均分

## 标签说明

- fail = 0：未挂科
- fail = 1：挂科

## 项目收获

通过本项目，我练习了一个完整机器学习分类项目的基本流程。