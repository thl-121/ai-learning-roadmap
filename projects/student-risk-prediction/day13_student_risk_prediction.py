import pandas as pd


df = pd.read_csv("student_scores.csv")

print("学生成绩数据:")
print(df)

print("数据基本消息:")
print(df.info())

print("统计信息:")
print(df.describe())

df["total"] = df["math"] + df["english"] + df["python"]
df["average"] = df["total"] / 3

print("添加总分和平均分后的数据:")
print(df.head())

x = df[["math","english","python","study_hours","absences","total","average"]]
y = df["fail"]

print("特征 x:")
print(x.head())

print("标签 y:")
print(y.head())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("训练集数量:",len(x_train))
print("测试集数量:",len(x_test))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(x_train,y_train)

print("模型训练完了")

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

print("模型准确率:",accuracy)
print("分类报告:")
print(classification_report(y_test,y_pred))

print("混淆矩阵:")
print(confusion_matrix(y_test,y_pred))

result = x_test.copy()
result["真实是否挂科"] = y_test.values
result["预测是否挂科"] = y_pred

print("预测结果:")
print(result)

result.to_csv("prediction_result.csv",index=False)

print("预测结果已保存到 prediction_result.csv")

new_student = pd.DataFrame({
    "math":[59],
    "english":[60],
    "python":[55],
    "study_hours":[2],
    "absences":[6],
    "total":[58+60+55],
    "average":[(58+60+55)/3]
})

new_prediction = model.predict(new_student)
new_probability =model.predict_proba(new_student)

print("新学生数据:")
print(new_student)

print("预测是否挂科:",new_prediction[0])
print("挂科概率:",new_probability[0][1])

feature_importance = pd.DataFrame({
    "feature":x.columns,
    "importance":model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("特征重要性:")
print(feature_importance)

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] =["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(10,6))

plt.barh(feature_importance["feature"],feature_importance["importance"])

plt.title("挂科风险预测特征重要性")
plt.xlabel("重要性")
plt.ylabel("特征")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()



