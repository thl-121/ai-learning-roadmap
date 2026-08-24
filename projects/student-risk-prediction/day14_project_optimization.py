import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv("student_scores.csv")

df["total"] =  df["math"]+df["english"]+df["python"]
df["average"] = df["total"] / 3

print("学生数据:")
print(df.head())

###画成绩分布图
plt.figure(figsize=(8, 5))

plt.hist(df["average"],bins=8,color="skyblue",edgecolor="black")###画直方图（频次分布图）

plt.title("学生平均分分布")
plt.xlabel("平均分")
plt.ylabel("人数")

plt.tight_layout()
plt.savefig("score_distribution.png")
plt.show()

###画缺勤次数和挂科关系
plt.figure(figsize=(8,5))

plt.scatter(df["absences"],df["average"],c=df["fail"],cmap="coolwarm",s=80)

plt.title("缺勤次数与平均分关系")
plt.xlabel("缺勤次数")
plt.ylabel("平均分")
plt.colorbar(label="是否挂科")

plt.tight_layout()
plt.savefig("absence_average_scatter.png")
plt.show()

###准备特征和标签
x = df[["math","english","python","study_hours","absences","total","average"]]
y = df["fail"]

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)

###多模型对比
models = {
    "逻辑回归":LogisticRegression(max_iter=1000),
    "决策树":DecisionTreeClassifier(random_state=42),
    "随机森林":RandomForestClassifier(n_estimators=100,random_state=42),
    "KNN":KNeighborsClassifier(n_neighbors=3)
}

model_names =[]
accuracies = []
for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test,y_pred)

    print("==========", name, "==========")
    print("准确率:",acc)
    print(classification_report(y_test,y_pred))

    model_names.append(name)
    accuracies.append(acc)

###画模型对比图
plt.figure(figsize=(8,5))

plt.bar(model_names,accuracies,color=["orange", "green", "blue", "purple"])

plt.title("不同模型挂科预测准确率对比")
plt.xlabel("模型")
plt.ylabel("准确率")

plt.ylim(0,1.1)

for  i,acc in enumerate(accuracies):
    plt.text(i,acc + 0.02,f"{acc:.2f}",ha="center")

plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

###选择随机森林做最终模型
final_model = RandomForestClassifier(n_estimators=100,random_state=42)
final_model.fit(x_train,y_train)

fail_probability = final_model.predict_proba(x)[:,1]
fail_prediction = final_model.predict(x)

df["fail_probability"] = fail_probability
df["predicted_fail"] = fail_prediction

print("加入预测结果后的数据:")
print(df[["name", "average", "absences", "fail_probability", "predicted_fail"]])

###生成风险等级
def get_risk_level(probability):
    if probability >= 0.7:
        return "高风险"
    elif probability >= 0.4:
        return "中风险"
    else:
        return "低风险"

df["risk_level"] = df["fail_probability"].apply(get_risk_level)
print("学生风险等级:")
print(df[["name", "average", "absences", "fail_probability", "risk_level"]])
df.to_csv("risk_prediction_result.csv", index=False)

print("最终预测结果已保存到 risk_prediction_result.csv")