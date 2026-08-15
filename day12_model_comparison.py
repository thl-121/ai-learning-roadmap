import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine = load_wine()

x = pd.DataFrame(wine.data,columns=wine.feature_names)
y = wine.target

print("数据维度:",x.shape)
print("类别名称:",wine.target_names)
print("前5行数据:")
print(x.head())

###划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("训练集数量:",len(x_train))
print("测试集数量:",len(y_test))

###标准化数据
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
###训练集用 fit_transform;测试集只用 transform;测试集不能参与标准化规则的学习，否则会数据泄露
print("标准化完成")

###导入多个模型
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

###创建模型列表
models = {
    "逻辑回归":LogisticRegression(max_iter=1000),
    "决策树":DecisionTreeClassifier(random_state=42),
    "随机森林":RandomForestClassifier(n_estimators=100,random_state=42),
    "KNN":KNeighborsClassifier(n_neighbors=5)
}

###循环训练和评估模型
model_names = []
accuracies = []

for name,model in models.items():
    print("==========", name, "==========")

    model.fit(x_train_scaled,y_train)

    y_pred = model.predict(x_test_scaled)

    acc  = accuracy_score(y_test,y_pred)

    print("准确率:",acc)
    print("分类报告:")
    print(classification_report(y_test,y_pred,target_names=wine.target_names))

    model_names.append(name)
    accuracies.append(acc)

###画模型准确率对比图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8,5))

plt.bar(model_names,accuracies)

plt.title("不同分类模型准确率对比")
plt.xlabel("模型")
plt.ylabel("准确率")

plt.ylim(0,1.1)###限制Y轴的显示范围：从0到1.1

for i,acc in enumerate(accuracies):
    plt.text(i,acc+0.02,f"{acc:.2f}",ha="center")

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

###找出表现最好的模型
best_index = accuracies.index(max(accuracies))
best_model_name = model_names[best_index]
best_accuracy =  accuracies[best_index]

print("表现最好的模型:",best_model_name)
print("最高准确率:",best_accuracy)

###单独测试随机森林的重要特征
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(x_train,y_train)

feature_importance =pd.DataFrame({
    "特征":x.columns,###`x.columns`就是 wine 数据集 13 个特征名字
    "重要性":rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="重要性",
    ascending=False###降序排序，从大到小
)

print("随机森林特征重要性:")
print(feature_importance)

###画特征重要性图
plt.figure(figsize=(10,6))

plt.barh(feature_importance["特征"],feature_importance["重要性"])

plt.title("随机森林特征重要性")
plt.xlabel("重要性")
plt.ylabel("特征")

plt.gca().invert_yaxis()###最重要的特征显示在图表最顶端，越往下越不重要

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()













