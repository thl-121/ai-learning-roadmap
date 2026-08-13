import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

###加载数据
wine = load_wine()
x = pd.DataFrame(wine.data,columns=wine.feature_names)
y = wine.target

print("特征数据维度:",x.shape)
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

###训练KNN模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

###初始化 KNN 模型(n_neighbors=5 表示找最近的5个邻居)
knn_model = KNeighborsClassifier(n_neighbors=3)

###训练模型
knn_model.fit(x_train,y_train)
print("KNN 模型准备好了!")

###预测与评估
y_pred = knn_model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("KNN 准确率 (k=5):",acc)

##可视化
import matplotlib.pyplot as plt

###解决中文问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

k_values = range(1,21)
accuracies = []

###循环测试不同 K 值
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    accuracies.append(accuracy_score(y_test,pred))

###画折线图
plt.figure(figsize=(10,6))
plt.plot(k_values,accuracies,marker='o',linestyle='dashed',color='red',markersize=8)###x 轴数据：k_values（1~20）/y 轴数据：accuracies（每个 K 对应的准确率）/marker='o'：线上每个数据点画一个圆圈/linestyle='dashed'：虚线连接各个点/color='red'：线条红色 /markersize=8：圆圈大小设为 8
plt.title("不同 K 值下的模型准确率")
plt.xlabel("K 值 (n_neighbors)")
plt.ylabel("准确率(Accuracy)")
plt.xticks(k_values)###横轴刻度强制标出 1、2、3…20，每个 K 都看得见
plt.grid(True,linestyle='--',alpha=0.6)###打开背景网格，虚线、透明度 0.6，方便看数值

plt.tight_layout()
plt.savefig("knn_k_chart.png")
plt.show()




















