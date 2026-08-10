from sklearn.datasets import load_iris###这是为了从工具包里把 “鸢尾花数据集加载工具” 借来，后面要用它调取数据
import pandas as pd

from day05_linear_regression import predicted_price

iris = load_iris()

print("特征名称:")
print(iris.feature_names)

print("类别名称:")
print(iris.target_names)

print("数据形状:")
print(iris.data.shape)###代表150 条样本，每条样本 4 个特征，每条样本对应 1 个类别标签
###iris.data：所有花朵的 4 项测量数值，纯数字表格，也就是特征数据 .shape：查看表格行列格式，格式为(行数,列数)
print("标签形状:")
print(iris.target.shape)###iris.target：每一朵花对应的品种编号（0/1/2），也就是标签

###把鸢尾花数据转成 DataFrame
df = pd.DataFrame(iris.data,columns=iris.feature_names)###iris.data表示150行4列的花朵4项尺寸原始数字
###columns=iris.feature_names：给表格的 4 列起列名，分别是花萼长、花萼宽、花瓣长、花瓣宽。
df["target"] = iris.target###在现有表格pd.DataFrame里面新增一列，列名叫target，赋值为iris.target中的 150 个品种数字（0、1、2）依次放进这一列，0、1、2 分别代表三种鸢尾花
df["flower_name"] = df["target"].apply(lambda x:iris.target_names[x])###在现有表格pd.DataFrame里面新增一列，列名叫flower_name
###df["target"].apply(lambda x:iris.target_names[x])表示遍历刚刚 target 列里所有 0、1、2 数字，自动把数字翻译成对应的花朵英文名字，生成新的一列真实花名
print("鸢尾花数据表:")
print(df.head())

print("每个类别数量:")
print(df["flower_name"].value_counts())###.value_counts表示统计该列每个值出现的次数

###准备特征 X 和标签 y
x = df[iris.feature_names]
y = df["target"]

print("特征 x 前5行:")
print(x.head())

print("标签 y 前5个:")
print(y.head())

###划分训练集和测试集
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("训练集数量:",len(x_train))
print("测试集数量:",len(x_test))
print("训练集数量:",len(y_train))
print("测试集数量:",len(y_test))

###训练逻辑回归模型(分类算法)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)###表示最多训练 200 次，避免模型没训练完就停止

model.fit(x_train,y_train)

print("模型训练完成")

###进行预测
y_pred = model.predict(x_test)

result  = pd.DataFrame({
    "真实类别编号":y_test.values,
    "预测类别编号":y_pred
})

result["真实花名"] = result["真实类别编号"].apply(lambda x: iris.target_names[x])
result["预测花名"] = result["预测类别编号"].apply(lambda x: iris.target_names[x])

print("预测结果:")
print(result)

###计算准确率
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)

print("模型准确率:",accuracy)
print("模型准确率百分比:",accuracy*100,"%")

###输出分类报告
from sklearn.metrics  import classification_report

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

print("分类报告:")
print(report)

###混淆矩阵
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print("混淆矩阵:")###混淆矩阵是一张表格，用来统计分类模型每一类的对错情况，行是真实标签，列是预测标签，对角线 = 猜对，其余 = 猜错
print(cm)

###画混淆矩阵图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

plt.figure(figsize=(6,5))

plt.imshow(cm,cmap="Blues")###imshow()：专门把二维数字矩阵渲染成色块图，数值越大，颜色越深。

plt.title("鸢尾花分类混淆矩阵")
plt.xlabel("预测类别")
plt.ylabel("真实类别")

plt.xticks([0,1,2],iris.target_names)###xticks：X 轴刻度，原本刻度是 0、1、2，替换成花的英文名 setosa、versicolor、virginica；
plt.yticks([0,1,2],iris.target_names)

for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(i,j,cm[i][j],ha="center",va="center",color="black")
###这两个for循环是为了双重循环，在格子里标注数字 plt.text(x坐标, y坐标, 要写的内容)：在指定坐标写入文字，ha="center"：水平居中；va="center"：垂直居中，数字刚好落在格子正中间，color="black"：字体黑色
plt.colorbar()###在图表右边生成一条颜色标尺，用来对照：颜色深浅对应数值大小。
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png")
plt.show()

###预测一朵新花
new_flower = pd.DataFrame({
    "sepal length (cm)":[5.1],
    "sepal width (cm)":[3.5],
    "petal length (cm)":[1.4],
    "petal width (cm)":[0.2]
})

new_prediction = model.predict(new_flower)
new_prediction_name = iris.target_names[new_prediction[0]]

print("新花数据:")
print(new_flower)

print("预测类别编号:",new_prediction[0])
print("预测花名:",new_prediction_name)

###查看每个类别的预测概率
new_probability = model.predict_proba(new_flower)###predict_proba() 是分类模型的预测概率函数，和只输出 0/1/2 标签的predict()不一样

probability_df = pd.DataFrame(
    new_probability,
    columns=iris.target_names
)
###columns=iris.target_names：给表格三列命名，依次为 setosa、versicolor、virginica
print("新花属于每个类别的概率:")
print(probability_df)

###画特征散点图
plt.figure(figsize=(8,6))

for target_id,flower_name in enumerate(iris.target_names):###enumerate(列表) 遍历的时候，同时拿到：下标序号 + 对应元素 enumerate(iris.target_names)依次拿到：编号 0-setosa、编号 1-versicolor、编号 2-virginica
    flower_data = df[df["target"] == target_id]###df[df["target"] == target_id] 表格筛选：只取出当前这一类花的所有数据。 举例：第一次循环 target_id=0，只筛选出所有山鸢尾的数据。
    plt.scatter(
        flower_data["petal length (cm)"],###X 轴：花瓣长度
        flower_data["petal width (cm)"],###Y 轴：花瓣宽度
        label=flower_name###label=flower_name：给这一类点打上标签，后续图例用来区分颜色。效果：3 种花会自动用 3 种不同颜色画出圆点
    )

plt.title("鸢尾花花瓣长度与宽度分布")
plt.xlabel("花瓣长度")
plt.ylabel("花瓣宽度")
plt.legend()###plt.legend ()：图例，在图表角落显示「颜色对应哪一种花」，看图一目了然

plt.savefig("iris_feature_scatter.png")
plt.show()

result.to_csv("iris_prediction_result.csv", index=False)

print("预测结果已保存到 iris_prediction_result.csv")




