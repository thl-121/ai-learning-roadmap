import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sympy.logic.inference import pl_true

###加载数据，为了方便画图，我们只取前2个特征
iris = load_iris()
x = pd.DataFrame(iris.data[:,:2], columns=iris.feature_names[:2])###iris.data[:,:2]切片语法：[行 , 列]第一个冒号 : ：所有行（全部 150 朵花全部保留） :2 ：只取前 2 列
y = iris.target

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
print("训练集与测试集划分完成!")

###训练SVM模型
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

###初始化 SVM 模型(kernel="linear" 表示我们用一条直线/平面划分)
svm_model = SVC(kernel="rbf",random_state=42)

###训练模型
svm_model.fit(x_train,y_train)
print("SVM 模型训练完成!")

###预测与评估
y_pred = svm_model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("SVM 准确率 (Accuracy):",acc)

###把SVM的"边界线"画出来
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

###解决中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(10,6))

###绘制决策边界背景
DecisionBoundaryDisplay.from_estimator(
    svm_model,
    x_train,
    response_method="predict",###调用模型的 predict 函数做类别预测，输出类别 0/1/2，用来区分不同颜色区域
    cmap=plt.cm.coolwarm,###配色方案，冷色到暖色渐变，不同类别对应不同颜色
    alpha=0.6,###透明度，0 完全透明，1 完全不透明。0.6 就是半透明，后面画真实样本点的时候，点不会被背景色块盖住
    ax=plt.gca()###拿到我们刚刚创建的画布，把边界画到这张图上，而不是新开一张图片
)

###把训练集的真实数据点也画上去
plt.scatter(x_train.iloc[:,0],x_train.iloc[:,1],c=y_train,cmap=plt.cm.coolwarm,edgecolors="k")###背景是 SVM 模型划分出来的颜色区域，上面叠着真实花朵样本点。如果某个点落在和自己颜色不一样的背景色块，代表这个样本被 SVM 分错了

plt.title("SVM 鸢尾花分类边界 (仅使用前2个特征)")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.tight_layout()
plt.savefig("svm_boundary.png")
plt.show()

print("支持向量的数量:", len(svm_model.support_vectors_))






