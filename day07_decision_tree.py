from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

from day06_classification import accuracy

###加载数据
iris = load_iris()
x = pd.DataFrame(iris.data,columns=iris.feature_names)
y = iris.target

###划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
print("数据加载与划分完成!")

###训练决策树模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

###初始化决策树模型（限制树的最大深度为 3）
tree_model  = DecisionTreeClassifier(max_depth=3,random_state=42)

###训练模型
tree_model.fit(x_train,y_train)
print("决策树模型训练完成！")

###预测与评估
y_pred = tree_model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("决策树模型准确率 (Accuracy):",acc)

###画出决策树
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axse.unicode_minus"] = False

###可视化决策树
plt.figure(figsize=(12,8))
plot_tree(
    tree_model,###训练好的决策树模型对象
    feature_names=iris.feature_names,###特征名字：花萼长、花萼宽、花瓣长、花瓣宽
    class_names=iris.target_names,###类别名字：setosa,versicolor,virginica三种鸢尾花
    filled=True,###给每个节点填充不同颜色，不同类别颜色不一样
    rounded=True###节点方框边角变圆润，好看一点
)

plt.title("鸢尾花分类决策树")
plt.tight_layout()
plt.savefig("iris_tree.png")
plt.show()
















