import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


###加载数据
cancer = load_breast_cancer()
x = pd.DataFrame(cancer.data,columns=cancer.feature_names)
y = cancer.target

print("特征数据维度:",x.shape)###x.shape pandas表格属性，输出格式(样本数量,特征数量)
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
print("测试集数量:",len(x_test))
print("训练集数量:",len(y_train))
print("测试集数量:",len(y_test))

###训练随机森林模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

###初始化随机森林模型 (n_estimators=100 表示种 100 棵树)
rf_model = RandomForestClassifier(n_estimators=100,random_state=42)###随机森林靠「随机 + 多棵树投票」天然抗过拟合，不像单棵决策树那么必须强制限制深度

###训练模型
rf_model.fit(x_train,y_train)
print("随机森林模型训练完成!")

###预测与评估
y_pred = rf_model.predict(x_test)
acc = accuracy_score(y_test,y_pred)
print("随机森林准确率 (Accuracy):",acc)

###把“特征重要性”画出来
import matplotlib.pyplot as plt

###解决中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

###获取特征重要性
importances = rf_model.feature_importances_###随机森林的王牌功能！随机森林训练完之后，会自动计算每一个特征的**重要性分数**

###把特征和重要性组合起来，按重要性从高到低排序
feature_imp = pd.DataFrame({
    "Feature":cancer.feature_names,
    "Importance":importances
}).sort_values(by="Importance",ascending=False)###按照重要性这一列从高到低排序

###取前 10 个最重要的特征图画
top_10_features = feature_imp.head(10)

plt.figure(figsize=(10,6))
plt.barh(top_10_features["Feature"],top_10_features["Importance"])###barh：画横向条形图 Y 轴：特征名称（各个肿瘤检测指标）X 轴：重要性分数，条形越长代表这个特征越重要(因为是横向条形图所以x y 的位置调换了)
plt.gca().invert_yaxis()###gca()拿到当前坐标轴；invert_yaxis()` Y 轴上下颠倒。默认 barh 画出来，第一名最重要的会跑到图表最下面；颠倒之后，**最重要的特征放在图表最上方**，符合阅读习惯
plt.title("乳腺癌预测 - Top 10 重要特征")
plt.xlabel("重要性分数")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()









