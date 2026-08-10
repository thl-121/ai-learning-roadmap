import pandas as pd

df = pd.read_csv("house_price.csv")

print("房价数据表:")
print(df)

print("前5行数据:")
print(df.head())

print("数据基本信息:")
print(df.info())

print("统计信息:")
print(df.describe())###df.describe()表示 数值列统计描述只自动筛选数字类型的列（文本列不会显示），计算出 8 个常用统计值；

###准备特征X和标签Y
x=df[["area","rooms","distance","age"]]
y=df["price"]

print("特征 x:")
print(x.head())

print("标签 y:")
print(y.head())

###划分训练集和测试集
from sklearn.model_selection import train_test_split###model_selection：sklearn 里负责数据划分、交叉验证的模块 train_test_split：直译：训练 - 测试 - 分割，专门把一整份数据集拆成两份的工具函数;

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.2,###测试集中的数据占全部数据的20%
    random_state=42###这是为了分割前固定随机打乱的数据顺序
)

print("训练集特征数量:",len(x_train))
print("测试集特征数量:",len(x_test))
print("训练集特征数量:",len(y_train))
print("测试集特征数量:",len(y_test))

###训练线性回归模型
from sklearn.linear_model import LinearRegression###sklearn.linear_model：sklearn 库中线性模型专属模块，专门存放线性回归、岭回归等线性算法；LinearRegression：线性回归算法

model = LinearRegression()


model.fit(x_train,y_train)###fit = 拟合、训练，是模型学习知识的唯一指令,让模型从训练数据里学习规律

print("模型训练完成")

###查看模型学习到的规律
print("模型系数:",model.coef_)###model.coef_表示存放所有特征对应的权重系数，是一个数组，顺序和特征列顺序完全一一对应
print("模型截距:",model.intercept_)###model.intercept_表示公式里的常数项b，全局基础截距

feature_names = x.columns###.columns：提取表格所有列名，也就是所有特征名字

for name,coef in zip(feature_names,model.coef_):###zip()作用把「特征名字列表」和「系数数组」按顺序两两绑定配对
    print(name,"对房价影响系数:",coef)


###预测测试集
y_pred = model.predict(x_test)

result = pd.DataFrame({
    "真实房价":y_test.values,###这是 列名:这一列填充的数据，加 .values 强制转为纯数组，兼容性更强，代码更稳定
    "预测房价":y_pred###这里加.values和不加都一样，这个y_pred是model.predict()产生的输出天生就是numpy数组
})###pd.DataFrame()表示pandas 创建一张全新表格，用来规整数据，方便肉眼对比

print("预测结果:")
print(result)

###评估模型效果
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score###sklearn.metrics：sklearn 专门存放模型评价指标的工具包；
###mean_absolute_error "平均绝对误差"
###mean_squared_error "均方误差"
###r2_score "R2系数"
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("平均绝对误差 MAE:",mae)
print("均方误差 MSE:",mse)
print("R2 分数:",r2)

####预测一套新房子
new_house = pd.DataFrame({
    "area":[150],
    "rooms":[4],
    "distance":[3],
    "age":[5]
})

predicted_price = model.predict(new_house)

print("新房子信息:")
print(new_house)

print("预测房价:",predicted_price[0],"万元")###这里加[0]是因为是打包好的一组数据如果不加 [0]，打印出来会带着括号：预测房价: [235.6] 万元，观感杂乱；

###可视化预测结果
import matplotlib .pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8,5))

plt.plot(range(len(y_test)),y_test.values,marker="o",label="真实房价")###plt.plot(x轴,y轴,样式,标签)x 轴和上面完全一致，样本编号一一对应；
###y 轴：模型输出的预测房价；
###marker="s"：方形标记，和圆圈区分开，一眼分清两条线；
###label 命名为预测房价
plt.plot(range(len(y_pred)),y_pred,marker="s",label="预测房价")

plt.title("真实房价 vs 预测房价")
plt.xlabel("测试样本编号")
plt.ylabel("房价（万元）")
plt.legend()###用来显示图例的来区分多条曲线

plt.savefig("house_price_prediction.png")
plt.show()

###面积和房价的散点图
plt.figure(figsize=(8,5))

plt.scatter(df["area"],df["price"])###scatter是绘制散点图

plt.title("面积与房价关系")
plt.xlabel("面积")
plt.ylabel("房价（万元）")

plt.savefig("area_price_scatter.png")
plt.show()

###练习1
new_house_2 = pd.DataFrame({
    "area":[100],
    "rooms":[3],
    "distance":[6],
    "age":[8]
})

predicted_price1=model.predict(new_house_2)
print("新房子信息:")
print(new_house_2)

print("预测房价:",predicted_price1[0],"万元")

###练习2
print("========== 练习2：只用面积预测房价 ==========")

X_area = df[["area"]]
y = df["price"]

X_area_train, X_area_test, y_area_train, y_area_test = train_test_split(
    X_area,
    y,
    test_size=0.2,
    random_state=42
)

area_model = LinearRegression()
area_model.fit(X_area_train, y_area_train)

y_area_pred = area_model.predict(X_area_test)

area_mae = mean_absolute_error(y_area_test, y_area_pred)
area_mse = mean_squared_error(y_area_test, y_area_pred)
area_r2 = r2_score(y_area_test, y_area_pred)

print("只用面积预测的 MAE:", area_mae)
print("只用面积预测的 MSE:", area_mse)
print("只用面积预测的 R2:", area_r2)

area_result = pd.DataFrame({
    "真实房价": y_area_test.values,
    "只用面积预测房价": y_area_pred
})

print("只用面积预测结果:")
print(area_result)

print("========== R2 对比 ==========")
print("多特征模型 R2:", r2)
print("只用面积模型 R2:", area_r2)

if r2 > area_r2:
    print("多特征模型效果更好")
elif r2 < area_r2:
    print("只用面积模型效果更好")
else:
    print("两个模型效果差不多")













