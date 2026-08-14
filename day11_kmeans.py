import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()

x = pd.DataFrame(wine.data,columns=wine.feature_names)
y = wine.target

print("特征数据维度:",x.shape)
print("真实类别:",wine.target_names)
print("前5行数据:")
print(x.head())

###标准化数据(KMeans 对数据大小很敏感，所以要标准化)
from sklearn.preprocessing import StandardScaler
###KMeans 靠距离分组，数值单位不一样、量级不一样，会直接 “欺负” 小数值特征，标准化就是把所有特征拉到同一个起跑线
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)###训练集用 fit_transform；测试集只能用 transform ()，不能 fit

print("标准化后的数据:")
print(x_scaled[:5])###只打印前 5 行，避免数据太长刷屏

###训练 KMeans 模型
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3,random_state=42,n_init=10)###n_clusters=3 ✅【最重要】告诉算法：最后总共分成 3 堆;n_init=10意思是：自动重复跑 10 次 KMeans，每次换一组初始队长，最后自动挑效果最好的那一次结果保存。
kmeans.fit(x_scaled)
cluster_labels = kmeans.labels_###labels_ 是 KMeans 跑完之后自带的结果：每个样本的组别号
print("聚类结果:")
print(cluster_labels)

###把聚类结果放进表格
result = x.copy()###把原始特征表 x 复制一份新表格，起名叫 result
result["真实类别"] = y
result["聚类类别"] = cluster_labels

print("前10条聚类结果:")
print(result[["真实类别","聚类类别"]].head(10))

###查看真实类别和聚类类别对照
comparison = pd.crosstab(result["真实类别"],result["聚类类别"])###crosstab = 做一张统计汇总表格

print("真实类别 vs 聚类类别:")
print(comparison)
###算法只是划分人群，聚类编号只是代号，没有 “聚类 0 = 真实 0” 这种绑定关系！
###只要同一堆真实样本，基本集中在同一个聚类列里，就说明聚类效果很好！

###使用PCA降维画图
from sklearn.decomposition import PCA###PCA 就是用来 “压缩维度”，把多维数据压成 2 列，方便画散点图直观观察聚类效果
pca = PCA(n_components=2)###创建 PCA 机器，n_components=2 意思：最后只保留 2 个新特征（PCA1、PCA2）
x_pca = pca.fit_transform(x_scaled)###PCA 生成的 PCA1、PCA2不是原来的任何原始字段，是算法合成出来的、浓缩信息的新坐标

pca_df = pd.DataFrame(x_pca,columns=["PCA1","PCA2"])
pca_df["聚类类别"] = cluster_labels
pca_df["真实类别"] = y

print("PCA 降维后的数据:")
print(pca_df.head())

###画聚类结果图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8,6))
plt.scatter(
    pca_df["PCA1"],
    pca_df["PCA2"],
    c=pca_df["聚类类别"],###按聚类类别自动上色！聚类 0、1、2 会自动分配不同颜色，和你之前交叉表的分组对应
    cmap="viridis",###配色方案（matplotlib 自带，蓝→绿→黄渐变，视觉舒服）
    s=60### 每个圆点的大小，数值越大点越大
)

plt.title("KMeans 葡萄酒聚类结果")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar(label="聚类类别")###在图边上生成颜色条作用：看图例，知道「哪种颜色 = 第几类」，比如深蓝 = 聚类 0，浅绿 = 聚类 1，黄色 = 聚类 2

plt.tight_layout()
plt.savefig("kmeans_cluster.png")
plt.show()

###肘部法则选择k值(KMeans 需要你提前指定分几类。那怎么知道分几类合适？可以用肘部法则)
sse = []

k_values = range(1,11)

for k in k_values:
    model = KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(x_scaled)
    sse.append(model.inertia_)###inertia_ 可以理解成聚类内部误差。一般来说，SSE 越小越好，但 K 越大 SSE 一定会下降，所以要找下降速度开始变慢的位置

print("不同 k 值的 SSE:")
print(sse)

###画肘部图
plt.figure(figsize=(8,5))

plt.plot(k_values,sse,marker="o")

plt.title("KMeans 肘部法则")
plt.xlabel("K 值")
plt.ylabel("SSE")

plt.xticks(k_values)
plt.grid(True,linestyle="--",alpha=0.6)

plt.tight_layout()
plt.savefig("kmeans_elbow.png")
plt.show()


















