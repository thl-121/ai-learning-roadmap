# Day 11 KMeans 聚类

## 今天学了什么

- 无监督学习
- 聚类
- KMeans
- 标准化 StandardScaler
- PCA 降维
- 肘部法则
- 使用 KMeans 对葡萄酒数据进行聚类

## KNN 和 KMeans 的区别

KNN 是监督学习，需要标签，用来分类。

KMeans 是无监督学习，不需要标签，让模型自己把数据分成几类。

## 我理解的 KMeans

KMeans 会先随机选择几个中心点，然后不断调整样本所属类别和中心点位置，最后把相似的数据分到同一类。

## 重要代码

```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(x_scaled)

cluster_labels = kmeans.labels_