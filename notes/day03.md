# Day 03 Pandas 基础

## 今天学了什么

- CSV 文件
- Pandas DataFrame
- read_csv 读取表格
- head 查看前几行
- info 查看表格信息
- 新增列
- 计算总分和平均分
- 筛选数据
- 排序数据
- 保存 CSV 文件

## 重要代码

```python
df = pd.read_csv("students.csv")
df["total"] = df["math"] + df["english"] + df["python"]
df[df["python"] >= 90]
df.sort_values(by="total", ascending=False)
df.to_csv("students_result.csv", index=False)