# Day 04 Matplotlib 数据可视化

## 今天学了什么

- 使用 Matplotlib 画图
- 柱状图 bar
- 折线图 plot
- 设置标题、横轴、纵轴
- 保存图片 savefig
- 使用 subplot 绘制多个子图
- 用图表展示学生成绩数据

## 重要代码

```python
plt.bar(df["name"], df["total"])
plt.plot(df["name"], df["python"], marker="o")
plt.title("学生总分柱状图")
plt.xlabel("学生姓名")
plt.ylabel("总分")
plt.savefig("score_chart.png")
plt.show()