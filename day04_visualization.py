import pandas as pd
import matplotlib.pyplot as plt

###解决中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


df = pd.read_csv("students.csv")

df["total"] = df["math"] + df["english"] + df["python"]
df["average"] = df["total"] / 3

print(df)
###画第一个图，总分柱状图
plt.figure(figsize=(8,5))###figure就是画布的意思，这里表示搭建一个宽度8英寸，高度5英寸；

plt.bar(df["name"],df["total"])###这里bar表示柱状图，且df里面 姓名 为横轴， 总分 为纵轴；

plt.title("学生总分柱状图")
plt.xlabel("学生姓名")
plt.ylabel("总分")

plt.savefig("total_score_bar.png")
plt.show()

###画第二个图，每门平均分柱状图
course_name = ["数学","英语","Python"]
course_averages = [
    df["math"].mean(),
    df["english"].mean(),
    df["python"].mean()
]
plt.figure(figsize=(8,5))

plt.bar(course_name,course_averages)

plt.title("每门平均分")
plt.xlabel("课程")
plt.ylabel("平均分")

plt.savefig("course_average_bar.png")
plt.show()

###画第三个图，Python成绩折线图
plt.figure(figsize=(8,5))

plt.plot(df["name"],df["python"],marker="o")###这里面的marker="o"代表额外的修饰，在每个数据点上画上一个实心小圆点

plt.title("学生 python 成绩折线图")
plt.xlabel("学生姓名")
plt.ylabel("Python 成绩")

plt.savefig("python_score_line.png")
plt.show()
###画一个综合图，保存为score_chart.png
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)###这个前两个 2 2 是为了将整张画布均匀分为2行2列，一共4个小格子，后面那个1代表将这个图放在第一个格子
plt.bar(df["name"],df["total"])
plt.title("学生总分")
plt.xlabel("学生")
plt.ylabel("总分")

plt.subplot(2,2,2)
plt.bar(course_name,course_averages)
plt.title("课程平均分")
plt.xlabel("课程")
plt.ylabel("平均分")

plt.subplot(2,2,3)
plt.plot(df["name"],df["python"],marker="o")
plt.title("Python 成绩")
plt.xlabel("学生")
plt.ylabel("分数")

plt.subplot(2,2,4)
plt.plot(df["name"],df["math"],marker="o")
plt.title("Python 成绩")
plt.xlabel("学生")
plt.ylabel("分数")

plt.tight_layout()###这个是为了自动排版紧凑布局
plt.savefig("score_chart.png")
plt.show()
