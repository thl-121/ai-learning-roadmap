import pandas as pd

df = pd.read_csv("students.csv")

print("学生成绩表:")
print(df)

print("前5行数据:")
print(df.head())###默认为前5行的数据，如果是df.head(10)就是查看前10行

print("表格基本信息:")
print(df.info())

###计算总分和平均分
df["total"] = df["math"] + df["english"] + df["python"]
df["average"] = df["total"] / 3

print("添加总分和平均分后的表格:")
print(df)

###统计每门平均分
print("数学平均分:",df["math"].mean())
print("英语平均分:",df["english"].mean())
print("Python平均分:",df["python"].mean())

print("数学最高分:",df["math"].max())
print("英语最高分:",df["english"].max())
print("Python最高分:",df["python"].max())

###筛选数据
high_python_students = df[df["python"]>=90]
fail_students = df[df["average"]<60]

print("Python成绩90分以上的学生:")
print(high_python_students)

print("平均分不及格的学生:")
print(fail_students)

###排序
ranked_df=df.sort_values(by="total",ascending=False)###这里ascending=Fasle表示从高到低排序，如果这里是ascending=True表示从低到高排序

print("按总分从高到低排序:")
print(ranked_df)

###保存分析结果
ranked_df.to_csv("students_result.csv",index=False)###index=Fasle表示不保存左侧序号，表格干净整洁，只有姓名，分数，总分等数据；
###若写成index=True会多出一列多余行号

print("分析结果已保存到 students_result.csv")
















