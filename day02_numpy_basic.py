# 基础NumPy代码
import numpy as np
scores=np.array([88,92,75,63,99,84,70,56,91,85])
print("成绩数组:",scores)
print("平均分:",np.mean(scores))
print("最高分:",np.max(scores))
print("最低分:",np.min(scores))
print("总分:",np.sum(scores))

#筛选数据
high_scores = scores[scores >= 90]###这是从scores里面筛选出大于等于90的成绩
fail_scores = scores[scores < 60]###这是从scores里面筛选出小于60的成绩
print("90分以上成绩:",high_scores)
print("不及格成绩:",fail_scores)
print("90分以上的人数:",len(high_scores))
print("不及格的人数:",len(fail_scores))

#二维数组
student_scores = np.array([
    [88,92,85],
    [75,80,78],
    [90,95,93],
    [60,70,65],
    [99,96,100]
])###一个学生成绩里面包含三门课程的成绩
print("学生成绩表:")
print(student_scores)

print("每个学生的平均分:",np.mean(student_scores,axis=1))###这里加一个axis=1 表示 按每一行 计算，也就是每个学生
print("每门课程的平均分:",np.mean(student_scores,axis=0))###这里加一个axis=0 表示 按每一列 计算，也就是每门课程
print("每个学生的总分:",np.sum(student_scores,axis=1))

np.random.seed(42)###固定随机数种子，让每次运行代码生成的随机序列完全一模一样，不再随机波动

random_scores = np.random.randint(40,101,size=50)###这里代表随机生成50个40到100之间的整数，写101是因为不包含101

print("随机生成的50个成绩:")
print(random_scores)

print("平均分:",np.mean(random_scores))
print("最高分:",np.max(random_scores))
print("最低分:",np.min(random_scores))
print("不及格人数:",np.sum(random_scores<60))
print("90分以上人数:",np.sum(random_scores>=90))
print("不及格总分:",np.sum(random_scores[random_scores<60]))
print("90分以上总分:",np.sum(random_scores[random_scores>=90]))
###注意别把最后4个语句搞混了，前2个是为了计算人数，后2个是为了计算总分的

def analyze_scores(score_array):
    print("成绩数量:",len(score_array))
    print("平均分:",np.mean(score_array))
    print("最高分:",np.max(score_array))
    print("最低分:",np.min(score_array))
    print("标准差:",np.std(score_array))
    print("不及格人数:",np.sum(score_array<60))
    print("优秀人数:",np.sum(score_array>=90))
print("成绩分析结果:")
analyze_scores(random_scores)