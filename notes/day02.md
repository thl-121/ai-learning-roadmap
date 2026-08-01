# Day 02 NumPy 基础

## 今天学了什么

- 创建 NumPy 数组
- 数组的平均值、最大值、最小值、总和
- 使用条件筛选数组
- 二维数组
- axis=0 和 axis=1
- 随机生成数据
- 用函数分析成绩数据

## 重要代码

```python
scores = np.array([88, 92, 75, 63, 99])
np.mean(scores)
np.max(scores)
np.min(scores)
scores[scores >= 90]
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