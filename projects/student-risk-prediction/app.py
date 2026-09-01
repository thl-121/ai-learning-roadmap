import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] =False

st.set_page_config(
    page_title="学生挂科风险预测系统",
    page_icon="📊",
    layout="wide"
)

st.title("学生挂科风险预测系统")
st.write("本系统应用于学生成绩,学习时间和缺勤次数，预测学生是否存在挂科风险。")
df = pd.read_csv("student_scores.csv")
df["total"] = df["math"] + df["english"] + df["python"]
df["average"] = df["total"] / 3

x = df[["math","english","python","study_hours","absences","total","average"]]
y = df["fail"]
x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)

model = joblib.load("student_risk_model.pkl")
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
###添加顶部指标卡片
st.subheader("项目概览")

col1,col2,col3,col4 = st.columns(4)
with col1:
    st.metric("学生数量",len(df))
with col2:
    st.metric("平均分均值",round(df["average"].mean(),2))
with col3:
    st.metric("挂科人数",int(df["fail"].sum()))
with col4:
    st.metric("模型准确率",f"{accuracy:.2f}")

###用侧边栏输入学生信息
st.header("输入学生信息")

math = st.sidebar.number_input("数学成绩",min_value=0,max_value=100,value=60)
english = st.sidebar.number_input("英语成绩",min_value=0,max_value=100,value=60)
python_score = st.sidebar.number_input("Python成绩",min_value=0,max_value=100,value=60)
study_hours = st.sidebar.number_input("每天学习小时数",min_value=0.0,max_value=12.0,value=3.0)
absences = st.sidebar.number_input("缺勤次数",min_value=0,max_value=30,value=3)

total =math + english + python_score
average = total / 3

st.sidebar.write("总分:",total)
st.sidebar.write("平均分:",round(average,2))

###添加预测结果
st.subheader("挂科风险预测")
if st.sidebar.button("开始预测"):
    new_student = pd.DataFrame({
        "math":[math],
        "english":[english],
        "python":[python_score],
        "study_hours":[study_hours],
        "absences":[absences],
        "total":[total],
        "average":[average]
    })
    prediction = model.predict(new_student)[0]
    probability = model.predict_proba(new_student)[0][1]

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("预测挂科概率",f"{probability:.2f}")
    with col2:
        st.metric("预测结果","可能挂科" if prediction ==1 else "风险较低")
    with col3:
        if probability >= 0.7:
            st.error("风险等级：高风险")
        elif probability >= 0.4:
            st.warning("风险等级：中风险")
        else:
            st.success("风险等级：低风险")
    st.write("输入数据:")
    st.dataframe(new_student)

st.subheader("学生数据表")
st.dataframe(df)

###添加成绩分布图
st.subheader("平均分分布")
fig1,ax1 = plt.subplots(figsize=(8,5))

ax1.hist(df["average"],bins=8,color="skyblue",edgecolor="black")
ax1.set_title("学生平均分分布")
ax1.set_xlabel("平均分")
ax1.set_ylabel("人数")

st.pyplot(fig1)

###添加缺勤和平均分关系图
st.subheader("缺勤次数与平均分关系")

fig2,ax2 = plt.subplots(figsize=(8,5))

scatter = ax2.scatter(
    df["absences"],
    df["average"],
    c=df["fail"],
    cmap="coolwarm",
    s=80
)
ax2.set_title("缺勤次数与平均分关系")
ax2.set_xlabel("缺勤次数")
ax2.set_ylabel("平均分")

st.pyplot(fig2)

###添加特征重要性图
st.subheader("模型特征重要性")
feature_importance = pd.DataFrame({
    "feature":x.columns,
    "importance":model.feature_importances_
})
feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

fig3,ax3 = plt.subplots(figsize=(8,5))

ax3.barh(feature_importance["feature"],feature_importance["importance"])
ax3.set_title("随机森林特征重要性")
ax3.set_xlabel("重要性")
ax3.set_ylabel("特征")
ax3.invert_yaxis()

st.pyplot(fig3)

st.dataframe(feature_importance)








































