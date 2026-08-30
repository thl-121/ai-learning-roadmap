import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("学生挂科风险预测系统")

st.write("输入学生成绩,学习时间和缺勤次数，预测该学生是否存在挂科风险。")

df = pd.read_csv("student_scores.csv")

df["total"] = df["math"] + df["english"] + df["python"]
df["average"] = df["total"] / 3

st.subheader("原始数据")
st.dataframe(df)

x = df[["math","english","python","study_hours","absences","total","average"]]
y = df["fail"]

x_train,x_test,y_train,y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
###添加输入框
st.subheader("请输入学生信息")

math = st.number_input("数学成绩",min_value=0,max_value=100,value=60)
english = st.number_input("英语成绩",min_value=0,max_value=100,value=60)
python_score =st.number_input("Python 成绩",min_value=0,max_value=200,value=60)
study_hours = st.number_input("每天学习小时数",min_value=0.0,max_value=12.0,value=3.0)
absences = st.number_input("缺勤次数",min_value=0,max_value=30,value=3)

###计算总分和平均分
total = math + english + python_score
average = total / 3

st.write("总分:",total)
st.write("平均分:",round(average,2))

###预测挂科风险
if st.button("预测挂科风险"):
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

    st.subheader("预测结果")

    st.write("挂科概率:",round(probability,2))

    if probability >= 0.7:
        risk_level = "高风险"
        st.error("风险等级:高风险")
    elif probability >= 0.4:
        risk_level = "中风险"
        st.warning("风险等级:中风险")
    else:
        risk_level = "低风险"
        st.success("风险等级:低风险")
    if prediction == 1:
        st.write("模型判断:可能挂科")
    else:
        st.write("模型判断:不太可能挂科")
