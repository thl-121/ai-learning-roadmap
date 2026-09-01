import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("student_scores.csv")

df["total"] = df["math"] + df["english"] + df["python"]
df["average"] = df["total"] / 3

X = df[["math", "english", "python", "study_hours", "absences", "total", "average"]]
y = df["fail"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("模型训练完成")
print("模型准确率:", accuracy)

joblib.dump(model, "student_risk_model.pkl")

print("模型已保存到 student_risk_model.pkl")