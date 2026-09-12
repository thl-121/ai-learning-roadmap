from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =========================
# 1. 读取数据
# =========================

current_dir = Path(__file__).parent
data_path = current_dir / "student_scores.csv"

df = pd.read_csv(data_path)

print("原始数据：")
print(df.head())
print("\n数据维度：", df.shape)


# =========================
# 2. 构造特征
# =========================

df["total"] = (
    df["math"]
    + df["english"]
    + df["python"]
)

df["average"] = df["total"] / 3

feature_columns = [
    "math",
    "english",
    "python",
    "study_hours",
    "absences",
    "total",
    "average"
]

X = df[feature_columns]
y = df["fail"]

print("\n使用的特征：")
print(feature_columns)

print("\n标签数量：")
print(y.value_counts())


# =========================
# 3. 划分训练集和测试集
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 4. 建立基础随机森林模型
# =========================

base_model = RandomForestClassifier(
    random_state=42
)

base_model.fit(X_train, y_train)

base_pred = base_model.predict(X_test)

base_accuracy = accuracy_score(y_test, base_pred)
base_precision = precision_score(
    y_test,
    base_pred,
    zero_division=0
)
base_recall = recall_score(
    y_test,
    base_pred,
    zero_division=0
)
base_f1 = f1_score(
    y_test,
    base_pred,
    zero_division=0
)

print("\n===== 调参前结果 =====")
print(f"Accuracy：{base_accuracy:.4f}")
print(f"Precision：{base_precision:.4f}")
print(f"Recall：{base_recall:.4f}")
print(f"F1：{base_f1:.4f}")


# =========================
# 5. 设置参数搜索范围
# =========================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5, 8],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2, 4]
}


# =========================
# 6. 网格搜索
# =========================

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

print("\n正在进行网格搜索，请稍等……")

grid_search.fit(X_train, y_train)

print("\n===== 网格搜索完成 =====")
print("最佳参数：")
print(grid_search.best_params_)

print(
    f"\n交叉验证最佳 F1："
    f"{grid_search.best_score_:.4f}"
)


# =========================
# 7. 使用最佳模型测试
# =========================

best_model = grid_search.best_estimator_

tuned_pred = best_model.predict(X_test)

tuned_accuracy = accuracy_score(
    y_test,
    tuned_pred
)

tuned_precision = precision_score(
    y_test,
    tuned_pred,
    zero_division=0
)

tuned_recall = recall_score(
    y_test,
    tuned_pred,
    zero_division=0
)

tuned_f1 = f1_score(
    y_test,
    tuned_pred,
    zero_division=0
)

print("\n===== 调参后结果 =====")
print(f"Accuracy：{tuned_accuracy:.4f}")
print(f"Precision：{tuned_precision:.4f}")
print(f"Recall：{tuned_recall:.4f}")
print(f"F1：{tuned_f1:.4f}")

print("\n===== 调参后分类报告 =====")
print(
    classification_report(
        y_test,
        tuned_pred,
        target_names=["不挂科", "挂科"],
        zero_division=0
    )
)


# =========================
# 8. 保存对比结果
# =========================

result_df = pd.DataFrame([
    {
        "model": "调参前随机森林",
        "accuracy": base_accuracy,
        "precision": base_precision,
        "recall": base_recall,
        "f1": base_f1
    },
    {
        "model": "调参后随机森林",
        "accuracy": tuned_accuracy,
        "precision": tuned_precision,
        "recall": tuned_recall,
        "f1": tuned_f1
    }
])

result_path = current_dir / "day22_tuning_result.csv"

result_df.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n对比结果已保存到：{result_path}")


# =========================
# 9. 保存最佳模型
# =========================

model_path = current_dir / "student_risk_model_tuned.pkl"

joblib.dump(
    best_model,
    model_path
)

print(f"最佳模型已保存到：{model_path}")


