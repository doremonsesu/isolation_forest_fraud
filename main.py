import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, auc)

# ===== 0. Tạo thư mục lưu output =====
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ===== 1. Load dữ liệu =====
df = pd.read_csv("creditcard.csv")

# ===== 2. EDA ngắn gọn =====
print("Tổng số giao dịch:", len(df))
print("Số lượng giao dịch gian lận:", df['Class'].sum())
print("Tỉ lệ gian lận:", round(df['Class'].mean() * 100, 4), "%")

plt.figure(figsize=(10, 6))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', label='Bình thường', kde=True)
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label='Gian lận', kde=True)
plt.title('Phân phối số tiền giao dịch theo lớp')
plt.xlabel('Số tiền (Amount)')
plt.ylabel('Tần suất')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "amount_distribution.png"))
plt.close()

# ===== 3. Tiền xử lý =====
df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])
df = df.drop(['Time', 'Amount'], axis=1)
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===== 4. Huấn luyện Isolation Forest =====
iso_forest = IsolationForest(n_estimators=100, contamination=y_train.mean(), random_state=42)
iso_forest.fit(X_train)

# ===== 5. Dự đoán & đánh giá =====
y_pred = iso_forest.predict(X_test)
y_pred = np.where(y_pred == -1, 1, 0)

conf_mat = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Confusion Matrix ===\n", conf_mat)
print("\n=== Classification Report ===\n", report)
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# ===== 6. Lưu báo cáo =====
with open(os.path.join(output_dir, "isolation_forest_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Tổng số giao dịch: {len(df)}\n")
    f.write(f"Số lượng giao dịch gian lận: {df['Class'].sum()}\n")
    f.write(f"Tỉ lệ gian lận: {round(df['Class'].mean() * 100, 4)} %\n\n")
    f.write("=== Confusion Matrix ===\n")
    f.write(str(conf_mat) + "\n\n")
    f.write("=== Classification Report ===\n")
    f.write(report + "\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-score:  {f1:.4f}\n")

# ===== 7. Vẽ Confusion Matrix =====
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Bình thường', 'Gian lận'],
            yticklabels=['Bình thường', 'Gian lận'])
plt.title('Confusion Matrix - Isolation Forest')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# ===== 8. Precision-Recall Curve =====
y_scores = iso_forest.decision_function(X_test)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, -y_scores)
pr_auc = auc(recall_curve, precision_curve)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, marker='.', label=f'PR AUC = {pr_auc:.4f}')
plt.title('Precision-Recall Curve - Isolation Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pr_curve.png"))
plt.close()

# ===== 9. Tuning contamination =====
contamination_values = [0.001, 0.002, 0.005, y_train.mean()]
tuning_results = []

for contam in contamination_values:
    model = IsolationForest(n_estimators=100, contamination=contam, random_state=42)
    model.fit(X_train)
    y_pred_tuned = np.where(model.predict(X_test) == -1, 1, 0)

    p = precision_score(y_test, y_pred_tuned)
    r = recall_score(y_test, y_pred_tuned)
    f1 = f1_score(y_test, y_pred_tuned)
    tuning_results.append((contam, p, r, f1))

# Ghi kết quả tuning vào file
with open(os.path.join(output_dir, "contamination_tuning.txt"), "w", encoding="utf-8") as f:
    f.write("Contamination\tPrecision\tRecall\tF1-score\n")
    for contam, p, r, f1 in tuning_results:
        f.write(f"{contam:.4f}\t\t{p:.4f}\t\t{r:.4f}\t{f1:.4f}\n")
