import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# 1. تنظیم رندوم سید برای تکرارپذیری
np.random.seed(42)

# 2. شبیه‌سازی دادگان بالینی و ژنومیک (مقاومت به پلاتین در سرطان تخمدان)
n_samples = 500
n_genes = 100

# شبیه‌سازی بیان ژن‌ها (مثلا ژن‌های کلیدی ترمیم DNA و مقاومت)
gene_columns = [f"Marker_Gene_{i+1}" for i in range(n_genes)]
X_genes = pd.DataFrame(np.random.randn(n_samples, n_genes), columns=gene_columns)

# اضافه کردن ویژگی‌های بالینی مهم
X_clinical = pd.DataFrame({
    'Patient_Age': np.random.randint(38, 82, size=n_samples),
    'FIGO_Stage': np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.1, 0.15, 0.55, 0.2]),
    'Residual_Tumor_Size': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]), # 0: بدون تومور باقی‌مانده، 1: >1cm
    'CA125_Level': np.random.exponential(scale=150.0, size=n_samples) + 20
})

# ادغام دیتاست
X_full = pd.concat([X_genes, X_clinical], axis=1)

# تعریف متغیر هدف (0: حساس به پلاتین / 1: مقاوم به پلاتین)
# فرض می‌کنیم تغییرات خاص در ژن‌های اولیه و مرحله پیشرفته احتمال مقاومت را افزایش می‌دهد
latent_risk = (
    0.4 * X_full['Marker_Gene_12'] + 
    0.5 * X_full['Marker_Gene_45'] + 
    0.6 * X_full['FIGO_Stage'] + 
    0.5 * X_full['Residual_Tumor_Size'] +
    np.random.normal(0, 1, n_samples)
)
prob = 1 / (1 + np.exp(-latent_risk))
y = (prob > np.median(prob)).astype(int)

# 3. ارزیابی مدل با Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

cv_scores = cross_val_score(model, X_full, y, cv=cv, scoring='roc_auc')
print(f"Cross-Validation ROC-AUC Scores: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# آموزش نهایی روی کل داده‌ها
model.fit(X_full, y)

# ذخیره مدل و مشخصات
joblib.dump(model, 'platinum_resistance_model.pkl')

# 4. استخراج و رسم اهمیت ویژگی‌ها (Feature Importance)
importances = model.feature_importances_
feature_names = X_full.columns
top_indices = np.argsort(importances)[-10:] # 10 ویژگی برتر

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[top_indices], y=feature_names[top_indices], palette='viridis')
plt.title("Top Biomarkers and Clinical Features for Platinum Resistance")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig('platinum_feature_importance.png', dpi=300)
plt.close()

print("Model trained successfully and feature importance plot saved!")
