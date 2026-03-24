import pandas as pd
import numpy as np
import joblib
import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve
)

# =========================================
# 1️⃣ Load Dataset
# =========================================

df = pd.read_csv("data.csv")

df = df.drop(["CustomerId","Surname","Year"],axis=1,errors="ignore")

# =========================================
# 2️⃣ Feature Engineering
# =========================================

df["BalanceSalaryRatio"] = df["Balance"]/(df["EstimatedSalary"]+1)

df["AgeTenureRatio"] = df["Age"]/(df["Tenure"]+1)

df["ProductPerYear"] = df["NumOfProducts"]/(df["Tenure"]+1)

# =========================================
# 3️⃣ Define Features & Target
# =========================================

X = df.drop("Exited",axis=1)

y = df["Exited"]

categorical_cols = ["Geography","Gender"]

numerical_cols = [col for col in X.columns if col not in categorical_cols]

# =========================================
# 4️⃣ Preprocessing Pipeline
# =========================================

preprocessor = ColumnTransformer(

transformers=[

("num",StandardScaler(),numerical_cols),

("cat",OneHotEncoder(drop="first"),categorical_cols)

]

)

# =========================================
# 5️⃣ Model Pipeline + Grid Search
# =========================================

rf = RandomForestClassifier(

random_state=42,
n_jobs=-1

)

pipeline = Pipeline([

("preprocessor",preprocessor),

("classifier",rf)

])

param_grid = {

"classifier__n_estimators":[200,300],

"classifier__max_depth":[None,10,20],

"classifier__min_samples_split":[2,5],

"classifier__min_samples_leaf":[1,2]

}

grid = GridSearchCV(

pipeline,

param_grid,

cv=5,

scoring="roc_auc",

n_jobs=-1

)

# =========================================
# 6️⃣ Train-Test Split
# =========================================

X_train,X_test,y_train,y_test = train_test_split(

X,
y,
test_size=0.2,
stratify=y,
random_state=42

)

# =========================================
# 7️⃣ Train Model
# =========================================

grid.fit(X_train,y_train)

model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

# =========================================
# 8️⃣ Threshold Optimization
# =========================================

y_prob = model.predict_proba(X_test)[:,1]

precisions,recalls,thresholds = precision_recall_curve(y_test,y_prob)

f1_scores = 2*(precisions*recalls)/(precisions+recalls+1e-8)

best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]

print("\nOptimal Threshold:",round(best_threshold,3))

y_pred = (y_prob >= best_threshold).astype(int)

# =========================================
# 9️⃣ Logistic Regression comparison
# =========================================

log_model = Pipeline([

("preprocessor",preprocessor),

("classifier",LogisticRegression(max_iter=1000))

])

log_model.fit(X_train,y_train)

log_prob = log_model.predict_proba(X_test)[:,1]

print("Logistic ROC-AUC:",roc_auc_score(y_test,log_prob))

# =========================================
# 🔟 Evaluation
# =========================================

print("\nClassification Report:")
print(classification_report(y_test,y_pred))

print("\nMetrics:")

print("Accuracy :",round(accuracy_score(y_test,y_pred),4))

print("Precision:",round(precision_score(y_test,y_pred),4))

print("Recall   :",round(recall_score(y_test,y_pred),4))

print("F1 Score :",round(f1_score(y_test,y_pred),4))

print("ROC-AUC  :",round(roc_auc_score(y_test,y_prob),4))

# =========================================
# 1️⃣1️⃣ Cross Validation
# =========================================

cv_scores = cross_val_score(

model,

X,

y,

cv=5,

scoring="roc_auc",

n_jobs=-1

)

print("\nCross ROC-AUC:",round(cv_scores.mean(),4))

# =========================================
# 1️⃣2️⃣ Save Model (FINAL CLEAN SAVE)
# =========================================

saved = {

"model":model,

"threshold":best_threshold,

"sklearn_version":sklearn.__version__

}

joblib.dump(saved,"churn_model.pkl")

print("\nModel saved successfully.")