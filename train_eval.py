from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

# =========================
# Models
# =========================

def get_models(random_state: int = 42):
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=random_state))
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            #min_samples_split=10,  # 提高分裂质量
            #min_samples_leaf=5,  # 让叶子更稳定
            random_state=random_state
        ),

    }


# =========================
# LOSO Evaluation
# =========================



def find_best_threshold(y_true, y_score):
    y_score = np.array(y_score)

    # ===== 修复异常值（不改变长度）=====
    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)

    thresholds = np.linspace(0.1, 0.9, 50)
    best_thr = 0.5
    best_f1 = 0

    for thr in thresholds:
        y_pred = (y_score > thr).astype(int)

        try:
            f1 = f1_score(y_true, y_pred)
        except:
            continue

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr



def loso_evaluate(features_df: pd.DataFrame, random_state: int = 42):
    X = features_df.drop(columns=["subject", "label", "label_name"])
    y = features_df["label"].values
    groups = features_df["subject"].values

    # 二分类：baseline(1) -> 0, stress(2) -> 1
    y = np.where(y == 2, 1, 0)

    logo = LeaveOneGroupOut()
    models = get_models(random_state)

    metrics_rows = []
    preds_rows = []

    for model_name, model in models.items():
        accs, f1s, recalls, aucs = [], [], [], []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            test_subject = groups[test_idx][0]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            '''if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]

                y_pred = (y_score > 0.455).astype(int)'''

            if hasattr(model, "predict_proba"):

                # ===== 训练集概率 =====
                y_train_score = model.predict_proba(X_train)[:, 1]
                y_train_score = np.nan_to_num(y_train_score, nan=0.5, posinf=1.0, neginf=0.0)

                # ===== 如果训练集只有一个类别，跳过threshold搜索 =====
                if len(np.unique(y_train)) < 2:
                    best_thr = 0.5
                else:
                    best_thr = find_best_threshold(y_train, y_train_score)

                # ===== 测试集概率 =====
                y_score = model.predict_proba(X_test)[:, 1]
                y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)

                # ===== 用该折最优 threshold =====
                y_pred = (y_score > best_thr).astype(int)
            else:
                y_pred = model.predict(X_test)#

                y_score = y_pred


            # metrics
            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))

            try:
                aucs.append(roc_auc_score(y_test, y_score))
            except:
                aucs.append(np.nan)

            # save predictions
            for i in range(len(y_test)):
                preds_rows.append({
                    "model": model_name,
                    "subject": test_subject,
                    "y_true": y_test[i],
                    "y_pred": y_pred[i],
                    "y_score": y_score[i],
                })

        metrics_rows.append({
            "model": model_name,
            "accuracy_mean": np.nanmean(accs),
            "f1_mean": np.nanmean(f1s),
            "recall_mean": np.nanmean(recalls),
            "roc_auc_mean": np.nanmean(aucs),
        })

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.DataFrame(preds_rows)

    return metrics_df, preds_df
