#挑出识别错误的数据，并分析其错误数据的特性
import pandas as pd

# =========================
# 全局特征列表（统一🔥）
# =========================
FEATURES = [
    # 原始特征
    "hr_mean_bpm",
    "rmssd_ms",
    "sdnn_ms",
    "resp_rate_mean_bpm",

    # 比值特征🔥
    "hr_div_rmssd",
    "hr_div_sdnn",
    "rmssd_div_sdnn",
    "hr_div_resp",
]


# 1. 读取预测结果
df = pd.read_csv("tables/predictions.csv")

# 2. 找出预测错误的样本
wrong_df = df[df["y_true"] != df["y_pred"]]

# 3. 保存
wrong_df.to_csv("tables/wrong_predictions.csv", index=False)

print("错误样本数量：", len(wrong_df))

# 4. 看哪个被试最容易出错
print("\n各被试错误数量：")
print(wrong_df.groupby("subject").size().sort_values(ascending=False))

# 5. 看混淆情况
print("\n混淆矩阵统计：")
print(pd.crosstab(df["y_true"], df["y_pred"]))



# =========================
# 6. 合并特征数据（关键！！）
# =========================
feat = pd.read_csv("tables/features.csv")

# ⚠️ 这里只能用同一模型（否则会乱）
best_model = df["model"].value_counts().idxmax()
df_model = df[df["model"] == best_model].reset_index(drop=True)

# 合并（按行对齐）
merged = pd.concat([feat, df_model[["y_true", "y_pred"]]], axis=1)

# 标记是否预测错误
merged["is_wrong"] = merged["y_true"] != merged["y_pred"]



# =========================
# 去除异常值（IQR方法🔥）
# =========================
def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df

# 要处理的特征（加上你新特征🔥）
'''features = [
    "hr_mean_bpm",
    "rmssd_ms",
    "sdnn_ms",
    "resp_rate_mean_bpm",

# =====================
    # 比值特征（新增🔥）
    # =====================
    "hr_div_rmssd",
    "hr_div_sdnn",
    "rmssd_div_sdnn",
    "hr_div_resp",
]'''

# 去异常值
merged = remove_outliers(merged, FEATURES)

print("去异常值后样本数：", len(merged))




print("\n===== 错误样本 vs 正确样本 特征对比 =====")

# =========================
# 7. 找“共性”（均值对比🔥）
# =========================
#features = ["hr_mean_bpm", "rmssd_ms", "resp_rate_mean_bpm"]

group_mean = merged.groupby("is_wrong")[FEATURES].mean()

print(group_mean)

# =========================
# 8. 哪些特征差异最大
# =========================
diff = group_mean.loc[True] - group_mean.loc[False]
print("\n===== 错误样本 - 正确样本 差值 =====")
print(diff.sort_values(ascending=False))

# =========================
# 9. 按类别分析（更深入🔥）
# =========================
print("\n===== 错误样本真实类别分布 =====")
print(merged[merged["is_wrong"]]["y_true"].value_counts())

# =========================
# 10. 每个被试错误率（LOSO关键🔥）
# =========================
subject_error_rate = merged.groupby("subject")["is_wrong"].mean()

print("\n===== 各被试错误率 =====")
print(subject_error_rate.sort_values(ascending=False))

print("\n===== 差异最大特征（按绝对值排序） =====")
print(diff.reindex(diff.abs().sort_values(ascending=False).index))


group_mean.to_csv("tables/error_feature_mean.csv")
diff.to_csv("tables/error_feature_diff.csv")



#将数据三维可视化
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 使用 merged 数据（已经合并好的）
df_plot = merged.copy()

# 只取需要的列
x = df_plot["hr_mean_bpm"]# 心率
y = df_plot["resp_rate_mean_bpm"]# 呼吸率
z = df_plot["rmssd_ms"]# HRV
labels = df_plot["y_true"]   # 用真实标签上色

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 不同类别分开画
for label in labels.unique():
    idx = labels == label
    ax.scatter(
        x[idx],
        y[idx],
        z[idx],
        label=f"Class {label}",
        alpha=0.7
    )

# 坐标轴标签
ax.set_xlabel("Heart Rate (HR)") # X轴：心率
ax.set_ylabel("Respiration Rate")# Y轴：呼吸率
ax.set_zlabel("RMSSD (HRV)")# Z轴：心率变异性

ax.set_title("3D Visualization of Physiological Features")

# 图例
ax.legend()

plt.show()
'''# =========================
# 选择特征（包含比值🔥）
# =========================
features = [
    "hr_mean_bpm",
    "rmssd_ms",
    "sdnn_ms",
    "resp_rate_mean_bpm",
    "hr_div_rmssd",
    "hr_div_sdnn",
    "rmssd_div_sdnn",
    "hr_div_resp"
]

df_plot = merged.copy()
labels = df_plot["y_true"]

# =========================
# 遍历所有3特征组合
# =========================
combos = list(itertools.combinations(features, 3))

print(f"总组合数: {len(combos)}")

# =========================
# 逐个画图
# =========================
for combo in combos:

    x_col, y_col, z_col = combo

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for label in labels.unique():
        idx = labels == label

        ax.scatter(
            df_plot.loc[idx, x_col],
            df_plot.loc[idx, y_col],
            df_plot.loc[idx, z_col],
            label=f"Class {label}",
            alpha=0.6
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)

    ax.set_title(f"3D Feature Space: {x_col}, {y_col}, {z_col}")
    ax.legend()

    plt.show()'''









