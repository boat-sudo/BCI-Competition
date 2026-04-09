# 论文结果撰写模板

## 1. 实验设置
- 数据集：WESAD
- 信号：ECG + Respiration
- 指标：HR / RR / HRV
- 任务：baseline vs stress 二分类
- 窗口长度：300 秒
- 评估方式：LOSO

## 2. 模型总体表现
- 最佳模型：**RandomForest**
- Accuracy_mean: **0.7746**
- F1_mean: **0.7705**
- Recall_mean: **0.9000**
- ROC_AUC_mean: **0.8556**

## 3. 生理指标差异解读
- **hr_mean_bpm** 在 stress 相比 baseline 中整体升高，均值差为 3.1752，Mann-Whitney U 检验 p = 0.022259。
- **rmssd_ms** 在 stress 相比 baseline 中整体升高，均值差为 72.3634，Mann-Whitney U 检验 p = 0.000002。
- **resp_rate_mean_bpm** 在 stress 相比 baseline 中整体降低，均值差为 -1.0461，Mann-Whitney U 检验 p = 0.000201。

## 4. 图表建议
- `figures/boxplot_hr_mean_bpm.png`
- `figures/boxplot_resp_rate_mean_bpm.png`
- `figures/boxplot_rmssd_ms.png`
- `figures/roc_*.png`
- `figures/confusion_matrix_*.png`
