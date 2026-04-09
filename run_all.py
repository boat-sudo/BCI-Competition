from __future__ import annotations

import argparse
from pathlib import Path

from config import Settings
from build_dataset import build_feature_table
from train_eval import loso_evaluate
from plot_analysis import (
    ensure_dir,
    save_basic_descriptive_stats,
    save_group_difference_stats,
    plot_boxplot,
    plot_histogram,
    plot_subject_level_mean,
    plot_roc_for_best_model,
    plot_confusion_for_best_model,
    plot_subject_score_distribution,
    pick_best_model,
    write_text_summary,
    FEATURES_TO_PLOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click pipeline: raw WESAD -> features -> model -> paper figures"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="WESAD 根目录")
    parser.add_argument("--output_dir", type=str, default="./paper_run_outputs", help="总输出目录")
    parser.add_argument("--window_seconds", type=int, default=300, help="窗口长度（秒）")
    parser.add_argument("--step_seconds", type=int, default=300, help="窗口步长（秒）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figs_dir = output_dir / "figures"

    ensure_dir(output_dir)
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    settings = Settings(
        data_dir=Path(args.data_dir),
        output_dir=output_dir,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )

    print(">>> Step 1/4: Building feature table from raw WESAD...")
    features_df = build_feature_table(settings)
    if features_df.empty:
        raise RuntimeError("特征表为空，请检查数据路径、窗口长度、标签映射或信号质量。")

    feature_csv = tables_dir / "features.csv"
    features_df.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    print(f">>> Saved: {feature_csv}")

    print(">>> Step 2/4: Training and LOSO evaluation...")
    metrics_df, preds_df = loso_evaluate(features_df, random_state=settings.random_state)

    metrics_csv = tables_dir / "metrics.csv"
    preds_csv = tables_dir / "predictions.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    preds_df.to_csv(preds_csv, index=False, encoding="utf-8-sig")

    print(f">>> Saved: {metrics_csv}")
    print(f">>> Saved: {preds_csv}")

    print(">>> Step 3/4: Generating statistics tables...")
    diff_df = save_group_difference_stats(features_df, tables_dir)
    save_basic_descriptive_stats(features_df, tables_dir)

    best_model = pick_best_model(metrics_df)

    print(">>> Step 4/4: Generating paper figures...")
    for feat in FEATURES_TO_PLOT:
        if feat in features_df.columns:
            plot_boxplot(features_df, feat, figs_dir)
            plot_histogram(features_df, feat, figs_dir)
            plot_subject_level_mean(features_df, feat, figs_dir)

    plot_roc_for_best_model(preds_df, best_model, figs_dir)
    plot_confusion_for_best_model(preds_df, best_model, figs_dir)
    plot_subject_score_distribution(preds_df, best_model, figs_dir)

    write_text_summary(metrics_df, diff_df, best_model, output_dir)

    report_md = output_dir / "paper_result_template.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# 论文结果撰写模板\n\n")
        f.write("## 1. 实验设置\n")
        f.write("- 数据集：WESAD\n")
        f.write("- 信号：ECG + Respiration\n")
        f.write("- 指标：HR / RR / HRV\n")
        f.write("- 任务：baseline vs stress 二分类\n")
        f.write(f"- 窗口长度：{args.window_seconds} 秒\n")
        f.write("- 评估方式：LOSO\n\n")

        f.write("## 2. 模型总体表现\n")
        best_row = metrics_df.loc[metrics_df["model"] == best_model].iloc[0]
        f.write(f"- 最佳模型：**{best_model}**\n")
        f.write(f"- Accuracy_mean: **{best_row['accuracy_mean']:.4f}**\n")
        f.write(f"- F1_mean: **{best_row['f1_mean']:.4f}**\n")
        f.write(f"- Recall_mean: **{best_row['recall_mean']:.4f}**\n")
        f.write(f"- ROC_AUC_mean: **{best_row['roc_auc_mean']:.4f}**\n\n")

        f.write("## 3. 生理指标差异解读\n")
        for _, row in diff_df.iterrows():
            feat = row["feature"]
            diff = row["mean_diff_stress_minus_baseline"]
            pval = row["mannwhitney_u_p"]
            direction = "升高" if diff > 0 else "降低"
            f.write(
                f"- **{feat}** 在 stress 相比 baseline 中整体{direction}，"
                f"均值差为 {diff:.4f}，Mann-Whitney U 检验 p = {pval:.6f}。\n"
            )

        f.write("\n## 4. 图表建议\n")
        f.write("- `figures/boxplot_hr_mean_bpm.png`\n")
        f.write("- `figures/boxplot_resp_rate_mean_bpm.png`\n")
        f.write("- `figures/boxplot_rmssd_ms.png`\n")
        f.write("- `figures/roc_*.png`\n")
        f.write("- `figures/confusion_matrix_*.png`\n")

    print("\n>>> 全流程完成")
    print(f">>> 输出目录: {output_dir.resolve()}")
    print(">>> 关键文件：")
    print("    tables/features.csv")
    print("    tables/metrics.csv")
    print("    tables/predictions.csv")
    print("    tables/descriptive_stats.csv")
    print("    tables/group_difference_stats.csv")
    print("    figures/*.png")
    print("    analysis_summary.txt")
    print("    paper_result_template.md")


if __name__ == "__main__":
    main()