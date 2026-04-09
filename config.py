from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    """
    全局配置类，用于在各模块之间传递参数
    """

    # ===== 路径 =====
    data_dir: Path                 # WESAD 数据根目录
    output_dir: Path               # 输出目录

    # ===== 窗口参数 =====
    window_seconds: int = 60       # 窗口长度（秒）
    step_seconds: int = 30         # 滑动步长（秒）

    # ===== 随机性 =====
    random_state: int = 42         # 随机种子（用于模型训练）

    # ===== 采样率（WESAD默认） =====
    chest_fs: int = 700            # 胸部信号采样率
    wrist_fs: int = 64             # 手腕信号采样率（如果以后扩展）

    # ===== 标签定义 =====
    LABEL_BASELINE: int = 1
    LABEL_STRESS: int = 2
    LABEL_AMUSEMENT: int = 3
    LABEL_MEDITATION: int = 4

    def __post_init__(self):
        """
        初始化后检查路径是否合法
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        # 自动创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ===== 可扩展：标签映射 =====
    @property
    def label_map(self) -> dict:
        """
        将标签映射为字符串
        """
        return {
            self.LABEL_BASELINE: "baseline",
            self.LABEL_STRESS: "stress",
            self.LABEL_AMUSEMENT: "amusement",
            self.LABEL_MEDITATION: "meditation",
        }

    # ===== 可扩展：任务标签过滤 =====
    @property
    def target_labels(self) -> list[int]:
        """
        当前任务只做 baseline vs stress
        """
        return [self.LABEL_BASELINE, self.LABEL_STRESS]
