"""A 数据工程师的 EDA 与无泄漏预处理接口。"""

from .config import DATA_PATH, RANDOM_STATE, TEST_SIZE
from .quality import build_quality_summary, load_raw_data

__all__ = [
    "DATA_PATH",
    "RANDOM_STATE",
    "TEST_SIZE",
    "PreparedData",
    "build_quality_summary",
    "load_raw_data",
    "prepare_data",
    "prepare_data_from_frame",
]


def __getattr__(name: str):
    """延迟导入预处理接口，避免命令行模块执行时被提前加载。"""

    if name in {"PreparedData", "prepare_data", "prepare_data_from_frame"}:
        from . import preprocessing

        return getattr(preprocessing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
