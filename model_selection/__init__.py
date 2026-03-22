"""
model_selection — LORIS 动态模型路由与选择。
"""


def __getattr__(name):
    """延迟导入，避免 python -m 直接执行时的循环导入警告。"""
    _public = {"SelectionNetwork", "HybridLoss", "FinalSelector", "build_oracle_mask"}
    if name in _public:
        from model_selection.dynamic_router import (
            SelectionNetwork,
            HybridLoss,
            FinalSelector,
            build_oracle_mask,
        )
        _map = {
            "SelectionNetwork": SelectionNetwork,
            "HybridLoss": HybridLoss,
            "FinalSelector": FinalSelector,
            "build_oracle_mask": build_oracle_mask,
        }
        return _map[name]
    raise AttributeError(f"module 'model_selection' has no attribute {name!r}")


__all__ = [
    "SelectionNetwork",
    "HybridLoss",
    "FinalSelector",
    "build_oracle_mask",
]
