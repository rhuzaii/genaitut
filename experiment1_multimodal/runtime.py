"""Small helpers shared by the generation stages."""

from __future__ import annotations

import gc
from typing import Any


def release_model(model: Any) -> None:
    """Drop a model and return its GPU memory to the allocator."""
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def describe_device(device: str) -> str:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"cuda ({name}, {total:.1f} GiB)"
    return device
