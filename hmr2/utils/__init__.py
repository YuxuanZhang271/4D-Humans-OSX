import torch
from typing import Any

from .renderer import Renderer
from .mesh_renderer import MeshRenderer
from .skeleton_renderer import SkeletonRenderer
from .pose_utils import eval_pose, Evaluator

def recursive_to(x: Any, target: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    # normalize target to string for simple checks (e.g., 'mps')
    target_str = str(target).lower()

    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        # On MPS, float64 (double) is not supported by the MPS backend.
        # Convert floating-point tensors to float32 before moving to MPS.
        if "mps" in target_str and x.is_floating_point() and x.dtype == torch.float64:
            return x.to(device=target, dtype=torch.float32)
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x
