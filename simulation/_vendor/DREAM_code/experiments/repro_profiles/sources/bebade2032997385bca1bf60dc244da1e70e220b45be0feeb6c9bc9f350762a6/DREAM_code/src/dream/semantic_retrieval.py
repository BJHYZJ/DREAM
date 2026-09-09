"""Hardware-independent semantic retrieval rules used by DREAM adapters.

The feature alignment is the same normalized voxel-feature/text product used
in SparseVoxelMap.find_alignment_over_model. This is not an object detector.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_alignment(text_features: torch.Tensor, voxel_features: torch.Tensor):
    features = F.normalize(voxel_features, p=2, dim=-1).cpu()
    return text_features.cpu().float() @ features.float().T
