# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.utils import is_torch_min_version

jit_fuser = torch.jit.script
# nvFuser is deprecated in PyTorch JIT starting from 2.2
# NOTE: torch.compile (inductor) has a known system RAM leak that grows with every
# compiled kernel call, causing OOM after hours of training. Using torch.jit.script
# instead avoids this. See: https://github.com/pytorch/pytorch/issues/96937
# For short-lived jobs (inference, probes), torch.compile is fine — set
# MEGATRON_USE_TORCH_COMPILE=1 to re-enable it.
import os
if is_torch_min_version("2.2.0a0") and os.environ.get("MEGATRON_USE_TORCH_COMPILE", "0") == "1":
    jit_fuser = torch.compile
