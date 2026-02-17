from .builder import build_criteria

from .misc import (
    CrossEntropyLoss,
    SmoothCELoss,
    DiceLoss,
    FocalLoss,
    BinaryFocalLoss,
    L1Loss,
)
from .lovasz import LovaszLoss

from .sim_dino_clstoken_loss import MCRLoss
from .sim_ibot_patch_loss import CosinePatchLoss
