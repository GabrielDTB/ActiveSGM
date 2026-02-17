import torch.nn as nn

from pointcept.models.modules import PointModule
from pointcept.models.builder import MODULES


@MODULES.register_module()
class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,  # If True, use independent norm layer per condition (training dataset).
        adaptive=False,  # If True, learn prompt-driven scale/shift parameters via an MLP and the per-dataset vector.
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys()), (
            f"feat and condition must be in point.keys(): {point.keys()}"
        )
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions, (
                f"[PDNorm] condition {condition} not in {self.conditions}"
            )
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(
                2, dim=1
            )  # split into two tensors
            point.feat = point.feat * (1.0 + scale) + shift
        return point
