import torch
import torch.nn as nn

from torchvision.models import resnet18


class RadarResNet18(nn.Module):
    """
    针对雷达STFT时频矩阵修改的ResNet18

    输入：
        (B, 1, 256, 33)

    输出：
        (B, 6)
    """

    def __init__(
        self,
        num_classes=6
    ):

        super().__init__()

        # 不使用ImageNet预训练
        self.model = resnet18(
            weights=None
        )

        # ====================================================
        # 修改输入层
        #
        # 原始ResNet18：
        # 3通道
        # 7×7
        # stride=2
        #
        # 雷达STFT：
        # 1通道
        # 3×3
        # stride=1
        #
        # 避免时间维过快缩小
        # ====================================================

        self.model.conv1 = nn.Conv2d(

            in_channels=1,

            out_channels=64,

            kernel_size=3,

            stride=1,

            padding=1,

            bias=False
        )

        # ====================================================
        # 删除开头MaxPool
        #
        # 原本会再次 /2
        # 对33个时间bin来说压缩太快
        # ====================================================

        self.model.maxpool = nn.Identity()

        # ====================================================
        # 修改分类器
        # ====================================================

        in_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )


    def forward(self, x):

        return self.model(x)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":

    model = RadarResNet18(
        num_classes=6
    )

    x = torch.randn(
        8,
        1,
        256,
        33
    )

    y = model(x)

    print(
        "输入：",
        x.shape
    )

    print(
        "输出：",
        y.shape
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "参数量：",
        f"{total_params:,}"
    )