import torch
import torch.nn as nn


class CNN2D(nn.Module):

    """
    STFT时频图雷达分类模型

    输入：
        (B, 1, F, T)

    典型：
        (B, 1, 256, 33)

    输出：
        (B, 6)
    """

    def __init__(
        self,
        num_classes=6
    ):

        super().__init__()


        self.features = nn.Sequential(

            # =================================================
            # Block 1
            # =================================================

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),


            # =================================================
            # Block 2
            # =================================================

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),


            # =================================================
            # Block 3
            # =================================================

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),


            # =================================================
            # Block 4
            # =================================================

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU()
        )


        # 不管STFT尺寸如何，
        # 最终变成：
        #
        # (B,256,1,1)

        self.global_pool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = self.global_pool(
            x
        )

        x = self.classifier(
            x
        )

        return x


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    model = CNN2D(
        num_classes=6
    )

    x = torch.randn(
        16,
        1,
        256,
        33
    )

    y = model(
        x
    )

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