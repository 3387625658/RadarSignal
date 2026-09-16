import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    用于雷达IQ信号分类的1D-CNN

    输入：
        (batch, 2, 2048)

    其中：
        channel 0 = I
        channel 1 = Q

    输出：
        (batch, 6)
    """

    def __init__(self, num_classes=6):

        super().__init__()

        # ====================================================
        # 第一层卷积
        # ====================================================

        self.block1 = nn.Sequential(

            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=9,
                stride=1,
                padding=4
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            )
        )

        # ====================================================
        # 第二层卷积
        # ====================================================

        self.block2 = nn.Sequential(

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            )
        )

        # ====================================================
        # 第三层卷积
        # ====================================================

        self.block3 = nn.Sequential(

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            )
        )

        # ====================================================
        # 第四层卷积
        # ====================================================

        self.block4 = nn.Sequential(

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            )
        )

        # ====================================================
        # 自适应全局池化
        #
        # 不管前面的长度是多少，
        # 最终压成：
        #
        # (batch, 256, 1)
        # ====================================================

        self.global_pool = (
            nn.AdaptiveAvgPool1d(1)
        )

        # ====================================================
        # 分类器
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


# ============================================================
# 模型测试
# ============================================================

if __name__ == "__main__":

    model = CNN1D(
        num_classes=6
    )

    # 模拟一个batch
    x = torch.randn(
        32,
        2,
        2048
    )

    y = model(x)

    print(
        "输入shape：",
        x.shape
    )

    print(
        "输出shape：",
        y.shape
    )

    # 应该：
    #
    # 输入：
    # torch.Size([32, 2, 2048])
    #
    # 输出：
    # torch.Size([32, 6])