import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN + BiLSTM 雷达信号分类模型

    输入：
        (batch, 2, 2048)

    输出：
        (batch, 6)
    """

    def __init__(
        self,
        num_classes=6,
        lstm_hidden_size=64
    ):

        super().__init__()

        # ====================================================
        # CNN特征提取
        # ====================================================

        self.cnn = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # (B, 2, 2048)
            # →
            # (B, 32, 1024)
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=9,
                padding=4
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),


            # ------------------------------------------------
            # Block 2
            #
            # (B,32,1024)
            # →
            # (B,64,512)
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),


            # ------------------------------------------------
            # Block 3
            #
            # (B,64,512)
            # →
            # (B,128,256)
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            )
        )


        # ====================================================
        # 双向LSTM
        #
        # CNN输出：
        # (B,128,256)
        #
        # 转置：
        # (B,256,128)
        #
        # 这里：
        #
        # sequence length = 256
        # features = 128
        # ====================================================

        self.lstm = nn.LSTM(

            input_size=128,

            hidden_size=lstm_hidden_size,

            num_layers=1,

            batch_first=True,

            bidirectional=True
        )


        # ====================================================
        # 分类层
        #
        # 双向LSTM：
        #
        # hidden × 2
        #
        # 64 × 2 = 128
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                lstm_hidden_size * 2,
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

        # ====================================================
        # CNN
        # ====================================================

        x = self.cnn(x)

        # 当前：
        #
        # (B,128,256)


        # ====================================================
        # Conv1D格式
        #
        # B,C,L
        #
        # 转为LSTM：
        #
        # B,L,C
        # ====================================================

        x = x.transpose(
            1,
            2
        )

        # 当前：
        #
        # (B,256,128)


        # ====================================================
        # LSTM
        # ====================================================

        lstm_output, _ = self.lstm(x)

        # lstm_output：
        #
        # (B,256,128)


        # ====================================================
        # 时间维平均池化
        # ====================================================

        x = torch.mean(
            lstm_output,
            dim=1
        )

        # 当前：
        #
        # (B,128)


        # ====================================================
        # 分类
        # ====================================================

        x = self.classifier(x)

        return x


# ============================================================
# 模型测试
# ============================================================

if __name__ == "__main__":

    model = CNNLSTM(
        num_classes=6
    )

    x = torch.randn(
        32,
        2,
        2048
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