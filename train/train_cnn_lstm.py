from pathlib import Path
import sys
import time

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# 1. 项目路径
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = (
    CURRENT_FILE
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)


# ============================================================
# 2. 导入
# ============================================================

from dataset.radar_dataset import create_dataloaders
from models.cnn_lstm import CNNLSTM


# ============================================================
# 3. 配置
# ============================================================

MODEL_NAME = "cnn_lstm"

NUM_CLASSES = 6

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 1e-3

NUM_WORKERS = 0


# ============================================================
# 4. 保存路径
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

TRAINING_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "training"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAINING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    MODEL_DIR
    / "cnn_lstm_best.pth"
)

HISTORY_PATH = (
    TRAINING_DIR
    / "cnn_lstm_history.pt"
)


# ============================================================
# 5. 设备
# ============================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


# ============================================================
# 6. 训练一个Epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0


    for signals, labels in dataloader:

        signals = signals.to(
            device
        )

        labels = labels.to(
            device
        )


        # 梯度清零
        optimizer.zero_grad()


        # 前向传播
        outputs = model(
            signals
        )


        # Loss
        loss = criterion(
            outputs,
            labels
        )


        # 反向传播
        loss.backward()


        # 更新
        optimizer.step()


        batch_size = (
            labels.size(0)
        )


        total_loss += (
            loss.item()
            * batch_size
        )


        predictions = torch.argmax(

            outputs,

            dim=1
        )


        total_correct += (

            predictions
            == labels

        ).sum().item()


        total_samples += (
            batch_size
        )


    average_loss = (

        total_loss

        / total_samples
    )


    accuracy = (

        total_correct

        / total_samples
    )


    return (
        average_loss,
        accuracy
    )


# ============================================================
# 7. 验证/测试
# ============================================================

def evaluate(
    model,
    dataloader,
    criterion
):

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0


    with torch.no_grad():

        for signals, labels in dataloader:

            signals = signals.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                signals
            )


            loss = criterion(
                outputs,
                labels
            )


            batch_size = (
                labels.size(0)
            )


            total_loss += (
                loss.item()
                * batch_size
            )


            predictions = torch.argmax(

                outputs,

                dim=1
            )


            total_correct += (

                predictions
                == labels

            ).sum().item()


            total_samples += (
                batch_size
            )


    average_loss = (

        total_loss

        / total_samples
    )


    accuracy = (

        total_correct

        / total_samples
    )


    return (
        average_loss,
        accuracy
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "雷达信号识别 - CNN-LSTM"
    )

    print("=" * 70)

    print(
        f"\n当前设备：{device}"
    )


    if torch.cuda.is_available():

        print(
            "GPU：",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "未检测到CUDA，将使用CPU"
        )


    # ========================================================
    # Dataset
    # ========================================================

    print()
    print("=" * 70)

    print(
        "加载数据库"
    )

    print("=" * 70)


    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(

        mode="iq",

        batch_size=BATCH_SIZE,

        num_workers=NUM_WORKERS,

        normalize=True,

        return_metadata=False
    )


    # ========================================================
    # Model
    # ========================================================

    model = CNNLSTM(

        num_classes=NUM_CLASSES,

        lstm_hidden_size=64

    ).to(device)


    print()
    print("=" * 70)

    print(
        "CNN-LSTM模型"
    )

    print("=" * 70)

    print(model)


    # ========================================================
    # 参数量
    # ========================================================

    total_params = sum(

        p.numel()

        for p in model.parameters()
    )


    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )


    print()

    print(
        f"总参数量："
        f"{total_params:,}"
    )

    print(
        f"可训练参数量："
        f"{trainable_params:,}"
    )


    # ========================================================
    # Loss
    # ========================================================

    criterion = (
        nn.CrossEntropyLoss()
    )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )


    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3
    )


    # ========================================================
    # History
    # ========================================================

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "val_loss": [],

        "val_accuracy": []
    }


    best_val_accuracy = 0.0


    print()
    print("=" * 70)

    print(
        "开始训练 CNN-LSTM"
    )

    print("=" * 70)


    # ========================================================
    # Epoch
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        start_time = time.time()


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy
        ) = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy
        ) = evaluate(

            model,

            val_loader,

            criterion
        )


        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_loss
        )


        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        elapsed = (
            time.time()
            - start_time
        )


        print(

            f"Epoch "
            f"[{epoch:02d}/{EPOCHS}] "

            f"| "

            f"Train Loss: "
            f"{train_loss:.4f} "

            f"| "

            f"Train Acc: "
            f"{train_accuracy * 100:.2f}% "

            f"| "

            f"Val Loss: "
            f"{val_loss:.4f} "

            f"| "

            f"Val Acc: "
            f"{val_accuracy * 100:.2f}% "

            f"| "

            f"LR: "
            f"{current_lr:.6f} "

            f"| "

            f"{elapsed:.1f}s"
        )


        # ====================================================
        # 保存Best
        # ====================================================

        if (
            val_accuracy
            >
            best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )


            torch.save(

                {

                    "model_state_dict":
                        model.state_dict(),

                    "val_accuracy":
                        best_val_accuracy,

                    "epoch":
                        epoch
                },

                MODEL_PATH
            )


            print(

                f"    → 保存新的最优模型 "

                f"Val Acc = "

                f"{best_val_accuracy * 100:.2f}%"
            )


    # ========================================================
    # 保存History
    # ========================================================

    torch.save(

        history,

        HISTORY_PATH
    )


    print()

    print(
        f"训练历史保存："
        f"{HISTORY_PATH}"
    )


    # ========================================================
    # 加载最优模型
    # ========================================================

    print()
    print("=" * 70)

    print(
        "加载最佳 CNN-LSTM"
    )

    print("=" * 70)


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=device
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    print(
        f"最佳 Epoch："
        f"{checkpoint['epoch']}"
    )


    print(
        f"最佳验证准确率："
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )


    # ========================================================
    # Test
    # ========================================================

    (
        test_loss,
        test_accuracy
    ) = evaluate(

        model,

        test_loader,

        criterion
    )


    print()

    print(
        f"Test Loss："
        f"{test_loss:.4f}"
    )


    print(
        f"Test Accuracy："
        f"{test_accuracy * 100:.2f}%"
    )


    print()
    print("=" * 70)

    print(
        "CNN-LSTM训练完成"
    )

    print("=" * 70)


    print(
        f"\n模型保存："
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":

    main()