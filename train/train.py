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

# 让Python能够找到项目模块
sys.path.append(
    str(PROJECT_ROOT)
)


# ============================================================
# 2. 导入自己的代码
# ============================================================

from dataset.radar_dataset import create_dataloaders
from models.cnn1d import CNN1D


# ============================================================
# 3. 配置
# ============================================================

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 1e-3

NUM_CLASSES = 6

NUM_WORKERS = 0

MODEL_NAME = "cnn1d"


# ============================================================
# 4. 输出目录
# ============================================================

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "training"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# 6. 单个epoch训练
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


    for batch_index, (
        signals,
        labels
    ) in enumerate(dataloader):

        # ----------------------------------------------------
        # 放到GPU / CPU
        # ----------------------------------------------------

        signals = signals.to(
            device
        )

        labels = labels.to(
            device
        )

        # ----------------------------------------------------
        # 梯度清零
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # 前向传播
        # ----------------------------------------------------

        outputs = model(
            signals
        )

        # ----------------------------------------------------
        # 计算Loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # 反向传播
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # 更新参数
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # 累加Loss
        # ----------------------------------------------------

        batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        # ----------------------------------------------------
        # 分类结果
        # ----------------------------------------------------

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
# 7. 验证
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
# 8. 保存训练历史
# ============================================================

def save_history(
    history
):

    output_path = (
        RESULT_DIR
        / f"{MODEL_NAME}_history.pt"
    )

    torch.save(
        history,
        output_path
    )

    print(
        f"\n训练历史已保存："
        f"{output_path}"
    )


# ============================================================
# 9. 主训练函数
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "雷达脉内信号识别 - 1D CNN训练"
    )

    print("=" * 70)

    print(
        f"\n当前设备：{device}"
    )

    # --------------------------------------------------------
    # GPU信息
    # --------------------------------------------------------

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
    # 创建DataLoader
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
    # 创建模型
    # ========================================================

    model = CNN1D(

        num_classes=NUM_CLASSES

    ).to(device)

    print()
    print("=" * 70)

    print(
        "模型结构"
    )

    print("=" * 70)

    print(model)

    # --------------------------------------------------------
    # 参数量
    # --------------------------------------------------------

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
        f"可训练参数："
        f"{trainable_params:,}"
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = (
        nn.CrossEntropyLoss()
    )

    # ========================================================
    # 优化器
    # ========================================================

    optimizer = Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )

    # ========================================================
    # 学习率调度器
    # ========================================================

    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.5,

        patience=3
    )

    # ========================================================
    # 训练记录
    # ========================================================

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "val_loss": [],

        "val_accuracy": []
    }

    # ========================================================
    # 最优验证准确率
    # ========================================================

    best_val_accuracy = 0.0

    best_model_path = (

        MODEL_DIR
        / f"{MODEL_NAME}_best.pth"
    )


    print()
    print("=" * 70)

    print(
        "开始训练"
    )

    print("=" * 70)


    # ========================================================
    # Epoch循环
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
        # 调整学习率
        # ----------------------------------------------------

        scheduler.step(
            val_loss
        )

        # ----------------------------------------------------
        # 保存history
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

        # ----------------------------------------------------
        # 当前学习率
        # ----------------------------------------------------

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        elapsed_time = (
            time.time()
            - start_time
        )

        # ----------------------------------------------------
        # 输出
        # ----------------------------------------------------

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

            f"{elapsed_time:.1f}s"
        )

        # ====================================================
        # 保存最优模型
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

                best_model_path
            )

            print(
                f"    → 保存新的最优模型 "
                f"Val Acc = "
                f"{best_val_accuracy * 100:.2f}%"
            )


    # ========================================================
    # 保存训练历史
    # ========================================================

    save_history(
        history
    )


    # ========================================================
    # 载入最佳模型
    # ========================================================

    print()
    print("=" * 70)

    print(
        "加载最佳模型进行测试"
    )

    print("=" * 70)

    checkpoint = torch.load(

        best_model_path,

        map_location=device
    )

    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"最佳Epoch："
        f"{checkpoint['epoch']}"
    )

    print(
        f"最佳验证准确率："
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )


    # ========================================================
    # Test
    # ========================================================

    test_loss, test_accuracy = evaluate(

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
        "训练完成"
    )

    print("=" * 70)

    print(
        f"\n最佳模型保存："
        f"{best_model_path}"
    )


# ============================================================
# 10. 程序入口
# ============================================================

if __name__ == "__main__":

    main()