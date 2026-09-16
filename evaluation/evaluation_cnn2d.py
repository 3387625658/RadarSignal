from pathlib import Path
import sys
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)


# ============================================================
# 1. 项目路径
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)


# ============================================================
# 2. 导入
# ============================================================

from dataset.radar_dataset import create_dataloaders
from models.cnn2d import CNN2D


# ============================================================
# 3. 配置
# ============================================================

NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_WORKERS = 0

CLASS_NAMES = [
    "LFM",
    "BPSK",
    "QPSK",
    "LFM_BPSK",
    "LFM_QPSK",
    "2FSK_BPSK"
]


# ============================================================
# 4. 文件路径
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "cnn2d_best.pth"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "cnn2d_history.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "cnn2d"
)

OUTPUT_DIR.mkdir(
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
# 6. 加载模型
# ============================================================

def load_model():

    model = CNN2D(
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"最佳 Epoch：{checkpoint['epoch']}"
    )

    print(
        f"最佳验证准确率："
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    return model


# ============================================================
# 7. 获取预测结果
# ============================================================

def get_predictions(
    model,
    dataloader
):

    labels_all = []
    predictions_all = []
    probabilities_all = []
    snr_all = []

    with torch.no_grad():

        for (
            signals,
            labels,
            metadata
        ) in dataloader:

            signals = signals.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                signals
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            labels_all.extend(
                labels.cpu().numpy()
            )

            predictions_all.extend(
                predictions.cpu().numpy()
            )

            probabilities_all.extend(
                probabilities.cpu().numpy()
            )

            snr_all.extend(
                metadata["snr"]
                .cpu()
                .numpy()
            )

    return (
        np.array(labels_all),
        np.array(predictions_all),
        np.array(probabilities_all),
        np.array(snr_all)
    )


# ============================================================
# 8. 分类指标
# ============================================================

def evaluate_metrics(
    labels,
    predictions
):

    accuracy = np.mean(
        labels == predictions
    )

    print()
    print("=" * 70)
    print("2D-CNN 测试结果")
    print("=" * 70)

    print(
        f"\nTest Accuracy："
        f"{accuracy * 100:.2f}%"
    )

    print()

    print(
        classification_report(
            labels,
            predictions,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    path = (
        OUTPUT_DIR
        / "classification_metrics.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Class",
            "Precision",
            "Recall",
            "F1",
            "Support"
        ])

        for i in range(NUM_CLASSES):

            writer.writerow([
                CLASS_NAMES[i],
                precision[i],
                recall[i],
                f1[i],
                int(support[i])
            ])


# ============================================================
# 9. 混淆矩阵
# ============================================================

def plot_confusion_matrix(
    labels,
    predictions
):

    cm = confusion_matrix(
        labels,
        predictions
    )

    # 普通混淆矩阵
    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(cm)

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_xticks(
        range(NUM_CLASSES)
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(
        "2D-CNN Confusion Matrix"
    )

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


    # ========================================================
    # 归一化混淆矩阵
    # ========================================================

    normalized = (
        cm.astype(np.float64)
        /
        cm.sum(
            axis=1,
            keepdims=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(
        normalized
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_xticks(
        range(NUM_CLASSES)
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(
        "2D-CNN Normalized Confusion Matrix"
    )

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            ax.text(
                j,
                i,
                f"{normalized[i, j] * 100:.1f}%",
                ha="center",
                va="center"
            )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "confusion_matrix_normalized.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 10. Accuracy-SNR
# ============================================================

def plot_accuracy_vs_snr(
    labels,
    predictions,
    snr
):

    snr_values = np.sort(
        np.unique(snr)
    )

    accuracies = []

    print()
    print("=" * 70)
    print("2D-CNN 不同SNR识别准确率")
    print("=" * 70)

    print()

    for value in snr_values:

        mask = (
            snr == value
        )

        accuracy = np.mean(
            labels[mask]
            ==
            predictions[mask]
        )

        accuracies.append(
            accuracy
        )

        print(
            f"SNR = {value:2d} dB"
            f"    Accuracy = "
            f"{accuracy * 100:.2f}%"
        )


    # 保存CSV
    csv_path = (
        OUTPUT_DIR
        / "accuracy_vs_snr.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "SNR_dB",
            "Accuracy"
        ])

        for value, acc in zip(
            snr_values,
            accuracies
        ):

            writer.writerow([
                int(value),
                float(acc)
            ])


    # 绘图
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        snr_values,
        np.array(accuracies) * 100,
        marker="o"
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "2D-CNN Accuracy vs SNR"
    )

    ax.set_xticks(
        snr_values
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "accuracy_vs_snr.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 11. 类别 × SNR
# ============================================================

def plot_class_snr(
    labels,
    predictions,
    snr
):

    snr_values = np.sort(
        np.unique(snr)
    )

    matrix = np.zeros(
        (
            NUM_CLASSES,
            len(snr_values)
        )
    )

    for class_id in range(
        NUM_CLASSES
    ):

        for j, snr_value in enumerate(
            snr_values
        ):

            mask = (
                (labels == class_id)
                &
                (snr == snr_value)
            )

            if np.sum(mask) > 0:

                matrix[
                    class_id,
                    j
                ] = np.mean(
                    predictions[mask]
                    ==
                    class_id
                )


    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    image = ax.imshow(
        matrix * 100,
        aspect="auto"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Accuracy (%)"
    )

    ax.set_xticks(
        range(
            len(snr_values)
        )
    )

    ax.set_xticklabels(
        snr_values
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Signal class"
    )

    ax.set_title(
        "2D-CNN Accuracy by Class and SNR"
    )

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            len(snr_values)
        ):

            ax.text(
                j,
                i,
                f"{matrix[i, j] * 100:.0f}",
                ha="center",
                va="center"
            )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "class_snr_accuracy.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 12. 训练历史
# ============================================================

def plot_history():

    history = torch.load(
        HISTORY_PATH,
        map_location="cpu"
    )

    epochs = np.arange(
        1,
        len(
            history["train_loss"]
        ) + 1
    )


    # Loss
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        epochs,
        history["train_loss"],
        label="Train"
    )

    ax.plot(
        epochs,
        history["val_loss"],
        label="Validation"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        "2D-CNN Loss"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "loss_curve.png",
        dpi=200
    )

    plt.show()


    # Accuracy
    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        epochs,
        np.array(
            history["train_accuracy"]
        ) * 100,
        label="Train"
    )

    ax.plot(
        epochs,
        np.array(
            history["val_accuracy"]
        ) * 100,
        label="Validation"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_title(
        "2D-CNN Accuracy"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "accuracy_curve.png",
        dpi=200
    )

    plt.show()


# ============================================================
# 13. 保存逐样本预测
# ============================================================

def save_predictions(
    labels,
    predictions,
    probabilities,
    snr
):

    path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "sample",
            "true_label",
            "true_class",
            "pred_label",
            "pred_class",
            "snr_db",
            "confidence"
        ])

        for i in range(
            len(labels)
        ):

            true_id = int(
                labels[i]
            )

            pred_id = int(
                predictions[i]
            )

            confidence = float(
                probabilities[
                    i,
                    pred_id
                ]
            )

            writer.writerow([
                i,
                true_id,
                CLASS_NAMES[true_id],
                pred_id,
                CLASS_NAMES[pred_id],
                int(snr[i]),
                confidence
            ])


# ============================================================
# 14. main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("2D-CNN 模型评价")
    print("=" * 70)

    print(
        f"\n设备：{device}"
    )

    model = load_model()

    (
        _,
        _,
        test_loader
    ) = create_dataloaders(

        # 注意这里一定是stft
        mode="stft",

        batch_size=BATCH_SIZE,

        num_workers=NUM_WORKERS,

        normalize=True,

        return_metadata=True
    )

    (
        labels,
        predictions,
        probabilities,
        snr
    ) = get_predictions(
        model,
        test_loader
    )

    evaluate_metrics(
        labels,
        predictions
    )

    plot_confusion_matrix(
        labels,
        predictions
    )

    plot_accuracy_vs_snr(
        labels,
        predictions,
        snr
    )

    plot_class_snr(
        labels,
        predictions,
        snr
    )

    plot_history()

    save_predictions(
        labels,
        predictions,
        probabilities,
        snr
    )

    print()
    print("=" * 70)
    print("2D-CNN评价完成")
    print("=" * 70)

    print(
        f"\n保存位置：{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()