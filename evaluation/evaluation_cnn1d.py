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

PROJECT_ROOT = (
    CURRENT_FILE
    .parent
    .parent
)

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

MODEL_NAME = "cnn1d"

NUM_CLASSES = 6

BATCH_SIZE = 64

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
    / "cnn1d_best.pth"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "cnn1d_history.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "cnn1d"
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

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"找不到模型：\n{MODEL_PATH}\n"
            f"请先运行 train.py"
        )

    model = CNN1D(
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
        f"模型加载完成：{MODEL_PATH}"
    )

    print(
        f"训练时最佳 Epoch："
        f"{checkpoint['epoch']}"
    )

    print(
        f"训练时最佳验证准确率："
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    return model


# ============================================================
# 7. 获得测试集全部预测结果
# ============================================================

def get_predictions(
    model,
    test_loader
):

    all_labels = []

    all_predictions = []

    all_probabilities = []

    all_snr = []


    with torch.no_grad():

        for (
            signals,
            labels,
            metadata
        ) in test_loader:

            signals = signals.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                signals
            )

            probabilities = (
                torch.softmax(
                    outputs,
                    dim=1
                )
            )

            predictions = (
                torch.argmax(
                    outputs,
                    dim=1
                )
            )

            all_labels.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
                .tolist()
            )

            all_probabilities.extend(
                probabilities
                .cpu()
                .numpy()
                .tolist()
            )

            all_snr.extend(
                metadata["snr"]
                .cpu()
                .numpy()
                .tolist()
            )


    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
        np.array(all_snr)
    )


# ============================================================
# 8. 总体Accuracy
# ============================================================

def calculate_accuracy(
    labels,
    predictions
):

    accuracy = np.mean(
        labels == predictions
    )

    print()
    print("=" * 70)
    print("总体测试结果")
    print("=" * 70)

    print(
        f"\nTest Accuracy："
        f"{accuracy * 100:.2f}%"
    )

    return accuracy


# ============================================================
# 9. Precision / Recall / F1
# ============================================================

def evaluate_classification_metrics(
    labels,
    predictions
):

    print()
    print("=" * 70)
    print("分类指标")
    print("=" * 70)

    report = classification_report(

        labels,

        predictions,

        target_names=CLASS_NAMES,

        digits=4,

        zero_division=0
    )

    print()
    print(report)

    (
        precision,
        recall,
        f1,
        support
    ) = precision_recall_fscore_support(

        labels,

        predictions,

        labels=np.arange(
            NUM_CLASSES
        ),

        zero_division=0
    )

    csv_path = (
        OUTPUT_DIR
        / "classification_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Class",
                "Precision",
                "Recall",
                "F1",
                "Support"
            ]
        )

        for i in range(
            NUM_CLASSES
        ):

            writer.writerow(
                [
                    CLASS_NAMES[i],
                    precision[i],
                    recall[i],
                    f1[i],
                    int(support[i])
                ]
            )

    print(
        f"\n分类指标已保存："
        f"{csv_path}"
    )


# ============================================================
# 10. 混淆矩阵
# ============================================================

def plot_confusion_matrix(
    labels,
    predictions
):

    cm = confusion_matrix(

        labels,

        predictions,

        labels=np.arange(
            NUM_CLASSES
        )
    )

    print()
    print("=" * 70)
    print("混淆矩阵")
    print("=" * 70)

    print()
    print(cm)


    # --------------------------------------------------------
    # 原始数量
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(
        cm
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_xticks(
        np.arange(
            NUM_CLASSES
        )
    )

    ax.set_yticks(
        np.arange(
            NUM_CLASSES
        )
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
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_title(
        "CNN1D Confusion Matrix"
    )

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "confusion_matrix.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


    # --------------------------------------------------------
    # 归一化混淆矩阵
    # --------------------------------------------------------

    cm_normalized = (

        cm.astype(
            np.float64
        )

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
        cm_normalized
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_xticks(
        np.arange(
            NUM_CLASSES
        )
    )

    ax.set_yticks(
        np.arange(
            NUM_CLASSES
        )
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
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_title(
        "CNN1D Normalized Confusion Matrix"
    )


    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            ax.text(
                j,
                i,
                f"{cm_normalized[i, j] * 100:.1f}%",
                ha="center",
                va="center"
            )


    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "confusion_matrix_normalized.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 11. 不同SNR下的Accuracy
# ============================================================

def plot_accuracy_vs_snr(
    labels,
    predictions,
    snr_array
):

    unique_snr = np.sort(
        np.unique(
            snr_array
        )
    )

    accuracies = []


    print()
    print("=" * 70)
    print("不同SNR下识别准确率")
    print("=" * 70)

    print()


    for snr in unique_snr:

        mask = (
            snr_array == snr
        )

        snr_accuracy = np.mean(

            labels[mask]

            ==

            predictions[mask]
        )

        accuracies.append(
            snr_accuracy
        )

        print(
            f"SNR = {snr:2d} dB"
            f"    "
            f"Accuracy = "
            f"{snr_accuracy * 100:.2f}%"
        )


    # --------------------------------------------------------
    # 保存CSV
    # --------------------------------------------------------

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

        writer.writerow(
            [
                "SNR_dB",
                "Accuracy"
            ]
        )

        for snr, acc in zip(
            unique_snr,
            accuracies
        ):

            writer.writerow(
                [
                    int(snr),
                    float(acc)
                ]
            )


    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        unique_snr,
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
        "CNN1D Accuracy vs SNR"
    )

    ax.set_xticks(
        unique_snr
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "accuracy_vs_snr.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 12. 每类信号在不同SNR下准确率
# ============================================================

def plot_class_snr_heatmap(
    labels,
    predictions,
    snr_array
):

    unique_snr = np.sort(
        np.unique(
            snr_array
        )
    )

    matrix = np.zeros(

        (
            NUM_CLASSES,
            len(unique_snr)
        ),

        dtype=np.float64
    )


    for class_id in range(
        NUM_CLASSES
    ):

        for j, snr in enumerate(
            unique_snr
        ):

            mask = (

                (labels == class_id)

                &

                (snr_array == snr)
            )

            if np.sum(mask) > 0:

                matrix[
                    class_id,
                    j
                ] = np.mean(

                    predictions[mask]

                    == class_id
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
        np.arange(
            len(unique_snr)
        )
    )

    ax.set_xticklabels(
        unique_snr
    )

    ax.set_yticks(
        np.arange(
            NUM_CLASSES
        )
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
        "CNN1D Accuracy by Class and SNR"
    )


    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            len(unique_snr)
        ):

            ax.text(

                j,

                i,

                f"{matrix[i, j] * 100:.0f}",

                ha="center",

                va="center"
            )


    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "class_snr_accuracy.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 13. 训练Loss曲线
# ============================================================

def plot_training_history():

    if not HISTORY_PATH.exists():

        print(
            "\n未找到训练历史："
            f"{HISTORY_PATH}"
        )

        return


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


    # ========================================================
    # Loss
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    ax.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        "CNN1D Training and Validation Loss"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "loss_curve.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


    # ========================================================
    # Accuracy
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(

        epochs,

        np.array(
            history["train_accuracy"]
        ) * 100,

        label="Train Accuracy"
    )

    ax.plot(

        epochs,

        np.array(
            history["val_accuracy"]
        ) * 100,

        label="Validation Accuracy"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "CNN1D Training and Validation Accuracy"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "accuracy_curve.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 14. 保存逐样本预测结果
# ============================================================

def save_predictions(
    labels,
    predictions,
    probabilities,
    snr_array
):

    output_path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        header = [

            "sample",

            "true_label",

            "true_class",

            "pred_label",

            "pred_class",

            "snr_db",

            "confidence"
        ]

        writer.writerow(
            header
        )


        for i in range(
            len(labels)
        ):

            pred = int(
                predictions[i]
            )

            true = int(
                labels[i]
            )

            confidence = float(
                probabilities[
                    i,
                    pred
                ]
            )

            writer.writerow(
                [
                    i,
                    true,
                    CLASS_NAMES[true],
                    pred,
                    CLASS_NAMES[pred],
                    int(snr_array[i]),
                    confidence
                ]
            )


    print(
        f"\n逐样本预测结果已保存："
        f"{output_path}"
    )


# ============================================================
# 15. main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "CNN1D 模型评价"
    )

    print("=" * 70)

    print(
        f"\n运行设备：{device}"
    )


    # ========================================================
    # 模型
    # ========================================================

    model = load_model()


    # ========================================================
    # Test DataLoader
    #
    # 注意：
    #
    # return_metadata=True
    #
    # 因为我们需要读取SNR
    # ========================================================

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(

        mode="iq",

        batch_size=BATCH_SIZE,

        num_workers=NUM_WORKERS,

        normalize=True,

        return_metadata=True
    )


    # ========================================================
    # 预测
    # ========================================================

    (
        labels,
        predictions,
        probabilities,
        snr_array
    ) = get_predictions(

        model,

        test_loader
    )


    # ========================================================
    # Accuracy
    # ========================================================

    calculate_accuracy(

        labels,

        predictions
    )


    # ========================================================
    # Precision / Recall / F1
    # ========================================================

    evaluate_classification_metrics(

        labels,

        predictions
    )


    # ========================================================
    # 混淆矩阵
    # ========================================================

    plot_confusion_matrix(

        labels,

        predictions
    )


    # ========================================================
    # Accuracy-SNR
    # ========================================================

    plot_accuracy_vs_snr(

        labels,

        predictions,

        snr_array
    )


    # ========================================================
    # 每类 × SNR
    # ========================================================

    plot_class_snr_heatmap(

        labels,

        predictions,

        snr_array
    )


    # ========================================================
    # 训练曲线
    # ========================================================

    plot_training_history()


    # ========================================================
    # 保存所有预测结果
    # ========================================================

    save_predictions(

        labels,

        predictions,

        probabilities,

        snr_array
    )


    print()
    print("=" * 70)

    print(
        "CNN1D评价完成"
    )

    print("=" * 70)

    print(
        f"\n结果保存在："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()