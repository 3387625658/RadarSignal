from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 项目目录
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = (
    CURRENT_FILE
    .parent
    .parent
)


# ============================================================
# 2. 配置
# ============================================================

MODEL_NAMES = [
    "cnn1d",
    "cnn_lstm",
    "cnn2d",
    "resnet18"
]

DISPLAY_NAMES = {
    "cnn1d": "CNN1D",
    "cnn_lstm": "CNN-LSTM",
    "cnn2d": "2D-CNN",
    "resnet18": "ResNet18"
}


EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 读取测试预测并计算总准确率
# ============================================================

def read_test_accuracy(
    model_name
):

    path = (
        EVALUATION_DIR
        / model_name
        / "test_predictions.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"找不到：{path}\n"
            f"请先运行 {model_name} 的评价程序。"
        )

    total = 0
    correct = 0

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            total += 1

            if (
                row["true_label"]
                ==
                row["pred_label"]
            ):

                correct += 1

    return (
        correct / total
    )


# ============================================================
# 4. 读取SNR准确率
# ============================================================

def read_snr_accuracy(
    model_name
):

    path = (
        EVALUATION_DIR
        / model_name
        / "accuracy_vs_snr.csv"
    )

    snr_values = []
    accuracies = []

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            snr_values.append(
                int(
                    row["SNR_dB"]
                )
            )

            accuracies.append(
                float(
                    row["Accuracy"]
                )
            )

    return (
        np.array(snr_values),
        np.array(accuracies)
    )


# ============================================================
# 5. 模型总体准确率对比
# ============================================================

def plot_overall_accuracy():

    accuracies = []

    print()
    print("=" * 70)
    print("四模型测试准确率")
    print("=" * 70)

    for model_name in MODEL_NAMES:

        accuracy = read_test_accuracy(
            model_name
        )

        accuracies.append(
            accuracy
        )

        print(
            f"{DISPLAY_NAMES[model_name]:12s}"
            f" : "
            f"{accuracy * 100:.2f}%"
        )


    # --------------------------------------------------------
    # 保存CSV
    # --------------------------------------------------------

    path = (
        OUTPUT_DIR
        / "model_accuracy_comparison.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Model",
            "Test Accuracy"
        ])

        for model_name, accuracy in zip(
            MODEL_NAMES,
            accuracies
        ):

            writer.writerow([
                DISPLAY_NAMES[
                    model_name
                ],
                accuracy
            ])


    # --------------------------------------------------------
    # 柱状图
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        [
            DISPLAY_NAMES[x]
            for x in MODEL_NAMES
        ],
        np.array(accuracies) * 100
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Test Accuracy (%)"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_title(
        "Radar Signal Classification Model Comparison"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bar, acc in zip(
        bars,
        accuracies
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc * 100:.2f}%",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "model_accuracy_comparison.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 6. 四模型 SNR 曲线
# ============================================================

def plot_snr_comparison():

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for model_name in MODEL_NAMES:

        (
            snr_values,
            accuracies
        ) = read_snr_accuracy(
            model_name
        )

        ax.plot(
            snr_values,
            accuracies * 100,
            marker="o",
            label=DISPLAY_NAMES[
                model_name
            ]
        )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "Model Accuracy vs SNR"
    )

    ax.set_xticks(
        range(0, 11)
    )

    ax.set_ylim(
        0,
        105
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "model_accuracy_vs_snr.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 7. main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("四模型统一对比")
    print("=" * 70)

    plot_overall_accuracy()

    plot_snr_comparison()

    print()
    print("=" * 70)
    print("模型比较完成")
    print("=" * 70)

    print(
        f"\n保存位置："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()