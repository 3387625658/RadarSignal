from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 项目路径
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent


FULL_PIPELINE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "full_pipeline"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_summary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 四模型最终测试准确率
# ============================================================

MODEL_NAMES = [
    "CNN1D",
    "CNN-LSTM",
    "2D-CNN",
    "ResNet18"
]

MODEL_ACCURACIES = [
    64.39,
    80.98,
    89.31,
    99.58
]


# ============================================================
# 读取CSV
# ============================================================

def read_csv(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


# ============================================================
# 1. 四模型准确率
# ============================================================

def plot_model_comparison():

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        MODEL_NAMES,
        MODEL_ACCURACIES
    )

    ax.set_ylabel(
        "Test Accuracy (%)"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_title(
        "Classification Accuracy of Different Models"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bar, accuracy in zip(
        bars,
        MODEL_ACCURACIES
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{accuracy:.2f}%",
            ha="center"
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "model_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 2. 六类分类准确率
# ============================================================

def plot_class_accuracy(
    summary_rows
):

    class_names = []

    accuracies = []


    for row in summary_rows:

        class_names.append(
            row["class"]
        )

        accuracies.append(

            float(
                row[
                    "classification_accuracy"
                ]
            )
            * 100
        )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        class_names,
        accuracies
    )

    ax.set_ylabel(
        "Classification Accuracy (%)"
    )

    ax.set_xlabel(
        "Signal Class"
    )

    ax.set_ylim(
        90,
        101
    )

    ax.set_title(
        "ResNet18 Classification Accuracy by Signal Class"
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )


    for bar, accuracy in zip(
        bars,
        accuracies
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{accuracy:.2f}%",
            ha="center"
        )


    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "class_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 3. 码参数恢复率
# ============================================================

def plot_code_recovery(
    summary_rows
):

    classes = []

    symbol_count_accuracy = []

    code_accuracy = []


    for row in summary_rows:

        symbol_value = row[
            "symbol_count_accuracy"
        ]

        code_value = row[
            "code_accuracy"
        ]


        if (
            symbol_value != ""
            and
            code_value != ""
            and
            symbol_value.lower()
            != "nan"
            and
            code_value.lower()
            != "nan"
        ):

            classes.append(
                row["class"]
            )

            symbol_count_accuracy.append(

                float(
                    symbol_value
                )
                * 100
            )

            code_accuracy.append(

                float(
                    code_value
                )
                * 100
            )


    x = np.arange(
        len(classes)
    )

    width = 0.36


    fig, ax = plt.subplots(
        figsize=(11, 6)
    )


    bars1 = ax.bar(

        x - width / 2,

        symbol_count_accuracy,

        width,

        label="Symbol Count Accuracy"
    )


    bars2 = ax.bar(

        x + width / 2,

        code_accuracy,

        width,

        label="Code Recovery Accuracy"
    )


    ax.set_xticks(x)

    ax.set_xticklabels(
        classes,
        rotation=25
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_title(
        "Symbol Parameter Recovery Performance"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )


    for bars in (
        bars1,
        bars2
    ):

        for bar in bars:

            height = (
                bar.get_height()
            )

            ax.text(

                bar.get_x()
                +
                bar.get_width() / 2,

                height + 1,

                f"{height:.1f}",

                ha="center",

                fontsize=8
            )


    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "code_parameter_recovery.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 4. 整理最终论文表
# ============================================================

def save_final_table(
    summary_rows
):

    output_path = (
        OUTPUT_DIR
        / "final_experiment_summary.csv"
    )


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)


        writer.writerow([
            "Signal Class",
            "Classification Accuracy (%)",
            "Center Frequency MAE (kHz)",
            "Bandwidth MAPE (%)",
            "Chirp Rate MAPE (%)",
            "Symbol Count Accuracy (%)",
            "Code Recovery Accuracy (%)",
            "Frequency Separation MAPE (%)",
            "FSK Sequence Accuracy (%)"
        ])


        for row in summary_rows:

            def percent_value(key):

                value = row[key]

                if (
                    value == ""
                    or
                    value.lower() == "nan"
                ):

                    return ""

                return (
                    float(value)
                    * 100
                )


            def normal_value(key):

                value = row[key]

                if (
                    value == ""
                    or
                    value.lower() == "nan"
                ):

                    return ""

                return float(value)


            fc_value = row[
                "fc_mae_hz"
            ]

            if (
                fc_value == ""
                or
                fc_value.lower() == "nan"
            ):

                fc_khz = ""

            else:

                fc_khz = (
                    float(fc_value)
                    / 1000
                )


            writer.writerow([

                row["class"],

                percent_value(
                    "classification_accuracy"
                ),

                fc_khz,

                normal_value(
                    "bandwidth_mape_percent"
                ),

                normal_value(
                    "chirp_rate_mape_percent"
                ),

                percent_value(
                    "symbol_count_accuracy"
                ),

                percent_value(
                    "code_accuracy"
                ),

                normal_value(
                    "freq_sep_mape_percent"
                ),

                percent_value(
                    "fsk_code_accuracy"
                )
            ])


# ============================================================
# main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "最终实验结果汇总"
    )

    print("=" * 70)


    summary_path = (
        FULL_PIPELINE_DIR
        / "class_summary.csv"
    )


    if not summary_path.exists():

        raise FileNotFoundError(
            f"找不到：{summary_path}"
        )


    summary_rows = read_csv(
        summary_path
    )


    print(
        "\n1. 绘制四模型准确率"
    )

    plot_model_comparison()


    print(
        "2. 绘制六类识别准确率"
    )

    plot_class_accuracy(
        summary_rows
    )


    print(
        "3. 绘制码参数恢复性能"
    )

    plot_code_recovery(
        summary_rows
    )


    print(
        "4. 生成最终论文数据表"
    )

    save_final_table(
        summary_rows
    )


    print()
    print("=" * 70)

    print(
        "最终实验汇总完成"
    )

    print("=" * 70)


    print(
        f"\n保存目录："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()