from pathlib import Path
import sys

import numpy as np


# ============================================================
# 1. 路径
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


from pipeline.radar_pipeline import (
    load_classifier,
    analyze_signal
)


# ============================================================
# 2. 数据路径
# ============================================================

DATA_PATH = (

    PROJECT_ROOT

    / "data"

    / "test.npz"
)


# ============================================================
# 3. 美化打印
# ============================================================

def print_value(
    key,
    value
):

    if isinstance(
        value,
        np.ndarray
    ):

        print(
            f"{key}: "
            f"{value.tolist()}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# 4. main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "雷达脉内复合信号智能识别与参数测量系统"
    )

    print("=" * 70)


    # ========================================================
    # 加载模型
    # ========================================================

    model = load_classifier()


    print(
        "\nResNet18加载完成"
    )


    # ========================================================
    # 加载测试数据
    # ========================================================

    data = np.load(
        DATA_PATH,
        allow_pickle=False
    )


    X = data["X"]

    y = data["y"]

    snr = data["snr"]

    fs = float(
        data["fs"]
    )

    class_names = (
        data[
            "class_names"
        ].tolist()
    )


    # ========================================================
    # 随机抽一条测试信号
    # ========================================================

    rng = np.random.default_rng()

    index = int(

        rng.integers(
            0,
            len(X)
        )
    )


    signal = (

        X[index, 0]

        +

        1j
        * X[index, 1]
    )


    true_class = (
        class_names[
            int(
                y[index]
            )
        ]
    )


    print()
    print("=" * 70)

    print(
        f"测试样本编号：{index}"
    )

    print(
        f"真实类型：{true_class}"
    )

    print(
        f"SNR：{snr[index]} dB"
    )

    print("=" * 70)


    # ========================================================
    # 完整分析
    # ========================================================

    result = analyze_signal(

        signal,

        fs,

        model
    )


    classification = (
        result[
            "classification"
        ]
    )


    parameters = (
        result[
            "parameters"
        ]
    )


    # ========================================================
    # 识别结果
    # ========================================================

    print()
    print("=" * 70)

    print(
        "信号识别结果"
    )

    print("=" * 70)


    print(

        f"\n预测类型："

        f"{classification['class_name']}"
    )


    print(

        f"置信度："

        f"{classification['confidence'] * 100:.2f}%"
    )


    print(
        "\n各类别概率："
    )


    for class_name, probability in (

        classification[
            "probabilities"
        ].items()
    ):

        print(

            f"  {class_name:12s}: "

            f"{probability * 100:.3f}%"
        )


    # ========================================================
    # 参数测量
    # ========================================================

    print()
    print("=" * 70)

    print(
        "参数测量结果"
    )

    print("=" * 70)


    for key, value in (
        parameters.items()
    ):

        # dechirped_signal太长
        # 不在终端打印
        if (
            key
            ==
            "dechirped_signal"
        ):

            continue


        # candidate_scores单独处理
        if (
            key
            ==
            "candidate_scores"
        ):

            print(
                f"{key}: {value}"
            )

            continue


        print_value(
            key,
            value
        )


    # ========================================================
    # 对比真实类别
    # ========================================================

    correct = (

        classification[
            "class_name"
        ]

        ==
        true_class
    )


    print()
    print("=" * 70)


    print(

        "识别是否正确：",

        "✓ 正确"
        if correct
        else "✗ 错误"
    )


    print("=" * 70)


    data.close()


if __name__ == "__main__":

    main()