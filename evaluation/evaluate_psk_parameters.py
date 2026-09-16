from pathlib import Path
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt


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


from estimation.psk_estimator import (
    estimate_psk_parameters
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "test.npz"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "parameter_evaluation"
    / "psk"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 码序列匹配
# ============================================================

def aligned_code_accuracy(
    estimated_code,
    true_code,
    modulation_order
):
    """
    PSK存在整体相位模糊。

    BPSK允许：
        整体 +0 / +1

    QPSK允许：
        整体 +0 / +1 / +2 / +3

    取最优循环偏移后的准确率。
    """

    estimated_code = np.asarray(
        estimated_code,
        dtype=np.int64
    )

    true_code = np.asarray(
        true_code,
        dtype=np.int64
    )


    if (
        len(estimated_code)
        !=
        len(true_code)
    ):

        return 0.0


    best_accuracy = 0.0


    for offset in range(
        modulation_order
    ):

        aligned = (

            estimated_code
            + offset

        ) % modulation_order


        accuracy = np.mean(

            aligned
            ==
            true_code
        )


        best_accuracy = max(

            best_accuracy,

            accuracy
        )


    return float(
        best_accuracy
    )


# ============================================================
# 3. RMSE
# ============================================================

def rmse(
    true,
    estimated
):

    return float(

        np.sqrt(

            np.mean(

                (
                    np.asarray(estimated)

                    -

                    np.asarray(true)
                ) ** 2
            )
        )
    )


# ============================================================
# 4. 对一种PSK评价
# ============================================================

def evaluate_one_modulation(
    data,
    class_name
):

    X = data["X"]

    y = data["y"]

    snr_array = data["snr"]

    params = data["params"]

    codes = data["code"]

    class_names = (
        data["class_names"]
        .tolist()
    )

    param_names = (
        data["param_names"]
        .tolist()
    )

    fs = float(
        data["fs"]
    )


    param_index = {

        name: index

        for index, name in enumerate(
            param_names
        )
    }


    label = class_names.index(
        class_name
    )


    indices = np.where(
        y == label
    )[0]


    modulation_order = (

        2

        if class_name == "BPSK"

        else 4
    )


    print()
    print("=" * 70)

    print(
        f"{class_name} 参数测量"
    )

    print("=" * 70)


    print(
        f"\n测试样本数："
        f"{len(indices)}"
    )


    rows = []


    true_fc_all = []
    est_fc_all = []

    count_correct_all = []

    code_accuracy_all = []

    exact_code_all = []


    for number, index in enumerate(
        indices,
        start=1
    ):

        # ----------------------------------------------------
        # IQ
        # ----------------------------------------------------

        signal = (

            X[index, 0]

            + 1j
            * X[index, 1]
        )


        # ----------------------------------------------------
        # 估计
        # ----------------------------------------------------

        result = (
            estimate_psk_parameters(

                signal,

                fs,

                modulation=class_name
            )
        )


        # ----------------------------------------------------
        # Ground Truth
        # ----------------------------------------------------

        sample_params = params[
            index
        ]


        true_fc = sample_params[

            param_index[
                "center_freq_hz"
            ]
        ]


        true_symbol_count = int(

            sample_params[

                param_index[
                    "symbol_count"
                ]

            ]
        )


        true_symbol_rate = sample_params[

            param_index[
                "symbol_rate_baud"
            ]
        ]


        true_code = codes[
            index,
            :true_symbol_count
        ].astype(
            np.int64
        )


        # ----------------------------------------------------
        # Estimate
        # ----------------------------------------------------

        est_fc = result[
            "center_freq_hz"
        ]


        est_symbol_count = result[
            "symbol_count"
        ]


        est_symbol_rate = result[
            "symbol_rate_baud"
        ]


        est_code = result[
            "symbol_sequence"
        ]


        # ----------------------------------------------------
        # 指标
        # ----------------------------------------------------

        center_frequency_error = abs(

            est_fc
            -
            true_fc
        )


        count_correct = (

            est_symbol_count
            ==
            true_symbol_count
        )


        symbol_rate_relative_error = (

            abs(
                est_symbol_rate
                -
                true_symbol_rate
            )

            /

            true_symbol_rate

            * 100
        )


        code_accuracy = (
            aligned_code_accuracy(

                est_code,

                true_code,

                modulation_order
            )
        )


        exact_code = (

            code_accuracy
            >= 1.0 - 1e-12
        )


        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        true_fc_all.append(
            true_fc
        )

        est_fc_all.append(
            est_fc
        )

        count_correct_all.append(
            count_correct
        )

        code_accuracy_all.append(
            code_accuracy
        )

        exact_code_all.append(
            exact_code
        )


        rows.append({

            "sample_index":
                int(index),

            "snr_db":
                int(
                    snr_array[index]
                ),

            "true_fc_hz":
                float(true_fc),

            "est_fc_hz":
                float(est_fc),

            "fc_abs_error_hz":
                float(
                    center_frequency_error
                ),

            "true_symbol_count":
                int(
                    true_symbol_count
                ),

            "est_symbol_count":
                int(
                    est_symbol_count
                ),

            "symbol_count_correct":
                int(
                    count_correct
                ),

            "true_symbol_rate_baud":
                float(
                    true_symbol_rate
                ),

            "est_symbol_rate_baud":
                float(
                    est_symbol_rate
                ),

            "symbol_rate_relative_error_percent":
                float(
                    symbol_rate_relative_error
                ),

            "code_accuracy":
                float(
                    code_accuracy
                ),

            "exact_code_recovery":
                int(
                    exact_code
                ),

            "fit_score":
                float(
                    result[
                        "fit_score"
                    ]
                )
        })


        if (
            number % 20 == 0
            or
            number == len(indices)
        ):

            print(
                f"已处理 "
                f"{number}/"
                f"{len(indices)}"
            )


    # ========================================================
    # 总体指标
    # ========================================================

    true_fc_all = np.asarray(
        true_fc_all
    )

    est_fc_all = np.asarray(
        est_fc_all
    )


    fc_mae = np.mean(

        np.abs(

            est_fc_all
            -
            true_fc_all
        )
    )


    fc_rmse = rmse(

        true_fc_all,

        est_fc_all
    )


    count_accuracy = np.mean(
        count_correct_all
    )


    mean_code_accuracy = np.mean(
        code_accuracy_all
    )


    exact_code_accuracy = np.mean(
        exact_code_all
    )


    print()
    print("-" * 70)

    print(
        f"{class_name} 总体结果"
    )

    print("-" * 70)


    print(
        f"\n中心频率 MAE："
        f"{fc_mae / 1e3:.3f} kHz"
    )

    print(
        f"中心频率 RMSE："
        f"{fc_rmse / 1e3:.3f} kHz"
    )


    print(
        f"\n码元数识别准确率："
        f"{count_accuracy * 100:.2f}%"
    )


    print(
        f"平均码序列恢复率："
        f"{mean_code_accuracy * 100:.2f}%"
    )


    print(
        f"整段码序列完全恢复率："
        f"{exact_code_accuracy * 100:.2f}%"
    )


    # ========================================================
    # 保存逐样本CSV
    # ========================================================

    csv_path = (

        OUTPUT_DIR

        / f"{class_name.lower()}_parameter_results.csv"
    )


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # ========================================================
    # 按SNR评价
    # ========================================================

    snr_values = np.sort(

        np.unique(

            snr_array[
                indices
            ]
        )
    )


    snr_rows = []


    for snr_value in snr_values:

        selected = [

            row

            for row in rows

            if row["snr_db"]
            == snr_value
        ]


        snr_rows.append({

            "snr_db":
                int(snr_value),

            "fc_mae_hz":
                float(
                    np.mean([
                        row[
                            "fc_abs_error_hz"
                        ]
                        for row in selected
                    ])
                ),

            "symbol_count_accuracy":
                float(
                    np.mean([
                        row[
                            "symbol_count_correct"
                        ]
                        for row in selected
                    ])
                ),

            "code_accuracy":
                float(
                    np.mean([
                        row[
                            "code_accuracy"
                        ]
                        for row in selected
                    ])
                ),

            "exact_code_recovery":
                float(
                    np.mean([
                        row[
                            "exact_code_recovery"
                        ]
                        for row in selected
                    ])
                )
        })


    snr_csv = (

        OUTPUT_DIR

        / f"{class_name.lower()}_metrics_by_snr.csv"
    )


    with open(
        snr_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=list(
                snr_rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            snr_rows
        )


    return rows, snr_rows


# ============================================================
# 5. 画图
# ============================================================

def plot_metrics(
    class_name,
    snr_rows
):

    snr = np.array([

        row["snr_db"]

        for row in snr_rows
    ])


    # ========================================================
    # 码元数准确率
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(

        snr,

        np.array([

            row[
                "symbol_count_accuracy"
            ]

            for row in snr_rows

        ]) * 100,

        marker="o"
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Symbol Count Accuracy (%)"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_title(
        f"{class_name} Symbol Count Accuracy vs SNR"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(

        OUTPUT_DIR
        / f"{class_name.lower()}_symbol_count_vs_snr.png",

        dpi=200,

        bbox_inches="tight"
    )

    plt.show()


    # ========================================================
    # 码序列恢复率
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(

        snr,

        np.array([

            row["code_accuracy"]

            for row in snr_rows

        ]) * 100,

        marker="o"
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Code Accuracy (%)"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_title(
        f"{class_name} Code Recovery vs SNR"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(

        OUTPUT_DIR
        / f"{class_name.lower()}_code_accuracy_vs_snr.png",

        dpi=200,

        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 6. main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "BPSK / QPSK 参数测量性能评价"
    )

    print("=" * 70)


    data = np.load(
        DATA_PATH,
        allow_pickle=False
    )


    # ========================================================
    # BPSK
    # ========================================================

    (
        _,
        bpsk_snr
    ) = evaluate_one_modulation(

        data,

        "BPSK"
    )


    plot_metrics(

        "BPSK",

        bpsk_snr
    )


    # ========================================================
    # QPSK
    # ========================================================

    (
        _,
        qpsk_snr
    ) = evaluate_one_modulation(

        data,

        "QPSK"
    )


    plot_metrics(

        "QPSK",

        qpsk_snr
    )


    data.close()


    print()
    print("=" * 70)

    print(
        "PSK参数评价完成"
    )

    print("=" * 70)


    print(
        f"\n结果保存："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()