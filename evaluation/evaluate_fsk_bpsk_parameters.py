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


from estimation.fsk_bpsk_estimator import (
    estimate_fsk_bpsk_parameters
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
    / "fsk_bpsk"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. BPSK码序列允许整体反相
# ============================================================

def aligned_bpsk_accuracy(
    estimated,
    true
):

    estimated = np.asarray(
        estimated,
        dtype=np.int64
    )

    true = np.asarray(
        true,
        dtype=np.int64
    )


    if len(estimated) != len(true):
        return 0.0


    accuracy_1 = np.mean(
        estimated == true
    )

    accuracy_2 = np.mean(
        (1 - estimated) == true
    )


    return float(
        max(
            accuracy_1,
            accuracy_2
        )
    )


# ============================================================
# 3. 普通码序列准确率
# ============================================================

def sequence_accuracy(
    estimated,
    true
):

    estimated = np.asarray(
        estimated,
        dtype=np.int64
    )

    true = np.asarray(
        true,
        dtype=np.int64
    )


    if len(estimated) != len(true):
        return 0.0


    return float(
        np.mean(
            estimated == true
        )
    )


# ============================================================
# 4. RMSE
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
# 5. 主程序
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "2FSK-BPSK 参数测量性能评价"
    )

    print("=" * 70)


    data = np.load(
        DATA_PATH,
        allow_pickle=False
    )


    X = data["X"]

    y = data["y"]

    snr_array = data["snr"]

    params = data["params"]

    codes = data["code"]

    fsk_codes = data["fsk_code"]

    fs = float(
        data["fs"]
    )


    class_names = (
        data["class_names"]
        .tolist()
    )

    param_names = (
        data["param_names"]
        .tolist()
    )


    param_index = {

        name: index

        for index, name in enumerate(
            param_names
        )
    }


    class_label = class_names.index(
        "2FSK_BPSK"
    )


    indices = np.where(
        y == class_label
    )[0]


    print(
        f"\n测试样本数："
        f"{len(indices)}"
    )


    rows = []


    fc_true_all = []
    fc_est_all = []

    sep_true_all = []
    sep_est_all = []

    symbol_count_correct_all = []

    bpsk_accuracy_all = []

    bpsk_exact_all = []

    fsk_accuracy_all = []

    fsk_exact_all = []


    # ========================================================
    # 遍历
    # ========================================================

    for number, index in enumerate(
        indices,
        start=1
    ):

        signal = (

            X[index, 0]

            +

            1j
            * X[index, 1]
        )


        result = (
            estimate_fsk_bpsk_parameters(

                signal,

                fs
            )
        )


        sample_params = params[
            index
        ]


        # ====================================================
        # Ground Truth
        # ====================================================

        true_fc = sample_params[

            param_index[
                "center_freq_hz"
            ]
        ]


        true_f1 = sample_params[

            param_index[
                "f1_hz"
            ]
        ]


        true_f2 = sample_params[

            param_index[
                "f2_hz"
            ]
        ]


        true_sep = sample_params[

            param_index[
                "freq_sep_hz"
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


        true_bpsk_code = codes[
            index,
            :true_symbol_count
        ].astype(
            np.int64
        )


        true_fsk_code = fsk_codes[
            index,
            :true_symbol_count
        ].astype(
            np.int64
        )


        # ====================================================
        # Estimate
        # ====================================================

        est_fc = result[
            "center_freq_hz"
        ]

        est_f1 = result[
            "f1_hz"
        ]

        est_f2 = result[
            "f2_hz"
        ]

        est_sep = result[
            "freq_sep_hz"
        ]

        est_symbol_count = result[
            "symbol_count"
        ]

        est_symbol_rate = result[
            "symbol_rate_baud"
        ]


        # ====================================================
        # 参数误差
        # ====================================================

        fc_error = abs(
            est_fc
            -
            true_fc
        )


        f1_error = abs(
            est_f1
            -
            true_f1
        )


        f2_error = abs(
            est_f2
            -
            true_f2
        )


        sep_error = abs(
            est_sep
            -
            true_sep
        )


        sep_relative_error = (

            sep_error

            /

            abs(true_sep)

            * 100
        )


        symbol_count_correct = (

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


        # ====================================================
        # 只有码长正确时才能逐码元比较
        # ====================================================

        if symbol_count_correct:

            bpsk_accuracy = (
                aligned_bpsk_accuracy(

                    result[
                        "bpsk_code"
                    ],

                    true_bpsk_code
                )
            )


            fsk_accuracy = (
                sequence_accuracy(

                    result[
                        "fsk_code"
                    ],

                    true_fsk_code
                )
            )

        else:

            bpsk_accuracy = 0.0

            fsk_accuracy = 0.0


        bpsk_exact = (
            bpsk_accuracy
            >=
            1.0 - 1e-12
        )


        fsk_exact = (
            fsk_accuracy
            >=
            1.0 - 1e-12
        )


        # ====================================================
        # 记录
        # ====================================================

        fc_true_all.append(
            true_fc
        )

        fc_est_all.append(
            est_fc
        )


        sep_true_all.append(
            true_sep
        )

        sep_est_all.append(
            est_sep
        )


        symbol_count_correct_all.append(
            symbol_count_correct
        )


        bpsk_accuracy_all.append(
            bpsk_accuracy
        )

        bpsk_exact_all.append(
            bpsk_exact
        )


        fsk_accuracy_all.append(
            fsk_accuracy
        )

        fsk_exact_all.append(
            fsk_exact
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
                float(fc_error),

            "true_f1_hz":
                float(true_f1),

            "est_f1_hz":
                float(est_f1),

            "f1_abs_error_hz":
                float(f1_error),

            "true_f2_hz":
                float(true_f2),

            "est_f2_hz":
                float(est_f2),

            "f2_abs_error_hz":
                float(f2_error),

            "true_freq_sep_hz":
                float(true_sep),

            "est_freq_sep_hz":
                float(est_sep),

            "freq_sep_abs_error_hz":
                float(sep_error),

            "freq_sep_relative_error_percent":
                float(
                    sep_relative_error
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
                    symbol_count_correct
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

            "bpsk_code_accuracy":
                float(
                    bpsk_accuracy
                ),

            "bpsk_exact_recovery":
                int(
                    bpsk_exact
                ),

            "fsk_code_accuracy":
                float(
                    fsk_accuracy
                ),

            "fsk_exact_recovery":
                int(
                    fsk_exact
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
    # 总体统计
    # ========================================================

    fc_mae = np.mean(

        np.abs(

            np.asarray(
                fc_est_all
            )

            -

            np.asarray(
                fc_true_all
            )
        )
    )


    fc_rmse = rmse(
        fc_true_all,
        fc_est_all
    )


    sep_mae = np.mean(

        np.abs(

            np.asarray(
                sep_est_all
            )

            -

            np.asarray(
                sep_true_all
            )
        )
    )


    sep_rmse = rmse(
        sep_true_all,
        sep_est_all
    )


    sep_mape = np.mean(

        np.abs(

            (
                np.asarray(
                    sep_est_all
                )

                -

                np.asarray(
                    sep_true_all
                )
            )

            /

            np.asarray(
                sep_true_all
            )
        )
    ) * 100


    symbol_count_accuracy = np.mean(
        symbol_count_correct_all
    )


    bpsk_code_accuracy = np.mean(
        bpsk_accuracy_all
    )


    bpsk_exact_accuracy = np.mean(
        bpsk_exact_all
    )


    fsk_code_accuracy = np.mean(
        fsk_accuracy_all
    )


    fsk_exact_accuracy = np.mean(
        fsk_exact_all
    )


    # ========================================================
    # 输出
    # ========================================================

    print()
    print("=" * 70)

    print(
        "2FSK-BPSK 总体结果"
    )

    print("=" * 70)


    print(
        f"\n中心频率 MAE："
        f"{fc_mae / 1e3:.3f} kHz"
    )


    print(
        f"中心频率 RMSE："
        f"{fc_rmse / 1e3:.3f} kHz"
    )


    print(
        f"\n频率间隔 MAE："
        f"{sep_mae / 1e3:.3f} kHz"
    )


    print(
        f"频率间隔 RMSE："
        f"{sep_rmse / 1e3:.3f} kHz"
    )


    print(
        f"频率间隔平均相对误差："
        f"{sep_mape:.3f}%"
    )


    print(
        f"\n码元数识别准确率："
        f"{symbol_count_accuracy * 100:.2f}%"
    )


    print(
        f"\nBPSK平均码恢复率："
        f"{bpsk_code_accuracy * 100:.2f}%"
    )


    print(
        f"BPSK整段完全恢复率："
        f"{bpsk_exact_accuracy * 100:.2f}%"
    )


    print(
        f"\nFSK跳频序列平均恢复率："
        f"{fsk_code_accuracy * 100:.2f}%"
    )


    print(
        f"FSK跳频序列完全恢复率："
        f"{fsk_exact_accuracy * 100:.2f}%"
    )


    # ========================================================
    # 保存逐样本CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "fsk_bpsk_parameter_results.csv"
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
    # 按SNR统计
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
            ==
            snr_value
        ]


        snr_rows.append({

            "snr_db":
                int(snr_value),

            "frequency_separation_mape":
                float(
                    np.mean([
                        row[
                            "freq_sep_relative_error_percent"
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

            "bpsk_code_accuracy":
                float(
                    np.mean([
                        row[
                            "bpsk_code_accuracy"
                        ]
                        for row in selected
                    ])
                ),

            "fsk_code_accuracy":
                float(
                    np.mean([
                        row[
                            "fsk_code_accuracy"
                        ]
                        for row in selected
                    ])
                )
        })


    with open(
        OUTPUT_DIR
        / "fsk_bpsk_metrics_by_snr.csv",
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


    # ========================================================
    # 画恢复率 vs SNR
    # ========================================================

    snr_axis = np.asarray([

        row["snr_db"]

        for row in snr_rows
    ])


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    ax.plot(

        snr_axis,

        np.asarray([

            row[
                "symbol_count_accuracy"
            ]

            for row in snr_rows

        ]) * 100,

        marker="o",

        label="Symbol Count"
    )


    ax.plot(

        snr_axis,

        np.asarray([

            row[
                "bpsk_code_accuracy"
            ]

            for row in snr_rows

        ]) * 100,

        marker="s",

        label="BPSK Code"
    )


    ax.plot(

        snr_axis,

        np.asarray([

            row[
                "fsk_code_accuracy"
            ]

            for row in snr_rows

        ]) * 100,

        marker="^",

        label="FSK Sequence"
    )


    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.set_title(
        "2FSK-BPSK Parameter Estimation vs SNR"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )


    fig.tight_layout()


    fig.savefig(

        OUTPUT_DIR
        / "fsk_bpsk_accuracy_vs_snr.png",

        dpi=200,

        bbox_inches="tight"
    )


    plt.show()


    data.close()


    print()
    print("=" * 70)

    print(
        "2FSK-BPSK 参数评价完成"
    )

    print("=" * 70)


    print(
        f"\n结果保存："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()