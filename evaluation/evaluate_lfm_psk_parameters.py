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


from estimation.lfm_psk_estimator import (
    estimate_lfm_psk_parameters
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
    / "lfm_psk"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. PSK码序列对齐
# ============================================================

def aligned_code_accuracy(
    estimated_code,
    true_code,
    modulation_order
):

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

        aligned_code = (

            estimated_code
            + offset

        ) % modulation_order


        accuracy = np.mean(

            aligned_code
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
    true_values,
    estimated_values
):

    true_values = np.asarray(
        true_values
    )

    estimated_values = np.asarray(
        estimated_values
    )

    return float(

        np.sqrt(

            np.mean(

                (
                    estimated_values
                    -
                    true_values
                ) ** 2
            )
        )
    )


# ============================================================
# 4. MAPE
# ============================================================

def mape(
    true_values,
    estimated_values
):

    true_values = np.asarray(
        true_values
    )

    estimated_values = np.asarray(
        estimated_values
    )


    return float(

        np.mean(

            np.abs(

                (
                    estimated_values
                    -
                    true_values
                )

                /

                (
                    np.abs(
                        true_values
                    )

                    + 1e-12
                )
            )
        )

        * 100
    )


# ============================================================
# 5. 评价一种复合信号
# ============================================================

def evaluate_one_class(
    data,
    class_name,
    modulation
):

    X = data["X"]

    y = data["y"]

    snr_array = data[
        "snr"
    ]

    params = data[
        "params"
    ]

    codes = data[
        "code"
    ]

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


    label = class_names.index(
        class_name
    )


    indices = np.where(
        y == label
    )[0]


    modulation_order = (

        2
        if modulation == "BPSK"
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

    true_bw_all = []
    est_bw_all = []

    true_k_all = []
    est_k_all = []

    symbol_count_correct_all = []

    code_accuracy_all = []

    exact_code_all = []

    direction_correct_all = []

    residual_freq_all = []


    # ========================================================
    # 遍历测试集
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


        # ----------------------------------------------------
        # 参数估计
        # ----------------------------------------------------

        result = (
            estimate_lfm_psk_parameters(

                signal,

                fs,

                modulation=modulation
            )
        )


        sample_params = (
            params[index]
        )


        # ----------------------------------------------------
        # Ground Truth：LFM
        # ----------------------------------------------------

        true_fc = sample_params[

            param_index[
                "center_freq_hz"
            ]
        ]


        true_bw = sample_params[

            param_index[
                "bandwidth_hz"
            ]
        ]


        true_k = sample_params[

            param_index[
                "chirp_rate_hz_per_s"
            ]
        ]


        true_direction = int(

            sample_params[

                param_index[
                    "chirp_direction"
                ]

            ]
        )


        # ----------------------------------------------------
        # Ground Truth：PSK
        # ----------------------------------------------------

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


        est_bw = result[
            "bandwidth_hz"
        ]


        est_k = result[
            "chirp_rate_hz_per_s"
        ]


        est_direction = result[
            "chirp_direction"
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
        # 误差
        # ----------------------------------------------------

        fc_error = abs(
            est_fc
            -
            true_fc
        )


        bw_relative_error = (

            abs(
                est_bw
                -
                true_bw
            )

            /

            abs(true_bw)

            * 100
        )


        k_relative_error = (

            abs(
                est_k
                -
                true_k
            )

            /

            abs(true_k)

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


        code_accuracy = (
            aligned_code_accuracy(

                est_code,

                true_code,

                modulation_order
            )
        )


        exact_code = (

            code_accuracy
            >=
            1.0 - 1e-12
        )


        direction_correct = (

            est_direction
            ==
            true_direction
        )


        # ----------------------------------------------------
        # 保存统计
        # ----------------------------------------------------

        true_fc_all.append(
            true_fc
        )

        est_fc_all.append(
            est_fc
        )

        true_bw_all.append(
            true_bw
        )

        est_bw_all.append(
            est_bw
        )

        true_k_all.append(
            true_k
        )

        est_k_all.append(
            est_k
        )

        symbol_count_correct_all.append(
            symbol_count_correct
        )

        code_accuracy_all.append(
            code_accuracy
        )

        exact_code_all.append(
            exact_code
        )

        direction_correct_all.append(
            direction_correct
        )

        residual_freq_all.append(
            abs(
                result[
                    "residual_frequency_hz"
                ]
            )
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

            "true_bw_hz":
                float(true_bw),

            "est_bw_hz":
                float(est_bw),

            "bw_relative_error_percent":
                float(
                    bw_relative_error
                ),

            "true_k":
                float(true_k),

            "est_k":
                float(est_k),

            "k_relative_error_percent":
                float(
                    k_relative_error
                ),

            "direction_correct":
                int(
                    direction_correct
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

            "code_accuracy":
                float(
                    code_accuracy
                ),

            "exact_code_recovery":
                int(
                    exact_code
                ),

            "residual_frequency_hz":
                float(
                    result[
                        "residual_frequency_hz"
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
    # 转NumPy
    # ========================================================

    true_fc_all = np.asarray(
        true_fc_all
    )

    est_fc_all = np.asarray(
        est_fc_all
    )

    true_bw_all = np.asarray(
        true_bw_all
    )

    est_bw_all = np.asarray(
        est_bw_all
    )

    true_k_all = np.asarray(
        true_k_all
    )

    est_k_all = np.asarray(
        est_k_all
    )


    # ========================================================
    # 总体指标
    # ========================================================

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


    bw_mape = mape(
        true_bw_all,
        est_bw_all
    )


    k_mape = mape(
        true_k_all,
        est_k_all
    )


    direction_accuracy = np.mean(
        direction_correct_all
    )


    symbol_count_accuracy = np.mean(
        symbol_count_correct_all
    )


    mean_code_accuracy = np.mean(
        code_accuracy_all
    )


    exact_code_accuracy = np.mean(
        exact_code_all
    )


    mean_residual_frequency = np.mean(
        residual_freq_all
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
        f"\n带宽平均相对误差："
        f"{bw_mape:.3f}%"
    )


    print(
        f"调频斜率平均相对误差："
        f"{k_mape:.3f}%"
    )


    print(
        f"扫频方向准确率："
        f"{direction_accuracy * 100:.2f}%"
    )


    print(
        f"\n码元数识别准确率："
        f"{symbol_count_accuracy * 100:.2f}%"
    )


    print(
        f"平均码序列恢复率："
        f"{mean_code_accuracy * 100:.2f}%"
    )


    print(
        f"整段码序列完全恢复率："
        f"{exact_code_accuracy * 100:.2f}%"
    )


    print(
        f"\nDechirp平均残余频偏："
        f"{mean_residual_frequency / 1e3:.3f} kHz"
    )


    # ========================================================
    # 保存逐样本CSV
    # ========================================================

    csv_path = (

        OUTPUT_DIR

        / f"{class_name.lower()}_results.csv"
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

            "bw_mape_percent":
                float(
                    np.mean([
                        row[
                            "bw_relative_error_percent"
                        ]
                        for row in selected
                    ])
                ),

            "k_mape_percent":
                float(
                    np.mean([
                        row[
                            "k_relative_error_percent"
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


    return snr_rows


# ============================================================
# 6. 绘制码恢复率 vs SNR
# ============================================================

def plot_code_accuracy(
    class_name,
    snr_rows
):

    snr = np.array([

        row["snr_db"]

        for row in snr_rows
    ])


    code_accuracy = np.array([

        row["code_accuracy"]

        for row in snr_rows

    ]) * 100


    symbol_accuracy = np.array([

        row[
            "symbol_count_accuracy"
        ]

        for row in snr_rows

    ]) * 100


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    ax.plot(
        snr,
        code_accuracy,
        marker="o",
        label="Code Accuracy"
    )


    ax.plot(
        snr,
        symbol_accuracy,
        marker="s",
        label="Symbol Count Accuracy"
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
        f"{class_name} Parameter Estimation vs SNR"
    )


    ax.legend()


    ax.grid(
        alpha=0.3
    )


    fig.tight_layout()


    fig.savefig(

        OUTPUT_DIR

        / f"{class_name.lower()}_accuracy_vs_snr.png",

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

    print(
        "LFM-BPSK / LFM-QPSK 参数测量性能评价"
    )

    print("=" * 70)


    data = np.load(
        DATA_PATH,
        allow_pickle=False
    )


    # ========================================================
    # LFM-BPSK
    # ========================================================

    lfm_bpsk_snr = (
        evaluate_one_class(

            data,

            "LFM_BPSK",

            "BPSK"
        )
    )


    plot_code_accuracy(

        "LFM_BPSK",

        lfm_bpsk_snr
    )


    # ========================================================
    # LFM-QPSK
    # ========================================================

    lfm_qpsk_snr = (
        evaluate_one_class(

            data,

            "LFM_QPSK",

            "QPSK"
        )
    )


    plot_code_accuracy(

        "LFM_QPSK",

        lfm_qpsk_snr
    )


    data.close()


    print()
    print("=" * 70)

    print(
        "复合LFM-PSK参数评价完成"
    )

    print("=" * 70)


    print(
        f"\n结果保存："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()