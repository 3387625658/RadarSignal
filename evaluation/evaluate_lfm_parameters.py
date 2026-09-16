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


from estimation.lfm_estimator import (
    estimate_lfm_parameters
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
    / "lfm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. RMSE
# ============================================================

def rmse(
    true,
    estimated
):

    true = np.asarray(true)

    estimated = np.asarray(
        estimated
    )

    return np.sqrt(

        np.mean(

            (
                estimated
                - true
            ) ** 2
        )
    )


# ============================================================
# 3. MAE
# ============================================================

def mae(
    true,
    estimated
):

    return np.mean(

        np.abs(

            np.asarray(estimated)

            -

            np.asarray(true)
        )
    )


# ============================================================
# 4. MAPE
# ============================================================

def mape(
    true,
    estimated
):

    true = np.asarray(
        true
    )

    estimated = np.asarray(
        estimated
    )

    return np.mean(

        np.abs(

            (
                estimated
                - true
            )

            /

            (
                np.abs(true)
                + 1e-12
            )
        )
    ) * 100


# ============================================================
# 5. main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "LFM 参数测量性能评价"
    )

    print("=" * 70)


    # ========================================================
    # 读取测试数据库
    # ========================================================

    data = np.load(
        DATA_PATH,
        allow_pickle=False
    )


    X = data["X"]

    y = data["y"]

    snr_array = data["snr"]

    params = data["params"]

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


    # ========================================================
    # 参数名称 → 下标
    # ========================================================

    param_index = {

        name: index

        for index, name in enumerate(
            param_names
        )
    }


    # ========================================================
    # 找到LFM类别编号
    # ========================================================

    lfm_label = (
        class_names.index(
            "LFM"
        )
    )


    lfm_indices = np.where(
        y == lfm_label
    )[0]


    print(
        f"\n测试集中LFM样本数："
        f"{len(lfm_indices)}"
    )


    # ========================================================
    # 用于保存结果
    # ========================================================

    results = []


    true_fc = []
    estimated_fc = []

    true_bw = []
    estimated_bw = []

    true_k = []
    estimated_k = []

    direction_correct = []


    # ========================================================
    # 遍历全部LFM测试数据
    # ========================================================

    for count, index in enumerate(
        lfm_indices,
        start=1
    ):

        # ----------------------------------------------------
        # 恢复复数IQ
        # ----------------------------------------------------

        I = X[
            index,
            0
        ]

        Q = X[
            index,
            1
        ]


        signal = (

            I

            + 1j * Q
        )


        # ----------------------------------------------------
        # 参数估计
        # ----------------------------------------------------

        estimate = (
            estimate_lfm_parameters(
                signal,
                fs
            )
        )


        # ----------------------------------------------------
        # Ground Truth
        # ----------------------------------------------------

        sample_params = params[
            index
        ]


        gt_fc = sample_params[

            param_index[
                "center_freq_hz"
            ]
        ]


        gt_bw = sample_params[

            param_index[
                "bandwidth_hz"
            ]
        ]


        gt_k = sample_params[

            param_index[
                "chirp_rate_hz_per_s"
            ]
        ]


        gt_direction = int(

            sample_params[

                param_index[
                    "chirp_direction"
                ]

            ]
        )


        # ----------------------------------------------------
        # Estimate
        # ----------------------------------------------------

        est_fc = estimate[
            "center_freq_hz"
        ]

        est_bw = estimate[
            "bandwidth_hz"
        ]

        est_k = estimate[
            "chirp_rate_hz_per_s"
        ]

        est_direction = estimate[
            "chirp_direction"
        ]


        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------

        true_fc.append(
            gt_fc
        )

        estimated_fc.append(
            est_fc
        )

        true_bw.append(
            gt_bw
        )

        estimated_bw.append(
            est_bw
        )

        true_k.append(
            gt_k
        )

        estimated_k.append(
            est_k
        )

        direction_correct.append(

            gt_direction
            ==
            est_direction
        )


        # ----------------------------------------------------
        # 单样本误差
        # ----------------------------------------------------

        fc_error = np.abs(
            est_fc - gt_fc
        )

        bw_error = np.abs(
            est_bw - gt_bw
        )

        bw_relative_error = (

            bw_error
            /
            np.abs(gt_bw)
            * 100
        )


        k_error = np.abs(
            est_k - gt_k
        )

        k_relative_error = (

            k_error
            /
            np.abs(gt_k)
            * 100
        )


        snr_value = int(
            snr_array[index]
        )


        results.append({

            "sample_index":
                int(index),

            "snr_db":
                snr_value,

            "true_fc_hz":
                gt_fc,

            "est_fc_hz":
                est_fc,

            "fc_abs_error_hz":
                fc_error,

            "true_bw_hz":
                gt_bw,

            "est_bw_hz":
                est_bw,

            "bw_abs_error_hz":
                bw_error,

            "bw_relative_error_percent":
                bw_relative_error,

            "true_k":
                gt_k,

            "est_k":
                est_k,

            "k_abs_error":
                k_error,

            "k_relative_error_percent":
                k_relative_error,

            "direction_correct":
                int(
                    gt_direction
                    ==
                    est_direction
                ),

            "ridge_rmse_hz":
                estimate[
                    "ridge_rmse_hz"
                ]
        })


        if (
            count % 20 == 0

            or

            count
            ==
            len(lfm_indices)
        ):

            print(
                f"已处理 "
                f"{count}/"
                f"{len(lfm_indices)}"
            )


    # ========================================================
    # NumPy
    # ========================================================

    true_fc = np.asarray(
        true_fc
    )

    estimated_fc = np.asarray(
        estimated_fc
    )

    true_bw = np.asarray(
        true_bw
    )

    estimated_bw = np.asarray(
        estimated_bw
    )

    true_k = np.asarray(
        true_k
    )

    estimated_k = np.asarray(
        estimated_k
    )


    # ========================================================
    # 总体指标
    # ========================================================

    fc_mae = mae(
        true_fc,
        estimated_fc
    )

    fc_rmse = rmse(
        true_fc,
        estimated_fc
    )


    bw_mae = mae(
        true_bw,
        estimated_bw
    )

    bw_rmse = rmse(
        true_bw,
        estimated_bw
    )

    bw_mape = mape(
        true_bw,
        estimated_bw
    )


    k_mae = mae(
        true_k,
        estimated_k
    )

    k_rmse = rmse(
        true_k,
        estimated_k
    )

    k_mape = mape(
        true_k,
        estimated_k
    )


    direction_accuracy = np.mean(
        direction_correct
    )


    print()
    print("=" * 70)

    print(
        "LFM 参数测量总体结果"
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
        f"\n带宽 MAE："
        f"{bw_mae / 1e3:.3f} kHz"
    )

    print(
        f"带宽 RMSE："
        f"{bw_rmse / 1e3:.3f} kHz"
    )

    print(
        f"带宽平均相对误差："
        f"{bw_mape:.3f}%"
    )


    print(
        f"\n调频斜率 MAE："
        f"{k_mae:.6e} Hz/s"
    )

    print(
        f"调频斜率 RMSE："
        f"{k_rmse:.6e} Hz/s"
    )

    print(
        f"调频斜率平均相对误差："
        f"{k_mape:.3f}%"
    )


    print(
        f"\n扫频方向识别准确率："
        f"{direction_accuracy * 100:.2f}%"
    )


    # ========================================================
    # 保存逐样本结果
    # ========================================================

    result_csv = (
        OUTPUT_DIR
        / "lfm_parameter_results.csv"
    )


    with open(
        result_csv,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = list(
            results[0].keys()
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # 各SNR统计
    # ========================================================

    snr_values = np.sort(
        np.unique(
            snr_array[
                lfm_indices
            ]
        )
    )


    snr_rows = []


    for snr_value in snr_values:

        selected = [

            row

            for row in results

            if row["snr_db"]
            == snr_value
        ]


        bw_relative = np.array([

            row[
                "bw_relative_error_percent"
            ]

            for row in selected
        ])


        k_relative = np.array([

            row[
                "k_relative_error_percent"
            ]

            for row in selected
        ])


        fc_absolute = np.array([

            row[
                "fc_abs_error_hz"
            ]

            for row in selected
        ])


        direction_acc = np.mean([

            row[
                "direction_correct"
            ]

            for row in selected
        ])


        snr_rows.append({

            "snr_db":
                int(snr_value),

            "fc_mae_hz":
                float(
                    np.mean(
                        fc_absolute
                    )
                ),

            "bw_mape_percent":
                float(
                    np.mean(
                        bw_relative
                    )
                ),

            "k_mape_percent":
                float(
                    np.mean(
                        k_relative
                    )
                ),

            "direction_accuracy":
                float(
                    direction_acc
                )
        })


    snr_csv = (
        OUTPUT_DIR
        / "lfm_metrics_by_snr.csv"
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


    # ========================================================
    # 带宽相对误差 vs SNR
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    ax.plot(

        [
            row["snr_db"]
            for row in snr_rows
        ],

        [
            row[
                "bw_mape_percent"
            ]
            for row in snr_rows
        ],

        marker="o"
    )


    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Bandwidth MAPE (%)"
    )

    ax.set_title(
        "LFM Bandwidth Estimation Error vs SNR"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()


    fig.savefig(

        OUTPUT_DIR
        / "bandwidth_error_vs_snr.png",

        dpi=200,

        bbox_inches="tight"
    )


    plt.show()


    # ========================================================
    # 调频斜率误差 vs SNR
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    ax.plot(

        [
            row["snr_db"]
            for row in snr_rows
        ],

        [
            row[
                "k_mape_percent"
            ]
            for row in snr_rows
        ],

        marker="o"
    )


    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "Chirp Rate MAPE (%)"
    )

    ax.set_title(
        "LFM Chirp Rate Estimation Error vs SNR"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()


    fig.savefig(

        OUTPUT_DIR
        / "chirp_rate_error_vs_snr.png",

        dpi=200,

        bbox_inches="tight"
    )


    plt.show()


    data.close()


    print()
    print("=" * 70)

    print(
        "LFM参数评价完成"
    )

    print("=" * 70)


    print(
        f"\n结果保存："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()