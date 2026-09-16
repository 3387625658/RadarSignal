from pathlib import Path
import sys
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt


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
# 2. 导入
# ============================================================

from dataset.radar_dataset import create_dataloaders

from pipeline.radar_pipeline import (
    load_classifier,
    estimate_parameters_by_class,
    CLASS_NAMES,
    DEVICE
)


# ============================================================
# 3. 路径
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "test.npz"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "full_pipeline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. 配置
# ============================================================

BATCH_SIZE = 64
NUM_WORKERS = 0


# ============================================================
# 5. PSK码序列对齐
# ============================================================

def aligned_psk_accuracy(
    estimated,
    true,
    modulation_order
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


    best_accuracy = 0.0


    for offset in range(
        modulation_order
    ):

        aligned = (

            estimated
            + offset

        ) % modulation_order


        accuracy = np.mean(
            aligned == true
        )


        best_accuracy = max(
            best_accuracy,
            accuracy
        )


    return float(
        best_accuracy
    )


# ============================================================
# 6. BPSK允许整体反相
# ============================================================

def aligned_bpsk_accuracy(
    estimated,
    true
):

    return aligned_psk_accuracy(
        estimated,
        true,
        modulation_order=2
    )


# ============================================================
# 7. 普通二进制序列
# ============================================================

def binary_sequence_accuracy(
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
# 8. 安全获取估计字段
# ============================================================

def get_value(
    result,
    key
):

    if result is None:

        return np.nan


    value = result.get(
        key,
        np.nan
    )


    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    ):

        return float(value)


    return np.nan


# ============================================================
# 9. 计算参数误差
# ============================================================

def calculate_parameter_metrics(
    true_class,
    estimate,
    true_params,
    param_index,
    true_code,
    true_fsk_code
):

    metrics = {

        "fc_abs_error_hz":
            np.nan,

        "bandwidth_relative_error_percent":
            np.nan,

        "chirp_rate_relative_error_percent":
            np.nan,

        "direction_correct":
            np.nan,

        "symbol_count_correct":
            np.nan,

        "symbol_rate_relative_error_percent":
            np.nan,

        "code_accuracy":
            np.nan,

        "exact_code_recovery":
            np.nan,

        "freq_sep_relative_error_percent":
            np.nan,

        "fsk_code_accuracy":
            np.nan,

        "fsk_exact_recovery":
            np.nan
    }


    # ========================================================
    # 中心频率
    # ========================================================

    if "center_freq_hz" in estimate:

        true_fc = true_params[

            param_index[
                "center_freq_hz"
            ]
        ]


        if not np.isnan(
            true_fc
        ):

            metrics[
                "fc_abs_error_hz"
            ] = abs(

                estimate[
                    "center_freq_hz"
                ]

                -

                true_fc
            )


    # ========================================================
    # LFM类
    # ========================================================

    if true_class in (
        "LFM",
        "LFM_BPSK",
        "LFM_QPSK"
    ):

        true_bw = true_params[

            param_index[
                "bandwidth_hz"
            ]
        ]


        true_k = true_params[

            param_index[
                "chirp_rate_hz_per_s"
            ]
        ]


        true_direction = int(

            true_params[

                param_index[
                    "chirp_direction"
                ]

            ]
        )


        if "bandwidth_hz" in estimate:

            metrics[
                "bandwidth_relative_error_percent"
            ] = (

                abs(
                    estimate[
                        "bandwidth_hz"
                    ]
                    -
                    true_bw
                )

                /

                abs(true_bw)

                * 100
            )


        if (
            "chirp_rate_hz_per_s"
            in estimate
        ):

            metrics[
                "chirp_rate_relative_error_percent"
            ] = (

                abs(
                    estimate[
                        "chirp_rate_hz_per_s"
                    ]
                    -
                    true_k
                )

                /

                abs(true_k)

                * 100
            )


        if "chirp_direction" in estimate:

            metrics[
                "direction_correct"
            ] = float(

                int(
                    estimate[
                        "chirp_direction"
                    ]
                )

                ==
                true_direction
            )


    # ========================================================
    # PSK相关类别
    # ========================================================

    if true_class in (
        "BPSK",
        "QPSK",
        "LFM_BPSK",
        "LFM_QPSK",
        "2FSK_BPSK"
    ):

        true_symbol_count = int(

            true_params[

                param_index[
                    "symbol_count"
                ]

            ]
        )


        true_symbol_rate = true_params[

            param_index[
                "symbol_rate_baud"
            ]
        ]


        if "symbol_count" in estimate:

            metrics[
                "symbol_count_correct"
            ] = float(

                int(
                    estimate[
                        "symbol_count"
                    ]
                )

                ==
                true_symbol_count
            )


        if "symbol_rate_baud" in estimate:

            metrics[
                "symbol_rate_relative_error_percent"
            ] = (

                abs(
                    estimate[
                        "symbol_rate_baud"
                    ]
                    -
                    true_symbol_rate
                )

                /

                true_symbol_rate

                * 100
            )


    # ========================================================
    # 普通PSK和LFM-PSK
    # ========================================================

    if true_class in (
        "BPSK",
        "QPSK",
        "LFM_BPSK",
        "LFM_QPSK"
    ):

        if true_class in (
            "BPSK",
            "LFM_BPSK"
        ):

            modulation_order = 2

        else:

            modulation_order = 4


        symbol_count = int(

            true_params[

                param_index[
                    "symbol_count"
                ]

            ]
        )


        valid_true_code = (
            true_code[
                :symbol_count
            ]
        )


        if (
            "symbol_sequence"
            in estimate
        ):

            code_accuracy = (
                aligned_psk_accuracy(

                    estimate[
                        "symbol_sequence"
                    ],

                    valid_true_code,

                    modulation_order
                )
            )


            metrics[
                "code_accuracy"
            ] = code_accuracy


            metrics[
                "exact_code_recovery"
            ] = float(

                code_accuracy
                >=
                1.0 - 1e-12
            )


    # ========================================================
    # 2FSK-BPSK
    # ========================================================

    if true_class == "2FSK_BPSK":

        true_sep = true_params[

            param_index[
                "freq_sep_hz"
            ]
        ]


        if "freq_sep_hz" in estimate:

            metrics[
                "freq_sep_relative_error_percent"
            ] = (

                abs(
                    estimate[
                        "freq_sep_hz"
                    ]
                    -
                    true_sep
                )

                /

                abs(true_sep)

                * 100
            )


        symbol_count = int(

            true_params[

                param_index[
                    "symbol_count"
                ]

            ]
        )


        valid_true_code = (
            true_code[
                :symbol_count
            ]
        )


        valid_true_fsk = (
            true_fsk_code[
                :symbol_count
            ]
        )


        if "bpsk_code" in estimate:

            bpsk_accuracy = (
                aligned_bpsk_accuracy(

                    estimate[
                        "bpsk_code"
                    ],

                    valid_true_code
                )
            )


            metrics[
                "code_accuracy"
            ] = bpsk_accuracy


            metrics[
                "exact_code_recovery"
            ] = float(

                bpsk_accuracy
                >=
                1.0 - 1e-12
            )


        if "fsk_code" in estimate:

            fsk_accuracy = (
                binary_sequence_accuracy(

                    estimate[
                        "fsk_code"
                    ],

                    valid_true_fsk
                )
            )


            metrics[
                "fsk_code_accuracy"
            ] = fsk_accuracy


            metrics[
                "fsk_exact_recovery"
            ] = float(

                fsk_accuracy
                >=
                1.0 - 1e-12
            )


    return metrics


# ============================================================
# 10. nan平均
# ============================================================

def safe_mean(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )


    valid = values[
        ~np.isnan(values)
    ]


    if len(valid) == 0:

        return np.nan


    return float(
        np.mean(valid)
    )


# ============================================================
# 11. main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "全测试集端到端识别与参数测量评价"
    )

    print("=" * 70)


    print(
        f"\n设备：{DEVICE}"
    )


    # ========================================================
    # 加载模型
    # ========================================================

    model = load_classifier()


    # ========================================================
    # 加载原始test.npz
    # ========================================================

    data = np.load(
        DATA_PATH,
        allow_pickle=False
    )


    X = data["X"]

    y = data["y"]

    snr_array = data["snr"]

    params = data["params"]

    codes = data["code"]

    fsk_codes = data[
        "fsk_code"
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


    # ========================================================
    # DataLoader
    #
    # 分类部分一次64条送进GPU
    # ========================================================

    (
        _,
        _,
        test_loader
    ) = create_dataloaders(

        mode="stft",

        batch_size=BATCH_SIZE,

        num_workers=NUM_WORKERS,

        normalize=True,

        return_metadata=True
    )


    results = []

    processed = 0


    # ========================================================
    # 推理
    # ========================================================

    for (
        signals,
        labels,
        metadata
    ) in test_loader:

        signals = signals.to(
            DEVICE
        )


        with torch.no_grad():

            logits = model(
                signals
            )


            probabilities = torch.softmax(

                logits,

                dim=1
            )


            predictions = torch.argmax(

                probabilities,

                dim=1
            )


        predictions = (
            predictions
            .cpu()
            .numpy()
        )


        probabilities = (
            probabilities
            .cpu()
            .numpy()
        )


        batch_indices = (
            metadata["index"]
        )


        if torch.is_tensor(
            batch_indices
        ):

            batch_indices = (

                batch_indices
                .cpu()
                .numpy()
            )

        else:

            batch_indices = (
                np.asarray(
                    batch_indices
                )
            )


        # ====================================================
        # 每个信号执行参数估计
        # ====================================================

        for j, sample_index in enumerate(
            batch_indices
        ):

            sample_index = int(
                sample_index
            )


            true_label = int(
                y[
                    sample_index
                ]
            )


            predicted_label = int(
                predictions[j]
            )


            true_class = (
                class_names[
                    true_label
                ]
            )


            predicted_class = (
                CLASS_NAMES[
                    predicted_label
                ]
            )


            confidence = float(

                probabilities[
                    j,
                    predicted_label
                ]
            )


            classification_correct = (

                true_class
                ==
                predicted_class
            )


            signal = (

                X[
                    sample_index,
                    0
                ]

                +

                1j
                * X[
                    sample_index,
                    1
                ]
            )


            estimate = None

            estimation_success = True

            error_message = ""


            try:

                estimate = (
                    estimate_parameters_by_class(

                        signal,

                        fs,

                        predicted_class
                    )
                )

            except Exception as error:

                estimation_success = False

                error_message = str(
                    error
                )


            # =================================================
            # 参数误差只有类别识别正确时才有物理可比性
            # =================================================

            if (
                classification_correct
                and
                estimation_success
            ):

                parameter_metrics = (
                    calculate_parameter_metrics(

                        true_class,

                        estimate,

                        params[
                            sample_index
                        ],

                        param_index,

                        codes[
                            sample_index
                        ],

                        fsk_codes[
                            sample_index
                        ]
                    )
                )

            else:

                parameter_metrics = {

                    "fc_abs_error_hz":
                        np.nan,

                    "bandwidth_relative_error_percent":
                        np.nan,

                    "chirp_rate_relative_error_percent":
                        np.nan,

                    "direction_correct":
                        np.nan,

                    "symbol_count_correct":
                        np.nan,

                    "symbol_rate_relative_error_percent":
                        np.nan,

                    "code_accuracy":
                        np.nan,

                    "exact_code_recovery":
                        np.nan,

                    "freq_sep_relative_error_percent":
                        np.nan,

                    "fsk_code_accuracy":
                        np.nan,

                    "fsk_exact_recovery":
                        np.nan
                }


            row = {

                "sample_index":
                    sample_index,

                "snr_db":
                    int(
                        snr_array[
                            sample_index
                        ]
                    ),

                "true_label":
                    true_label,

                "true_class":
                    true_class,

                "pred_label":
                    predicted_label,

                "pred_class":
                    predicted_class,

                "confidence":
                    confidence,

                "classification_correct":
                    int(
                        classification_correct
                    ),

                "estimation_success":
                    int(
                        estimation_success
                    ),

                "end_to_end_route_success":
                    int(
                        classification_correct
                        and
                        estimation_success
                    ),

                "error_message":
                    error_message
            }


            # =================================================
            # 保存估计参数
            # =================================================

            if estimate is not None:

                row.update({

                    "est_center_freq_hz":
                        get_value(
                            estimate,
                            "center_freq_hz"
                        ),

                    "est_bandwidth_hz":
                        get_value(
                            estimate,
                            "bandwidth_hz"
                        ),

                    "est_chirp_rate_hz_per_s":
                        get_value(
                            estimate,
                            "chirp_rate_hz_per_s"
                        ),

                    "est_symbol_count":
                        get_value(
                            estimate,
                            "symbol_count"
                        ),

                    "est_symbol_rate_baud":
                        get_value(
                            estimate,
                            "symbol_rate_baud"
                        ),

                    "est_f1_hz":
                        get_value(
                            estimate,
                            "f1_hz"
                        ),

                    "est_f2_hz":
                        get_value(
                            estimate,
                            "f2_hz"
                        ),

                    "est_freq_sep_hz":
                        get_value(
                            estimate,
                            "freq_sep_hz"
                        )
                })

            else:

                row.update({

                    "est_center_freq_hz":
                        np.nan,

                    "est_bandwidth_hz":
                        np.nan,

                    "est_chirp_rate_hz_per_s":
                        np.nan,

                    "est_symbol_count":
                        np.nan,

                    "est_symbol_rate_baud":
                        np.nan,

                    "est_f1_hz":
                        np.nan,

                    "est_f2_hz":
                        np.nan,

                    "est_freq_sep_hz":
                        np.nan
                })


            row.update(
                parameter_metrics
            )


            results.append(
                row
            )


            processed += 1


        print(
            f"已处理："
            f"{processed}/"
            f"{len(X)}"
        )


    # ========================================================
    # 12. 保存逐样本结果
    # ========================================================

    result_path = (
        OUTPUT_DIR
        / "full_pipeline_results.csv"
    )


    with open(
        result_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=list(
                results[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    # ========================================================
    # 13. 总体结果
    # ========================================================

    classification_accuracy = (
        safe_mean([

            row[
                "classification_correct"
            ]

            for row in results
        ])
    )


    estimation_success_rate = (
        safe_mean([

            row[
                "estimation_success"
            ]

            for row in results
        ])
    )


    route_success_rate = (
        safe_mean([

            row[
                "end_to_end_route_success"
            ]

            for row in results
        ])
    )


    print()
    print("=" * 70)

    print(
        "全测试集总体结果"
    )

    print("=" * 70)


    print(
        f"\n总样本数："
        f"{len(results)}"
    )


    print(
        f"识别准确率："
        f"{classification_accuracy * 100:.2f}%"
    )


    print(
        f"参数算法执行成功率："
        f"{estimation_success_rate * 100:.2f}%"
    )


    print(
        f"端到端正确路由率："
        f"{route_success_rate * 100:.2f}%"
    )


    # ========================================================
    # 14. 各类别端到端指标
    # ========================================================

    summary_rows = []


    print()
    print("=" * 70)

    print(
        "各类别端到端结果"
    )

    print("=" * 70)


    for class_name in class_names:

        selected = [

            row

            for row in results

            if row[
                "true_class"
            ]
            ==
            class_name
        ]


        classification_acc = (
            safe_mean([

                row[
                    "classification_correct"
                ]

                for row in selected
            ])
        )


        route_acc = (
            safe_mean([

                row[
                    "end_to_end_route_success"
                ]

                for row in selected
            ])
        )


        fc_mae = safe_mean([

            row[
                "fc_abs_error_hz"
            ]

            for row in selected
        ])


        bandwidth_mape = safe_mean([

            row[
                "bandwidth_relative_error_percent"
            ]

            for row in selected
        ])


        chirp_mape = safe_mean([

            row[
                "chirp_rate_relative_error_percent"
            ]

            for row in selected
        ])


        direction_acc = safe_mean([

            row[
                "direction_correct"
            ]

            for row in selected
        ])


        symbol_count_acc = safe_mean([

            row[
                "symbol_count_correct"
            ]

            for row in selected
        ])


        code_accuracy = safe_mean([

            row[
                "code_accuracy"
            ]

            for row in selected
        ])


        exact_code = safe_mean([

            row[
                "exact_code_recovery"
            ]

            for row in selected
        ])


        freq_sep_mape = safe_mean([

            row[
                "freq_sep_relative_error_percent"
            ]

            for row in selected
        ])


        fsk_code_accuracy = safe_mean([

            row[
                "fsk_code_accuracy"
            ]

            for row in selected
        ])


        summary_row = {

            "class":
                class_name,

            "samples":
                len(selected),

            "classification_accuracy":
                classification_acc,

            "route_success_rate":
                route_acc,

            "fc_mae_hz":
                fc_mae,

            "bandwidth_mape_percent":
                bandwidth_mape,

            "chirp_rate_mape_percent":
                chirp_mape,

            "direction_accuracy":
                direction_acc,

            "symbol_count_accuracy":
                symbol_count_acc,

            "code_accuracy":
                code_accuracy,

            "exact_code_recovery":
                exact_code,

            "freq_sep_mape_percent":
                freq_sep_mape,

            "fsk_code_accuracy":
                fsk_code_accuracy
        }


        summary_rows.append(
            summary_row
        )


        print()
        print(
            f"{class_name}:"
        )

        print(
            f"  分类准确率："
            f"{classification_acc * 100:.2f}%"
        )

        print(
            f"  端到端正确路由率："
            f"{route_acc * 100:.2f}%"
        )


        if not np.isnan(
            fc_mae
        ):

            print(
                f"  中心频率MAE："
                f"{fc_mae / 1e3:.3f} kHz"
            )


        if not np.isnan(
            bandwidth_mape
        ):

            print(
                f"  带宽MAPE："
                f"{bandwidth_mape:.3f}%"
            )


        if not np.isnan(
            chirp_mape
        ):

            print(
                f"  调频斜率MAPE："
                f"{chirp_mape:.3f}%"
            )


        if not np.isnan(
            symbol_count_acc
        ):

            print(
                f"  码元数准确率："
                f"{symbol_count_acc * 100:.2f}%"
            )


        if not np.isnan(
            code_accuracy
        ):

            print(
                f"  平均码恢复率："
                f"{code_accuracy * 100:.2f}%"
            )


        if not np.isnan(
            freq_sep_mape
        ):

            print(
                f"  频率间隔MAPE："
                f"{freq_sep_mape:.3f}%"
            )


        if not np.isnan(
            fsk_code_accuracy
        ):

            print(
                f"  FSK序列恢复率："
                f"{fsk_code_accuracy * 100:.2f}%"
            )


    # ========================================================
    # 15. 保存类别总结
    # ========================================================

    with open(
        OUTPUT_DIR
        / "class_summary.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=list(
                summary_rows[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )


    # ========================================================
    # 16. 按SNR统计端到端成功率
    # ========================================================

    snr_values = sorted(

        set(

            row[
                "snr_db"
            ]

            for row in results
        )
    )


    snr_rows = []


    for snr_value in snr_values:

        selected = [

            row

            for row in results

            if row[
                "snr_db"
            ]
            ==
            snr_value
        ]


        classification_acc = (
            safe_mean([

                row[
                    "classification_correct"
                ]

                for row in selected
            ])
        )


        route_acc = (
            safe_mean([

                row[
                    "end_to_end_route_success"
                ]

                for row in selected
            ])
        )


        snr_rows.append({

            "snr_db":
                snr_value,

            "classification_accuracy":
                classification_acc,

            "route_success_rate":
                route_acc
        })


    # ========================================================
    # 17. 保存SNR CSV
    # ========================================================

    with open(
        OUTPUT_DIR
        / "pipeline_metrics_by_snr.csv",
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
    # 18. SNR曲线
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
                "classification_accuracy"
            ]
            * 100

            for row in snr_rows
        ],

        marker="o",

        label="Classification Accuracy"
    )


    ax.plot(

        [
            row["snr_db"]
            for row in snr_rows
        ],

        [
            row[
                "route_success_rate"
            ]
            * 100

            for row in snr_rows
        ],

        marker="s",

        label="End-to-End Route Success"
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
        "Full Pipeline Performance vs SNR"
    )


    ax.grid(
        alpha=0.3
    )


    ax.legend()


    fig.tight_layout()


    fig.savefig(

        OUTPUT_DIR
        / "full_pipeline_vs_snr.png",

        dpi=200,

        bbox_inches="tight"
    )


    plt.show()


    # ========================================================
    # 19. 错误分类样本
    # ========================================================

    wrong_rows = [

        row

        for row in results

        if row[
            "classification_correct"
        ]
        == 0
    ]


    print()
    print("=" * 70)

    print(
        "错误识别样本"
    )

    print("=" * 70)


    print(
        f"\n错误数量："
        f"{len(wrong_rows)}"
    )


    for row in wrong_rows:

        print(

            f"样本 {row['sample_index']:4d}"

            f" | SNR={row['snr_db']:2d} dB"

            f" | 真值={row['true_class']}"

            f" | 预测={row['pred_class']}"

            f" | confidence="
            f"{row['confidence'] * 100:.2f}%"
        )


    data.close()


    print()
    print("=" * 70)

    print(
        "全测试集端到端评价完成"
    )

    print("=" * 70)


    print(
        f"\n结果保存："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()