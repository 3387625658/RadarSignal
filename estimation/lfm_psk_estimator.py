import numpy as np

from estimation.lfm_estimator import (
    estimate_lfm_parameters
)

from estimation.psk_estimator import (
    estimate_psk_parameters
)


# ============================================================
# LFM + PSK复合信号参数估计
# ============================================================

def estimate_lfm_psk_parameters(
    signal,
    fs,
    modulation="BPSK",
    symbol_count_candidates=(8, 16, 32)
):
    """
    LFM-BPSK / LFM-QPSK 参数估计

    流程：

        原始复合信号
            ↓
        LFM参数估计
            ↓
        Dechirp
            ↓
        PSK参数估计

    Parameters
    ----------
    signal:
        一维复数IQ

    fs:
        采样率

    modulation:
        "BPSK"
        "QPSK"

    Returns
    -------
    dict
    """

    signal = np.asarray(signal)

    if signal.ndim != 1:

        raise ValueError(
            "signal必须是一维复数IQ信号"
        )


    n = len(signal)

    duration = (
        n / fs
    )

    t = (
        np.arange(n)
        / fs
    )


    # ========================================================
    # 1. LFM参数估计
    # ========================================================

    lfm_result = (
        estimate_lfm_parameters(
            signal,
            fs
        )
    )


    chirp_rate = (
        lfm_result[
            "chirp_rate_hz_per_s"
        ]
    )


    start_frequency = (
        lfm_result[
            "start_freq_hz"
        ]
    )


    # ========================================================
    # 2. 构造估计的LFM参考信号
    #
    # 瞬时频率：
    #
    # f(t) = f_start + k*t
    #
    # 对应相位：
    #
    # phi(t)
    # =
    # 2*pi*f_start*t
    # +
    # pi*k*t²
    #
    # 与数据库生成式只相差一个常数相位，
    # 不影响PSK恢复。
    # ========================================================

    reference_phase = (

        2
        * np.pi
        * start_frequency
        * t

        +

        np.pi
        * chirp_rate
        * t ** 2
    )


    reference_signal = np.exp(

        1j
        * reference_phase
    )


    # ========================================================
    # 3. Dechirp
    #
    # x(t) × conj(LFM)
    #
    # 理想情况下剩余：
    #
    # BPSK / QPSK + 常数相位
    # ========================================================

    dechirped_signal = (

        signal

        * np.conj(
            reference_signal
        )
    )


    # ========================================================
    # 4. PSK参数估计
    # ========================================================

    psk_result = (
        estimate_psk_parameters(

            dechirped_signal,

            fs,

            modulation=modulation,

            symbol_count_candidates=(
                symbol_count_candidates
            )
        )
    )


    # ========================================================
    # 5. 注意：
    #
    # Dechirp之后PSK估计出来的中心频率
    # 理论上应该接近0。
    #
    # 真正原始信号的中心频率使用
    # LFM阶段估计值。
    # ========================================================

    return {

        # ----------------------------------------------------
        # LFM部分
        # ----------------------------------------------------

        "center_freq_hz":
            lfm_result[
                "center_freq_hz"
            ],

        "bandwidth_hz":
            lfm_result[
                "bandwidth_hz"
            ],

        "chirp_rate_hz_per_s":
            lfm_result[
                "chirp_rate_hz_per_s"
            ],

        "chirp_direction":
            lfm_result[
                "chirp_direction"
            ],

        "start_freq_hz":
            lfm_result[
                "start_freq_hz"
            ],

        "end_freq_hz":
            lfm_result[
                "end_freq_hz"
            ],

        "ridge_rmse_hz":
            lfm_result[
                "ridge_rmse_hz"
            ],


        # ----------------------------------------------------
        # PSK部分
        # ----------------------------------------------------

        "modulation":
            psk_result[
                "modulation"
            ],

        "symbol_count":
            psk_result[
                "symbol_count"
            ],

        "symbol_rate_baud":
            psk_result[
                "symbol_rate_baud"
            ],

        "symbol_duration_s":
            psk_result[
                "symbol_duration_s"
            ],

        "symbol_sequence":
            psk_result[
                "symbol_sequence"
            ],

        "fit_score":
            psk_result[
                "fit_score"
            ],

        # ----------------------------------------------------
        # Dechirp残余频偏
        #
        # 理想值≈0
        # 可作为算法诊断量
        # ----------------------------------------------------

        "residual_frequency_hz":
            psk_result[
                "center_freq_hz"
            ],

        "dechirped_signal":
            dechirped_signal
    }