import numpy as np
from scipy.signal import stft


def estimate_lfm_parameters(
    signal,
    fs,
    nperseg=128,
    noverlap=112,
    nfft=2048
):
    """
    LFM参数估计

    Parameters
    ----------
    signal:
        一维复数IQ信号

    fs:
        采样率 Hz

    Returns
    -------
    result:
        {
            center_freq_hz,
            bandwidth_hz,
            chirp_rate_hz_per_s,
            chirp_direction,
            start_freq_hz,
            end_freq_hz,
            ridge_rmse_hz
        }
    """

    signal = np.asarray(signal)

    if signal.ndim != 1:

        raise ValueError(
            "signal 必须是一维复数IQ信号"
        )

    if len(signal) < nperseg:

        raise ValueError(
            "信号长度小于STFT窗口长度"
        )

    # ========================================================
    # 1. 信号持续时间
    # ========================================================

    n = len(signal)

    duration = n / fs


    # ========================================================
    # 2. STFT
    # ========================================================

    frequencies, times, Zxx = stft(

        signal,

        fs=fs,

        window="hann",

        nperseg=nperseg,

        noverlap=noverlap,

        nfft=nfft,

        return_onesided=False,

        boundary=None,

        padded=False
    )


    # ========================================================
    # 3. FFT Shift
    #
    # 让频率变成：
    #
    # -fs/2 ... 0 ... fs/2
    # ========================================================

    frequencies = np.fft.fftshift(
        frequencies
    )

    Zxx = np.fft.fftshift(
        Zxx,
        axes=0
    )


    magnitude = np.abs(
        Zxx
    )


    # ========================================================
    # 4. 提取时频脊线
    # ========================================================

    ridge_freq = []

    ridge_strength = []

    frequency_bins = len(
        frequencies
    )


    for time_index in range(
        magnitude.shape[1]
    ):

        column = magnitude[
            :,
            time_index
        ]


        # 最大能量频率位置
        peak_index = int(
            np.argmax(column)
        )


        # ----------------------------------------------------
        # 在峰值附近做局部频率质心
        #
        # 比直接取FFT格点精度更高
        # ----------------------------------------------------

        left = max(
            0,
            peak_index - 2
        )

        right = min(
            frequency_bins,
            peak_index + 3
        )


        local_mag = column[
            left:right
        ]

        local_freq = frequencies[
            left:right
        ]


        local_power = (
            local_mag ** 2
        )


        power_sum = np.sum(
            local_power
        )


        if power_sum > 0:

            frequency_estimate = (
                np.sum(
                    local_freq
                    * local_power
                )
                /
                power_sum
            )

        else:

            frequency_estimate = (
                frequencies[
                    peak_index
                ]
            )


        ridge_freq.append(
            frequency_estimate
        )


        ridge_strength.append(
            column[
                peak_index
            ]
        )


    ridge_freq = np.asarray(
        ridge_freq,
        dtype=np.float64
    )

    ridge_strength = np.asarray(
        ridge_strength,
        dtype=np.float64
    )


    # ========================================================
    # 5. 加权直线拟合
    #
    # f(t) = k*t + b
    # ========================================================

    weights = (

        ridge_strength

        /

        (
            np.max(ridge_strength)
            + 1e-12
        )
    )


    weights = np.clip(
        weights,
        1e-3,
        None
    )


    slope, intercept = np.polyfit(

        times,

        ridge_freq,

        deg=1,

        w=weights
    )


    # ========================================================
    # 6. 第一次拟合之后去除明显离群点
    # ========================================================

    fitted = (
        slope * times
        + intercept
    )


    residual = (
        ridge_freq
        - fitted
    )


    residual_median = np.median(
        residual
    )


    mad = np.median(

        np.abs(
            residual
            - residual_median
        )
    )


    robust_sigma = (
        1.4826 * mad
    )


    # 至少允许两个FFT bin误差
    minimum_threshold = (
        2 * fs / nfft
    )


    threshold = max(

        3.0 * robust_sigma,

        minimum_threshold
    )


    valid_mask = (

        np.abs(
            residual
            - residual_median
        )

        <= threshold
    )


    # ========================================================
    # 7. 重新拟合
    # ========================================================

    if np.sum(valid_mask) >= 5:

        slope, intercept = np.polyfit(

            times[
                valid_mask
            ],

            ridge_freq[
                valid_mask
            ],

            deg=1,

            w=weights[
                valid_mask
            ]
        )


    # ========================================================
    # 8. 参数计算
    # ========================================================

    # 根据直线：
    #
    # f(t) = slope*t + intercept

    start_freq = (
        intercept
    )

    end_freq = (
        slope * duration
        + intercept
    )


    center_freq = (

        start_freq
        + end_freq

    ) / 2.0


    bandwidth = np.abs(

        end_freq
        - start_freq
    )


    chirp_rate = (
        slope
    )


    chirp_direction = (

        1
        if chirp_rate >= 0

        else -1
    )


    # ========================================================
    # 9. 脊线拟合误差
    # ========================================================

    final_fit = (
        slope * times
        + intercept
    )


    ridge_rmse = np.sqrt(

        np.mean(

            (
                ridge_freq
                - final_fit
            ) ** 2
        )
    )


    # ========================================================
    # 10. 返回结果
    # ========================================================

    return {

        "center_freq_hz":
            float(center_freq),

        "bandwidth_hz":
            float(bandwidth),

        "chirp_rate_hz_per_s":
            float(chirp_rate),

        "chirp_direction":
            int(chirp_direction),

        "start_freq_hz":
            float(start_freq),

        "end_freq_hz":
            float(end_freq),

        "ridge_rmse_hz":
            float(ridge_rmse)
    }