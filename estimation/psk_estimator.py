import numpy as np


# ============================================================
# 1. 抛物线峰值插值
# ============================================================

def _parabolic_offset(values, index):

    if (
        index <= 0
        or
        index >= len(values) - 1
    ):
        return 0.0

    y1 = values[index - 1]
    y2 = values[index]
    y3 = values[index + 1]

    denominator = (
        y1
        - 2 * y2
        + y3
    )

    if abs(denominator) < 1e-20:

        return 0.0

    offset = (
        0.5
        * (y1 - y3)
        / denominator
    )

    return float(offset)


# ============================================================
# 2. 利用自相关得到载频粗估计
# ============================================================

def _coarse_frequency_estimate(
    signal,
    fs,
    lag=6
):

    """
    对PSK信号进行延迟自相关：

    R(L) = sum x[n+L] x*[n]

    码元内部的相位编码会抵消，
    剩下主要是载频造成的相位旋转。

    当前数据库：
        fc ∈ [-1.5 MHz, 1.5 MHz]

    lag=6 时不会发生相位模糊。
    """

    correlation = np.sum(

        signal[lag:]

        * np.conj(
            signal[:-lag]
        )
    )

    phase = np.angle(
        correlation
    )

    frequency = (

        phase
        * fs

        /

        (
            2
            * np.pi
            * lag
        )
    )

    return float(frequency)


# ============================================================
# 3. M次方载频精细估计
# ============================================================

def _refine_frequency_mth_power(
    signal,
    fs,
    modulation_order,
    coarse_frequency,
    nfft=131072,
    search_hz=300_000
):

    """
    BPSK：
        x^2 消除 0/pi 编码

    QPSK：
        x^4 消除
        0/pi/2/pi/3pi/2 编码

    然后在粗估计附近寻找窄带谱峰。
    """

    signal = np.asarray(
        signal
    )

    n = len(signal)

    t = (
        np.arange(n)
        / fs
    )

    # --------------------------------------------------------
    # 先去除粗估计载频
    # --------------------------------------------------------

    baseband = (

        signal

        * np.exp(

            -1j
            * 2
            * np.pi
            * coarse_frequency
            * t
        )
    )

    # --------------------------------------------------------
    # M次方去PSK调制
    # --------------------------------------------------------

    powered = (
        baseband
        ** modulation_order
    )

    window = np.hanning(n)

    spectrum = np.fft.fftshift(

        np.fft.fft(

            powered * window,

            n=nfft
        )
    )

    frequencies = np.fft.fftshift(

        np.fft.fftfreq(

            nfft,

            d=1 / fs
        )
    )

    # --------------------------------------------------------
    # 只在0附近搜索
    #
    # 避免低SNR时被远处噪声峰骗走
    # --------------------------------------------------------

    search_mask = (

        np.abs(frequencies)
        <= search_hz
    )

    valid_indices = np.where(
        search_mask
    )[0]

    log_magnitude = np.log(

        np.abs(spectrum)

        + 1e-15
    )

    local_values = log_magnitude[
        valid_indices
    ]

    local_peak = int(
        np.argmax(local_values)
    )

    peak_index = int(
        valid_indices[
            local_peak
        ]
    )

    # --------------------------------------------------------
    # 亚频点插值
    # --------------------------------------------------------

    offset = _parabolic_offset(

        log_magnitude,

        peak_index
    )

    frequency_resolution = (

        fs / nfft
    )

    powered_residual_frequency = (

        frequencies[
            peak_index
        ]

        + offset
        * frequency_resolution
    )

    # M次方后频率也乘了M
    residual_frequency = (

        powered_residual_frequency

        / modulation_order
    )

    refined_frequency = (

        coarse_frequency

        + residual_frequency
    )

    return float(
        refined_frequency
    )


# ============================================================
# 4. 对一个候选码元数评分
# ============================================================

def _evaluate_symbol_candidate(
    baseband,
    modulation_order,
    symbol_count
):

    n = len(baseband)

    if n % symbol_count != 0:

        raise ValueError(
            "采样点数必须能够被码元数整除"
        )

    samples_per_symbol = (

        n // symbol_count
    )

    # --------------------------------------------------------
    # 分段
    # --------------------------------------------------------

    segments = baseband.reshape(

        symbol_count,

        samples_per_symbol
    )

    # 每个码元求复均值
    segment_mean = np.mean(

        segments,

        axis=1
    )

    unit_segments = (

        segment_mean

        /

        (
            np.abs(segment_mean)
            + 1e-12
        )
    )


    # ========================================================
    # 估计公共初始相位
    #
    # z^M 可以消除PSK码元相位
    # ========================================================

    common_phase = (

        np.angle(

            np.mean(

                unit_segments
                ** modulation_order
            )
        )

        /

        modulation_order
    )


    phase_step = (

        2
        * np.pi

        /

        modulation_order
    )


    # ========================================================
    # 判断每段属于哪个PSK状态
    # ========================================================

    relative_phase = np.mod(

        np.angle(

            segment_mean

            * np.exp(
                -1j * common_phase
            )
        ),

        2 * np.pi
    )


    symbol_sequence = (

        np.rint(

            relative_phase
            / phase_step
        )

        .astype(np.int64)

        % modulation_order
    )


    # ========================================================
    # 根据估计码元重建理想相位序列
    # ========================================================

    reconstructed_symbol_phase = (

        common_phase

        +

        symbol_sequence
        * phase_step
    )


    reconstructed = np.repeat(

        np.exp(

            1j
            * reconstructed_symbol_phase
        ),

        samples_per_symbol
    )


    # ========================================================
    # 计算拟合程度
    #
    # 越接近1越好
    # ========================================================

    residual = (

        baseband

        * np.conj(
            reconstructed
        )
    )


    coherence = (

        np.abs(
            np.mean(residual)
        )

        /

        (
            np.mean(
                np.abs(baseband)
            )

            + 1e-12
        )
    )


    return {

        "symbol_count":
            int(symbol_count),

        "samples_per_symbol":
            int(samples_per_symbol),

        "symbol_sequence":
            symbol_sequence,

        "common_phase_rad":
            float(common_phase),

        "score":
            float(coherence)
    }


# ============================================================
# 5. PSK参数估计
# ============================================================

def estimate_psk_parameters(
    signal,
    fs,
    modulation="BPSK",
    symbol_count_candidates=(8, 16, 32),
    score_tolerance=0.01
):

    """
    BPSK / QPSK 参数估计。

    Parameters
    ----------
    signal:
        复数IQ

    fs:
        采样率

    modulation:
        "BPSK"
        "QPSK"

    symbol_count_candidates:
        当前数据库的候选码长

    Returns
    -------
    dict
    """

    signal = np.asarray(
        signal
    )

    if signal.ndim != 1:

        raise ValueError(
            "signal必须是一维复数IQ"
        )


    modulation_upper = (
        modulation.upper()
    )


    if modulation_upper == "BPSK":

        modulation_order = 2

    elif modulation_upper == "QPSK":

        modulation_order = 4

    else:

        raise ValueError(
            "当前只支持 BPSK 和 QPSK"
        )


    n = len(signal)

    duration = (
        n / fs
    )


    # ========================================================
    # 6. 载频粗估计
    # ========================================================

    coarse_frequency = (
        _coarse_frequency_estimate(

            signal,

            fs,

            lag=6
        )
    )


    # ========================================================
    # 7. 载频精确估计
    # ========================================================

    center_frequency = (
        _refine_frequency_mth_power(

            signal,

            fs,

            modulation_order,

            coarse_frequency
        )
    )


    # ========================================================
    # 8. 去载频
    # ========================================================

    t = (
        np.arange(n)
        / fs
    )


    baseband = (

        signal

        * np.exp(

            -1j
            * 2
            * np.pi
            * center_frequency
            * t
        )
    )


    # ========================================================
    # 9. 依次尝试 8 / 16 / 32
    # ========================================================

    candidates = []


    for symbol_count in (
        symbol_count_candidates
    ):

        candidate = (
            _evaluate_symbol_candidate(

                baseband,

                modulation_order,

                symbol_count
            )
        )

        candidates.append(
            candidate
        )


    # ========================================================
    # 10. 选择码元数量
    #
    # 如果大码长只是把一个码元重复切成两段，
    # 分数可能和真正码长几乎一样。
    #
    # 因此：
    # 在接近最佳分数的候选中，
    # 优先选择最小码长。
    # ========================================================

    best_score = max(

        item["score"]

        for item in candidates
    )


    eligible_candidates = [

        item

        for item in candidates

        if item["score"]

        >=
        (
            best_score
            - score_tolerance
        )
    ]


    selected = min(

        eligible_candidates,

        key=lambda item:
            item["symbol_count"]
    )


    symbol_count = (
        selected[
            "symbol_count"
        ]
    )


    symbol_rate = (

        symbol_count

        / duration
    )


    symbol_duration = (

        duration

        / symbol_count
    )


    return {

        "modulation":
            modulation_upper,

        "modulation_order":
            modulation_order,

        "center_freq_hz":
            float(
                center_frequency
            ),

        "symbol_count":
            int(symbol_count),

        "symbol_rate_baud":
            float(symbol_rate),

        "symbol_duration_s":
            float(symbol_duration),

        "symbol_sequence":
            selected[
                "symbol_sequence"
            ],

        "common_phase_rad":
            selected[
                "common_phase_rad"
            ],

        "fit_score":
            selected["score"],

        "candidate_scores": {

            item["symbol_count"]:
                item["score"]

            for item in candidates
        }
    }