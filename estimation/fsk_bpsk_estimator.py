import numpy as np
from scipy.signal import find_peaks


# ============================================================
# 1. 抛物线插值
# ============================================================

def _parabolic_offset(values, index):

    if index <= 0 or index >= len(values) - 1:
        return 0.0

    y1 = values[index - 1]
    y2 = values[index]
    y3 = values[index + 1]

    denominator = (
        y1
        - 2.0 * y2
        + y3
    )

    if abs(denominator) < 1e-20:
        return 0.0

    return float(
        0.5
        * (y1 - y3)
        / denominator
    )


# ============================================================
# 2. 平方谱粗估计 f1 / f2
# ============================================================

def _estimate_two_frequencies(
    signal,
    fs,
    nfft=131072
):

    n = len(signal)

    # BPSK平方以后消失
    squared = signal ** 2

    window = np.hanning(n)

    spectrum = np.fft.fftshift(
        np.fft.fft(
            squared * window,
            n=nfft
        )
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            nfft,
            d=1 / fs
        )
    )

    magnitude = np.abs(
        spectrum
    )

    # 平方后的频率范围
    mask = (
        np.abs(frequencies)
        <= 8e6
    )

    valid_indices = np.where(
        mask
    )[0]

    local_magnitude = magnitude[
        valid_indices
    ]

    frequency_resolution = (
        fs / nfft
    )

    min_distance = max(
        1,
        int(
            0.8e6
            /
            frequency_resolution
        )
    )

    peaks, _ = find_peaks(
        local_magnitude,
        distance=min_distance
    )

    if len(peaks) < 2:

        sorted_indices = np.argsort(
            local_magnitude
        )[::-1]

        selected = []

        for candidate in sorted_indices:

            candidate_frequency = frequencies[
                valid_indices[candidate]
            ]

            if len(selected) == 0:

                selected.append(
                    candidate
                )

            else:

                first_frequency = frequencies[
                    valid_indices[
                        selected[0]
                    ]
                ]

                if abs(
                    candidate_frequency
                    - first_frequency
                ) >= 0.8e6:

                    selected.append(
                        candidate
                    )

                    break

        if len(selected) < 2:

            raise RuntimeError(
                "无法检测两个FSK频率"
            )

        peaks = np.asarray(
            selected
        )

    peak_strength = local_magnitude[
        peaks
    ]

    strongest = np.argsort(
        peak_strength
    )[::-1][:2]

    selected_peaks = peaks[
        strongest
    ]

    log_magnitude = np.log(
        magnitude + 1e-15
    )

    powered_frequencies = []

    for local_peak in selected_peaks:

        global_index = valid_indices[
            local_peak
        ]

        offset = _parabolic_offset(
            log_magnitude,
            global_index
        )

        frequency = (
            frequencies[global_index]
            +
            offset
            * frequency_resolution
        )

        powered_frequencies.append(
            frequency
        )

    # 平方后频率是原来的2倍
    original = (
        np.asarray(
            powered_frequencies
        )
        / 2.0
    )

    original = np.sort(
        original
    )

    return (
        float(original[0]),
        float(original[1])
    )


# ============================================================
# 3. 判断每个码元属于 f1 还是 f2
# ============================================================

def _evaluate_fsk_candidate(
    signal,
    fs,
    f1,
    f2,
    symbol_count
):

    n = len(signal)

    if n % symbol_count != 0:

        raise ValueError(
            "采样点数不能被码元数整除"
        )

    samples_per_symbol = (
        n // symbol_count
    )

    t = (
        np.arange(n)
        / fs
    )

    fsk_code = []

    quality = []

    for symbol_index in range(
        symbol_count
    ):

        start = (
            symbol_index
            * samples_per_symbol
        )

        end = (
            start
            + samples_per_symbol
        )

        segment = signal[
            start:end
        ]

        segment_t = t[
            start:end
        ]

        c1 = np.mean(
            segment
            * np.exp(
                -1j
                * 2
                * np.pi
                * f1
                * segment_t
            )
        )

        c2 = np.mean(
            segment
            * np.exp(
                -1j
                * 2
                * np.pi
                * f2
                * segment_t
            )
        )

        a1 = abs(c1)
        a2 = abs(c2)

        if a1 >= a2:

            fsk_code.append(0)
            best = a1

        else:

            fsk_code.append(1)
            best = a2

        normalization = (
            np.mean(
                np.abs(segment)
            )
            + 1e-12
        )

        quality.append(
            best / normalization
        )

    return {
        "symbol_count":
            int(symbol_count),

        "samples_per_symbol":
            int(samples_per_symbol),

        "fsk_code":
            np.asarray(
                fsk_code,
                dtype=np.int64
            ),

        "score":
            float(
                np.mean(
                    quality
                )
            )
    }


# ============================================================
# 4. 对某一路FSK频率进一步精细搜索
# ============================================================

def _refine_one_frequency(
    signal,
    fs,
    symbol_count,
    fsk_code,
    state,
    initial_frequency,
    search_hz=600_000
):

    n = len(signal)

    samples_per_symbol = (
        n // symbol_count
    )

    t = (
        np.arange(n)
        / fs
    )

    segments = []

    for symbol_index in range(
        symbol_count
    ):

        if fsk_code[
            symbol_index
        ] != state:

            continue

        start = (
            symbol_index
            * samples_per_symbol
        )

        end = (
            start
            + samples_per_symbol
        )

        segments.append(
            (start, end)
        )

    # 没有观测到这个频率
    if len(segments) == 0:

        return float(
            initial_frequency
        )


    # ========================================================
    # 评分函数
    #
    # signal² 消除BPSK
    #
    # 每一个码元分别取幅度，
    # 因而不要求不同码元之间载波相位连续。
    # ========================================================

    def score_frequency(
        frequency
    ):

        scores = []

        for start, end in segments:

            segment = signal[
                start:end
            ]

            segment_t = t[
                start:end
            ]

            value = np.mean(

                segment ** 2

                * np.exp(

                    -1j
                    * 4
                    * np.pi
                    * frequency
                    * segment_t
                )
            )

            scores.append(
                abs(value)
            )

        return float(
            np.mean(scores)
        )


    # ========================================================
    # 第一级粗搜索
    # ========================================================

    coarse_offsets = np.arange(

        -search_hz,

        search_hz + 1,

        10_000.0
    )

    coarse_scores = []

    for offset in coarse_offsets:

        coarse_scores.append(

            score_frequency(

                initial_frequency
                + offset
            )
        )

    best_index = int(
        np.argmax(
            coarse_scores
        )
    )

    coarse_best = (

        initial_frequency

        + coarse_offsets[
            best_index
        ]
    )


    # ========================================================
    # 第二级精搜索
    # ========================================================

    fine_offsets = np.arange(

        -15_000.0,

        15_000.0 + 1,

        250.0
    )

    fine_scores = []

    for offset in fine_offsets:

        fine_scores.append(

            score_frequency(

                coarse_best
                + offset
            )
        )

    best_index = int(
        np.argmax(
            fine_scores
        )
    )

    final_frequency = (

        coarse_best

        + fine_offsets[
            best_index
        ]
    )

    return float(
        final_frequency
    )


# ============================================================
# 5. 选择码元数
# ============================================================

def _select_symbol_count(
    signal,
    fs,
    f1,
    f2,
    candidates,
    score_tolerance
):

    results = []

    for symbol_count in candidates:

        result = (
            _evaluate_fsk_candidate(
                signal,
                fs,
                f1,
                f2,
                symbol_count
            )
        )

        results.append(
            result
        )

    best_score = max(

        item["score"]

        for item in results
    )

    eligible = [

        item

        for item in results

        if item["score"]
        >=
        best_score
        -
        score_tolerance
    ]

    return min(

        eligible,

        key=lambda item:
            item["symbol_count"]
    )


# ============================================================
# 6. 给定FSK载波，恢复BPSK
# ============================================================

def _recover_bpsk(
    signal,
    fs,
    f1,
    f2,
    fsk_code,
    symbol_count,
    carrier_mode
):

    n = len(signal)

    samples_per_symbol = (
        n // symbol_count
    )

    t = (
        np.arange(n)
        / fs
    )

    # 每个采样点对应的频率
    instantaneous_frequency = np.zeros(
        n,
        dtype=np.float64
    )

    for symbol_index in range(
        symbol_count
    ):

        start = (
            symbol_index
            * samples_per_symbol
        )

        end = (
            start
            + samples_per_symbol
        )

        if fsk_code[
            symbol_index
        ] == 0:

            frequency = f1

        else:

            frequency = f2

        instantaneous_frequency[
            start:end
        ] = frequency


    # ========================================================
    # 两种可能的FSK相位模型
    # ========================================================

    if carrier_mode == "continuous":

        carrier_phase = (

            2
            * np.pi
            * np.cumsum(
                instantaneous_frequency
            )
            / fs
        )

    elif carrier_mode == "absolute":

        carrier_phase = (

            2
            * np.pi
            * instantaneous_frequency
            * t
        )

    else:

        raise ValueError(
            "未知carrier_mode"
        )


    baseband = (

        signal

        * np.exp(
            -1j
            * carrier_phase
        )
    )


    # ========================================================
    # 每个码元求复均值
    # ========================================================

    segments = baseband.reshape(

        symbol_count,

        samples_per_symbol
    )

    segment_mean = np.mean(

        segments,

        axis=1
    )


    unit_values = (

        segment_mean

        /

        (
            np.abs(
                segment_mean
            )

            + 1e-12
        )
    )


    # ========================================================
    # 平方消掉BPSK ±π
    # 得到公共相位
    # ========================================================

    common_phase = (

        np.angle(

            np.mean(
                unit_values ** 2
            )
        )

        / 2.0
    )


    corrected = (

        segment_mean

        * np.exp(
            -1j
            * common_phase
        )
    )


    bpsk_code = (

        np.real(
            corrected
        )

        < 0
    ).astype(
        np.int64
    )


    # ========================================================
    # 拟合评分
    # ========================================================

    bpsk_sign = (

        1.0

        - 2.0
        * bpsk_code
    )


    reconstructed = np.repeat(

        bpsk_sign

        * np.exp(
            1j
            * common_phase
        ),

        samples_per_symbol
    )


    residual = (

        baseband

        * np.conj(
            reconstructed
        )
    )


    fit_score = (

        abs(
            np.mean(
                residual
            )
        )

        /

        (
            np.mean(
                np.abs(
                    baseband
                )
            )

            + 1e-12
        )
    )


    return {

        "bpsk_code":
            bpsk_code,

        "common_phase_rad":
            float(common_phase),

        "fit_score":
            float(fit_score),

        "carrier_mode":
            carrier_mode
    }


# ============================================================
# 7. 总参数估计
# ============================================================

def estimate_fsk_bpsk_parameters(
    signal,
    fs,
    symbol_count_candidates=(8, 16, 32),
    score_tolerance=0.01
):

    signal = np.asarray(
        signal
    )

    if signal.ndim != 1:

        raise ValueError(
            "signal必须是一维复数IQ"
        )


    n = len(signal)

    duration = (
        n / fs
    )


    # ========================================================
    # 第一步：频率粗估计
    # ========================================================

    f1, f2 = (
        _estimate_two_frequencies(
            signal,
            fs
        )
    )


    # ========================================================
    # 第二步：第一次码长判断
    # ========================================================

    selected = (
        _select_symbol_count(

            signal,
            fs,
            f1,
            f2,
            symbol_count_candidates,
            score_tolerance
        )
    )


    symbol_count = (
        selected[
            "symbol_count"
        ]
    )

    fsk_code = (
        selected[
            "fsk_code"
        ]
    )


    # ========================================================
    # 第三步：根据跳频序列精细校正f1/f2
    # ========================================================

    f1 = _refine_one_frequency(

        signal,
        fs,
        symbol_count,
        fsk_code,
        0,
        f1
    )


    f2 = _refine_one_frequency(

        signal,
        fs,
        symbol_count,
        fsk_code,
        1,
        f2
    )


    if f1 > f2:

        f1, f2 = f2, f1


    # ========================================================
    # 第四步：用精确频率重新判断码元数
    # ========================================================

    selected = (
        _select_symbol_count(

            signal,
            fs,
            f1,
            f2,
            symbol_count_candidates,
            score_tolerance
        )
    )


    symbol_count = (
        selected[
            "symbol_count"
        ]
    )

    fsk_code = (
        selected[
            "fsk_code"
        ]
    )


    # ========================================================
    # 再精炼一次频率
    # ========================================================

    f1 = _refine_one_frequency(

        signal,
        fs,
        symbol_count,
        fsk_code,
        0,
        f1,
        search_hz=100_000
    )


    f2 = _refine_one_frequency(

        signal,
        fs,
        symbol_count,
        fsk_code,
        1,
        f2,
        search_hz=100_000
    )


    if f1 > f2:

        f1, f2 = f2, f1


    # ========================================================
    # 最终FSK序列
    # ========================================================

    final_fsk = (
        _evaluate_fsk_candidate(

            signal,
            fs,
            f1,
            f2,
            symbol_count
        )
    )


    fsk_code = (
        final_fsk[
            "fsk_code"
        ]
    )


    # ========================================================
    # 第五步：恢复BPSK
    #
    # 两种FSK相位模型都试
    # ========================================================

    continuous_result = (
        _recover_bpsk(

            signal,
            fs,
            f1,
            f2,
            fsk_code,
            symbol_count,
            "continuous"
        )
    )


    absolute_result = (
        _recover_bpsk(

            signal,
            fs,
            f1,
            f2,
            fsk_code,
            symbol_count,
            "absolute"
        )
    )


    if (
        continuous_result[
            "fit_score"
        ]
        >=
        absolute_result[
            "fit_score"
        ]
    ):

        bpsk_result = (
            continuous_result
        )

    else:

        bpsk_result = (
            absolute_result
        )


    # ========================================================
    # 最终参数
    # ========================================================

    center_frequency = (
        f1 + f2
    ) / 2.0


    frequency_separation = (
        abs(
            f2 - f1
        )
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

        "center_freq_hz":
            float(
                center_frequency
            ),

        "f1_hz":
            float(f1),

        "f2_hz":
            float(f2),

        "freq_sep_hz":
            float(
                frequency_separation
            ),

        "symbol_count":
            int(
                symbol_count
            ),

        "symbol_rate_baud":
            float(
                symbol_rate
            ),

        "symbol_duration_s":
            float(
                symbol_duration
            ),

        "fsk_code":
            fsk_code,

        "bpsk_code":
            bpsk_result[
                "bpsk_code"
            ],

        "common_phase_rad":
            bpsk_result[
                "common_phase_rad"
            ],

        "carrier_mode":
            bpsk_result[
                "carrier_mode"
            ],

        "fit_score":
            bpsk_result[
                "fit_score"
            ]
    }