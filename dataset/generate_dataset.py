import json
from pathlib import Path

import numpy as np


# ============================================================
# 1. 全局配置
# ============================================================

# 随机种子，保证每次生成结果可复现
SEED = 2026

# 采样率：20 MHz
FS = 20_000_000

# 每条信号采样点数
N = 2048

# 一条信号的总持续时间
DURATION = N / FS

# SNR：0 ~ 10 dB，全部为整数
SNR_VALUES = list(range(0, 11))

# 每一种「类别 × SNR」生成多少条样本
SAMPLES_PER_CLASS_SNR = 120

# 每个类别/SNR内部进行 70% / 15% / 15% 划分
TRAIN_PER_GROUP = 84
VAL_PER_GROUP = 18
TEST_PER_GROUP = 18

# 支持的码元数量
SYMBOL_COUNTS = [8, 16, 32]

# 6种信号
CLASS_NAMES = [
    "LFM",
    "BPSK",
    "QPSK",
    "LFM_BPSK",
    "LFM_QPSK",
    "2FSK_BPSK"
]

CLASS_TO_LABEL = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# 2. Ground Truth 参数名称
# ============================================================

# 所有信号共用一个参数表
# 不适用的参数保存为 NaN
PARAM_NAMES = [
    "amplitude",             # 信号幅度
    "center_freq_hz",        # 中心频率
    "bandwidth_hz",          # LFM带宽
    "chirp_rate_hz_per_s",   # 调频斜率
    "symbol_count",          # 码元数量
    "symbol_rate_baud",      # 码元速率
    "f1_hz",                 # 2FSK频率1
    "f2_hz",                 # 2FSK频率2
    "freq_sep_hz",           # FSK频率间隔
    "chirp_direction",       # LFM方向：+1/-1
    "initial_phase_rad"      # 初始相位
]


# ============================================================
# 3. 工具函数
# ============================================================

def random_symbol_sequence(rng, length, alphabet_size):
    """
    生成随机码元序列。

    为了避免出现：
        00000000
        11111111

    这种完全没有跳变的退化信号，
    强制序列至少包含两种不同码元。
    """

    while True:
        sequence = rng.integers(
            low=0,
            high=alphabet_size,
            size=length,
            dtype=np.int8
        )

        if len(np.unique(sequence)) >= 2:
            return sequence


def expand_symbols(symbols):
    """
    把码元序列扩展到 N 个采样点。

    例如：

        [0, 1, 0, 1]

    变成：

        [0,0,...,0,
         1,1,...,1,
         0,0,...,0,
         1,1,...,1]
    """

    symbol_count = len(symbols)

    samples_per_symbol = N // symbol_count

    return np.repeat(
        symbols,
        samples_per_symbol
    )


def add_awgn(signal, snr_db, rng):
    """
    给复数IQ信号加入复高斯白噪声。

    SNR单位：dB
    """

    signal_power = np.mean(
        np.abs(signal) ** 2
    )

    snr_linear = 10 ** (snr_db / 10.0)

    noise_power = signal_power / snr_linear

    noise_real = rng.standard_normal(N)
    noise_imag = rng.standard_normal(N)

    noise = np.sqrt(noise_power / 2) * (
        noise_real + 1j * noise_imag
    )

    noisy_signal = signal + noise

    return noisy_signal


def complex_to_iq(signal):
    """
    将复数信号：

        x = I + jQ

    转换为神经网络更方便使用的：

        shape = (2, N)

    第0行：I
    第1行：Q
    """

    iq = np.stack(
        [
            signal.real,
            signal.imag
        ],
        axis=0
    )

    return iq.astype(np.float32)


def create_empty_params():
    """
    创建一条空参数记录。
    不适用的数据填充 NaN。
    """

    return np.full(
        len(PARAM_NAMES),
        np.nan,
        dtype=np.float64
    )


def create_empty_code():
    """
    最大码长设置为32。

    -1 表示当前位置没有码元。
    """

    return np.full(
        32,
        -1,
        dtype=np.int8
    )


# ============================================================
# 4. LFM
# ============================================================

def generate_lfm(snr_db, rng):
    """
    生成LFM线性调频信号。
    """

    t = np.arange(N) / FS

    # 以信号中间为时间零点
    t_centered = t - DURATION / 2

    # 幅度
    amplitude = rng.uniform(
        0.8,
        1.2
    )

    # 中心频率
    center_freq = rng.uniform(
        -1.5e6,
        1.5e6
    )

    # 带宽：1 ~ 6 MHz
    bandwidth = rng.uniform(
        1e6,
        6e6
    )

    # 上扫频或者下扫频
    chirp_direction = rng.choice(
        [-1, 1]
    )

    # 调频斜率
    chirp_rate = (
        chirp_direction
        * bandwidth
        / DURATION
    )

    # 初始相位
    initial_phase = rng.uniform(
        0,
        2 * np.pi
    )

    phase = (
        2 * np.pi * center_freq * t
        + np.pi * chirp_rate * t_centered ** 2
        + initial_phase
    )

    signal = amplitude * np.exp(
        1j * phase
    )

    signal = add_awgn(
        signal,
        snr_db,
        rng
    )

    params = create_empty_params()

    params[0] = amplitude
    params[1] = center_freq
    params[2] = bandwidth
    params[3] = chirp_rate
    params[9] = chirp_direction
    params[10] = initial_phase

    code = create_empty_code()
    fsk_code = create_empty_code()

    return signal, params, code, fsk_code


# ============================================================
# 5. BPSK
# ============================================================

def generate_bpsk(snr_db, rng):

    t = np.arange(N) / FS

    amplitude = rng.uniform(
        0.8,
        1.2
    )

    center_freq = rng.uniform(
        -1.5e6,
        1.5e6
    )

    initial_phase = rng.uniform(
        0,
        2 * np.pi
    )

    symbol_count = int(
        rng.choice(SYMBOL_COUNTS)
    )

    # BPSK：
    #
    # 0 -> 0°
    # 1 -> 180°
    #
    code_sequence = random_symbol_sequence(
        rng,
        symbol_count,
        2
    )

    phase_symbols = (
        code_sequence * np.pi
    )

    expanded_phase = expand_symbols(
        phase_symbols
    )

    phase = (
        2 * np.pi * center_freq * t
        + expanded_phase
        + initial_phase
    )

    signal = amplitude * np.exp(
        1j * phase
    )

    signal = add_awgn(
        signal,
        snr_db,
        rng
    )

    symbol_rate = (
        symbol_count / DURATION
    )

    params = create_empty_params()

    params[0] = amplitude
    params[1] = center_freq
    params[4] = symbol_count
    params[5] = symbol_rate
    params[10] = initial_phase

    code = create_empty_code()

    code[:symbol_count] = (
        code_sequence
    )

    fsk_code = create_empty_code()

    return signal, params, code, fsk_code


# ============================================================
# 6. QPSK
# ============================================================

def generate_qpsk(snr_db, rng):

    t = np.arange(N) / FS

    amplitude = rng.uniform(
        0.8,
        1.2
    )

    center_freq = rng.uniform(
        -1.5e6,
        1.5e6
    )

    initial_phase = rng.uniform(
        0,
        2 * np.pi
    )

    symbol_count = int(
        rng.choice(SYMBOL_COUNTS)
    )

    # QPSK：
    #
    # 0 -> 0°
    # 1 -> 90°
    # 2 -> 180°
    # 3 -> 270°
    #
    code_sequence = random_symbol_sequence(
        rng,
        symbol_count,
        4
    )

    phase_symbols = (
        code_sequence
        * np.pi
        / 2
    )

    expanded_phase = expand_symbols(
        phase_symbols
    )

    phase = (
        2 * np.pi * center_freq * t
        + expanded_phase
        + initial_phase
    )

    signal = amplitude * np.exp(
        1j * phase
    )

    signal = add_awgn(
        signal,
        snr_db,
        rng
    )

    symbol_rate = (
        symbol_count / DURATION
    )

    params = create_empty_params()

    params[0] = amplitude
    params[1] = center_freq
    params[4] = symbol_count
    params[5] = symbol_rate
    params[10] = initial_phase

    code = create_empty_code()

    code[:symbol_count] = (
        code_sequence
    )

    fsk_code = create_empty_code()

    return signal, params, code, fsk_code


# ============================================================
# 7. LFM-BPSK
# ============================================================

def generate_lfm_bpsk(snr_db, rng):

    t = np.arange(N) / FS

    t_centered = (
        t - DURATION / 2
    )

    amplitude = rng.uniform(
        0.8,
        1.2
    )

    center_freq = rng.uniform(
        -1.5e6,
        1.5e6
    )

    bandwidth = rng.uniform(
        1e6,
        6e6
    )

    chirp_direction = rng.choice(
        [-1, 1]
    )

    chirp_rate = (
        chirp_direction
        * bandwidth
        / DURATION
    )

    initial_phase = rng.uniform(
        0,
        2 * np.pi
    )

    symbol_count = int(
        rng.choice(SYMBOL_COUNTS)
    )

    code_sequence = random_symbol_sequence(
        rng,
        symbol_count,
        2
    )

    bpsk_phase = (
        code_sequence * np.pi
    )

    expanded_phase = expand_symbols(
        bpsk_phase
    )

    phase = (
        2 * np.pi * center_freq * t
        + np.pi * chirp_rate * t_centered ** 2
        + expanded_phase
        + initial_phase
    )

    signal = amplitude * np.exp(
        1j * phase
    )

    signal = add_awgn(
        signal,
        snr_db,
        rng
    )

    symbol_rate = (
        symbol_count / DURATION
    )

    params = create_empty_params()

    params[0] = amplitude
    params[1] = center_freq
    params[2] = bandwidth
    params[3] = chirp_rate
    params[4] = symbol_count
    params[5] = symbol_rate
    params[9] = chirp_direction
    params[10] = initial_phase

    code = create_empty_code()

    code[:symbol_count] = (
        code_sequence
    )

    fsk_code = create_empty_code()

    return signal, params, code, fsk_code


# ============================================================
# 8. LFM-QPSK
# ============================================================

def generate_lfm_qpsk(snr_db, rng):

    t = np.arange(N) / FS

    t_centered = (
        t - DURATION / 2
    )

    amplitude = rng.uniform(
        0.8,
        1.2
    )

    center_freq = rng.uniform(
        -1.5e6,
        1.5e6
    )

    bandwidth = rng.uniform(
        1e6,
        6e6
    )

    chirp_direction = rng.choice(
        [-1, 1]
    )

    chirp_rate = (
        chirp_direction
        * bandwidth
        / DURATION
    )

    initial_phase = rng.uniform(
        0,
        2 * np.pi
    )

    symbol_count = int(
        rng.choice(SYMBOL_COUNTS)
    )

    code_sequence = random_symbol_sequence(
        rng,
        symbol_count,
        4
    )

    qpsk_phase = (
        code_sequence
        * np.pi
        / 2
    )

    expanded_phase = expand_symbols(
        qpsk_phase
    )

    phase = (
        2 * np.pi * center_freq * t
        + np.pi * chirp_rate * t_centered ** 2
        + expanded_phase
        + initial_phase
    )

    signal = amplitude * np.exp(
        1j * phase
    )

    signal = add_awgn(
        signal,
        snr_db,
        rng
    )

    symbol_rate = (
        symbol_count / DURATION
    )

    params = create_empty_params()

    params[0] = amplitude
    params[1] = center_freq
    params[2] = bandwidth
    params[3] = chirp_rate
    params[4] = symbol_count
    params[5] = symbol_rate
    params[9] = chirp_direction
    params[10] = initial_phase

    code = create_empty_code()

    code[:symbol_count] = (
        code_sequence
    )

    fsk_code = create_empty_code()

    return signal, params, code, fsk_code


# ============================================================
# 9. 2FSK-BPSK
# ============================================================

def generate_2fsk_bpsk(snr_db, rng):

    amplitude = rng.uniform(
        0.8,
        1.2
    )

    center_freq = rng.uniform(
        -1.5e6,
        1.5e6
    )

    initial_phase = rng.uniform(
        0,
        2 * np.pi
    )

    symbol_count = int(
        rng.choice(SYMBOL_COUNTS)
    )

    # BPSK序列
    bpsk_sequence = random_symbol_sequence(
        rng,
        symbol_count,
        2
    )

    # FSK频率选择序列
    fsk_sequence = random_symbol_sequence(
        rng,
        symbol_count,
        2
    )

    # 两个FSK频率之间间隔
    freq_sep = rng.uniform(
        0.8e6,
        3.0e6
    )

    f1 = (
        center_freq
        - freq_sep / 2
    )

    f2 = (
        center_freq
        + freq_sep / 2
    )

    # 每个码元选择一个频率
    symbol_frequencies = np.where(
        fsk_sequence == 0,
        f1,
        f2
    )

    instantaneous_frequency = expand_symbols(
        symbol_frequencies
    )

    # 对瞬时频率积分得到相位
    frequency_phase = (
        2
        * np.pi
        * np.cumsum(instantaneous_frequency)
        / FS
    )

    # BPSK部分
    bpsk_phase = (
        bpsk_sequence * np.pi
    )

    expanded_bpsk_phase = expand_symbols(
        bpsk_phase
    )

    phase = (
        frequency_phase
        + expanded_bpsk_phase
        + initial_phase
    )

    signal = amplitude * np.exp(
        1j * phase
    )

    signal = add_awgn(
        signal,
        snr_db,
        rng
    )

    symbol_rate = (
        symbol_count / DURATION
    )

    params = create_empty_params()

    params[0] = amplitude
    params[1] = center_freq
    params[4] = symbol_count
    params[5] = symbol_rate
    params[6] = f1
    params[7] = f2
    params[8] = freq_sep
    params[10] = initial_phase

    code = create_empty_code()

    code[:symbol_count] = (
        bpsk_sequence
    )

    fsk_code = create_empty_code()

    fsk_code[:symbol_count] = (
        fsk_sequence
    )

    return signal, params, code, fsk_code


# ============================================================
# 10. 根据类别调用相应生成器
# ============================================================

def generate_signal(class_name, snr_db, rng):

    if class_name == "LFM":

        return generate_lfm(
            snr_db,
            rng
        )

    elif class_name == "BPSK":

        return generate_bpsk(
            snr_db,
            rng
        )

    elif class_name == "QPSK":

        return generate_qpsk(
            snr_db,
            rng
        )

    elif class_name == "LFM_BPSK":

        return generate_lfm_bpsk(
            snr_db,
            rng
        )

    elif class_name == "LFM_QPSK":

        return generate_lfm_qpsk(
            snr_db,
            rng
        )

    elif class_name == "2FSK_BPSK":

        return generate_2fsk_bpsk(
            snr_db,
            rng
        )

    else:

        raise ValueError(
            f"未知信号类型：{class_name}"
        )


# ============================================================
# 11. 创建一个数据集
# ============================================================

def create_dataset(
    split_name,
    samples_per_group,
    rng
):

    class_count = len(CLASS_NAMES)

    snr_count = len(SNR_VALUES)

    total_samples = (
        class_count
        * snr_count
        * samples_per_group
    )

    print()
    print("=" * 60)

    print(
        f"正在生成 {split_name} 数据集"
    )

    print(
        f"总样本数量：{total_samples}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # 预先申请内存
    # --------------------------------------------------------

    X = np.zeros(
        (
            total_samples,
            2,
            N
        ),
        dtype=np.float32
    )

    y = np.zeros(
        total_samples,
        dtype=np.int64
    )

    snr_array = np.zeros(
        total_samples,
        dtype=np.int16
    )

    params_array = np.zeros(
        (
            total_samples,
            len(PARAM_NAMES)
        ),
        dtype=np.float64
    )

    code_array = np.zeros(
        (
            total_samples,
            32
        ),
        dtype=np.int8
    )

    fsk_code_array = np.zeros(
        (
            total_samples,
            32
        ),
        dtype=np.int8
    )

    index = 0

    # --------------------------------------------------------
    # 每个类别，每个SNR都生成相同数量
    # --------------------------------------------------------

    for class_name in CLASS_NAMES:

        label = CLASS_TO_LABEL[
            class_name
        ]

        print(
            f"\n正在生成：{class_name}"
        )

        for snr_db in SNR_VALUES:

            for _ in range(
                samples_per_group
            ):

                (
                    signal,
                    params,
                    code,
                    fsk_code
                ) = generate_signal(
                    class_name,
                    snr_db,
                    rng
                )

                X[index] = complex_to_iq(
                    signal
                )

                y[index] = label

                snr_array[index] = (
                    snr_db
                )

                params_array[index] = (
                    params
                )

                code_array[index] = (
                    code
                )

                fsk_code_array[index] = (
                    fsk_code
                )

                index += 1

            print(
                f"SNR = {snr_db:2d} dB 完成"
            )

    # --------------------------------------------------------
    # 打乱数据顺序
    # --------------------------------------------------------

    permutation = rng.permutation(
        total_samples
    )

    X = X[permutation]

    y = y[permutation]

    snr_array = snr_array[
        permutation
    ]

    params_array = params_array[
        permutation
    ]

    code_array = code_array[
        permutation
    ]

    fsk_code_array = fsk_code_array[
        permutation
    ]

    return {
        "X": X,
        "y": y,
        "snr": snr_array,
        "params": params_array,
        "code": code_array,
        "fsk_code": fsk_code_array
    }


# ============================================================
# 12. 保存数据
# ============================================================

def save_dataset(
    output_path,
    dataset
):

    print()
    print(
        f"正在保存：{output_path}"
    )

    np.savez_compressed(

        output_path,

        X=dataset["X"],

        y=dataset["y"],

        snr=dataset["snr"],

        params=dataset["params"],

        code=dataset["code"],

        fsk_code=dataset[
            "fsk_code"
        ],

        class_names=np.array(
            CLASS_NAMES
        ),

        param_names=np.array(
            PARAM_NAMES
        ),

        fs=np.int64(FS),

        n_points=np.int64(N),

        duration=np.float64(
            DURATION
        )
    )

    print(
        "保存完成。"
    )


# ============================================================
# 13. 保存数据库信息
# ============================================================

def save_dataset_info(output_dir):

    total_per_class = (
        len(SNR_VALUES)
        * SAMPLES_PER_CLASS_SNR
    )

    total_samples = (
        len(CLASS_NAMES)
        * total_per_class
    )

    info = {

        "seed": SEED,

        "sample_rate_hz": FS,

        "sample_points": N,

        "signal_duration_s": DURATION,

        "signal_duration_us": (
            DURATION * 1e6
        ),

        "classes": CLASS_NAMES,

        "number_of_classes": len(
            CLASS_NAMES
        ),

        "snr_db": SNR_VALUES,

        "samples_per_class_per_snr":
            SAMPLES_PER_CLASS_SNR,

        "samples_per_class":
            total_per_class,

        "total_samples":
            total_samples,

        "split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },

        "split_samples": {
            "train":
                len(CLASS_NAMES)
                * len(SNR_VALUES)
                * TRAIN_PER_GROUP,

            "validation":
                len(CLASS_NAMES)
                * len(SNR_VALUES)
                * VAL_PER_GROUP,

            "test":
                len(CLASS_NAMES)
                * len(SNR_VALUES)
                * TEST_PER_GROUP
        },

        "parameter_ranges": {

            "amplitude": [
                0.8,
                1.2
            ],

            "center_frequency_hz": [
                -1_500_000,
                1_500_000
            ],

            "lfm_bandwidth_hz": [
                1_000_000,
                6_000_000
            ],

            "symbol_counts": [
                8,
                16,
                32
            ],

            "2fsk_frequency_separation_hz": [
                800_000,
                3_000_000
            ],

            "initial_phase_rad": [
                0,
                2 * np.pi
            ]
        },

        "param_names": PARAM_NAMES
    }

    info_path = (
        output_dir
        / "dataset_info.json"
    )

    with open(
        info_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            info,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"数据库配置已保存到：{info_path}"
    )


# ============================================================
# 14. 主函数
# ============================================================

def main():

    # generate_dataset.py
    # 位于：
    #
    # RadarSignal/dataset/
    #
    # 输出到：
    #
    # RadarSignal/data/

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    output_dir = (
        project_root
        / "data"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)

    print(
        "雷达脉内信号数据库生成程序"
    )

    print("=" * 60)

    print(
        f"采样率：{FS / 1e6:.1f} MHz"
    )

    print(
        f"采样点：{N}"
    )

    print(
        f"信号长度：{DURATION * 1e6:.1f} us"
    )

    print(
        f"信号类别：{len(CLASS_NAMES)}"
    )

    print(
        f"SNR：{SNR_VALUES}"
    )

    print(
        f"总样本数量："
        f"{len(CLASS_NAMES) * len(SNR_VALUES) * SAMPLES_PER_CLASS_SNR}"
    )

    # 创建独立随机数发生器
    rng = np.random.default_rng(
        SEED
    )

    # ========================================================
    # 训练集
    # ========================================================

    train_dataset = create_dataset(
        split_name="训练集",
        samples_per_group=TRAIN_PER_GROUP,
        rng=rng
    )

    save_dataset(
        output_dir
        / "train.npz",
        train_dataset
    )

    # 释放内存
    del train_dataset

    # ========================================================
    # 验证集
    # ========================================================

    val_dataset = create_dataset(
        split_name="验证集",
        samples_per_group=VAL_PER_GROUP,
        rng=rng
    )

    save_dataset(
        output_dir
        / "val.npz",
        val_dataset
    )

    del val_dataset

    # ========================================================
    # 测试集
    # ========================================================

    test_dataset = create_dataset(
        split_name="测试集",
        samples_per_group=TEST_PER_GROUP,
        rng=rng
    )

    save_dataset(
        output_dir
        / "test.npz",
        test_dataset
    )

    del test_dataset

    # 保存参数说明
    save_dataset_info(
        output_dir
    )

    print()
    print("=" * 60)

    print(
        "全部数据库生成完成！"
    )

    print("=" * 60)

    print()
    print(
        f"数据库位置：{output_dir}"
    )


# ============================================================
# 15. 程序入口
# ============================================================

if __name__ == "__main__":

    main()
