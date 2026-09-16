"""RadarSignal 数据集检查与绘图，可在 PyCharm 中直接运行。

推荐放置：RadarSignal/dataset/check_dataset.py
数据位置：RadarSignal/data/{train,val,test}.npz
依赖：python -m pip install numpy matplotlib
命令行示例：python dataset/check_dataset.py --data-dir data --no-show
说明：SNR 分布检查的是保存的标签，不能仅凭带噪 I/Q 验证真实 SNR。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib


# 在 PyCharm 中可只修改本区域，然后右键 Run。
# None 表示自动查找脚本同级或上一级的 data 文件夹。
DATA_DIR = None  # 自动按脚本位置定位项目 data 目录
OUTPUT_DIR = None  # None：保存到项目根目录/check_results/本次运行时间/
SAMPLE_SPLIT = "test"
RANDOM_SEED = 2026
SELECT_SNR = None  # None：随机 SNR；设为 10 可只抽取 10 dB 样本
SHOW_PLOTS = True
DEFAULT_FS = 20_000_000.0  # 文件没有 fs 时，采用上一版生成代码的 20 MHz
EXPECTED_LENGTH = 2048
STFT_WINDOW = 256
STFT_HOP = 64
STFT_NFFT = 512

CLASS_NAMES = ("LFM", "BPSK", "QPSK", "LFM_BPSK", "LFM_QPSK", "2FSK_BPSK")
SNR_VALUES = np.arange(11)
EXPECTED_PER_GROUP = {"train": 84, "val": 18, "test": 18}
PARAM_NAMES = (
    "amplitude", "center_freq_hz", "bandwidth_hz", "chirp_rate_hz_per_s",
    "symbol_count", "symbol_rate_baud", "f1_hz", "f2_hz", "freq_sep_hz",
    "chirp_direction", "initial_phase_rad",
)

# 如旧脚本使用其他字段名，将它放在对应元组中即可。
# 多个别名同时出现时拒绝猜测，避免读错数据。
FIELD_ALIASES = {
    "iq": ("X", "x", "iq", "IQ", "signals", "signal", "data", "X_iq"),
    "label": ("y", "labels", "label", "class_labels"),
    "snr": ("snr", "snrs", "SNR", "snr_db", "snr_values"),
    "params": ("params", "parameters", "ground_truth"),
    "fs": ("fs", "sampling_rate", "sample_rate", "fs_hz"),
}


class Report:
    def __init__(self):
        self.lines = []
        self.warnings = []
        self.errors = []

    def log(self, message=""):
        self.lines.append(str(message))
        print(message)

    def warn(self, message):
        self.warnings.append(message)
        self.log(f"[警告] {message}")

    def error(self, message):
        self.errors.append(message)
        self.log(f"[错误] {message}")


def find_data_dir(value):
    if value is not None:
        return Path(value).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    candidates = (script_dir / "data", script_dir.parent / "data")
    for candidate in candidates:
        if any((candidate / f"{s}.npz").is_file() for s in EXPECTED_PER_GROUP):
            return candidate
    return candidates[1] if script_dir.name == "dataset" else candidates[0]


def get_key(archive, field, required=False):
    matches = [key for key in FIELD_ALIASES[field] if key in archive.files]
    if len(matches) > 1:
        raise ValueError(f"{field} 有多个候选字段 {matches}，请修改 FIELD_ALIASES 明确使用哪个。")
    if not matches and required:
        raise ValueError(f"缺少 {field} 字段；接受的名称：{FIELD_ALIASES[field]}")
    return matches[0] if matches else None


def numeric_finite(array, name):
    if array.dtype.kind not in "fiu c".replace(" ", ""):
        raise ValueError(f"{name} 必须是数值数组，当前 dtype={array.dtype}。")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 含 NaN 或 Inf。")


def vector(array, count, name):
    # 不使用任意 flatten，防止把错误的二维标签悄悄展开。
    if array.shape == (count, 1):
        array = array[:, 0]
    if array.shape != (count,):
        raise ValueError(f"{name} shape 应为 ({count},) 或 ({count}, 1)，实际为 {array.shape}。")
    return array


def normalized_class_name(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).upper().replace("-", "_").strip()


def decode_labels(raw, count, archive):
    raw = vector(raw, count, "类别标签")
    if raw.dtype.kind in "US":
        names = [normalized_class_name(v) for v in raw]
    else:
        numeric_finite(raw, "类别标签")
        if np.iscomplexobj(raw) or not np.equal(raw, np.floor(raw)).all():
            raise ValueError("类别标签必须是整数 0~5 或六类信号名称。")
        if np.any((raw < 0) | (raw >= len(CLASS_NAMES))):
            raise ValueError("类别标签超出 0~5 范围。")
        ids = raw.astype(np.int64)
        if "class_names" not in archive.files:
            return ids
        source_names = archive["class_names"]
        if source_names.shape != (6,):
            raise ValueError("class_names 应为长度为 6 的一维数组。")
        source_names = [normalized_class_name(v) for v in source_names]
        if set(source_names) != set(CLASS_NAMES):
            raise ValueError(f"class_names 与预期六类不一致：{source_names}")
        names = [source_names[i] for i in ids]
    lookup = {name: i for i, name in enumerate(CLASS_NAMES)}
    unknown = sorted(set(names) - set(lookup))
    if unknown:
        raise ValueError(f"未知类别名称：{unknown}")
    return np.array([lookup[name] for name in names], dtype=np.int64)


def validate_iq(raw):
    if raw.dtype.kind == "c" and raw.ndim == 2:
        layout, length = "complex", raw.shape[1]
    elif raw.dtype.kind in "fiu" and raw.ndim == 3:
        if raw.shape[1] == 2 and raw.shape[2] != 2:
            layout, length = "channels_first", raw.shape[2]
        elif raw.shape[2] == 2 and raw.shape[1] != 2:
            layout, length = "channels_last", raw.shape[1]
        else:
            raise ValueError(f"无法确定 I/Q 通道轴：{raw.shape}")
    else:
        raise ValueError("I/Q 应为复数 (M,N)、实数 (M,2,N) 或实数 (M,N,2) 数组；不接受仅实部。")
    if raw.shape[0] == 0 or length < 8:
        raise ValueError("数据集不能为空，每条信号至少需要 8 个采样点。")
    zero_count = 0
    for start in range(0, raw.shape[0], 256):
        block = raw[start:start + 256]
        numeric_finite(block, "I/Q")
        zero_count += int(np.all(block == 0, axis=tuple(range(1, block.ndim))).sum())
    if zero_count:
        raise ValueError(f"发现 {zero_count} 条全零 I/Q 样本。")
    return layout, length


def complex_sample(raw, index, layout):
    sample = raw[index]
    if layout == "complex":
        return sample.astype(np.complex128)
    if layout == "channels_first":
        return sample[0].astype(np.float64) + 1j * sample[1].astype(np.float64)
    return sample[:, 0].astype(np.float64) + 1j * sample[:, 1].astype(np.float64)


def load_and_check(path, split, report, fs_override):
    report.log(f"\n{'=' * 64}\n{split}: {path}")
    if not path.is_file():
        raise ValueError(f"找不到文件：{path}。请设置 DATA_DIR 或 --data-dir。")
    with np.load(path, allow_pickle=False) as archive:
        report.log("字段清单（原始 shape / dtype）：")
        for key in archive.files:
            try:
                value = archive[key]
                report.log(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                del value
            except ValueError as exc:
                report.warn(f"{split}/{key} 无法安全读取：{exc}；必需字段不可读时本数据集检查失败。")

        keys = {field: get_key(archive, field, field in ("iq", "label", "snr"))
                for field in FIELD_ALIASES}
        report.log(f"采用字段：{keys}")
        raw = archive[keys["iq"]]
        layout, length = validate_iq(raw)
        count = len(raw)
        labels = decode_labels(archive[keys["label"]], count, archive)
        if "class_names" not in archive.files:
            report.log("类别约定：0=LFM, 1=BPSK, 2=QPSK, 3=LFM_BPSK, 4=LFM_QPSK, 5=2FSK_BPSK。")
        snr = vector(archive[keys["snr"]], count, "SNR")
        numeric_finite(snr, "SNR")
        if np.iscomplexobj(snr) or np.any((snr < 0) | (snr > 10)) or not np.equal(snr, np.floor(snr)).all():
            raise ValueError("SNR 标签必须为 0~10 dB 的整数。")
        snr = snr.astype(np.int64)

        if fs_override is not None:
            fs = np.full(count, fs_override, dtype=float)
            report.log(f"采样率：使用用户指定值 {fs_override:g} Hz。")
        elif keys["fs"]:
            saved_fs = np.asarray(archive[keys["fs"]])
            if np.iscomplexobj(saved_fs) or saved_fs.dtype.kind not in "fiu":
                raise ValueError("fs 必须是实数数值。")
            if saved_fs.size == 1:
                fs = np.full(count, saved_fs.item(), dtype=float)
            else:
                fs = vector(saved_fs, count, "fs").astype(float)
            report.log(f"采样率：读取字段 {keys['fs']}。")
        else:
            fs = np.full(count, DEFAULT_FS, dtype=float)
            report.warn(f"{split} 未保存 fs，按上一版配置采用 {DEFAULT_FS:g} Hz；如不同，请设置 --fs。")
        if not np.isfinite(fs).all() or np.any(fs <= 0):
            raise ValueError("采样率必须是有限正数。")

        params = None
        param_names = None
        if keys["params"]:
            params = archive[keys["params"]]
            if params.ndim != 2 or params.shape[0] != count or params.dtype.kind not in "fiu":
                raise ValueError(f"params 应为 (M,P) 实数数组，实际 {params.shape} / {params.dtype}。")
            if np.isinf(params).any():
                raise ValueError("params 含 Inf；仅不适用的参数允许 NaN。")
            if "param_names" in archive.files:
                stored_names = archive["param_names"]
                if stored_names.shape != (params.shape[1],):
                    raise ValueError("param_names 长度与 params 列数不一致。")
                param_names = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in stored_names]
            elif params.shape[1] == len(PARAM_NAMES):
                param_names = list(PARAM_NAMES)
                report.log("参数名称按历史预览中的 11 列顺序解释。")
            else:
                param_names = [f"param_{i}" for i in range(params.shape[1])]
                report.warn(f"{split} 参数列数为 {params.shape[1]} 且缺少 param_names，使用列编号。")
            report.log(f"Ground Truth：{params.shape}，NaN={int(np.isnan(params).sum())}（不适用参数允许为空）。")
        else:
            report.warn(f"{split} 缺少 params，无法检查参数真值表；仍可检查分类数据并绘图。")

        for key in ("codes", "code", "phase_codes", "fsk_codes", "sample_ids"):
            if key in archive.files:
                extra = archive[key]
                if extra.ndim == 0 or extra.shape[0] != count:
                    raise ValueError(f"逐样本字段 {key} 第一维应为 {count}，实际为 {extra.shape}。")

    report.log(f"样本数={count}；I/Q 布局={layout}；每条采样点={length}。")
    report.log(f"采样率范围={fs.min():g}~{fs.max():g} Hz；记录时长范围={length/fs.max()*1e6:.3f}~{length/fs.min()*1e6:.3f} us。")
    if length != EXPECTED_LENGTH:
        report.warn(f"{split} 采样点数为 {length}，上一版计划值为 {EXPECTED_LENGTH}。")
    counts = np.zeros((6, 11), dtype=np.int64)
    np.add.at(counts, (labels, snr), 1)
    report.log("类别 × SNR 数量（列依次为 0~10 dB）：")
    for name, row in zip(CLASS_NAMES, counts):
        report.log(f"  {name:12s}: {' '.join(f'{v:4d}' for v in row)} | 合计 {row.sum()}")
    report.log(f"SNR 合计：{counts.sum(axis=0).tolist()}")
    expected = EXPECTED_PER_GROUP[split]
    if not np.all(counts == expected):
        report.warn(f"{split} 分布与原计划每类每 SNR {expected} 条不符；共 {(counts != expected).sum()} 个组合不同。")
    else:
        report.log(f"[通过] {split} 每类每 SNR 均为 {expected} 条。")
    return dict(raw=raw, layout=layout, length=length, labels=labels, snr=snr,
                fs=fs, params=params, param_names=param_names, counts=counts, count=count)


def spectrum_db(signal, fs):
    """双边 Hann 加窗周期图，单位 dB/Hz，参考 1 个幅度单位平方/Hz。"""
    window = np.hanning(len(signal))
    transform = np.fft.fftshift(np.fft.fft(signal * window))
    psd = np.abs(transform) ** 2 / (fs * np.sum(window ** 2))
    frequency = np.fft.fftshift(np.fft.fftfreq(len(signal), 1 / fs))
    return frequency, 10 * np.log10(np.maximum(psd, np.finfo(float).tiny))


def stft_db(signal, fs):
    """双边复数 STFT；只使用完整窗，不补边；时间坐标位于窗中心。"""
    window_length = min(STFT_WINDOW, len(signal))
    hop = min(STFT_HOP, window_length)
    nfft = max(STFT_NFFT, window_length)
    starts = np.arange(0, len(signal) - window_length + 1, hop)
    frames = np.stack([signal[start:start + window_length] for start in starts])
    window = np.hanning(window_length)
    transform = np.fft.fftshift(np.fft.fft(frames * window, n=nfft, axis=1), axes=1)
    power = np.abs(transform) ** 2 / (fs * np.sum(window ** 2))
    frequency = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    times = (starts + (window_length - 1) / 2) / fs
    db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
    return times, frequency, db.T


def plot_distributions(summaries, output_dir, plt):
    fig, axes = plt.subplots(1, len(summaries), figsize=(6 * len(summaries), 4), squeeze=False)
    vmax = max(int(summary["counts"].max()) for summary in summaries.values())
    for ax, (split, summary) in zip(axes[0], summaries.items()):
        counts = summary["counts"]
        chart = ax.imshow(counts, aspect="auto", cmap="Blues", vmin=0, vmax=max(vmax, 1))
        ax.set_xticks(SNR_VALUES)
        ax.set_yticks(range(6), CLASS_NAMES)
        ax.set_xlabel("SNR label (dB)")
        ax.set_title(f"{split}: {summary['count']} samples")
        for row in range(6):
            for column in range(11):
                value = counts[row, column]
                ax.text(column, row, str(value), ha="center", va="center", fontsize=8,
                        color="white" if value > vmax * 0.55 else "black")
        fig.colorbar(chart, ax=ax, label="Sample count", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "distribution.png", dpi=160)
    return fig


def plot_samples(dataset, split, selected_snr, seed, output_dir, plt, report):
    rng = np.random.default_rng(seed)
    figures, selected_rows = [], []
    for label, name in enumerate(CLASS_NAMES):
        mask = dataset["labels"] == label
        if selected_snr is not None:
            mask &= dataset["snr"] == selected_snr
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            report.error(f"{split}/{name} 在当前 SNR 筛选条件下没有样本，无法完成六类绘图。")
            continue
        index = int(rng.choice(candidates))
        signal = complex_sample(dataset["raw"], index, dataset["layout"])
        fs, snr = float(dataset["fs"][index]), int(dataset["snr"][index])
        time_us = np.arange(len(signal)) / fs * 1e6
        frequency, psd = spectrum_db(signal, fs)
        stft_time, stft_freq, stft_power = stft_db(signal, fs)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
        fig.suptitle(f"{name} | {split}[{index}] | SNR label={snr} dB | fs={fs/1e6:g} MHz")
        axes[0].plot(time_us, signal.real, lw=0.7, label="I (real)", alpha=0.85)
        axes[0].plot(time_us, signal.imag, lw=0.7, label="Q (imag)", alpha=0.75)
        axes[0].set(xlabel="Time (us)", ylabel="Amplitude", title="Time-domain I/Q")
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.25)
        axes[1].plot(frequency / 1e6, psd, lw=0.8)
        axes[1].set(xlabel="Baseband frequency (MHz)", ylabel="PSD (dB/Hz)",
                    title="Two-sided spectrum (Hann window)", xlim=(-fs / 2e6, fs / 2e6))
        axes[1].grid(alpha=0.25)
        peak = float(stft_power.max())
        mesh = axes[2].pcolormesh(stft_time * 1e6, stft_freq / 1e6, stft_power,
                                  shading="auto", cmap="turbo", vmin=peak - 60, vmax=peak)
        axes[2].set(xlabel="Time (us)", ylabel="Baseband frequency (MHz)",
                    title="STFT (Hann; color range: peak - 60 dB to peak)")
        fig.colorbar(mesh, ax=axes[2], label="PSD (dB/Hz)")
        filename = f"{label}_{name}_{split}_index{index}_snr{snr}.png"
        fig.savefig(output_dir / filename, dpi=160)
        figures.append(fig)
        selected_rows.append([split, index, label, name, snr, fs, filename])
        report.log(f"抽样：{split}[{index}]，{name}，SNR={snr} dB -> {filename}")
        if dataset["params"] is not None:
            parameter_text = ", ".join(f"{key}={value:g}" for key, value in
                                       zip(dataset["param_names"], dataset["params"][index]) if np.isfinite(value))
            report.log(f"  参数真值：{parameter_text}")
    with (output_dir / "selected_samples.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "index_zero_based", "label", "class_name", "snr_db", "fs_hz", "figure"])
        writer.writerows(selected_rows)
    return figures


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="含 train/val/test.npz 的文件夹")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="输出根目录，每次新建时间戳子目录")
    parser.add_argument("--split", choices=tuple(EXPECTED_PER_GROUP), default=SAMPLE_SPLIT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--snr", type=int, choices=range(11), default=SELECT_SNR)
    parser.add_argument("--fs", type=float, help="覆盖采样率，单位 Hz，例如 20000000")
    parser.add_argument("--no-show", action="store_true", help="只保存图片，不弹窗")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    show = SHOW_PLOTS and not args.no_show
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = find_data_dir(args.data_dir)
    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir.parent / "check_results"
    output_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = Report()
    report.log("RadarSignal 数据集检查")
    report.log(f"数据目录：{data_dir}\n输出目录：{output_dir}")
    report.log(f"随机种子={args.seed}；抽样集合={args.split}；SNR 筛选={args.snr}")
    report.log("SNR 是文件标签，未通过干净信号/噪声对验证实际 SNR；本脚本不计算识别或参数测量准确率。")
    summaries, plot_data = {}, None
    for split in EXPECTED_PER_GROUP:
        try:
            dataset = load_and_check(data_dir / f"{split}.npz", split, report, args.fs)
            summaries[split] = {key: dataset[key] for key in ("counts", "count", "length")}
            if split == args.split:
                plot_data = dataset
            del dataset
        except (ValueError, OSError, KeyError, EOFError) as exc:
            report.error(f"{split}: {exc}")

    figures = []
    if summaries:
        if len(summaries) == 3:
            total = sum(summary["count"] for summary in summaries.values())
            report.log(f"\n总样本数：{total}（原计划 7920）")
            for split, summary in summaries.items():
                report.log(f"{split}：{summary['count']} 条，占 {summary['count'] / total:.2%}")
            if len({summary["length"] for summary in summaries.values()}) != 1:
                report.error("train/val/test 的采样点数不一致，不能直接使用同一固定输入维度模型。")
        with (output_dir / "distribution.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "label", "class_name", "snr_db", "count"])
            for split, summary in summaries.items():
                for label, name in enumerate(CLASS_NAMES):
                    for snr in SNR_VALUES:
                        writer.writerow([split, label, name, snr, summary["counts"][label, snr]])
        figures.append(plot_distributions(summaries, output_dir, plt))
    if plot_data is not None:
        figures.extend(plot_samples(plot_data, args.split, args.snr, args.seed, output_dir, plt, report))
    else:
        report.error(f"{args.split} 不可用，未生成信号示例图。")
    report.log(f"\n检查结束：{len(report.errors)} 个错误，{len(report.warnings)} 个警告。")
    report.log("结构与分布检查不等于确认信号公式、实际 SNR 或训练/测试独立性。")
    report.log(f"报告与图片已保存到：{output_dir}")
    (output_dir / "check_report.txt").write_text("\n".join(report.lines) + "\n", encoding="utf-8-sig")
    if show and figures:
        plt.show()
    for fig in figures:
        plt.close(fig)
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
