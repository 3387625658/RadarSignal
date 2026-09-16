from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 项目路径
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent

DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# 2. RadarDataset
# ============================================================

class RadarDataset(Dataset):

    """
    雷达脉内信号 PyTorch Dataset

    支持两种输入模式：

    mode="iq"
        输出原始 I/Q
        shape:
            (2, 2048)

        用于：
            1D-CNN
            CNN-LSTM


    mode="stft"
        在线计算 STFT 时频图
        shape:
            (1, frequency_bins, time_bins)

        用于：
            2D-CNN
            ResNet


    Parameters
    ----------
    npz_path:
        train.npz / val.npz / test.npz

    mode:
        "iq" 或 "stft"

    normalize:
        是否对输入信号进行归一化

    return_metadata:
        是否额外返回：
            SNR
            params
            code
            fsk_code
    """

    def __init__(
        self,
        npz_path,
        mode="iq",
        normalize=True,
        return_metadata=False,
        stft_n_fft=256,
        stft_hop_length=64,
        stft_win_length=256
    ):

        super().__init__()

        self.npz_path = Path(
            npz_path
        )

        self.mode = mode

        self.normalize = normalize

        self.return_metadata = (
            return_metadata
        )

        self.stft_n_fft = (
            stft_n_fft
        )

        self.stft_hop_length = (
            stft_hop_length
        )

        self.stft_win_length = (
            stft_win_length
        )

        # ----------------------------------------------------
        # 模式检查
        # ----------------------------------------------------

        if self.mode not in [
            "iq",
            "stft"
        ]:

            raise ValueError(
                "mode 只能是 "
                "'iq' 或 'stft'"
            )

        # ----------------------------------------------------
        # 检查文件
        # ----------------------------------------------------

        if not self.npz_path.exists():

            raise FileNotFoundError(
                f"找不到数据文件：\n"
                f"{self.npz_path}"
            )

        # ----------------------------------------------------
        # 加载数据库
        # ----------------------------------------------------

        print(
            f"正在加载："
            f"{self.npz_path.name}"
        )

        data = np.load(
            self.npz_path,
            allow_pickle=False
        )

        # 信号
        self.X = data[
            "X"
        ].astype(
            np.float32
        )

        # 标签
        self.y = data[
            "y"
        ].astype(
            np.int64
        )

        # SNR
        self.snr = data[
            "snr"
        ].astype(
            np.int16
        )

        # 参数真实值
        self.params = data[
            "params"
        ].astype(
            np.float64
        )

        # PSK码
        self.code = data[
            "code"
        ].astype(
            np.int8
        )

        # FSK码
        self.fsk_code = data[
            "fsk_code"
        ].astype(
            np.int8
        )

        # 类别名
        self.class_names = (
            data["class_names"]
            .tolist()
        )

        # 参数名
        self.param_names = (
            data["param_names"]
            .tolist()
        )

        # 采样率
        self.fs = float(
            data["fs"]
        )

        # 采样点
        self.n_points = int(
            data["n_points"]
        )

        data.close()

        # ----------------------------------------------------
        # Hann窗
        # ----------------------------------------------------

        if self.mode == "stft":

            self.window = (
                torch.hann_window(
                    self.stft_win_length
                )
            )

        # ----------------------------------------------------
        # 打印基本信息
        # ----------------------------------------------------

        print(
            f"加载完成："
            f"{len(self.X)} 条样本"
        )

        print(
            f"输入模式："
            f"{self.mode}"
        )


    # ========================================================
    # Dataset长度
    # ========================================================

    def __len__(self):

        return len(
            self.y
        )


    # ========================================================
    # IQ归一化
    # ========================================================

    @staticmethod
    def normalize_iq(iq):

        """
        RMS功率归一化。

        iq shape:
            (2, N)
        """

        power = np.mean(
            iq[0] ** 2
            +
            iq[1] ** 2
        )

        rms = np.sqrt(
            power
            + 1e-12
        )

        iq = iq / rms

        return iq.astype(
            np.float32
        )


    # ========================================================
    # IQ → STFT
    # ========================================================

    def iq_to_stft(
        self,
        iq_tensor
    ):

        """
        输入：

            iq_tensor
            shape = (2, N)

        输出：

            STFT log magnitude
            shape = (1, F, T)
        """

        # ----------------------------------------------------
        # I/Q恢复为复数信号
        # ----------------------------------------------------

        complex_signal = torch.complex(

            iq_tensor[0],

            iq_tensor[1]
        )

        # ----------------------------------------------------
        # STFT
        # ----------------------------------------------------

        stft_result = torch.stft(

            complex_signal,

            n_fft=self.stft_n_fft,

            hop_length=self.stft_hop_length,

            win_length=self.stft_win_length,

            window=self.window,

            center=True,

            return_complex=True,

            onesided=False
        )

        # ----------------------------------------------------
        # FFT shift
        # ----------------------------------------------------

        stft_result = torch.fft.fftshift(

            stft_result,

            dim=0
        )

        # ----------------------------------------------------
        # 幅度
        # ----------------------------------------------------

        magnitude = torch.abs(
            stft_result
        )

        # ----------------------------------------------------
        # 转为dB
        # ----------------------------------------------------

        log_magnitude = (

            20

            * torch.log10(

                magnitude
                + 1e-8
            )
        )

        # ----------------------------------------------------
        # 每张STFT相对自己的峰值归一化
        #
        # 最高 = 0 dB
        # ----------------------------------------------------

        log_magnitude = (

            log_magnitude

            - torch.max(
                log_magnitude
            )
        )

        # ----------------------------------------------------
        # 截取动态范围
        #
        # -60 dB ~ 0 dB
        # ----------------------------------------------------

        log_magnitude = torch.clamp(

            log_magnitude,

            min=-60.0,

            max=0.0
        )

        # ----------------------------------------------------
        # 映射到 0 ~ 1
        # ----------------------------------------------------

        log_magnitude = (

            log_magnitude
            + 60.0

        ) / 60.0

        # ----------------------------------------------------
        # 增加channel
        #
        # (F,T)
        #
        # →
        #
        # (1,F,T)
        # ----------------------------------------------------

        log_magnitude = (
            log_magnitude
            .unsqueeze(0)
        )

        return log_magnitude


    # ========================================================
    # 获取一条样本
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        # ----------------------------------------------------
        # 原始IQ
        # ----------------------------------------------------

        iq = self.X[
            index
        ].copy()

        # ----------------------------------------------------
        # 归一化
        # ----------------------------------------------------

        if self.normalize:

            iq = self.normalize_iq(
                iq
            )

        # ----------------------------------------------------
        # numpy → torch
        # ----------------------------------------------------

        iq_tensor = torch.from_numpy(
            iq
        ).float()

        # ----------------------------------------------------
        # 标签
        # ----------------------------------------------------

        label = torch.tensor(

            self.y[index],

            dtype=torch.long
        )

        # ----------------------------------------------------
        # 根据mode决定输入
        # ----------------------------------------------------

        if self.mode == "iq":

            signal = iq_tensor

        else:

            signal = self.iq_to_stft(
                iq_tensor
            )

        # ----------------------------------------------------
        # 普通训练
        # ----------------------------------------------------

        if not self.return_metadata:

            return (
                signal,
                label
            )

        # ----------------------------------------------------
        # 带Ground Truth
        #
        # 以后模型评价 / 参数测量使用
        # ----------------------------------------------------

        metadata = {

            "snr": torch.tensor(
                int(
                    self.snr[index]
                ),
                dtype=torch.int64
            ),

            "params": torch.tensor(
                self.params[index],
                dtype=torch.float64
            ),

            "code": torch.tensor(
                self.code[index],
                dtype=torch.int64
            ),

            "fsk_code": torch.tensor(
                self.fsk_code[index],
                dtype=torch.int64
            ),

            "index": torch.tensor(
                index,
                dtype=torch.int64
            )
        }

        return (
            signal,
            label,
            metadata
        )


# ============================================================
# 3. 创建三个 Dataset
# ============================================================

def create_datasets(
    mode="iq",
    normalize=True,
    return_metadata=False
):

    train_dataset = RadarDataset(

        DATA_DIR / "train.npz",

        mode=mode,

        normalize=normalize,

        return_metadata=return_metadata
    )

    val_dataset = RadarDataset(

        DATA_DIR / "val.npz",

        mode=mode,

        normalize=normalize,

        return_metadata=return_metadata
    )

    test_dataset = RadarDataset(

        DATA_DIR / "test.npz",

        mode=mode,

        normalize=normalize,

        return_metadata=return_metadata
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# 4. 创建 DataLoader
# ============================================================

def create_dataloaders(
    mode="iq",
    batch_size=64,
    num_workers=0,
    normalize=True,
    return_metadata=False
):

    """

    Windows + PyCharm：

        num_workers 建议先使用 0

    等程序完全跑通以后，
    可以尝试：

        num_workers = 2
        num_workers = 4

    """

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets(

        mode=mode,

        normalize=normalize,

        return_metadata=return_metadata
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        drop_last=False
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        drop_last=False
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        drop_last=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# 5. 测试 IQ 模式
# ============================================================

def test_iq_mode():

    print()
    print("=" * 70)
    print("测试 IQ 数据模式")
    print("=" * 70)

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(

        mode="iq",

        batch_size=32,

        num_workers=0
    )

    X, y = next(
        iter(train_loader)
    )

    print()

    print(
        "一个Batch的X：",
        X.shape
    )

    print(
        "一个Batch的y：",
        y.shape
    )

    print(
        "数据类型：",
        X.dtype
    )

    print(
        "标签类型：",
        y.dtype
    )

    print(
        "X最小值：",
        X.min().item()
    )

    print(
        "X最大值：",
        X.max().item()
    )

    print(
        "标签：",
        y[:10]
    )

    # 应该：
    #
    # X:
    # (32, 2, 2048)
    #
    # y:
    # (32,)


# ============================================================
# 6. 测试 STFT 模式
# ============================================================

def test_stft_mode():

    print()
    print("=" * 70)
    print("测试 STFT 数据模式")
    print("=" * 70)

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(

        mode="stft",

        batch_size=16,

        num_workers=0
    )

    X, y = next(
        iter(train_loader)
    )

    print()

    print(
        "一个Batch的STFT：",
        X.shape
    )

    print(
        "标签shape：",
        y.shape
    )

    print(
        "STFT最小值：",
        X.min().item()
    )

    print(
        "STFT最大值：",
        X.max().item()
    )

    # 应该类似：
    #
    # (16, 1, 256, 33)
    #
    # 最小值：
    # 0
    #
    # 最大值：
    # 1


# ============================================================
# 7. 测试 Metadata
# ============================================================

def test_metadata():

    print()
    print("=" * 70)
    print("测试 Ground Truth Metadata")
    print("=" * 70)

    dataset = RadarDataset(

        DATA_DIR / "test.npz",

        mode="iq",

        normalize=False,

        return_metadata=True
    )

    signal, label, metadata = (
        dataset[0]
    )

    print()

    print(
        "signal shape：",
        signal.shape
    )

    print(
        "label：",
        label.item()
    )

    print(
        "class：",
        dataset.class_names[
            label.item()
        ]
    )

    print(
        "SNR：",
        metadata["snr"].item(),
        "dB"
    )

    print(
        "params："
    )

    for name, value in zip(

        dataset.param_names,

        metadata["params"]
    ):

        value = value.item()

        if np.isfinite(value):

            print(
                f"    "
                f"{name:25s}"
                f" = "
                f"{value}"
            )

    print(
        "code：",
        metadata["code"]
    )


# ============================================================
# 8. main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("RadarDataset 测试程序")
    print("=" * 70)

    # --------------------------------------------------------
    # 先测试IQ
    # --------------------------------------------------------

    test_iq_mode()

    # --------------------------------------------------------
    # 再测试STFT
    # --------------------------------------------------------

    test_stft_mode()

    # --------------------------------------------------------
    # 最后测试Ground Truth
    # --------------------------------------------------------

    test_metadata()

    print()
    print("=" * 70)
    print("RadarDataset 全部测试完成")
    print("=" * 70)


if __name__ == "__main__":

    main()