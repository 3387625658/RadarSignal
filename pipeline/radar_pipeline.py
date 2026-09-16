from pathlib import Path
import sys

import numpy as np
import torch


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
# 2. 导入模型
# ============================================================

from models.resnet18_radar import RadarResNet18


# ============================================================
# 3. 导入参数估计器
# ============================================================

from estimation.lfm_estimator import (
    estimate_lfm_parameters
)

from estimation.psk_estimator import (
    estimate_psk_parameters
)

from estimation.lfm_psk_estimator import (
    estimate_lfm_psk_parameters
)

from estimation.fsk_bpsk_estimator import (
    estimate_fsk_bpsk_parameters
)


# ============================================================
# 4. 类别
# ============================================================

CLASS_NAMES = [

    "LFM",

    "BPSK",

    "QPSK",

    "LFM_BPSK",

    "LFM_QPSK",

    "2FSK_BPSK"
]


# ============================================================
# 5. 模型路径
# ============================================================

MODEL_PATH = (

    PROJECT_ROOT

    / "outputs"

    / "models"

    / "resnet18_best.pth"
)


# ============================================================
# 6. 设备
# ============================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"
)


# ============================================================
# 7. IQ归一化
#
# 必须与训练阶段保持一致
# ============================================================

def normalize_iq(
    signal
):

    signal = np.asarray(
        signal,
        dtype=np.complex64
    )


    power = np.mean(

        np.abs(signal) ** 2
    )


    signal = (

        signal

        /

        np.sqrt(
            power + 1e-12
        )
    )


    return signal


# ============================================================
# 8. IQ → STFT
#
# 必须与RadarDataset训练预处理完全一致
# ============================================================

def iq_to_stft(
    signal,
    fs,
    n_fft=256,
    hop_length=64,
    win_length=256
):

    signal = normalize_iq(
        signal
    )


    # 转成PyTorch复数
    signal_tensor = torch.tensor(

        signal,

        dtype=torch.complex64
    )


    window = torch.hann_window(
        win_length
    )


    # ========================================================
    # STFT
    # ========================================================

    stft_result = torch.stft(

        signal_tensor,

        n_fft=n_fft,

        hop_length=hop_length,

        win_length=win_length,

        window=window,

        center=True,

        return_complex=True,

        onesided=False
    )


    # ========================================================
    # fftshift
    # ========================================================

    stft_result = torch.fft.fftshift(

        stft_result,

        dim=0
    )


    # ========================================================
    # 幅度谱
    # ========================================================

    magnitude = torch.abs(
        stft_result
    )


    # ========================================================
    # 转dB
    # ========================================================

    magnitude_db = (

        20

        * torch.log10(

            magnitude

            + 1e-12
        )
    )


    # ========================================================
    # 最大值归一化为0 dB
    # ========================================================

    magnitude_db = (

        magnitude_db

        -

        torch.max(
            magnitude_db
        )
    )


    # ========================================================
    # 限制动态范围
    # ========================================================

    magnitude_db = torch.clamp(

        magnitude_db,

        min=-60.0,

        max=0.0
    )


    # ========================================================
    # 映射到0~1
    # ========================================================

    magnitude_normalized = (

        magnitude_db
        + 60.0

    ) / 60.0


    # ========================================================
    # 增加Channel维
    #
    # (F,T)
    # →
    # (1,F,T)
    # ========================================================

    magnitude_normalized = (
        magnitude_normalized
        .unsqueeze(0)
    )


    # ========================================================
    # 再增加Batch维
    #
    # (1,F,T)
    # →
    # (1,1,F,T)
    # ========================================================

    magnitude_normalized = (
        magnitude_normalized
        .unsqueeze(0)
    )


    return magnitude_normalized


# ============================================================
# 9. 加载ResNet18
# ============================================================

def load_classifier():

    model = RadarResNet18(
        num_classes=6
    )


    checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE
    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    return model


# ============================================================
# 10. 信号识别
# ============================================================

def classify_signal(
    signal,
    fs,
    model
):

    stft_input = iq_to_stft(

        signal,

        fs
    )


    stft_input = (
        stft_input
        .to(DEVICE)
    )


    with torch.no_grad():

        logits = model(
            stft_input
        )


        probabilities = torch.softmax(

            logits,

            dim=1
        )


        predicted_label = int(

            torch.argmax(

                probabilities,

                dim=1

            ).item()
        )


        confidence = float(

            probabilities[
                0,
                predicted_label
            ].item()
        )


    predicted_class = (

        CLASS_NAMES[
            predicted_label
        ]
    )


    probability_dict = {

        CLASS_NAMES[i]:

            float(
                probabilities[
                    0,
                    i
                ].item()
            )

        for i in range(
            len(CLASS_NAMES)
        )
    }


    return {

        "label":
            predicted_label,

        "class_name":
            predicted_class,

        "confidence":
            confidence,

        "probabilities":
            probability_dict
    }


# ============================================================
# 11. 根据识别结果自动参数测量
# ============================================================

def estimate_parameters_by_class(
    signal,
    fs,
    class_name
):

    # ========================================================
    # LFM
    # ========================================================

    if class_name == "LFM":

        result = (
            estimate_lfm_parameters(

                signal,

                fs
            )
        )


    # ========================================================
    # BPSK
    # ========================================================

    elif class_name == "BPSK":

        result = (
            estimate_psk_parameters(

                signal,

                fs,

                modulation="BPSK"
            )
        )


    # ========================================================
    # QPSK
    # ========================================================

    elif class_name == "QPSK":

        result = (
            estimate_psk_parameters(

                signal,

                fs,

                modulation="QPSK"
            )
        )


    # ========================================================
    # LFM-BPSK
    # ========================================================

    elif class_name == "LFM_BPSK":

        result = (
            estimate_lfm_psk_parameters(

                signal,

                fs,

                modulation="BPSK"
            )
        )


    # ========================================================
    # LFM-QPSK
    # ========================================================

    elif class_name == "LFM_QPSK":

        result = (
            estimate_lfm_psk_parameters(

                signal,

                fs,

                modulation="QPSK"
            )
        )


    # ========================================================
    # 2FSK-BPSK
    # ========================================================

    elif class_name == "2FSK_BPSK":

        result = (
            estimate_fsk_bpsk_parameters(

                signal,

                fs
            )
        )


    else:

        raise ValueError(

            f"未知信号类型："
            f"{class_name}"
        )


    return result


# ============================================================
# 12. 完整流水线
# ============================================================

def analyze_signal(
    signal,
    fs,
    model
):

    # ========================================================
    # 第一步：识别
    # ========================================================

    classification = (
        classify_signal(

            signal,

            fs,

            model
        )
    )


    predicted_class = (
        classification[
            "class_name"
        ]
    )


    # ========================================================
    # 第二步：参数估计
    # ========================================================

    parameters = (
        estimate_parameters_by_class(

            signal,

            fs,

            predicted_class
        )
    )


    # ========================================================
    # 第三步：组合结果
    # ========================================================

    return {

        "classification":
            classification,

        "parameters":
            parameters
    }