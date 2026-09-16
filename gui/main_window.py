from pathlib import Path
import sys

import numpy as np
import torch

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QTextEdit,
    QGroupBox,
    QMessageBox,
    QFileDialog,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ============================================================
# 项目路径
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
# 导入完整处理流水线
# ============================================================

from pipeline.radar_pipeline import (
    load_classifier,
    analyze_signal,
    iq_to_stft
)


# ============================================================
# 数据路径
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "test.npz"
)


# ============================================================
# Matplotlib画布
# ============================================================

class PlotCanvas(FigureCanvas):

    def __init__(
        self,
        width=5,
        height=4
    ):

        self.figure = Figure(
            figsize=(width, height),
            tight_layout=True
        )

        super().__init__(
            self.figure
        )


# ============================================================
# 主窗口
# ============================================================

class RadarMainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "雷达脉内复合信号智能识别与参数测量系统"
        )

        self.resize(
            1500,
            980
        )

        # ----------------------------------------------------
        # 数据
        # ----------------------------------------------------

        self.data = None

        self.X = None
        self.y = None
        self.snr = None
        self.fs = None
        self.class_names = None

        self.current_index = 0
        self.current_signal = None
        # 最近一次完整分析结果
        self.last_result = None

        # 当前信号来源
        # "dataset" / "external"
        self.signal_source = "dataset"

        # 外部文件路径
        self.external_file_path = None

        # 外部信号没有真实类别
        self.external_true_class = None

        # ----------------------------------------------------
        # 模型
        # ----------------------------------------------------

        self.model = None

        # ----------------------------------------------------
        # 初始化
        # ----------------------------------------------------

        self.load_data()

        self.init_model()

        self.init_ui()

        self.load_sample(0)


    # ========================================================
    # 加载数据库
    # ========================================================

    def load_data(self):

        if not DATA_PATH.exists():

            raise FileNotFoundError(
                f"找不到测试集：{DATA_PATH}"
            )

        self.data = np.load(
            DATA_PATH,
            allow_pickle=False
        )

        self.X = self.data["X"]
        self.y = self.data["y"]
        self.snr = self.data["snr"]
        self.params = self.data["params"]

        self.codes = self.data["code"]

        self.fsk_codes = self.data["fsk_code"]

        self.param_names = (
            self.data["param_names"]
            .tolist()
        )

        self.param_index = {

            name: index

            for index, name in enumerate(
                self.param_names
            )
        }

        self.fs = float(
            self.data["fs"]
        )

        self.class_names = (
            self.data[
                "class_names"
            ].tolist()
        )


    # ========================================================
    # 加载ResNet18
    # ========================================================

    def init_model(self):

        print(
            "正在加载ResNet18..."
        )

        self.model = (
            load_classifier()
        )

        print(
            "ResNet18加载完成"
        )


    # ========================================================
    # UI
    # ========================================================

    def init_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(
            central_widget
        )

        main_layout = QVBoxLayout(
            central_widget
        )


        # ====================================================
        # 顶部控制区
        # ====================================================

        control_group = QGroupBox(
            "样本控制"
        )

        control_layout = QHBoxLayout(
            control_group
        )


        # 样本编号
        control_layout.addWidget(
            QLabel(
                "测试样本编号："
            )
        )


        self.index_spin = QSpinBox()

        self.index_spin.setMinimum(0)

        self.index_spin.setMaximum(
            len(self.X) - 1
        )

        self.index_spin.setValue(0)

        control_layout.addWidget(
            self.index_spin
        )


        # 加载按钮
        self.load_button = QPushButton(
            "加载样本"
        )


        self.load_button.clicked.connect(
            self.on_load_sample
        )

        control_layout.addWidget(
            self.load_button
        )


        # 随机样本
        self.random_button = QPushButton(
            "随机样本"
        )
        # ============================================================
        # 导入外部IQ文件
        # ============================================================

        self.import_button = QPushButton(
            "导入 IQ 文件"
        )

        self.import_button.clicked.connect(
            self.on_import_iq
        )

        control_layout.addWidget(
            self.import_button
        )
        control_layout.addWidget(
            QLabel(
                "采样率："
            )
        )

        self.fs_spin = QDoubleSpinBox()

        self.fs_spin.setRange(
            0.1,
            1000.0
        )

        self.fs_spin.setDecimals(3)

        # MHz
        self.fs_spin.setValue(
            self.fs / 1e6
        )

        self.fs_spin.setSuffix(
            " MHz"
        )

        control_layout.addWidget(
            self.fs_spin
        )

        self.random_button.clicked.connect(
            self.on_random_sample
        )

        control_layout.addWidget(
            self.random_button
        )


        # 分析按钮
        self.analyze_button = QPushButton(
            "开始识别与参数测量"
        )
        # ============================================================
        # 导出分析报告
        # ============================================================

        self.export_button = QPushButton(
            "导出分析报告"
        )

        self.export_button.clicked.connect(
            self.on_export_report
        )

        self.export_button.setEnabled(
            False
        )

        control_layout.addWidget(
            self.export_button
        )

        self.analyze_button.clicked.connect(
            self.on_analyze
        )

        control_layout.addWidget(
            self.analyze_button
        )


        control_layout.addStretch()


        # 当前样本信息
        self.sample_info_label = QLabel()

        control_layout.addWidget(
            self.sample_info_label
        )


        main_layout.addWidget(
            control_group
        )


        # ====================================================
        # 中部图像区
        # ====================================================

        plot_layout = QHBoxLayout()


        # ----------------------------------------------------
        # IQ波形
        # ----------------------------------------------------

        iq_group = QGroupBox(
            "时域 I/Q 波形"
        )

        iq_layout = QVBoxLayout(
            iq_group
        )

        self.iq_canvas = PlotCanvas(
            width=7,
            height=5
        )

        iq_layout.addWidget(
            self.iq_canvas
        )


        # ----------------------------------------------------
        # STFT
        # ----------------------------------------------------

        stft_group = QGroupBox(
            "STFT 时频图"
        )

        stft_layout = QVBoxLayout(
            stft_group
        )

        self.stft_canvas = PlotCanvas(
            width=7,
            height=5
        )

        stft_layout.addWidget(
            self.stft_canvas
        )


        plot_layout.addWidget(
            iq_group
        )

        plot_layout.addWidget(
            stft_group
        )


        main_layout.addLayout(
            plot_layout,
            stretch=3
        )


        # ====================================================
        # 底部结果区
        # ====================================================

        result_layout = QHBoxLayout()


        # ----------------------------------------------------
        # 分类结果
        # ----------------------------------------------------

        classification_group = QGroupBox(
            "信号识别结果"
        )

        classification_layout = QVBoxLayout(
            classification_group
        )


        self.classification_text = QTextEdit()

        self.classification_text.setReadOnly(
            True
        )

        classification_layout.addWidget(
            self.classification_text
        )


        # ----------------------------------------------------
        # 参数测量结果
        # ----------------------------------------------------

        parameter_group = QGroupBox(
            "参数测量结果"
        )

        parameter_layout = QVBoxLayout(
            parameter_group
        )


        self.parameter_text = QTextEdit()

        self.parameter_text.setReadOnly(
            True
        )

        parameter_layout.addWidget(
            self.parameter_text
        )


        result_layout.addWidget(
            classification_group
        )

        result_layout.addWidget(
            parameter_group
        )


        main_layout.addLayout(
            result_layout,
            stretch=2
        )

        # ============================================================
        # 参数测量精度对比
        # ============================================================

        comparison_group = QGroupBox(
            "真实参数与测量结果对比"
        )

        comparison_layout = QVBoxLayout(
            comparison_group
        )

        self.comparison_table = QTableWidget()

        self.comparison_table.setColumnCount(4)

        self.comparison_table.setHorizontalHeaderLabels([
            "参数",
            "真实值",
            "测量值",
            "误差 / 结果"
        ])

        header = (
            self.comparison_table
            .horizontalHeader()
        )

        header.setSectionResizeMode(
            QHeaderView.Stretch
        )

        self.comparison_table.setAlternatingRowColors(
            True
        )

        comparison_layout.addWidget(
            self.comparison_table
        )

        main_layout.addWidget(
            comparison_group,
            stretch=2
        )


    # ========================================================
    # 加载指定样本
    # ========================================================

    def load_sample(
        self,
        index
    ):

        self.signal_source = "dataset"

        self.external_file_path = None

        # 恢复数据库固定采样率
        self.fs = float(
            self.data["fs"]
        )

        self.fs_spin.setValue(
            self.fs / 1e6
        )

        self.current_index = int(
            index
        )


        I = self.X[
            self.current_index,
            0
        ]

        Q = self.X[
            self.current_index,
            1
        ]


        self.current_signal = (

            I

            + 1j * Q
        )


        true_label = int(
            self.y[
                self.current_index
            ]
        )


        true_class = (
            self.class_names[
                true_label
            ]
        )


        snr_value = int(
            self.snr[
                self.current_index
            ]
        )


        self.sample_info_label.setText(

            f"真实类别：{true_class}"
            f"    "
            f"SNR：{snr_value} dB"
        )

        self.last_result = None

        self.export_button.setEnabled(
            False
        )

        # 清空旧结果
        self.classification_text.clear()

        self.parameter_text.clear()


        # 画图
        self.plot_iq()

        self.plot_stft()


    # ========================================================
    # 点击加载
    # ========================================================

    def on_load_sample(self):

        index = (
            self.index_spin.value()
        )

        self.load_sample(
            index
        )


    # ========================================================
    # 随机样本
    # ========================================================

    def on_random_sample(self):

        index = np.random.randint(
            0,
            len(self.X)
        )


        self.index_spin.setValue(
            index
        )


        self.load_sample(
            index
        )


    # ========================================================
    # IQ波形
    # ========================================================

    def plot_iq(self):

        figure = (
            self.iq_canvas.figure
        )

        figure.clear()


        ax = figure.add_subplot(
            111
        )


        n = len(
            self.current_signal
        )


        time_us = (

            np.arange(n)

            / self.fs

            * 1e6
        )


        ax.plot(
            time_us,
            np.real(
                self.current_signal
            ),
            label="I"
        )


        ax.plot(
            time_us,
            np.imag(
                self.current_signal
            ),
            label="Q",
            alpha=0.8
        )


        ax.set_xlabel(
            "Time (μs)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.set_title(
            "Time-Domain I/Q Signal"
        )

        ax.legend()

        ax.grid(
            alpha=0.3
        )


        self.iq_canvas.draw()


    # ========================================================
    # STFT绘图
    # ========================================================

    def plot_stft(self):

        stft_tensor = iq_to_stft(

            self.current_signal,

            self.fs
        )


        # shape:
        #
        # (1,1,F,T)
        stft_image = (

            stft_tensor[
                0,
                0
            ]

            .detach()
            .cpu()
            .numpy()
        )


        figure = (
            self.stft_canvas.figure
        )

        figure.clear()


        ax = figure.add_subplot(
            111
        )


        duration_us = (

            len(
                self.current_signal
            )

            / self.fs

            * 1e6
        )


        frequency_mhz = (

            self.fs

            / 2

            / 1e6
        )


        image = ax.imshow(

            stft_image,

            origin="lower",

            aspect="auto",

            extent=[
                0,
                duration_us,
                -frequency_mhz,
                frequency_mhz
            ]
        )


        ax.set_xlabel(
            "Time (μs)"
        )

        ax.set_ylabel(
            "Frequency (MHz)"
        )

        ax.set_title(
            "STFT Time-Frequency Representation"
        )


        figure.colorbar(
            image,
            ax=ax,
            label="Normalized Magnitude"
        )


        self.stft_canvas.draw()


    # ========================================================
    # 分析
    # ========================================================

    def on_analyze(self):

        if self.current_signal is None:

            QMessageBox.warning(
                self,
                "提示",
                "请先加载信号"
            )

            return


        self.last_result = None
        self.export_button.setEnabled(False)

        # 暂时禁用按钮
        self.analyze_button.setEnabled(
            False
        )

        self.analyze_button.setText(
            "正在分析..."
        )


        QApplication.processEvents()


        try:

            result = analyze_signal(

                self.current_signal,

                self.fs,

                self.model
            )

            # 保存最近一次结果
            self.last_result = result

            self.show_classification_result(
                result["classification"]
            )

            self.show_parameter_result(
                result["parameters"]
            )

            self.show_comparison_table(
                result["parameters"]
            )

            self.export_button.setEnabled(
                True
            )


        except Exception as error:

            QMessageBox.critical(

                self,

                "分析错误",

                str(error)
            )


        finally:

            self.analyze_button.setEnabled(
                True
            )

            self.analyze_button.setText(
                "开始识别与参数测量"
            )


    # ========================================================
    # 显示分类结果
    # ========================================================

    def show_classification_result(
        self,
        result
    ):

        predicted_class = (
            result[
                "class_name"
            ]
        )


        confidence = (
            result[
                "confidence"
            ]
        )


        lines = []


        lines.append(
            f"预测类别：{predicted_class}"
        )


        lines.append(

            f"识别置信度："
            f"{confidence * 100:.2f}%"
        )


        lines.append("")

        lines.append(
            "各类别概率："
        )


        for class_name, probability in (

            result[
                "probabilities"
            ].items()
        ):

            lines.append(

                f"{class_name:12s}"
                f" : "
                f"{probability * 100:.4f}%"
            )


        # ====================================================
        # 与真实类别比较
        # ====================================================

        # ============================================================
        # 如果来自数据库
        # 才能比较真实类别
        # ============================================================

        if self.signal_source == "dataset":

            true_label = int(
                self.y[
                    self.current_index
                ]
            )

            true_class = (
                self.class_names[
                    true_label
                ]
            )

            lines.append("")

            if predicted_class == true_class:

                lines.append(
                    "识别结果：正确 ✓"
                )

            else:

                lines.append(
                    "识别结果：错误 ✗"
                )

                lines.append(
                    f"真实类别：{true_class}"
                )


        # ============================================================
        # 外部未知信号
        # ============================================================

        else:

            lines.append("")

            lines.append(
                "信号来源：外部未知IQ信号"
            )


        self.classification_text.setPlainText(

            "\n".join(
                lines
            )
        )


    # ========================================================
    # 数值格式
    # ========================================================

    def format_parameter(
        self,
        key,
        value
    ):

        # ----------------------------------------------------
        # 频率
        # ----------------------------------------------------

        if key in (
            "center_freq_hz",
            "start_freq_hz",
            "end_freq_hz",
            "f1_hz",
            "f2_hz",
            "freq_sep_hz",
            "residual_frequency_hz",
            "ridge_rmse_hz"
        ):

            return (
                f"{value / 1e6:.6f} MHz"
            )


        # ----------------------------------------------------
        # 带宽
        # ----------------------------------------------------

        if key == "bandwidth_hz":

            return (
                f"{value / 1e6:.6f} MHz"
            )


        # ----------------------------------------------------
        # 调频斜率
        # ----------------------------------------------------

        if key == "chirp_rate_hz_per_s":

            return (
                f"{value:.6e} Hz/s"
            )


        # ----------------------------------------------------
        # 码元率
        # ----------------------------------------------------

        if key == "symbol_rate_baud":

            return (
                f"{value / 1e3:.3f} kBaud"
            )


        # ----------------------------------------------------
        # 码元时间
        # ----------------------------------------------------

        if key == "symbol_duration_s":

            return (
                f"{value * 1e6:.3f} μs"
            )


        # ----------------------------------------------------
        # ndarray
        # ----------------------------------------------------

        if isinstance(
            value,
            np.ndarray
        ):

            return str(
                value.tolist()
            )


        return str(value)


    # ========================================================
    # 显示参数
    # ========================================================

    def show_parameter_result(
        self,
        result
    ):

        lines = []


        # 中文名称
        parameter_names = {

            "center_freq_hz":
                "中心频率",

            "bandwidth_hz":
                "带宽",

            "chirp_rate_hz_per_s":
                "调频斜率",

            "chirp_direction":
                "扫频方向",

            "start_freq_hz":
                "起始频率",

            "end_freq_hz":
                "终止频率",

            "ridge_rmse_hz":
                "脊线拟合RMSE",

            "symbol_count":
                "码元数",

            "symbol_rate_baud":
                "码元速率",

            "symbol_duration_s":
                "码元持续时间",

            "symbol_sequence":
                "PSK码序列",

            "f1_hz":
                "FSK频率1",

            "f2_hz":
                "FSK频率2",

            "freq_sep_hz":
                "FSK频率间隔",

            "fsk_code":
                "FSK跳频序列",

            "bpsk_code":
                "BPSK码序列",

            "residual_frequency_hz":
                "Dechirp残余频偏",

            "fit_score":
                "拟合评分",

            "carrier_mode":
                "载波相位模型"
        }


        for key, value in result.items():

            # ------------------------------------------------
            # 不显示内部诊断大数据
            # ------------------------------------------------

            if key in (
                "dechirped_signal",
                "candidate_scores",
                "common_phase_rad",
                "modulation_order"
            ):

                continue


            display_name = (
                parameter_names.get(
                    key,
                    key
                )
            )


            # ------------------------------------------------
            # 扫频方向特殊显示
            # ------------------------------------------------

            if key == "chirp_direction":

                if value == 1:

                    formatted = (
                        "上扫频"
                    )

                else:

                    formatted = (
                        "下扫频"
                    )

            else:

                formatted = (
                    self.format_parameter(
                        key,
                        value
                    )
                )


            lines.append(

                f"{display_name}："
                f"{formatted}"
            )


        self.parameter_text.setPlainText(

            "\n".join(
                lines
            )
        )

    # ============================================================
    # 导入外部IQ
    # ============================================================

    def on_import_iq(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 IQ 信号文件",
            "",
            "IQ Files (*.npy *.npz);;All Files (*)"
        )

        if not file_path:
            return

        try:

            path = Path(
                file_path
            )

            # ====================================================
            # NPY
            # ====================================================

            if path.suffix.lower() == ".npy":

                raw = np.load(
                    file_path,
                    allow_pickle=False
                )

                signal = (
                    self.parse_iq_array(
                        raw
                    )
                )


            # ====================================================
            # NPZ
            # ====================================================

            elif path.suffix.lower() == ".npz":

                npz_data = np.load(
                    file_path,
                    allow_pickle=False
                )

                signal = None

                # -----------------------------------------------
                # 优先找常见字段名
                # -----------------------------------------------

                candidate_keys = [
                    "signal",
                    "iq",
                    "IQ",
                    "x",
                    "X"
                ]

                for key in candidate_keys:

                    if key in npz_data.files:

                        raw = npz_data[
                            key
                        ]

                        # 如果X是一个完整数据库
                        # 例如 (N,2,2048)
                        if (
                                key == "X"
                                and
                                raw.ndim == 3
                        ):
                            raw = raw[0]

                        signal = (
                            self.parse_iq_array(
                                raw
                            )
                        )

                        break

                if signal is None:
                    npz_data.close()

                    raise ValueError(
                        "NPZ文件中没有找到 "
                        "signal / iq / IQ / x / X 字段"
                    )

                # -----------------------------------------------
                # 如果文件本身包含fs
                # 自动读取
                # -----------------------------------------------

                if "fs" in npz_data.files:
                    external_fs = float(
                        npz_data["fs"]
                    )

                    self.fs_spin.setValue(
                        external_fs
                        / 1e6
                    )

                npz_data.close()


            else:

                raise ValueError(
                    "目前只支持 .npy 和 .npz 文件"
                )

            # ====================================================
            # 长度检查
            # ====================================================

            if len(signal) != 2048:
                raise ValueError(

                    f"当前模型要求2048个采样点，"
                    f"导入文件为 {len(signal)} 个采样点。"
                )

            # ====================================================
            # 保存
            # ====================================================

            self.current_signal = (
                signal.astype(
                    np.complex64
                )
            )

            self.external_file_path = (
                file_path
            )

            self.signal_source = (
                "external"
            )

            self.external_true_class = None

            # 新信号不能复用上一条信号的分析报告。
            self.last_result = None
            self.export_button.setEnabled(False)

            # GUI中的采样率
            self.fs = (

                    self.fs_spin.value()

                    * 1e6
            )

            # ====================================================
            # 信息显示
            # ====================================================

            self.sample_info_label.setText(

                f"外部文件：{path.name}"
                f"    "
                f"Fs：{self.fs / 1e6:.3f} MHz"
            )

            self.classification_text.clear()

            self.parameter_text.clear()

            self.comparison_table.setRowCount(
                0
            )

            self.plot_iq()

            self.plot_stft()

            QMessageBox.information(

                self,

                "导入成功",

                f"IQ文件加载成功\n\n"
                f"文件：{path.name}\n"
                f"采样点：{len(signal)}\n"
                f"采样率：{self.fs / 1e6:.3f} MHz"
            )

        except Exception as error:

            QMessageBox.critical(

                self,

                "IQ文件读取失败",

                str(error)
            )

    # ============================================================
    # 将不同格式的数组转换成复数IQ
    # ============================================================

    def parse_iq_array(
            self,
            array
    ):

        array = np.asarray(
            array
        )

        # ========================================================
        # 情况1：
        #
        # complex ndarray
        #
        # shape:
        # (2048,)
        # ========================================================

        if (
                array.ndim == 1
                and
                np.iscomplexobj(array)
        ):
            return array.astype(
                np.complex64
            )

        # ========================================================
        # 情况2：
        #
        # (2,N)
        #
        # 第一行为I
        # 第二行为Q
        # ========================================================

        if (
                array.ndim == 2
                and
                array.shape[0] == 2
        ):
            I = array[0]

            Q = array[1]

            return (

                    I

                    + 1j * Q

            ).astype(
                np.complex64
            )

        # ========================================================
        # 情况3：
        #
        # (N,2)
        # ========================================================

        if (
                array.ndim == 2
                and
                array.shape[1] == 2
        ):
            I = array[:, 0]

            Q = array[:, 1]

            return (

                    I

                    + 1j * Q

            ).astype(
                np.complex64
            )

        raise ValueError(

            "无法识别IQ数组格式。\n\n"

            "支持：\n"
            "1. complex数组：(N,)\n"
            "2. I/Q双通道：(2,N)\n"
            "3. I/Q双通道：(N,2)"
        )

    def get_ground_truth(
            self
    ):

        # ========================================================
        # 外部IQ没有Ground Truth
        # ========================================================

        if self.signal_source != "dataset":
            return None

        index = self.current_index

        true_class = self.class_names[
            int(
                self.y[index]
            )
        ]

        sample_params = self.params[
            index
        ]

        def value(
                name
        ):

            return sample_params[
                self.param_index[
                    name
                ]
            ]

        ground_truth = {}

        # ========================================================
        # LFM
        # ========================================================

        if true_class == "LFM":

            ground_truth = {

                "center_freq_hz":
                    value(
                        "center_freq_hz"
                    ),

                "bandwidth_hz":
                    value(
                        "bandwidth_hz"
                    ),

                "chirp_rate_hz_per_s":
                    value(
                        "chirp_rate_hz_per_s"
                    ),

                "chirp_direction":
                    int(
                        value(
                            "chirp_direction"
                        )
                    )
            }


        # ========================================================
        # BPSK / QPSK
        # ========================================================

        elif true_class in (
                "BPSK",
                "QPSK"
        ):

            symbol_count = int(
                value(
                    "symbol_count"
                )
            )

            ground_truth = {

                "center_freq_hz":
                    value(
                        "center_freq_hz"
                    ),

                "symbol_count":
                    symbol_count,

                "symbol_rate_baud":
                    value(
                        "symbol_rate_baud"
                    ),

                "symbol_sequence":
                    self.codes[
                    index,
                    :symbol_count
                    ].astype(
                        np.int64
                    )
            }


        # ========================================================
        # LFM-BPSK / LFM-QPSK
        # ========================================================

        elif true_class in (
                "LFM_BPSK",
                "LFM_QPSK"
        ):

            symbol_count = int(
                value(
                    "symbol_count"
                )
            )

            ground_truth = {

                "center_freq_hz":
                    value(
                        "center_freq_hz"
                    ),

                "bandwidth_hz":
                    value(
                        "bandwidth_hz"
                    ),

                "chirp_rate_hz_per_s":
                    value(
                        "chirp_rate_hz_per_s"
                    ),

                "chirp_direction":
                    int(
                        value(
                            "chirp_direction"
                        )
                    ),

                "symbol_count":
                    symbol_count,

                "symbol_rate_baud":
                    value(
                        "symbol_rate_baud"
                    ),

                "symbol_sequence":
                    self.codes[
                    index,
                    :symbol_count
                    ].astype(
                        np.int64
                    )
            }


        # ========================================================
        # 2FSK-BPSK
        # ========================================================

        elif true_class == "2FSK_BPSK":

            symbol_count = int(
                value(
                    "symbol_count"
                )
            )

            ground_truth = {

                "center_freq_hz":
                    value(
                        "center_freq_hz"
                    ),

                "f1_hz":
                    value(
                        "f1_hz"
                    ),

                "f2_hz":
                    value(
                        "f2_hz"
                    ),

                "freq_sep_hz":
                    value(
                        "freq_sep_hz"
                    ),

                "symbol_count":
                    symbol_count,

                "symbol_rate_baud":
                    value(
                        "symbol_rate_baud"
                    ),

                "bpsk_code":
                    self.codes[
                    index,
                    :symbol_count
                    ].astype(
                        np.int64
                    ),

                "fsk_code":
                    self.fsk_codes[
                    index,
                    :symbol_count
                    ].astype(
                        np.int64
                    )
            }

        return ground_truth

    def calculate_psk_code_accuracy(
            self,
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

    def calculate_sequence_accuracy(
            self,
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

    def calculate_error_text(
            self,
            key,
            true_value,
            estimated_value,
            true_class
    ):

        # ========================================================
        # 码序列
        # ========================================================

        if key == "symbol_sequence":
            modulation_order = (

                2

                if true_class in (
                    "BPSK",
                    "LFM_BPSK"
                )

                else 4
            )

            accuracy = (
                self.calculate_psk_code_accuracy(

                    estimated_value,

                    true_value,

                    modulation_order
                )
            )

            return (
                f"恢复率 "
                f"{accuracy * 100:.2f}%"
            )

        # ========================================================
        # BPSK序列
        # ========================================================

        if key == "bpsk_code":
            accuracy = (
                self.calculate_psk_code_accuracy(

                    estimated_value,

                    true_value,

                    2
                )
            )

            return (
                f"恢复率 "
                f"{accuracy * 100:.2f}%"
            )

        # ========================================================
        # FSK序列
        # ========================================================

        if key == "fsk_code":
            accuracy = (
                self.calculate_sequence_accuracy(

                    estimated_value,

                    true_value
                )
            )

            return (
                f"恢复率 "
                f"{accuracy * 100:.2f}%"
            )

        # ========================================================
        # 扫频方向
        # ========================================================

        if key == "chirp_direction":

            if int(
                    true_value
            ) == int(
                estimated_value
            ):

                return "正确"

            else:

                return "错误"

        # ========================================================
        # 码元数
        # ========================================================

        if key == "symbol_count":

            if int(
                    true_value
            ) == int(
                estimated_value
            ):

                return "正确"

            else:

                return (
                    f"误差 "
                    f"{abs(int(estimated_value) - int(true_value))}"
                )

        # ========================================================
        # 普通连续参数
        # ========================================================

        try:

            true_value = float(
                true_value
            )

            estimated_value = float(
                estimated_value
            )

        except Exception:

            return "—"

        absolute_error = abs(

            estimated_value
            -
            true_value
        )

        # ========================================================
        # 频率
        # ========================================================

        if key in (
                "center_freq_hz",
                "f1_hz",
                "f2_hz"
        ):
            return (
                f"{absolute_error / 1e3:.3f} kHz"
            )

        # ========================================================
        # 相对误差
        # ========================================================

        if abs(true_value) > 1e-12:
            relative_error = (

                    absolute_error

                    /
                    abs(true_value)

                    * 100
            )

            return (
                f"{relative_error:.3f}%"
            )

        return (
            str(
                absolute_error
            )
        )

    def show_comparison_table(
            self,
            estimated_result
    ):

        self.comparison_table.setRowCount(
            0
        )

        # ========================================================
        # 外部未知IQ
        # ========================================================

        if self.signal_source != "dataset":

            keys = [

                key

                for key in estimated_result.keys()

                if key not in (
                    "dechirped_signal",
                    "candidate_scores",
                    "common_phase_rad",
                    "modulation_order"
                )
            ]

            self.comparison_table.setRowCount(
                len(keys)
            )

            for row, key in enumerate(
                    keys
            ):
                value = estimated_result[
                    key
                ]

                self.comparison_table.setItem(

                    row,
                    0,

                    QTableWidgetItem(
                        key
                    )
                )

                self.comparison_table.setItem(

                    row,
                    1,

                    QTableWidgetItem(
                        "—"
                    )
                )

                self.comparison_table.setItem(

                    row,
                    2,

                    QTableWidgetItem(

                        self.format_parameter(
                            key,
                            value
                        )
                    )
                )

                self.comparison_table.setItem(

                    row,
                    3,

                    QTableWidgetItem(
                        "无 Ground Truth"
                    )
                )

            return

        # ========================================================
        # 数据库样本
        # ========================================================

        ground_truth = (
            self.get_ground_truth()
        )

        true_class = self.class_names[

            int(
                self.y[
                    self.current_index
                ]
            )
        ]

        # 中文参数名
        name_map = {

            "center_freq_hz":
                "中心频率",

            "bandwidth_hz":
                "带宽",

            "chirp_rate_hz_per_s":
                "调频斜率",

            "chirp_direction":
                "扫频方向",

            "symbol_count":
                "码元数",

            "symbol_rate_baud":
                "码元速率",

            "symbol_sequence":
                "PSK码序列",

            "f1_hz":
                "FSK频率1",

            "f2_hz":
                "FSK频率2",

            "freq_sep_hz":
                "FSK频率间隔",

            "bpsk_code":
                "BPSK码序列",

            "fsk_code":
                "FSK跳频序列"
        }

        rows = []

        for key, true_value in (
                ground_truth.items()
        ):

            if key not in estimated_result:
                continue

            estimated_value = (
                estimated_result[
                    key
                ]
            )

            error_text = (
                self.calculate_error_text(

                    key,

                    true_value,

                    estimated_value,

                    true_class
                )
            )

            rows.append((

                key,

                true_value,

                estimated_value,

                error_text
            ))

        self.comparison_table.setRowCount(
            len(rows)
        )

        for row_index, (
                key,
                true_value,
                estimated_value,
                error_text
        ) in enumerate(
            rows
        ):
            # 参数名称
            self.comparison_table.setItem(

                row_index,
                0,

                QTableWidgetItem(

                    name_map.get(
                        key,
                        key
                    )
                )
            )

            # 真值
            self.comparison_table.setItem(

                row_index,
                1,

                QTableWidgetItem(

                    self.format_parameter(
                        key,
                        true_value
                    )
                )
            )

            # 测量值
            self.comparison_table.setItem(

                row_index,
                2,

                QTableWidgetItem(

                    self.format_parameter(
                        key,
                        estimated_value
                    )
                )
            )

            # 误差
            self.comparison_table.setItem(

                row_index,
                3,

                QTableWidgetItem(
                    error_text
                )
            )

    # ============================================================
    # 导出分析报告
    # ============================================================

    def on_export_report(self):

        if self.last_result is None:
            QMessageBox.warning(
                self,
                "提示",
                "请先完成一次信号识别与参数测量。"
            )

            return

        # ========================================================
        # 默认文件名
        # ========================================================

        if self.signal_source == "dataset":

            default_name = (
                f"radar_report_sample_"
                f"{self.current_index}.txt"
            )

        else:

            if self.external_file_path:

                file_stem = Path(
                    self.external_file_path
                ).stem

                default_name = (
                    f"radar_report_"
                    f"{file_stem}.txt"
                )

            else:

                default_name = (
                    "radar_analysis_report.txt"
                )

        default_path = str(

            PROJECT_ROOT
            /
            default_name
        )

        file_path, _ = (
            QFileDialog.getSaveFileName(

                self,

                "保存分析报告",

                default_path,

                "Text Report (*.txt)"
            )
        )

        if not file_path:
            return

        try:

            classification = (
                self.last_result[
                    "classification"
                ]
            )

            parameters = (
                self.last_result[
                    "parameters"
                ]
            )

            lines = []

            # ====================================================
            # 标题
            # ====================================================

            lines.append(
                "=" * 70
            )

            lines.append(
                "雷达脉内复合信号智能识别与参数测量分析报告"
            )

            lines.append(
                "=" * 70
            )

            lines.append("")

            # ====================================================
            # 信号信息
            # ====================================================

            lines.append(
                "一、信号基本信息"
            )

            lines.append(
                "-" * 70
            )

            if self.signal_source == "dataset":

                true_label = int(

                    self.y[
                        self.current_index
                    ]
                )

                true_class = (

                    self.class_names[
                        true_label
                    ]
                )

                snr_value = int(

                    self.snr[
                        self.current_index
                    ]
                )

                lines.append(
                    f"信号来源：测试数据库"
                )

                lines.append(
                    f"样本编号："
                    f"{self.current_index}"
                )

                lines.append(
                    f"真实类别："
                    f"{true_class}"
                )

                lines.append(
                    f"SNR："
                    f"{snr_value} dB"
                )

            else:

                lines.append(
                    "信号来源：外部IQ文件"
                )

                lines.append(
                    f"文件："
                    f"{Path(self.external_file_path).name}"
                )

            lines.append(
                f"采样率："
                f"{self.fs / 1e6:.6f} MHz"
            )

            lines.append(
                f"采样点数："
                f"{len(self.current_signal)}"
            )

            lines.append("")

            # ====================================================
            # 识别结果
            # ====================================================

            lines.append(
                "二、信号识别结果"
            )

            lines.append(
                "-" * 70
            )

            predicted_class = (
                classification[
                    "class_name"
                ]
            )

            confidence = (
                classification[
                    "confidence"
                ]
            )

            lines.append(
                f"预测类别："
                f"{predicted_class}"
            )

            lines.append(
                f"识别置信度："
                f"{confidence * 100:.4f}%"
            )

            if self.signal_source == "dataset":

                if predicted_class == true_class:

                    lines.append(
                        "识别结果：正确"
                    )

                else:

                    lines.append(
                        "识别结果：错误"
                    )

            lines.append("")

            lines.append(
                "各类别概率："
            )

            for (
                    class_name,
                    probability
            ) in classification[
                "probabilities"
            ].items():
                lines.append(

                    f"  "
                    f"{class_name:12s}"
                    f" : "
                    f"{probability * 100:.6f}%"
                )

            lines.append("")

            # ====================================================
            # 参数测量
            # ====================================================

            lines.append(
                "三、参数测量结果"
            )

            lines.append(
                "-" * 70
            )

            parameter_names = {

                "center_freq_hz":
                    "中心频率",

                "bandwidth_hz":
                    "带宽",

                "chirp_rate_hz_per_s":
                    "调频斜率",

                "chirp_direction":
                    "扫频方向",

                "start_freq_hz":
                    "起始频率",

                "end_freq_hz":
                    "终止频率",

                "ridge_rmse_hz":
                    "脊线拟合RMSE",

                "symbol_count":
                    "码元数",

                "symbol_rate_baud":
                    "码元速率",

                "symbol_duration_s":
                    "码元持续时间",

                "symbol_sequence":
                    "PSK码序列",

                "f1_hz":
                    "FSK频率1",

                "f2_hz":
                    "FSK频率2",

                "freq_sep_hz":
                    "FSK频率间隔",

                "fsk_code":
                    "FSK跳频序列",

                "bpsk_code":
                    "BPSK码序列",

                "residual_frequency_hz":
                    "Dechirp残余频偏",

                "fit_score":
                    "拟合评分",

                "carrier_mode":
                    "载波相位模型"
            }

            ignored_keys = {
                "dechirped_signal",
                "candidate_scores",
                "common_phase_rad",
                "modulation_order"
            }

            for key, value in parameters.items():

                if key in ignored_keys:
                    continue

                display_name = (
                    parameter_names.get(
                        key,
                        key
                    )
                )

                if key == "chirp_direction":

                    formatted_value = (

                        "上扫频"

                        if int(value) == 1

                        else "下扫频"
                    )

                else:

                    formatted_value = (

                        self.format_parameter(
                            key,
                            value
                        )
                    )

                lines.append(

                    f"{display_name}："
                    f"{formatted_value}"
                )

            # ====================================================
            # 数据库样本再增加误差对比
            # ====================================================

            if self.signal_source == "dataset":

                lines.append("")

                lines.append(
                    "四、真实参数与测量误差"
                )

                lines.append(
                    "-" * 70
                )

                ground_truth = (
                    self.get_ground_truth()
                )

                if ground_truth:

                    for (
                            key,
                            true_value
                    ) in ground_truth.items():

                        if key not in parameters:
                            continue

                        estimated_value = (
                            parameters[
                                key
                            ]
                        )

                        error_text = (
                            self.calculate_error_text(

                                key,

                                true_value,

                                estimated_value,

                                true_class
                            )
                        )

                        display_name = (
                            parameter_names.get(
                                key,
                                key
                            )
                        )

                        true_text = (
                            self.format_parameter(

                                key,

                                true_value
                            )
                        )

                        estimated_text = (
                            self.format_parameter(

                                key,

                                estimated_value
                            )
                        )

                        lines.append(
                            f"{display_name}"
                        )

                        lines.append(
                            f"  真实值："
                            f"{true_text}"
                        )

                        lines.append(
                            f"  测量值："
                            f"{estimated_text}"
                        )

                        lines.append(
                            f"  误差/结果："
                            f"{error_text}"
                        )

            lines.append("")

            lines.append(
                "=" * 70
            )

            lines.append(
                "报告生成完成"
            )

            lines.append(
                "=" * 70
            )

            # ====================================================
            # 写文件
            # ====================================================

            with open(
                    file_path,
                    "w",
                    encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(
                        lines
                    )
                )

            QMessageBox.information(

                self,

                "导出成功",

                f"分析报告已保存：\n"
                f"{file_path}"
            )


        except Exception as error:

            QMessageBox.critical(

                self,

                "导出失败",

                str(error)
            )


# ============================================================
# 程序入口
# ============================================================

def main():

    app = QApplication(
        sys.argv
    )

    app.setStyleSheet(
        """
        QMainWindow {
            background: #f4f6f8;
        }

        QGroupBox {
            font-size: 14px;
            font-weight: bold;

            border: 1px solid #c9d2dc;
            border-radius: 8px;

            margin-top: 12px;
            padding-top: 12px;

            background: white;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;

            padding: 0 6px;
        }

        QPushButton {
            min-height: 30px;

            padding-left: 14px;
            padding-right: 14px;

            border: 1px solid #b6c2cf;
            border-radius: 6px;

            background: white;

            font-size: 13px;
        }

        QPushButton:hover {
            background: #eef4ff;

            border: 1px solid #6699dd;
        }

        QPushButton:pressed {
            background: #dceaff;
        }

        QPushButton:disabled {
            color: #999999;

            background: #eeeeee;
        }

        QTextEdit {
            border: 1px solid #d0d7de;
            border-radius: 5px;

            background: white;

            font-family: Consolas;
            font-size: 13px;
        }

        QSpinBox,
        QDoubleSpinBox {
            min-height: 28px;

            border: 1px solid #b6c2cf;
            border-radius: 5px;

            background: white;

            padding-left: 6px;
        }

        QTableWidget {
            border: 1px solid #d0d7de;

            background: white;

            alternate-background-color: #f7f9fb;

            gridline-color: #e4e8ec;

            font-size: 12px;
        }

        QHeaderView::section {
            background: #eaf0f6;

            border: none;
            border-right: 1px solid #d0d7de;

            padding: 6px;

            font-weight: bold;
        }

        QLabel {
            font-size: 13px;
        }
        """
    )

    window = RadarMainWindow()

    window.show()


    sys.exit(
        app.exec_()
    )


if __name__ == "__main__":

    main()
