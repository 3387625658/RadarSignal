from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


def main():
    data = np.load(

        PROJECT_ROOT
        / "data"
        / "test.npz",

        allow_pickle=False
    )


    # 随便选一个测试样本
    index = 500


    I = data["X"][
        index,
        0
    ]

    Q = data["X"][
        index,
        1
    ]


    signal = (

        I

        + 1j * Q
    )


    output_path = (

        PROJECT_ROOT

        / "external_test_signal.npy"
    )


    np.save(
        output_path,
        signal
    )


    print(
        "导出成功：",
        output_path
    )

    print(
        "真实类别：",
        data["class_names"][
            int(
                data["y"][index]
            )
        ]
    )

    print(
        "SNR：",
        data["snr"][index],
        "dB"
    )


    data.close()

if __name__ == "__main__":
    main()
