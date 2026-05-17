"""Проверка PyTorch на Windows (c10.dll / WinError 1114)."""
from __future__ import annotations

import platform
import sys


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("---")
    try:
        import torch
    except ModuleNotFoundError:
        print("torch не установлен (pip install torch).")
        print("Без PyTorch: python train_sklearn.py  →  weights/diff_cnn_sklearn.joblib")
        raise SystemExit(1)

    try:
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        x = torch.randn(2, 1, 64, 64)
        print("tensor OK:", tuple(x.shape))
        print("\nPyTorch работает. Запускайте: python train.py")
    except OSError as e:
        print("ОШИБКА загрузки PyTorch:", e)
        print(
            "\nИсправление:\n"
            "1) VC++ Redistributable x64: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist\n"
            "2) pip uninstall torch torchvision torchaudio -y\n"
            "   pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "3) Перезагрузка ПК\n"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
