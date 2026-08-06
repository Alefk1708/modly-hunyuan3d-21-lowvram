"""Create the isolated Modly runtime for Hunyuan3D 2.1 Low VRAM."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from pathlib import Path


MIN_DRIVER = (576, 57)


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def _run(*args: str) -> None:
    print("[setup]", " ".join(args))
    subprocess.run(list(args), check=True)


def _pip(python: Path, *args: str) -> None:
    _run(str(python), "-m", "pip", *args)


def _python_tag(python: Path) -> str:
    return subprocess.check_output(
        [str(python), "-c", "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()


def _driver_version() -> tuple[int, int] | None:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
        parts = [int(value) for value in re.findall(r"\d+", raw)[:2]]
        return (parts[0], parts[1]) if len(parts) == 2 else None
    except Exception:
        return None


def setup(
    python_exe: str,
    ext_dir: Path,
    gpu_sm: int,
    cuda_version: int = 0,
    accelerator: str = "",
    platform_name: str = "",
    **_ignored,
) -> None:
    del platform_name
    ext_dir = ext_dir.resolve()
    venv = ext_dir / "venv"
    if accelerator and accelerator != "cuda":
        raise RuntimeError("This extension is built for NVIDIA CUDA; CPU/MPS/ROCm are not supported.")
    if gpu_sm and gpu_sm < 70:
        raise RuntimeError(f"CUDA GPU is too old (SM {gpu_sm}); this extension requires SM 70 or newer.")

    driver = _driver_version()
    if driver is not None and driver < MIN_DRIVER:
        raise RuntimeError(
            f"NVIDIA driver {driver[0]}.{driver[1]:02d} detected; the WinPortable reference requires "
            "576.57 or newer. Update the driver, then use Repair in Modly."
        )
    if driver is None:
        print("[setup] WARNING: nvidia-smi did not report the driver version; verify the installed version manually.")
    else:
        print(f"[setup] NVIDIA driver {driver[0]}.{driver[1]:02d} (minimum 576.57): OK")

    print(f"[setup] GPU SM={gpu_sm or 'unknown'}, detected CUDA={cuda_version or 'unknown'}")
    _run(python_exe, "-m", "venv", str(venv))
    python = _venv_python(venv)
    tag = _python_tag(python)
    if tag not in {"cp310", "cp311", "cp312"}:
        raise RuntimeError(
            f"Python {tag} is not supported by this extension. Use Modly's Python 3.10, 3.11, or 3.12."
        )

    _pip(python, "install", "--upgrade", "pip==25.1.1", "setuptools==75.8.0", "wheel==0.45.1")
    print("[setup] Installing PyTorch 2.6 / CUDA 12.6...")
    _pip(
        python,
        "install",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
        "torch==2.6.0",
        "torchvision==0.21.0",
    )
    print("[setup] Installing inference and low-VRAM dependencies...")
    requirements = ext_dir / "requirements.txt"
    if not requirements.is_file():
        raise RuntimeError(f"Required file is missing: {requirements}")
    _pip(
        python,
        "install",
        "--retries",
        "10",
        "--timeout",
        "180",
        "-r",
        str(requirements),
    )

    report = {
        "python_tag": tag,
        "torch": "2.6.0+cu126",
        "gpu_sm": gpu_sm,
        "cuda_detected": cuda_version,
        "nvidia_driver": None if driver is None else f"{driver[0]}.{driver[1]:02d}",
    }
    (ext_dir / "install-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[setup] Environment setup complete:", venv)


def _parse_args() -> dict:
    if len(sys.argv) == 2:
        return json.loads(sys.argv[1])
    if len(sys.argv) >= 4:
        return {
            "python_exe": sys.argv[1],
            "ext_dir": sys.argv[2],
            "gpu_sm": int(sys.argv[3]),
            "cuda_version": int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
            "accelerator": "cuda",
        }
    raise SystemExit("Usage: python setup.py '<Modly JSON>'")


if __name__ == "__main__":
    args = _parse_args()
    setup(
        python_exe=args["python_exe"],
        ext_dir=Path(args["ext_dir"]),
        gpu_sm=int(args.get("gpu_sm", 0)),
        cuda_version=int(args.get("cuda_version", 0)),
        accelerator=args.get("accelerator", ""),
        platform_name=args.get("platform", ""),
    )
