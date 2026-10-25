# Real-Time Mask R-CNN GPU Webcam Detection & Instance Segmentation 🚀

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B%20(CUDA%2012.4)-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](./LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey?style=for-the-badge)]()

An end-to-end, high-performance deep learning instance segmentation application designed for real-time live webcam streams and recorded video footage. Powered by **PyTorch**, **Torchvision**, **NVIDIA CUDA GPU acceleration**, and **OpenCV DirectShow backend**.

Utilizing **Mask R-CNN ResNet-50 FPN V2**, the system performs pixel-level instance segmentation, generates translucent color-coded masks with anti-aliased contour borders, draws bounding boxes and confidence pill badges, and displays real-time telemetry (FPS and inference latency) with live interactive keyboard controls.

---

## 📑 Table of Contents

- [✨ Key Features](#-key-features)
- [🏗️ Pipeline Architecture](#️-pipeline-architecture)
- [📦 Hardware & Environment Compatibility](#-hardware--environment-compatibility)
- [⚡ Quick Start](#-quick-start)
- [🛠️ Installation & Setup](#️-installation--setup)
- [⌨️ Interactive Keyboard Controls](#️-interactive-keyboard-controls)
- [⚙️ Configuration (`config.yaml`)](#️-configuration-configyaml)
- [💻 Command-Line Interface (CLI)](#-command-line-interface-cli)
- [📄 Printable Reference Sheet (PDF)](#-printable-reference-sheet-pdf)
- [📁 Project Structure](#-project-structure)
- [🔧 Diagnostics & Troubleshooting](#-diagnostics--troubleshooting)
- [📜 License](#-license)

---

## ✨ Key Features

- **⚡ NVIDIA CUDA GPU Acceleration**: Built for real-time performance with native CUDA support, FP16 mixed precision (`torch.amp.autocast`), and startup CUDA kernel warm-up to prevent initial frame stutter.
- **🎯 Mask R-CNN ResNet-50 FPN V2**: Employs Torchvision's state-of-the-art V2 architecture pre-trained on the COCO dataset (80 object classes) with an optional V1 fallback.
- **🖥️ Real-Time Telemetry & Heads-Up Display (HUD)**: On-screen monitoring displaying current FPS, neural network inference latency in milliseconds, detected object counts, active device identifier, and recording status.
- **🎛️ Live Dynamic Hotkeys**: Adjust detection confidence thresholds, mask opacity, and display layers (masks, boxes, labels) in real time while the stream is running without restarting.
- **📸 Built-in Media Capture**: Save timestamped high-resolution snapshot images (`.png`) and recorded video clips (`.mp4`) to the `captures/` folder with one keystroke.
- **🎯 Specific Object Class Filtering**: Easily restrict detection to specific targets (e.g., only `["person", "laptop", "cell phone"]`) via YAML config.
- **⚙️ Centralized Configuration**: All camera dimensions, detection thresholds, rendering aesthetics, and storage settings are managed via a single `config.yaml` file.
- **📄 Exportable PDF Cheat Sheet**: Built-in ReportLab generator produces a ready-to-print PDF cheat sheet summarizing all commands, shortcuts, and COCO classes.
- **🚀 One-Click Launchers**: Includes native Windows Batch (`run.bat`) and PowerShell (`run.ps1`) scripts for instant launch.

---

## 🏗️ Pipeline Architecture

```mermaid
flowchart LR
    A[Webcam / Video Stream] -->|cv2.CAP_DSHOW| B[Frame Preprocessing]
    B -->|Tensor Conversion & Normalization| C[GPU VRAM Transfer]
    C -->|torch.cuda.amp.autocast| D[Mask R-CNN ResNet-50 FPN V2]
    D --> E[Predictions: Boxes, Labels, Masks, Scores]
    E -->|Confidence & Mask Thresholds| F[Filtered Detections]
    F --> G[Translucent Mask Blending & Contours]
    G --> H[Bounding Boxes & Label Pills]
    H --> I[HUD Telemetry & Controls Overlay]
    I --> J[Display Window / MP4 Recorder]
```

---

## 📦 Hardware & Environment Compatibility

| Component | Tested & Recommended | Minimum |
| :--- | :--- | :--- |
| **OS** | Windows 10/11 (64-bit), Ubuntu 20.04+ | Windows 10 (64-bit), Linux |
| **Python** | Python 3.11.x | Python 3.10 |
| **GPU** | NVIDIA GeForce RTX 4060 / 3060 or higher | NVIDIA GTX 1060 (6GB) / CPU Mode |
| **CUDA** | CUDA 12.4 / cu124 | CUDA 11.8 or CPU fallback |
| **PyTorch** | PyTorch >= 2.4.0 + Torchvision >= 0.19.0 | PyTorch >= 2.0.0 |
| **Webcam** | 720p / 1080p HD Webcam | Any USB / Integrated Camera |

---

## ⚡ Quick Start

### Windows One-Click Launch
Double-click:
```cmd
run.bat
```
Or run the PowerShell launcher:
```powershell
.\run.ps1
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pmofakher/mask-rcnn-gpu-webcam.git
cd mask-rcnn-gpu-webcam
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA Support
For NVIDIA GPUs with CUDA 12.4 support (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
*(For CPU-only machines: `pip install torch torchvision`)*

### 4. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify GPU Environment
Run the included hardware diagnostic script:
```bash
python test_gpu.py
```
*Expected output: Displays your GPU model, VRAM, and runs a warm-up matrix multiplication test.*

### 6. (Optional) Pre-cache Model Weights
To prevent downloading weights on first webcam launch:
```bash
python download_weights.py
```

---

## ⌨️ Interactive Keyboard Controls

While the detection window is focused, interact with the application using the following keys:

| Key | Action | Description |
| :---: | :--- | :--- |
| `M` | **Toggle Masks** | Toggle translucent segmentation mask overlays on/off |
| `B` | **Toggle Bounding Boxes** | Toggle colored bounding box rectangles on/off |
| `L` | **Toggle Labels** | Toggle class label names and confidence score pills on/off |
| `+` / `=` | **Increase Confidence** | Increase minimum confidence score (+0.05) |
| `-` / `_` | **Decrease Confidence** | Decrease minimum confidence score (-0.05) |
| `[` | **Decrease Opacity** | Lower mask opacity by 0.05 (more transparent) |
| `]` | **Increase Opacity** | Increase mask opacity by 0.05 (more solid) |
| `S` | **Snapshot** | Capture and save current visual frame to `captures/snapshot_*.png` |
| `R` | **Record Video** | Start or stop real-time video recording to `captures/recording_*.mp4` |
| `H` | **Toggle Help HUD** | Show or hide the bottom shortcut assistance bar |
| `Q` / `ESC` | **Quit** | Stop all recording, release camera, and exit cleanly |

---

## ⚙️ Configuration (`config.yaml`)

All parameters are centralized in [`config.yaml`](./config.yaml). Edit this file to modify detection behavior without changing code:

```yaml
model:
  version: "v2"          # "v2" (ResNet-50 FPN V2, higher mAP) or "v1"
  device: "cuda"         # "cuda" (NVIDIA GPU) or "cpu"
  use_fp16: true         # Mixed precision FP16 (improves throughput on Tensor Cores)
  warmup: true           # Pre-warm GPU kernels on startup to eliminate first-frame latency

detection:
  conf_threshold: 0.50   # Minimum confidence threshold (range: 0.05 - 0.95)
  mask_threshold: 0.50   # Mask binarization threshold (range: 0.10 - 0.90)
  target_classes: []     # Filter target objects (e.g. ["person", "laptop", "cell phone"]). Empty = all 80 classes

camera:
  index: 0               # Camera device index (0 = default webcam, 1 = external USB camera)
  width: 1280            # Video stream resolution width
  height: 720            # Video stream resolution height
  video_file: null       # Optional path to a video file (e.g. "test.mp4") for offline testing
  buffer_size: 1         # OpenCV buffer size (1 keeps latency lowest)

visualization:
  draw_masks: true       # Render segmentation masks
  draw_boxes: true       # Render bounding box rectangles
  draw_labels: true      # Render label badges and confidence scores
  show_hud: true         # Render top telemetry bar & bottom controls
  mask_alpha: 0.45       # Mask transparency level (0.10 transparent - 0.90 opaque)
  box_thickness: 2       # Bounding box border thickness
  contour_thickness: 2   # Mask outer contour border thickness
  font_scale: 0.52       # HUD and label text scale

captures:
  save_dir: "captures"   # Directory for saved snapshots and video recordings
  recording_fps: 20.0    # Frame rate for saved MP4 recordings
  video_codec: "mp4v"    # Video encoding codec (mp4v, avc1, XVID)
```

---

## 💻 Command-Line Interface (CLI)

You can temporarily override configuration options via command-line arguments:

```bash
# Launch with default configuration
python main.py

# Use an alternate configuration file
python main.py --config configs/meeting_room.yaml

# Test an existing video file instead of live camera
python main.py --video path/to/sample.mp4

# Select external USB camera (index 1)
python main.py --camera 1

# Override resolution to Full HD (1080p)
python main.py --width 1920 --height 1080

# Override confidence threshold
python main.py --conf 0.65

# Force CPU inference
python main.py --cpu
```

### CLI Arguments Summary

| Flag | Type | Description |
| :--- | :---: | :--- |
| `--config` | `str` | Path to custom YAML configuration file (default: `config.yaml`) |
| `--camera` | `int` | Override camera index from config |
| `--video` | `str` | Path to recorded video file for evaluation |
| `--width` | `int` | Override capture stream width |
| `--height` | `int` | Override capture stream height |
| `--conf` | `float` | Override minimum detection confidence threshold |
| `--cpu` | `flag` | Force CPU execution regardless of GPU presence |

---

## 📄 Printable Reference Sheet (PDF)

The project includes an automated PDF generator that compiles all operational commands, hotkeys, and COCO classes into a professional, printable document:

- **PDF File**: [`Mask_RCNN_Commands_and_Shortkeys.pdf`](./Mask_RCNN_Commands_and_Shortkeys.pdf)

To regenerate the PDF after updating shortcuts or configuration:
```bash
python export_pdf.py
```

---

## 📁 Project Structure

```
mask-rcnn-gpu-webcam/
│
├── captures/                                # Directory for saved snapshots and MP4 recordings
│   └── .gitkeep
├── config.yaml                              # ⭐ Centralized hyperparameters and settings
├── config_loader.py                         # YAML configuration parser with type validation
├── download_weights.py                      # Utility to pre-cache Mask R-CNN COCO weights
├── export_pdf.py                            # ReportLab generator for the PDF cheat sheet
├── main.py                                  # Live webcam capture loop, HUD rendering, and event handler
├── Mask_RCNN_Commands_and_Shortkeys.pdf    # 📄 Exported reference cheat sheet
├── mask_rcnn_detector.py                    # Mask R-CNN neural network wrapper & mask rendering
├── requirements.txt                         # Pip package dependencies
├── run.bat                                  # Windows 1-click batch launcher
├── run.ps1                                  # PowerShell launcher with literal path resolution
├── test_gpu.py                              # Hardware & PyTorch CUDA diagnostics
├── .gitattributes                           # Cross-platform line ending normalization
├── .gitignore                               # Git ignore rules for virtual environments and captures
├── LICENSE                                  # MIT License
└── README.md                                # Official documentation
```

---

## 🔧 Diagnostics & Troubleshooting

<details>
<summary><b>1. Camera does not open or black screen appears</b></summary>

- On Windows, OpenCV uses DirectShow (`cv2.CAP_DSHOW`) for low-latency initialization. Ensure other applications (Zoom, Teams, Skype, or Windows Camera app) are not locking the webcam.
- If using an external USB camera, change the camera index in `config.yaml` to `index: 1` or pass `--camera 1`.
</details>

<details>
<summary><b>2. PyTorch reports CUDA is not available</b></summary>

- Run `python test_gpu.py`.
- If CUDA reports `False`, reinstall PyTorch with the CUDA wheel matching your driver:
  ```bash
  pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu124
  ```
</details>

<details>
<summary><b>3. PowerShell Execution Policy Restriction</b></summary>

- If running `.\run.ps1` gives an execution policy error, run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\run.ps1
  ```
</details>

<details>
<summary><b>4. GPU Memory (VRAM) Management</b></summary>

- Mask R-CNN ResNet-50 FPN V2 occupies approximately 1.8 GB – 2.5 GB of VRAM.
- With FP16 mixed precision enabled (`use_fp16: true`), memory utilization is minimized and throughput increases significantly on RTX Tensor Cores.
</details>

---

## 📜 License

This project is licensed under the **MIT License** - see the [`LICENSE`](./LICENSE) file for details.
