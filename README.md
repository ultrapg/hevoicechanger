# HEVoicechanger

<div align="center">

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Language: Rust](https://img.shields.io/badge/Language-Rust_2021-orange.svg)](https://www.rust-lang.org/)
[![Inference: ONNX Runtime](https://img.shields.io/badge/Inference-ONNX_Runtime_v2-brightgreen.svg)](https://onnxruntime.ai/)
[![GUI: egui](https://img.shields.io/badge/GUI-egui_/_eframe-blueviolet.svg)](https://github.com/emilk/egui)
[![Audio: ALSA / CPAL](https://img.shields.io/badge/Audio-ALSA_/_CPAL-red.svg)](https://github.com/RustAudio/cpal)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux_x86__64-informational.svg)](https://kernel.org/)
[![Latest Release](https://img.shields.io/github/v/release/ultrapg/hevoicechanger?color=blue&label=Latest%20Release)](https://github.com/ultrapg/hevoicechanger/releases/latest)

**Ultra-low latency, real-time Linux AI Voice Changer powered by causal streaming neural networks.**

[Download Latest Release](https://github.com/ultrapg/hevoicechanger/releases/latest) •
[Key Features](#key-features) •
[Architecture](#architecture) •
[Prerequisites](#prerequisites) •
[Installation](#installation--building) •
[Usage](#usage) •
[Virtual Microphone Setup](#virtual-microphone-setup) •
[Voice Models](#voice-models) •
[Performance Tuning](#performance-tuning) •
[License](#license)

</div>

---

## Overview

**HEVoicechanger** is a high-performance, real-time AI voice conversion suite engineered in pure Rust for Linux systems. Designed from the ground up for minimal latency and audio stability, it converts live microphone input into target voice profiles using causal streaming ONNX models (such as LLVC).

Unlike traditional voice changers that introduce hundreds of milliseconds of buffer lag or require heavy Python runtimes, HEVoicechanger utilizes:
- Causal convolutional architectures that process audio incrementally without future context.
- Lock-free single-producer single-consumer ring buffers for glitch-free audio transfer.
- Native multithreaded ONNX Runtime (`ort`) inference with Level 3 graph optimizations.
- A lightweight, responsive dark-mode GUI built with `egui`/`eframe`, alongside a headless CLI daemon mode.

---

## Key Features

- **Ultra-Low Latency Inference**: Real-time sliding receptive field history buffer (16 kHz, streaming causal inference) with typical response times under 20-50 ms depending on hardware.
- **Accelerated Execution Backends**:
  - **CPU**: Multi-core SIMD vectorization (AVX2 / FMA) with customizable thread affinity.
  - **WebGPU (Vulkan)**: Hardware-accelerated GPU inference support.
- **Lock-Free Audio Pipeline**: CPAL-based audio capture and playback backed by `rtrb` lock-free ring buffers. The inference worker thread runs asynchronously and never blocks audio I/O callbacks.
- **Dual Soft-Decay Noise Gates**: Independent intelligent noise gating on both microphone input and model output to eliminate background hum, static hiss, and breathing artifacts.
- **Real-Time Pitch Shifter**: Live formant/pitch transformation adjustable from `-12` to `+12` semitones.
- **Hot-Pluggable Models**: Automatically discovers all `.onnx` models placed in the `models/` directory with live refresh and dropdown selection in the GUI.
- **Dual Mode (GUI & Headless)**:
  - **Native GUI**: Visual RMS level meters, live Real-Time Factor (RTF) gauge, latency telemetry, gain sliders, and backend controls.
  - **Headless Mode (`--headless`)**: Zero-overhead command-line daemon mode ideal for background execution, low-spec systems, or systemd services.
- **Standalone Deployment Script**: Includes `make_deploy.sh` to produce a completely self-contained distribution archive (`hevoicechanger_deploy.tar.gz`) with bundled shared libraries and models.

---

## Architecture

```text
╔═════════════════════════════════════════════════════════════════════════════════╗
║                                AUDIO PIPELINE                                   ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║   ┌───────────────────────────────────────────┐                                 ║
║   │             Microphone (CPAL)             │                                 ║
║   └─────────────────────┬─────────────────────┘                                 ║
║                         │                                                       ║
║                         ▼                                                       ║
║   ┌───────────────────────────────────────────┐                                 ║
║   │         Linear Resampler (16 kHz)         │                                 ║
║   └─────────────────────┬─────────────────────┘                                 ║
║                         │                                                       ║
║                         ▼                                                       ║
║   ┌───────────────────────────────────────────┐                                 ║
║   │        Input Noise Gate & RMS Meter       │                                 ║
║   └─────────────────────┬─────────────────────┘                                 ║
║                         │                                                       ║
║                         ▼                                                       ║
║   ┌───────────────────────────────────────────┐                                 ║
║   │  Lock-Free In-RingBuffer (65,536 samples) │                                 ║
║   └─────────────────────┬─────────────────────┘                                 ║
║                         │                                                       ║
║                         ▼                                                       ║
║   ┌───────────────────────────────────────────────────────────┐                 ║
║   │                AI INFERENCE WORKER THREAD                 │                 ║
║   │                                                           │                 ║
║   │         [ Sliding Receptive History Context ]             │                 ║
║   │                           │                               │                 ║
║   │                           ▼                               │                 ║
║   │         [ ONNX Runtime Session (Level 3 Opt) ]            │                 ║
║   │                           │                               │                 ║
║   │                           ▼                               │                 ║
║   │         [ Pitch Shifter (-12 .. +12 Semitones) ]          │                 ║
║   └─────────────────────┬─────────────────────────────────────┘                 ║
║                         │                                                       ║
║                         ▼                                                       ║
║   ┌───────────────────────────────────────────┐                                 ║
║   │ Lock-Free Out-RingBuffer (65,536 samples) │                                 ║
║   └─────────────────────┬─────────────────────┘                                 ║
║                         │                                                       ║
║                         ▼                                                       ║
║   ┌───────────────────────────────────────────┐                                 ║
║   │       Output Noise Gate & RMS Meter       │                                 ║
║   └─────────────────────┬─────────────────────┘                                 ║
║                         │                                                       ║
║                         ▼                                                       ║
║   ┌───────────────────────────────────────────┐                                 ║
║   │    Speakers / Audio Output Device (CPAL)  │                                 ║
║   └───────────────────────────────────────────┘                                 ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
```

---

## Prerequisites

HEVoicechanger requires a Linux distribution with standard audio and development packages installed.

### Ubuntu / Debian / Pop!_OS / Linux Mint
```bash
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev libasound2-dev
```

### Arch Linux / Manjaro
```bash
sudo pacman -S --needed base-devel alsa-lib openssl pkgconf
```

### Fedora / RHEL
```bash
sudo dnf install -y gcc gcc-c++ alsa-lib-devel openssl-devel pkgconf-pkg-config
```

### Rust Toolchain
Ensure you have the latest stable Rust toolchain installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

---

## Installation & Building

### Prebuilt Binaries

Precompiled standalone bundles are available on the [Latest Release](https://github.com/ultrapg/hevoicechanger/releases/latest) page. You can download `hevoicechanger_deploy.tar.gz`, extract it, and run the binary directly with bundled dependencies and models without installing a Rust compiler:

```bash
tar -xzvf hevoicechanger_deploy.tar.gz
cd hevoicechanger_deploy
./hevoicechanger
```

### Building from Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ultrapg/hevoicechanger.git
   cd hevoicechanger
   ```

2. **Build the optimized release binary**:
   ```bash
   cargo build --release
   ```
   The compiled binary will be located at `target/release/hevoicechanger`.

3. **(Optional) Build Standalone Distribution Package**:
   To create a portable tarball with bundled shared libraries and models:
   ```bash
   chmod +x make_deploy.sh
   ./make_deploy.sh
   ```
   This generates `hevoicechanger_deploy.tar.gz`, which can be extracted and run on any compatible Linux desktop without compiling from source.

---

## Usage

### 1. Launching the GUI (Default)
Simply run the release executable:
```bash
cargo run --release
```
or run the binary directly:
```bash
./target/release/hevoicechanger
```

### 2. Command-Line Options
You can inspect all available flags with `--help`:
```bash
./target/release/hevoicechanger --help
```

```text
Usage: hevoicechanger [OPTIONS]

Options:
  -m, --model <MODEL>                  Path to initial ONNX voice model [default: models/female_voice_hfg_new.onnx]
  -i, --input-device <INPUT_DEVICE>    Specific microphone device name to use (defaults to system default)
  -o, --output-device <OUTPUT_DEVICE>  Specific speaker/output device name to use (defaults to system default)
  -g, --gain <GAIN>                    Initial mic input gain multiplier [default: 1.0]
  -v, --volume <VOLUME>                Initial voice output volume multiplier [default: 1.0]
      --headless                       Run in headless daemon mode (no GUI window)
      --list-devices                   List available audio devices and exit
  -h, --help                           Print help
  -V, --version                        Print version
```

### 3. Listing Available Audio Devices
To identify the exact system name of your microphone or speaker:
```bash
cargo run --release -- --list-devices
```

### 4. Running Headless (Background Daemon)
For headless environments, game servers, or low-overhead background operation:
```bash
cargo run --release -- --headless --model models/female_voice_hfg_new.onnx --gain 1.2 --volume 1.0
```

---

## Virtual Microphone Setup

To route your transformed voice into external applications such as **Discord**, **OBS Studio**, **Zoom**, or games, create a virtual audio loopback sink in **PipeWire** or **PulseAudio**.

### Method A: Quick Command Line (PulseAudio / PipeWire-Pulse)
Run these commands in your terminal to create a virtual microphone:
```bash
# 1. Create a null sink (an audio destination)
pactl load-module module-null-sink sink_name=VoiceChangerSink sink_properties=device.description="VoiceChangerSink"

# 2. Remap the monitor of that sink into a virtual recording microphone
pactl load-module module-remap-source master=VoiceChangerSink.monitor source_name=VirtualMic source_properties=device.description="Virtual_AI_Microphone"
```

Once created:
1. In **HEVoicechanger**, select `VoiceChangerSink` as your **Speaker / Output Device**.
2. In **Discord / OBS / Settings**, select `Virtual_AI_Microphone` as your **Input Device (Microphone)**.

To unload the virtual sink when done:
```bash
pactl unload-module module-remap-source
pactl unload-module module-null-sink
```

### Method B: Graphical Patchbay (PipeWire)
If using native PipeWire, you can use visual patchbay tools like **qpwgraph** or **Helvum**:
1. Connect `hevoicechanger` output ports to `Discord` / `OBS` input ports.
2. Monitor yourself by linking `hevoicechanger` to your headphones simultaneously.

---

## Voice Models

All voice models are stored in the `./models/` folder. The application automatically discovers all `.onnx` files in this directory on startup or whenever you click **"Refresh Models"** in the GUI.

### Included Models:
- `female_voice_hfg_new.onnx`: High-fidelity, 200-epoch causal model trained via teacher-student distillation on a clean 16 kHz paired dataset with HiFi-GAN vocoder topology.
- `female_voice_old_hq.onnx`: High-quality baseline reference model.
- `female_voice_old.onnx`: Standard baseline reference model.

### The Model Lore and Training Saga

The path to `female_voice_hfg_new.onnx` came after extensive experimentation with low-latency neural audio architectures.

#### The Early Struggles (The "Old" Models)
Initial testing relied on generic pre-trained weights (`female_voice_old.onnx` and `female_voice_old_hq.onnx`). While they proved that the real-time Rust engine worked, the vocal fidelity left much to be desired. Early custom training runs suffered from sample-rate mismatches: microphone audio recorded at 24 kHz or 48 kHz was fed into training pipelines expecting 16 kHz, causing severe spectral smearing, robotic distortion, and phase artifacts. Extended 500-epoch runs took hours of compute only to yield muffled results.

#### The Breakthrough: Teacher-Student Knowledge Distillation
To achieve crystalline real-time conversion with zero lookahead, a hybrid distillation pipeline was engineered:

1. **Neutral Source Audio Sourcing**:
   500 clean, diverse speech audio clips were extracted from the LibriSpeech (`dev-clean`) dataset and normalized to 16,000 Hz mono. These acted as the neutral linguistic reference.

2. **Teacher Model Training (RVC / Applio)**:
   A high-capacity Retrieval-based Voice Conversion (RVC v2) model was fine-tuned for 200 epochs on curated voice data using RMVPE (Robust Model for Vocal Pitch Estimation) and a HiFi-GAN vocoder. This model served as the offline "Teacher" possessing deep acoustic representation of the voice.

3. **Batch Offline Inference**:
   The RVC Teacher translated all 500 neutral LibriSpeech clips into target voice samples, producing a perfectly parallel corpus of spoken sentences.

4. **16 kHz Alignment and Pairing**:
   Because RVC operates at 40 kHz, all generated audio was resampled to exactly 16,000 Hz mono via FFmpeg. 500 strictly synchronized audio pairs (`sample_XXXX_original.wav` and `sample_XXXX_converted.wav`) were generated and divided into training (90%), validation (8%), and test/dev (2%) sets.

5. **Student Training (LLVC Streaming Causal CNN)**:
   A lightweight Low-Latency Voice Conversion (LLVC) student model was trained for 200 epochs (10,000 global gradient steps) on a cloud GPU. The architecture was tailored specifically for the Rust engine:
   - Receptive field context: `dec_buf_len = 100`
   - Chunk step size: `dec_chunk_size = 72`
   - Causal convolutions with zero future lookahead.

6. **TorchScript ONNX Compilation**:
   The trained generator checkpoint was serialized into ONNX (opset 17) with constant-folding and dynamic input/output axes (`audio_in` / `audio_out`).

The resulting 13.7 MB model delivers clean timbre matching, zero hallucination, and instantaneous inference speeds.

### Custom Model Requirements:
To add your own voice model:
1. Export your causal voice conversion network (e.g. LLVC / Streaming RVC) to **ONNX** (opset 17 recommended).
2. Ensure the model expects **16,000 Hz** mono input audio.
3. Place the `.onnx` file directly into the `models/` folder:
   ```bash
   cp /path/to/my_voice.onnx ./models/
   ```
4. Open the GUI and choose it from the **Voice Profile** dropdown.

---

## Performance Tuning

In the **Advanced Performance Settings** panel of the GUI:

| Setting | Default | Description |
|---|---|---|
| **CPU Cores (Intra)** | `4 - 8` | Controls ONNX Runtime parallel intra-op execution threads. Set this equal to your physical CPU core count for lowest latency. |
| **GPU Backend** | `CPU` | Switch between `CPU (AVX2/SIMD)` and `WebGPU (Vulkan)`. CPU is typically optimal for ultra-small chunk sizes; GPU excels with larger batches. |
| **Input Noise Gate** | `0.01` | Mutes background room ambience and mic hiss before feeding audio into the AI model, preventing artifact generation. |
| **Output Noise Gate** | `0.00` | Optional output gate to ensure silence when no speech is synthesized. |
| **Pitch Shift** | `0.0` | Shifts output pitch in semitones (+/- 12) in real-time. |

### Understanding the Real-Time Factor (RTF) Meter:
The GUI displays live **RTF** telemetry:
- **RTF < 0.6x (Green)**: Outstanding performance. The engine processes audio much faster than real-time; audio is smooth and lag-free.
- **0.6x - 1.0x (Yellow)**: Real-time capable. Audio will play correctly without drops.
- **RTF > 1.0x (Red)**: The CPU/GPU is unable to process audio in real-time. Increase chunk size or core allocation to avoid buffer under-runs.

---

## Project Structure

```text
hevoicechanger/
├── Cargo.toml               # Rust package manifest & dependencies
├── Cargo.lock               # Deterministic dependency lockfile
├── LICENSE                  # GNU General Public License v3.0
├── README.md                # Comprehensive documentation
├── make_deploy.sh           # Standalone packaging script
├── models/                  # ONNX AI voice models folder
│   ├── female_voice_hfg_new.onnx
│   ├── female_voice_old_hq.onnx
│   └── female_voice_old.onnx
└── src/
    ├── lib.rs               # Library root (audio, engine, ui modules)
    ├── main.rs              # CLI parser, worker orchestration, runtime setup
    ├── engine.rs            # ONNX Runtime session, causal history, inference
    ├── audio.rs             # CPAL stream management, lock-free ring buffers, noise gates
    └── ui.rs                # egui / eframe GUI, RMS meters, interactive controls
```

---

## Contributing

Contributions, bug reports, and pull requests are welcomed:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Open a Pull Request.

---

## License

GNU General Public License v3.0
([License](LICENSE))
