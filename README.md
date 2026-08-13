# NoiseGate

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows%2010%2F11-0078d4.svg)](#install)
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange.svg)](https://www.rust-lang.org/)
[![Model: RNNoise + ONNX](https://img.shields.io/badge/model-RNNoise%20%2B%20ONNX-purple.svg)](#noise-suppression-model)
[![Latest release](https://img.shields.io/github/v/release/Yashsomalkar/noisegate?include_prereleases&label=download)](https://github.com/Yashsomalkar/noisegate/releases)

**Keeps your voice on the call and leaves the rest of the room behind.** Real-time microphone processing for Windows that removes the people talking *around* you — a child in the next room, a partner on a phone call, an office two desks away — not just fans and keyboard clatter. Pure-Rust, WASAPI low-latency capture/render, system-tray UI. Cleaned audio goes into a virtual cable so Zoom, Teams, Discord, OBS or a browser call just sees a quiet microphone.

> Status: pre-alpha. Builds on Windows 10/11 with the MSVC toolchain.

## Why this and not the others

Ordinary noise suppression solves the easy half of the problem. Fans, hum and typing are *stationary* noise, and everything removes those — Windows does some of it for free. What almost nothing removes is **other people's voices**, because every speech-enhancement model is trained to *preserve* speech, and your neighbour is speech.

That turns out to be measurable. Same 60-second recording — one person talking at the microphone, a child talking loudly across the room — through four backends. The middle of the level distribution is where background speech lives; the top is the near voice:

| backend | background speech | your voice |
|---|---|---|
| **DeepFilterNet3** | **−19.3 dB** | −0.1 dB |
| GTCRN | −11.1 dB | −0.9 dB |
| RNNoise | −6.5 dB | −0.4 dB |
| MossFormer2 (48 kHz, newer, larger) | −5.5 dB | −0.2 dB |

The newest, largest model came last: it is very good at enhancing speech, so it faithfully preserves the voice you wanted gone. DeepFilterNet3 is the outlier — it discriminates on proximity and reverberation, which is exactly the cue that separates you from the room.

So NoiseGate ships DeepFilterNet3 as its default and treats "surrounding conversation" as the problem worth solving. It's free, open source, runs on any CPU — no Krisp subscription, no RTX card, no Linux.

*(Reproduce it yourself: `noisegate --denoise noisy.wav clean.wav`, which reports these numbers. RNNoise stays available as a low-CPU fallback at ~50× less compute.)*

## Stack

- **Audio I/O:** WASAPI shared low-latency, event-driven, MMCSS Pro Audio scheduling.
- **Routing:** VB-Cable (free virtual audio cable, [vb-audio.com](https://vb-audio.com/Cable/)).
- **DSP model:** **DeepFilterNet3** by default via ONNX Runtime. **RNNoise** (the pure-Rust [`nnnoiseless`](https://github.com/jneem/nnnoiseless) port) as a zero-dependency, low-CPU fallback.
- **UI:** `tray-icon` + `winit` event loop.
- **Lang:** Rust 2021, single static binary, ~10 MB stripped (RNNoise default), ~25 MB with ONNX runtime.

## Architecture

```
physical mic ─► WASAPI capture ─► ring A ─► DSP thread (denoiser) ─► ring B ─► WASAPI render ─► VB-Cable Input
                                                                                                        │
                                                               other apps choose "CABLE Output" as mic ◄┘
```

Three dedicated MMCSS-priority threads. Lock-free SPSC ring buffers (8 frames ≈ 80 ms headroom). 480-sample (10 ms) frames end-to-end — the native frame size for both RNNoise and DeepFilterNet, so no reblocking inside the DSP path.

## Noise suppression model

Two backends, both real-time, switchable from the tray while running:

| Backend | Default? | Background speech | CPU | Needs |
|---|---|---|---|---|
| **DeepFilterNet3** (ONNX) | ✅ | −19 dB | ~15% of one core (RTF 0.15) | `onnxruntime.dll` (bundled) + a model file |
| **RNNoise** (`nnnoiseless`) | fallback | −6.5 dB | ~0.3% of one core (RTF 0.003) | nothing — model embedded |

RNNoise is still worth having: it needs no download at all, runs on any architecture Rust targets, and costs ~50× less compute, which matters on an old laptop or on battery. It's genuinely good at fans, hum and hiss. It just can't touch a voice in the next room, because it was trained not to.

NoiseGate falls back to RNNoise on its own if no ONNX model is present, so it always does something useful.

## Install

### Option A — Run the installer (recommended)

Grab `NoiseGate-x.y.z-setup.exe` from the **[Releases page](https://github.com/Yashsomalkar/noisegate/releases)** and run it. It installs per-user, so there's no administrator prompt, and it offers to start NoiseGate when you sign in.

The installer includes the ONNX Runtime it needs. On first run NoiseGate will tell you if a virtual audio cable is missing and offer to open the download page — see [Install a virtual audio cable](#install-a-virtual-audio-cable).

Building the installer yourself:

```powershell
cargo build --release --features onnx
ISCC.exe installer\noisegate.iss     # Inno Setup 6
```

### Option B — Build from source

You need three things on your Windows 10/11 machine:

| # | Component | Download | Notes |
|---|---|---|---|
| 1 | **Rust toolchain** | [rustup-init.exe](https://win.rustup.rs/x86_64) — or visit [rustup.rs](https://rustup.rs/) | Pick the default (`stable`, `x86_64-pc-windows-msvc`). |
| 2 | **MSVC C++ build tools** | [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/) | In the installer, check **"Desktop development with C++"** (gives you `link.exe` + the Windows SDK). |
| 3 | **VB-Cable virtual audio driver** | [vb-audio.com/Cable](https://vb-audio.com/Cable/) ([direct ZIP](https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip)) | Extract, run `VBCABLE_Setup_x64.exe` as Administrator, reboot. |
| 4 | **Git** *(probably already installed)* | [git-scm.com/download/win](https://git-scm.com/download/win) | Used to clone the repo. |

Then in PowerShell:

```powershell
git clone https://github.com/Yashsomalkar/noisegate.git
cd noisegate
cargo build --release
.\target\release\noisegate.exe
```

First build pulls ~300 MB of crates and takes 5-10 minutes. Subsequent builds are seconds.

A tray icon appears in the system tray (bottom-right corner of your taskbar) — no console window; NoiseGate is a GUI-subsystem binary. Right-click the icon for the menu:

| Item | What it does |
|---|---|
| **Enabled** | Toggles denoising. Unchecked = bypass, audio still flows. **Left-clicking the tray icon** does the same thing, so the common action needs one click. |
| **Microphone ▸** | Pick the input device. **Windows default** is selected out of the box and follows whatever Windows is using. Switching restarts the audio pipeline in place and is remembered. |
| **Start with Windows** | Adds/removes a `NoiseGate` value under `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`. Per-user, no elevation. There is deliberately no config setting for this — the registry is the state, so the checkbox can never disagree with what Windows will actually do. |
| **Open log folder** | Opens `%APPDATA%\NoiseGate\logs`. |
| **Quit NoiseGate** | Exits. |

The icon itself reports state at a glance:

| Icon | Meaning |
|---|---|
| Blue-green disc | Denoising is **on**. |
| Orange disc | **Bypassed** — audio still flows, unprocessed. |
| ⚠ badge on the corner | **Audio isn't running at all** (no cable, no mic, device failed). |

Hovering shows the active backend, ON/BYPASS, and a CPU meter.

The command-line flags still work when you run it from a terminal — NoiseGate attaches to the calling console (and leaves redirection to a file or pipe alone). If audio can't start, you get a dialog explaining why and the tray stays up, so you can fix the problem and pick a microphone without relaunching.

### Install a virtual audio cable

NoiseGate is a normal user-mode application. Windows lets it *read* a microphone and *write* to an output, but there is no user-mode API to publish a new microphone that other apps can select — that needs a kernel-mode audio driver. So a virtual cable supplies the "microphone" half:

```
real mic ─► NoiseGate ─► CABLE Input  ║  CABLE Output ─► Zoom / Teams / Discord
                                      ╚══ the cable ══╝
```

**VB-Cable** is the usual choice: free ([donationware](https://vb-audio.com/Cable/)), properly signed, five minutes to install.

1. Download the driver pack from **<https://vb-audio.com/Cable/>**.
2. Extract the zip anywhere.
3. Right-click **`VBCABLE_Setup_x64.exe`** → **Run as administrator**, then click **Install Driver**.
4. Reboot if it asks. The endpoints often appear immediately.

Confirm it worked:

```powershell
.\noisegate.exe --list-devices
```

You should see `CABLE Output (VB-Audio Virtual Cable)` under inputs and `CABLE Input (VB-Audio Virtual Cable)` under outputs, the latter tagged `[VB-Cable]`.

> **Two things the installer changes that catch people out.**
>
> It makes the cable the **default device for both playback and recording**. That means your speakers/headphones go silent — system audio is now going into the cable. Open **Sound settings** and set your real output back as default.
>
> It also becomes the default **microphone**, which for NoiseGate specifically means "use the Windows default mic" would capture from the cable we render into — the cable feeding itself. NoiseGate detects that and refuses to start rather than looping; pick your real microphone from the tray menu.

Any cable works, not just VB-Audio's — NoiseGate also recognises VoiceMeeter and Virtual Audio Cable endpoints. For anything else, set `output_device_id` in `config.toml` to its id from `--list-devices`.

Open-source virtual audio drivers exist ([Virtual-Audio-Driver](https://github.com/VirtualDrivers/Virtual-Audio-Driver), [AudioMirror](https://github.com/JannesP/AudioMirror), and others) and are perfectly good code, but none ship a production-signed binary — they need Windows test-signing mode enabled, which disables a system-wide security boundary. That's why the recommendation is a signed third-party cable rather than one we bundle. NoiseGate can't ship its own driver either: Microsoft [attestation signing](https://learn.microsoft.com/en-us/windows-hardware/drivers/dashboard/code-signing-attestation) requires an EV certificate (~$280–580/year) held by a registered company.

### Set up audio routing

Once NoiseGate is running, point your communication apps at VB-Cable instead of your real microphone:

1. Open **Windows Sound Settings** → **Sound** → **Input**.
2. Verify **CABLE Output (VB-Audio Virtual Cable)** is in the device list. If not, the VB-Cable install didn't finish — re-run step 3 above.
3. In Zoom / Teams / Discord / OBS / browser settings, choose **CABLE Output** as the microphone.

NoiseGate captures from your real mic, denoises, and writes the cleaned signal into **CABLE Input**. Apps that listen to **CABLE Output** then receive your noise-free voice.

> If NoiseGate can't find the CABLE Input endpoint it **stops with an error** instead of picking another output. That's deliberate: falling back to the default render device would play your microphone out of whatever speakers, headset or meeting-room display happens to be default. Install VB-Cable, or set `output_device_id` explicitly.

> **Sanity test**: open the Windows **Voice Recorder** app, set its mic to **CABLE Output**, record 10 seconds with a fan / typing in the background. Toggle NoiseGate's tray Enable off and re-record. The difference should be obvious.

## Picking a specific microphone

By default NoiseGate captures from your **system default mic**. If that's a Bluetooth headset, Windows will switch the headset into **HFP/Hands-Free mode** as soon as we open the mic — that's a Windows-wide behavior, not a NoiseGate bug, and it sounds awful (16 kHz mono, glitchy). Pick your USB or built-in mic instead.

Easiest way is the tray: right-click the icon → **Microphone** → pick one. The choice is saved and applied immediately. From the command line:

```powershell
# See what's available:
.\noisegate.exe --list-devices

# Run with a specific mic (substring match on the friendly name):
.\noisegate.exe --mic "USB"
.\noisegate.exe --mic "Yeti"
.\noisegate.exe --mic "Realtek"
```

For a permanent choice, copy the device `id` from `--list-devices` into `%APPDATA%\NoiseGate\config.toml`:

```toml
input_device_id = "{0.0.1.00000000}.{...your-id...}"
```

## Configuration

`%APPDATA%\NoiseGate\config.toml` — created on first run:

```toml
input_device = ""            # microphone by name; empty = Windows default
output_device = ""           # by name; empty = auto-detect a virtual cable
enabled = true
attenuation_db = 100.0       # 6.0 = subtle, 100.0 = max. ONNX models only;
                             # RNNoise has no equivalent knob.
model_path = ""              # ONNX model to use instead of RNNoise
```

Logs at `%APPDATA%\NoiseGate\logs\noisegate.log` (rolled over at 5 MB). Tune verbosity with `RUST_LOG=noisegate=debug`.

## Trying it without VB-Cable

Two offline modes let you hear what the denoiser does before setting up any routing:

```powershell
# Record 10 seconds from your mic (48 kHz mono WAV):
.\noisegate.exe --record 10 raw.wav

# Run it through the denoiser:
.\noisegate.exe --denoise raw.wav clean.wav

# ...or with an ONNX model, capping suppression at 12 dB:
.\noisegate.exe --denoise raw.wav clean.wav --model model.onnx --atten 12
```

`--denoise` reports the noise floor and the speech level separately, because whole-file loudness barely moves even when all the noise is gone:

```
noise_floor="-58.1 -> -106.0 dB (-47.9)" speech="-21.9 -> -23.4 dB (-1.6)" rtf="0.003"
```

## Cargo features

| Flag | Default | What it does |
|---|---|---|
| `rnnoise` | ✅ on | RNNoise backend via `nnnoiseless`. Pure-Rust, model embedded, no extra runtime deps. |
| `onnx` | off | Adds ONNX Runtime as a dependency so you can load a streaming noise-suppression ONNX model (e.g. DFN3). Needs `onnxruntime.dll` **next to `noisegate.exe`**, or `ORT_DYLIB_PATH` pointing at it — we don't let the OS search PATH and the working directory for it. |

The loader expects a **streaming** export: one frame of raw audio in, one frame out, with the recurrent state handed back on every call (`input_frame` / `states` / `atten_lim_db` in; enhanced audio + new states out). Tensor names are matched loosely and the state width is read from the model, so most DeepFilterNet3 stream exports load with no code change. Encoder/decoder-split exports, and models wanting a spectrum rather than raw audio, need their own front-end and won't load.

Build with the ONNX backend in addition to RNNoise:

```powershell
cargo build --release -p noisegate -F onnx
```

To get DeepFilterNet3 quality:
1. Build with `-F onnx`.
2. Download the DFN3 ONNX export from Hugging Face: <https://huggingface.co/Rikorose/DeepFilterNet3>.
3. Drop `onnxruntime.dll` next to `noisegate.exe` (download from <https://github.com/microsoft/onnxruntime/releases> — pick the `win-x64` zip). It must sit beside the exe: NoiseGate loads it by absolute path rather than searching PATH.
4. Point `model_path` in `config.toml` at the ONNX file.

Confirm it loads and actually suppresses before touching any audio routing:

```powershell
.\noisegate.exe --denoise noisy.wav clean.wav --model .\model.onnx
```

Pick an ONNX Runtime whose API version matches the `ort` crate this is pinned to — `ort` 2.0.0-rc.10 wants ONNX Runtime **1.22.x**.

## License

Code: dual MIT / Apache-2.0 — your choice.

The bundled RNNoise model (via `nnnoiseless`) is BSD-licensed — fine for any use including commercial. If you opt into the ONNX backend with DeepFilterNet3 weights, those are research / non-commercial; for commercial use, retrain on your own data or pick a different ONNX model.

## Cost

$0 for personal use. Every component is free. No driver-signing certs required (we don't ship a driver — VB-Cable does).

## Contributing

Every pull request runs format, lint, build, test and a coverage ratchet on
Windows — see [`docs/testing.md`](docs/testing.md). The coverage threshold is a
floor that may be raised but never lowered; the long-term goal is 100%, and the
doc is honest about which parts of the codebase can't get there without a
hardware abstraction that doesn't exist yet.

```powershell
cargo fmt --all -- --check
cargo clippy --all-targets --all-features
cargo test --all-features
```

## Not included (yet)

- macOS / Linux backends (the audio-io crate is Windows-only; the rest is portable).
- Far-end denoising (cleaning the audio you *hear* from the call — only your mic is cleaned).
- Acoustic echo cancellation (combine with an AEC frontend if needed).
- Auto-update.

## Credits

Built on top of excellent open-source work:

- **[RNNoise](https://gitlab.xiph.org/xiph/rnnoise)** by Jean-Marc Valin / Xiph.Org — the recurrent-network noise-suppression model that powers the default backend.
- **[`nnnoiseless`](https://github.com/jneem/nnnoiseless)** by jneem — pure-Rust port of RNNoise; this is the actual crate doing the work.
- **[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)** by Hendrik Schröter et al. — the modern speech-enhancement model that the optional ONNX backend can run.
- **[ONNX Runtime](https://onnxruntime.ai/)** + the **[`ort`](https://github.com/pykeio/ort)** Rust bindings — power the optional `onnx` backend.
- **[VB-Cable](https://vb-audio.com/Cable/)** by VB-Audio — the free virtual audio driver every Windows audio-routing app depends on.
- **[`windows`](https://github.com/microsoft/windows-rs)** crate by Microsoft — official Win32 bindings for Rust (WASAPI, MMCSS, COM).
- **[`tray-icon`](https://github.com/tauri-apps/tray-icon)** + **[`winit`](https://github.com/rust-windowing/winit)** — system-tray UI and event loop.
- **[`ringbuf`](https://github.com/agerasev/ringbuf)** — the lock-free SPSC buffers that connect the audio threads.
- Inspired by **[NoiseTorch](https://github.com/noisetorch/NoiseTorch)** (Linux-only equivalent that uses RNNoise).

## Issues & discussion

- Bug reports / feature requests: [open an issue](https://github.com/Yashsomalkar/noisegate/issues).
- Questions, ideas, "is X possible?" — [Discussions](https://github.com/Yashsomalkar/noisegate/discussions).
