# Training a model in WSL2

Notes for rebuilding the DeepFilterNet training and reference environment
without rediscovering the workarounds. Everything here was learned the hard
way; the non-obvious bits are called out.

## Why we train our own

Not quality — the DeepFilterNet3 checkpoint is good, and NoiseGate's ONNX
loader has been verified to reproduce the reference implementation almost
exactly. The reasons are **ownership** and **availability**: the only export
matching our loader came from a third-party repository that could disappear,
and the licence covering the published weights is unstated.

## Environment

Source [`scripts/wsl_env.sh`](../scripts/wsl_env.sh) before anything else.

| Trap | Symptom | Fix |
|---|---|---|
| WSL inherits the **Windows PATH** | `bash: syntax error near unexpected token '('` | rebuild `PATH` rather than appending to it |
| miniforge's `sqlite3` needs miniforge's `libstdc++` | imports fail as a misleading *"Failed to import monkeytype"* | `LD_LIBRARY_PATH=/opt/miniforge3/lib` |
| `$VAR` and `$(...)` are eaten by `wsl.exe -d Ubuntu -- bash -c '...'` | silent missing-argument errors | put anything with shell variables in a `.sh` file and run that |
| stdout is lossy over the same bridge | truncated output on long/fast runs | redirect to a file and read it back |
| HDF5 locking fails on `drvfs` | dataset creation or training dies | keep datasets on the WSL filesystem (`/root/...`), never `/mnt/...` |
| `requests`/`urllib3` in that Python cannot TLS to **GitHub** | `ConnectionError: RemoteDisconnected` | `curl` works fine, and so does `pip` against PyPI — this is GitHub-specific, **not** a WSL networking problem |

## Reference implementation (for validating a port)

```bash
pip install deepfilternet 'torch==2.1.2' 'torchaudio==2.1.2' \
  --index-url https://download.pytorch.org/whl/cpu
```

`torchaudio` must be pinned: 2.2 moved `torchaudio.backend.common.AudioMetaData`
and 2.9 removed it. Place the checkpoint at
`/root/.cache/DeepFilterNet/DeepFilterNet3/` by hand — the automatic download
hits the GitHub TLS problem above.

`df.enhance.df_features()` returns exactly the tensors a host must compute
(`spec`, `feat_erb`, `feat_spec`), which is how
[`crates/dsp/src/dfn_frontend.rs`](../crates/dsp/src/dfn_frontend.rs) was
verified: on real audio all three match to floating-point noise.

## The data loader

```bash
python3 -m maturin build --release -m pyDF-data/Cargo.toml \
  --features hdf5-static --compatibility linux -o /tmp/wheels
```

`--compatibility linux` (or installing `patchelf`) is **required**: without it
the manylinux repair step fails and maturin writes a 22-byte empty wheel that
installs without complaint and fails at import. `--features hdf5-static` does
not actually static-link; `libhdf5_serial.so.103` still has to be present.

`df/train.py` also imports `icecream`, which is not pulled in as a dependency.

## AMD GPU under WSL2

ROCm **does** work in WSL2 — an RX 7900 XTX reaches 14.3 TFLOP/s fp32,
including GRU forward and backward, which is what matters for a GRU-heavy
model.

Install `torch` for ROCm into its own venv, then apply the one fix that makes
it see the card:

```bash
cp -f /opt/rocm/lib/libhsa-runtime64.so.1.14.0 \
      "$VENV/lib/python3.11/site-packages/torch/lib/libhsa-runtime64.so"
```

PyTorch ships a generic HSA runtime that talks to `/dev/kfd`; WSL2 has only
`/dev/dxg`, so ROCm's own runtime has to replace it. `rocm-smi` failing with
*"amdgpu not found in modules"* is expected on WSL and harmless — that is the
monitoring tool, not compute.

Newer `torchaudio` in the ROCm venv needs `torchaudio.save` bound to
soundfile at `df.io` import time, because `df/train.py` calls
`torchaudio.save` directly as well. Pinned memory must also be disabled —
it fails with `HIP error: out of memory` on WSL regardless of free RAM.

Measured, batch 64, 5 s segments, warm cache:

| | epoch | throughput | 100 epochs on VCTK |
|---|---|---|---|
| RX 7900 XTX | 9 s | ~430 audio-s/s | ~10 h |
| 7950X3D, 32 threads | 33 s | ~116 audio-s/s | ~38 h |

The GPU is only ~3.7× the CPU here: the model is small and the run is partly
dataloader-bound. CPU-only training is viable.

## Datasets

Clean licences and 48 kHz throughout, 33 GB downloaded, 23.7 GB of HDF5.

| Corpus | Size | Licence | Rate |
|---|---|---|---|
| VCTK 0.92 | 10.94 GB | CC BY 4.0 | 48 kHz |
| DEMAND (17 environments) | 4.91 GB | CC BY-SA 3.0 | 48 kHz |
| FSD50K dev | 17.15 GB | per-clip CC-BY / CC0 | 44.1 kHz → resampled |
| RIRs | — | synthesised, pyroomacoustics (BSD) | 48 kHz |

Deliberately **not** used:

- **MUSAN** and **OpenSLR28** RIRs — 16 kHz only, useless for a 48 kHz model.
- **DNS-Challenge noise** — Audioset/YouTube-derived, which would reintroduce
  exactly the provenance problem we are escaping.
- `assets/clean_freesound_33711.wav` — **CC BY-NC**, fine for plumbing tests,
  must never reach a shipped model.

The Edinburgh DataShare URL for VCTK returns a *wrapper* zip containing the
real `VCTK-Corpus-0.92.zip`; extracting it looks like a no-op because the inner
archive has the same name.

## Training for competing speech

The point of training our own model is not to reproduce DeepFilterNet3 but to
be *deliberately* good at what it is accidentally good at.

Measured on real recordings, DFN3 removes ~19 dB of background speech where
newer and larger models remove ~5 dB, because speech-enhancement models are
trained to *preserve* speech and a neighbour is speech. DFN3 appears to
discriminate on proximity and reverberation instead. Training on
speech-plus-noise alone would very likely lose that property.

So interfering speech is a first-class noise class:

- **Speaker-disjoint pools** — interferers are never targets, so the model
  cannot shortcut on voice identity.
- **Two separate RIR families.** Near (RT60 0.15–0.35 s, 0.3–1.2 m) for the
  loader's own augmentation; far (RT60 0.40–1.00 s, 3–8 m, plus a 900 Hz–3 kHz
  low-pass for the wall) used *only* to build interferers offline. Keeping them
  apart matters — far RIRs in the loader's pool would make the near target
  itself heavily reverberant.
- **Mixture rate solved, not guessed.** libDF combines `uniform(2,6)` noise
  clips per mixture, so the interferer share `f` of the noise pool follows
  `1-(1-f)^3.5 = 1/3` → `f = 0.109`. Verified at 0.332 across all splits.
- `p_reverb = 0.2`, above DFN3's 0.1, because libDF makes the *input* more
  reverberant than the *target*; that dereverberation pressure is the likely
  mechanism behind the proximity discrimination, so it is leaned on
  deliberately.

Use the **official DFN3 recipe**, not the generated defaults — the defaults
leave the multi-resolution spectral loss at zero and shrink `conv_ch` from 64
to 16, training a different and much weaker network.

### A bug in libDF's own interferer path

`libDF/src/dataset.rs:1327`, inside the interferer loop:

```rust
self.reverb.transform_single(&mut speech, rir)?;   // `speech` is the TARGET
```

It reverberates the training **target**, once per interferer and therefore
compounded, instead of `sample`, the interferer. The target becomes reverberant
while the interferer stays dry — exactly backwards. Set `p_interfer_sp = 0.0`
and build interference yourself.

## Judging the result

Use the project's percentile measure (`noisegate --denoise`), against the
fixtures in `samples/`: p25–p75 for background, p95 for the near voice, with
`samples/kid_dfn.wav` as the bar.

Be careful with it. **The measure rewards a model that simply outputs less** —
a p25 of −50 dB is near-silence in the quiet parts, which looks like a triumph
and may not be one. Pair it with listening and a speech-quality metric before
concluding anything.
