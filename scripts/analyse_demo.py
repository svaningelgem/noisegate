# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "scipy>=1.10",
#     "soundfile>=0.12",
#     "matplotlib>=3.7",
# ]
# ///
"""Measure and draw what the denoiser did to the demo sample.

    uv run scripts/analyse_demo.py

Expects scripts/make_demo_sample.py to have run, plus:

    noisegate --denoise samples/demo_raw.wav samples/demo_cleaned.wav

Prints a per-segment table and writes docs/demo-spectrogram.png.

Per segment rather than per file, because a whole-file average hides the only
question that matters: a denoiser can score well by removing a fan and nothing
else. The segments are ten seconds each and each holds a different problem, so
the neighbour row is directly comparable with the traffic row.
"""

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

SR = 48_000
SEG = 10
S = Path(__file__).resolve().parent.parent / "samples"
DOCS = Path(__file__).resolve().parent.parent / "docs"

SEGMENTS = [
    "clean",
    "office ventilation",
    "cafeteria babble",
    "street traffic",
    "neighbour through wall",
    "everything at once",
]


def rd(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return d.astype(np.float64) / 32768


def level(x: np.ndarray, pct: float, win: int = 4800) -> float:
    n = len(x) // win
    if n == 0:
        return -200.0
    r = [np.sqrt(np.mean(x[i * win : (i + 1) * win] ** 2)) for i in range(n)]
    return 20 * np.log10(max(np.percentile(r, pct), 1e-10))


def main() -> None:
    raw = rd(S / "demo_raw.wav")
    have = {}
    for tag, name in (
        ("DeepFilterNet3", "demo_cleaned.wav"),
        ("RNNoise", "demo_rnnoise.wav"),
    ):
        p = S / name
        if p.exists():
            have[tag] = rd(p)
    if not have:
        sys.exit("run --denoise first; see the module docstring")

    # make_demo_sample lays each segment out as 6 s of speech then 4 s of
    # pause. Measure background in the pause and voice in the speech, rather
    # than taking a percentile over the mixture and hoping it lands in the
    # right place — with a 60/40 duty cycle the median window is speech, so a
    # p50 "background" figure is really measuring the voice.
    seg = SEG * SR
    speech_on = int(6.0 * SR)

    def window(i: int, speech: bool) -> slice:
        start = i * seg
        return (
            slice(start, start + speech_on)
            if speech
            else slice(start + speech_on, start + seg)
        )

    header = f"{'segment':<24}" + "".join(f"{t:>16}" for t in have)

    # The measurement that matters: how much closer to the clean voice the
    # output is than the input was, *while the person is speaking*.
    #
    # Measuring background in the pauses instead rewards gating — anything
    # that outputs silence when nobody is talking scores perfectly, and by
    # that metric RNNoise beats DeepFilterNet3 on every row. It is the wrong
    # question. Removing a competing voice is only hard while the near voice
    # is also present, because then you cannot simply mute.
    #
    # Possible because this mixture is synthetic: demo_clean_reference.wav is
    # the near voice alone, so the error signal is exactly the part that
    # should not be there.
    clean_ref = rd(S / "demo_clean_reference.wav")

    def snr_db(signal: np.ndarray, estimate: np.ndarray) -> float:
        n = min(len(signal), len(estimate))
        err = estimate[:n] - signal[:n]
        return 10 * np.log10(
            (np.sum(signal[:n] ** 2) + 1e-12) / (np.sum(err**2) + 1e-12)
        )

    def align(x: np.ndarray, ref: np.ndarray, span: int = 4000) -> np.ndarray:
        """Undo the denoiser's latency before comparing sample by sample."""
        n = min(len(x), len(ref))
        a, b = x[:n] - x[:n].mean(), ref[:n] - ref[:n].mean()
        c = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)))
        c = np.concatenate([c[-span:], c[:span]])
        lag = int(np.argmax(np.abs(c))) - span
        if lag > 0:
            return np.concatenate([x[lag:], np.zeros(lag)])
        if lag < 0:
            return np.concatenate([np.zeros(-lag), x[:lag]])
        return x

    aligned = {t: align(x, raw) for t, x in have.items()}

    print("\nSNR improvement while speaking (dB — higher is better)\n")
    print(header)
    print("-" * len(header))
    for i, label in enumerate(SEGMENTS):
        sl = window(i, speech=True)
        before = snr_db(clean_ref[sl], raw[sl])
        row = f"{label:<24}"
        for x in aligned.values():
            # The first segment has no noise, so "improvement" is a ratio of
            # two near-zero error signals and means nothing. Say so.
            cell = (
                "—" if before > 40 else f"{snr_db(clean_ref[sl], x[sl]) - before:+.1f}"
            )
            row += f"{cell:>16}"
        print(row)

    print("\nNear voice kept, measured while speaking (dB — closer to 0 is better)\n")
    print(header)
    print("-" * len(header))
    for i, label in enumerate(SEGMENTS):
        sl = window(i, speech=True)
        base = level(raw[sl], 95)
        row = f"{label:<24}"
        for x in have.values():
            row += f"{level(x[sl], 95) - base:>+16.1f}"
        print(row)

    # Spectrograms: the picture makes the neighbour segment obvious in a way
    # the numbers do not.
    best = "DeepFilterNet3" if "DeepFilterNet3" in have else next(iter(have))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, (title, x) in zip(axes, [("Input", raw), (f"After {best}", have[best])]):
        f, t, sxx = spectrogram(x, SR, nperseg=1024, noverlap=768)
        ax.pcolormesh(
            t, f / 1000, 10 * np.log10(sxx + 1e-12), vmin=-120, vmax=-40, shading="auto"
        )
        ax.set_ylabel("kHz")
        ax.set_ylim(0, 8)
        ax.set_title(title, loc="left", fontsize=11)
        for i in range(1, 6):
            ax.axvline(i * SEG, color="w", lw=0.8, alpha=0.5)
        for i, label in enumerate(SEGMENTS):
            ax.text(i * SEG + 0.3, 7.4, label, color="w", fontsize=8, va="top")
    axes[1].set_xlabel("seconds")
    fig.tight_layout()
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "demo-spectrogram.png"
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")

    # The WAVs are 5.8 MB each and reproducible from the script, so the repo
    # carries MP3s instead — small enough to click from the README. The
    # measurements above are taken on the WAVs, never on these.
    if shutil.which("ffmpeg") is None:
        print("no ffmpeg on PATH; skipping the MP3s the README links to")
        return
    for name in ("demo_raw", "demo_cleaned", "demo_rnnoise"):
        src = S / f"{name}.wav"
        if not src.exists():
            continue
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-b:a",
                "96k",
                str(S / f"{name}.mp3"),
            ],
            check=True,
        )
        print(f"wrote {S / f'{name}.mp3'}")


if __name__ == "__main__":
    main()
