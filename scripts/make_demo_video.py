# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "matplotlib>=3.7",
# ]
# ///
"""Turn the demo sample into an MP4 a README can actually play.

    uv run scripts/make_demo_video.py             # the whole 60 s tour, twice
    uv run scripts/make_demo_video.py --segment   # "everything at once", ~20 s

GitHub strips <audio> from markdown, so a playable clip has to be a video.

Before and after sit one above the other on the same timeline, scrolling
together past a fixed playhead. Whichever one you are hearing is drawn in
orange and the other is dimmed, so the difference is visible at the instant
it is audible rather than remembered from twenty seconds ago. The clip runs
the segment twice: once hearing the input, once hearing the output.

It is drawn as two wide stills and scrolled by ffmpeg's crop filter, so
nothing is rendered frame by frame; encoding takes seconds and the file
stays a few MB.

Also writes docs/demo-waveform.png: both halves stacked, static, for the
README itself — that renders inline without anyone having to upload anything.

Needs ffmpeg on PATH, and scripts/make_demo_sample.py plus a --denoise run
to have produced samples/demo_raw.wav and samples/demo_cleaned.wav.

The video still has to be uploaded by hand: GitHub only renders <video> from
its own user-attachments URLs, which you get by dropping the file into any
comment box. A path inside the repo is stripped from the markdown entirely.
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

SR = 48_000
SEG = 10

# Pixels per second. High enough that ~5 s is on screen at a time: at 100 the
# window held 13 s and the speech ran together into an unreadable block.
PPS = 250
VW = 1280
VH_WAVE = 460  # the two stacked panels together
VH_MAP = 96  # the minimap under them
# Peaks are brief and the speech sits far below them, so drawing to the true
# peak leaves a thin line in an empty box. Clipping the display here fills the
# frame. Every panel uses the same value, so the comparison stays honest.
CEILING = 0.35

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "samples"

SEGMENTS = [
    "clean",
    "office ventilation",
    "cafeteria babble",
    "street traffic",
    "neighbour through wall",
    "everything at once",
]

INK = "#0d1117"
GRID = "#30363d"
MUTED = "#8b949e"
ACTIVE = "#f0883e"  # what you are hearing
IDLE = "#39414d"  # the other one, for comparison
TEAL = "#2dd4bf"


def rd(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return d.astype(np.float32) / 32768


def envelope(x: np.ndarray, buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Peak envelope, which is what a waveform is at this zoom.

    Plotting millions of samples gives a solid block; the min and max of each
    bucket keeps the shape of the speech and the floor between phrases.
    """
    buckets = max(1, min(buckets, len(x)))
    n = (len(x) // buckets) * buckets
    b = x[:n].reshape(buckets, -1)
    return b.min(axis=1), b.max(axis=1)


def draw(ax, x: np.ndarray, t0: float, colour: str, buckets: int) -> None:
    lo, hi = envelope(x, buckets)
    t = np.linspace(t0, t0 + len(x) / SR, len(lo))
    ax.fill_between(t, lo, hi, color=colour, linewidth=0)


def style(ax, span: tuple[float, float]) -> None:
    ax.set_facecolor(INK)
    ax.set_xlim(*span)
    ax.set_ylim(-CEILING, CEILING)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def pass_strip(
    raw: np.ndarray, clean: np.ndarray, active: int, labels: list[str], out: Path
) -> Path:
    """Both takes on one timeline, with `active` (0=before, 1=after) lit up.

    Padded by half a window at each end. Without it the crop has to be clamped
    so it does not run off the image, and the waveform then sits still for the
    first and last few seconds while the audio plays on.
    """
    span = len(raw) / SR
    pad = (VW / 2) / PPS
    fig, axes = plt.subplots(
        2, 1, figsize=((span + 2 * pad) * PPS / 100, VH_WAVE / 100), dpi=100
    )
    fig.patch.set_facecolor(INK)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.07)

    buckets = int(span * PPS)
    for i, (ax, x, tag) in enumerate(
        ((axes[0], raw, "BEFORE"), (axes[1], clean, "AFTER"))
    ):
        colour = ACTIVE if i == active else IDLE
        draw(ax, x, 0, colour, buckets)
        style(ax, (-pad, span + pad))
        for k in range(1, len(labels)):
            ax.axvline(k * SEG, color=GRID, lw=1)
        # Repeated along the strip rather than written once at the start: only
        # ~5 s is on screen, so a label at t=0 is gone by the third second.
        # Never past `span` -- a label out in the trailing pad reads as the
        # clip starting over after the audio has already stopped.
        step = 2.5
        for k in range(int(span / step) + 1):
            t = k * step + 0.25
            if t >= span:
                break
            ax.text(t, CEILING * 0.72, tag, color=colour, fontsize=19, weight="bold")
            if i == 1:
                ax.text(
                    t, -CEILING * 0.92, labels[min(int(t / SEG), len(labels) - 1)],
                    color=MUTED, fontsize=13,
                )

    fig.savefig(out, facecolor=INK)
    plt.close(fig)
    return out


def minimap(raw: np.ndarray, clean: np.ndarray, out: Path) -> Path:
    """The whole clip at a glance, for the position marker to travel along."""
    span = len(raw) / SR
    fig = plt.figure(figsize=(VW / 100, VH_MAP / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    fig.patch.set_facecolor(INK)

    draw(ax, raw, 0, ACTIVE, 1200)
    draw(ax, clean, span, TEAL, 1200)
    style(ax, (0, 2 * span))
    ax.axvline(span, color=MUTED, lw=1)

    fig.savefig(out, facecolor=INK)
    plt.close(fig)
    return out


def overview(raw: np.ndarray, clean: np.ndarray) -> Path:
    """Both halves stacked and static — the picture the README embeds."""
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(INK)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.06, hspace=0.3)

    span = len(raw) / SR
    for ax, x, colour, title in (
        (axes[0], raw, ACTIVE, "BEFORE  —  straight from the microphone"),
        (axes[1], clean, TEAL, "AFTER  —  DeepFilterNet3"),
    ):
        draw(ax, x, 0, colour, 2400)
        style(ax, (0, span))
        ax.text(
            0.008, 0.96, title, transform=ax.transAxes,
            color=colour, fontsize=13, weight="bold", va="top",
        )
        for i in range(1, 6):
            ax.axvline(i * SEG, color=GRID, lw=1)
        for i, label in enumerate(SEGMENTS):
            ax.text(i * SEG + 0.4, -CEILING * 0.93, label, color=MUTED, fontsize=8)

    out = ROOT / "docs" / "demo-waveform.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=INK)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not on PATH")
    raw_p, clean_p = S / "demo_raw.wav", S / "demo_cleaned.wav"
    for p in (raw_p, clean_p):
        if not p.exists():
            sys.exit(f"missing {p}; see the module docstring")

    raw, clean = rd(raw_p), rd(clean_p)
    overview(raw, clean)

    # `--segment [n]` cuts one segment out of each take. This is what the
    # README leads with: "everything at once" is the whole product in twenty
    # seconds, where the full tour asks for two minutes before it convinces.
    labels, stem = SEGMENTS, "demo_tour"
    if "--segment" in sys.argv:
        rest = sys.argv[sys.argv.index("--segment") + 1 :]
        i = int(rest[0]) if rest and rest[0].isdigit() else SEGMENTS.index("everything at once")
        cut = slice(i * SEG * SR, (i + 1) * SEG * SR)
        raw, clean, labels = raw[cut], clean[cut], [SEGMENTS[i]]
        stem = "demo_showcase"
        print(f"segment {i}: {SEGMENTS[i]}")

    span = len(raw) / SR
    before = pass_strip(raw, clean, 0, labels, S / "strip-before.png")
    after = pass_strip(raw, clean, 1, labels, S / "strip-after.png")
    small = minimap(raw, clean, S / "strip-map.png")

    joined = S / f"{stem}.wav"
    with wave.open(str(joined), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        both = np.concatenate([raw, clean])
        w.writeframes((np.clip(both, -1, 1) * 32767).astype("<i2").tobytes())

    # Each take is scrolled over the same timeline, then the two are joined:
    # first pass lit on top, second lit underneath. The padding above means no
    # clamping is needed -- at t=0 the crop is already centred, and at the end
    # it lands exactly on iw-VW.
    centre = VW // 2
    scroll = ";".join(
        f"[{i}:v]crop={VW}:{VH_WAVE}:x='t*{PPS}':y=0,"
        f"trim=0:{span:.4f},setpts=PTS-STARTPTS[p{i}]"
        for i in (0, 1)
    )
    waves = (
        f"[p0][p1]concat=n=2:v=1:a=0,"
        f"drawbox=x={centre}:y=0:w=2:h={VH_WAVE}:color=white@0.85:t=fill[top]"
    )
    # overlay, not drawbox: inside a drawbox expression `t` is the box's
    # *thickness*, not the timestamp -- only `enable=` gets timeline time, so
    # a drawbox that looks like it should animate silently never moves.
    marker = f"[2:v][3:v]overlay=x='(t/{2 * span:.4f})*(W-w)':y=0[bot]"

    out = S / f"{stem}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-loop", "1", "-framerate", "30", "-i", str(before),
            "-loop", "1", "-framerate", "30", "-i", str(after),
            "-loop", "1", "-framerate", "30", "-i", str(small),
            "-f", "lavfi", "-i", f"color=c=white:s=3x{VH_MAP}:r=30",
            "-i", str(joined),
            "-filter_complex",
            f"{scroll};{waves};{marker};[top][bot]vstack=inputs=2,format=yuv420p[v]",
            "-map", "[v]", "-map", "4:a",
            "-c:v", "libx264", "-preset", "slow", "-crf", "24",
            "-c:a", "aac", "-b:a", "128k",
            # -shortest is not enough: the minimap overlay inherits the
            # infinitely looped still, so the video ran ~1.7 s past the audio
            # and ended on a silent, motionless frame.
            "-t", f"{2 * span:.4f}",
            "-movflags", "+faststart",
            str(out),
        ],
        check=True,
    )
    for tmp in (joined, before, after, small):
        tmp.unlink()
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("\nGitHub only plays <video> from its own uploads. Drag this into any")
    print("comment box to get a user-attachments URL, then put that in README.md.")


if __name__ == "__main__":
    main()
