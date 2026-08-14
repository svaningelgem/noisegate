# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "scipy>=1.10",
#     "soundfile>=0.12",
# ]
# ///
"""Build a reproducible demo recording from openly licensed audio.

    uv run scripts/make_demo_sample.py

Produces a 60-second tour, one voice throughout, in six ten-second segments:

    0:00  clean                  the reference — what should survive
    0:10  office ventilation     stationary noise, the easy case
    0:20  cafeteria babble       many distant voices
    0:30  street traffic         broadband, non-stationary
    0:40  neighbour through wall a single competing voice — the hard case
    0:50  everything at once

The last two are the point. Anything removes a fan; the neighbour is what
RoomMute exists for, and it is deliberately built the way the problem actually
occurs: a real speaker, low-passed and reverberated as a wall would, sitting
well below the near voice.

The README links to this sample, so the mixture *is* redistributed — as MP3,
under CC BY-SA 3.0, with attribution in samples/demo_ATTRIBUTION.txt. The
48 kHz WAVs are not committed: they are 5.8 MB each and this script rebuilds
them byte for byte.

## Sources

* **LibriSpeech** `test-clean`, CC BY 4.0 — <https://www.openslr.org/12/>
  Near voice and the neighbour. 16 kHz, so the mixture is band-limited above
  8 kHz; everything below, including the 4–8 kHz presence band that decides
  whether a denoiser has hollowed out the voice, is intact.
* **DEMAND**, 48 kHz — <https://zenodo.org/records/1227121>
  Real environmental recordings. Note the licence is stated inconsistently:
  Zenodo's record says CC BY 4.0, the original paper says CC BY-SA 3.0. Treat
  it as share-alike if you intend to publish a mixture.

Roughly 1.3 GB is downloaded the first time and cached; later runs are instant.

## Reproducibility

Speaker and clip choices are taken in sorted order, and the seed is fixed, so
two people running this get the same file. Every level is set by measured SNR
rather than by a gain that happens to sound right here.
"""

import io
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

SR = 48_000
SEG = 10  # seconds per segment
SEED = 20260814

CACHE = Path(__file__).resolve().parent.parent / ".demo-cache"
OUT = Path(__file__).resolve().parent.parent / "samples"

LIBRISPEECH = "https://openslr.trmal.net/resources/12/test-clean.tar.gz"
DEMAND = "https://zenodo.org/records/1227121/files/{}_48k.zip?download=1"

# Environment -> what it stands in for. DEMAND has no airshow; the nearest
# permissively licensed thing is a public square or station, and inventing a
# synthetic aeroplane would prove nothing about a real denoiser.
ENVIRONMENTS = {
    "OOFFICE": "office ventilation",
    "PCAFETER": "cafeteria babble",
    "STRAFFIC": "street traffic",
}

# Signal-to-noise ratio per segment, in dB, measured against the near voice.
SNR_DB = {
    "OOFFICE": 6.0,
    "PCAFETER": 6.0,
    "STRAFFIC": 3.0,
    "neighbour": 9.0,
}


def fetch(url: str, name: str) -> Path:
    """Download once, cache forever."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if dest.exists():
        return dest
    print(f"  fetching {name} ...", end="", flush=True)
    with urlopen(url) as r:
        blob = r.read()
    dest.write_bytes(blob)
    print(f" {len(blob) / 1e6:.0f} MB")
    return dest


def librispeech_voices(n: int) -> list[np.ndarray]:
    """The first `n` speakers' longest utterance each, at 48 kHz.

    Sorted order, so the same speakers come out on every machine.
    """
    archive = fetch(LIBRISPEECH, "librispeech-test-clean.tar.gz")
    by_speaker: dict[str, list[tuple[int, bytes]]] = {}
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            if not m.name.endswith(".flac"):
                continue
            speaker = Path(m.name).parts[-3]
            by_speaker.setdefault(speaker, []).append((m.size, m.name))
        chosen = []
        for speaker in sorted(by_speaker)[:n]:
            # Longest file for this speaker: the most continuous speech.
            _, member = max(by_speaker[speaker])
            data = tf.extractfile(member).read()
            audio, sr = sf.read(io.BytesIO(data), dtype="float32")
            chosen.append(resample_poly(audio, SR // 1000, sr // 1000))
            print(f"  speaker {speaker}: {len(chosen[-1]) / SR:.1f}s")
    return chosen


def demand_noise(env: str) -> np.ndarray:
    """One channel of a DEMAND environment, at 48 kHz."""
    archive = fetch(DEMAND.format(env), f"{env}_48k.zip")
    with zipfile.ZipFile(archive) as z:
        # ch01 is the first microphone of the array; any single channel is a
        # normal mono recording of the scene.
        name = next(n for n in sorted(z.namelist()) if n.endswith("ch01.wav"))
        audio, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32")
    assert sr == SR, f"{env} is {sr} Hz, expected {SR}"
    return audio


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def speech_rms(x: np.ndarray, win: int = 4800) -> float:
    """RMS of the parts that are actually speech.

    Now that the voice has pauses, a plain RMS would be pulled down by the
    silence and every noise would be mixed in quieter than the SNR claims.
    Take the loudest quarter of windows instead.
    """
    n = len(x) // win
    if n == 0:
        return rms(x)
    levels = np.array([rms(x[i * win : (i + 1) * win]) for i in range(n)])
    return float(np.mean(levels[levels >= np.percentile(levels, 75)]))


def at_snr(voice: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` so speech-to-noise is `snr_db`."""
    return noise * (speech_rms(voice) / rms(noise)) * (10 ** (-snr_db / 20))


def through_a_wall(speech: np.ndarray) -> np.ndarray:
    """Make a clean speaker sound like the neighbour.

    A wall is a low-pass and a delay spread, not a volume knob — which is the
    whole reason this case is hard. Roll off above 2.5 kHz, then smear with a
    short exponential-decay reverb. What is left is unmistakably speech, with
    the proximity cues removed.
    """
    sos = butter(4, 2500, btype="low", fs=SR, output="sos")
    muffled = sosfilt(sos, speech)

    rng = np.random.default_rng(SEED)
    n = int(0.25 * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir[0] = 1.0
    wet = np.convolve(muffled, ir, mode="full")[: len(muffled)]
    return wet / (np.max(np.abs(wet)) + 1e-9) * np.max(np.abs(muffled))


def tile(x: np.ndarray, n: int) -> np.ndarray:
    """Repeat/trim to exactly n samples."""
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    return x[:n]


def with_pauses(speech: np.ndarray, total: int) -> np.ndarray:
    """Lay speech out in turns, with the gaps a real conversation has.

    This is not cosmetic. A voice that never stops leaves no window containing
    background alone, so every measurement — and every listening impression —
    is dominated by the speech and the noise is barely assessable. The first
    version of this script tiled speech continuously and reported that
    DeepFilterNet3 removed 0.5 dB of an interfering speaker, which says
    nothing about the model and everything about the sample.

    Six seconds on, four off, per ten-second segment, with short fades so the
    edges are not clicks.
    """
    on, off = int(6.0 * SR), int(4.0 * SR)
    fade = int(0.02 * SR)
    ramp = np.linspace(0.0, 1.0, fade)

    out = np.zeros(total, dtype=np.float32)
    src = tile(speech, total)  # keep advancing through the speech, not looping a phrase
    pos = 0
    while pos < total:
        end = min(pos + on, total)
        block = src[pos:end].copy()
        if len(block) > 2 * fade:
            block[:fade] *= ramp
            block[-fade:] *= ramp[::-1]
        out[pos:end] = block
        pos = end + off
    return out


def main() -> None:
    print("sources")
    voices = librispeech_voices(2)
    near_src, neighbour_src = voices[0], voices[1]

    seg = SEG * SR
    total = 6 * seg
    near = with_pauses(near_src, total) * 0.35  # headroom for what gets added

    noises = {}
    for env in ENVIRONMENTS:
        noises[env] = tile(demand_noise(env), seg)
    neighbour = tile(through_a_wall(neighbour_src), seg)

    out = near.copy()
    layout = [("clean", None)]
    for i, (env, label) in enumerate(ENVIRONMENTS.items(), start=1):
        s = i * seg
        out[s : s + seg] += at_snr(near[s : s + seg], noises[env], SNR_DB[env])
        layout.append((label, env))

    s = 4 * seg
    out[s : s + seg] += at_snr(near[s : s + seg], neighbour, SNR_DB["neighbour"])
    layout.append(("neighbour through wall", "LibriSpeech + synthetic wall"))

    s = 5 * seg  # everything, each 3 dB quieter so the sum is not absurd
    block = near[s : s + seg].copy()
    for env in ENVIRONMENTS:
        block = block + at_snr(near[s : s + seg], noises[env], SNR_DB[env] + 3)
    block = block + at_snr(near[s : s + seg], neighbour, SNR_DB["neighbour"] + 3)
    out[s : s + seg] = block
    layout.append(("everything at once", "all of the above"))

    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out * (0.99 / peak)
        print(f"\nscaled by {0.99 / peak:.3f} to avoid clipping")

    OUT.mkdir(parents=True, exist_ok=True)
    noisy = OUT / "demo_raw.wav"
    clean = OUT / "demo_clean_reference.wav"
    sf.write(noisy, out, SR, subtype="PCM_16")
    sf.write(clean, near, SR, subtype="PCM_16")

    print(f"\nwrote {noisy}")
    print(f"wrote {clean}  (the voice alone, for comparison)")
    print("\nlayout:")
    for i, (label, src) in enumerate(layout):
        print(f"  {i * SEG // 60}:{i * SEG % 60:02d}  {label:<24} {src or ''}")

    (OUT / "demo_ATTRIBUTION.txt").write_text(
        "The samples/demo_* files are built by scripts/make_demo_sample.py and\n"
        "redistributed under CC BY-SA 3.0. They are NOT covered by this\n"
        "project's MIT/Apache-2.0 licence.\n\n"
        "Sources:\n\n"
        "  LibriSpeech test-clean - CC BY 4.0\n"
        "    https://www.openslr.org/12/\n"
        "    Panayotov et al., 'Librispeech: an ASR corpus based on public\n"
        "    domain audio books', ICASSP 2015.\n\n"
        "  DEMAND (OOFFICE, PCAFETER, STRAFFIC) - see note below\n"
        "    https://zenodo.org/records/1227121\n"
        "    Thiemann, Ito, Vincent, 'The Diverse Environments Multi-channel\n"
        "    Acoustic Noise Database', Proc. Meetings on Acoustics, 2013.\n\n"
        "The DEMAND licence is stated inconsistently: the Zenodo record says\n"
        "CC BY 4.0, the paper says CC BY-SA 3.0. If you publish a mixture that\n"
        "includes it, the safe reading is share-alike.\n\n"
        "The 'neighbour through wall' segment is a second LibriSpeech speaker\n"
        "low-passed and reverberated by this script, not a recording of anyone.\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT / 'demo_ATTRIBUTION.txt'}")
    print(f"\n  roommute --denoise {noisy} demo_cleaned.wav")


if __name__ == "__main__":
    main()
