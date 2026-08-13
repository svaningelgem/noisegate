"""Run DeepFilterNet3 from our own 3-file ONNX export.

This validates the export and, more importantly, the post-processing that the
Rust side must reproduce: an ERB gain mask onto the spectrum, then a complex
"deep filter" across DF_ORDER frames on the lowest NB_DF bins. It mirrors
libDF/src/tract.rs, written in Python first so the maths can be checked apart
from the ort plumbing.

The exported graphs carry a dynamic time axis, so the whole file goes through
in one call per graph. Real-time streaming (one 480-sample hop at a time) is
the Rust implementation's job; crates/dsp/src/dfn_frontend.rs already keeps
the analysis/normalisation state needed for that.

Usage: python onnx3_runner.py <export_dir> <in.wav> <out.wav>
"""

import sqlite3  # noqa: F401  MUST precede df/torch imports; see export_dfn3.py
import sys
import wave

import numpy as np
import onnxruntime as ort
from libdf import DF, erb, erb_norm, unit_norm
from df.config import config
from df.modules import get_norm_alpha

SR, HOP, FFT = 48000, 480, 960
NB_ERB, NB_DF = 32, 96
DF_ORDER, DF_LOOKAHEAD = 5, 2


def read_wav(p):
    with wave.open(p, "rb") as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (d.astype(np.float32) / 32768.0).reshape(1, -1)


def write_wav(p, x):
    with wave.open(p, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    export_dir, inp, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    # get_norm_alpha reads norm_tau from the model config, so it has to be
    # loaded before any feature call.
    config.load(f"{export_dir}/config.ini")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    enc = ort.InferenceSession(f"{export_dir}/enc.onnx", so)
    erb_dec = ort.InferenceSession(f"{export_dir}/erb_dec.onnx", so)
    df_dec = ort.InferenceSession(f"{export_dir}/df_dec.onnx", so)

    state = DF(sr=SR, fft_size=FFT, hop_size=HOP, nb_bands=NB_ERB)
    widths = state.erb_widths()
    alpha = get_norm_alpha(False)

    audio = read_wav(inp)
    spec = state.analysis(audio)  # [C, T, F] complex
    T = spec.shape[1]

    feat_erb = erb_norm(erb(spec, widths), alpha).astype(np.float32)
    feat_erb = feat_erb.reshape(1, 1, T, NB_ERB)
    u = unit_norm(spec[..., :NB_DF], alpha)
    feat_spec = np.stack([u[0].real, u[0].imag]).astype(np.float32).reshape(1, 2, T, NB_DF)

    e0, e1, e2, e3, emb, c0, lsnr = enc.run(
        None, {"feat_erb": feat_erb, "feat_spec": feat_spec}
    )
    m = erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})[0]
    coefs = df_dec.run(None, {"emb": emb, "c0": c0})[0]

    # The model emits its estimate for frame t at index t - DF_LOOKAHEAD, so
    # line the spectrum up before applying anything.
    out = spec[0].copy()  # [T, F]

    # Stage 1: ERB gains, expanded from 32 bands back to 481 bins.
    gains = m.reshape(T, NB_ERB)
    for t in range(T):
        b = 0
        for width, g in zip(widths, gains[t]):
            out[t, b : b + int(width)] *= g
            b += int(width)

    # Stage 2: deep filter. coefs is [T, NB_DF, 2*DF_ORDER] -> complex taps.
    c = coefs.reshape(T, NB_DF, DF_ORDER, 2)
    c = c[..., 0] + 1j * c[..., 1]
    noisy = spec[0]
    for t in range(T):
        acc = np.zeros(NB_DF, np.complex64)
        for k in range(DF_ORDER):
            src = t + DF_LOOKAHEAD - k
            if 0 <= src < T:
                acc += noisy[src, :NB_DF] * c[t, :, k]
        out[t, :NB_DF] = acc

    enhanced = state.synthesis(out.reshape(1, T, -1).astype(np.complex64))
    write_wav(outp, enhanced[0])
    print(f"wrote {outp}: {T} frames, lsnr mean {float(np.mean(lsnr)):.1f} dB")


if __name__ == "__main__":
    main()
