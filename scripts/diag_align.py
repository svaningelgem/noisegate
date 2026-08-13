"""Which stage is wrong: alignment, the ERB mask, or the deep filter?

Runs the 3-file export three ways and correlates each against the reference
kid_dfn.wav, scanning lags so a pure delay cannot masquerade as a broken stage.
"""

import sqlite3  # noqa: F401
import sys
import wave

import numpy as np
import onnxruntime as ort
from libdf import DF, erb, erb_norm, unit_norm
from df.config import config
from df.modules import get_norm_alpha

SR, HOP, FFT, NB_ERB, NB_DF, DF_ORDER = 48000, 480, 960, 32, 96, 5
EXPORT = "/root/dfn3export"


def rd(p):
    with wave.open(p, "rb") as w:
        d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (d.astype(np.float32) / 32768.0)


def best_lag(a, b, span=4000):
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    fa, fb = np.fft.rfft(a, 2 * n), np.fft.rfft(b, 2 * n)
    x = np.fft.irfft(fa * np.conj(fb))
    x = np.concatenate([x[-span:], x[:span]])
    lag = int(np.argmax(np.abs(x))) - span
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return lag, float(np.max(np.abs(x)) / denom) if denom else 0.0


def main():
    config.load(f"{EXPORT}/config.ini")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    enc = ort.InferenceSession(f"{EXPORT}/enc.onnx", so)
    erb_dec = ort.InferenceSession(f"{EXPORT}/erb_dec.onnx", so)
    df_dec = ort.InferenceSession(f"{EXPORT}/df_dec.onnx", so)

    state = DF(sr=SR, fft_size=FFT, hop_size=HOP, nb_bands=NB_ERB)
    widths = state.erb_widths()  # libdf wants its own array back, not a list
    widths_i = [int(x) for x in widths]
    alpha = get_norm_alpha(False)

    audio = rd(sys.argv[1] if len(sys.argv) > 1 else "/mnt/e/noisegate/samples/kid_raw.wav")
    ref = rd("/mnt/e/noisegate/samples/kid_dfn.wav")
    spec = state.analysis(audio.reshape(1, -1))
    T = spec.shape[1]

    feat_erb = erb_norm(erb(spec, widths), alpha).astype(np.float32).reshape(1, 1, T, NB_ERB)
    u = unit_norm(spec[..., :NB_DF], alpha)
    feat_spec = np.stack([u[0].real, u[0].imag]).astype(np.float32).reshape(1, 2, T, NB_DF)
    e0, e1, e2, e3, emb, c0, lsnr = enc.run(None, {"feat_erb": feat_erb, "feat_spec": feat_spec})
    m = erb_dec.run(None, {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0})[0].reshape(T, NB_ERB)
    coefs = df_dec.run(None, {"emb": emb, "c0": c0})[0].reshape(T, NB_DF, DF_ORDER, 2)
    coefs = coefs[..., 0] + 1j * coefs[..., 1]

    def masked():
        out = spec[0].copy()
        for t in range(T):
            b = 0
            for w_, g in zip(widths_i, m[t]):
                out[t, b : b + w_] *= g
                b += w_
        return out

    def synth(o):
        return state.synthesis(o.reshape(1, T, -1).astype(np.complex64))[0]

    variants = {}
    variants["mask only"] = synth(masked())
    for la in (0, 1, 2, 3, 4):
        o = masked()
        noisy = spec[0]
        for t in range(T):
            acc = np.zeros(NB_DF, np.complex64)
            for k in range(DF_ORDER):
                s = t + la - k
                if 0 <= s < T:
                    acc += noisy[s, :NB_DF] * coefs[t, :, k]
            o[t, :NB_DF] = acc
        variants[f"mask+df lookahead={la}"] = synth(o)

    print(f"{'variant':26s} {'best lag':>9s} {'corr':>8s}")
    for k, v in variants.items():
        lag, c = best_lag(v, ref)
        print(f"{k:26s} {lag:9d} {c:8.4f}")


if __name__ == "__main__":
    main()
