# Where the model comes from, and how to rebuild it

Everything here is reproducible from a clean machine. Nothing depends on a
file that only exists on one PC — which was the point: the model NoiseGate
shipped with came from a third party and could not be regenerated.

## What NoiseGate actually runs today

`target/release/model.onnx`: a single 16 MB streaming graph.

    input_frame[480], states[45304], atten_lim_db  ->
    enhanced_audio_frame, new_states[45304], lsnr

Its weights are genuinely DeepFilterNet3's — 23 of its large tensors match the
published `model_120.ckpt.best` byte for byte, including
`enc.df_fc_emb.0.weight` and `enc.emb_gru.linear_in.0.weight`. The tensors that
don't match are ONNX Runtime's fused conv/bias forms, expected for a graph
produced by pytorch 1.13.1 and then ORT-optimised.

**It is not upstream's artefact.** Upstream ships DeepFilterNet3 as *three*
ONNX files — `enc`, `erb_dec`, `df_dec` — because the STFT front-end is
deliberately left outside the graph. Someone repackaged the model into one
streaming graph, and that repackaging is what we depend on. If it vanishes
from wherever it came from, the copy in `<local backup>\runtime`
is the only one we have.

Removing that dependency is what the rest of this document is for.

## Licence: we may redistribute

DeepFilterNet is dual-licensed **MIT or Apache-2.0**, and the statement in its
README covers the whole repository, including `models/` where the weights
live. There is no carve-out for the checkpoints. Both licences permit
redistribution provided the licence text and copyright notice travel with it
(`LICENSE-MIT`, `LICENSE-APACHE`, © 2021 Hendrik Schröter).

So shipping the weights inside NoiseGate is allowed. That undercuts the
original reason for training our own model — the goal was never quality, it
was being allowed to ship *something*, and we are.

One caveat worth stating rather than hiding: weights are shaped by their
training data, and whether a dataset's licence reaches through to the trained
weights is unsettled in general. Upstream distributes these weights under
MIT/Apache-2.0 without qualification and that is the licence we would be
relying on.

## Rebuilding the export from the published checkpoint

No GPU. The whole thing is CPU work and takes about a minute.

```bash
git clone https://github.com/Rikorose/DeepFilterNet.git /e/DeepFilterNet
cd /e/DeepFilterNet/models && unzip DeepFilterNet3.zip -d /root/dfn3ref
export PYTHONPATH=/e/DeepFilterNet/DeepFilterNet
python scripts/export_dfn3.py /root/dfn3ref/DeepFilterNet3 /root/dfn3export
```

Produces `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`, plus `*_input.npz` /
`*_output.npz` reference tensors for each stage — those let a reimplementation
be checked stage by stage instead of guessing from bad audio.

Two of the three came out **byte-identical to upstream's release**
(`enc.onnx`, `erb_dec.onnx`); `df_dec.onnx` differs by 1.4 KB. That is the
evidence the export is faithful.

### The trap that cost an hour

The export dies with:

    Failed to import monkeytype. Please install it via
    $ pip install MonkeyType

and installing MonkeyType does not help, because that is not the problem.
`df` imports torch, torch loads the system `libstdc++`, and miniforge's
`libicui18n` then needs a newer `CXXABI` than that provides — so `sqlite3`
fails to load. Upstream guards its export with `import monkeytype`, which
imports `sqlite3` transitively, so an unrelated ABI mismatch surfaces as a
missing package.

Fix: `import sqlite3` **before** anything imports torch. That is all
`scripts/export_dfn3.py` does beyond calling upstream's own script.

## The contract, and what the host has to do

| graph | in | out |
|---|---|---|
| `enc` | `feat_erb[1,1,S,32]`, `feat_spec[1,2,S,96]` | `e0..e3`, `emb`, `c0`, `lsnr` |
| `erb_dec` | `emb`, `e0..e3` | `m` — ERB gains, 32 bands |
| `df_dec` | `emb`, `c0` | `coefs[S,96,10]` — 5 complex taps per bin |

The front-end and both output stages live in the host application. Reference:
`libDF/src/tract.rs`. Per frame:

1. `analysis(480 samples)` -> spectrum, 481 bins
2. `feat_erb` = `erb_norm(erb(spec))`, `feat_spec` = `unit_norm(spec[..96])`
3. run `enc`, then `erb_dec` and `df_dec`
4. **ERB mask**: expand the 32 gains back over the 481 bins by band width and
   multiply (`apply_interp_band_gain`)
5. **Deep filter**: for the lowest 96 bins, replace the value with a complex
   FIR across 5 frames of the *noisy* history, taps from `coefs`
6. `synthesis(spec)` -> 480 samples

Steps 1 and 2 already exist in Rust as `crates/dsp/src/dfn_frontend.rs`, which
is bit-exact against libDF. Steps 4–6 are what remains.

`df_lookahead = 2`, so output for frame *t* uses noisy frames *t+2 … t-2* and
the result lags the input by two hops (960 samples).

## Stage switching is off, deliberately

`libDF`'s runner picks a different treatment for every 10 ms frame from the
model's own SNR estimate:

| lsnr | treatment |
|---|---|
| < `min_db_thresh` | output zeroed |
| > `max_db_erb_thresh` | passed through untouched |
| > `max_db_df_thresh` | ERB mask only |
| otherwise | ERB mask + deep filter |

On speech with background chatter the estimate sits near a boundary
constantly, so neighbouring frames get wholly different processing and the
seams are audible — it sounds like the voice cracking. On a 60-second sample
it also produced **17 seconds of exact digital silence**.

`crates/dsp/src/tract.rs` pushes the thresholds out of reach (`NEVER` /
`ALWAYS`) so one treatment stays in force. This is not a tuning compromise. The
single-file DeepFilterNet3 export everyone compares against has no equivalent
switching — it is one graph that always runs both stages — so disabling it is
what makes us *match* the reference rather than approximate it:

| variant | correlation vs reference | silent seconds |
|---|---|---|
| stage switching on | 0.9295 | 17 |
| **switching off (shipped)** | **1.0000** | 1 |
| switching off + 25 dB limit | 0.9997 | 0 |

Measured end to end, the shipped build differs from the reference by at most
**one sample step out of 32768** — float rounding through a differently shaped
graph, inaudible.

The 25 dB variant is worth remembering: it never emits digital silence at all,
at the cost of a little background. If the reference's silences ever feel like
a dropped call, that is the knob — `attenuation_db` in config.toml.

## Verification status

`scripts/onnx3_runner.py` runs the three graphs over a whole file and applies
both stages. Against `samples/kid_dfn.wav`, which the shipped model produces
**bit-for-bit**, so it is an exact target rather than a listening judgement:

| variant | best lag | correlation |
|---|---|---|
| mask only | −960 | 0.852 |
| mask + df, lookahead 0 | 0 | 0.950 |
| **mask + df, lookahead 2** | **−960** | **0.960** |
| mask + df, lookahead 3 | −1440 | 0.918 |

Lookahead 2 wins and the −960 lag is exactly the two-hop delay, which
confirms both the export and the post-processing. It is **not yet exact**. The
most likely remaining difference is the post-filter (`pf_beta = 0.02`), which
`onnx3_runner.py` does not implement; see `mask_pf` and `pf_beta` in the
config and the post-filter in `tract.rs`.

Use `scripts/diag_align.py` to re-run that table after any change. It scans
lags, so a pure delay cannot be mistaken for a broken stage — which is how the
lookahead was pinned down in the first place.

## Backup

`<local backup>`, every file checksummed in `SHA256SUMS.txt`:

- `runtime/` — the third-party single-file model we ship today, and
  `onnxruntime.dll` 1.22.0. The irreplaceable one.
- `official/` — upstream `model_120.ckpt.best`, `config.ini`, the release
  archives, and both licence texts.
- `our-export/` — the three graphs we built, their reference tensors, and the
  scripts above.
- `our-runs/` — two unsuccessful training attempts, kept as controls. See
  `docs/training.md` for why they failed.
