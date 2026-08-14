# Bundled model

Two files, both **DeepFilterNet3** by Hendrik Schröter and contributors.

`dfn3_ours.tar.gz` is **what the app ships and loads**: upstream's three-graph
export, built from the published checkpoint by `scripts/export_dfn3.py`
(`uv run scripts/export_dfn3.py`). Every artefact in it is one we can
regenerate.

`model.onnx` is kept for provenance and backwards compatibility — see the
caveat below. Installs from before the switch have it beside the executable and
it still loads.

- Upstream: <https://github.com/Rikorose/DeepFilterNet>
- Paper: *DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement*,
  <https://arxiv.org/abs/2305.08227>
- Copyright © 2021 Hendrik Schröter
- Licensed under **MIT** or **Apache-2.0** at your option — see `LICENSE-MIT`
  and `LICENSE-APACHE` in this directory. Both permit redistribution; this file
  and those licence texts are the attribution that goes with it.

NoiseGate ships this model so the app works on first run. Nothing is downloaded
and nothing is phoned home — the app has no network code at all.

## What this file is, exactly

A single-file streaming ONNX graph:

    input_frame[480], states[45304], atten_lim_db
      -> enhanced_audio_frame[480], new_states[45304], lsnr

The weights are DeepFilterNet3's published ones. Verified rather than assumed:
23 of its large tensors match upstream's `model_120.ckpt.best` byte for byte,
including `enc.df_fc_emb.0.weight` and `enc.emb_gru.linear_in.0.weight`. The
rest are ONNX Runtime's fused conv/bias forms, as expected for a graph produced
by pytorch 1.13.1 and then ORT-optimised.

## The honest caveat

**This exact file is not obtainable from upstream.** Upstream publishes DFN3 as
*three* ONNX files — `enc`, `erb_dec`, `df_dec` — because the STFT front-end is
meant to live in the host application. Somebody repackaged those into the
single streaming graph here, and we do not know who.

The weights are provably genuine, so this is not a trust problem so much as a
reproducibility one: we cannot regenerate this file from source. Producing an
equivalent export ourselves is tracked in `docs/model-pipeline.md`, which
covers what has been rebuilt so far (the three-graph export, byte-identical to
upstream for two of the three) and what remains.

Until that lands, `<local backup>\runtime\model.onnx` is the
backup of record, checksummed in `SHA256SUMS.txt` alongside it.
