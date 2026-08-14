# Bundled model

Two files, both **DeepFilterNet3** by Hendrik Schröter and contributors.

- Upstream: <https://github.com/Rikorose/DeepFilterNet>
- Paper: *DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement*,
  <https://arxiv.org/abs/2305.08227>
- Copyright © 2021 Hendrik Schröter
- Licensed under **MIT** or **Apache-2.0** at your option — see `LICENSE-MIT`
  and `LICENSE-APACHE` in this directory. Both permit redistribution; this file
  and those licence texts are the attribution that goes with it.

RoomMute ships the model so the app works on first run. Nothing is downloaded
and nothing is phoned home — the app has no network code at all.

## `dfn3_ours.tar.gz` — what the app loads

Upstream's own three-graph export (`enc`, `erb_dec`, `df_dec`) with its config,
built from the published checkpoint by:

```bash
uv run scripts/export_dfn3.py
```

Every artefact in it can be regenerated on any machine in about a minute of CPU
work. Two of the three graphs come out byte-identical to upstream's release.

## `model.onnx` — kept for older installs

A single-file streaming graph:

    input_frame[480], states[45304], atten_lim_db
      -> enhanced_audio_frame[480], new_states[45304], lsnr

Installs from before the switch have this beside the executable and it still
loads, and `model_path` can still be pointed at any single-file streaming
export. It is not what ships.

Its weights are provably DeepFilterNet3's: 23 of its large tensors match
upstream's `model_120.ckpt.best` byte for byte, including
`enc.df_fc_emb.0.weight` and `enc.emb_gru.linear_in.0.weight`. The rest are
ONNX Runtime's fused conv/bias forms, as expected for a graph produced by
pytorch 1.13.1 and then ORT-optimised.

**It is not obtainable from upstream, though.** Upstream publishes the model as
three graphs, because the STFT front-end is meant to live in the host
application; somebody repackaged those into this single graph, and we do not
know who. The weights are genuine, so this is a reproducibility problem rather
than a trust one — and it is exactly why `dfn3_ours.tar.gz` exists.

`docs/model-pipeline.md` has the full picture.
