#!/usr/bin/env bash
# Run 2: p_reverb 0.1 instead of 0.2.
#
# Run 1 (/root/dfn-run) stopped early at epoch 40, best at 29. Its noise-floor
# numbers were spectacular and its near voice was wrecked: -13.7 dB in the
# 4-8 kHz presence band against DeepFilterNet3's +1.7, and -33.9 dB at p95 on
# the chat sample. It learned to suppress rather than to separate.
#
# The one suspect is p_reverb. DeepFilterNet3 ships 0.1; run 1 used 0.2 on the
# theory that dereverberation pressure is what teaches proximity
# discrimination. Too much of it and the model treats the near voice's own room
# reflections as noise, taking the brightness with them.
#
# So this changes exactly two things against run 1:
#   p_reverb                 0.2 -> 0.1   the experiment
#   early_stopping_patience  12  -> 8     stops sooner once it plateaus; run 1
#                                         spent 11 epochs (~85 min) after its
#                                         best proving it was done
#
# batch_size stays at 64 deliberately. Raising it to 96 is the obvious speed
# win, but it changes gradient noise and how the LR schedule lands, so a better
# result would not be attributable to p_reverb. Bank the speed after this
# question is settled.
#
# Compare against: /root/dfn-run/checkpoints/model_29.ckpt.best
#   eval:  python eval_ours.py /root/dfn-run2 kid,chat OURS-r2
#   bands: python diag_p95.py
set -u
source "$(dirname "$0")/wsl_env.sh"
cd /root/dfn-run2
export DF_NO_PIN_MEMORY=1   # WSL cannot allocate pinned host memory
exec /root/dfn-rocm/bin/python -m df.train \
  /root/dfn-run2/dataset.cfg /root/dfn-data/hdf5/ /root/dfn-run2
