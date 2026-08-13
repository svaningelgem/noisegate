#!/usr/bin/env bash
# Environment for DeepFilterNet work inside WSL2.
#
# Source this, don't run it:  source wsl_env.sh
#
# Two non-obvious bits:
#  * LD_LIBRARY_PATH — miniforge's sqlite3/icu need miniforge's own libstdc++;
#    the system one lacks CXXABI_1.3.15 and imports fail with a confusing
#    "Failed to import monkeytype".
#  * PATH is rebuilt rather than appended to, because WSL inherits the Windows
#    PATH, which contains spaces and parentheses that break naive quoting.
set -u
# Committed copy: see docs/training.md for what each of these works around.

export PATH="$HOME/.cargo/bin:/opt/miniforge3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/opt/miniforge3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# libdfdata links against HDF5; Debian/Ubuntu put the serial build here.
export HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial
export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

# Keep cargo's build products off the Windows filesystem — /mnt is slow enough
# to dominate build times.
export CARGO_TARGET_DIR=/root/.cache/dfn-target
