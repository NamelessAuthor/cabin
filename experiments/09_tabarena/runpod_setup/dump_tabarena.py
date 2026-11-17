#!/usr/bin/env python3
"""
Dump all TabArena baseline results to CSV.

Usage:
    python dump_tabarena.py

Output:
    /tmp/tabarena_baselines.csv
"""
import os
import sys
# local clone of https://github.com/autogluon/tabarena
sys.path.insert(0, os.path.expanduser('~/tabarena/tabarena'))
sys.path.insert(0, os.path.expanduser('~/tabarena/bencheval'))

from tabarena.nips2025_utils.tabarena_context import TabArenaContext

ctx = TabArenaContext(include_unverified=False)
df = ctx.load_results_paper(download_results='auto')
df.to_csv('/tmp/tabarena_baselines.csv', index=False)
print(f"{len(df)} rows, {df['method'].nunique()} methods, {df['dataset'].nunique()} datasets")
print(f"Saved to /tmp/tabarena_baselines.csv")
