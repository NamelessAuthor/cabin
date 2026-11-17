#!/usr/bin/env python3
"""Rebuild the CC18 baseline comparison (Table 1 of the paper)
from the per-fold JSON results in this directory.

Reproduces every mean of the table (52 common tasks). Both the tie-inclusive and the
strict win counts are printed.

Usage: python table2.py
"""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Paper row name -> tub model name
MODELS = {
    'TabICL': 'tabicl',
    'Cabin+Contrastive': 'cabin-v33-contrastive',
    'Cabin+MoE': 'cabin-v33-moe4-shared',
    'Cabin': 'cabin-v33-2slots',
    'TabM': 'tabm',
    'Cabin+Anneal': 'cabin-v33-anneal',
    'CatBoost': 'catboost',
    'RealMLP': 'realmlp',
    'XGBoost': 'xgboost',
    'LightGBM': 'lightgbm',
}


def load(model, root='results'):
    """Mean accuracy per task, keeping only tasks with all 5 folds."""
    out = {}
    for f in glob.glob(os.path.join(HERE, root, model, '*.json')):
        with open(f) as fh:
            j = json.load(fh)
        accs = [r['metrics'].get('accuracy') for r in j.get('fold_results', []) if r.get('metrics')]
        if len(accs) == 5 and None not in accs:
            out[j['task']] = sum(accs) / 5
    return pd.Series(out, dtype=float)


if __name__ == '__main__':
    P = pd.DataFrame({name: load(m) for name, m in MODELS.items()})
    C = P.dropna()
    best = C.max(axis=1)
    is_best = C.eq(best, axis=0)
    table = pd.DataFrame({
        'mean': C.mean(),
        'std': C.std(),
        'wins_ties_all': is_best.sum(),
        'wins_strict': (is_best & (is_best.sum(axis=1) == 1).values[:, None]).sum(),
    }).sort_values('mean', ascending=False)
    print(f'{len(C)} common tasks')
    print(table.round(3).to_string())
