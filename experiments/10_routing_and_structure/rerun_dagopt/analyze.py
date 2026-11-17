"""Paired comparison of Track A variants against rerun_interact/out_highd references."""
import glob, json, os, sys
import pandas as pd
from scipy.stats import wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for base, src in ((os.path.join(HERE, '..', 'rerun_interact', 'out_highd'), 'ref'), (os.path.join(HERE, 'out'), 'new')):
    for f in glob.glob(f'{base}/*/*.json'):
        m = os.path.basename(os.path.dirname(f)); r = json.load(open(f))
        if r.get('status') != 'ok':
            continue
        k = dict(d=r['d'], n=r['n'], lam=r['lam'], rep=r['rep'])
        if m == 'baselines':
            for b in ('lgbm', 'lgbm_stumps', 'lgbm_mb'):
                rows.append({**k, 'model': b, 'acc': r[b]})
        else:
            rows.append({**k, 'model': m, 'acc': r['acc'], 'auc_skel': r.get('auc_skel')})
D = pd.DataFrame(rows)
P = D.pivot_table(index=['lam', 'd', 'n', 'rep'], columns='model', values='acc')
new = sorted(set(D.model) - set(pd.read_json if False else []) - {'lgbm', 'lgbm_stumps', 'lgbm_mb', 'add_true', 'mlp_true',
             'add_rand', 'mlp_rand', 'add_nodag', 'mlp_nodag', 'cabin'})
refs = ['add_true', 'mlp_true', 'add_nodag', 'mlp_nodag', 'add_rand', 'cabin', 'lgbm']
out = []
for v in new:
    for lam in (0.0, 1.0):
        g = P.xs(lam, level='lam')
        g = g[g[v].notna()]
        row = dict(variant=v, lam=lam, n=len(g), acc=g[v].mean())
        for r in refs:
            x = g[v] - g[r]
            p = wilcoxon(x).pvalue if (x != 0).any() and len(x) > 1 else 1.0
            row[f'vs_{r}'] = f'{100*x.mean():+.1f} ({(x>0).sum()}/{len(x)}, p={p:.3f})'
        out.append(row)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(pd.DataFrame(out).set_index(['variant', 'lam']).to_string())
aucs = D[D.model.isin(new)].groupby('model').auc_skel.mean()
print('\nmean skeleton AUC of the graph used:'); print(aucs.round(3).to_string())
