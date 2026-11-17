"""Main-text Table 1: CC18 columns plus TabArena ELO, rank and win rate, from the generated tables."""
import os, re
GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'paper', 'generated')
TYPE = {'TabICL': 'pre-trained ICL', 'Cabin': 'graph-informed NN', 'TabM': 'MLP ensemble',
        'CatBoost': 'GBDT', 'RealMLP': 'MLP', 'XGBoost': 'GBDT', 'LightGBM': 'GBDT'}

def key(name):
    name = re.sub(r'\\textbf\{(.*)\}', r'\1', name).replace('$^\\dagger$', '')
    return name.replace('(k=32)', '').strip()

ta = {}
for line in open(os.path.join(GEN, 'table_tabarena.tex')):
    p = [x.strip() for x in line.replace('\\\\', '').split('&')]
    if len(p) >= 5:
        ta[key(p[1])] = (p[2], p[0], p[4])
out = []
for line in open(os.path.join(GEN, 'table_cc18.tex')):
    if line.strip().startswith('\\midrule'):
        out.append(line.rstrip('\n'))
        continue
    p = [x.strip() for x in line.replace('\\\\', '').split('&')]
    k = key(p[0])
    elo, rank, win = ta.get(k, ('--', '--', '--'))
    out.append(' & '.join([k, TYPE[k]] + p[1:] + [elo, rank, win]) + ' \\\\')
open(os.path.join(GEN, 'table_main_results.tex'), 'w').write('\n'.join(out) + '\n')
