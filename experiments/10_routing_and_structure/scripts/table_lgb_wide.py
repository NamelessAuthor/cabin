"""Transposed version of generated/table_lightgbm_bins.tex for the main text."""
import os
GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'paper', 'generated')
rows = [[c.strip() for c in l.replace('\\\\', '').split('&')]
        for l in open(os.path.join(GEN, 'table_lightgbm_bins.tex')) if '&' in l]
cols = list(zip(*rows))
names = ['Datasets', 'Mean $\\Delta$', '\\Cabin{} better']
out = ['Subset & ' + ' & '.join(cols[0]) + ' \\\\', '\\midrule']
out += [n + ' & ' + ' & '.join(c) + ' \\\\' for n, c in zip(names, cols[1:])]
open(os.path.join(GEN, 'table_lightgbm_bins_wide.tex'), 'w').write('\n'.join(out) + '\n')
